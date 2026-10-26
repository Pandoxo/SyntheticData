"""Scene generation: one call renders one scene, end to end.

One invocation renders exactly one scene. Running each scene in its own process is
deliberate: Blender never frees appended materials, images and meshes, so looping
scenes inside a single process grows memory until it dies. `run_batch.py` drives
the loop from the outside.

Order matters, and is:
    environment -> lighting -> litter (deformed, dirtied) -> physics drop
    -> after_physics (bury, clutter, canopy) -> cameras -> after_cameras (grass)
    -> segmentation on -> render -> filter -> write
Grass comes after the cameras because it is cleared where the robot stands, and
segmentation comes last because it only labels objects that already exist.
"""

import json
import os
import random
import shutil

import blenderproc as bproc
import bpy
import numpy as np

from . import assets as assets_mod
from . import cameras, capture, environment, procedural, realism, scenes


def _seed_everything(seed):
    random.seed(seed)          # BlenderProc's own samplers use the stdlib RNG
    np.random.seed(seed % (2 ** 32))
    capture.set_render_seed(seed)
    return np.random.default_rng(seed)


_PROFILE = {"t": None, "stages": {}}


def _log_mem(stage):
    """Peak memory and time spent since the previous stage.

    Both end up in scene_meta.json ("profile"), which is what capacity planning for
    a big run is based on - and peak memory is the number to watch when a biome or
    a scene gets OOM-killed.
    """
    import resource  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel
    now = time.time()
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    elapsed = now - _PROFILE["t"] if _PROFILE["t"] is not None else 0.0
    _PROFILE["t"] = now
    _PROFILE["stages"][stage] = {"s": round(elapsed, 2), "peak_rss_gb": round(peak_gb, 2)}
    print(f"[mem] {stage}: peak {peak_gb:.1f} GB, {elapsed:.1f}s")


def _choose_env_type(cfg, rng, forced):
    if forced:
        return forced
    weights = cfg.get("environments", {"procedural": 1.0})
    names = sorted(k for k, v in weights.items() if v > 0)
    probs = np.array([float(weights[n]) for n in names])
    return str(rng.choice(names, p=probs / probs.sum()))


def _load_trash(catalog, env, cfg, rng):
    """Load, label, deform and position the trash objects, then let them fall."""
    obj_cfg = cfg["objects"]
    n_objects = int(rng.integers(obj_cfg["min_objects_per_scene"],
                                 obj_cfg["max_objects_per_scene"] + 1))
    real_cfg = cfg.get("realism", {})

    objects, zone_per_object, records = [], {}, []
    for index, asset in enumerate(catalog.sample(rng, n_objects)):
        try:
            obj = assets_mod.load(asset, rng, index)
        except Exception as err:  # pylint: disable=broad-except
            # One broken download must not cost the whole scene.
            print(f"[trash] could not load {asset.path}: {err}")
            continue
        record = {"object": obj.get_name(), "asset": asset.name, "path": asset.path,
                  "class": asset.class_name, "category_id": asset.category_id,
                  "source": asset.source, "license": asset.license}
        if real_cfg.get("deform") and asset.deformable:
            record["deform"] = realism.deform(obj, asset.class_name, real_cfg["deform"], rng)
        if real_cfg.get("grime"):
            record["grime"] = realism.add_grime(obj, real_cfg["grime"], rng)

        zone = env.zones[int(rng.integers(0, len(env.zones)))]
        record["zone"] = zone.get_name()
        objects.append(obj)
        zone_per_object[obj] = zone
        records.append(record)

        obj.enable_rigidbody(
            active=True,
            collision_shape=asset.collision_shape,
            mass=asset.mass_kg,
            friction=cfg["physics"]["object_friction"],
            linear_damping=cfg["physics"]["linear_damping"],
            angular_damping=cfg["physics"]["angular_damping"],
        )

    if not objects:
        return [], records
    environment.sample_drop_poses(objects, zone_per_object, env.colliders, cfg["physics"], rng)

    # BlenderProc's physics run leaves some Python references to objects dangling
    # ("StructRNA of type Object has been removed") although the objects are intact -
    # it happens to models imported from glTF in particular. Keep names across the
    # simulation and look the objects up again afterwards.
    names = [o.get_name() for o in objects]
    zone_names = [z.get_name() for z in env.zones]
    environment.simulate_drop(cfg["physics"])
    objects = [bproc.types.MeshObject(bpy.data.objects[n]) for n in names if n in bpy.data.objects]
    env.zones = [bproc.types.MeshObject(bpy.data.objects[n]) for n in zone_names if n in bpy.data.objects]

    kept, _ = environment.discard_escapees(objects, env.zones,
                                           cfg["physics"]["escape_tolerance_m"])
    kept_names = {o.get_name() for o in kept}
    records = [r for r in records if r["object"] in kept_names]
    return kept, records


def generate(cfg, scene_index, catalog, resources, output_dir, env_type=None):
    """Render one scene. Returns a metadata dict, or None if nothing usable came out."""
    seed = int(cfg["seed"]) + scene_index
    rng = _seed_everything(seed)
    _log_mem("start")

    env_type = _choose_env_type(cfg, rng, env_type)
    if env_type == "procedural":
        env = procedural.build_environment(cfg, rng, resources["index"])
    elif env_type == "scene":
        env = scenes.build_environment(cfg, rng)
    else:
        raise ValueError(f"unknown environment type {env_type!r}")
    if not env.zones:
        raise RuntimeError("The environment defined no spawn zones - there is nowhere "
                           "to put trash.")
    print(f"[scene {scene_index}] environment: {env.description}")
    _log_mem("environment")

    world = environment.setup_world(resources["hdri_paths"], cfg["hdri"], rng,
                                    resources.get("dark_hdris", frozenset()))
    trash, records = _load_trash(catalog, env, cfg, rng)
    _log_mem("litter + physics")
    if not trash:
        print(f"[scene {scene_index}] every object left its spawn zone, skipping")
        return None

    statics = list(env.statics)
    extra = {}
    if env.after_physics is not None:
        extra = env.after_physics(trash, rng) or {}
        statics += extra.get("statics", [])
        _log_mem("clutter")

    poi = bproc.object.compute_poi(trash)
    lights = []
    if world["is_dark"] or cfg["lights"]["always"]:
        lights = environment.add_lights(cfg["lights"], poi, rng)

    frames = cameras.sample_views(trash, env, cfg["camera"], rng, cfg["views_per_scene"])
    _log_mem("cameras")
    if not frames:
        print(f"[scene {scene_index}] no valid camera pose found, skipping render")
        return None

    after_cam = {}
    if env.after_cameras is not None:
        after_cam = env.after_cameras(trash, frames, rng) or {}
        _log_mem("grass")

    # Only now, with every object placed, is it safe to switch segmentation on.
    capture.enable_annotations(cfg["annotations"])

    data = bproc.renderer.render()
    _log_mem("render")
    filtered, stats = capture.filter_by_visibility(data, frames, cfg["annotations"])

    # A scene directory belongs to exactly one scene index. Clear leftovers of an
    # earlier attempt: the COCO writer appends to an existing file, while the dense
    # maps overwrite theirs, and a mix of both silently pairs images with the wrong
    # masks.
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    n_written = capture.write_coco(output_dir, filtered, catalog.label_mapping(),
                                   cfg["annotations"])
    instance_table = {}
    if n_written:
        instance_table = capture.write_dense_maps(output_dir, filtered,
                                                  [o.get_name() for o in trash])
    if cfg["annotations"].get("write_hdf5"):
        capture.write_hdf5(output_dir, data)

    print(f"[scene {scene_index}] {n_written} image(s), "
          f"{stats['annotations_kept']} annotation(s) "
          f"(dropped {stats['annotations_dropped']} tiny, "
          f"{stats['frames_dropped']} empty frame(s), kept {stats['empty_frames_kept']} negative(s))")

    # Frames dropped by the visibility filter shift the remaining ones, so only keep
    # the camera metadata for frames that actually made it into the dataset. The order
    # matches the image order, so index i here describes images/00000i.png.
    kept_frames = [frames[i] for i in filtered["kept_indices"]]

    world_meta = getattr(env, "world", None)
    meta = {
        "scene_index": scene_index,
        "seed": seed,
        "environment_type": env_type,
        "environment": env.description,
        "environment_meta": env.meta,
        "materials": world_meta.materials.used if world_meta is not None else [],
        "world": world,
        "lights": lights,
        "objects": records,
        "instances": instance_table,
        "buried": extra.get("buried", {}),
        "grass": after_cam,
        "frames": kept_frames,
        "stats": stats,
        "profile": _PROFILE["stages"],
    }
    _log_mem("write")
    meta["profile"] = dict(_PROFILE["stages"])
    if n_written:
        with open(os.path.join(output_dir, "scene_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=str)
    return meta
