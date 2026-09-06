"""Scene generation, shared by both entry points.

One invocation renders exactly one scene. Running each scene in its own process is
deliberate: Blender never frees appended materials, images and meshes, so looping
scenes inside a single process grows memory until it dies. `run_batch.py` drives
the loop from the outside.
"""

import json
import os
import random

import blenderproc as bproc
import numpy as np

from . import assets as assets_mod
from . import capture, environment


def _seed_everything(seed):
    random.seed(seed)          # BlenderProc's own samplers use the stdlib RNG
    np.random.seed(seed % (2 ** 32))
    capture.set_render_seed(seed)
    return np.random.default_rng(seed)


def _load_trash(catalog, env, cfg, rng):
    """Load, label and position the trash objects, then let them fall."""
    obj_cfg = cfg["objects"]
    n_objects = int(rng.integers(obj_cfg["min_objects_per_scene"],
                                 obj_cfg["max_objects_per_scene"] + 1))

    objects, zone_per_object, records = [], {}, []
    for index, asset in enumerate(catalog.sample(rng, n_objects)):
        obj = assets_mod.load(asset, rng, index)
        zone = env.zones[int(rng.integers(0, len(env.zones)))]
        objects.append(obj)
        zone_per_object[obj] = zone
        records.append({"asset": asset.name, "class": asset.class_name,
                        "category_id": asset.category_id, "zone": zone.get_name()})

        obj.enable_rigidbody(
            active=True,
            collision_shape=asset.collision_shape,
            mass=asset.mass_kg,
            friction=cfg["physics"]["object_friction"],
            linear_damping=cfg["physics"]["linear_damping"],
            angular_damping=cfg["physics"]["angular_damping"],
        )

    environment.sample_drop_poses(objects, zone_per_object, env.colliders, cfg["physics"], rng)
    environment.simulate_drop(cfg["physics"])

    kept, _ = environment.discard_escapees(objects, env.zones,
                                           cfg["physics"]["escape_tolerance_m"])
    return kept, records


def generate(cfg, scene_index, catalog, hdri_paths, build_environment, output_dir):
    """Render one scene. Returns a metadata dict, or None if nothing usable came out."""
    seed = int(cfg["seed"]) + scene_index
    rng = _seed_everything(seed)

    env = build_environment(cfg, rng)
    if not env.zones:
        raise RuntimeError("The environment defined no spawn zones - there is nowhere "
                           "to put trash. See the `spawn_zone_pattern` setting.")

    world = environment.setup_world(hdri_paths, cfg["hdri"], rng)
    trash, records = _load_trash(catalog, env, cfg, rng)
    if not trash:
        print(f"[scene {scene_index}] every object left its spawn zone, skipping")
        return None

    poi = bproc.object.compute_poi(trash)
    lights = []
    if world["is_dark"] or cfg["lights"]["always"]:
        lights = environment.add_lights(cfg["lights"], poi, rng)

    # The BVH is what the camera validation ray-casts against: visible scene geometry
    # plus the trash itself, but not the invisible zones or colliders.
    occluders = list(env.statics) + list(trash)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(occluders) if occluders else None

    frames = capture.sample_views(trash, bvh_tree, cfg["camera"], rng, cfg["views_per_scene"])
    if not frames:
        print(f"[scene {scene_index}] no valid camera pose found, skipping render")
        return None

    # Only now, with every object placed, is it safe to switch segmentation on.
    capture.enable_annotations(cfg["annotations"])

    data = bproc.renderer.render()
    filtered, stats = capture.filter_by_visibility(data, cfg["annotations"])

    os.makedirs(output_dir, exist_ok=True)
    n_written = capture.write_coco(output_dir, filtered, catalog.label_mapping(),
                                   cfg["annotations"])
    if cfg["annotations"].get("write_hdf5"):
        capture.write_hdf5(output_dir, data)

    print(f"[scene {scene_index}] {n_written} image(s), "
          f"{stats['annotations_kept']} annotation(s) "
          f"(dropped {stats['annotations_dropped']} tiny, "
          f"{stats['frames_dropped']} empty frame(s))")

    # Frames dropped by the visibility filter shift the remaining ones, so only keep
    # the camera metadata for frames that actually made it into the dataset. The order
    # matches the COCO image order, so index i here describes the i-th written image.
    kept_frames = [frames[i] for i in filtered["kept_indices"]]

    meta = {
        "scene_index": scene_index,
        "seed": seed,
        "environment": env.description,
        "world": world,
        "lights": lights,
        "objects": records,
        "frames": kept_frames,
        "stats": stats,
    }
    with open(os.path.join(output_dir, "scene_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
