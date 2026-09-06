import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Scene generator: trash dropped into a real environment loaded from a .blend file.
#
# Marking where trash may appear
# ------------------------------
# 1. Open the scene .blend in Blender.
# 2. Add a plane (Add > Mesh > Plane), scale and move it so it lies flat ON the
#    ground exactly where you want objects to appear - e.g. a 2x2 m patch of
#    pavement next to a bin.
# 3. Name it so it matches `scene.spawn_zone_pattern` from the config, by default
#    anything starting with "spawn_" (spawn_pavement, spawn_gutter, spawn_grass...).
# 4. Add as many zones as you like and save the file.
#
# Zones are never rendered and never annotated - they only say "here". By default an
# invisible box collider is built with its top face on the zone surface, so objects
# land on the zone regardless of what the scene's real ground mesh looks like. Keep
# zones flat and axis-aligned; for sloped ground set physics.collider to "mesh".
#
# Use `blenderproc run inspect_blend.py -- --scene <file.blend>` to list what a scene
# contains and check that your zones are picked up.
#
# Render one scene:
#     blenderproc run generate_scene.py -- --config config_scene.yaml --scene-index 0
# Render the whole dataset (one process per scene):
#     .venv/bin/python run_batch.py --config config_scene.yaml --script generate_scene.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from sdlib import cli, config, environment, pipeline
from sdlib.environment import Environment


def _pick_scene(cfg, rng):
    paths = config.resolve_all(cfg["scene"]["files"])
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"scene.files entries do not exist: {missing}")
    if not paths:
        raise ValueError("scene.files is empty - point it at at least one .blend file")
    return paths[int(rng.integers(0, len(paths)))]


def build_environment(cfg, rng):
    scene_cfg = cfg["scene"]
    path = _pick_scene(cfg, rng)
    print(f"[scene] loading {path}")

    obj_types = ["mesh"]
    if scene_cfg.get("load_scene_lights"):
        obj_types.append("light")
    loaded = bproc.loader.load_blend(path, obj_types=obj_types)
    meshes = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]

    zones = environment.find_spawn_zones(meshes, scene_cfg["spawn_zone_pattern"])
    if not zones:
        names = sorted(o.get_name() for o in meshes)[:40]
        raise RuntimeError(
            f"No object in {path} matches spawn_zone_pattern "
            f"{scene_cfg['spawn_zone_pattern']!r}.\n"
            f"Add a flat plane on the ground named e.g. 'spawn_pavement' and save the "
            f"file. Objects currently in the scene include: {names}")

    statics = [o for o in meshes if o not in zones]
    for obj in statics:
        # Everything from the scene is background. Without this the segmentation
        # renderer would have no category_id for it, and any scene object left
        # unlabelled could end up annotated as trash.
        obj.set_cp("category_id", 0)
    for zone in zones:
        environment.prepare_zone(zone)

    colliders = [
        environment.build_collider(
            zone,
            mode=cfg["physics"]["collider"],
            thickness=cfg["physics"]["collider_thickness_m"],
            friction=cfg["physics"]["ground_friction"],
            margin=cfg["physics"]["collider_margin_m"],
        )
        for zone in zones
    ]

    # Optionally let the trash also collide with the scene's own geometry, so items
    # can come to rest against a kerb or a wall instead of only on the flat proxy.
    # This is much slower and depends on the scene being clean, hence opt-in.
    if cfg["physics"].get("collide_with_scene"):
        for obj in _statics_near_zones(statics, zones, cfg["physics"]["scene_collider_margin_m"]):
            obj.enable_rigidbody(active=False, collision_shape="MESH",
                                 friction=cfg["physics"]["ground_friction"])
            colliders.append(obj)

    print(f"[scene] {len(zones)} spawn zone(s), {len(statics)} static mesh(es)")
    return Environment(zones=zones, colliders=colliders, statics=statics,
                       description=os.path.basename(path))


def _statics_near_zones(statics, zones, margin):
    """Scene meshes whose bounding box overlaps a zone's, expanded by `margin`."""
    zone_boxes = []
    for zone in zones:
        bb = np.asarray(zone.get_bound_box())
        zone_boxes.append((bb.min(axis=0) - margin, bb.max(axis=0) + margin))

    near = []
    for obj in statics:
        bb = np.asarray(obj.get_bound_box())
        low, high = bb.min(axis=0), bb.max(axis=0)
        if any(np.all(low <= zhigh) and np.all(high >= zlow) for zlow, zhigh in zone_boxes):
            near.append(obj)
    return near


def main():
    args = cli.parse_args("config_scene.yaml")
    cfg, catalog, hdri_paths, output_dir = cli.bootstrap(args)
    pipeline.generate(cfg, args.scene_index, catalog, hdri_paths,
                      build_environment, output_dir)


if __name__ == "__main__":
    main()
