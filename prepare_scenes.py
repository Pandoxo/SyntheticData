import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Find spawn patches in .blend scenes and record which objects surround each one.
#
#     blenderproc run prepare_scenes.py -- --config config.yaml
#     blenderproc run prepare_scenes.py -- --config config.yaml --scene scenes/forest_2.blend
#
# Writes <scene>.zones.json next to every scene. The generator then loads only the
# objects within `scene.crop_radius_m` of the chosen patch, which is what keeps big
# scenes (forest_2.blend: 13 GB when loaded whole) inside a render node's memory.
# Hand-made `spawn_*` planes are kept as patches of their own. Run it once per scene,
# and again after editing a scene.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from sdlib import config, environment, scenes


def prepare(path, cfg):
    scene_cfg = cfg["scene"]
    prep_cfg = scene_cfg["auto_zones"]
    rng = np.random.default_rng(0)

    bproc.clean_up(clean_up_camera=False)
    loaded = bproc.loader.load_blend(path, obj_types=["mesh"])
    meshes = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
    manual = environment.find_spawn_zones(meshes, scene_cfg["spawn_zone_pattern"])
    geometry = [o for o in meshes if o not in manual]
    radius = float(scene_cfg["crop_radius_m"])

    zones = []
    for zone in manual:
        bb = np.asarray(zone.get_bound_box())
        low, high = bb.min(axis=0), bb.max(axis=0)
        centre = [float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2), float(high[2])]
        zones.append({"name": zone.get_name(), "manual": True, "center": centre,
                      "size": [float(high[0] - low[0]), float(high[1] - low[1])],
                      "objects": scenes.objects_near(geometry, centre, radius) + [zone.get_name()]})

    if prep_cfg.get("enabled", True):
        for k, (x, y, z) in enumerate(scenes.find_ground_patches(geometry, prep_cfg, rng)):
            size = float(prep_cfg["patch_size_m"])
            zones.append({"name": f"auto_{k:02d}", "manual": False, "center": [x, y, z],
                          "size": [size, size],
                          "objects": scenes.objects_near(geometry, (x, y, z), radius)})

    out = {"scene": os.path.basename(path), "crop_radius_m": radius,
           "total_objects": len(meshes), "zones": zones}
    with open(scenes.zones_file(path), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    sizes = [len(z["objects"]) for z in zones]
    print(f"[prepare] {path}: {len(zones)} zone(s) ({len(manual)} manual), objects per "
          f"crop {min(sizes) if sizes else 0}-{max(sizes) if sizes else 0} of {len(meshes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scene", action="append", default=None,
                        help="scene file(s) to prepare (default: every scene.files entry)")
    parser.add_argument("--missing-only", action="store_true",
                        help="skip scenes that already have a .zones.json")
    args = parser.parse_args()
    cfg = config.load(config.resolve(args.config))
    bproc.init()
    paths = config.resolve_all(args.scene or cfg["scene"]["files"])
    for path in paths:
        if not os.path.isfile(path):
            print(f"[prepare] missing, skipped: {path}")
            continue
        if args.missing_only and os.path.isfile(scenes.zones_file(path)):
            continue
        try:
            prepare(path, cfg)
        except OSError as err:
            # Typically a scene saved by a newer Blender than BlenderProc's 4.2.
            print(f"[prepare] cannot read {path}: {err}")


if __name__ == "__main__":
    main()
