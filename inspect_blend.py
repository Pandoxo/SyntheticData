import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Inspection helper. Renders nothing; it just tells you what is inside your .blend
# files so you can calibrate assets.yaml and verify your spawn zones.
#
#   blenderproc run inspect_blend.py -- --assets
#       For every catalogued asset: native bounding box, the scale factor implied by
#       `longest_m`, and how many mesh parts it has. A factor that is not roughly 1,
#       0.01 or 100 means the assumed real-world size is probably wrong.
#
#   blenderproc run inspect_blend.py -- --scene path/to/scene.blend
#       Lists the scene's objects with their sizes and flags which ones match the
#       spawn-zone pattern. Run this after adding zones in Blender.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from sdlib import assets as assets_mod
from sdlib import config, environment


def _dims(obj):
    return assets_mod.world_dimensions(obj)


def inspect_assets(cfg_path):
    cfg = config.load(config.resolve(cfg_path))
    catalog = assets_mod.Catalog.from_config(
        config.resolve(cfg["objects"]["assets_file"]),
        config.resolve_all(cfg["objects"]["model_dirs"]),
    )

    print(f"\n{'asset':<34} {'class':<10} {'parts':>5} "
          f"{'native size (m)':>24} {'target':>8} {'scale':>10}")
    print("-" * 100)
    for asset in catalog.assets:
        loaded = bproc.loader.load_blend(asset.path, obj_types=["mesh"])
        meshes = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
        if not meshes:
            print(f"{asset.name:<34} {asset.class_name:<10} {'0':>5}   (no mesh objects!)")
            continue

        allbb = np.concatenate([np.asarray(o.get_bound_box()) for o in meshes], axis=0)
        size = allbb.max(axis=0) - allbb.min(axis=0)
        longest = float(size.max())
        factor = asset.longest_m / longest if longest > 0 else float("nan")

        # USD-derived models routinely import at 1000x or more, so only a factor far
        # outside even that range says anything - usually a degenerate mesh.
        flag = "" if 1e-6 <= factor <= 1e4 else "  <-- suspicious"
        print(f"{asset.name[:34]:<34} {asset.class_name:<10} {len(meshes):>5} "
              f"{size[0]:>7.3f} {size[1]:>7.3f} {size[2]:>7.3f} "
              f"{asset.longest_m:>8.3f} {factor:>10.4g}{flag}")

        # Deleting a parent takes its children with it, so later references in this
        # list can already be dead by the time we reach them.
        for obj in meshes:
            if obj.is_valid():
                obj.delete(remove_all_offspring=True)

    print("\nA scale factor far from 1 is normal (assets ship in cm, mm or arbitrary units).\n"
          "What matters is that it is consistent for similar objects and that `longest_m`\n"
          "in assets.yaml matches the real-world item the model depicts.")


def inspect_scene(scene_path, pattern):
    path = config.resolve(scene_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    print(f"\nLoading {path}")

    loaded = bproc.loader.load_blend(path, obj_types=["mesh", "empty", "light", "camera"])
    meshes = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
    others = [o for o in loaded if not isinstance(o, bproc.types.MeshObject)]
    zones = environment.find_spawn_zones(meshes, pattern)
    zone_names = {z.get_name() for z in zones}

    print(f"\n{len(loaded)} object(s): {len(meshes)} mesh, {len(others)} other\n")
    print(f"{'name':<44} {'size (m)':>24}   role")
    print("-" * 90)
    for obj in sorted(meshes, key=lambda o: o.get_name()):
        size = _dims(obj)
        role = "SPAWN ZONE" if obj.get_name() in zone_names else "background"
        print(f"{obj.get_name()[:44]:<44} {size[0]:>7.2f} {size[1]:>7.2f} {size[2]:>7.2f}   {role}")

    print()
    if zones:
        print(f"{len(zones)} zone(s) match {pattern!r}:")
        for zone in zones:
            bb = np.asarray(zone.get_bound_box())
            low, high = bb.min(axis=0), bb.max(axis=0)
            size = high - low
            if size[2] > 0.05 * max(size[0], size[1]):
                print(f"  ! {zone.get_name()} is {size[2]:.2f} m tall - a zone should be a "
                      f"flat plane, not a solid")
            print(f"  - {zone.get_name()}: {size[0]:.2f} x {size[1]:.2f} m, "
                  f"surface at z = {high[2]:.3f}")
    else:
        print(f"No object matches {pattern!r}. Open the file in Blender, add a flat plane "
              f"lying on the ground where you want trash, name it e.g. 'spawn_pavement', "
              f"and save.")


def main():
    parser = argparse.ArgumentParser(description="Inspect assets or a scene .blend")
    parser.add_argument("--config", default="config_scene.yaml")
    parser.add_argument("--assets", action="store_true", help="inspect the asset catalogue")
    parser.add_argument("--scene", default=None, help="path to a scene .blend to inspect")
    parser.add_argument("--pattern", default=None, help="override the spawn zone regex")
    args = parser.parse_args()

    if not args.assets and not args.scene:
        parser.error("pass --assets and/or --scene <file.blend>")

    bproc.init()
    if args.assets:
        inspect_assets(args.config)
    if args.scene:
        pattern = args.pattern
        if pattern is None:
            cfg = config.load(config.resolve(args.config))
            pattern = cfg.get("scene", {}).get("spawn_zone_pattern", "spawn_.*")
        inspect_scene(args.scene, pattern)


if __name__ == "__main__":
    main()
