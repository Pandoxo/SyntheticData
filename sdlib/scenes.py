"""Real .blend environments: find where litter can lie, and load only that corner.

Two problems with dropping trash into downloaded scenes:

1. Someone has to mark where trash may land. Manual `spawn_*` planes still work, but
   `find_ground_patches` also finds them automatically: flat, upward-facing ground
   with nothing above it in the immediate neighbourhood (so not a table top or a car
   roof).
2. Large scenes do not fit in memory - forest_2.blend took Blender to 13 GB during
   the drop simulation and was killed. Litter photographed from 10 cm never shows
   anything 30 m away anyway, so `prepare_scenes.py` records, per spawn patch, which
   objects lie within `crop_radius_m`, and the generator loads only those.

The per-scene result lives next to the scene as `<scene>.zones.json`.
"""

import json
import os
import re

import blenderproc as bproc
import numpy as np
from mathutils import Vector

from . import config, environment
from .environment import Environment


def zones_file(scene_path):
    return scene_path + ".zones.json"


# ---------------------------------------------------------------- preparation

def _ray_down(depsgraph, x, y, z_top):
    import bpy  # pylint: disable=import-outside-toplevel
    ok, loc, normal, _, _, _ = bpy.context.scene.ray_cast(
        depsgraph, Vector((x, y, z_top)), Vector((0, 0, -1)))
    if not ok:
        return None
    return float(loc[2]), float(normal[2])


def find_ground_patches(meshes, cfg, rng):
    """Flat patches of ground big enough for a spawn zone.

    One downward ray per grid cell gives a height map; window filters over it then
    find cells whose whole patch is flat, faces up, and sits near the lowest ground
    around it (which rules out table tops, benches and car roofs).
    """
    from scipy import ndimage  # pylint: disable=import-outside-toplevel

    import bpy  # pylint: disable=import-outside-toplevel
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners = np.concatenate([np.asarray(o.get_bound_box()) for o in meshes])
    low, high = corners.min(axis=0), corners.max(axis=0)
    # Downloaded scenes often sit on a terrain plane hundreds of metres wide; search
    # the middle of the scene only, where the modelled detail is.
    centre = np.median(corners, axis=0)
    half_span = float(cfg.get("search_half_m", 60.0))
    low[:2] = np.maximum(low[:2], centre[:2] - half_span)
    high[:2] = np.minimum(high[:2], centre[:2] + half_span)
    z_top = float(high[2]) + 1.0
    step = float(cfg["grid_step_m"])
    xs = np.arange(low[0], high[0], step)
    ys = np.arange(low[1], high[1], step)
    heights = np.full((len(xs), len(ys)), np.nan)
    normal_z = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hit = _ray_down(depsgraph, x, y, z_top)
            if hit is not None:
                heights[i, j], normal_z[i, j] = hit

    patch_cells = max(int(round(float(cfg["patch_size_m"]) / step)), 1) | 1  # odd
    local_cells = max(int(round(10.0 / step)), 3) | 1
    filled_hi = np.where(np.isnan(heights), np.inf, heights)
    filled_lo = np.where(np.isnan(heights), -np.inf, heights)
    win_max = ndimage.maximum_filter(filled_lo, size=patch_cells, mode="constant", cval=np.inf)
    win_min = ndimage.minimum_filter(filled_hi, size=patch_cells, mode="constant", cval=-np.inf)
    win_nz = ndimage.minimum_filter(normal_z, size=patch_cells, mode="constant", cval=0.0)
    local_ground = ndimage.minimum_filter(filled_hi, size=local_cells, mode="nearest")

    ok = ((win_max - win_min) <= float(cfg["flatness_tol_m"])) & (win_nz >= 0.95) \
        & ((heights - local_ground) <= float(cfg["max_above_local_ground_m"])) \
        & np.isfinite(heights)
    candidates = [(float(xs[i]), float(ys[j]), float(heights[i, j])) for i, j in zip(*np.nonzero(ok))]

    rng.shuffle(candidates)
    chosen = []
    min_sep = float(cfg["min_separation_m"])
    for p in candidates:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in chosen):
            chosen.append(p)
        if len(chosen) >= int(cfg["max_patches"]):
            break
    return chosen


def objects_near(meshes, point, radius):
    names = []
    for obj in meshes:
        bb = np.asarray(obj.get_bound_box())
        low, high = bb.min(axis=0), bb.max(axis=0)
        nearest = np.clip(np.asarray(point[:2]), low[:2], high[:2])
        if np.linalg.norm(nearest - np.asarray(point[:2])) <= radius:
            names.append(obj.get_name())
    return names


# --------------------------------------------------------------------- render

def _readable(path):
    """Blender 4.2 cannot open files saved by Blender 5 ("BLENDER17-01" header)."""
    with open(path, "rb") as f:
        head = f.read(12)
    if head.startswith(b"BLENDER17"):
        print(f"[scene] skipping {path}: saved by Blender 5, BlenderProc runs 4.2 - "
              f"re-save it from Blender 4.2 (File > Save As)")
        return False
    return True


def _pick_scene(cfg, rng):
    paths = config.resolve_all(cfg["scene"]["files"])
    paths = [p for p in paths if os.path.isfile(p) and _readable(p)]
    if not paths:
        raise FileNotFoundError(f"none of scene.files exist: {cfg['scene']['files']}")
    return paths[int(rng.integers(0, len(paths)))]


def build_environment(cfg, rng):
    scene_cfg = cfg["scene"]
    path = _pick_scene(cfg, rng)
    obj_types = ["mesh"] + (["light"] if scene_cfg.get("load_scene_lights") else [])

    prepared = None
    if os.path.isfile(zones_file(path)):
        with open(zones_file(path), encoding="utf-8") as f:
            prepared = json.load(f)

    if prepared and prepared.get("zones"):
        zone_spec = prepared["zones"][int(rng.integers(0, len(prepared["zones"])))]
        pattern = "^(" + "|".join(re.escape(n) for n in zone_spec["objects"]) + ")$"
        print(f"[scene] loading {path} - {len(zone_spec['objects'])} object(s) near "
              f"zone {zone_spec['name']}")
        loaded = bproc.loader.load_blend(path, obj_types=obj_types, name_regrex=pattern)
        meshes = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
        zones = environment.find_spawn_zones(meshes, scene_cfg["spawn_zone_pattern"])
        if not zones:
            size = zone_spec["size"]
            zone = bproc.object.create_primitive("PLANE")
            zone.set_name(f"spawn_auto_{zone_spec['name']}")
            zone.set_scale([size[0] / 2.0, size[1] / 2.0, 1.0])
            zone.set_location(zone_spec["center"])
            zones = [zone]
        description = f"{os.path.basename(path)}#{zone_spec['name']}"
    else:
        print(f"[scene] loading {path} (no .zones.json - whole scene, manual zones only)")
        loaded = bproc.loader.load_blend(path, obj_types=obj_types)
        meshes = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
        zones = environment.find_spawn_zones(meshes, scene_cfg["spawn_zone_pattern"])
        if not zones:
            raise RuntimeError(f"No spawn zones in {path}: run prepare_scenes.py on it, or "
                               f"add a flat plane named e.g. 'spawn_pavement'.")
        description = os.path.basename(path)

    statics = [o for o in meshes if o not in zones]
    for obj in statics:
        obj.set_cp("category_id", 0)
    for zone in zones:
        environment.prepare_zone(zone)

    colliders = [
        environment.build_collider(zone, mode=cfg["physics"]["collider"],
                                   thickness=cfg["physics"]["collider_thickness_m"],
                                   friction=cfg["physics"]["ground_friction"],
                                   margin=cfg["physics"]["collider_margin_m"])
        for zone in zones
    ]

    env = Environment(zones=zones, colliders=colliders, statics=statics,
                      description=description)
    zone_z = float(np.mean([np.asarray(z.get_bound_box())[:, 2].max() for z in zones]))

    def ground(x, y):
        # Blender's scene ray cast rather than a BVH copy of a possibly huge scene.
        import bpy  # pylint: disable=import-outside-toplevel
        ok, loc, normal, _, obj, _ = bpy.context.scene.ray_cast(
            bpy.context.evaluated_depsgraph_get(), Vector((x, y, zone_z + 3.0)), Vector((0, 0, -1)))
        if not ok or normal[2] < 0.8 or (obj is not None and obj.name.startswith("trash_")):
            return None
        return float(loc[2])

    env.ground_fn = ground
    env.meta = {"scene_file": path, "zones": [z.get_name() for z in zones],
                "static_objects": len(statics)}
    print(f"[scene] {len(zones)} spawn zone(s), {len(statics)} static mesh(es)")
    return env
