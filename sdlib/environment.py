"""World lighting, spawn zones, physics proxies and the drop simulation."""

import glob
import math
import os

import blenderproc as bproc
import bpy
import numpy as np

HDRI_EXTENSIONS = (".hdr", ".exr")


class Environment:
    """Everything the pipeline needs to know about the world it is dropping trash into.

    zones     - invisible surfaces marking where trash is allowed to land
    colliders - invisible passive rigid bodies the trash actually lands on
    statics   - visible scene geometry (background, labelled category_id 0)

    Builders can also provide:
    ground_fn(x, y)                      height of the walkable ground, or None where a
                                         robot cannot stand (inside a wall, off the map)
    after_physics(trash, rng)            runs once the litter has landed; may return
                                         {"statics": [...]} to add to the ray checks
    after_cameras(trash, frames, rng)    runs once the views are fixed (grass is grown
                                         here, clear of where the robot stands)
    meta                                 dict merged into scene_meta.json
    """

    def __init__(self, zones, colliders, statics, description=""):
        self.zones = zones
        self.colliders = colliders
        self.statics = statics
        self.description = description
        self.ground_fn = None
        self.after_physics = None
        self.after_cameras = None
        self.meta = {}

    def all_objects(self):
        return list(self.zones) + list(self.colliders) + list(self.statics)

    def ground_height(self, x, y):
        return self.ground_fn(x, y) if self.ground_fn is not None else None


# --------------------------------------------------------------------------- HDRI

def list_hdris(folders):
    """Every .hdr/.exr under the given folders, at any depth.

    Deliberately not `bproc.loader.get_random_world_background_hdr_img_path_from_haven`:
    that helper only globs `<root>/hdris/*/*.hdr`, so it silently finds nothing for
    both the flat layout in resources/haven/hdris and the BlenderKit layout.
    """
    found = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"[hdri] WARNING folder does not exist, skipping: {folder}")
            continue
        for ext in HDRI_EXTENSIONS:
            found.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
    found = sorted(set(found))
    if not found:
        raise RuntimeError(f"No .hdr/.exr files found under {folders}")
    return found


def is_dark_hdri(path, patterns):
    name = os.path.basename(path).lower()
    return any(p.lower() in name for p in patterns)


def setup_world(hdri_paths, cfg, rng, dark_names=frozenset()):
    """Pick an HDRI, set it with a random yaw, and report what was chosen.

    `dark_names` are HDRI folder/file stems known to be night shots (from the asset
    manifest's categories); filename patterns from the config catch the rest.
    """
    path = hdri_paths[int(rng.integers(0, len(hdri_paths)))]
    strength = float(rng.uniform(*cfg["strength_range"]))
    yaw_deg = float(rng.uniform(*cfg["rotation_range_deg"]))

    # Rotating the environment moves the sun and every shadow with it. This is the
    # cheapest variation axis available.
    bproc.world.set_world_background_hdr_img(
        path, strength=strength, rotation_euler=[0.0, 0.0, math.radians(yaw_deg)])

    stem = os.path.basename(os.path.dirname(path))
    is_dark = is_dark_hdri(path, cfg.get("dark_patterns", [])) or stem in dark_names
    return {"path": path, "strength": strength, "rotation_deg": yaw_deg, "is_dark": is_dark}


def add_lights(cfg, poi, rng):
    """Add point lights. Only meaningful for night/indoor HDRIs.

    Under a daylight HDRI the environment already provides physically correct sun
    and sky; extra point lights cast a second set of shadows with no visible source,
    which is a domain gap a detector will happily learn.
    """
    lights = []
    count = int(rng.integers(cfg["count_range"][0], cfg["count_range"][1] + 1))
    for _ in range(count):
        location = bproc.sampler.sphere(
            center=poi, radius=float(rng.uniform(*cfg["radius_range"])), mode="SURFACE")
        # Keep lights above the ground - a lamp under the street lights nothing.
        location[2] = abs(location[2] - poi[2]) + poi[2]
        energy = float(rng.uniform(*cfg["energy_range"]))

        light = bproc.types.Light()
        light.set_type(cfg.get("type", "POINT"))
        light.set_location(location)
        light.set_energy(energy)
        lights.append({"location": np.asarray(location).tolist(), "energy": energy})
    return lights


# --------------------------------------------------------------------- spawn zones

def find_spawn_zones(objects, pattern):
    zones = bproc.filter.by_attr(objects, "name", pattern, regex=True)
    return [z for z in zones if isinstance(z, bproc.types.MeshObject)]


def prepare_zone(zone):
    """A zone is a marker, never a rendered surface and never an annotation."""
    zone.hide(True)
    zone.set_cp("category_id", 0)


def build_collider(zone, mode, thickness, friction, margin=0.0):
    """Return the passive rigid body that trash dropped into `zone` lands on.

    "box" (default) replaces the zone with a thin, invisible box whose top face sits
    exactly on the zone surface. It is cheap, cannot be tunnelled through by a
    falling object, and is immune to whatever the scene's real ground mesh looks
    like. It assumes the zone is roughly flat and axis-aligned.

    "mesh" turns the zone mesh itself into the collider instead, which is the right
    choice for sloped or uneven zones at the cost of a slower simulation.

    `margin` extends the box past the zone in x and y. Without it an item that rolls
    a few centimetres past the edge finds no floor at all - the scene's own geometry
    has no rigid body unless physics.collide_with_scene is on - and falls out of the
    world.
    """
    if mode == "mesh":
        zone.enable_rigidbody(active=False, collision_shape="MESH", friction=friction)
        return zone

    if mode != "box":
        raise ValueError(f"Unknown physics.collider mode {mode!r} (expected 'box' or 'mesh')")

    bb = np.asarray(zone.get_bound_box())
    low, high = bb.min(axis=0), bb.max(axis=0)
    size = np.maximum(high - low, 1e-4) + np.array([2 * margin, 2 * margin, 0.0])
    center = (low + high) / 2.0

    box = bproc.object.create_primitive("CUBE")  # default cube spans -1..1 on each axis
    box.set_name(f"collider_{zone.get_name()}")
    box.set_scale([size[0] / 2.0, size[1] / 2.0, thickness / 2.0])
    box.set_location([center[0], center[1], high[2] - thickness / 2.0])
    box.hide(True)
    box.set_cp("category_id", 0)
    box.enable_rigidbody(active=False, collision_shape="BOX", friction=friction)
    return box


# ------------------------------------------------------------------------ dropping

def place_bbox_bottom_at(obj, target):
    """Move `obj` so the bottom-centre of its world bounding box sits at `target`.

    Assets have arbitrary origins - some at the geometric centre, some metres away.
    Positioning by origin would bury half of them in the ground and leave the rest
    floating, so we position by bounding box instead. Call this *after* setting the
    rotation, since rotating changes the bounding box.
    """
    bpy.context.view_layer.update()
    bb = np.asarray(obj.get_bound_box())
    low, high = bb.min(axis=0), bb.max(axis=0)
    anchor = np.array([(low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0, low[2]])
    obj.set_location(np.asarray(obj.get_location()) + (np.asarray(target) - anchor))
    bpy.context.view_layer.update()


def sample_drop_poses(objects, zone_per_object, obstacles, cfg, rng):
    """Give every object a random orientation and a non-overlapping start position
    above its zone, so the simulation begins from a valid state."""
    drop_low, drop_high = cfg["drop_height_range"]
    # Keep the drop point away from the very edge of the zone, so an object that lands
    # half-overhanging does not immediately topple out of it.
    inset = float(cfg["zone_inset_frac"])
    face_range = [inset, 1.0 - inset]

    def sample_pose(obj):
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
        # upper_region samples across the top face of the zone's bounding box, so a
        # zone should be a flat, roughly axis-aligned plane laid on the ground.
        point = bproc.sampler.upper_region(
            [zone_per_object[obj]],
            face_sample_range=face_range,
            min_height=drop_low,
            max_height=drop_high,
            use_upper_dir=True,
        )
        place_bbox_bottom_at(obj, point)

    return bproc.object.sample_poses(
        objects_to_sample=objects,
        sample_pose_func=sample_pose,
        objects_to_check_collisions=list(objects) + list(obstacles),
        max_tries=cfg["max_placement_tries"],
        mode_on_failure="last_pose",
    )


def simulate_drop(cfg):
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=cfg["min_simulation_time"],
        max_simulation_time=cfg["max_simulation_time"],
        check_object_interval=cfg["check_object_interval"],
        substeps_per_frame=cfg["substeps_per_frame"],
        solver_iters=cfg["solver_iters"],
    )


def discard_escapees(objects, zones, tolerance):
    """Delete objects that rolled off every zone or fell through the world.

    Physics occasionally throws an item off a curb or tunnels a thin one through a
    collider. Keeping it would put a label on something lying somewhere we never
    intended to photograph, so it is dropped from the scene entirely.
    """
    if not zones:
        return list(objects), []

    bounds = []
    for zone in zones:
        bb = np.asarray(zone.get_bound_box())
        bounds.append((bb.min(axis=0), bb.max(axis=0)))

    kept, removed = [], []
    for obj in objects:
        bb = np.asarray(obj.get_bound_box())
        centre = (bb.min(axis=0) + bb.max(axis=0)) / 2.0
        inside = any(
            low[0] - tolerance <= centre[0] <= high[0] + tolerance
            and low[1] - tolerance <= centre[1] <= high[1] + tolerance
            and low[2] - tolerance <= centre[2] <= high[2] + tolerance + 1.0
            for low, high in bounds
        )
        (kept if inside else removed).append(obj)

    for obj in removed:
        obj.delete(remove_all_offspring=True)
    if removed:
        print(f"[physics] discarded {len(removed)} object(s) that left their spawn zone")
    return kept, removed
