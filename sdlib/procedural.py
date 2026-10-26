"""Procedural outdoor environments: a believable patch of the world around the litter.

The problem this solves: a camera 10 cm above a flat textured plane under an HDRI
sees one texture repeating to the horizon and nothing else. Real robot footage has
curbs, gutters, grass blades in front of the lens, weeds, stones, bins, walls, trees
and shadows. So every scene builds a small world from parts:

* a **layout** of parallel ground strips (road | curb | sidewalk | verge | wall ...),
  each a solid box with its own PBR material - also the physics floor;
* **vegetation**: 3D grass and moss instanced by geometry nodes on grassy strips;
* **clutter** near the litter: stones, weeds, branches, roots - never annotated,
  they are the hard negatives;
* **props** further out: bins, hydrants, lamps, benches, barriers, trees;
* **facades** - large textured walls that fill the horizon in urban biomes;
* an optional **canopy** that throws dappled shadows from outside the frame.

The biome (sidewalk, lawn, park path, forest, ...) decides which of those appear
and with what materials. Everything is drawn from `assets/` via sdlib.library.

Coordinates: x runs across the strips, y along them, z up. The litter is dropped
into spawn zones around y = 0 on the biome's focus strips.
"""

import blenderproc as bproc
import bpy
import numpy as np

from . import environment, realism, scatter
from .environment import Environment
from .library import MaterialLibrary, ModelLibrary

# Keywords picking a fitting texture out of the "ground" pool for each surface role.
SURFACE_KEYWORDS = {
    "asphalt": ["asphalt", "road"],
    "paving": ["paving", "pavement", "sidewalk", "tiles", "brick", "cobble", "floor",
               "concrete", "herringbone", "tactile"],
    "curb": ["concrete", "curb", "kerb", "stone"],
    "grass": ["grass", "lawn", "leafy"],
    "dirt": ["dirt", "soil", "ground", "mud", "clay", "park"],
    "gravel": ["gravel", "pebble", "rocky", "stones", "rock"],
    "path": ["gravel", "dirt", "path", "woodchip", "wood_chip", "park", "pathway"],
    "forest": ["forest", "leaves", "leaf", "moss", "bark", "decay", "needle", "root"],
    "plaza": ["cobble", "paving", "tiles", "brick", "floor", "concrete"],
    "any": [],
}

# Surfaces an item can sink into, and how deep (fraction of its height).
SOFT_SURFACES = {"grass": 0.25, "dirt": 0.2, "forest": 0.3, "gravel": 0.15, "path": 0.12}


class Strip:
    def __init__(self, name, x0, x1, top, surface, grassy=False, spawn=0.0, props=True,
                 clutter=1.0, walkable=True):
        self.name, self.x0, self.x1, self.top = name, float(x0), float(x1), float(top)
        self.surface = surface
        self.grassy = grassy
        self.spawn = spawn          # relative chance a spawn zone lands on this strip
        self.props = props          # may props stand here
        self.clutter = clutter      # clutter density multiplier
        self.walkable = walkable    # False behind a facade: no robot stands in a building
        self.obj = None

    def contains(self, x):
        return self.x0 <= x < self.x1

    def to_dict(self):
        return {"name": self.name, "x": [round(self.x0, 3), round(self.x1, 3)],
                "top": round(self.top, 3), "surface": self.surface, "grassy": self.grassy}


# ------------------------------------------------------------------------ layouts

def _urban(rng, extent, focus):
    """Building | (verge) | sidewalk | curb | road | curb | sidewalk | (building)."""
    curb_h = float(rng.uniform(0.08, 0.16))
    sw = float(rng.uniform(1.6, 4.5))
    cw = float(rng.uniform(0.12, 0.3))
    road_w = float(rng.uniform(5.5, 9.0))
    verge_w = float(rng.uniform(0.8, 3.5)) if rng.random() < 0.55 else 0.0
    verge_surface = str(rng.choice(["grass", "dirt", "gravel"], p=[0.6, 0.25, 0.15]))

    strips, x = [], -sw / 2.0
    left_end = x - verge_w
    if verge_w:
        strips.append(Strip("verge", left_end, x, curb_h, verge_surface,
                            grassy=verge_surface == "grass", spawn=0.6 if focus != "road" else 0.0))
    strips.append(Strip("sidewalk", x, x + sw, curb_h, "paving",
                        spawn=1.0 if focus != "road" else 0.3, clutter=0.3))
    x += sw
    strips.append(Strip("curb", x, x + cw, curb_h + float(rng.uniform(0.0, 0.02)), "curb",
                        props=False, clutter=0.0))
    x += cw
    # The gutter - the strip of road along the curb - is where litter collects.
    strips.append(Strip("road", x, x + road_w, 0.0, "asphalt",
                        spawn=1.0 if focus == "road" else 0.5, props=False, clutter=0.2))
    x += road_w
    strips.append(Strip("curb_far", x, x + cw, curb_h, "curb", props=False, clutter=0.0))
    x += cw
    strips.append(Strip("sidewalk_far", x, x + float(rng.uniform(2.0, 4.0)), curb_h, "paving",
                        clutter=0.2))
    far_end = strips[-1].x1

    facades = []
    if rng.random() < 0.8:
        facades.append(("left", left_end))
    else:
        strips.insert(0, Strip("lawn_left", -extent, left_end, curb_h, "grass", grassy=True))
    if rng.random() < 0.6:
        facades.append(("right", far_end))
    else:
        strips.append(Strip("lawn_right", far_end, extent, curb_h, "grass", grassy=True))
    if facades and facades[0][0] == "left":
        strips.insert(0, Strip("behind_left", -extent, left_end, curb_h, "paving", props=False,
                               clutter=0.0, walkable=False))
    if any(side == "right" for side, _ in facades):
        strips.append(Strip("behind_right", far_end, extent, curb_h, "paving", props=False,
                            clutter=0.0, walkable=False))
    return strips, facades


def _lawn(rng, extent, with_path):
    strips = []
    if with_path:
        pw = float(rng.uniform(1.0, 3.5))
        off = float(rng.uniform(-1.5, 1.5))
        surface = str(rng.choice(["path", "gravel", "paving", "dirt"], p=[0.35, 0.25, 0.25, 0.15]))
        top = float(rng.uniform(-0.02, 0.03))
        strips += [Strip("lawn_left", -extent, off - pw / 2, 0.0, "grass", grassy=True, spawn=0.6),
                   Strip("path", off - pw / 2, off + pw / 2, top, surface, spawn=1.0, clutter=0.6),
                   Strip("lawn_right", off + pw / 2, extent, 0.0, "grass", grassy=True, spawn=0.6)]
    else:
        strips.append(Strip("lawn", -extent, extent, 0.0, "grass", grassy=True, spawn=1.0))
    return strips, []


def _single(rng, extent, surface, name, grassy=False):
    return [Strip(name, -extent, extent, 0.0, surface, grassy=grassy, spawn=1.0)], []


BIOMES = {
    "sidewalk": lambda rng, e: _urban(rng, e, "sidewalk"),
    "street": lambda rng, e: _urban(rng, e, "road"),
    "lawn": lambda rng, e: _lawn(rng, e, with_path=rng.random() < 0.35),
    "park_path": lambda rng, e: _lawn(rng, e, with_path=True),
    "forest": lambda rng, e: _single(rng, e, "forest", "forest_floor"),
    "gravel": lambda rng, e: _single(rng, e, str(rng.choice(["gravel", "dirt"])), "gravel"),
    "plaza": lambda rng, e: (_single(rng, e, "plaza", "plaza")[0],
                             [("left", float(rng.uniform(-9, -4)))] if rng.random() < 0.7 else []),
    "studio": lambda rng, e: _single(rng, e, "any", "studio"),
}

# Per biome: prop groups and how many, clutter level, grass style.
BIOME_STYLE = {
    "sidewalk": {"props": {"urban_props": (3, 9), "trees": (0, 3)}, "clutter": 0.4},
    "street": {"props": {"urban_props": (3, 8), "trees": (0, 2)}, "clutter": 0.4},
    "lawn": {"props": {"urban_props": (0, 3), "trees": (1, 5)}, "clutter": 0.8},
    "park_path": {"props": {"urban_props": (1, 4), "trees": (2, 6)}, "clutter": 0.8},
    "forest": {"props": {"trees": (4, 10), "rocks": (1, 4)}, "clutter": 1.6},
    "gravel": {"props": {"rocks": (1, 4), "urban_props": (0, 2)}, "clutter": 1.2},
    "plaza": {"props": {"urban_props": (4, 10), "trees": (0, 3)}, "clutter": 0.3},
    "studio": {"props": {}, "clutter": 0.5},
}


# ------------------------------------------------------------------------- build

def _box_mesh(name, x0, x1, y0, y1, z0, z1):
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj["category_id"] = 0
    return bproc.types.MeshObject(obj)


class ProceduralWorld:
    """Builds one procedural scene and serves the pipeline's hooks for it."""

    def __init__(self, cfg, index, rng):
        self.cfg = cfg["procedural"]
        self.full_cfg = cfg
        self.index = index
        self.rng = rng
        self.materials = MaterialLibrary(index, rng, default_tile_m=self.cfg.get("default_tile_m", 2.0))
        self.models = ModelLibrary(index, rng)
        self.meta = {}

    # ---------------------------------------------------------------- layout

    def build(self):
        cfg, rng = self.cfg, self.rng
        names = sorted(cfg["biome_weights"])
        probs = np.array([float(cfg["biome_weights"][n]) for n in names])
        self.biome = str(rng.choice(names, p=probs / probs.sum()))
        extent = float(cfg["extent_m"])
        self.strips, facades = BIOMES[self.biome](rng, extent)
        self.style = BIOME_STYLE[self.biome]
        self.extent = extent

        surface_mats = {}
        statics = []
        for strip in self.strips:
            if strip.surface not in surface_mats:
                group = "ground"
                if strip.surface == "any":
                    surface_mats[strip.surface] = self.materials.pick(group)
                else:
                    surface_mats[strip.surface] = self.materials.pick_matching(
                        group, SURFACE_KEYWORDS[strip.surface])
            obj = _box_mesh(f"ground_{strip.name}", strip.x0, strip.x1, -extent, extent,
                            strip.top - 0.3, strip.top)
            obj.blender_obj.data.materials.append(surface_mats[strip.surface])
            obj.enable_rigidbody(active=False, collision_shape="BOX",
                                 friction=self.full_cfg["physics"]["ground_friction"])
            strip.obj = obj
            statics.append(obj)

        self.facade_objs = []
        for side, x_edge in facades:
            height = float(rng.uniform(*cfg["facade_height_m"]))
            depth = 1.0
            x0, x1 = (x_edge - depth, x_edge) if side == "left" else (x_edge, x_edge + depth)
            wall = _box_mesh(f"facade_{side}", x0, x1, -extent, extent, -0.3, height)
            wall.blender_obj.data.materials.append(
                self.materials.pick("facade", projection="BOX", tile_scale=(0.9, 1.6)))
            self.facade_objs.append(wall)
            statics.append(wall)

        # Spawn zones on the focus strips, all within a few metres of the origin so
        # cameras see several items together.
        zones = []
        focus_half = float(rng.uniform(*cfg["focus_half_m"]))
        weights = np.array([s.spawn for s in self.strips])
        n_zones = int(rng.integers(cfg["zones"][0], cfg["zones"][1] + 1))
        for k in range(n_zones):
            if weights.sum() <= 0:
                break
            strip = self.strips[int(rng.choice(len(self.strips), p=weights / weights.sum()))]
            x0 = max(strip.x0, -focus_half)
            x1 = min(strip.x1, focus_half)
            if strip.name == "road":
                x1 = min(x1, strip.x0 + 1.5)  # the gutter, not the middle of the road
            if x1 - x0 < 0.2:
                continue
            y_c = float(rng.uniform(-focus_half / 2, focus_half / 2))
            y_half = float(rng.uniform(0.6, focus_half))
            zone = bproc.object.create_primitive("PLANE")
            zone.set_name(f"spawn_{strip.name}_{k}")
            zone.set_scale([(x1 - x0) / 2.0, y_half, 1.0])
            zone.set_location([(x0 + x1) / 2.0, y_c, strip.top])
            environment.prepare_zone(zone)
            zones.append(zone)

        if not zones:
            # Every draw landed on a strip too narrow inside the focus area (a curb, a
            # thin verge). Fall back to the widest spawn strip rather than fail.
            strip = max((s for s in self.strips if s.spawn > 0), key=lambda s: min(s.x1, focus_half) - max(s.x0, -focus_half))
            x0, x1 = max(strip.x0, -focus_half), min(strip.x1, focus_half)
            zone = bproc.object.create_primitive("PLANE")
            zone.set_name(f"spawn_{strip.name}_fallback")
            zone.set_scale([max(x1 - x0, 0.3) / 2.0, focus_half / 2.0, 1.0])
            zone.set_location([(x0 + x1) / 2.0, 0.0, strip.top])
            environment.prepare_zone(zone)
            zones.append(zone)

        colliders = [s.obj for s in self.strips]
        self.meta = {"biome": self.biome, "strips": [s.to_dict() for s in self.strips],
                     "facades": [{"side": s, "x": round(x, 3)} for s, x in facades],
                     "focus_half_m": round(focus_half, 3)}
        self.focus_half = focus_half

        self.pool = self._sample_pool()
        props = self._place_props()
        statics += props

        # Everything the later stages will instance or copy is chosen now, so the rest
        # of what the model files brought in can be dropped before the simulation.
        self.clutter_items = {}
        for label, group in (("rocks_small", "rocks"), ("ground_cover", "ground_cover")):
            items = self._items(group, max_tris=60000)
            if label == "rocks_small":
                # Only stones a robot could drive past: big boulders are props.
                items = [it for it in items
                         if float(np.max(np.ptp(np.vstack(self.models.item_size(it)), axis=0))) < 0.6]
            self.clutter_items[label] = [[o.name for o in it] for it in items]
        self.grass_protos = []
        if any(s.grassy for s in self.strips):
            for parts in self._items("grass", max_tris=4000):
                self.grass_protos += [o.name for o in parts]
        self.meta["purged_objects"] = self.models.purge_unused()
        env = Environment(zones=zones, colliders=colliders, statics=statics,
                          description=f"procedural:{self.biome}")
        env.ground_fn = self.ground_height
        env.after_physics = self.after_physics
        env.after_cameras = self.after_cameras
        env.meta = self.meta
        return env

    def strip_at(self, x):
        return next((s for s in self.strips if s.contains(x)), None)

    def ground_height(self, x, y):
        if abs(y) > self.extent:
            return None
        strip = self.strip_at(x)
        return None if strip is None or not strip.walkable else strip.top

    # ----------------------------------------------------------------- props

    def _sample_pool(self):
        """Which model files of each group this scene may use.

        A few per group, not all of them: loading every tree and every prop file for
        every scene costs minutes and gigabytes, and the variety comes from the many
        scenes anyway.
        """
        counts = self.cfg.get("models_per_scene", {})
        pool = {}
        for group in self.index.index.get("polyhaven_models", {}):
            ids = self.index.ids("polyhaven_models", group)
            k = min(int(counts.get(group, len(ids))), len(ids))
            pool[group] = sorted(self.rng.choice(ids, size=k, replace=False).tolist()) if k else []
        self.meta["model_pool"] = pool
        return pool

    def _items(self, group, max_tris=None, prefer_lod=0):
        items = []
        for asset_id in self.pool.get(group, []):
            items += self.models.variants(asset_id, max_tris=max_tris, prefer_lod=prefer_lod)
        return items

    @staticmethod
    def _resolve(names_lists):
        import bpy  # pylint: disable=import-outside-toplevel
        return [[bpy.data.objects[n] for n in names if n in bpy.data.objects]
                for names in names_lists]

    def _place_props(self):
        rng, cfg = self.rng, self.cfg
        placed, records = [], []
        r_min = self.focus_half + float(cfg["props_min_gap_m"])
        r_max = float(cfg["props_max_dist_m"])
        for group, (lo, hi) in self.style["props"].items():
            # Props stand metres away; their lightest LOD is indistinguishable there and
            # keeps a forest of 1.6M-triangle trees from exhausting memory.
            items = self._items(group, prefer_lod=3)
            if not items:
                continue
            count = int(rng.integers(lo, hi + 1))
            pts = []
            for _ in range(count * 6):
                if len(pts) >= count:
                    break
                r = float(rng.uniform(r_min, r_max))
                a = float(rng.uniform(0, 2 * np.pi))
                x, y = r * np.cos(a), r * np.sin(a)
                strip = self.strip_at(x)
                if strip is None or not strip.props:
                    continue
                if group == "trees" and strip.surface in ("paving", "asphalt", "plaza", "curb"):
                    continue
                pts.append((x, y, strip.top))
            objs, recs = scatter.place_items(self.models, items, pts, rng, name=f"prop_{group}_")
            placed += objs
            records += [{**r, "group": group} for r in recs]
        self.meta["props"] = records
        return [bproc.types.MeshObject(o) for o in placed if o.type == "MESH"]

    # ---------------------------------------------------------------- hooks

    def _footprints(self, trash, shrink):
        discs = []
        for obj in trash:
            bb = np.asarray(obj.get_bound_box())
            low, high = bb.min(axis=0), bb.max(axis=0)
            r = 0.5 * float(np.linalg.norm(high[:2] - low[:2])) * shrink
            discs.append(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, r))
        return discs

    def after_physics(self, trash, rng):
        """Bury, then add clutter around the fallen litter. Returns new static objects."""
        cfg = self.cfg
        out = {"buried": {}}
        for obj in trash:
            bb = np.asarray(obj.get_bound_box())
            centre = (bb.min(axis=0) + bb.max(axis=0)) / 2
            strip = self.strip_at(centre[0])
            if strip is not None and strip.surface in SOFT_SURFACES and rng.random() < cfg["bury_probability"]:
                out["buried"][obj.get_name()] = round(realism.bury(obj, SOFT_SURFACES[strip.surface], rng), 4)

        discs = self._footprints(trash, shrink=1.0)
        level = float(self.style["clutter"]) * float(rng.uniform(*cfg["clutter_level"]))
        new_statics, records = [], []
        for label, weight in (("rocks_small", 0.6), ("ground_cover", 1.0)):
            items = [it for it in self._resolve(self.clutter_items.get(label, [])) if it]
            if not items:
                continue
            density = float(cfg["clutter_density_per_m2"]) * level * weight

            def region(x, y):
                strip = self.strip_at(x)
                if strip is None or rng.random() > strip.clutter:
                    return None
                return strip.top

            pts = scatter.sample_points(region, (0.0, 0.0), float(cfg["clutter_radius_m"]),
                                        density, rng, discs=discs, falloff=0.5)
            objs, recs = scatter.place_items(self.models, items, pts, rng,
                                             scale_range=cfg["clutter_scale"],
                                             sink_frac=(0.0, 0.15), name=f"clutter_{label}_")
            new_statics += [bproc.types.MeshObject(o) for o in objs if o.type == "MESH"]
            records += [{**r, "group": label} for r in recs]
        self.meta["clutter"] = {"level": round(level, 3), "count": len(records)}
        if rng.random() < float(cfg["canopy_probability"]):
            self.meta["canopy"] = realism.add_canopy((0.0, 0.0, 0.0), cfg["canopy"], rng)
        out["statics"] = new_statics
        return out

    def after_cameras(self, trash, frames, rng):
        """Grow grass last, clear of the litter and of the robot's own position."""
        cfg = self.cfg["grass"]
        grassy = [s for s in self.strips if s.grassy]
        if not grassy:
            return {}
        import bpy  # pylint: disable=import-outside-toplevel
        protos = [bpy.data.objects[n] for n in self.grass_protos if n in bpy.data.objects]
        if not protos:
            return {}
        for proto in protos:
            proto["category_id"] = 0

        style = str(rng.choice(sorted(cfg["styles"])))
        scale_range = cfg["styles"][style]
        density = float(rng.uniform(*cfg["density_per_m2"]))
        discs = self._footprints(trash, shrink=float(cfg["trash_clearance"]))
        for frame in frames:
            if frame["rig"] == "robot":
                m = np.asarray(frame["cam2world"])
                discs.append((m[0, 3], m[1, 3], float(cfg["robot_clearance_m"])))

        def region(x, y):
            strip = self.strip_at(x)
            return strip.top if strip is not None and strip.grassy else None

        pts = scatter.sample_points(region, (0.0, 0.0), float(cfg["radius_m"]), density, rng,
                                    discs=discs, falloff=float(cfg["falloff"]))
        scatter.instance_field("grass", pts, protos, scale_range, rng, tilt=0.25,
                               library_collection=self.models.collection)
        info = {"style": style, "density_per_m2": round(density, 1), "instances": int(len(pts))}
        self.meta["grass"] = info
        return info


def build_environment(cfg, rng, index):
    world = ProceduralWorld(cfg, index, rng)
    env = world.build()
    env.world = world
    return env
