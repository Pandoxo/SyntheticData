"""Environment asset library: PBR ground materials and Poly Haven models.

Everything here reads the tree fetch_assets.py writes (`assets/index.json` says which
ids belong to which group). Two things are built from it:

* materials - a Principled BSDF wired to the image maps of one Poly Haven texture or
  ambientCG material, mapped in world-scale object coordinates so a 2 m tile is 2 m
  on the ground regardless of how big the mesh is, with a low-frequency colour
  variation on top. Without that variation a single texture repeated across 40 m of
  ground at a grazing 10 cm camera angle turns into an obvious grid.

* model variants - a Poly Haven .blend is not one object. It holds LOD chains,
  "_aged" duplicates, the source pieces of its geometry-nodes scatter, and often a
  whole set of separate items laid out side by side (six rocks, five dandelions).
  `ModelLibrary.variants(id)` reduces a file to a list of distinct physical items,
  each one a list of mesh parts that belong together (a hydrant and its caps).
"""

import glob
import json
import os
import re

import bpy
import numpy as np

from . import config

LOD_SUFFIX = re.compile(r"_LOD([0-9])$", re.IGNORECASE)
TEXTURE_KEYS = {
    # role: substrings identifying the file, Poly Haven first, ambientCG second
    "color": ("_diff_", "_Color."),
    "normal": ("_nor_gl_", "_NormalGL."),
    "rough": ("_rough_", "_Roughness."),
    "disp": ("_disp_", "_Displacement."),
    "ao": ("_ao_", "_AmbientOcclusion."),
    "metal": ("_metal_", "_Metalness."),
    "alpha": ("_alpha_", "_Opacity."),
}


class AssetIndex:
    """`assets/index.json` + `assets/manifest.jsonl`, with paths resolved."""

    def __init__(self, root):
        self.root = config.resolve(root)
        index_path = os.path.join(self.root, "index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{index_path} not found - run `.venv/bin/python fetch_assets.py` first")
        with open(index_path, encoding="utf-8") as f:
            self.index = json.load(f)
        self.manifest = {}
        manifest_path = os.path.join(self.root, "manifest.jsonl")
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.manifest[(row["source"], row["id"])] = row

    def ids(self, kind, group):
        return list(self.index.get(kind, {}).get(group, []))

    # ---------------------------------------------------------------- textures

    def material_ids(self, group):
        """Every downloaded material in `group`, as (source, id) pairs."""
        out = [("polyhaven", i) for i in self.ids("polyhaven_textures", group)]
        out += [("ambientcg", i) for i in self.ids("ambientcg", group)]
        return [(s, i) for s, i in out if self.texture_maps(s, i).get("color")]

    def texture_maps(self, source, asset_id):
        folder = (os.path.join(self.root, "polyhaven", "textures", asset_id) if source == "polyhaven"
                  else os.path.join(self.root, "ambientcg", asset_id))
        if not os.path.isdir(folder):
            return {}
        files = os.listdir(folder)
        maps = {}
        for role, keys in TEXTURE_KEYS.items():
            hit = next((f for f in sorted(files) if any(k in f for k in keys)), None)
            if hit:
                maps[role] = os.path.join(folder, hit)
        return maps

    def tile_size_m(self, source, asset_id, default):
        """Real-world size of one texture tile, from Poly Haven's metadata if known."""
        row = self.manifest.get((source, asset_id), {})
        dims = row.get("dimensions_mm")
        if dims and dims[0]:
            return float(dims[0]) / 1000.0
        return float(default)

    # ------------------------------------------------------------------- HDRIs

    def hdri_paths(self, groups=("outdoor", "skies")):
        paths = []
        for group in groups:
            for asset_id in self.ids("polyhaven_hdris", group):
                paths += glob.glob(os.path.join(self.root, "polyhaven", "hdris", asset_id, "*.hdr"))
        return sorted(paths)

    # ------------------------------------------------------------------ models

    def model_path(self, asset_id):
        hits = sorted(glob.glob(os.path.join(self.root, "polyhaven", "models", asset_id, "*.blend")))
        return hits[0] if hits else None


# ------------------------------------------------------------------- materials

def _image(path, non_color):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def make_pbr_material(name, maps, tile_m, rng, projection="FLAT", variation=0.12,
                      bump_strength=1.0):
    """Build a material from image maps, mapped at `tile_m` metres per repeat.

    Mapping runs on Object coordinates, so the mesh must be built at real size with
    an identity scale (all builders in procedural.py do this). `projection="BOX"`
    is for vertical surfaces (facades, curbs): the image is projected along whichever
    axis the face points.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    out = nodes.get("Material Output")

    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    inv = 1.0 / max(tile_m, 1e-3)
    mapping.inputs["Scale"].default_value = (inv, inv, inv)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, float(rng.uniform(0, 2 * np.pi)))
    mapping.inputs["Location"].default_value = tuple(rng.uniform(0, 1, 3))
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def tex(path, non_color):
        node = nodes.new("ShaderNodeTexImage")
        node.image = _image(path, non_color)
        node.projection = projection
        if projection == "BOX":
            node.projection_blend = 0.2
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    color = tex(maps["color"], False)
    base = color.outputs["Color"]

    # Low-frequency variation: a large noise field nudging value and hue across the
    # ground, so repeats of the same tile do not look identical. Features of a few
    # metres, i.e. bigger than any tile.
    if variation > 0:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = float(rng.uniform(0.15, 0.6))
        noise.inputs["Detail"].default_value = 2.0
        links.new(coord.outputs["Object"], noise.inputs["Vector"])
        hsv = nodes.new("ShaderNodeHueSaturation")
        ramp = nodes.new("ShaderNodeMapRange")
        ramp.inputs["To Min"].default_value = 1.0 - variation
        ramp.inputs["To Max"].default_value = 1.0 + variation
        links.new(noise.outputs["Fac"], ramp.inputs["Value"])
        links.new(ramp.outputs["Result"], hsv.inputs["Value"])
        hsv.inputs["Hue"].default_value = 0.5 + float(rng.uniform(-0.015, 0.015))
        links.new(base, hsv.inputs["Color"])
        base = hsv.outputs["Color"]

    if "ao" in maps:
        ao = tex(maps["ao"], True)
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 0.6
        links.new(base, mix.inputs["A"])
        links.new(ao.outputs["Color"], mix.inputs["B"])
        base = mix.outputs["Result"]
    links.new(base, bsdf.inputs["Base Color"])

    if "rough" in maps:
        rough = tex(maps["rough"], True)
        links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    if "metal" in maps:
        metal = tex(maps["metal"], True)
        links.new(metal.outputs["Color"], bsdf.inputs["Metallic"])

    normal_out = None
    if "normal" in maps:
        nrm_tex = tex(maps["normal"], True)
        nrm = nodes.new("ShaderNodeNormalMap")
        links.new(nrm_tex.outputs["Color"], nrm.inputs["Color"])
        normal_out = nrm.outputs["Normal"]
    if "disp" in maps:
        # Displacement as bump only. True displacement would need a dense mesh over
        # 40 m of ground; at the scales that matter (cobbles, gravel) bump plus the
        # normal map reads correctly from a few centimetres up.
        disp = tex(maps["disp"], True)
        bump = nodes.new("ShaderNodeBump")
        # Real relief on a road or a paving stone is millimetres. Scaling the bump with
        # the tile size made a 3 m asphalt tile look like molten plastic.
        bump.inputs["Strength"].default_value = 0.35 * bump_strength
        bump.inputs["Distance"].default_value = 0.004
        links.new(disp.outputs["Color"], bump.inputs["Height"])
        if normal_out is not None:
            links.new(normal_out, bump.inputs["Normal"])
        normal_out = bump.outputs["Normal"]
    if normal_out is not None:
        links.new(normal_out, bsdf.inputs["Normal"])

    mat.cycles.displacement_method = "BUMP"
    _ = out
    return mat


class MaterialLibrary:
    """Hands out PBR materials by group, built on demand and reused within a scene."""

    def __init__(self, index, rng, default_tile_m=2.0):
        self.index = index
        self.rng = rng
        self.default_tile_m = default_tile_m
        self.used = []

    def pick(self, groups, projection="FLAT", tile_scale=(0.8, 1.25), name=None):
        """A new material from a random asset in any of `groups` (list or str)."""
        if isinstance(groups, str):
            groups = [groups]
        pool = []
        for group in groups:
            pool += self.index.material_ids(group)
        if not pool:
            raise RuntimeError(f"No downloaded materials in group(s) {groups} - check "
                               f"asset_sources.yaml and rerun fetch_assets.py")
        source, asset_id = pool[int(self.rng.integers(0, len(pool)))]
        return self.build(source, asset_id, projection, tile_scale, name)

    def pick_matching(self, groups, keywords, **kwargs):
        """Like pick(), restricted to ids containing one of `keywords` when any do."""
        if isinstance(groups, str):
            groups = [groups]
        pool = []
        for group in groups:
            pool += self.index.material_ids(group)
        kws = [k.lower() for k in keywords]
        hits = [p for p in pool if any(k in p[1].lower() for k in kws)]
        if not hits:
            return self.pick(groups, **kwargs)
        source, asset_id = hits[int(self.rng.integers(0, len(hits)))]
        return self.build(source, asset_id, kwargs.get("projection", "FLAT"),
                          kwargs.get("tile_scale", (0.8, 1.25)), kwargs.get("name"))

    def build(self, source, asset_id, projection="FLAT", tile_scale=(0.8, 1.25), name=None):
        maps = self.index.texture_maps(source, asset_id)
        tile = self.index.tile_size_m(source, asset_id, self.default_tile_m)
        tile *= float(self.rng.uniform(*tile_scale))
        mat = make_pbr_material(name or f"{source}_{asset_id}", maps, tile, self.rng,
                                projection=projection)
        mat["sd_source"] = source
        mat["sd_asset"] = asset_id
        self.used.append({"source": source, "id": asset_id, "tile_m": round(tile, 3)})
        return mat


# ---------------------------------------------------------------------- models

def _tris(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def _aabb(obj):
    corners = np.array([obj.matrix_world @ v.co for v in obj.data.vertices]) if len(obj.data.vertices) \
        else np.zeros((1, 3))
    return corners.min(axis=0), corners.max(axis=0)


def _overlaps(a, b, pad=0.01):
    return bool(np.all(a[0] - pad <= b[1]) and np.all(b[0] - pad <= a[1]))


class ModelLibrary:
    """Loads Poly Haven model files once per scene and exposes their items.

    Prototype objects live in a collection excluded from the view layer: they never
    render themselves, but geometry nodes can instance them and `place()` can make
    linked copies (shared mesh data, so a hundred rocks cost one rock of memory).
    """

    COLLECTION = "_sd_library"

    def __init__(self, index, rng):
        self.index = index
        self.rng = rng
        self._variants = {}
        collection = bpy.data.collections.new(self.COLLECTION)
        # BlenderProc's load_blend purges every data block that became an orphan while
        # it ran; a fake user keeps the library out of that sweep.
        collection.use_fake_user = True
        bpy.context.scene.collection.children.link(collection)
        layer = bpy.context.view_layer.layer_collection.children[collection.name]
        layer.exclude = True

    @property
    def collection(self):
        # Looked up by name every time: BlenderProc's physics simulation invalidates
        # Python references to collections (the collection itself survives), so a
        # reference kept from before the drop raises "StructRNA has been removed".
        return bpy.data.collections[self.COLLECTION]

    def variants(self, asset_id, max_tris=None, prefer_lod=0):
        """Distinct items in a model file, each a list of mesh objects (prototypes).

        `prefer_lod` picks the level of detail closest to it from every LOD chain -
        a tree 15 m away needs its LOD2 (~100k triangles), not its 1.6M-triangle LOD0.
        """
        key = (asset_id, max_tris, prefer_lod)
        if key in self._variants:
            return [[bpy.data.objects[n] for n in group] for group in self._variants[key]]
        path = self.index.model_path(asset_id)
        if path is None:
            self._variants[key] = []
            return []

        loaded_key = ("_loaded", asset_id)
        if loaded_key in self._variants:
            loaded = [bpy.data.objects[n] for n in self._variants[loaded_key]]
        else:
            try:
                with bpy.data.libraries.load(path, link=False) as (src, dst):
                    dst.objects = list(src.objects)
            except OSError as err:
                # Saved by a Blender newer than 4.2; scenery can do without it.
                print(f"[library] skipping {asset_id}: {err}")
                self._variants[key] = []
                return []
            loaded = [o for o in dst.objects if o is not None]
            library = self.collection
            for obj in loaded:
                library.objects.link(obj)
                obj["category_id"] = 0
            self._variants[loaded_key] = [o.name for o in loaded]

        meshes = [o for o in loaded if o.type == "MESH"]
        # Instance-source pieces of the file's own geometry-nodes scatter sit in
        # sub-collections; the GN system objects themselves are named *geometry_nodes.
        candidates = [o for o in meshes
                      if "geometry_nodes" not in o.name
                      and not any(c.name != self.COLLECTION for c in o.users_collection)]
        if not candidates:
            candidates = [o for o in meshes if "geometry_nodes" not in o.name]

        # Keep one LOD of every chain: the one closest to the preferred level.
        best = {}
        for obj in candidates:
            m = LOD_SUFFIX.search(obj.name)
            base, lod = (obj.name[:m.start()], int(m.group(1))) if m else (obj.name, 0)
            if base not in best or abs(lod - prefer_lod) < abs(best[base][1] - prefer_lod):
                best[base] = (obj, lod)
        candidates = [o for o, _ in best.values()]

        # "_aged" twins sit exactly on top of their clean counterparts; pick one look.
        aged = [o for o in candidates if "_aged" in o.name]
        if aged:
            use_aged = bool(self.rng.random() < 0.5)
            candidates = [o for o in candidates if ("_aged" in o.name) == use_aged]

        if max_tris is not None:
            candidates = [o for o in candidates if _tris(o) <= max_tris]

        # Parts that touch belong to one item (hydrant + caps); separated ones are
        # separate items of a set (six rocks laid out in a row).
        boxes = {o.name: _aabb(o) for o in candidates}
        groups, seen = [], set()
        for obj in candidates:
            if obj.name in seen:
                continue
            group, stack = [], [obj]
            seen.add(obj.name)
            while stack:
                cur = stack.pop()
                group.append(cur)
                for other in candidates:
                    if other.name not in seen and _overlaps(boxes[cur.name], boxes[other.name]):
                        seen.add(other.name)
                        stack.append(other)
            groups.append(group)

        self._variants[key] = [[o.name for o in group] for group in groups]
        return groups

    def purge_unused(self):
        """Delete every loaded object no variant uses (other LODs, GN source pieces,
        the file's own scatter systems) along with its mesh. Returns how many.

        A Poly Haven tree file is ~2.5M triangles across its LOD chain; a scene keeps
        one LOD of it. Purging before the drop simulation is what keeps a forest
        scene inside a render node's memory.
        """
        keep = set()
        for key, groups in self._variants.items():
            if key[0] == "_loaded":
                continue
            for group in groups:
                keep.update(group)
        removed, meshes = 0, set()
        for key in [k for k in self._variants if k[0] == "_loaded"]:
            for name in self._variants[key]:
                obj = bpy.data.objects.get(name)
                if obj is None or name in keep:
                    continue
                if obj.type == "MESH":
                    meshes.add(obj.data.name)
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
            self._variants[key] = [n for n in self._variants[key] if n in keep]
        for name in meshes:
            mesh = bpy.data.meshes.get(name)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        return removed

    def item_size(self, parts):
        lows, highs = zip(*(_aabb(p) for p in parts))
        return np.min(lows, axis=0), np.max(highs, axis=0)

    def place(self, parts, location, yaw, scale=1.0, collection=None, name=None,
              sink=0.0):
        """Linked copies of `parts`, re-centred so the item's footprint centre and
        bottom sit at `location`. Returns the new objects."""
        low, high = self.item_size(parts)
        anchor = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2] + sink])
        target = collection or bpy.context.scene.collection
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        out = []
        for part in parts:
            copy = part.copy()  # shares mesh data with the prototype
            if name:
                copy.name = f"{name}_{part.name}"
            target.objects.link(copy)
            m = np.array(part.matrix_world)
            m[:3, 3] -= anchor
            m[:3, :3] *= scale
            m[:3, 3] *= scale
            m[:3, :] = rot @ m[:3, :]
            m[:3, 3] += np.asarray(location)
            copy.matrix_world = _to_matrix(m)
            copy.hide_render = False
            copy.hide_viewport = False
            copy["category_id"] = 0
            out.append(copy)
        return out


def _to_matrix(m):
    from mathutils import Matrix  # pylint: disable=import-outside-toplevel
    return Matrix([list(row) for row in m])
