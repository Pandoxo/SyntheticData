"""Asset catalogue: which model files are trash, what class they belong to, and
how big they are in the real world.

Models from different sources do not share a unit convention - a can and a bottle
downloaded from two different authors can differ by orders of magnitude in native
size. So every asset is rescaled on load so that its longest bounding-box axis
matches the `longest_m` value declared in its catalogue entry. Scale then stays a
*usable* cue for a detector instead of being noise, and only a few percent of
jitter is applied on top.

A catalogue is one or more YAML files (`objects.catalogs`). The first one defines
the classes; every file contributes rules of three kinds:

    match: <substring>   any model under `objects.model_dirs` whose path contains it
    path:  <file>        one explicit file, relative to the project root
    glob:  <pattern>     like path, but the first file matching the pattern - used for
                         Poly Haven models, whose file name carries the resolution
"""

import glob
import os

import blenderproc as bproc
import bpy
import numpy as np

from . import config

MODEL_EXTENSIONS = (".blend", ".fbx", ".glb", ".gltf", ".obj")


class Asset:
    """One trash model on disk, plus what we know about it."""

    def __init__(self, path, class_name, category_id, longest_m, collision_shape,
                 scale_jitter, mass_kg, source=None, license=None, deformable=True):
        self.path = path
        self.class_name = class_name
        self.category_id = category_id
        self.longest_m = longest_m
        self.collision_shape = collision_shape
        self.scale_jitter = scale_jitter
        self.mass_kg = mass_kg
        self.source = source
        self.license = license
        self.deformable = deformable

    @property
    def name(self):
        # Every source lays models out as <dir>/<name>/<file>; the containing
        # directory is the most human-readable handle we have.
        return os.path.basename(os.path.dirname(self.path))

    def __repr__(self):
        return f"<Asset {self.name} class={self.class_name} longest={self.longest_m}m>"


class Catalog:
    """The set of usable trash assets, and the class -> category_id mapping."""

    def __init__(self, assets, class_to_id, class_weights=None):
        self.assets = assets
        self.class_to_id = class_to_id
        self.by_class = {}
        for asset in assets:
            self.by_class.setdefault(asset.class_name, []).append(asset)

        # Class-balanced sampling: pick a class first, then a model within it. Drawing
        # uniformly over models would make the dataset's class mix a by-product of
        # whichever classes happen to have the most downloads (60 "other" models
        # against one cigarette, at the time of writing).
        weights = class_weights or {}
        self.classes = sorted(self.by_class)
        w = np.array([float(weights.get(c, 1.0)) for c in self.classes])
        self.class_probs = w / w.sum()

    def __len__(self):
        return len(self.assets)

    def label_mapping(self):
        """COCO category names. Category 0 is reserved for everything unlabelled."""
        return bproc.utility.LabelIdMapping.from_dict({"background": 0, **self.class_to_id})

    def sample(self, rng, count):
        """Pick `count` assets with replacement - duplicates in one scene are realistic."""
        picked = []
        for class_name in rng.choice(self.classes, size=count, p=self.class_probs):
            pool = self.by_class[class_name]
            picked.append(pool[int(rng.integers(0, len(pool)))])
        return picked

    def summary(self):
        return {c: len(self.by_class[c]) for c in self.classes}

    @classmethod
    def from_config(cls, catalog_paths, model_dirs, class_weights=None):
        specs = [(p, config.load(p)) for p in catalog_paths]
        class_to_id = next((dict(s["classes"]) for _, s in specs if s.get("classes")), None)
        if class_to_id is None:
            raise ValueError(f"None of {catalog_paths} defines `classes`")
        if 0 in class_to_id.values():
            raise ValueError("catalogue: category_id 0 is reserved for background "
                             "(the COCO writer drops every instance with category_id 0)")

        files = []
        for model_dir in model_dirs:
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"objects.model_dirs entry does not exist: {model_dir}")
            for ext in MODEL_EXTENSIONS:
                files.extend(glob.glob(os.path.join(model_dir, "**", f"*{ext}"), recursive=True))
        files = sorted(set(files))

        assets, unmatched, missing = [], set(files), []
        for spec_path, spec in specs:
            defaults = spec.get("defaults", {})
            default_shape = defaults.get("collision_shape", "CONVEX_HULL")
            default_jitter = float(defaults.get("scale_jitter", 0.05))
            ignore = [p.lower() for p in spec.get("ignore", [])]

            def make(path, rule, spec_path=spec_path, default_shape=default_shape,
                     default_jitter=default_jitter):
                class_name = rule["class"]
                if class_name not in class_to_id:
                    raise ValueError(f"{spec_path}: asset {rule} has class {class_name!r}, "
                                     f"which is not listed under `classes`")
                return Asset(
                    path=path,
                    class_name=class_name,
                    category_id=class_to_id[class_name],
                    longest_m=float(rule["longest_m"]),
                    collision_shape=rule.get("collision_shape", default_shape),
                    scale_jitter=float(rule.get("scale_jitter", default_jitter)),
                    mass_kg=rule.get("mass_kg"),
                    source=rule.get("source"),
                    license=rule.get("license"),
                    deformable=bool(rule.get("deformable", True)),
                )

            match_rules = [r for r in spec.get("assets", []) or [] if "match" in r]
            for path in files:
                key = path.lower()
                if any(pat in key for pat in ignore):
                    unmatched.discard(path)
                    continue
                rule = next((r for r in match_rules if r["match"].lower() in key), None)
                if rule is not None:
                    assets.append(make(path, rule))
                    unmatched.discard(path)

            for rule in spec.get("assets", []) or []:
                if "path" in rule:
                    path = config.resolve(rule["path"])
                    if os.path.isfile(path):
                        assets.append(make(path, rule))
                    else:
                        missing.append(rule["path"])
                elif "glob" in rule:
                    hits = sorted(glob.glob(config.resolve(rule["glob"])))
                    if hits:
                        assets.append(make(hits[0], rule))
                    else:
                        missing.append(rule["glob"])

        for path in sorted(unmatched):
            print(f"[assets] WARNING no rule matches {path} - skipping. Add a `match:` "
                  f"entry, or list it under `ignore:` to silence this.")
        if missing:
            # Explicit entries for assets that were not downloaded on this machine (a
            # dev subset, say) are expected; say how many rather than listing them all.
            print(f"[assets] {len(missing)} catalogue entr(y/ies) point at files that are "
                  f"not on disk - skipped (e.g. {missing[0]})")
        if not assets:
            raise RuntimeError(f"No usable assets found from {catalog_paths} under {model_dirs}")

        catalog = cls(assets, class_to_id, class_weights)
        print(f"[assets] catalogue: {len(assets)} usable assets - {catalog.summary()}")
        return catalog


def world_dimensions(obj):
    """Axis-aligned world-space size of a mesh object."""
    bb = np.asarray(obj.get_bound_box())
    return bb.max(axis=0) - bb.min(axis=0)


def _load_meshes(path):
    if path.lower().endswith(".blend"):
        try:
            loaded = bproc.loader.load_blend(path, obj_types=["mesh"])
            return [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
        except Exception:  # pylint: disable=broad-except
            # A .blend saved by a newer Blender cannot be read by 4.2; Poly Haven
            # litter ships a glTF copy next to it for exactly this case.
            fallback = sorted(glob.glob(os.path.join(os.path.dirname(path), "gltf", "*.gltf")))
            if not fallback:
                raise
            path = fallback[0]
    before = {o.name for o in bpy.data.objects}
    # load_obj also returns the file's lights and cameras (an FBX exported from a
    # scene often carries its studio light); keep the meshes, by name.
    mesh_names = [m.blender_obj.name for m in bproc.loader.load_obj(path)
                  if m.blender_obj.type == "MESH"]
    # Importers leave the file's empties and armatures behind; they are never
    # rendered but they are parents, and a stray parent fights the physics. Names
    # first, then remove: an empty can be the child of another empty.
    extra = [n for n in {o.name for o in bpy.data.objects} - before
             if bpy.data.objects[n].type != "MESH"]
    for name in extra:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        for child in list(obj.children):
            matrix = child.matrix_world.copy()
            child.parent = None
            child.matrix_world = matrix
        bpy.data.objects.remove(obj, do_unlink=True)
    return [bproc.types.MeshObject(bpy.data.objects[n]) for n in mesh_names
            if n in bpy.data.objects]


def _delete_new_objects(names_before):
    """Remove whatever a failed load left in the scene."""
    for name in [o.name for o in bpy.data.objects if o.name not in names_before]:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def _image_ok(img):
    # Checked without touching the pixels: images load lazily, so `has_data` is False
    # for perfectly good packed textures until something renders them.
    if img is None:
        return False
    if img.packed_file is not None or img.source in ("GENERATED", "VIEWER", "MOVIE"):
        return True
    return os.path.isfile(bpy.path.abspath(img.filepath, library=img.library))


def fix_missing_images(blender_obj):
    """Unplug image textures whose file was not found; returns how many.

    Blender renders a missing image as bright magenta, which would teach a detector
    that magenta means litter. A model whose texture went missing keeps its
    material's base colour instead, and the catalogue QA sheet reports it.
    """
    fixed = 0
    for slot in blender_obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in list(mat.node_tree.nodes):
            if node.type != "TEX_IMAGE" or _image_ok(node.image):
                continue
            for link in list(node.outputs[0].links) + list(node.outputs[1].links):
                mat.node_tree.links.remove(link)
            fixed += 1
    return fixed


def load(asset, rng, instance_index):
    """Load one asset, collapse it to a single mesh, label it and normalise its size.

    Returns the loaded MeshObject, already scaled to real-world size but not yet
    positioned.
    """
    names_before = {o.name for o in bpy.data.objects}
    try:
        meshes = _load_meshes(asset.path)
    except Exception:
        _delete_new_objects(names_before)
        raise
    if not meshes:
        _delete_new_objects(names_before)
        raise RuntimeError(f"{asset.path} contains no mesh objects")

    # Models are often parented to an empty for placement. A parent transform fights
    # the rigid body solver once the object starts falling.
    for mesh in meshes:
        if mesh.get_parent() is not None:
            mesh.clear_parent()

    # A single physical item that ships as several meshes (bottle + cap + label)
    # would otherwise get one segmentation instance - and one COCO box - per part.
    # Joining also gives the drop simulation a single rigid body instead of parts
    # that fall away from each other.
    obj = max(meshes, key=lambda o: o.get_bound_box_volume())
    others = [o for o in meshes if o != obj]
    if others:
        try:
            obj.join_with_other_objects(others)
        except Exception as err:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Could not join the {len(meshes)} parts of {asset.name} into one object "
                f"({err}). Open the file and join them by hand, otherwise every part "
                f"becomes its own annotation.") from err

    # Bake the import transform into the mesh so later modifiers (deformation) and the
    # convex-hull collider work in the object's real proportions.
    obj.persist_transformation_into_mesh(location=False, rotation=True, scale=True)

    missing = fix_missing_images(obj.blender_obj)
    if missing:
        print(f"[assets] {asset.name}: {missing} texture(s) missing on disk - unplugged")
    obj.set_cp("sd_missing_textures", missing)

    obj.set_name(f"trash_{instance_index:03d}_{asset.class_name}")
    obj.set_cp("category_id", asset.category_id)
    obj.set_cp("sd_class", asset.class_name)
    obj.set_cp("sd_asset", asset.name)

    scale_factor = normalize_size(obj, asset, rng)
    obj.set_cp("sd_scale_factor", scale_factor)
    return obj


def normalize_size(obj, asset, rng):
    """Uniformly rescale `obj` so its longest world axis is `asset.longest_m`."""
    bpy.context.view_layer.update()
    longest = float(world_dimensions(obj).max())
    if longest < 1e-9:
        raise RuntimeError(f"{asset.name} has a degenerate bounding box - cannot normalise size")

    factor = asset.longest_m / longest
    jitter = float(rng.uniform(1.0 - asset.scale_jitter, 1.0 + asset.scale_jitter))
    obj.set_scale(np.asarray(obj.get_scale()) * factor * jitter)
    bpy.context.view_layer.update()
    return factor * jitter
