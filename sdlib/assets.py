"""Asset catalogue: which .blend files are trash, what class they belong to, and
how big they are in the real world.

BlenderKit assets do not share a unit convention - a can and a bottle downloaded
from two different authors can differ by orders of magnitude in native size. So
every asset is rescaled on load so that its longest bounding-box axis matches the
`longest_m` value declared in assets.yaml. Scale then stays a *usable* cue for a
detector instead of being noise, and only a few percent of jitter is applied on
top.
"""

import glob
import os

import blenderproc as bproc
import bpy
import numpy as np

from . import config


class Asset:
    """One trash model on disk, plus what we know about it."""

    def __init__(self, path, class_name, category_id, longest_m, collision_shape,
                 scale_jitter, mass_kg):
        self.path = path
        self.class_name = class_name
        self.category_id = category_id
        self.longest_m = longest_m
        self.collision_shape = collision_shape
        self.scale_jitter = scale_jitter
        self.mass_kg = mass_kg

    @property
    def name(self):
        # BlenderKit lays assets out as <dir>/<slug>_<uuid>/<file>_<uuid>.blend;
        # the containing directory is the most human-readable handle we have.
        return os.path.basename(os.path.dirname(self.path))

    def __repr__(self):
        return f"<Asset {self.name} class={self.class_name} longest={self.longest_m}m>"


class Catalog:
    """The set of usable trash assets, and the class -> category_id mapping."""

    def __init__(self, assets, class_to_id):
        self.assets = assets
        self.class_to_id = class_to_id

    def __len__(self):
        return len(self.assets)

    def label_mapping(self):
        """COCO category names. Category 0 is reserved for everything unlabelled."""
        return bproc.utility.LabelIdMapping.from_dict({"background": 0, **self.class_to_id})

    def sample(self, rng, count):
        """Pick `count` assets with replacement - duplicates in one scene are realistic."""
        return [self.assets[i] for i in rng.integers(0, len(self.assets), size=count)]

    @classmethod
    def from_config(cls, assets_yaml_path, model_dirs):
        spec = config.load(assets_yaml_path)
        class_to_id = dict(spec["classes"])
        if 0 in class_to_id.values():
            raise ValueError("assets.yaml: category_id 0 is reserved for background "
                             "(the COCO writer drops every instance with category_id 0)")

        defaults = spec.get("defaults", {})
        default_shape = defaults.get("collision_shape", "CONVEX_HULL")
        default_jitter = float(defaults.get("scale_jitter", 0.05))

        ignore = [p.lower() for p in spec.get("ignore", [])]
        rules = spec["assets"]

        files = []
        for model_dir in model_dirs:
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"objects.model_dirs entry does not exist: {model_dir}")
            files.extend(glob.glob(os.path.join(model_dir, "**", "*.blend"), recursive=True))
        files = sorted(set(files))
        if not files:
            raise RuntimeError(f"No .blend files found under any of: {model_dirs}")

        assets, unmatched = [], []
        for path in files:
            key = path.lower()
            if any(pat in key for pat in ignore):
                continue
            rule = next((r for r in rules if r["match"].lower() in key), None)
            if rule is None:
                unmatched.append(path)
                continue
            class_name = rule["class"]
            if class_name not in class_to_id:
                raise ValueError(f"assets.yaml: asset {rule['match']!r} has class "
                                 f"{class_name!r}, which is not listed under `classes`")
            assets.append(Asset(
                path=path,
                class_name=class_name,
                category_id=class_to_id[class_name],
                longest_m=float(rule["longest_m"]),
                collision_shape=rule.get("collision_shape", default_shape),
                scale_jitter=float(rule.get("scale_jitter", default_jitter)),
                mass_kg=rule.get("mass_kg"),
            ))

        for path in unmatched:
            print(f"[assets] WARNING no rule in assets.yaml matches {path} - skipping. "
                  f"Add a `match:` entry, or list it under `ignore:` to silence this.")
        if not assets:
            raise RuntimeError(
                f"None of the {len(files)} .blend files under {model_dirs} matched a rule in "
                f"{assets_yaml_path}. Check the `match:` substrings against the real filenames.")

        print(f"[assets] catalogue: {len(assets)} usable assets across "
              f"{len(set(a.class_name for a in assets))} classes")
        return cls(assets, class_to_id)


def world_dimensions(obj):
    """Axis-aligned world-space size of a mesh object."""
    bb = np.asarray(obj.get_bound_box())
    return bb.max(axis=0) - bb.min(axis=0)


def load(asset, rng, instance_index):
    """Load one asset, collapse it to a single mesh, label it and normalise its size.

    Returns the loaded MeshObject, already scaled to real-world size but not yet
    positioned.
    """
    loaded = bproc.loader.load_blend(asset.path, obj_types=["mesh"])
    meshes = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
    if not meshes:
        raise RuntimeError(f"{asset.path} contains no mesh objects")

    # BlenderKit models are usually parented to an empty for placement. That empty is
    # not returned by load_blend, but it still sits in the scene, and a parent
    # transform fights the rigid body solver once the object starts falling.
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
                f"({err}). Open the .blend and join them by hand, otherwise every part "
                f"becomes its own annotation.") from err

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
