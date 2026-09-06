import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Converts .usdz (and other USD flavours) into .blend files, one .blend per input.
#
#     blenderproc run usdz_to_blend.py -- --input ~/downloads/models --asset
#     blenderproc run usdz_to_blend.py -- --input can.usdz --output-dir resources/objects
#
# The point of this script is the asset pipeline: Catalog.from_config globs `*.blend`
# under objects.model_dirs, so a USDZ has to become a .blend before generate.py can
# ever see it. `--asset` writes the result in the layout the catalogue expects and
# prepares the mesh the way sdlib/assets.py wants it (see the flag's help text).
#
# Textures are the reason this is not a one-liner: a .usdz is a zip, and its images
# only exist inside that archive. They are packed into the .blend on import, so the
# output is a single self-contained file you can move anywhere.
#
# Two things Blender's USD importer does and does not do, both verified on 4.2:
#   - it *does* convert a Y-up stage (which is what most .usdz in the wild are) to
#     Blender's Z-up, so models do not arrive lying on their side;
#   - it *ignores* the stage's metersPerUnit, so a model authored in centimetres
#     imports 100x too large. `--scale 0.01` fixes those. It does not matter for the
#     trash catalogue, where assets.yaml `longest_m` rescales every asset anyway -
#     which is why the native size is printed for each file.

import argparse
import glob
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
from mathutils import Vector

from sdlib import config

USD_EXTENSIONS = (".usdz", ".usd", ".usda", ".usdc")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert .usdz files into .blend files.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, metavar="PATH",
                        help="files and/or directories. Directories are searched "
                             "recursively for --pattern; files are converted whatever "
                             "their extension.")
    parser.add_argument("--output-dir", default="resources/objects",
                        help="where the .blend files go (default: %(default)s)")
    parser.add_argument("--pattern", default="*.usdz",
                        help="glob used when --input names a directory "
                             "(default: %(default)s). Use '*.usd*' to sweep every "
                             "USD flavour.")
    parser.add_argument("--asset", action="store_true",
                        help="prepare the output for the trash catalogue: geometry "
                             "only, all meshes joined into one object, origin at the "
                             "bounding-box centre and the object at the world origin. "
                             "Equivalent to --geometry-only --join --center.")
    parser.add_argument("--geometry-only", action="store_true",
                        help="drop cameras and lights, keep meshes and materials")
    parser.add_argument("--join", action="store_true",
                        help="join every mesh into a single object, so the scene "
                             "generator produces one annotation per item instead of "
                             "one per part")
    parser.add_argument("--center", action="store_true",
                        help="give each object its own bounding-box centre as origin, "
                             "then sit the model's combined bounding box on the world "
                             "origin (several meshes keep their relative layout)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="uniform scale applied on import (default: %(default)s). "
                             "Blender ignores the stage's metersPerUnit, so a model "
                             "authored in centimetres imports 100x too big - pass "
                             "--scale 0.01 for those. Assets headed for the catalogue "
                             "do not need it: assets.yaml `longest_m` rescales them.")
    parser.add_argument("--flat", action="store_true",
                        help="write <output-dir>/<name>.blend. The default is "
                             "<output-dir>/<name>/<name>.blend, which is the layout "
                             "sdlib/assets.py reads the asset name from.")
    parser.add_argument("--compress", action="store_true",
                        help="compress the .blend (smaller on disk, slower to load)")
    parser.add_argument("--overwrite", action="store_true",
                        help="reconvert inputs whose .blend already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be converted and where, then stop")
    args = parser.parse_args()
    if args.asset:
        args.geometry_only = args.join = args.center = True
    return args


def collect_inputs(inputs, pattern):
    """Expand the --input list into a sorted, de-duplicated list of USD files."""
    found = []
    for entry in inputs:
        path = config.resolve(entry)
        if os.path.isdir(path):
            matches = glob.glob(os.path.join(path, "**", pattern), recursive=True)
            if not matches:
                print(f"[usdz] WARNING no {pattern} under {path}")
            found.extend(matches)
        elif os.path.isfile(path):
            # An explicitly named file is converted even if it does not match
            # --pattern; only warn when the extension is not USD at all.
            if not path.lower().endswith(USD_EXTENSIONS):
                print(f"[usdz] WARNING {path} is not a USD file - trying anyway")
            found.append(path)
        else:
            raise FileNotFoundError(f"--input entry does not exist: {path}")
    return sorted(set(os.path.normpath(p) for p in found))


def output_path(src, output_dir, flat):
    """Where the .blend for `src` goes.

    The nested default mirrors BlenderKit's <dir>/<slug>/<file>.blend layout, because
    Asset.name in sdlib/assets.py reads the asset name off the *containing directory*.
    """
    stem = os.path.splitext(os.path.basename(src))[0]
    if flat:
        return os.path.join(output_dir, f"{stem}.blend")
    return os.path.join(output_dir, stem, f"{stem}.blend")


def reset_scene():
    """Start from an empty file, so nothing leaks from the previous conversion."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_usd(src, scale, geometry_only):
    """Import one USD file, packing its textures into the current .blend.

    Only the operator properties this Blender actually has are passed:
    `import_textures_mode` arrived in 4.1, and without it a .usdz would import with
    its images pointing into a temporary extraction directory that is gone by the
    time anyone opens the result.
    """
    supported = set(bpy.ops.wm.usd_import.get_rna_type().properties.keys())
    kwargs = {
        "filepath": src,
        "scale": scale,
        "import_textures_mode": "IMPORT_PACK",
        "import_cameras": not geometry_only,
        "import_lights": not geometry_only,
        "set_frame_range": False,
    }
    dropped = [k for k in kwargs if k not in supported]
    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    if "import_textures_mode" in dropped:
        print(f"[usdz] WARNING this Blender ({bpy.app.version_string}) cannot pack USD "
              f"textures on import; materials in the output may have broken image "
              f"paths. Blender 4.1+ fixes this.")

    result = bpy.ops.wm.usd_import(**kwargs)
    if "FINISHED" not in result:
        raise RuntimeError(f"wm.usd_import returned {set(result)}")


def mesh_objects():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def unparent_meshes():
    """Clear parents, keeping each mesh exactly where it is in world space.

    USD hierarchies put the geometry under empties, and every later step here removes
    or joins those - dropping a parent without re-applying its matrix would silently
    move the geometry.
    """
    for obj in mesh_objects():
        if obj.parent is not None:
            matrix = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = matrix
    bpy.context.view_layer.update()


def drop_non_meshes():
    """Delete everything that is not a mesh.

    `import_cameras=False` / `import_lights=False` do not skip those prims - Blender
    still imports them, as empties, to keep the transform hierarchy intact. So asking
    for geometry only leaves a litter of empties named Camera, Point and _materials
    unless they are removed here.
    """
    unparent_meshes()
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)


def join_meshes():
    """Collapse every mesh into one object.

    A single physical item that ships as several prims (bottle + cap + label) would
    otherwise become one segmentation instance - and one COCO box - per part. This is
    the same collapse sdlib/assets.py does at load time, done once here instead.
    """
    unparent_meshes()  # a parent transform would be baked in wrong by the join
    meshes = mesh_objects()
    if len(meshes) < 2:
        return meshes[0] if meshes else None

    target = max(meshes, key=lambda o: sum(o.dimensions))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def center_objects():
    """Put the model's combined bounding-box centre at the world origin.

    Each object's origin is moved to its own geometry first, so the drop simulation
    spins it about its own centre rather than some distant pivot inherited from the
    USD hierarchy. Several meshes are then moved as one group - dropping each of them
    individually onto the origin would stack the parts on top of each other.
    """
    unparent_meshes()
    objs = mesh_objects()
    if not objs:
        return

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.context.view_layer.update()

    corners = [obj.matrix_world @ Vector(c) for obj in objs for c in obj.bound_box]
    center = Vector(tuple((max(c[i] for c in corners) + min(c[i] for c in corners)) / 2.0
                          for i in range(3)))
    for obj in objs:
        obj.location -= center
    bpy.context.view_layer.update()


def describe():
    """The numbers worth knowing before writing an assets.yaml rule for this model."""
    meshes = mesh_objects()
    packed = sum(1 for img in bpy.data.images if img.packed_file)
    unpacked = [img.name for img in bpy.data.images
                if not img.packed_file and img.source == "FILE"]

    # World-space corners of each object's local bounding box - the same measurement
    # sdlib/assets.py:normalize_size makes, so the size printed here is the number
    # `longest_m` will be compared against.
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    size = (0.0, 0.0, 0.0)
    if corners:
        size = tuple(max(c[i] for c in corners) - min(c[i] for c in corners)
                     for i in range(3))

    # A texture node with no image is how a lost texture actually shows up: the
    # material survives, the image silently does not, and the asset renders flat
    # without anything failing. Worth saying out loud.
    empty_texture_nodes = sum(
        1 for mat in bpy.data.materials if mat.use_nodes
        for node in mat.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is None)

    return {
        "objects": len(bpy.data.objects),
        "meshes": len(meshes),
        "faces": sum(len(o.data.polygons) for o in meshes),
        "materials": len(bpy.data.materials),
        "images_packed": packed,
        "images_unpacked": unpacked,
        "empty_texture_nodes": empty_texture_nodes,
        "size_m": size,
    }


def convert(src, dst, args):
    reset_scene()
    import_usd(src, args.scale, args.geometry_only)

    if not mesh_objects():
        raise RuntimeError("no mesh objects were imported - the file may be empty, or "
                           "hold only prims Blender's USD importer skips")

    if args.geometry_only:
        drop_non_meshes()
    if args.join:
        join_meshes()
    if args.center:
        center_objects()

    # Belt and braces: IMPORT_PACK covers the USD's own textures, pack_all catches
    # anything else that ended up referencing a path outside the file.
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as err:
        print(f"[usdz] WARNING could not pack every file ({err})")

    bpy.context.view_layer.update()
    stats = describe()

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=dst, compress=args.compress)

    size = " x ".join(f"{v:.3f}" for v in stats["size_m"])
    print(f"[usdz] wrote {dst}")
    print(f"[usdz]   {stats['objects']} object(s), {stats['meshes']} mesh(es), "
          f"{stats['faces']} faces, {stats['materials']} material(s), "
          f"{stats['images_packed']} image(s) packed")
    print(f"[usdz]   native size {size} (USD units) - "
          f"longest axis {max(stats['size_m']):.3f}")
    if stats["images_unpacked"]:
        print(f"[usdz]   WARNING unpacked images: {', '.join(stats['images_unpacked'])} "
              f"- the .blend depends on files outside it and will render untextured if "
              f"moved")
    if stats["empty_texture_nodes"]:
        print(f"[usdz]   WARNING {stats['empty_texture_nodes']} texture node(s) have no "
              f"image: the source pointed at textures that are not there, so this asset "
              f"renders flat. A .usdz carries its textures inside the archive, so this "
              f"usually means a plain .usd/.usda whose sibling texture files are missing.")
    return stats


def main():
    args = parse_args()
    output_dir = config.resolve(args.output_dir)
    sources = collect_inputs(args.input, args.pattern)
    if not sources:
        raise RuntimeError(f"nothing to convert: no input matched {args.pattern!r}")

    jobs = [(src, output_path(src, output_dir, args.flat)) for src in sources]
    if args.dry_run:
        for src, dst in jobs:
            state = "overwrite" if os.path.exists(dst) else "new"
            print(f"[usdz] {src} -> {dst} ({state})")
        print(f"[usdz] dry run: {len(jobs)} file(s)")
        return 0

    converted, skipped, failed = 0, 0, []
    for index, (src, dst) in enumerate(jobs, start=1):
        if os.path.exists(dst) and not args.overwrite:
            print(f"[usdz] [{index}/{len(jobs)}] skipping {os.path.basename(src)} - "
                  f"{dst} exists (use --overwrite)")
            skipped += 1
            continue
        print(f"[usdz] [{index}/{len(jobs)}] {src}")
        try:
            convert(src, dst, args)
            converted += 1
        except Exception as err:  # pylint: disable=broad-except
            # One bad model should not cost you the rest of the batch. A RuntimeError
            # is the importer refusing a file and says so clearly on one line; anything
            # else is unexpected and worth a full traceback.
            if not isinstance(err, RuntimeError):
                traceback.print_exc()
            print(f"[usdz] FAILED {src}: {err}")
            failed.append(src)

    print(f"[usdz] done: {converted} converted, {skipped} skipped, {len(failed)} failed")
    for src in failed:
        print(f"[usdz]   failed: {src}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
