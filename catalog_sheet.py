import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Render a thumbnail of every catalogue model, as it will appear in the dataset.
#
#     blenderproc run catalog_sheet.py -- --config config.yaml
#     blenderproc run catalog_sheet.py -- --config config.yaml --class cigarette --class cup
#     blenderproc run catalog_sheet.py -- --config config.yaml --source objaverse --out qa/objaverse
#
# Each model is loaded through the same code path as the generator (joined, size
# normalised, labelled), placed on a grey floor next to a 10 cm reference cube and
# shot from a 3/4 view. Thumbnails are captioned "<class> <longest_m> <name>" and
# packed into contact sheets of 48 - the fastest way to catch a mislabelled class,
# a wrong real-world size, a missing texture or a model with stray geometry.
#
# A stray-geometry check also runs on every model and its findings go to
# <out>/suspects.json: models whose vertices reach far beyond the bulk of the mesh
# (a lost vertex, a string, a ground plane) produce masks and boxes that do not
# match the visible object.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from PIL import Image, ImageDraw

from sdlib import assets as assets_mod
from sdlib import config


def stray_ratio(obj):
    mesh = obj.blender_obj.data
    if not len(mesh.vertices):
        return 0.0
    m = np.array(obj.blender_obj.matrix_world)
    v = np.array([tuple(x.co) for x in mesh.vertices])
    v = (np.c_[v, np.ones(len(v))] @ m.T)[:, :3]
    d = np.linalg.norm(v - np.median(v, axis=0), axis=1)
    return float(d.max() / max(np.percentile(d, 90), 1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--class", dest="classes", action="append", default=None)
    parser.add_argument("--source", default=None, help="only assets with this `source:`")
    parser.add_argument("--out", default="qa/catalog")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = config.load(config.resolve(args.config))
    out_dir = config.resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)

    bproc.init()
    bproc.camera.set_resolution(args.size, args.size)
    bproc.renderer.set_max_amount_of_samples(16)
    bproc.renderer.set_denoiser("OPTIX" if cfg.get("denoiser") == "OPTIX" else None)
    bproc.renderer.set_output_format(file_format="PNG")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    bg.inputs["Strength"].default_value = 1.5

    obj_cfg = cfg["objects"]
    catalog = assets_mod.Catalog.from_config(config.resolve_all(obj_cfg["catalogs"]),
                                             config.resolve_all(obj_cfg["model_dirs"]))
    chosen = [a for a in catalog.assets
              if (not args.classes or a.class_name in args.classes)
              and (not args.source or a.source == args.source)]
    chosen.sort(key=lambda a: (a.class_name, a.name))
    if args.limit:
        chosen = chosen[:args.limit]
    print(f"[sheet] {len(chosen)} model(s)")

    floor = bproc.object.create_primitive("PLANE", scale=[3, 3, 1])
    floor.set_location([0, 0, 0])
    ref = bproc.object.create_primitive("CUBE", scale=[0.05, 0.05, 0.05])
    mat = ref.new_material("ref")
    mat.set_principled_shader_value("Base Color", [0.9, 0.1, 0.1, 1])
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_rotation_euler([0.6, 0.2, 0.8])
    light.set_energy(3)

    rng = np.random.default_rng(0)
    thumbs, suspects = [], []
    for index, asset in enumerate(chosen):
        try:
            obj = assets_mod.load(asset, rng, index)
        except Exception as err:  # pylint: disable=broad-except
            suspects.append({"path": asset.path, "problem": f"load failed: {err}"})
            continue
        bb = np.asarray(obj.get_bound_box())
        low, high = bb.min(axis=0), bb.max(axis=0)
        obj.set_location(np.asarray(obj.get_location()) - np.array(
            [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]]))
        extent = float(np.max(high - low))
        ref.set_location([extent * 0.75 + 0.06, 0, 0.05])
        ratio = stray_ratio(obj)
        if ratio > 3.0:
            suspects.append({"path": asset.path, "problem": f"stray geometry, max/p90 = {ratio:.1f}"})
        missing = int(obj.get_cp("sd_missing_textures"))
        if missing:
            suspects.append({"path": asset.path, "problem": f"{missing} texture(s) missing"})

        span = extent + 0.2
        cam_loc = np.array([span * 1.1, -span * 1.3, span * 0.9])
        target = np.array([0.03, 0, (high[2] - low[2]) / 3])
        rot = bproc.camera.rotation_from_forward_vec(target - cam_loc)
        bproc.utility.reset_keyframes()
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_loc, rot))
        data = bproc.renderer.render()
        img = Image.fromarray(np.asarray(data["colors"][0])[:, :, :3])
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, args.size, 26], fill=(0, 0, 0))
        draw.text((3, 1), f"{asset.class_name} {asset.longest_m}m", fill=(255, 255, 0))
        draw.text((3, 13), asset.name[:40], fill=(255, 255, 255))
        if ratio > 3.0 or missing:
            draw.rectangle([0, 0, args.size - 1, args.size - 1], outline=(255, 0, 0), width=4)
        thumbs.append(img)
        obj.delete()

    per_sheet, cols = 48, 8
    for sheet, start in enumerate(range(0, len(thumbs), per_sheet)):
        block = thumbs[start:start + per_sheet]
        rows = (len(block) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * args.size, rows * args.size), (40, 40, 40))
        for k, img in enumerate(block):
            canvas.paste(img, ((k % cols) * args.size, (k // cols) * args.size))
        canvas.save(os.path.join(out_dir, f"catalog_{sheet:02d}.jpg"), quality=85)
    with open(os.path.join(out_dir, "suspects.json"), "w", encoding="utf-8") as f:
        json.dump(suspects, f, indent=1)
    print(f"[sheet] {len(thumbs)} thumbnail(s), {len(suspects)} suspect(s) -> {out_dir}")


if __name__ == "__main__":
    main()
