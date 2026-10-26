#!/usr/bin/env python3
"""Look at a rendered dataset: contact sheets and summary statistics.

Run with the venv python (no Blender needed):

    .venv/bin/python qa_report.py output/robot_v1                 # 24 random frames
    .venv/bin/python qa_report.py output/robot_v1 --count 48 --rig robot
    .venv/bin/python qa_report.py output/robot_v1/scene_00003     # one scene, all frames

Writes into <dataset>/qa/:
    sheet_XX.jpg   rows of [RGB | masks + boxes + class names | depth], 8 rows a sheet
    stats.json     frames per rig and biome, annotations per class, empty frames,
                   brightness outliers, object sizes in pixels

Read the sheets before trusting a new config: every systematic problem so far
(all-black frames, objects outside the frame, wrong scale) was obvious at a glance
and invisible in the numbers.
"""

import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

PALETTE = np.array([
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
    [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40]], dtype=np.uint8)


def scene_dirs(path):
    if os.path.isfile(os.path.join(path, "scene_meta.json")):
        return [path]
    return sorted(d for d in glob.glob(os.path.join(path, "scene_*")) if os.path.isdir(d))


def load_scene(directory):
    meta_path = os.path.join(directory, "scene_meta.json")
    coco_path = os.path.join(directory, "coco_annotations.json")
    if not (os.path.isfile(meta_path) and os.path.isfile(coco_path)):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)
    return meta, coco


def depth_vis(depth_mm):
    d = depth_mm.astype(np.float32) / 1000.0
    valid = d > 0
    out = np.zeros(d.shape + (3,), dtype=np.uint8)
    if valid.any():
        # log scale: 10 cm and 20 m both stay readable
        v = np.log(np.clip(d[valid], 0.05, 60.0))
        v = (v - np.log(0.05)) / (np.log(60.0) - np.log(0.05))
        g = (255 * (1.0 - v)).astype(np.uint8)
        out[valid] = np.stack([g, (g * 0.7).astype(np.uint8), 255 - g], axis=1)
    return out


def frame_row(directory, image_index, coco, names, thumb_w):
    name = f"{image_index:06d}.png"
    rgb = Image.open(os.path.join(directory, "images", name)).convert("RGB")
    w, h = rgb.size
    scale = thumb_w / w
    size = (thumb_w, int(h * scale))

    sem_path = os.path.join(directory, "semantic", name)
    overlay = np.asarray(rgb).copy()
    if os.path.isfile(sem_path):
        sem = np.asarray(Image.open(sem_path))
        mask = sem > 0
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * PALETTE[sem[mask] % len(PALETTE)]).astype(np.uint8)
    over = Image.fromarray(overlay)
    draw = ImageDraw.Draw(over)
    image_id = next((im["id"] for im in coco["images"] if im["file_name"].endswith(name)), None)
    for ann in coco["annotations"]:
        if ann["image_id"] != image_id:
            continue
        x, y, bw, bh = ann["bbox"]
        color = tuple(int(c) for c in PALETTE[ann["category_id"] % len(PALETTE)])
        draw.rectangle([x, y, x + bw, y + bh], outline=color, width=max(2, w // 320))
        draw.text((x + 2, max(0, y - 12)), names.get(ann["category_id"], "?"), fill=color)

    depth_path = os.path.join(directory, "depth", name)
    if os.path.isfile(depth_path):
        dep = Image.fromarray(depth_vis(np.asarray(Image.open(depth_path))))
    else:
        dep = Image.new("RGB", (w, h))

    row = Image.new("RGB", (size[0] * 3, size[1]))
    for k, im in enumerate((rgb, over, dep)):
        row.paste(im.resize(size), (k * size[0], 0))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="dataset directory or one scene directory")
    parser.add_argument("--count", type=int, default=24, help="frames on the sheets")
    parser.add_argument("--rig", default=None, help="only frames from this camera rig")
    parser.add_argument("--thumb", type=int, default=420, help="thumbnail width in pixels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output folder (default <path>/qa)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.path, "qa")
    os.makedirs(out_dir, exist_ok=True)

    frames, stats = [], {"scenes": 0, "frames": 0, "by_rig": {}, "by_biome": {},
                         "by_env": {}, "annotations_by_class": {}, "empty_frames": 0,
                         "dark_frames": 0, "box_area_px": [], "failed_scenes": 0}
    names = {}
    for directory in scene_dirs(args.path):
        loaded = load_scene(directory)
        if loaded is None:
            stats["failed_scenes"] += 1
            continue
        meta, coco = loaded
        names.update({c["id"]: c["name"] for c in coco["categories"]})
        stats["scenes"] += 1
        biome = (meta.get("environment_meta") or {}).get("biome", meta.get("environment_type"))
        stats["by_biome"][biome] = stats["by_biome"].get(biome, 0) + len(meta["frames"])
        env = meta.get("environment_type", "?")
        stats["by_env"][env] = stats["by_env"].get(env, 0) + len(meta["frames"])
        per_image = {}
        for ann in coco["annotations"]:
            per_image[ann["image_id"]] = per_image.get(ann["image_id"], 0) + 1
            cname = names.get(ann["category_id"], str(ann["category_id"]))
            stats["annotations_by_class"][cname] = stats["annotations_by_class"].get(cname, 0) + 1
            stats["box_area_px"].append(ann["bbox"][2] * ann["bbox"][3])
        for i, frame in enumerate(meta["frames"]):
            rig = frame.get("rig", "general")
            stats["by_rig"][rig] = stats["by_rig"].get(rig, 0) + 1
            stats["frames"] += 1
            image = coco["images"][i] if i < len(coco["images"]) else None
            if image is not None and per_image.get(image["id"], 0) == 0:
                stats["empty_frames"] += 1
            if args.rig is None or rig == args.rig:
                frames.append((directory, i, coco))

    areas = np.asarray(stats.pop("box_area_px") or [0])
    stats["box_side_px_percentiles"] = {p: round(float(np.sqrt(np.percentile(areas, p))), 1)
                                        for p in (5, 25, 50, 75, 95)}

    random.Random(args.seed).shuffle(frames)
    chosen = frames[:args.count]
    rows = []
    for directory, i, coco in chosen:
        rgb = np.asarray(Image.open(os.path.join(directory, "images", f"{i:06d}.png")).convert("L"))
        if rgb.mean() < 30:
            stats["dark_frames"] += 1
        rows.append(frame_row(directory, i, coco, names, args.thumb))

    for sheet, start in enumerate(range(0, len(rows), 8)):
        block = rows[start:start + 8]
        sheet_img = Image.new("RGB", (block[0].width, sum(r.height for r in block)))
        y = 0
        for r in block:
            sheet_img.paste(r, (0, y))
            y += r.height
        sheet_img.save(os.path.join(out_dir, f"sheet_{sheet:02d}.jpg"), quality=88)

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    print(f"[qa] {len(rows)} frame(s) on {(len(rows) + 7) // 8} sheet(s) in {out_dir}")


if __name__ == "__main__":
    main()
