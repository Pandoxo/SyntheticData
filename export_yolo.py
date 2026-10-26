#!/usr/bin/env python3
"""Convert a rendered dataset into Ultralytics YOLO segmentation format.

Run with the venv python (needs opencv-python-headless, see brev/requirements.txt):

    .venv/bin/python export_yolo.py output/robot_v1 --out exports/robot_v1_yolo
    .venv/bin/python export_yolo.py output/robot_v1 --out exports/robot_v1_yolo --rig robot --rig arm

Produces
    <out>/images/{train,val}/<scene>_<frame>.png   (symlinks unless --copy)
    <out>/labels/{train,val}/<scene>_<frame>.txt   one polygon per instance
    <out>/data.yaml                                 for `yolo segment train data=...`

Polygons are traced from the rendered instance masks (instance/*.png), not from
COCO: the masks are exact, already account for occlusion, and an instance split
into pieces by a grass blade gets one polygon per piece. The train/val split is by
scene, never by frame - frames of one scene share litter, ground and light, and
splitting them would leak the validation set into training.
"""

import argparse
import glob
import json
import os
import random
import shutil

import cv2
import numpy as np
import yaml


def polygons(mask, min_area):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.003 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            out.append(c.reshape(-1, 2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--rig", action="append", default=None, help="only these camera rigs")
    parser.add_argument("--min-area", type=float, default=30.0, help="pixels; smaller pieces dropped")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets.yaml"),
                        help="catalogue file whose `classes` define the label ids")
    args = parser.parse_args()

    scenes = sorted(d for d in glob.glob(os.path.join(args.dataset, "scene_*"))
                    if os.path.isfile(os.path.join(d, "scene_meta.json")))
    random.Random(args.seed).shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * args.val_fraction))) if len(scenes) > 1 else 0
    split = {d: ("val" if k < n_val else "train") for k, d in enumerate(scenes)}

    # Class names from the catalogue, not from the per-scene COCO files: those list
    # only the classes that happened to appear, and the ids must be the full set.
    with open(args.classes, encoding="utf-8") as f:
        names = {int(v): k for k, v in yaml.safe_load(f)["classes"].items()}
    counts = {"train": 0, "val": 0, "instances": 0}
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    for scene in scenes:
        with open(os.path.join(scene, "scene_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        part = split[scene]
        for i, frame in enumerate(meta["frames"]):
            if args.rig and frame.get("rig") not in args.rig:
                continue
            stem = f"{os.path.basename(scene)}_{i:06d}"
            img_src = os.path.abspath(os.path.join(scene, "images", f"{i:06d}.png"))
            inst = cv2.imread(os.path.join(scene, "instance", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            sem = cv2.imread(os.path.join(scene, "semantic", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            if inst is None or sem is None:
                continue
            h, w = inst.shape[:2]
            lines = []
            for k in np.unique(inst):
                if k == 0:
                    continue
                mask = inst == k
                cls = int(np.bincount(sem[mask]).argmax())
                for poly in polygons(mask, args.min_area):
                    coords = " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly)
                    # YOLO class ids are 0-based; ours start at 1 (0 = background).
                    lines.append(f"{cls - 1} {coords}")
                    counts["instances"] += 1
            img_dst = os.path.join(args.out, "images", part, stem + ".png")
            if os.path.lexists(img_dst):
                os.remove(img_dst)
            if args.copy:
                shutil.copy2(img_src, img_dst)
            else:
                os.symlink(img_src, img_dst)
            with open(os.path.join(args.out, "labels", part, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            counts[part] += 1

    max_id = max(names) if names else 0
    class_names = [names.get(i, f"class_{i}") for i in range(1, max_id + 1)]
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(class_names)))
    print(f"[yolo] {counts['train']} train / {counts['val']} val image(s), "
          f"{counts['instances']} polygon(s), {len(class_names)} classes -> {args.out}")


if __name__ == "__main__":
    main()
