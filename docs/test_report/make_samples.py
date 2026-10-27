#!/usr/bin/env python3
"""Pick sample frames for the report: a grid of RGB images and one strip of
RGB | masks | depth, chosen across camera rigs and environments.

    python docs/test_report/make_samples.py <dataset> docs/test_report/figures
"""

import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from qa_report import PALETTE, depth_vis  # noqa: E402


def frames(dataset):
    out = []
    for meta_path in sorted(glob.glob(os.path.join(dataset, "scene_*", "scene_meta.json"))):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        scene = os.path.dirname(meta_path)
        biome = (meta.get("environment_meta") or {}).get("biome") or meta["environment"].split("#")[0]
        for i, fr in enumerate(meta["frames"]):
            out.append({"scene": scene, "i": i, "rig": fr.get("rig"), "biome": biome,
                        "empty": fr.get("intended_empty", False)})
    return out


def overlay(scene, i):
    name = f"{i:06d}.png"
    rgb = np.asarray(Image.open(os.path.join(scene, "images", name)).convert("RGB")).copy()
    sem = np.asarray(Image.open(os.path.join(scene, "semantic", name)))
    m = sem > 0
    rgb[m] = (0.45 * rgb[m] + 0.55 * PALETTE[sem[m] % len(PALETTE)]).astype(np.uint8)
    return Image.fromarray(rgb)


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return img


def main():
    dataset, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rng = random.Random(3)
    pool = [f for f in frames(dataset) if not f["empty"]]
    rng.shuffle(pool)

    # 3 x 3 grid: 4 robot, 3 arm, 2 general, preferring distinct biomes
    want = {"robot": 4, "arm": 3, "general": 2}
    chosen, used = [], set()
    for rig, n in want.items():
        for f in pool:
            if len([c for c in chosen if c["rig"] == rig]) >= n:
                break
            if f["rig"] == rig and (f["biome"], rig) not in used:
                chosen.append(f)
                used.add((f["biome"], rig))
        for f in pool:  # top up if there were not enough distinct biomes
            if len([c for c in chosen if c["rig"] == rig]) >= n:
                break
            if f["rig"] == rig and f not in chosen:
                chosen.append(f)
    w, h = 480, 360
    grid = Image.new("RGB", (3 * w, 3 * h))
    for k, f in enumerate(chosen[:9]):
        img = Image.open(os.path.join(f["scene"], "images", f"{f['i']:06d}.png")).convert("RGB").resize((w, h))
        grid.paste(label(img, f"{f['rig']} | {f['biome']}"), ((k % 3) * w, (k // 3) * h))
    grid.save(os.path.join(out, "samples.jpg"), quality=88)

    # strip: two frames as RGB | masks | depth
    strip = Image.new("RGB", (3 * w, 2 * h))
    for row, f in enumerate([c for c in chosen if c["rig"] == "robot"][:1] + [c for c in chosen if c["rig"] == "arm"][:1]):
        name = f"{f['i']:06d}.png"
        rgb = Image.open(os.path.join(f["scene"], "images", name)).convert("RGB").resize((w, h))
        ov = overlay(f["scene"], f["i"]).resize((w, h))
        dep = Image.fromarray(depth_vis(np.asarray(Image.open(os.path.join(f["scene"], "depth", name))))).resize((w, h))
        for col, im in enumerate((rgb, ov, dep)):
            strip.paste(im, (col * w, row * h))
    strip.save(os.path.join(out, "modalities.jpg"), quality=88)
    print(f"[samples] {len(chosen)} frames -> {out}")


if __name__ == "__main__":
    main()
