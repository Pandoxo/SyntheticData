#!/usr/bin/env python3
"""Render a whole dataset, one Blender process per scene, then merge the COCO files.

Run with the venv python, NOT through `blenderproc run`:

    .venv/bin/python run_batch.py --config config.yaml
    .venv/bin/python run_batch.py --config config_scene.yaml --script generate_scene.py --jobs 4

Why a process per scene rather than a loop inside one process: Blender never frees
appended materials, images and meshes, so an in-process loop over full scenes grows
memory until it is killed. A separate process also means one bad scene cannot take
the run down, and scenes can be rendered in parallel.

Each scene writes its own `output/<name>/scene_XXXXX/` directory - separate COCO
files avoid a write race between parallel workers. `--merge-only` combines them into
one `coco_annotations.json` at the dataset root afterwards.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdlib import config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def scene_dir(dataset_dir, index):
    return os.path.join(dataset_dir, f"scene_{index:05d}")


def is_done(directory):
    return os.path.isfile(os.path.join(directory, "coco_annotations.json"))


def blenderproc_executable():
    """Locate the `blenderproc` console script.

    It has to be the console script - `python -m blenderproc` is rejected by the
    package itself. Prefer the one next to the interpreter running this file, since
    the venv's bin/ is not on PATH unless the venv was activated.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "blenderproc")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which("blenderproc")
    if found:
        return found
    raise RuntimeError(
        f"Could not find the `blenderproc` executable next to {sys.executable} or on "
        f"PATH. Install it into this environment with `pip install blenderproc`.")


def render_scene(script, cfg_path, index, directory, quiet):
    cmd = [
        blenderproc_executable(),
        "run", os.path.join(PROJECT_ROOT, script), "--",
        "--config", cfg_path,
        "--scene-index", str(index),
        "--output-dir", directory,
    ]
    started = time.time()
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, check=False,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        tail = ""
        if quiet and result.stdout:
            tail = "\n" + result.stdout.decode("utf-8", "replace")[-2000:]
        return index, False, elapsed, f"exit code {result.returncode}{tail}"
    return index, True, elapsed, None


def merge_coco(dataset_dir):
    """Combine every per-scene coco_annotations.json into one at the dataset root.

    Image ids and annotation ids are renumbered, and file names are rewritten to be
    relative to the dataset root so the merged file resolves against the same tree.
    """
    scene_dirs = sorted(
        os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir)
        if name.startswith("scene_") and os.path.isdir(os.path.join(dataset_dir, name))
    )

    merged = None
    categories = {}
    images, annotations = [], []
    next_image_id, next_annotation_id = 0, 0

    for directory in scene_dirs:
        path = os.path.join(directory, "coco_annotations.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            part = json.load(f)
        if merged is None:
            merged = {"info": part.get("info", {}), "licenses": part.get("licenses", [])}

        for category in part.get("categories", []):
            categories.setdefault(category["id"], category)

        prefix = os.path.relpath(directory, dataset_dir)
        remap = {}
        for image in part.get("images", []):
            remap[image["id"]] = next_image_id
            image = dict(image, id=next_image_id,
                         file_name=os.path.join(prefix, image["file_name"]))
            images.append(image)
            next_image_id += 1

        for annotation in part.get("annotations", []):
            annotation = dict(annotation, id=next_annotation_id,
                              image_id=remap[annotation["image_id"]])
            annotations.append(annotation)
            next_annotation_id += 1

    if merged is None:
        print("[merge] nothing to merge - no per-scene coco_annotations.json found")
        return None

    merged["categories"] = [categories[k] for k in sorted(categories)]
    merged["images"] = images
    merged["annotations"] = annotations

    out_path = os.path.join(dataset_dir, "coco_annotations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)

    per_class = {}
    names = {c["id"]: c["name"] for c in merged["categories"]}
    for annotation in annotations:
        key = names.get(annotation["category_id"], annotation["category_id"])
        per_class[key] = per_class.get(key, 0) + 1

    print(f"\n[merge] {out_path}")
    print(f"[merge] {len(images)} images, {len(annotations)} annotations")
    for name, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"[merge]   {name:<12} {count}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--script", default="generate.py",
                        help="generate.py (ground plane) or generate_scene.py (.blend scenes)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="scenes to render in parallel; each is a full Blender process")
    parser.add_argument("--scenes", type=int, default=None,
                        help="override num_scenes from the config")
    parser.add_argument("--start", type=int, default=0, help="first scene index")
    parser.add_argument("--resume", action="store_true",
                        help="skip scenes that already produced annotations")
    parser.add_argument("--merge-only", action="store_true",
                        help="only merge existing per-scene COCO files")
    parser.add_argument("--quiet", action="store_true",
                        help="capture Blender's output and only show it for failures")
    args = parser.parse_args()

    cfg_path = config.resolve(args.config)
    cfg = config.load(cfg_path)
    dataset_dir = config.resolve(cfg["output_dir"])
    os.makedirs(dataset_dir, exist_ok=True)

    if args.merge_only:
        merge_coco(dataset_dir)
        return 0

    total = args.scenes if args.scenes is not None else int(cfg["num_scenes"])
    indices = list(range(args.start, args.start + total))
    if args.resume:
        before = len(indices)
        indices = [i for i in indices if not is_done(scene_dir(dataset_dir, i))]
        print(f"[batch] resuming: {before - len(indices)} scene(s) already done")

    print(f"[batch] {len(indices)} scene(s) via {args.script}, {args.jobs} job(s) "
          f"-> {dataset_dir}")

    failures, started = [], time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(render_scene, args.script, cfg_path, i,
                        scene_dir(dataset_dir, i), args.quiet)
            for i in indices
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index, ok, elapsed, error = future.result()
            status = "ok" if ok else "FAILED"
            print(f"[batch] {done}/{len(indices)} scene {index}: {status} ({elapsed:.0f}s)")
            if not ok:
                failures.append((index, error))

    print(f"[batch] finished in {time.time() - started:.0f}s, {len(failures)} failure(s)")
    for index, error in failures:
        print(f"[batch]   scene {index}: {error}")

    merge_coco(dataset_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
