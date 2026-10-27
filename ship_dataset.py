#!/usr/bin/env python3
"""Pack rendered scenes into tar shards and upload them to Google Drive (rclone).

Run with the venv python on the render machine, during or after rendering:

    .venv/bin/python ship_dataset.py output/robot_v1
    .venv/bin/python ship_dataset.py output/robot_v1 --final          # also partial shards
    .venv/bin/python ship_dataset.py output/robot_v1 --remote gdrive:syntheticData/datasets

Why shards: one frame is four PNGs (RGB, depth, instance, semantic), so 100k frames
are 400k files - Google Drive accepts a few files per second, which would take days.
A shard is one tar of `--shard-size` consecutive scene indices (scene_00000-00099),
named by that range, so machines rendering disjoint ranges never collide.

Layout on the remote:
    <remote>/<dataset>/shards/scenes_00000-00099.tar
    <remote>/<dataset>/meta/<host>/coco_annotations.json, config.yaml, qa/...

A shard is only packed once every scene in its range has finished (or failed for
good, with --final). Uploaded shards are recorded in <dataset>/_shards/uploaded.json,
so rerunning ships only what is new. `--delete-local` frees the disk afterwards.
Unpack on the other side with `tar xf` into one folder and run
`run_batch.py --merge-only` there to get one COCO file for everything.
"""

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile


def rclone(*args):
    exe = shutil.which("rclone") or os.path.expanduser("~/.local/bin/rclone")
    cmd = [exe, *args, "--drive-chunk-size", "128M", "--retries", "5",
           "--low-level-retries", "20", "--stats-one-line", "--stats", "30s"]
    return subprocess.run(cmd, check=False).returncode


def scene_index(path):
    return int(os.path.basename(path).split("_")[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset directory, e.g. output/robot_v1")
    parser.add_argument("--remote", default="gdrive:syntheticData/datasets")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--start", type=int, default=0, help="first scene index this machine renders")
    parser.add_argument("--scenes", type=int, default=None, help="how many it renders (for --final)")
    parser.add_argument("--final", action="store_true",
                        help="pack shards even if some of their scenes are missing")
    parser.add_argument("--delete-local", action="store_true",
                        help="remove shard tars and scene folders once uploaded")
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    name = os.path.basename(dataset.rstrip("/"))
    shard_dir = os.path.join(dataset, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    state_path = os.path.join(shard_dir, "uploaded.json")
    uploaded = json.load(open(state_path, encoding="utf-8")) if os.path.isfile(state_path) else {}

    done = {scene_index(d): d for d in glob.glob(os.path.join(dataset, "scene_*"))
            if os.path.isfile(os.path.join(d, "coco_annotations.json"))}
    present = {scene_index(d) for d in glob.glob(os.path.join(dataset, "scene_*"))}
    if not done:
        print("[ship] no finished scenes yet")
    end = args.start + args.scenes if args.scenes else (max(present) + 1 if present else 0)

    shards = {}
    for idx in done:
        first = (idx // args.shard_size) * args.shard_size
        shards.setdefault(first, []).append(idx)

    failures = 0
    for first in sorted(shards):
        last = first + args.shard_size - 1
        shard_name = f"scenes_{first:05d}-{last:05d}.tar"
        if shard_name in uploaded:
            continue
        expected = set(range(max(first, args.start), min(last + 1, end)))
        complete = expected <= set(shards[first]) or (expected <= present and args.final)
        if not complete and not args.final:
            continue
        tar_path = os.path.join(shard_dir, shard_name)
        with tarfile.open(tar_path + ".part", "w") as tar:
            for idx in sorted(shards[first]):
                tar.add(done[idx], arcname=os.path.basename(done[idx]))
        os.replace(tar_path + ".part", tar_path)
        size_mb = os.path.getsize(tar_path) / 1e6
        print(f"[ship] {shard_name}: {len(shards[first])} scene(s), {size_mb:.0f} MB -> uploading")
        if rclone("copyto", tar_path, f"{args.remote}/{name}/shards/{shard_name}") != 0:
            print(f"[ship] upload FAILED for {shard_name} - will retry on the next run")
            failures += 1
            continue
        uploaded[shard_name] = {"scenes": sorted(shards[first]), "mb": round(size_mb, 1)}
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(uploaded, f, indent=1)
        if args.delete_local:
            os.remove(tar_path)
            for idx in shards[first]:
                shutil.rmtree(done[idx], ignore_errors=True)

    # Small files that describe the run: merged COCO (this machine's part), config,
    # QA sheets. One folder per host so several machines do not overwrite each other.
    host = socket.gethostname()
    meta_remote = f"{args.remote}/{name}/meta/{host}"
    for item in ("coco_annotations.json", "config.yaml", "run_info.json"):
        path = os.path.join(dataset, item)
        if os.path.isfile(path):
            rclone("copyto", path, f"{meta_remote}/{item}")
    if os.path.isdir(os.path.join(dataset, "qa")):
        rclone("copy", os.path.join(dataset, "qa"), f"{meta_remote}/qa")

    print(f"[ship] {len(uploaded)} shard(s) uploaded in total, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
