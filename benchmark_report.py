#!/usr/bin/env python3
"""Turn a render run into numbers and plots for capacity planning.

Run with the venv python on a dataset that render.sh produced (or a copy of it):

    .venv/bin/python benchmark_report.py output/robot_v1 --logs logs --price-per-hour 4.66
    .venv/bin/python benchmark_report.py run/dataset --logs run/logs --out docs/test_report/figures

Reads every scene_meta.json (per-stage time and peak memory, written by
sdlib/pipeline.py), run_info.json (wall time, GPU count), and the utilisation traces
render.sh records (logs/gpu_util.csv, logs/host_util.csv). Writes into --out:

    summary.json          every number the report quotes
    stages.png            where a scene's time goes, by environment type
    scene_times.png       distribution of per-scene wall time
    utilisation.png       GPU load / VRAM and host RAM over the run
    projection.png        wall time and cost for a dataset size vs number of GPUs
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STAGES = ["init", "environment", "litter + physics", "clutter", "cameras", "grass", "render", "write"]
COLORS = ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]


def load_scenes(dataset):
    rows = []
    for path in sorted(glob.glob(os.path.join(dataset, "scene_*", "scene_meta.json"))):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        prof = meta.get("profile", {})
        stages = {k: float(v.get("s", 0.0)) for k, v in prof.items()}
        peak = max((float(v.get("peak_rss_gb", 0.0)) for v in prof.values()), default=0.0)
        env = meta.get("environment_type", "?")
        biome = (meta.get("environment_meta") or {}).get("biome") or meta.get("environment", "?").split("#")[0]
        rows.append({
            "scene": meta["scene_index"], "env": env, "biome": biome,
            "frames_rendered": meta["stats"]["frames_in"], "frames_written": meta["stats"]["frames_out"],
            "annotations": meta["stats"]["annotations_kept"], "stages": stages,
            "wall_s": sum(stages.values()), "peak_rss_gb": peak,
            "rigs": [f.get("rig") for f in meta["frames"]],
        })
    return rows


def read_gpu_trace(path):
    trace = {}
    if not os.path.isfile(path):
        return trace
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            try:
                t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                idx = int(row[1])
                util = float(row[2].strip().split()[0])
                mem = float(row[3].strip().split()[0])
                total = float(row[4].strip().split()[0])
            except (ValueError, IndexError):
                continue
            trace.setdefault(idx, []).append((t, util, mem, total))
    return trace


def read_host_trace(path):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 4:
                try:
                    out.append(tuple(float(p) for p in parts))
                except ValueError:
                    pass
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--run-info", default=None, help="default <dataset>/run_info.json")
    parser.add_argument("--price-per-hour", type=float, required=True,
                        help="what the measured machine costs per hour, all GPUs")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sizes", default="10000,50000,100000,250000",
                        help="dataset sizes (images) to project")
    args = parser.parse_args()

    out = args.out or os.path.join(args.dataset, "benchmark")
    os.makedirs(out, exist_ok=True)
    scenes = load_scenes(args.dataset)
    if not scenes:
        raise SystemExit("no scene_meta.json with a profile found")
    with open(args.run_info or os.path.join(args.dataset, "run_info.json"), encoding="utf-8") as f:
        info = json.load(f)

    gpus = int(info["gpus"])
    elapsed_h = float(info["elapsed_s"]) / 3600.0
    images = sum(s["frames_written"] for s in scenes)
    rendered = sum(s["frames_rendered"] for s in scenes)
    per_gpu_hour = images / elapsed_h / gpus
    price_per_gpu_hour = args.price_per_hour / gpus

    summary = {
        "machine": {k: info.get(k) for k in ("host", "gpus", "gpu_model", "jobs_per_gpu")},
        "price_per_hour": args.price_per_hour,
        "scenes": len(scenes), "images": images, "frames_rendered": rendered,
        "kept_fraction": round(images / max(rendered, 1), 3),
        "annotations": sum(s["annotations"] for s in scenes),
        "elapsed_h": round(elapsed_h, 3),
        "images_per_hour_node": round(images / elapsed_h, 1),
        "images_per_hour_per_gpu": round(per_gpu_hour, 1),
        "cost_per_1000_images": round(1000 / per_gpu_hour * price_per_gpu_hour, 2),
        "scene_wall_s": {p: round(float(np.percentile([s["wall_s"] for s in scenes], p)), 1)
                         for p in (10, 50, 90, 99)},
        "peak_rss_gb": {p: round(float(np.percentile([s["peak_rss_gb"] for s in scenes], p)), 2)
                        for p in (50, 90, 100)},
        "stage_mean_s": {st: round(float(np.mean([s["stages"].get(st, 0.0) for s in scenes])), 2)
                         for st in STAGES},
        "by_env": {},
        "rigs": {},
    }
    for s in scenes:
        e = summary["by_env"].setdefault(s["env"], {"scenes": 0, "wall_s": [], "rss": []})
        e["scenes"] += 1
        e["wall_s"].append(s["wall_s"])
        e["rss"].append(s["peak_rss_gb"])
        for r in s["rigs"]:
            summary["rigs"][r] = summary["rigs"].get(r, 0) + 1
    for e in summary["by_env"].values():
        e["wall_s_median"] = round(float(np.median(e.pop("wall_s"))), 1)
        e["rss_max_gb"] = round(float(np.max(e.pop("rss"))), 2)
    biomes = {}
    for s in scenes:
        biomes.setdefault(s["biome"], []).append(s["wall_s"])
    summary["biome_wall_s_median"] = {b: round(float(np.median(v)), 1) for b, v in sorted(biomes.items())}

    # ---- utilisation
    gpu_trace = read_gpu_trace(os.path.join(args.logs, "gpu_util.csv"))
    host_trace = read_host_trace(os.path.join(args.logs, "host_util.csv"))
    if gpu_trace:
        summary["gpu_util_mean_pct"] = round(float(np.mean([u for tr in gpu_trace.values() for _, u, _, _ in tr])), 1)
        summary["gpu_mem_peak_mib"] = int(max(m for tr in gpu_trace.values() for _, _, m, _ in tr))
        summary["gpu_mem_total_mib"] = int(max(t for tr in gpu_trace.values() for _, _, _, t in tr))
    if host_trace:
        summary["host_ram_peak_gib"] = round(max(r[1] for r in host_trace) / 1024, 1)
        summary["host_ram_total_gib"] = round(max(r[2] for r in host_trace) / 1024, 1)
        summary["host_load_mean"] = round(float(np.mean([r[3] for r in host_trace])), 1)

    # ---- projections: linear in GPU count (render nodes do not talk to each other)
    sizes = [int(x) for x in args.sizes.split(",")]
    gpu_counts = [1, 2, 4, 8, 16, 32]
    proj = {}
    for n in sizes:
        proj[n] = {g: {"hours": round(n / (per_gpu_hour * g), 2),
                       "cost": round(n / per_gpu_hour * price_per_gpu_hour, 0)} for g in gpu_counts}
    summary["projection"] = proj
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    # ---- figures
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

    envs = sorted({s["env"] for s in scenes})
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    for k, env in enumerate(envs):
        sub = [s for s in scenes if s["env"] == env]
        left = 0.0
        for st, col in zip(STAGES, COLORS):
            v = float(np.mean([s["stages"].get(st, 0.0) for s in sub]))
            ax.barh(k, v, left=left, color=col, label=st if k == 0 else None)
            left += v
        ax.text(left + 1, k, f"{left:.0f} s (n={len(sub)})", va="center", fontsize=8)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels(envs)
    ax.set_xlabel("mean seconds per scene")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stages.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.2))
    axes[0].hist([s["wall_s"] for s in scenes], bins=20, color="#4c72b0")
    axes[0].set_xlabel("scene wall time [s]")
    axes[0].set_ylabel("scenes")
    axes[1].hist([s["peak_rss_gb"] for s in scenes], bins=20, color="#dd8452")
    axes[1].set_xlabel("peak RAM per Blender process [GB]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scene_times.png"), bbox_inches="tight")
    plt.close(fig)

    if gpu_trace or host_trace:
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.2), sharex=True)
        t0 = min([tr[0][0] for tr in gpu_trace.values()] + [h[0] for h in host_trace[:1]])
        for idx, tr in sorted(gpu_trace.items()):
            t = [(x[0] - t0) / 60 for x in tr]
            axes[0].plot(t, [x[1] for x in tr], lw=0.6, label=f"GPU {idx}")
            axes[1].plot(t, [x[2] / 1024 for x in tr], lw=0.6)
        axes[0].set_ylabel("GPU util [%]")
        axes[0].legend(ncol=4, fontsize=7, frameon=False)
        axes[1].set_ylabel("VRAM [GiB]")
        if host_trace:
            ax2 = axes[1].twinx()
            ax2.plot([(h[0] - t0) / 60 for h in host_trace], [h[1] / 1024 for h in host_trace],
                     color="black", lw=0.8, ls="--")
            ax2.set_ylabel("host RAM [GiB] (dashed)")
        axes[1].set_xlabel("minutes since start")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "utilisation.png"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    for n in sizes:
        ax.plot(gpu_counts, [proj[n][g]["hours"] for g in gpu_counts], marker="o",
                label=f"{n // 1000}k images (${proj[n][1]['cost']:.0f})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(gpu_counts)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel("GPUs rendering in parallel")
    ax.set_ylabel("wall time [h]")
    ax.grid(True, which="both", lw=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "projection.png"), bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "projection"}, indent=1))
    print(f"[bench] figures and summary.json in {out}")


if __name__ == "__main__":
    main()
