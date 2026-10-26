#!/usr/bin/env python3
"""Pull litter models from Objaverse (Sketchfab uploads, mostly CC-BY) into the catalogue.

Run with the venv python (no Blender needed):

    .venv/bin/python fetch_objaverse.py --dry-run              # counts per class only
    .venv/bin/python fetch_objaverse.py --max-per-class 150 --jobs 16

What it does:
1. reads Objaverse-LVIS (46k objects hand-sorted into 1156 categories) and keeps the
   categories in CATEGORIES below, each mapped to one of our classes;
2. reads each candidate's Sketchfab metadata and drops: non-commercial and
   share-alike licences, multi-item uploads ("set", "pack", "collection" ...),
   huge meshes, and anything the per-category keyword rules reject (a "cup" that
   is a porcelain tea cup, a "box" that is not cardboard);
3. downloads the GLBs into assets/objaverse/glbs/;
4. writes catalog/objaverse.yaml - class, real-world size, licence, author and URL
   per model (CC-BY requires attribution: the dataset's credits come from here).

LVIS labels are noisy. Before a big render, look at the models:
    blenderproc run catalog_sheet.py -- --source objaverse --out qa/objaverse
and list the uids of the bad ones in catalog/objaverse_rejects.txt (one per line,
`#` comments allowed); rerunning this script drops them from the catalogue.
"""

import argparse
import fcntl
import concurrent.futures
import gzip
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdlib import config

HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
USER_AGENT = "syntheticData-objaverse-fetch/1.0"
COMMERCIAL_OK = {"by", "cc0"}   # CC-BY-SA renders would have to be shared alike; NC is out

# LVIS category -> (class, longest_m, scale_jitter, keyword rule)
# A keyword rule is either None (take the category as is) or a dict:
#   require: at least one of these words must appear in name/tags/description
#   class_by: [(words, class), ...] first match decides the class; no match = skip
#   reject: any of these words drops the model
CATEGORIES = {
    "beer_bottle": ("bottle_glass", 0.24, 0.12, None),
    "wine_bottle": ("bottle_glass", 0.30, 0.10, None),
    "water_bottle": ("bottle_plastic", 0.22, 0.20, {"reject": ["metal", "steel", "thermos", "flask", "sport"]}),
    "bottle": (None, 0.25, 0.25, {"class_by": [
        (["glass", "wine", "beer", "whisky", "whiskey", "vodka", "rum", "liquor", "champagne"], "bottle_glass"),
        (["plastic", "pet", "water", "soda", "cola", "coke", "shampoo", "detergent", "milk", "juice"], "bottle_plastic")]}),
    "jar": ("bottle_glass", 0.12, 0.25, {"reject": ["cookie", "ceramic", "clay", "pot"]}),
    "beer_can": ("can", 0.12, 0.10, None),
    "can": ("can", 0.12, 0.25, {"reject": ["trash can", "garbage can", "watering", "jerry", "oil can"]}),
    "pop_(soda)": (None, 0.13, 0.15, {"class_by": [(["bottle"], "bottle_plastic"), (["can", "soda", "cola", "drink"], "can")]}),
    "aerosol_can": ("can", 0.19, 0.15, None),
    "carton": ("paper", 0.22, 0.25, {"reject": ["cigarette carton"]}),
    "fruit_juice": (None, 0.20, 0.20, {"class_by": [(["carton", "box", "tetra", "pack"], "paper"), (["bottle"], "bottle_plastic")]}),
    "orange_juice": (None, 0.20, 0.20, {"class_by": [(["carton", "box", "tetra", "pack"], "paper"), (["bottle"], "bottle_plastic")]}),
    "milk": (None, 0.22, 0.20, {"class_by": [(["carton", "box", "tetra"], "paper"), (["bottle", "plastic"], "bottle_plastic")]}),
    "box": ("paper", 0.30, 0.40, {"require": ["cardboard", "carton", "cereal", "pizza", "package", "parcel", "shipping", "packaging", "paper"]}),
    "cup": ("cup", 0.11, 0.20, {"require": ["paper", "plastic", "disposable", "takeaway", "take away", "coffee cup", "starbucks", "foam", "styrofoam", "to go", "soda cup"],
                                "reject": ["mug", "ceramic", "porcelain", "tea cup", "teacup", "glass", "saucer", "trophy"]}),
    "Dixie_cup": ("cup", 0.09, 0.20, None),
    "paper_plate": ("paper", 0.22, 0.15, None),
    "shopping_bag": (None, 0.40, 0.25, {"class_by": [(["paper", "kraft"], "paper"), (["plastic", "bag"], "plastic_film")]}),
    "grocery_bag": (None, 0.40, 0.25, {"class_by": [(["paper", "kraft"], "paper"), (["plastic", "bag"], "plastic_film")]}),
    "plastic_bag": ("plastic_film", 0.35, 0.25, None),
    "packet": ("plastic_film", 0.14, 0.30, None),
    "pouch": ("plastic_film", 0.17, 0.30, {"reject": ["leather", "belt", "military", "tactical", "ammo"]}),
    "crisp_(potato_chip)": ("plastic_film", 0.25, 0.20, {"require": ["bag", "pack", "packet", "lays", "pringles", "chips"]}),
    "chocolate_bar": ("plastic_film", 0.15, 0.20, {"require": ["wrap", "wrapper", "pack", "bar"]}),
    "candy_bar": ("plastic_film", 0.13, 0.20, None),
    "tinfoil": ("plastic_film", 0.15, 0.40, None),
    "cigarette": ("cigarette", 0.06, 0.40, {"reject": ["pack", "box", "case", "lighter", "ashtray"]}),
    "bottle_cap": ("small_item", 0.03, 0.20, None),
    "cork_(bottle_plug)": ("small_item", 0.04, 0.20, None),
    "straw_(for_drinking)": ("small_item", 0.20, 0.20, None),
    "newspaper": ("paper", 0.35, 0.20, None),
    "magazine": ("paper", 0.28, 0.15, None),
    "envelope": ("paper", 0.22, 0.20, None),
    "napkin": ("paper", 0.20, 0.25, None),
    "paper_towel": ("paper", 0.25, 0.25, None),
    "toilet_tissue": ("paper", 0.11, 0.10, None),
    "tissue_paper": ("paper", 0.20, 0.25, None),
    "tray": ("container", 0.25, 0.25, {"require": ["plastic", "food", "takeaway", "foam", "styrofoam", "disposable", "aluminium", "aluminum"]}),
    "mask": ("other", 0.17, 0.10, {"require": ["face", "surgical", "medical", "covid", "n95", "ffp2"]}),
    "diaper": ("other", 0.30, 0.15, None),
}

# Uploads holding several items, or a whole scene, become one box spanning them all.
MULTI_ITEM = re.compile(r"\b(set|sets|pack of|collection|bundle|kit|pile|scene|room|shelf|"
                        r"assets|props|asset pack|lot|group|several|various)\b", re.IGNORECASE)


def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as err:
            if attempt == 4:
                raise
            print(f"[objaverse] retry {url}: {err}")
            time.sleep(2 ** attempt)
    return None


def cached_json_gz(url, cache):
    path = os.path.join(cache, os.path.basename(url))
    if not os.path.isfile(path):
        os.makedirs(cache, exist_ok=True)
        with open(path, "wb") as f:
            f.write(get(url).content)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def text_of(meta):
    tags = " ".join(t.get("name", "") if isinstance(t, dict) else str(t) for t in meta.get("tags", []))
    return f"{meta.get('name', '')} {tags} {meta.get('description', '')[:300]}".lower()


def decide(category, meta):
    """Return (class, reason) - class None means rejected."""
    cls, _, _, rule = CATEGORIES[category]
    text = text_of(meta)
    if MULTI_ITEM.search(meta.get("name", "")):
        return None, "multi-item name"
    if rule:
        if any(w in text for w in rule.get("reject", [])):
            return None, "reject keyword"
        if rule.get("require") and not any(w in text for w in rule["require"]):
            return None, "missing required keyword"
        if rule.get("class_by"):
            cls = next((c for words, c in rule["class_by"] if any(w in text for w in words)), None)
            if cls is None:
                return None, "class undecidable"
    return cls, "ok"


def _lock(root):
    """One fetcher at a time per asset folder: two would race on the same files."""
    handle = open(os.path.join(root, ".fetch.lock"), "w")  # pylint: disable=consider-using-with
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets-root", default="assets")
    parser.add_argument("--catalog", default="catalog/objaverse.yaml")
    parser.add_argument("--rejects", default="catalog/objaverse_rejects.txt")
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--max-faces", type=int, default=400000)
    parser.add_argument("--max-mb", type=float, default=60.0)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = config.resolve(args.assets_root)
    os.makedirs(root, exist_ok=True)
    lock = _lock(root)  # noqa: F841 - held until exit
    cache = os.path.join(root, "objaverse", "meta")
    glb_dir = os.path.join(root, "objaverse", "glbs")

    rejects = set()
    rejects_path = config.resolve(args.rejects)
    if os.path.isfile(rejects_path):
        with open(rejects_path, encoding="utf-8") as f:
            rejects = {line.split("#")[0].strip() for line in f if line.split("#")[0].strip()}

    lvis = cached_json_gz(f"{HF}/lvis-annotations.json.gz", cache)
    paths = cached_json_gz(f"{HF}/object-paths.json.gz", cache)
    candidates = {}
    for category in CATEGORIES:
        for uid in lvis.get(category, []):
            candidates.setdefault(uid, category)
    print(f"[objaverse] {len(candidates)} candidates in {len(CATEGORIES)} LVIS categories")

    # Metadata comes in 160 shards keyed by the GLB's folder.
    shards = sorted({paths[uid].split("/")[1] for uid in candidates if uid in paths})
    metadata = {}
    for shard in shards:
        metadata.update({k: v for k, v in cached_json_gz(f"{HF}/metadata/{shard}.json.gz", cache).items()
                         if k in candidates})

    kept, reasons, per_class = [], {}, {}
    for uid, category in sorted(candidates.items()):
        meta = metadata.get(uid)
        if meta is None or uid not in paths:
            reasons["no metadata"] = reasons.get("no metadata", 0) + 1
            continue
        if uid in rejects:
            reasons["manually rejected"] = reasons.get("manually rejected", 0) + 1
            continue
        lic = str(meta.get("license", "")).lower()
        if lic not in COMMERCIAL_OK:
            reasons[f"license {lic}"] = reasons.get(f"license {lic}", 0) + 1
            continue
        if int(meta.get("faceCount") or 0) > args.max_faces:
            reasons["too many faces"] = reasons.get("too many faces", 0) + 1
            continue
        cls, why = decide(category, meta)
        if cls is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        if per_class.get(cls, 0) >= args.max_per_class:
            reasons["class full"] = reasons.get("class full", 0) + 1
            continue
        per_class[cls] = per_class.get(cls, 0) + 1
        _, longest, jitter, _ = CATEGORIES[category]
        user = meta.get("user") or {}
        kept.append({"uid": uid, "category": category, "class": cls, "longest_m": longest,
                     "scale_jitter": jitter, "license": "CC-BY-4.0" if lic == "by" else "CC0",
                     "name": meta.get("name", ""), "author": user.get("displayName") or user.get("username", ""),
                     "url": meta.get("viewerUrl", f"https://sketchfab.com/3d-models/{uid}"),
                     "glb": paths[uid]})

    print(f"[objaverse] kept {len(kept)}: {dict(sorted(per_class.items()))}")
    print(f"[objaverse] dropped: {dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}")
    if args.dry_run:
        return 0

    os.makedirs(glb_dir, exist_ok=True)

    def fetch(entry):
        dest = os.path.join(glb_dir, f"{entry['uid']}.glb")
        if os.path.isfile(dest):
            return entry, True
        r = get(f"{HF}/{entry['glb']}")
        if len(r.content) > args.max_mb * 1e6:
            return entry, False
        with open(dest + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dest + ".part", dest)
        return entry, True

    ok = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fetch, e) for e in kept]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                entry, good = future.result()
                if good:
                    ok.append(entry)
            except Exception as err:  # pylint: disable=broad-except
                print(f"[objaverse] download failed: {err}")
            if done % 100 == 0:
                print(f"[objaverse] {done}/{len(kept)} downloaded")

    rel_root = os.path.relpath(glb_dir, config.PROJECT_ROOT)
    lines = ["# Generated by fetch_objaverse.py - do not edit by hand; reject models via",
             "# catalog/objaverse_rejects.txt and rerun. Licences: CC-BY-4.0 needs the",
             "# author and url credited wherever the dataset is published.", "", "assets:"]
    for e in sorted(ok, key=lambda e: (e["class"], e["uid"])):
        name = json.dumps(e["name"][:80], ensure_ascii=False)
        author = json.dumps(e["author"][:60], ensure_ascii=False)
        lines.append(f"  - {{path: \"{rel_root}/{e['uid']}.glb\", class: {e['class']}, "
                     f"longest_m: {e['longest_m']}, scale_jitter: {e['scale_jitter']}, "
                     f"source: objaverse, license: {e['license']}, lvis: \"{e['category']}\", "
                     f"name: {name}, author: {author}, url: \"{e['url']}\"}}")
    with open(config.resolve(args.catalog), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[objaverse] {len(ok)} model(s) in {args.catalog}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
