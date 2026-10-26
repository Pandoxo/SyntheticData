#!/usr/bin/env python3
"""Download the environment assets: ground materials, HDRIs, vegetation, props.

Run with the venv python (plain Python, no Blender needed):

    .venv/bin/python fetch_assets.py --profile dev            # small laptop subset
    .venv/bin/python fetch_assets.py --profile full --jobs 16 # render machines
    .venv/bin/python fetch_assets.py --profile full --dry-run # just list and size

What to fetch lives in asset_sources.yaml. Files land under `assets/` in the layouts
BlenderProc's own loaders expect, so `bproc.loader.load_haven_mat` and
`load_ccmaterials` work on them unchanged:

    assets/polyhaven/textures/<id>/<id>_diff_2k.jpg ...
    assets/polyhaven/hdris/<id>/<id>_2k.hdr
    assets/polyhaven/models/<id>/<id>_2k.blend  (+ textures/ next to it)
    assets/ambientcg/<AssetId>/<AssetId>_2K-JPG_Color.jpg ...

Two files tie it together:
    assets/index.json      group name -> ids, which the generators read
    assets/manifest.jsonl  one line per asset: source, id, licence, authors, url

Reruns only download what is missing, so an interrupted fetch is resumed by running
the same command again.
"""

import argparse
import fcntl
import concurrent.futures
import io
import json
import os
import sys
import threading
import time
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdlib import config

USER_AGENT = "syntheticData-asset-fetch/1.0 (synthetic litter dataset; contact via github.com/Pandoxo)"
PH_API = "https://api.polyhaven.com"
ACG_API = "https://ambientcg.com/api/v2/full_json"
ACG_GET = "https://ambientcg.com/get?file={file}"

# Poly Haven map key -> what BlenderProc's haven loader looks for in the filename.
PH_TEXTURE_MAPS = ("Diffuse", "nor_gl", "Rough", "Displacement", "AO")
# ambientCG zip members worth keeping; the loader ignores the rest anyway.
ACG_KEEP = ("_Color.", "_NormalGL.", "_Roughness.", "_Displacement.",
            "_AmbientOcclusion.", "_Metalness.", "_Opacity.")

_session_local = threading.local()


class Gone(Exception):
    """The source no longer serves this file (HTTP 404) - skip it, do not retry."""


def session():
    if not hasattr(_session_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _session_local.s = s
    return _session_local.s


def get_json(url, params=None):
    for attempt in range(5):
        try:
            r = session().get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as err:
            if attempt == 4:
                raise
            print(f"[fetch] retry {url}: {err}")
            time.sleep(2 ** attempt)
    return None


def download(url, dest, expected_size=None):
    """Fetch `url` to `dest` atomically. Returns bytes written (0 if already there)."""
    if os.path.isfile(dest) and (expected_size is None or os.path.getsize(dest) == expected_size):
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(5):
        try:
            with session().get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                raise Gone(url) from err
            if attempt == 4:
                raise
            print(f"[fetch] retry {os.path.basename(dest)}: {err}")
            time.sleep(2 ** attempt)
        except requests.RequestException as err:
            if attempt == 4:
                raise
            print(f"[fetch] retry {os.path.basename(dest)}: {err}")
            time.sleep(2 ** attempt)
    return 0


# ------------------------------------------------------------------- Poly Haven

def _ph_select(listing, spec, limit):
    any_of = set(spec.get("any_of", []))
    require = set(spec.get("require", []))
    exclude = [e.lower() for e in spec.get("exclude", [])]
    exclude_cats = set(spec.get("exclude_categories", []))
    chosen = []
    for asset_id in sorted(listing):
        cats = set(listing[asset_id].get("categories", []))
        if any_of and not cats & any_of:
            continue
        if not require <= cats or cats & exclude_cats:
            continue
        if any(e in asset_id.lower() for e in exclude):
            continue
        chosen.append(asset_id)
    group_limit = spec.get("limit")
    for cap in (group_limit, limit):
        if cap is not None:
            chosen = chosen[:cap]
    return chosen


def _ph_manifest(kind, asset_id, info, group):
    return {"source": "polyhaven", "type": kind, "id": asset_id, "group": group,
            "license": "CC0", "authors": sorted(info.get("authors", {})),
            "url": f"https://polyhaven.com/a/{asset_id}",
            "categories": info.get("categories", []),
            "dimensions_mm": info.get("dimensions")}


def plan_polyhaven(spec, profile, root):
    """Return (jobs, manifest_rows, index) for everything Poly Haven contributes."""
    jobs, rows, index = [], [], {}
    limit = profile["limit_per_group"]

    tex_listing = get_json(f"{PH_API}/assets", {"t": "textures"})
    for group, gspec in spec.get("textures", {}).items():
        ids = _ph_select(tex_listing, gspec, limit)
        index.setdefault("polyhaven_textures", {})[group] = ids
        res = profile["texture_res"]
        for asset_id in ids:
            files = get_json(f"{PH_API}/files/{asset_id}")
            for key in PH_TEXTURE_MAPS:
                by_fmt = files.get(key, {}).get(res, {})
                entry = by_fmt.get("jpg") or by_fmt.get("png")
                if entry:
                    name = os.path.basename(entry["url"])
                    jobs.append((entry["url"], os.path.join(root, "polyhaven", "textures", asset_id, name),
                                 entry.get("size")))
            rows.append(_ph_manifest("texture", asset_id, tex_listing[asset_id], group))

    hdri_listing = get_json(f"{PH_API}/assets", {"t": "hdris"})
    for group, gspec in spec.get("hdris", {}).items():
        ids = _ph_select(hdri_listing, gspec, limit)
        index.setdefault("polyhaven_hdris", {})[group] = ids
        res = profile["hdri_res"]
        for asset_id in ids:
            files = get_json(f"{PH_API}/files/{asset_id}")
            entry = files.get("hdri", {}).get(res, {}).get("hdr")
            if entry:
                jobs.append((entry["url"], os.path.join(root, "polyhaven", "hdris", asset_id,
                                                        os.path.basename(entry["url"])), entry.get("size")))
                rows.append(_ph_manifest("hdri", asset_id, hdri_listing[asset_id], group))

    model_listing = get_json(f"{PH_API}/assets", {"t": "models"})
    for group, ids in spec.get("models", {}).items():
        ids = [i for i in ids if i in model_listing]
        if limit is not None:
            ids = ids[:limit]
        index.setdefault("polyhaven_models", {})[group] = ids
        res = profile["model_res"]
        for asset_id in ids:
            files = get_json(f"{PH_API}/files/{asset_id}")
            blend = files.get("blend", {}).get(res, {}).get("blend")
            if not blend:
                print(f"[polyhaven] {asset_id}: no {res} .blend, skipping")
                continue
            model_dir = os.path.join(root, "polyhaven", "models", asset_id)
            jobs.append((blend["url"], os.path.join(model_dir, os.path.basename(blend["url"])),
                         blend.get("size")))
            for relpath, inc in blend.get("include", {}).items():
                jobs.append((inc["url"], os.path.join(model_dir, relpath), inc.get("size")))
            # Some Poly Haven .blend files are saved by a Blender newer than the 4.2
            # BlenderProc runs, and 4.2 cannot open them. Litter models also get the
            # version-independent glTF, which sdlib/assets.py falls back to.
            gltf = files.get("gltf", {}).get(res, {}).get("gltf")
            if group in spec.get("gltf_fallback_groups", []) and gltf:
                gltf_dir = os.path.join(model_dir, "gltf")
                jobs.append((gltf["url"], os.path.join(gltf_dir, os.path.basename(gltf["url"])),
                             gltf.get("size")))
                for relpath, inc in gltf.get("include", {}).items():
                    jobs.append((inc["url"], os.path.join(gltf_dir, relpath), inc.get("size")))
            rows.append(_ph_manifest("model", asset_id, model_listing[asset_id], group))
    return jobs, rows, index


# --------------------------------------------------------------------- ambientCG

def _acg_listing():
    assets, offset = [], 0
    while True:
        page = get_json(ACG_API, {"type": "Material", "limit": 250, "offset": offset,
                                  "sort": "Alphabet"})
        found = page.get("foundAssets", [])
        assets.extend(found)
        offset += len(found)
        if not found or offset >= int(page.get("numberOfResults", 0)):
            return assets


def plan_ambientcg(spec, profile, root):
    jobs, rows, index = [], [], {}
    listing = _acg_listing()
    limit = profile["limit_per_group"]
    res = profile["acg_res"]
    for group, cats in spec.get("materials", {}).items():
        wanted = set(cats)
        ids = sorted(a["assetId"] for a in listing if a.get("displayCategory") in wanted)
        if limit is not None:
            ids = ids[:limit]
        index.setdefault("ambientcg", {})[group] = ids
        by_id = {a["assetId"]: a for a in listing}
        for asset_id in ids:
            jobs.append(("acg-zip", os.path.join(root, "ambientcg", asset_id), f"{asset_id}_{res}.zip"))
            rows.append({"source": "ambientcg", "type": "material", "id": asset_id, "group": group,
                         "license": "CC0", "authors": ["ambientCG"],
                         "url": f"https://ambientcg.com/a/{asset_id}",
                         "categories": [by_id[asset_id].get("displayCategory")]})
    return jobs, rows, index


def fetch_acg_zip(dest_dir, file_name):
    """Download one ambientCG material zip and keep only the maps we use."""
    if os.path.isdir(dest_dir) and any(n.endswith("_Color.jpg") for n in os.listdir(dest_dir)):
        return 0
    url = ACG_GET.format(file=file_name)
    for attempt in range(5):
        try:
            r = session().get(url, timeout=180)
            if r.status_code == 404:
                raise Gone(url)
            r.raise_for_status()
            break
        except requests.RequestException as err:
            if attempt == 4:
                raise
            print(f"[fetch] retry {file_name}: {err}")
            time.sleep(2 ** attempt)
    os.makedirs(dest_dir, exist_ok=True)
    written = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".jpg") and any(k in member for k in ACG_KEEP):
                data = zf.read(member)
                with open(os.path.join(dest_dir, os.path.basename(member)), "wb") as f:
                    f.write(data)
                written += len(data)
    return written


# -------------------------------------------------------------------------- main

def run_job(job):
    if job[0] == "acg-zip":
        return fetch_acg_zip(job[1], job[2])
    url, dest, size = job
    return download(url, dest, size)


def merge_manifest(path, rows):
    existing = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[(row["source"], row["type"], row["id"])] = row
    for row in rows:
        existing[(row["source"], row["type"], row["id"])] = row
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(existing):
            f.write(json.dumps(existing[key], ensure_ascii=False) + "\n")
    return len(existing)


def merge_index(path, index):
    current = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            current = json.load(f)
    for kind, groups in index.items():
        for group, ids in groups.items():
            merged = sorted(set(current.get(kind, {}).get(group, [])) | set(ids))
            current.setdefault(kind, {})[group] = merged
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=1, sort_keys=True)


def _lock(root):
    """One fetcher at a time per asset folder: two would race on the same files."""
    handle = open(os.path.join(root, ".fetch.lock"), "w")  # pylint: disable=consider-using-with
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default="asset_sources.yaml")
    parser.add_argument("--profile", default="dev", help="profile name from the sources file")
    parser.add_argument("--only", choices=["polyhaven", "ambientcg"], default=None)
    parser.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = parser.parse_args()

    spec = config.load(config.resolve(args.sources))
    root_dir = config.resolve(spec.get("root", "assets"))
    os.makedirs(root_dir, exist_ok=True)
    lock = _lock(root_dir)  # noqa: F841 - held until exit
    profile = spec["profiles"][args.profile]
    root = config.resolve(spec.get("root", "assets"))
    os.makedirs(root, exist_ok=True)

    jobs, rows, index = [], [], {}
    if args.only in (None, "polyhaven"):
        j, r, i = plan_polyhaven(spec.get("polyhaven", {}), profile, root)
        jobs += j; rows += r; index.update(i)
        print(f"[polyhaven] {len(r)} assets, {len(j)} files")
    if args.only in (None, "ambientcg"):
        j, r, i = plan_ambientcg(spec.get("ambientcg", {}), profile, root)
        jobs += j; rows += r; index.update(i)
        print(f"[ambientcg] {len(r)} materials")

    for kind, groups in index.items():
        for group, ids in groups.items():
            print(f"[plan] {kind}/{group}: {len(ids)}")

    if args.dry_run:
        known = sum(j[2] or 0 for j in jobs if j[0] != "acg-zip")
        print(f"[plan] dry run - {len(jobs)} downloads, {known / 1e9:.1f} GB known size "
              f"(ambientCG zips not included)")
        return 0

    started, total, failed, gone = time.time(), 0, [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                total += future.result()
            except Gone as err:
                gone.append(str(err))
            except Exception as err:  # pylint: disable=broad-except
                failed.append((futures[future], err))
            if done % 50 == 0 or done == len(jobs):
                print(f"[fetch] {done}/{len(jobs)} files, {total / 1e9:.2f} GB new, "
                      f"{time.time() - started:.0f}s")

    n = merge_manifest(os.path.join(root, "manifest.jsonl"), rows)
    merge_index(os.path.join(root, "index.json"), index)
    print(f"[fetch] manifest: {n} assets; {len(failed)} failed download(s), "
          f"{len(gone)} no longer offered by the source (404, skipped)")
    for job, err in failed[:20]:
        print(f"[fetch]   {job[1]}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
