"""Shared command line handling and start-up for the generator entry points."""

import argparse
import os

import blenderproc as bproc

from . import assets as assets_mod
from . import capture, config, environment
from .library import AssetIndex


def parse_args(default_config):
    parser = argparse.ArgumentParser(
        description="Render one scene of the synthetic trash dataset. "
                    "Use run_batch.py to render many.")
    parser.add_argument("--config", default=default_config,
                        help="path to the YAML config (default: %(default)s)")
    parser.add_argument("--scene-index", type=int, default=0,
                        help="index of the scene to render; also drives the seed")
    parser.add_argument("--output-dir", default=None,
                        help="where to write this scene (default: <output_dir>/scene_XXXXX)")
    parser.add_argument("--env", choices=["procedural", "scene"], default=None,
                        help="force one environment type instead of the config's mix")
    parser.add_argument("--biome", default=None,
                        help="force one procedural biome (sidewalk, lawn, forest, ...)")
    parser.add_argument("--debug", action="store_true",
                        help="wait for a debugpy client on port 5678 before running")
    return parser.parse_args()


def bootstrap(args):
    """Load the config, initialise BlenderProc and build the shared resources."""
    if args.debug:
        import debugpy  # pylint: disable=import-outside-toplevel
        debugpy.listen(5678)
        print("Waiting for debugger attach on port 5678...")
        debugpy.wait_for_client()

    cfg = config.load(config.resolve(args.config))
    if getattr(args, "biome", None):
        cfg["procedural"]["biome_weights"] = {args.biome: 1.0}

    output_dir = args.output_dir or os.path.join(
        config.resolve(cfg["output_dir"]), f"scene_{args.scene_index:05d}")
    output_dir = config.resolve(output_dir)

    bproc.init()
    capture.configure_renderer(cfg)

    obj_cfg = cfg["objects"]
    catalog = assets_mod.Catalog.from_config(
        config.resolve_all(obj_cfg["catalogs"]),
        config.resolve_all(obj_cfg["model_dirs"]),
        obj_cfg.get("class_weights"),
    )

    index = AssetIndex(cfg.get("assets_root", "assets"))
    hdri_paths = index.hdri_paths()
    extra_folders = [f for f in config.resolve_all(cfg["hdri"].get("folders")) if os.path.isdir(f)]
    if extra_folders:
        hdri_paths += environment.list_hdris(extra_folders)
    if not hdri_paths:
        raise RuntimeError("No HDRIs found - run fetch_assets.py or set hdri.folders")
    dark = frozenset(asset_id for (source, asset_id), row in index.manifest.items()
                     if row.get("type") == "hdri" and "night" in row.get("categories", []))
    print(f"[init] {len(hdri_paths)} HDRI(s) available ({len(dark)} night)")

    resources = {"index": index, "hdri_paths": sorted(set(hdri_paths)), "dark_hdris": dark}
    return cfg, catalog, resources, output_dir
