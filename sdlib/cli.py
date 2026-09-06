"""Shared command line handling and start-up for the generator entry points."""

import argparse
import os

import blenderproc as bproc

from . import assets as assets_mod
from . import capture, config, environment


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

    output_dir = args.output_dir or os.path.join(
        config.resolve(cfg["output_dir"]), f"scene_{args.scene_index:05d}")
    output_dir = config.resolve(output_dir)

    bproc.init()
    capture.configure_renderer(cfg)

    catalog = assets_mod.Catalog.from_config(
        config.resolve(cfg["objects"]["assets_file"]),
        config.resolve_all(cfg["objects"]["model_dirs"]),
    )
    hdri_paths = environment.list_hdris(config.resolve_all(cfg["hdri"]["folders"]))
    print(f"[init] {len(hdri_paths)} HDRI(s) available")

    return cfg, catalog, hdri_paths, output_dir
