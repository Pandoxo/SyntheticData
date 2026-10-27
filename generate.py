import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Render one scene of the dataset.
#
# Which kind of world the scene is built in comes from `environments` in the config:
#   procedural - a generated patch of sidewalk / street / lawn / park / forest ...
#                (sdlib/procedural.py), the bulk of the dataset
#   scene      - a corner of a real .blend environment (sdlib/scenes.py)
# `--env` forces one of them, which is handy when working on either.
#
# Render one scene:
#     blenderproc run generate.py -- --config config.yaml --scene-index 0
# Render the whole dataset (one process per scene):
#     .venv/bin/python run_batch.py --config config.yaml

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdlib import cli, pipeline


def main():
    import time  # pylint: disable=import-outside-toplevel
    started = time.time()
    args = cli.parse_args("config.yaml")
    cfg, catalog, resources, output_dir = cli.bootstrap(args)
    pipeline._PROFILE["stages"]["init"] = {"s": round(time.time() - started, 2)}  # pylint: disable=protected-access
    pipeline.generate(cfg, args.scene_index, catalog, resources, output_dir, env_type=args.env)


if __name__ == "__main__":
    main()
