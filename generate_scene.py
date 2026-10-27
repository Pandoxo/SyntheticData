import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Render one scene in a real .blend environment only - the same as
#     blenderproc run generate.py -- --env scene ...
# kept as its own entry point so `run_batch.py --script generate_scene.py` renders a
# scenes-only dataset.
#
# Marking where trash may appear: run prepare_scenes.py once per scene (it finds flat
# ground automatically and records which objects surround each patch), and/or add flat
# planes named `spawn_*` lying on the ground in the .blend. See sdlib/scenes.py.

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
    pipeline.generate(cfg, args.scene_index, catalog, resources, output_dir,
                      env_type=args.env or "scene")


if __name__ == "__main__":
    main()
