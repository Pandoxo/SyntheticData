import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Ground-plane generator: trash dropped onto a bare textured plane under an HDRI.
#
# This is the fast, controlled variant - useful for sanity-checking assets, scale
# normalisation and annotation quality without paying for a full scene. Use
# generate_scene.py once you want real backgrounds.
#
# Render one scene:
#     blenderproc run generate.py -- --config config.yaml --scene-index 0
# Render the whole dataset (one process per scene):
#     .venv/bin/python run_batch.py --config config.yaml

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdlib import cli, environment, pipeline
from sdlib.environment import Environment


def build_environment(cfg, rng):
    """A flat ground slab plus an invisible spawn zone floating on its surface."""
    ground_cfg = cfg["ground"]
    half = float(ground_cfg["size_m"]) / 2.0
    thickness = float(ground_cfg["thickness_m"])

    # A flat cube rather than a plane: a plane with a BOX collision shape is a
    # degenerate zero-thickness box that Bullet handles badly, and fast-falling
    # objects can tunnel straight through it.
    ground = bproc.object.create_primitive("CUBE")
    ground.set_name("ground")
    ground.set_scale([half, half, thickness / 2.0])
    ground.set_location([0.0, 0.0, -thickness / 2.0])  # top face sits at z = 0
    ground.set_cp("category_id", 0)
    ground.enable_rigidbody(active=False, collision_shape="BOX",
                            friction=cfg["physics"]["ground_friction"])

    material = ground.new_material("ground_material")
    material.set_principled_shader_value("Base Color", list(rng.uniform(*ground_cfg["color_range"])) + [1.0])
    material.set_principled_shader_value("Roughness", float(rng.uniform(*ground_cfg["roughness_range"])))

    # The zone is a marker only; the ground slab above is the actual collider.
    zone_half = float(cfg["spawn"]["zone_size_m"]) / 2.0
    zone = bproc.object.create_primitive("PLANE")
    zone.set_name("spawn_ground")
    zone.set_scale([zone_half, zone_half, 1.0])
    zone.set_location([0.0, 0.0, 0.0])
    environment.prepare_zone(zone)

    return Environment(zones=[zone], colliders=[ground], statics=[ground],
                       description="ground_plane")


def main():
    args = cli.parse_args("config.yaml")
    cfg, catalog, hdri_paths, output_dir = cli.bootstrap(args)
    pipeline.generate(cfg, args.scene_index, catalog, hdri_paths,
                      build_environment, output_dir)


if __name__ == "__main__":
    main()
