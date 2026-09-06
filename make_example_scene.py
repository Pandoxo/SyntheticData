import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Builds resources/example_scene.blend: a minimal environment laid out the way
# generate_scene.py expects, so you have something runnable before you have marked up
# your own BlenderKit scenes.
#
#     blenderproc run make_example_scene.py
#
# It is also the shortest description of the convention: a scene is just geometry,
# plus one or more flat planes named `spawn_*` lying on the ground where trash may
# appear. Open the result in Blender to see the layout, then reproduce it in your own
# scenes by hand.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy

from sdlib import config

OUTPUT = config.resolve("resources/example_scene.blend")


def add_box(name, size, location, color):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    material = bpy.data.materials.new(f"{name}_material")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
    obj.data.materials.append(material)
    return obj


def add_spawn_zone(name, size_xy, location):
    """A spawn zone: a flat plane sitting on the surface trash should land on.

    Nothing about it is special beyond the name - generate_scene.py hides it from the
    render, keeps it out of the annotations, and builds an invisible collider whose
    top face sits on it.
    """
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size_xy[0], size_xy[1], 1.0)
    return obj


def main():
    bproc.init()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # Pavement at z = 0, a kerb, a road slightly lower, and a wall to occlude things.
    add_box("pavement", (8.0, 4.0, 0.2), (0.0, 1.0, -0.1), (0.55, 0.54, 0.52, 1.0))
    add_box("kerb", (8.0, 0.15, 0.3), (0.0, -1.0, -0.05), (0.62, 0.61, 0.58, 1.0))
    add_box("road", (8.0, 4.0, 0.2), (0.0, -3.1, -0.22), (0.18, 0.18, 0.19, 1.0))
    add_box("wall", (8.0, 0.3, 2.6), (0.0, 3.1, 1.3), (0.48, 0.40, 0.35, 1.0))

    # Two zones: an open stretch of pavement, and the gutter against the kerb where
    # litter actually collects. Both lie exactly on the surface below them.
    add_spawn_zone("spawn_pavement", (1.6, 1.2), (-1.5, 0.6, 0.0))
    add_spawn_zone("spawn_gutter", (2.0, 0.5, ), (2.0, -1.35, -0.12))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT)
    print(f"[example] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
