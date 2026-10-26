"""Put vegetation and clutter on the ground.

Two mechanisms, chosen by count:

instance_field   tens of thousands of small things (grass clumps, moss) - a point
                 cloud computed in numpy, instanced by a geometry-nodes modifier.
                 Instances share their mesh, so a lawn of 50k clumps costs about as
                 much memory as one clump.
place_items      tens of larger things (stones, weeds, branches, props) - linked
                 copies via ModelLibrary.place, which also makes them solid for the
                 camera's ray checks.

Both take "keep-out" discs so nothing grows through a lying bottle or under the
robot's wheels.
"""

import bpy
import numpy as np


def _keep_out_mask(points_xy, discs):
    keep = np.ones(len(points_xy), dtype=bool)
    for (cx, cy, r) in discs:
        keep &= (points_xy[:, 0] - cx) ** 2 + (points_xy[:, 1] - cy) ** 2 > r * r
    return keep


def sample_points(region_fn, center, radius, density, rng, discs=(), falloff=0.0):
    """Uniform points in a disc around `center`, filtered by `region_fn(x, y) -> z or None`.

    `falloff` thins density with distance (0 = flat, 1 = linear to zero at the rim):
    a lawn 9 m away is a texture at 10 cm camera height, so spending instances there
    buys nothing.
    """
    area = np.pi * radius * radius
    n = int(area * density)
    if n <= 0:
        return np.zeros((0, 3))
    r = radius * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2 * np.pi
    xy = np.stack([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)], axis=1)
    if falloff > 0:
        keep_p = 1.0 - falloff * (r / radius)
        xy = xy[rng.random(len(xy)) < keep_p]
    xy = xy[_keep_out_mask(xy, discs)]
    pts = []
    for x, y in xy:
        z = region_fn(x, y)
        if z is not None:
            pts.append((x, y, z))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def _instance_tree(name, collection, count, scale_range, tilt, seed):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    gin = nodes.new("NodeGroupInput")
    gout = nodes.new("NodeGroupOutput")

    info = nodes.new("GeometryNodeCollectionInfo")
    info.inputs["Collection"].default_value = collection
    info.inputs["Separate Children"].default_value = True
    info.inputs["Reset Children"].default_value = True

    inst = nodes.new("GeometryNodeInstanceOnPoints")
    inst.inputs["Pick Instance"].default_value = True
    links.new(gin.outputs["Geometry"], inst.inputs["Points"])
    links.new(info.outputs["Instances"], inst.inputs["Instance"])

    pick = nodes.new("FunctionNodeRandomValue")
    pick.data_type = "INT"
    # Random Value keeps one Min/Max pair per data type; by name, "Min" would be
    # the vector one. 4/5 are the integer pair.
    pick.inputs[4].default_value = 0
    pick.inputs[5].default_value = max(count - 1, 0)
    pick.inputs["Seed"].default_value = seed
    links.new(pick.outputs[2], inst.inputs["Instance Index"])

    rand_rot = nodes.new("FunctionNodeRandomValue")
    rand_rot.data_type = "FLOAT_VECTOR"
    rand_rot.inputs[0].default_value = (-tilt, -tilt, 0.0)
    rand_rot.inputs[1].default_value = (tilt, tilt, 6.2831853)
    rand_rot.inputs["Seed"].default_value = seed + 1
    to_rot = nodes.new("FunctionNodeEulerToRotation")
    links.new(rand_rot.outputs[0], to_rot.inputs[0])
    links.new(to_rot.outputs[0], inst.inputs["Rotation"])

    rand_scale = nodes.new("FunctionNodeRandomValue")
    rand_scale.data_type = "FLOAT"
    rand_scale.inputs[2].default_value = float(scale_range[0])
    rand_scale.inputs[3].default_value = float(scale_range[1])
    rand_scale.inputs["Seed"].default_value = seed + 2
    links.new(rand_scale.outputs[1], inst.inputs["Scale"])

    links.new(inst.outputs["Instances"], gout.inputs["Geometry"])
    return tree


def instance_field(name, points, prototypes, scale_range, rng, tilt=0.2, library_collection=None):
    """One object whose geometry-nodes modifier puts a random prototype on every point."""
    if len(points) == 0 or not prototypes:
        return None
    coll = bpy.data.collections.new(f"_sd_src_{name}")
    coll.use_fake_user = True  # see ModelLibrary: survives BlenderProc's orphan purge
    parent = library_collection or bpy.context.scene.collection
    parent.children.link(coll)
    for proto in prototypes:
        coll.objects.link(proto)

    mesh = bpy.data.meshes.new(f"sd_points_{name}")
    mesh.from_pydata([tuple(p) for p in points], [], [])
    obj = bpy.data.objects.new(f"sd_field_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj["category_id"] = 0

    tree = _instance_tree(f"sd_gn_{name}", coll, len(prototypes), scale_range, tilt,
                          int(rng.integers(0, 2 ** 20)))
    mod = obj.modifiers.new("sd_instances", "NODES")
    mod.node_group = tree
    return obj


def place_items(library, items, points, rng, scale_range=(1.0, 1.0), sink_frac=(0.0, 0.0),
                name="clutter"):
    """Linked copies of random `items` (variant part lists) at `points`.

    Returns (objects, records). Each placed item is one or more objects (multi-part).
    """
    placed, records = [], []
    if not items:
        return placed, records
    for k, p in enumerate(points):
        parts = items[int(rng.integers(0, len(items)))]
        scale = float(rng.uniform(*scale_range))
        low, high = library.item_size(parts)
        sink = float(rng.uniform(*sink_frac)) * float(high[2] - low[2])
        objs = library.place(parts, p, float(rng.uniform(0, 2 * np.pi)), scale=scale,
                             name=f"{name}{k:03d}", sink=sink)
        placed.extend(objs)
        records.append({"item": parts[0].name, "location": [round(float(c), 3) for c in p],
                        "scale": round(scale, 3)})
    return placed, records
