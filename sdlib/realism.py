"""Make clean catalogue models look like litter that has been lying around.

Three cheap tricks that multiply the effective size of the catalogue:

deform   crush, bend or flatten the mesh of an instance before it falls, so one can
         model yields many different-looking cans. Baked into the mesh, so the
         physics collider and the masks follow the deformed shape.
grime    dirt and faded colour on top of the model's own material.
bury     after landing, sink part of an item into soft ground (grass, soil, leaves,
         gravel). Litter on a lawn is never perched on top of it.

And one for the lighting: a dappled-shadow "canopy" above the scene that the camera
never sees, standing in for trees and buildings outside the frame.
"""

import bpy
import numpy as np

DEFORMABLE_CLASSES = {"can", "bottle_plastic", "cup", "container", "paper", "plastic_film"}


def _bake_modifiers(blender_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = blender_obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                           depsgraph=depsgraph)
    old = blender_obj.data
    blender_obj.modifiers.clear()
    blender_obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)


def deform(obj, class_name, cfg, rng):
    """Randomly crush / bend / flatten one instance. Returns what was done, or None."""
    if class_name not in DEFORMABLE_CLASSES or rng.random() >= float(cfg["probability"]):
        return None
    bo = obj.blender_obj
    local_dims = np.array(bo.dimensions) / np.maximum(np.abs(np.array(bo.scale)), 1e-9)
    size = float(local_dims.max())
    kind = str(rng.choice(["crush", "bend", "flatten"], p=[0.45, 0.3, 0.25]))

    if len(bo.data.vertices) < 1500 and kind == "crush":
        # Displacement needs vertices to move; a 32-segment can has almost none.
        sub = bo.modifiers.new("sd_subdiv", "SUBSURF")
        sub.subdivision_type = "SIMPLE"
        sub.levels = sub.render_levels = 2

    if kind == "crush":
        tex = bpy.data.textures.new("sd_crush", type="CLOUDS")
        tex.noise_scale = float(rng.uniform(0.15, 0.4)) * size
        disp = bo.modifiers.new("sd_crush", "DISPLACE")
        disp.texture = tex
        disp.texture_coords = "LOCAL"
        disp.strength = float(rng.uniform(*cfg["crush_strength_frac"])) * size
        disp.mid_level = 0.5
        if rng.random() < 0.5:
            taper = bo.modifiers.new("sd_taper", "SIMPLE_DEFORM")
            taper.deform_method = "TAPER"
            taper.factor = float(rng.uniform(-0.5, 0.5))
            taper.deform_axis = str(rng.choice(["X", "Y", "Z"]))
    elif kind == "bend":
        bend = bo.modifiers.new("sd_bend", "SIMPLE_DEFORM")
        bend.deform_method = "BEND"
        bend.angle = float(np.radians(rng.uniform(*cfg["bend_deg"])) * rng.choice([-1, 1]))
        bend.deform_axis = str(rng.choice(["X", "Y", "Z"]))
    else:
        axis = int(rng.integers(0, 3))
        factor = float(rng.uniform(*cfg["flatten_factor"]))
        for v in bo.data.vertices:
            v.co[axis] *= factor

    _bake_modifiers(bo)
    return kind


def add_grime(obj, cfg, rng):
    """Dirt patches and sun-faded colour layered over every material of `obj`."""
    if rng.random() >= float(cfg["probability"]):
        return None
    strength = float(rng.uniform(*cfg["strength"]))
    dirt = np.array([rng.uniform(0.08, 0.22), rng.uniform(0.06, 0.16), rng.uniform(0.03, 0.10), 1.0])
    fade = float(rng.uniform(*cfg["desaturate"]))
    scale = float(rng.uniform(8, 40))

    for slot in obj.blender_obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        nodes, links = nt.nodes, nt.links
        base_in = bsdf.inputs["Base Color"]
        source = base_in.links[0].from_socket if base_in.is_linked else None

        coord = nodes.new("ShaderNodeTexCoord")
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = scale
        noise.inputs["Detail"].default_value = 6.0
        links.new(coord.outputs["Object"], noise.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = float(rng.uniform(0.45, 0.6))
        ramp.color_ramp.elements[1].position = min(ramp.color_ramp.elements[0].position + 0.15, 0.99)
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        factor = nodes.new("ShaderNodeMath")
        factor.operation = "MULTIPLY"
        factor.inputs[1].default_value = strength
        links.new(ramp.outputs["Color"], factor.inputs[0])

        hsv = nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = 1.0 - fade
        if source is not None:
            links.new(source, hsv.inputs["Color"])
        else:
            hsv.inputs["Color"].default_value = base_in.default_value
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        mix.inputs["B"].default_value = dirt
        links.new(hsv.outputs["Color"], mix.inputs["A"])
        links.new(factor.outputs["Value"], mix.inputs["Factor"])
        links.new(mix.outputs["Result"], base_in)

        rough = bsdf.inputs["Roughness"]
        if not rough.is_linked:
            rough.default_value = max(rough.default_value, 0.45 + 0.4 * strength)
    return {"strength": round(strength, 3), "desaturate": round(fade, 3)}


def bury(obj, max_frac, rng):
    """Sink `obj` by up to `max_frac` of its height. Returns the depth in metres."""
    bb = np.asarray(obj.get_bound_box())
    height = float(bb[:, 2].max() - bb[:, 2].min())
    depth = float(rng.uniform(0.0, max_frac)) * height
    loc = np.asarray(obj.get_location())
    obj.set_location(loc - np.array([0.0, 0.0, depth]))
    return depth


def add_canopy(focus, cfg, rng):
    """A camera-invisible plane with holes, casting dappled shadows over the focus.

    Litter on a pavement is very often half in the shadow of a tree, a fence or a
    building that is not in the picture. Without it, every scene is either fully lit
    or fully overcast, and a detector learns that hard shadow edges never cross
    objects.
    """
    size = float(rng.uniform(*cfg["size_m"]))
    height = float(rng.uniform(*cfg["height_m"]))
    bpy.ops.mesh.primitive_plane_add(size=size, location=(focus[0] + rng.uniform(-2, 2),
                                                          focus[1] + rng.uniform(-2, 2),
                                                          focus[2] + height))
    plane = bpy.context.active_object
    plane.name = "sd_canopy"
    plane.rotation_euler = (float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3)),
                            float(rng.uniform(0, np.pi)))
    plane.visible_camera = False
    # Out of the viewport depsgraph too, so the cameras' ray checks see through it
    # just like the rendered camera does.
    plane.hide_viewport = True
    plane.visible_glossy = False
    plane.visible_transmission = False
    plane["category_id"] = 0

    mat = bpy.data.materials.new("sd_canopy")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.08, 0.03, 1.0)
    noise = nodes.new("ShaderNodeTexVoronoi" if rng.random() < 0.5 else "ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(rng.uniform(1.5, 6.0))
    ramp = nodes.new("ShaderNodeValToRGB")
    cover = float(rng.uniform(*cfg["coverage"]))
    ramp.color_ramp.elements[0].position = max(cover - 0.05, 0.0)
    ramp.color_ramp.elements[1].position = min(cover + 0.05, 1.0)
    links.new(noise.outputs[0], ramp.inputs["Fac"])
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    out = nodes.get("Material Output")
    links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    links.new(transparent.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    plane.data.materials.append(mat)
    return {"size_m": size, "height_m": height, "coverage": cover}
