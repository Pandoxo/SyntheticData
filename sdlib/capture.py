"""Renderer setup, camera pose sampling and annotation writing."""

import os

import blenderproc as bproc
import bpy
import numpy as np


def configure_renderer(cfg):
    bproc.camera.set_resolution(*cfg["resolution"])
    bproc.renderer.set_max_amount_of_samples(cfg["render_samples"])
    if cfg.get("noise_threshold") is not None:
        bproc.renderer.set_noise_threshold(float(cfg["noise_threshold"]))
    # Cycles at a low sample count is visibly noisy; denoising is what makes 64-128
    # samples usable. "OPTIX" needs an NVIDIA GPU, "INTEL" (OpenImageDenoise) is the
    # portable choice, null disables it.
    bproc.renderer.set_denoiser(cfg.get("denoiser"))
    bproc.renderer.set_output_format(file_format="PNG")

    if cfg.get("depth_output"):
        bproc.renderer.enable_depth_output(activate_antialiasing=False)


def enable_annotations(_cfg):
    """Turn on instance + class segmentation.

    Must be called *after* every object exists in the scene. `enable_segmentation_output`
    stamps a `pass_index` onto each mesh object at call time, and anything created later
    keeps pass_index 0 - which the mapper reads as the world background. Calling this
    during start-up therefore yields a dataset in which every object is background, with
    no error anywhere.
    """
    # The default covers every mesh we never labelled (scene geometry, the ground, the
    # invisible colliders), so it lands in category 0 and the COCO writer skips it.
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "class"], default_values={"category_id": 0})


def set_render_seed(seed):
    """Make Cycles' sampling pattern reproducible along with everything else."""
    bpy.context.scene.cycles.seed = int(seed) % (2 ** 31)


# ------------------------------------------------------------------- camera poses

def _bbox_of(objects):
    corners = np.concatenate([np.asarray(o.get_bound_box()) for o in objects], axis=0)
    return corners.min(axis=0), corners.max(axis=0)


def _framing_distance(extent, fraction):
    """How far back the camera has to be for `extent` metres to fill `fraction` of the frame.

    Distance is derived from the subject size and the current focal length rather than
    drawn from a fixed metre range. A fixed range cannot work here: a 12 cm can shot
    from 4 m is a handful of pixels, and the same range on a 20 cm bottle cluster
    behaves completely differently. Deriving it means a long lens automatically backs
    off and a wide lens moves in, which is what a photographer would do.
    """
    fov = min(bproc.camera.get_fov())
    return (extent / max(fraction, 1e-3)) / (2.0 * np.tan(fov / 2.0))


def _choose_subject(targets, cfg, rng):
    """Pick what this view is a photograph *of*.

    Never the full target list: with several spawn zones metres apart, framing all of
    them aims the camera at the empty ground between them and pushes every object to
    the edge of the frame. Instead pick one item and take its neighbours, which is
    both spatially coherent and what a pile of litter actually looks like.
    """
    anchor = targets[int(rng.integers(0, len(targets)))]
    if len(targets) == 1 or rng.random() < cfg["close_up_probability"]:
        return [anchor]

    low, high = _bbox_of([anchor])
    centre = (low + high) / 2.0
    radius = float(cfg["cluster_radius_m"])

    cluster = []
    for target in targets:
        t_low, t_high = _bbox_of([target])
        if np.linalg.norm((t_low + t_high) / 2.0 - centre) <= radius:
            cluster.append(target)
    return cluster or [anchor]


def _sample_pose(subject, cfg, rng):
    """Place the camera on a sphere around `subject`, framed to fill the viewport."""
    low, high = _bbox_of(subject)
    poi = (low + high) / 2.0
    extent = float(np.max(high - low))

    fraction = float(rng.uniform(*cfg["framing_fraction_range"]))
    distance = _framing_distance(extent, fraction)

    # Framing a widely scattered pile would otherwise push the camera so far back that
    # every individual item is a handful of pixels. Cap the distance so the *smallest*
    # item in the subject still spans a usable share of the frame.
    smallest = cfg.get("min_object_frame_fraction")
    if smallest:
        per_object = min(float(np.max(np.ptp(np.asarray(o.get_bound_box()), axis=0)))
                         for o in subject)
        distance = min(distance, _framing_distance(per_object, smallest))

    distance = float(np.clip(distance, *cfg["distance_limits_m"]))

    azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    elevation = np.radians(float(rng.uniform(*cfg["elevation_range_deg"])))
    offset = distance * np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    location = poi + offset

    # Aim slightly off-centre so the subject is not always dead centre of the frame.
    jitter = cfg["aim_jitter_frac"] * extent
    aim = poi + rng.uniform(-jitter, jitter, size=3)

    rotation = bproc.camera.rotation_from_forward_vec(
        aim - location,
        inplane_rot=np.radians(float(rng.uniform(*cfg["inplane_rotation_range_deg"]))),
    )
    return bproc.math.build_transformation_mat(location, rotation), distance, location, poi


def _keyframe_lens(frame, lens):
    """Persist the focal length for this frame.

    `bproc.camera.add_camera_pose` only keyframes location and rotation, and Blender
    renders the whole frame range in one `render(animation=True)` pass. Setting
    `camera.data.lens` alone would therefore apply the *last* sampled focal length to
    every frame while the metadata claimed otherwise. Keyframing the lens is what
    actually makes focal length vary per image.
    """
    cam_data = bpy.context.scene.camera.data
    cam_data.lens = lens
    cam_data.keyframe_insert(data_path="lens", frame=frame)

    # Constant interpolation, so a frame can never render a value halfway between two
    # sampled focal lengths.
    action = cam_data.animation_data.action if cam_data.animation_data else None
    if action is not None:
        for fcurve in action.fcurves:
            if fcurve.data_path == "lens":
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "CONSTANT"


def _pose_is_valid(cam2world, targets, bvh_tree, cfg):
    """Reject camera poses that a human photographer never would."""
    min_distance = cfg.get("min_obstacle_distance")
    if min_distance and bvh_tree is not None:
        # Rejects cameras that clipped into a wall, a parked car or the ground. Rays
        # that hit nothing (sky) are fine; only a hit closer than min_distance fails.
        if not bproc.camera.perform_obstacle_in_view_check(
                cam2world, {"min": float(min_distance)}, bvh_tree,
                sqrt_number_of_rays=cfg["check_rays"]):
            return False, "obstacle_too_close"

    # A ray grid through the frame; the subject is already framed to fill a good part
    # of it, so a target that no ray hits is genuinely hidden or out of view.
    # `scene_coverage_score` is deliberately not used - it scores indoor-scene class
    # diversity for SUNCG/3D-Front and says nothing useful about litter on a street.
    visible = bproc.camera.visible_objects(cam2world, sqrt_number_of_rays=cfg["check_rays"])
    if len(visible.intersection(set(targets))) < cfg["min_visible_targets"]:
        return False, "targets_occluded"

    return True, None


def sample_views(targets, bvh_tree, cfg, rng, n_views):
    """Add up to `n_views` validated camera poses, each with its own focal length.

    Poses that fail validation are discarded rather than kept as a "best effort"
    fallback - a rejected pose means the objects were hidden or the camera was inside
    geometry, and rendering it anyway just adds mislabelled images to the dataset.
    """
    frames, rejections = [], {}

    for _ in range(n_views):
        for _attempt in range(cfg["max_pose_attempts"]):
            subject = _choose_subject(targets, cfg, rng)

            lens = float(rng.uniform(*cfg["focal_length_range_mm"]))
            # Set the lens before anything else: the framing distance, the obstacle
            # check and the visibility check all depend on the field of view.
            bproc.camera.set_intrinsics_from_blender_params(lens=lens, lens_unit="MILLIMETERS")

            cam2world, distance, location, poi = _sample_pose(subject, cfg, rng)
            if cfg.get("min_height_above_poi") is not None and \
                    location[2] - poi[2] < cfg["min_height_above_poi"]:
                rejections["too_low"] = rejections.get("too_low", 0) + 1
                continue

            ok, reason = _pose_is_valid(cam2world, targets, bvh_tree, cfg)
            if not ok:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            frame = bproc.camera.add_camera_pose(cam2world)
            _keyframe_lens(frame, lens)
            frames.append({
                "frame": frame,
                "focal_length_mm": lens,
                "camera_distance_m": distance,
                "close_up": len(subject) == 1,
                "cam2world": np.asarray(cam2world).tolist(),
                "K": np.asarray(bproc.camera.get_intrinsics_as_K_matrix()).tolist(),
            })
            break

    if rejections:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(rejections.items()))
        print(f"[camera] kept {len(frames)}/{n_views} views (rejected: {summary})")
    return frames


# --------------------------------------------------------------------- annotations

def _visible_pixel_counts(instance_segmap):
    flat = np.asarray(instance_segmap).ravel()
    if flat.size == 0:
        return {}
    counts = np.bincount(flat.astype(np.int64))
    return {int(idx): int(n) for idx, n in enumerate(counts) if n > 0}


def filter_by_visibility(data, cfg):
    """Drop annotations for objects that are barely visible, and optionally whole frames.

    The instance segmap already accounts for occlusion, so a target hidden behind a
    parked car simply has very few pixels. Anything under the threshold is demoted to
    category_id 0, which the COCO writer skips - that removes the sliver boxes that
    otherwise teach a detector to hallucinate.
    """
    min_pixels = int(cfg["min_visible_pixels"])
    drop_empty = bool(cfg["drop_frames_without_targets"])

    colors, segmaps, attribute_maps, kept_indices = [], [], [], []
    stats = {"frames_in": len(data["colors"]), "frames_dropped": 0,
             "annotations_kept": 0, "annotations_dropped": 0}

    for index, (color, segmap, attrs) in enumerate(zip(data["colors"],
                                                       data["instance_segmaps"],
                                                       data["instance_attribute_maps"])):
        counts = _visible_pixel_counts(segmap)
        filtered_attrs, kept_here = [], 0

        for entry in attrs:
            entry = dict(entry)
            if int(entry.get("category_id", 0)) != 0:
                if counts.get(int(entry["idx"]), 0) < min_pixels:
                    entry["category_id"] = 0
                    stats["annotations_dropped"] += 1
                else:
                    kept_here += 1
                    stats["annotations_kept"] += 1
            filtered_attrs.append(entry)

        if kept_here == 0 and drop_empty:
            stats["frames_dropped"] += 1
            continue

        colors.append(color)
        segmaps.append(segmap)
        attribute_maps.append(filtered_attrs)
        kept_indices.append(index)

    if stats["frames_in"] and not stats["annotations_kept"] and not stats["annotations_dropped"]:
        print("[annotations] WARNING nothing in these frames carried a non-zero category_id. "
              "Either no trash was in view, or the objects lost their `category_id` custom "
              "property before rendering - check that assets.load ran on them.")

    stats["frames_out"] = len(colors)
    return {"colors": colors, "instance_segmaps": segmaps,
            "instance_attribute_maps": attribute_maps,
            "kept_indices": kept_indices}, stats


def write_coco(output_dir, filtered, label_mapping, cfg):
    """Write COCO annotations for the frames that survived filtering.

    The writer walks `scene.frame_start .. scene.frame_end` and indexes the lists it
    is given by that range, so the range has to be narrowed to match after frames
    have been filtered out.
    """
    if not filtered["colors"]:
        return 0

    scene = bpy.context.scene
    saved = (scene.frame_start, scene.frame_end)
    scene.frame_start = 0
    scene.frame_end = len(filtered["colors"])
    try:
        bproc.writer.write_coco_annotations(
            output_dir,
            instance_segmaps=filtered["instance_segmaps"],
            instance_attribute_maps=filtered["instance_attribute_maps"],
            colors=filtered["colors"],
            color_file_format=cfg.get("color_file_format", "PNG"),
            mask_encoding_format=cfg.get("mask_encoding_format", "rle"),
            label_mapping=label_mapping,
            append_to_existing_output=True,
        )
    finally:
        scene.frame_start, scene.frame_end = saved
    return len(filtered["colors"])


def write_hdf5(output_dir, data):
    bproc.writer.write_hdf5(os.path.join(output_dir, "hdf5"), data,
                            append_to_existing_output=True)
