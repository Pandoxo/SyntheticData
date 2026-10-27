"""Renderer setup and everything written per image.

Per scene directory:

    images/000000.png        RGB (written by BlenderProc's COCO writer)
    depth/000000.png         uint16, depth along the optical axis in millimetres.
                             0 = no valid depth (sky, or further than 65.535 m).
    instance/000000.png      uint16, 0 = not trash, k = the k-th trash instance of
                             the scene (same k across all frames of the scene)
    semantic/000000.png      uint8, the class id of every pixel (0 = background)
    coco_annotations.json    COCO instances with RLE masks
    scene_meta.json          every random choice that produced the scene

The depth and masks are rendered, not estimated: they are exact, which is the point
of synthetic data. Sensor-like depth noise is a training-time augmentation.
"""

import os

import blenderproc as bproc
import bpy
import cv2
import numpy as np

DEPTH_SCALE = 1000.0          # metres -> millimetres
DEPTH_MAX_MM = 65535


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
    # Light paths: grass and leaves are alpha-mapped cards; with Blender's default
    # transparency bounces deep layers of them render black.
    bpy.context.scene.cycles.transparent_max_bounces = 16
    if cfg.get("depth_output", True):
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
    # "name" puts each object's name into the attribute maps, which is how the dense
    # instance masks keep one stable id per trash object across frames.
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "class", "name"], default_values={"category_id": 0})


def set_render_seed(seed):
    """Make Cycles' sampling pattern reproducible along with everything else."""
    bpy.context.scene.cycles.seed = int(seed) % (2 ** 31)


# ----------------------------------------------------------------- annotations

def _visible_pixel_counts(instance_segmap):
    flat = np.asarray(instance_segmap).ravel()
    if flat.size == 0:
        return {}
    counts = np.bincount(flat.astype(np.int64))
    return {int(idx): int(n) for idx, n in enumerate(counts) if n > 0}


def filter_by_visibility(data, frames_meta, cfg):
    """Drop annotations for objects that are barely visible, and optionally whole frames.

    The instance segmap already accounts for occlusion, so a target hidden behind a
    bin or deep in the grass simply has very few pixels. Anything under the threshold
    is demoted to category_id 0, which the COCO writer skips - that removes the sliver
    boxes that otherwise teach a detector to hallucinate.

    Frames the camera sampler deliberately aimed away from all trash (hard negatives)
    are kept even though they carry no annotation.
    """
    min_pixels = int(cfg["min_visible_pixels"])
    drop_empty = bool(cfg["drop_frames_without_targets"])

    keep = {k: [] for k in ("colors", "instance_segmaps", "instance_attribute_maps",
                            "class_segmaps", "depth")}
    kept_indices = []
    stats = {"frames_in": len(data["colors"]), "frames_dropped": 0,
             "annotations_kept": 0, "annotations_dropped": 0, "empty_frames_kept": 0}

    for index in range(len(data["colors"])):
        segmap = data["instance_segmaps"][index]
        counts = _visible_pixel_counts(segmap)
        filtered_attrs, kept_here = [], 0
        for entry in data["instance_attribute_maps"][index]:
            entry = dict(entry)
            if int(entry.get("category_id", 0)) != 0:
                if counts.get(int(entry["idx"]), 0) < min_pixels:
                    entry["category_id"] = 0
                    stats["annotations_dropped"] += 1
                else:
                    kept_here += 1
                    stats["annotations_kept"] += 1
            filtered_attrs.append(entry)

        intended_empty = bool(frames_meta[index].get("intended_empty")) if index < len(frames_meta) else False
        if kept_here == 0 and drop_empty and not intended_empty:
            stats["frames_dropped"] += 1
            continue
        if kept_here == 0:
            stats["empty_frames_kept"] += 1

        keep["colors"].append(data["colors"][index])
        keep["instance_segmaps"].append(segmap)
        keep["instance_attribute_maps"].append(filtered_attrs)
        keep["class_segmaps"].append(data["class_segmaps"][index] if "class_segmaps" in data else None)
        keep["depth"].append(data["depth"][index] if "depth" in data else None)
        kept_indices.append(index)

    if stats["frames_in"] and not stats["annotations_kept"] and not stats["annotations_dropped"]:
        print("[annotations] WARNING nothing in these frames carried a non-zero category_id. "
              "Either no trash was in view, or the objects lost their `category_id` custom "
              "property before rendering - check that assets.load ran on them.")

    stats["frames_out"] = len(keep["colors"])
    keep["kept_indices"] = kept_indices
    return keep, stats


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


def write_dense_maps(output_dir, filtered, trash_names):
    """Depth, instance and semantic PNGs, one of each per kept frame.

    Instance ids are stable across the frames of a scene: trash object `trash_names[k]`
    is id k+1 everywhere. Returns the id -> (object name, category) table that
    scene_meta.json records, so a pixel can always be traced back to its model.
    """
    for sub in ("depth", "instance", "semantic"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    stable_id = {name: k + 1 for k, name in enumerate(trash_names)}
    table = {}

    for i, attrs in enumerate(filtered["instance_attribute_maps"]):
        segmap = np.asarray(filtered["instance_segmaps"][i])
        instance = np.zeros(segmap.shape[:2], dtype=np.uint16)
        semantic = np.zeros(segmap.shape[:2], dtype=np.uint8)
        for entry in attrs:
            category = int(entry.get("category_id", 0))
            name = entry.get("name")
            if category == 0 or name not in stable_id:
                continue
            mask = segmap == int(entry["idx"])
            instance[mask] = stable_id[name]
            semantic[mask] = category
            table[stable_id[name]] = {"object": name, "category_id": category}

        name = f"{i:06d}.png"
        cv2.imwrite(os.path.join(output_dir, "instance", name), instance)
        cv2.imwrite(os.path.join(output_dir, "semantic", name), semantic)

        depth = filtered["depth"][i]
        if depth is not None:
            d = np.asarray(depth, dtype=np.float64) * DEPTH_SCALE
            d[~np.isfinite(d)] = 0
            d[(d <= 0) | (d > DEPTH_MAX_MM)] = 0
            cv2.imwrite(os.path.join(output_dir, "depth", name), d.astype(np.uint16))
    return {str(k): v for k, v in sorted(table.items())}


def write_hdf5(output_dir, data):
    bproc.writer.write_hdf5(os.path.join(output_dir, "hdf5"), data,
                            append_to_existing_output=True)
