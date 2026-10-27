"""Camera rigs: where the camera is and what lens it has, per rendered frame.

Three rigs, mixed per frame by `camera.rig_weights`:

robot    the robot's own front camera. A few centimetres above the ground it stands
         on, looking roughly level, wide lens. The trash in view can be anywhere from
         25 cm to 4 m away and anywhere across the frame - the robot is driving, not
         composing a photograph.
arm      the camera on the robotic arm, looking down at an item from 15-60 cm,
         as it would just before a grasp.
general  a person-height "photographer" orbiting a subject. Not what the robot
         ever sees, but extra viewpoint diversity helps a detector generalise.

Every rig returns poses through the same `add_frame` path, so the per-frame lens and
the metadata written next to the images always agree.
"""

import math

import blenderproc as bproc
import bpy
import numpy as np

SENSOR_WIDTH_MM = 36.0


# --------------------------------------------------------------------- helpers

def _bbox_of(objects):
    corners = np.concatenate([np.asarray(o.get_bound_box()) for o in objects], axis=0)
    return corners.min(axis=0), corners.max(axis=0)


def _hfov_to_lens_mm(hfov_rad):
    return (SENSOR_WIDTH_MM / 2.0) / math.tan(hfov_rad / 2.0)


def _set_lens(lens_mm, clip_start):
    cam = bpy.context.scene.camera
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = SENSOR_WIDTH_MM
    cam.data.lens = lens_mm
    cam.data.clip_start = clip_start
    cam.data.clip_end = 1000.0


def _keyframe_lens(frame, lens):
    """Persist the focal length for this frame.

    `bproc.camera.add_camera_pose` only keyframes location and rotation, and Blender
    renders the whole frame range in one `render(animation=True)` pass. Setting
    `camera.data.lens` alone would therefore apply the *last* sampled focal length to
    every frame while the metadata claimed otherwise.
    """
    cam_data = bpy.context.scene.camera.data
    cam_data.lens = lens
    cam_data.keyframe_insert(data_path="lens", frame=frame)
    action = cam_data.animation_data.action if cam_data.animation_data else None
    if action is not None:
        for fcurve in action.fcurves:
            if fcurve.data_path == "lens":
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "CONSTANT"


def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _look(location, forward, roll_rad):
    rotation = bproc.camera.rotation_from_forward_vec(forward, inplane_rot=roll_rad)
    return bproc.math.build_transformation_mat(location, rotation)


# ------------------------------------------------------------------------ rigs

def sample_robot(targets, env, cfg, rng, want_target=True):
    """Robot front camera. Returns (cam2world, info) or None."""
    hfov = math.radians(float(rng.uniform(*cfg["hfov_deg"])))
    target = targets[int(rng.integers(0, len(targets)))]
    low, high = _bbox_of([target])
    centre = (low + high) / 2.0

    distance = _loguniform(rng, *cfg["target_distance_m"])
    bearing = float(rng.uniform(0, 2 * np.pi))
    ground_xy = centre[:2] + distance * np.array([np.cos(bearing), np.sin(bearing)])
    ground_z = env.ground_height(ground_xy[0], ground_xy[1])
    if ground_z is None:
        return None
    # The robot stands on the ground the trash lies on, give or take a curb; a pose
    # that would have it on a roof or a bench is not a robot pose.
    if abs(ground_z - low[2]) > float(cfg.get("max_ground_step_m", 0.35)):
        return None

    height = float(rng.uniform(*cfg["height_m"]))
    location = np.array([ground_xy[0], ground_xy[1], ground_z + height])

    # Face the target, then turn away by up to most of half the field of view, so
    # the item lands anywhere across the frame instead of always dead centre. For a
    # deliberately empty frame, turn far enough that it is out of view entirely.
    yaw_to_target = math.atan2(centre[1] - location[1], centre[0] - location[0])
    half = hfov / 2.0
    if want_target:
        yaw = yaw_to_target + float(rng.uniform(-1, 1)) * half * float(cfg["yaw_offset_frac"])
    else:
        yaw = yaw_to_target + float(rng.choice([-1, 1])) * float(rng.uniform(half * 1.3, np.pi))
    pitch = math.radians(float(rng.uniform(*cfg["pitch_deg"])))
    roll = math.radians(float(rng.uniform(*cfg["roll_deg"])))
    forward = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
                        math.sin(pitch)])
    info = {"rig": "robot", "hfov_deg": math.degrees(hfov), "height_m": height,
            "pitch_deg": math.degrees(pitch), "roll_deg": math.degrees(roll),
            "target_distance_m": distance,
            "target": target.get_name() if want_target else None}
    return _look(location, forward, roll), hfov, info


def sample_arm(targets, env, cfg, rng, want_target=True):
    """Gripper camera above one item, looking down."""
    hfov = math.radians(float(rng.uniform(*cfg["hfov_deg"])))
    target = targets[int(rng.integers(0, len(targets)))]
    low, high = _bbox_of([target])
    centre = (low + high) / 2.0
    ground_z = env.ground_height(centre[0], centre[1])
    ground_z = low[2] if ground_z is None else ground_z

    height = float(rng.uniform(*cfg["height_m"]))
    elevation = math.radians(float(rng.uniform(*cfg["elevation_deg"])))
    azimuth = float(rng.uniform(0, 2 * np.pi))
    aim = centre + np.append(rng.uniform(-1, 1, 2) * float(cfg["aim_jitter_m"]), 0.0)
    if not want_target:
        # Same height and tilt, aimed at bare ground a little way off.
        offset = float(rng.uniform(0.6, 1.5))
        aim = aim + offset * np.array([np.cos(azimuth + np.pi), np.sin(azimuth + np.pi), 0.0])
    horiz = (ground_z + height - aim[2]) / max(math.tan(elevation), 1e-3)
    location = np.array([aim[0] + horiz * math.cos(azimuth), aim[1] + horiz * math.sin(azimuth),
                         ground_z + height])
    roll = math.radians(float(rng.uniform(*cfg["roll_deg"])))
    info = {"rig": "arm", "hfov_deg": math.degrees(hfov), "height_m": height,
            "elevation_deg": math.degrees(elevation),
            "target": target.get_name() if want_target else None}
    return _look(location, aim - location, roll), hfov, info


def _framing_distance(extent, fraction, hfov):
    return (extent / max(fraction, 1e-3)) / (2.0 * np.tan(hfov / 2.0))


def sample_general(targets, env, cfg, rng, want_target=True):
    """Person-height photographer orbiting one item or a cluster."""
    hfov = 2.0 * math.atan((SENSOR_WIDTH_MM / 2.0) / float(rng.uniform(*cfg["focal_length_range_mm"])))
    anchor = targets[int(rng.integers(0, len(targets)))]
    subject = [anchor]
    if len(targets) > 1 and rng.random() >= cfg["close_up_probability"]:
        a_low, a_high = _bbox_of([anchor])
        a_centre = (a_low + a_high) / 2.0
        for target in targets:
            t_low, t_high = _bbox_of([target])
            if target is not anchor and np.linalg.norm((t_low + t_high) / 2.0 - a_centre) <= cfg["cluster_radius_m"]:
                subject.append(target)

    low, high = _bbox_of(subject)
    poi = (low + high) / 2.0
    extent = float(np.max(high - low))
    distance = _framing_distance(extent, float(rng.uniform(*cfg["framing_fraction_range"])), hfov)
    smallest = min(float(np.max(np.ptp(np.asarray(o.get_bound_box()), axis=0))) for o in subject)
    distance = min(distance, _framing_distance(smallest, cfg["min_object_frame_fraction"], hfov))
    distance = float(np.clip(distance, *cfg["distance_limits_m"]))

    azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    elevation = math.radians(float(rng.uniform(*cfg["elevation_range_deg"])))
    location = poi + distance * np.array([math.cos(elevation) * math.cos(azimuth),
                                          math.cos(elevation) * math.sin(azimuth),
                                          math.sin(elevation)])
    jitter = cfg["aim_jitter_frac"] * extent
    aim = poi + rng.uniform(-jitter, jitter, size=3)
    if not want_target:
        aim = aim + np.array([-math.cos(azimuth), -math.sin(azimuth), 0.0]) * distance
    roll = math.radians(float(rng.uniform(*cfg["inplane_rotation_range_deg"])))
    info = {"rig": "general", "hfov_deg": math.degrees(hfov), "camera_distance_m": distance,
            "elevation_deg": math.degrees(elevation), "close_up": len(subject) == 1,
            "target": anchor.get_name() if want_target else None}
    if location[2] - poi[2] < cfg.get("min_height_above_poi", 0.0):
        return None
    return _look(location, aim - location, roll), hfov, info


RIGS = {"robot": sample_robot, "arm": sample_arm, "general": sample_general}


# ----------------------------------------------------------------- validation

def _ray_grid(cam2world, sqrt_rays):
    """Cast a grid of rays through the image; returns [(hit object name, distance)].

    Blender's own scene ray cast, on the evaluated scene: no copy of the geometry is
    made. BlenderProc's BVH helper copies every triangle into Python lists first,
    which cost 4-7 GB once a scene had a lawn of weeds and a few scanned rocks.
    """
    from mathutils import Matrix  # pylint: disable=import-outside-toplevel
    m = Matrix(np.asarray(cam2world).tolist())
    cam = bpy.context.scene.camera.data
    frame = [m @ v for v in cam.view_frame(scene=bpy.context.scene)]
    vec_x, vec_y = frame[1] - frame[0], frame[3] - frame[0]
    origin = m.to_translation()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hits = []
    for i in range(sqrt_rays):
        for j in range(sqrt_rays):
            end = frame[0] + vec_x * i / float(sqrt_rays - 1) + vec_y * j / float(sqrt_rays - 1)
            ok, loc, _, _, obj, _ = bpy.context.scene.ray_cast(depsgraph, origin, end - origin)
            hits.append((obj.name if ok and obj is not None else None,
                         (loc - origin).length if ok else None))
    return hits


def _pose_is_valid(cam2world, targets, cfg, want_target):
    hits = _ray_grid(cam2world, int(cfg["check_rays"]))

    min_distance = cfg.get("min_obstacle_distance")
    if min_distance:
        # Rejects cameras inside a prop, a wall or the ground. Rays that hit nothing
        # (sky) are fine; only a hit closer than min_distance fails.
        if any(d is not None and d < float(min_distance) for _, d in hits):
            return False, "obstacle_too_close"

    names = {t.get_name() for t in targets}
    n_visible = len({n for n, _ in hits if n in names})
    if want_target and n_visible < cfg["min_visible_targets"]:
        return False, "targets_occluded"
    if not want_target and n_visible > 0:
        return False, "trash_in_empty_frame"
    return True, None


def sample_views(targets, env, cfg, rng, n_views):
    """Add up to `n_views` validated camera poses, each with its own rig and lens.

    Poses that fail validation are discarded rather than kept as a "best effort"
    fallback - a rejected pose means the objects were hidden or the camera was inside
    geometry, and rendering it anyway just adds mislabelled images to the dataset.
    """
    names = sorted(cfg["rig_weights"])
    probs = np.array([float(cfg["rig_weights"][n]) for n in names])
    probs = probs / probs.sum()
    frames, rejections = [], {}
    clip_start = float(cfg.get("clip_start_m", 0.005))

    for _ in range(n_views):
        rig = str(rng.choice(names, p=probs))
        rig_cfg = {**cfg, **cfg.get(rig, {})}
        # A few frames per dataset show no trash at all: hard negatives, so the model
        # learns that a stone, a leaf or a crack in the asphalt is not litter.
        want_target = bool(rng.random() >= float(cfg.get("empty_frame_probability", 0.0)))
        for _attempt in range(rig_cfg["max_pose_attempts"]):
            sampled = RIGS[rig](targets, env, rig_cfg, rng, want_target)
            if sampled is None:
                rejections["no_ground"] = rejections.get("no_ground", 0) + 1
                continue
            cam2world, hfov, info = sampled
            lens = _hfov_to_lens_mm(hfov)
            # The lens must be set before validating: the ray grid follows the FOV.
            _set_lens(lens, clip_start)

            ok, reason = _pose_is_valid(cam2world, targets, rig_cfg, want_target)
            if not ok:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            frame = bproc.camera.add_camera_pose(cam2world)
            _keyframe_lens(frame, lens)
            frames.append({
                "frame": frame,
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()},
                "focal_length_mm": round(lens, 3),
                "intended_empty": not want_target,
                "cam2world": np.asarray(cam2world).tolist(),
                "K": np.asarray(bproc.camera.get_intrinsics_as_K_matrix()).tolist(),
            })
            break

    if rejections:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(rejections.items()))
        print(f"[camera] kept {len(frames)}/{n_views} views (rejected: {summary})")
    return frames


def robot_ground_points(frames):
    """Ground positions of every robot/arm camera - grass gets cleared under them."""
    points = []
    for frame in frames:
        if frame["rig"] in ("robot",):
            m = np.asarray(frame["cam2world"])
            points.append(m[:3, 3])
    return points
