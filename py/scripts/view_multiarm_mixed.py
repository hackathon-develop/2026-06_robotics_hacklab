#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""View a multi-arm scene with shared overhead-camera perception.

Example:
    python py/scripts/view_multiarm_mixed.py --overhead-camera 0 --track-cube --track-drop-zone
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from pick_and_place.arm_config import ArmConfig, DEFAULT_TWO_ARM_CONFIGS, names_for_arm
from pick_and_place.arm_workspace import arm_can_pick
from pick_and_place.camera_compare import RealSource, draw_hud, draw_tag_detections, load_intrinsics
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.cam_align_solve import parse_index_or_path
from pick_and_place.cube_detection import CUBE_TAG_IDS, CubePose, CubeTracker, detect_tags
from pick_and_place.kinematics import So101Kinematics, derive_kinematics
from pick_and_place.multiarm_scene import build_multiarm_scene
from pick_and_place.paper_detection import (
    PaperTarget,
    PaperTracker,
    add_paper_target_marker,
    detect_paper_target,
    draw_paper_target,
    set_paper_target_marker,
)
from pick_and_place.workspace_overlays import (
    CUBE_PLACEMENT_OVERLAY,
    WORKSPACE_OVERLAYS,
    workspace_interior_corners_world,
)

WINDOW_TITLE = "view_multiarm_mixed  (m mode  , . alpha  q quit)"


class OpenCvCuda:
    """Small optional CUDA wrapper for OpenCV operations that support it."""

    def __init__(self, enabled: bool):
        self.enabled = False
        self._warned_remap = False
        self._warned_edges = False
        self._map_cache: dict[int, tuple[Any, Any]] = {}
        if not enabled:
            return
        count = 0
        cuda = getattr(cv2, "cuda", None)
        if cuda is not None:
            try:
                count = cuda.getCudaEnabledDeviceCount()
            except cv2.error:
                count = 0
        if count <= 0:
            print(
                "Warning: --opencv-cuda requested, but this OpenCV build has no CUDA "
                "device; using CPU."
            )
            return
        self.enabled = True
        print(f"OpenCV CUDA enabled ({count} device{'s' if count != 1 else ''}).")

    def remap(self, image: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        if not self.enabled:
            return cv2.remap(image, *maps, cv2.INTER_LINEAR)
        try:
            map1, map2 = self._gpu_maps(maps)
            cuda = getattr(cv2, "cuda")
            gpu_image = getattr(cv2, "cuda_GpuMat")()
            gpu_image.upload(image)
            return cuda.remap(gpu_image, map1, map2, cv2.INTER_LINEAR).download()
        except (AttributeError, cv2.error) as exc:
            if not self._warned_remap:
                print(f"Warning: OpenCV CUDA remap failed; using CPU remap. {exc}")
                self._warned_remap = True
            return cv2.remap(image, *maps, cv2.INTER_LINEAR)

    def edge_overlay(self, sim_rgb: np.ndarray, frame_rgb: np.ndarray) -> np.ndarray:
        if self.enabled:
            try:
                cuda = getattr(cv2, "cuda")
                gpu_sim = getattr(cv2, "cuda_GpuMat")()
                gpu_sim.upload(sim_rgb)
                gpu_gray = cuda.cvtColor(gpu_sim, cv2.COLOR_RGB2GRAY)
                detector = cuda.createCannyEdgeDetector(60, 160)
                edges = detector.detect(gpu_gray).download()
                out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out[edges > 0] = (0, 255, 0)
                return out
            except (AttributeError, cv2.error) as exc:
                if not self._warned_edges:
                    print(f"Warning: OpenCV CUDA Canny failed; using CPU Canny. {exc}")
                    self._warned_edges = True
        gray = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out[edges > 0] = (0, 255, 0)
        return out

    def _gpu_maps(self, maps: tuple[np.ndarray, np.ndarray]) -> tuple[Any, Any]:
        key = id(maps[0])
        cached = self._map_cache.get(key)
        if cached is not None:
            return cached
        xmap, ymap = cv2.convertMaps(maps[0], maps[1], cv2.CV_32FC1)
        gpu_mat = getattr(cv2, "cuda_GpuMat")
        gpu_x = gpu_mat()
        gpu_y = gpu_mat()
        gpu_x.upload(xmap)
        gpu_y.upload(ymap)
        cached = (gpu_x, gpu_y)
        self._map_cache[key] = cached
        return cached


@dataclass(frozen=True)
class ArmRuntime:
    config: ArmConfig
    kinematics: So101Kinematics


@dataclass(frozen=True)
class ReachDebug:
    distance: float | None
    inner_radius: float
    outer_radius: float
    radius_ok: bool
    pan_ok: bool
    ik_ok: bool | None = None

    @property
    def rough_ok(self) -> bool:
        return self.radius_ok and self.pan_ok


def _yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) / 2.0
    return math.cos(half), 0.0, 0.0, math.sin(half)


def _parse_arm(value: str) -> ArmConfig:
    """Parse ``name,x,y,yaw_deg`` or ``name:x:y:yaw_deg``."""
    parts = value.replace(":", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("arm must be name,x,y,yaw_deg")
    name, x, y, yaw = parts
    return ArmConfig(
        name=name,
        base_pos=(float(x), float(y), 0.0072),
        base_quat=_yaw_quat(float(yaw)),
    )


def _reach_debug(
    arm: ArmRuntime,
    xy: np.ndarray,
    *,
    inner_radius: float,
    outer_radius: float,
) -> ReachDebug:
    dx = float(xy[0] - arm.kinematics.pan_axis[0])
    dy = float(xy[1] - arm.kinematics.pan_axis[1])
    distance = float(math.hypot(dx, dy))
    radius_ok = inner_radius <= distance <= outer_radius
    shoulder_pan = -math.atan2(dy, dx)
    limit = arm.kinematics.joint_limits["shoulder_pan"]
    pan_ok = limit.min <= shoulder_pan <= limit.max
    return ReachDebug(
        distance=distance,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        radius_ok=radius_ok,
        pan_ok=pan_ok,
    )


def _pick_debug(arm: ArmRuntime, cube_pose: CubePose | None) -> ReachDebug:
    if cube_pose is None:
        overlay = WORKSPACE_OVERLAYS[-1]
        return ReachDebug(
            distance=None,
            inner_radius=overlay.inner_radius,
            outer_radius=overlay.outer_radius,
            radius_ok=False,
            pan_ok=False,
            ik_ok=None,
        )
    overlay = WORKSPACE_OVERLAYS[-1]
    rough = _reach_debug(
        arm,
        cube_pose.position[:2],
        inner_radius=overlay.inner_radius,
        outer_radius=overlay.outer_radius,
    )
    ik_ok = False
    if rough.rough_ok:
        ik_ok = arm_can_pick(arm.kinematics, cube_pose.position, cube_pose.rotation)
    return ReachDebug(
        distance=rough.distance,
        inner_radius=rough.inner_radius,
        outer_radius=rough.outer_radius,
        radius_ok=rough.radius_ok,
        pan_ok=rough.pan_ok,
        ik_ok=ik_ok,
    )


def _drop_debug(arm: ArmRuntime, target: PaperTarget | None) -> ReachDebug:
    if target is None:
        return ReachDebug(
            distance=None,
            inner_radius=CUBE_PLACEMENT_OVERLAY.inner_radius,
            outer_radius=CUBE_PLACEMENT_OVERLAY.outer_radius,
            radius_ok=False,
            pan_ok=False,
        )
    return _reach_debug(
        arm,
        target.center_world[:2],
        inner_radius=CUBE_PLACEMENT_OVERLAY.inner_radius,
        outer_radius=CUBE_PLACEMENT_OVERLAY.outer_radius,
    )


def _distance_text(distance: float | None) -> str:
    return "n/a" if distance is None else f"{distance:.3f}m"


def _pick_state(reach: ReachDebug, can_pick: bool) -> str:
    if can_pick:
        return "ok"
    if reach.distance is None:
        return "no-obj"
    if not reach.radius_ok:
        return "range"
    if not reach.pan_ok:
        return "pan"
    if reach.ik_ok is False:
        return "ik"
    return "no"


def _drop_state(reach: ReachDebug, can_drop: bool) -> str:
    if can_drop:
        return "ok"
    if reach.distance is None:
        return "no-target"
    if not reach.radius_ok:
        return "range"
    if not reach.pan_ok:
        return "pan"
    return "no"


def _choose_arm(
    arms: list[ArmRuntime],
    cube_pose: CubePose | None,
    target: PaperTarget | None,
) -> tuple[str | None, list[str], bool]:
    statuses = []
    candidates: list[tuple[float, str]] = []
    drop_reachable = False
    for arm in arms:
        score: float | None = None
        pick = _pick_debug(arm, cube_pose)
        drop = _drop_debug(arm, target)
        can_pick = pick.rough_ok and bool(pick.ik_ok)
        can_drop = drop.rough_ok

        if can_drop:
            drop_reachable = True

        if can_pick and can_drop and cube_pose is not None and target is not None:
            pan_xy = arm.kinematics.pan_axis[:2]
            score = float(
                np.linalg.norm(cube_pose.position[:2] - pan_xy)
                + np.linalg.norm(target.center_world[:2] - pan_xy)
            )
            candidates.append((score, arm.config.name))

        score_text = "n/a" if score is None else f"{score:.3f}m"
        pick_text = f"pick={_pick_state(pick, can_pick)} d={_distance_text(pick.distance)}"
        drop_text = f"drop={_drop_state(drop, can_drop)} d={_distance_text(drop.distance)}"
        statuses.append(
            f"{arm.config.name}: {pick_text}  {drop_text}  score={score_text}"
        )

    selected = min(candidates)[1] if candidates else None
    return selected, statuses, drop_reachable


def _format_xyz(label: str, value: np.ndarray | None) -> str:
    if value is None:
        return f"{label}: not seen"
    return f"{label}: x={value[0]:.3f} y={value[1]:.3f} z={value[2]:.3f}"


def _put_lines(bgr: np.ndarray, lines: list[str], *, y0: int = 32) -> None:
    for index, line in enumerate(lines):
        y = y0 + index * 24
        cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--overhead-camera", help="OpenCV camera index or device path")
    source.add_argument("--real-image", type=Path, help="captured real frame")
    parser.add_argument("--arm", action="append", type=_parse_arm, help="arm as name,x,y,yaw_deg")
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--track-cube", action="store_true")
    parser.add_argument("--track-drop-zone", action="store_true")
    parser.add_argument("--drop-zone-color", choices=["black", "white"], default="black")
    parser.add_argument("--cube-smooth", type=float, default=0.3)
    parser.add_argument("--drop-zone-smooth", type=float, default=0.3)
    parser.add_argument(
        "--opencv-cuda",
        action="store_true",
        help="use OpenCV CUDA for supported image operations, with CPU fallback",
    )
    args = parser.parse_args()

    arms = tuple(args.arm) if args.arm else DEFAULT_TWO_ARM_CONFIGS
    spec = build_multiarm_scene(arms)
    if args.track_drop_zone:
        add_paper_target_marker(spec)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.height)

    model = spec.compile()
    data = mujoco.MjData(model)
    applied = apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    if "overhead_camera" not in applied:
        print("Warning: no local extrinsics applied for 'overhead_camera'")

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
    if camera_id < 0:
        raise SystemExit("scene has no 'overhead_camera'")

    intrinsics = args.intrinsics
    if intrinsics is None:
        candidate = LOCAL_CAMERA_INTRINSICS_DIR / "overhead_camera.json"
        intrinsics = candidate if candidate.exists() else None

    undistort_map = None
    if intrinsics is not None:
        rect_matrix, undistort_map = load_intrinsics(
            intrinsics, args.width, args.height, cv2
        )
        rect_fy = float(rect_matrix[1, 1])
        model.cam_fovy[camera_id] = float(
            np.degrees(2.0 * np.arctan((args.height / 2.0) / rect_fy))
        )
    else:
        focal = (args.height / 2.0) / np.tan(
            np.radians(model.cam_fovy[camera_id]) / 2.0
        )
        rect_matrix = np.array(
            [[focal, 0, args.width / 2.0], [0, focal, args.height / 2.0], [0, 0, 1]],
            dtype=float,
        )

    detection_size = (args.camera_width, args.camera_height)
    if intrinsics is not None:
        detection_matrix, detection_map = load_intrinsics(intrinsics, *detection_size, cv2)
    else:
        sx = args.camera_width / args.width
        sy = args.camera_height / args.height
        detection_matrix = np.array(
            [
                [rect_matrix[0, 0] * sx, 0, rect_matrix[0, 2] * sx],
                [0, rect_matrix[1, 1] * sy, rect_matrix[1, 2] * sy],
                [0, 0, 1],
            ],
            dtype=float,
        )
        detection_map = None

    cv_cuda = OpenCvCuda(args.opencv_cuda)

    runtimes = [
        ArmRuntime(
            config=arm,
            kinematics=derive_kinematics(model, names_for_arm(arm.name)),
        )
        for arm in arms
    ]
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    real = RealSource(
        image_path=args.real_image,
        camera=(
            parse_index_or_path(args.overhead_camera)
            if args.overhead_camera is not None
            else None
        ),
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        cv2_module=cv2,
    )
    cube_tracker = CubeTracker(smooth=args.cube_smooth) if args.track_cube else None
    drop_tracker = PaperTracker(alpha=args.drop_zone_smooth) if args.track_drop_zone else None
    workspace_corners = workspace_interior_corners_world() if args.track_drop_zone else None

    mode = "blend"
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    try:
        while True:
            mujoco.mj_forward(model, data)
            frame = real.read(args.width, args.height)
            if undistort_map is not None:
                frame = cv_cuda.remap(frame, undistort_map)

            cube_pose = None
            drop_target = None
            tag_detections = []
            if cube_tracker is not None or drop_tracker is not None:
                det_rgb = real.read(*detection_size)
                if detection_map is not None:
                    det_rgb = cv_cuda.remap(det_rgb, detection_map)

                cam_pos = data.cam_xpos[camera_id]
                cam_rot = data.cam_xmat[camera_id].reshape(3, 3)
                if drop_tracker is not None:
                    raw_target = detect_paper_target(
                        det_rgb,
                        detection_matrix,
                        cam_pos,
                        cam_rot,
                        target_color=args.drop_zone_color,
                        workspace_corners_world=workspace_corners,
                    )
                    drop_target = drop_tracker.update(raw_target)
                if cube_tracker is not None:
                    tag_detections = detect_tags(det_rgb, cube_tracker.detector)
                    cube_detections = [
                        det for det in tag_detections if det.tag_id in CUBE_TAG_IDS
                    ]
                    cube_pose = cube_tracker.update(
                        cube_detections, detection_matrix, cam_pos, cam_rot
                    )

            renderer.update_scene(data, camera="overhead_camera")
            sim = renderer.render()
            if mode == "edges":
                out = cv_cuda.edge_overlay(sim, frame)
            else:
                blended = cv2.addWeighted(frame, alpha, sim, 1.0 - alpha, 0.0)
                out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            if drop_target is not None:
                draw_paper_target(
                    out,
                    drop_target,
                    args.width / detection_size[0],
                    args.height / detection_size[1],
                )
            if tag_detections:
                draw_tag_detections(
                    out,
                    tag_detections,
                    args.width / detection_size[0],
                    args.height / detection_size[1],
                )

            selected, statuses, drop_reachable = _choose_arm(runtimes, cube_pose, drop_target)
            if drop_target is not None:
                set_paper_target_marker(model, data, drop_target, usable=drop_reachable)
            object_position = None if cube_pose is None else cube_pose.position
            target_position = None if drop_target is None else drop_target.center_world
            lines = [
                _format_xyz("object", object_position),
                _format_xyz("target", target_position),
                "select: lowest score where pick=ok and drop=ok",
                f"opencv: {'cuda' if cv_cuda.enabled else 'cpu'}",
                f"selected: {selected or 'none'}",
                *statuses,
            ]
            out = draw_hud(out, mode=mode, alpha=alpha, intrinsics=intrinsics)
            _put_lines(out, lines)
            cv2.imshow(WINDOW_TITLE, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = "edges" if mode == "blend" else "blend"
            elif key == ord(","):
                alpha = float(np.clip(alpha - 0.05, 0.0, 1.0))
            elif key == ord("."):
                alpha = float(np.clip(alpha + 0.05, 0.0, 1.0))
            elif key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"/tmp/multiarm_overhead_{stamp}.png", out)
            if args.real_image is not None and key == -1:
                cv2.waitKey(0)
    finally:
        renderer.close()
        real.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
