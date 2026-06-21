#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD
"""First perception sequence for object-conditioned pick-and-place.

This script wires the current-branch overhead ``pick_pose_node`` with the
``wrist_align`` yaw estimate used as a local reference. It does not move the arm;
it emits the structured pose contract the next arm/VLA orchestrator can consume.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_SRC = REPO_ROOT / "py" / "src"
sys.path.insert(0, str(PY_SRC))
sys.path.insert(0, str(Path(__file__).parent))
from pick_pose_node import (
    _DEFAULT_INTRINSICS_PATH as DEFAULT_OVERHEAD_INTRINSICS_PATH,
    _annotate_frame,
    _load_intrinsics,
    _scale_K,
    load_pick_pose_model,
    pick_poses_from_frame,
    poses_by_label,
    target_classes_from_target_class,
)
from wrist_align import (
    _DEFAULT_INTRINSICS_PATH as DEFAULT_WRIST_INTRINSICS_PATH,
    _load_intrinsics as load_wrist_intrinsics,
    _scale_K as scale_wrist_K,
    wrist_alignment_from_frame,
)


class OverheadCameraReader:
    """Continuously read the overhead camera and optionally show a live window."""

    def __init__(self, camera_id: int, *, show_window: bool) -> None:
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open overhead camera {camera_id}.")
        self._show_window = show_window
        self._window_name = "first_pick_sequence overhead - press q to quit"
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._detections: list[dict[str, Any]] = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        if self._show_window:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame.copy()
                detections = list(self._detections)
            if self._show_window:
                display = frame.copy()
                if detections:
                    _annotate_frame(display, detections)
                cv2.imshow(self._window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self._running = False

    def latest(self, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.01)
        raise RuntimeError("timed out reading overhead camera")

    def update_detections(self, poses: list[Any]) -> None:
        with self._lock:
            self._detections = _detections_from_poses(poses)

    def wait_for_window(self, timeout_ms: int) -> None:
        if not self._show_window:
            return
        if timeout_ms > 0:
            deadline = time.monotonic() + timeout_ms / 1000.0
            while self._running and time.monotonic() < deadline:
                time.sleep(0.05)
            return
        while self._running:
            time.sleep(0.05)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
        if self._show_window:
            cv2.destroyWindow(self._window_name)


def _read_frame(cap: Any, timeout_s: float, label: str) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.01)
    raise RuntimeError(f"timed out reading {label} camera")


def _scaled_intrinsics_for_frame(
    frame: np.ndarray,
    K: np.ndarray,
    calib_w: int,
    calib_h: int,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    return _scale_K(K, calib_w, calib_h, frame_w, frame_h)


def _mujoco_camera_extrinsics(camera_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return world camera position/rotation from the calibrated MuJoCo scene."""
    import mujoco

    from pick_and_place import build_scene
    from pick_and_place.camera_extrinsics import (
        apply_camera_extrinsics_to_model,
        load_local_camera_extrinsics,
    )

    model = build_scene(include_environment=True).compile()
    data = mujoco.MjData(model)
    applied = apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise RuntimeError(f"MuJoCo scene has no camera named {camera_name!r}")
    return (
        data.cam_xpos[camera_id].copy(),
        data.cam_xmat[camera_id].reshape(3, 3).copy(),
        applied,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run overhead pick pose plus wrist yaw handoff."
    )
    parser.add_argument("camera_id", type=int, nargs="?", default=0, help="Overhead camera index.")
    parser.add_argument("--wrist-camera-id", type=int, default=1, help="Wrist camera index.")
    parser.add_argument("--no-wrist-align", action="store_true", help="Skip wrist-camera rz estimate.")
    parser.add_argument(
        "--model", default="rf-detr-base-o365", help="Overhead detection model."
    )
    parser.add_argument("--wrist-model", default="rf-detr-base-o365", help="Wrist detection model.")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Overhead confidence threshold."
    )
    parser.add_argument(
        "--wrist-confidence", type=float, default=0.10, help="Wrist confidence threshold."
    )
    parser.add_argument(
        "--target-class", default="", help="Object class to select; empty means best object."
    )
    parser.add_argument("--intrinsics", default="", help="Path to overhead_camera.json.")
    parser.add_argument("--wrist-intrinsics", default="", help="Path to wrist_camera.json.")
    parser.add_argument(
        "--camera-name",
        default="overhead_camera",
        help="MuJoCo camera name used for overhead extrinsics.",
    )
    parser.add_argument(
        "--manual-camera-extrinsics",
        action="store_true",
        help="Use --cam-pos/--cam-xmat instead of MuJoCo camera extrinsics.",
    )
    parser.add_argument(
        "--cam-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 1.0],
        help="Overhead camera world position from MuJoCo data.cam_xpos.",
    )
    parser.add_argument(
        "--cam-xmat",
        type=float,
        nargs=9,
        metavar="V",
        default=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        help="Overhead camera world rotation, row-major MuJoCo data.cam_xmat.",
    )
    parser.add_argument("--pick-z", type=float, default=0.0, help="Table plane z in world meters.")
    parser.add_argument(
        "--hover-offset",
        type=float,
        default=0.15,
        help="Hover height above each detected object's world z, in meters.",
    )
    parser.add_argument(
        "--hover-z",
        type=float,
        default=None,
        help="Absolute arm hover z in world meters. Overrides --hover-offset when set.",
    )
    parser.add_argument(
        "--hover-z-reference",
        choices=("gripper_body", "ik_target"),
        default="gripper_body",
        help="What arm_hover_pose.z refers to for IK movement.",
    )
    parser.add_argument("--rx", type=float, default=180.0, help="Default hover rx in degrees.")
    parser.add_argument("--ry", type=float, default=0.0, help="Default hover ry in degrees.")
    parser.add_argument("--rz", type=float, default=0.0, help="Fallback hover rz in degrees.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Camera read timeout in seconds.")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not show the annotated overhead camera frame.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=0,
        help="Milliseconds to keep the live overhead window after the sequence; 0 waits for q.",
    )
    parser.add_argument("--move-arm", action="store_true", help="Move the follower to arm_hover_pose.")
    parser.add_argument("--follower-port", default="", help="Serial port of the SO-101 follower.")
    parser.add_argument("--follower-id", default="folly", help="Follower calibration id.")
    parser.add_argument("--offsets-path", default=None, help="Optional sim->real joint offsets JSON.")
    parser.add_argument("--move-duration", type=float, default=2.0, help="Minimum arm ramp duration.")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Arm command streaming rate.")
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=10.0,
        help="Arm joint speed cap in deg/s.",
    )
    parser.add_argument(
        "--dry-run-arm",
        action="store_true",
        help="Solve the arm target without connecting or moving.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print detailed detection, projection, and arm target debug.",
    )
    return parser.parse_args()


def _detections_from_poses(poses: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": pose.label,
            "confidence": pose.confidence,
            "xyxy": pose.xyxy,
        }
        for pose in poses
    ]


def _hover_pose_from_pick(pose: Any, hover_z: float, z_reference: str, rz: float) -> dict[str, Any]:
    return {
        "label": pose.label,
        "confidence": pose.confidence,
        "x": pose.x,
        "y": pose.y,
        "z": hover_z,
        "z_reference": z_reference,
        "rx": pose.rx,
        "ry": pose.ry,
        "rz": rz,
        "source_pixel": [pose.cx, pose.cy],
        "source_xyxy": list(pose.xyxy),
    }


def _hover_z_for_pick(pose: Any, absolute_hover_z: float | None, hover_offset: float) -> float:
    if absolute_hover_z is not None:
        return float(absolute_hover_z)
    return float(pose.z) + float(hover_offset)


def _print_detection_debug(
    poses: list[Any],
    hover_poses: list[dict[str, Any]],
    *,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    extrinsics_source: str,
    applied_extrinsics: list[str],
) -> None:
    print("[debug] overhead camera extrinsics", file=sys.stderr)
    print(f"[debug]   source={extrinsics_source}", file=sys.stderr)
    print(f"[debug]   applied_local_extrinsics={applied_extrinsics}", file=sys.stderr)
    print(f"[debug]   cam_pos={cam_pos.tolist()}", file=sys.stderr)
    print(f"[debug]   cam_xmat={cam_rot.reshape(-1).tolist()}", file=sys.stderr)
    print(f"[debug] detected {len(poses)} object(s), sorted by confidence", file=sys.stderr)
    for index, (pose, hover) in enumerate(zip(poses, hover_poses), start=1):
        selected = "yes" if index == 1 else "no"
        print(
            "[debug] "
            f"#{index} selected={selected} label={pose.label!r} conf={pose.confidence:.3f} "
            f"pixel=({pose.cx:.1f},{pose.cy:.1f}) bbox={list(pose.xyxy)} "
            f"world=({pose.x:.4f},{pose.y:.4f},{pose.z:.4f}) "
            f"hover=({hover['x']:.4f},{hover['y']:.4f},{hover['z']:.4f}) "
            f"rz={hover['rz']:.2f} z_ref={hover['z_reference']}",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    if args.move_arm and not args.dry_run_arm and not args.follower_port:
        raise SystemExit("--follower-port is required with --move-arm")
    target_classes = target_classes_from_target_class(args.target_class)

    overhead_intrinsics = (
        Path(args.intrinsics)
        if args.intrinsics.strip()
        else DEFAULT_OVERHEAD_INTRINSICS_PATH
    )
    if not overhead_intrinsics.is_file():
        raise SystemExit(f"Overhead intrinsics not found: {overhead_intrinsics}")
    K, dist, calib_w, calib_h = _load_intrinsics(overhead_intrinsics)

    print(f"Loading overhead model '{args.model}' ...", file=sys.stderr)
    overhead_model = load_pick_pose_model(args.model, target_classes)

    if args.manual_camera_extrinsics:
        cam_pos = np.array(args.cam_pos, dtype=float)
        cam_rot = np.array(args.cam_xmat, dtype=float).reshape(3, 3)
        applied_extrinsics: list[str] = []
        extrinsics_source = "manual --cam-pos/--cam-xmat"
    else:
        cam_pos, cam_rot, applied_extrinsics = _mujoco_camera_extrinsics(args.camera_name)
        extrinsics_source = f"MuJoCo camera {args.camera_name!r}"

    overhead = OverheadCameraReader(args.camera_id, show_window=not args.no_window)

    wrist = None
    wrist_model = None
    wrist_K = wrist_dist = None
    wrist_calib_w = wrist_calib_h = None
    if not args.no_wrist_align:
        wrist_intrinsics = (
            Path(args.wrist_intrinsics)
            if args.wrist_intrinsics.strip()
            else DEFAULT_WRIST_INTRINSICS_PATH
        )
        if not wrist_intrinsics.is_file():
            raise SystemExit(f"Wrist intrinsics not found: {wrist_intrinsics}")
        wrist_K, wrist_dist, wrist_calib_w, wrist_calib_h = load_wrist_intrinsics(
            wrist_intrinsics
        )
        print(f"Loading wrist model '{args.wrist_model}' ...", file=sys.stderr)
        wrist_model = load_pick_pose_model(args.wrist_model, target_classes)
        wrist = cv2.VideoCapture(args.wrist_camera_id)
        if not wrist.isOpened():
            raise SystemExit(f"Could not open wrist camera {args.wrist_camera_id}.")

    try:
        overhead_frame = overhead.latest(args.timeout)
        K_scaled = _scaled_intrinsics_for_frame(overhead_frame, K, calib_w, calib_h)
        poses = pick_poses_from_frame(
            overhead_frame,
            overhead_model,
            target_classes,
            args.confidence,
            K_scaled,
            dist,
            cam_pos,
            cam_rot,
            args.pick_z,
            rx=args.rx,
            ry=args.ry,
            rz=args.rz,
        )
        if not poses:
            target_desc = args.target_class if args.target_class else "any object"
            raise SystemExit(f"No overhead detection for {target_desc}.")
        overhead.update_detections(poses)

        selected = poses[0]
        wrist_alignment = None
        final_rz = selected.rz
        if wrist is not None and wrist_model is not None:
            wrist_frame = _read_frame(wrist, args.timeout, "wrist")
            assert wrist_K is not None and wrist_dist is not None
            assert wrist_calib_w is not None and wrist_calib_h is not None
            wrist_K_scaled = scale_wrist_K(
                wrist_K,
                wrist_calib_w,
                wrist_calib_h,
                wrist_frame.shape[1],
                wrist_frame.shape[0],
            )
            alignment, _, _ = wrist_alignment_from_frame(
                wrist_frame,
                wrist_model,
                target_classes,
                args.wrist_confidence,
                K=wrist_K_scaled,
                dist=wrist_dist,
            )
            if alignment is not None:
                wrist_alignment = alignment.as_dict()
                if alignment.rz is not None:
                    final_rz = alignment.rz

        selected_pose = selected.as_dict()
        selected_pose["rz"] = final_rz
        hover_poses = [
            _hover_pose_from_pick(
                pose,
                _hover_z_for_pick(pose, args.hover_z, args.hover_offset),
                args.hover_z_reference,
                final_rz if index == 0 else pose.rz,
            )
            for index, pose in enumerate(poses)
        ]
        if args.debug:
            _print_detection_debug(
                poses,
                hover_poses,
                cam_pos=cam_pos,
                cam_rot=cam_rot,
                extrinsics_source=extrinsics_source,
                applied_extrinsics=applied_extrinsics,
            )
        payload = {
            "objects": poses_by_label(poses),
            "selected_label": selected.label,
            "selected_pose": selected_pose,
            "wrist_alignment": wrist_alignment,
            "arm_hover_pose": hover_poses[0],
            "arm_hover_poses": hover_poses,
            "camera_extrinsics": {
                "source": extrinsics_source,
                "applied_local_extrinsics": applied_extrinsics,
                "cam_pos": cam_pos.tolist(),
                "cam_xmat": cam_rot.reshape(-1).tolist(),
            },
            "units": {
                "position": "meters",
                "rotation": "degrees",
                "frame": "MuJoCo/world",
            },
        }
        print(json.dumps(payload, indent=2), flush=True)
        if args.move_arm:
            from arm_hover_move import move_to_hover_payload

            diagnostics = move_to_hover_payload(
                payload,
                follower_port=args.follower_port,
                follower_id=args.follower_id,
                offsets_path=args.offsets_path,
                duration=args.move_duration,
                control_hz=args.control_hz,
                max_joint_speed=args.max_joint_speed,
                dry_run=args.dry_run_arm,
            )
            print(json.dumps({"arm_move": diagnostics}, indent=2), flush=True)
        overhead.wait_for_window(args.window_ms)
    finally:
        overhead.close()
        if wrist is not None:
            wrist.release()


if __name__ == "__main__":
    main()
