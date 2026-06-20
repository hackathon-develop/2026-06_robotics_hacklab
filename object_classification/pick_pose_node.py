#!/usr/bin/env python3
"""Overhead camera object detection → pick pose printer.

Detects objects in an overhead camera feed and prints the best match as a
world-frame (x, y, z) pick coordinate, so a robot arm can reach and pick it.

Pixel → world XY transform uses calibrated camera intrinsics (K matrix +
distortion coefficients from overhead_camera.json) plus camera extrinsics
(position and rotation in world frame, from MuJoCo).  Each detected
bounding-box centre is undistorted, back-projected into a ray in the camera
frame, converted from OpenCV to MuJoCo convention (y and z flipped), rotated
to world frame with cam_rot, and intersected with the table plane (z = pick_z).

Usage
-----
python pick_pose_node.py 0 \\
    --cam-pos 0.0 0.0 0.65 \\
    --cam-xmat 1 0 0 0 0 -1 0 1 0 \\
    --pick-z 0.0 \\
    --target-class bottle

Press q in the camera window or Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow importing the sibling module without installing it as a package.
sys.path.insert(0, str(Path(__file__).parent))
from object_classification import (
    camera_frame_to_image,
    class_name_from_names,
    configure_model_environment,
    load_detection_model,
    matches_target_classes,
    normalize_label,
    rf_detr_label_for_detection,
)

import json

# Default intrinsics JSON (two levels up from this file → repo root).
_DEFAULT_INTRINSICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "camera_intrinsics"
    / "overhead_camera.json"
)


# ---------------------------------------------------------------------------
# Camera intrinsics helpers
# ---------------------------------------------------------------------------

def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load K matrix, distortion coefficients, and calibration resolution.

    Returns (K 3×3, dist 1-D, calib_width, calib_height).
    """
    data = json.loads(path.read_text())
    K = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(data["dist_coeffs"], dtype=float)
    return K, dist, int(data["width"]), int(data["height"])


def _scale_K(K: np.ndarray, calib_w: int, calib_h: int, frame_w: int, frame_h: int) -> np.ndarray:
    """Scale K from calibration resolution to actual frame resolution."""
    if frame_w == calib_w and frame_h == calib_h:
        return K
    K = K.copy()
    K[0, :] *= frame_w / calib_w  # fx, cx
    K[1, :] *= frame_h / calib_h  # fy, cy
    return K


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def _pixel_to_world_xy(
    u: float,
    v: float,
    K: np.ndarray,
    dist: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    pick_z: float,
) -> tuple[float, float]:
    """Back-project pixel (u, v) to world (x, y) at table height z = pick_z.

    Steps:
    1. cv2.undistortPoints removes lens distortion → normalised camera coords.
    2. Flip to MuJoCo convention (x right, y up, z backward): [x_n, -y_n, -1].
    3. Rotate ray to world frame via cam_rot (MuJoCo cam_xmat).
    4. Intersect ray with plane z = pick_z.
    """
    pts = np.array([[[u, v]]], dtype=np.float32)
    nd = cv2.undistortPoints(pts, K, dist)  # normalised, no P
    x_n, y_n = float(nd[0, 0, 0]), float(nd[0, 0, 1])

    ray_mj = np.array([x_n, -y_n, -1.0])       # OpenCV → MuJoCo frame
    ray_world = cam_rot @ ray_mj

    if abs(ray_world[2]) < 1e-6:
        raise ValueError(f"Ray parallel to table for pixel ({u:.1f}, {v:.1f}).")

    t = (pick_z - cam_pos[2]) / ray_world[2]
    if t < 0:
        raise ValueError(f"Ray intersects table behind the camera for pixel ({u:.1f}, {v:.1f}).")

    return float(cam_pos[0] + t * ray_world[0]), float(cam_pos[1] + t * ray_world[1])


# ---------------------------------------------------------------------------
# Detection collectors
# ---------------------------------------------------------------------------

def _collect_yolo_detections(
    results: list[Any],
    target_classes: list[str],
    class_name_lookup: list[str] | None,
) -> list[dict]:
    detections = []
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            label = class_name_from_names(names, int(box.cls[0]), class_name_lookup=class_name_lookup)
            if target_classes and not matches_target_classes(label, target_classes):
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "xyxy": (x1, y1, x2, y2),
            })
    return detections


def _collect_rfdetr_detections(
    detections_raw: Any,
    target_classes: list[str],
    class_name_lookup: list[str] | None,
) -> list[dict]:
    detections = []
    class_names = detections_raw.data.get("class_name", [])
    for idx, (xyxy, conf, class_id) in enumerate(
        zip(detections_raw.xyxy, detections_raw.confidence, detections_raw.class_id)
    ):
        label = rf_detr_label_for_detection(class_id, idx, class_names, class_name_lookup)
        if target_classes and not matches_target_classes(label, target_classes):
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        detections.append({
            "label": label,
            "confidence": float(conf),
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "xyxy": (x1, y1, x2, y2),
        })
    return detections


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------

def _detect(
    pil_image: Any,
    loaded_model: tuple[str, Any, list[str] | None],
    target_classes: list[str],
    confidence: float,
) -> list[dict]:
    model_type, model, class_name_lookup = loaded_model
    if model_type == "rf-detr":
        raw = model.predict(pil_image, threshold=confidence)
        dets = _collect_rfdetr_detections(raw, target_classes, class_name_lookup)
    else:
        results = list(model.predict(source=pil_image, conf=confidence, verbose=False))
        dets = _collect_yolo_detections(results, target_classes, class_name_lookup)
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _annotate_frame(frame: np.ndarray, detections: list[dict]) -> None:
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = (0, 255, 0) if i == 0 else (0, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['confidence']:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects from overhead camera and print pick pose in world frame."
    )
    parser.add_argument(
        "camera_id",
        type=int,
        nargs="?",
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Detection model: yolo11m.pt, rf-detr-base-o365, yolo-world-tools, etc. (default: yolo11m.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25).",
    )
    parser.add_argument(
        "--target-class",
        default="",
        help="Object class to pick. Empty = highest-confidence object. Example: bottle",
    )
    parser.add_argument(
        "--intrinsics",
        default="",
        help="Path to overhead_camera.json. Defaults to config/camera_intrinsics/overhead_camera.json.",
    )
    parser.add_argument(
        "--cam-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 1.0],
        help="Camera position in world frame from MuJoCo data.cam_xpos. Example: 0.0 0.0 0.65",
    )
    parser.add_argument(
        "--cam-xmat",
        type=float,
        nargs=9,
        metavar="V",
        default=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        help=(
            "Camera rotation matrix (row-major 3×3) from MuJoCo data.cam_xmat. "
            "9 floats. Example: 1 0 0 0 0 -1 0 1 0"
        ),
    )
    parser.add_argument(
        "--pick-z",
        type=float,
        default=0.0,
        help="Table surface height in world frame (default: 0.0).",
    )
    parser.add_argument(
        "--max-hz",
        type=float,
        default=10.0,
        help="Maximum processing rate in Hz (default: 10.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- intrinsics ---
    intrinsics_path = Path(args.intrinsics) if args.intrinsics.strip() else _DEFAULT_INTRINSICS_PATH
    if not intrinsics_path.is_file():
        raise SystemExit(
            f"Camera intrinsics file not found: {intrinsics_path}\n"
            "Place overhead_camera.json at config/camera_intrinsics/overhead_camera.json "
            "or pass --intrinsics <path>."
        )
    K, dist, calib_w, calib_h = _load_intrinsics(intrinsics_path)
    print(f"Loaded intrinsics from '{intrinsics_path}'.")

    # --- extrinsics ---
    cam_pos = np.array(args.cam_pos, dtype=float)
    cam_rot = np.array(args.cam_xmat, dtype=float).reshape(3, 3)
    pick_z = args.pick_z

    # --- target class ---
    target_classes = [normalize_label(args.target_class)] if args.target_class.strip() else []

    # --- model ---
    configure_model_environment()
    print(f"Loading model '{args.model}' …")
    loaded_model = load_detection_model(args.model, target_classes=target_classes or None)
    model_type = loaded_model[0]
    print(f"Model loaded (type={model_type}).")

    # --- camera ---
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera_id}.")
    print(f"Opened camera {args.camera_id}. Press q to quit.")

    min_interval = 1.0 / max(args.max_hz, 1.0)
    last_t = 0.0
    K_scaled: np.ndarray | None = None
    last_frame_size: tuple[int, int] | None = None
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            # Rate cap.
            now = time.monotonic()
            if now - last_t < min_interval:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_t = now
            frame_number += 1

            # Scale K to live resolution once (or on resolution change).
            frame_h, frame_w = frame.shape[:2]
            if last_frame_size != (frame_w, frame_h):
                last_frame_size = (frame_w, frame_h)
                K_scaled = _scale_K(K, calib_w, calib_h, frame_w, frame_h)
                print(f"Frame {frame_w}×{frame_h} (calib {calib_w}×{calib_h}) — K scaled.")

            pil_image = camera_frame_to_image(frame)
            detections = _detect(pil_image, loaded_model, target_classes, args.confidence)

            print(f"frame={frame_number}")
            if detections:
                best = detections[0]
                try:
                    world_x, world_y = _pixel_to_world_xy(
                        best["cx"], best["cy"],
                        K_scaled, dist, cam_pos, cam_rot, pick_z,
                    )
                    print(
                        f"  PICK label={best['label']} conf={best['confidence']:.2f} "
                        f"pixel=({best['cx']:.1f}, {best['cy']:.1f}) "
                        f"world=({world_x:.4f}, {world_y:.4f}, {pick_z:.4f})"
                    )
                except ValueError as exc:
                    print(f"  SKIP {exc}")
            else:
                target_desc = ", ".join(target_classes) if target_classes else "any object"
                print(f"  No detections for {target_desc}.")

            _annotate_frame(frame, detections)
            cv2.imshow("pick_pose — press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
