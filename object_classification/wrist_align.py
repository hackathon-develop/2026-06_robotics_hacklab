#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD
"""Wrist-camera yaw estimate for the local pick sequence.

This is the reusable version of the wrist-align reference flow: detect the object
under the wrist camera, crop its bounding box, fit a contour orientation, and
return the gripper yaw ``rz`` in degrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from object_classification import normalize_label
from pick_pose_node import _annotate_frame, _detect, load_pick_pose_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INTRINSICS_PATH = _REPO_ROOT / "config" / "camera_intrinsics" / "wrist_camera.json"


@dataclass(frozen=True)
class WristAlignment:
    label: str
    confidence: float
    rz: float | None
    bbox_area: float
    xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    data = json.loads(path.read_text())
    K = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(data["dist_coeffs"], dtype=float)
    return K, dist, int(data["width"]), int(data["height"])


def _scale_K(
    K: np.ndarray,
    calib_w: int,
    calib_h: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    if frame_w == calib_w and frame_h == calib_h:
        return K
    scaled = K.copy()
    scaled[0, :] *= frame_w / calib_w
    scaled[1, :] *= frame_h / calib_h
    return scaled


def _compute_rz(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> float | None:
    """Return object long-axis yaw in image/gripper coordinates, in degrees."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None

    _, (box_w, box_h), angle = cv2.minAreaRect(largest)
    if box_h > box_w:
        angle += 90.0
    if angle > 90.0:
        angle -= 180.0
    return float(angle)


def wrist_alignment_from_frame(
    frame: np.ndarray,
    loaded_model: tuple[str, Any, list[str] | None],
    target_classes: list[str],
    confidence: float,
    *,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> tuple[WristAlignment | None, np.ndarray, list[dict]]:
    """Estimate wrist-camera object yaw from one BGR frame.

    Returns ``(alignment, processed_frame, detections)``. ``alignment`` is the
    highest-confidence detection, or ``None`` when no target is visible.
    """
    processed = (
        cv2.undistort(frame, K, dist) if K is not None and dist is not None else frame
    )
    from object_classification import camera_frame_to_image

    detections = _detect(
        camera_frame_to_image(processed), loaded_model, target_classes, confidence
    )
    if not detections:
        return None, processed, detections

    best = detections[0]
    x1, y1, x2, y2 = [float(v) for v in best["xyxy"]]
    alignment = WristAlignment(
        label=str(best["label"]),
        confidence=float(best["confidence"]),
        rz=_compute_rz(processed, (x1, y1, x2, y2)),
        bbox_area=(x2 - x1) * (y2 - y1),
        xyxy=(x1, y1, x2, y2),
    )
    return alignment, processed, detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate object yaw from the wrist camera.")
    parser.add_argument(
        "--camera-id", type=int, default=1, help="Wrist camera index (default: 1)."
    )
    parser.add_argument("--model", default="rf-detr-base-o365", help="Detection model.")
    parser.add_argument("--confidence", type=float, default=0.10, help="Minimum confidence.")
    parser.add_argument("--target-class", default="", help="Object class to align to.")
    parser.add_argument("--intrinsics", default="", help="Path to wrist_camera.json.")
    parser.add_argument("--max-hz", type=float, default=10.0, help="Maximum processing rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intrinsics_path = (
        Path(args.intrinsics) if args.intrinsics.strip() else _DEFAULT_INTRINSICS_PATH
    )
    if not intrinsics_path.is_file():
        raise SystemExit(f"Wrist camera intrinsics not found: {intrinsics_path}")
    K, dist, calib_w, calib_h = _load_intrinsics(intrinsics_path)

    target_classes = [normalize_label(args.target_class)] if args.target_class.strip() else []
    print(f"Loading model '{args.model}' ...")
    loaded_model = load_pick_pose_model(args.model, target_classes)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open wrist camera {args.camera_id}.")

    min_interval = 1.0 / max(args.max_hz, 1.0)
    last_t = 0.0
    K_scaled: np.ndarray | None = None
    last_frame_size: tuple[int, int] | None = None
    frame_number = 0
    window_name = "wrist_align - press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from wrist camera.")
                break
            now = time.monotonic()
            if now - last_t < min_interval:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_t = now
            frame_number += 1

            frame_h, frame_w = frame.shape[:2]
            if last_frame_size != (frame_w, frame_h):
                last_frame_size = (frame_w, frame_h)
                K_scaled = _scale_K(K, calib_w, calib_h, frame_w, frame_h)
            assert K_scaled is not None

            alignment, processed, detections = wrist_alignment_from_frame(
                frame,
                loaded_model,
                target_classes,
                args.confidence,
                K=K_scaled,
                dist=dist,
            )
            if alignment is None:
                print(f"frame={frame_number} no detection")
            else:
                rz_text = "no_contour" if alignment.rz is None else f"{alignment.rz:.1f}deg"
                print(
                    f"frame={frame_number} label={alignment.label} "
                    f"conf={alignment.confidence:.2f} "
                    f"bbox_area={alignment.bbox_area:.0f}px2 rz={rz_text}"
                )
            _annotate_frame(processed, detections)
            cv2.imshow(window_name, processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
