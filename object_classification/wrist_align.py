#!/usr/bin/env python3
"""Wrist camera orientation alignment — phase 2 of pick-and-place.

Uses the wrist-mounted camera (camera 1) to detect the target object up close
and compute the gripper yaw (rz) needed to align with the object's orientation.
Run this while the arm is descending toward the object; the rz output updates
automatically as the arm approaches and the object fills more of the frame.

Pipeline per frame
------------------
1. Detect the target object with RF-DETR / YOLO.
2. Crop the bounding box region from the undistorted frame.
3. Threshold the crop and find the largest contour.
4. Fit cv2.minAreaRect to the contour → orientation angle.
5. Convert angle to gripper rz (degrees).
6. Print: rz=<angle>  bbox_area=<px²>  conf=<score>
   (bbox_area grows as the arm descends — bigger = closer = more reliable)

Usage
-----
python wrist_align.py --target-class bottle
python wrist_align.py --target-class screwdriver --model rf-detr-base-o365
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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

_DEFAULT_INTRINSICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "camera_intrinsics"
    / "wrist_camera.json"
)


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    data = json.loads(path.read_text())
    K = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(data["dist_coeffs"], dtype=float)
    return K, dist, int(data["width"]), int(data["height"])


def _scale_K(K: np.ndarray, calib_w: int, calib_h: int, frame_w: int, frame_h: int) -> np.ndarray:
    if frame_w == calib_w and frame_h == calib_h:
        return K
    K = K.copy()
    K[0, :] *= frame_w / calib_w
    K[1, :] *= frame_h / calib_h
    return K


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _collect_yolo_detections(
    results: list[Any], target_classes: list[str], class_name_lookup: list[str] | None
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
    detections_raw: Any, target_classes: list[str], class_name_lookup: list[str] | None
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
# Orientation from contour
# ---------------------------------------------------------------------------

def _compute_rz(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> float | None:
    """Fit a rotated rectangle to the object contour inside the bounding box.

    Returns the gripper yaw (rz) in degrees, or None if no contour is found.

    cv2.minAreaRect angle is in (-90, 0]:
      -  0° → box is horizontal  → object long axis is horizontal
      - -90° → box is vertical   → object long axis is vertical
    We convert to a positive clockwise angle for the gripper.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    # Clamp to frame bounds.
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to handle varying lighting as arm approaches.
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:  # too small to be reliable
        return None

    _, (box_w, box_h), angle = cv2.minAreaRect(largest)

    # If the box is taller than wide, the object's long axis is vertical.
    # Rotate angle by 90° so rz always refers to the long axis of the object.
    if box_h > box_w:
        angle += 90.0

    # angle is now in (-90, 90]: normalise to (-90, 90].
    if angle > 90.0:
        angle -= 180.0

    return float(angle)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _annotate(
    frame: np.ndarray,
    detections: list[dict],
    rz: float | None,
) -> None:
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = (0, 255, 0) if i == 0 else (0, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{d['label']} {d['confidence']:.2f}"
        cv2.putText(frame, label_text, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if i == 0 and rz is not None:
            # Draw the orientation arrow from the bounding box centre.
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            length = int(min(x2 - x1, y2 - y1) / 2)
            angle_rad = np.radians(rz)
            ex = int(cx + length * np.cos(angle_rad))
            ey = int(cy - length * np.sin(angle_rad))
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 165, 255), 2, tipLength=0.3)
            cv2.putText(frame, f"rz={rz:.1f}deg", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    if not detections:
        cv2.putText(frame, "No detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrist camera orientation alignment for gripper yaw (rz)."
    )
    parser.add_argument(
        "--camera-id", type=int, default=1,
        help="Wrist camera OpenCV index (default: 1).",
    )
    parser.add_argument(
        "--model", default="rf-detr-base-o365",
        help="Detection model (default: rf-detr-base-o365).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.10,
        help="Minimum detection confidence (default: 0.10).",
    )
    parser.add_argument(
        "--target-class", default="",
        help="Object class to align to. Example: bottle",
    )
    parser.add_argument(
        "--intrinsics", default="",
        help="Path to wrist_camera.json. Defaults to config/camera_intrinsics/wrist_camera.json.",
    )
    parser.add_argument(
        "--max-hz", type=float, default=10.0,
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
            f"Wrist camera intrinsics not found: {intrinsics_path}\n"
            "Pass --intrinsics <path> or place wrist_camera.json at "
            "config/camera_intrinsics/wrist_camera.json."
        )
    K, dist, calib_w, calib_h = _load_intrinsics(intrinsics_path)
    print(f"Loaded wrist camera intrinsics from '{intrinsics_path}'.")

    target_classes = [normalize_label(args.target_class)] if args.target_class.strip() else []

    # --- model ---
    configure_model_environment()
    print(f"Loading model '{args.model}' …")
    loaded_model = load_detection_model(args.model, target_classes=target_classes or None)
    print(f"Model loaded (type={loaded_model[0]}).")

    # --- camera ---
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open wrist camera {args.camera_id}.")
    print(f"Opened wrist camera {args.camera_id}. Press q to quit.")
    print("Move the arm toward the object — rz updates automatically.\n")

    window_name = "wrist_align — press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    min_interval = 1.0 / max(args.max_hz, 1.0)
    last_t = 0.0
    K_scaled: np.ndarray | None = None
    last_frame_size: tuple[int, int] | None = None
    frame_number = 0

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

            # Undistort frame for cleaner contour extraction.
            frame_h, frame_w = frame.shape[:2]
            if last_frame_size != (frame_w, frame_h):
                last_frame_size = (frame_w, frame_h)
                K_scaled = _scale_K(K, calib_w, calib_h, frame_w, frame_h)
                print(f"Frame {frame_w}×{frame_h} (calib {calib_w}×{calib_h}) — K scaled.")
            frame = cv2.undistort(frame, K_scaled, dist)

            pil_image = camera_frame_to_image(frame)
            detections = _detect(pil_image, loaded_model, target_classes, args.confidence)

            rz = None
            if detections:
                best = detections[0]
                x1, y1, x2, y2 = best["xyxy"]
                bbox_area = (x2 - x1) * (y2 - y1)
                rz = _compute_rz(frame, best["xyxy"])

                if rz is not None:
                    print(
                        f"frame={frame_number}  "
                        f"label={best['label']}  conf={best['confidence']:.2f}  "
                        f"bbox_area={bbox_area:.0f}px²  "
                        f"rz={rz:.1f}deg"
                    )
                else:
                    print(
                        f"frame={frame_number}  "
                        f"label={best['label']}  conf={best['confidence']:.2f}  "
                        f"bbox_area={bbox_area:.0f}px²  "
                        f"rz=no_contour"
                    )
            else:
                target_desc = ", ".join(target_classes) if target_classes else "any object"
                print(f"frame={frame_number}  no detection for {target_desc}")

            _annotate(frame, detections, rz)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
