#!/usr/bin/env python3
"""ROS2 node: overhead camera object detection → robot pick pose publisher.

Detects objects in an overhead camera feed and publishes the best match as a
geometry_msgs/PoseStamped in the robot base frame, so a LeRobot arm can reach
and pick the object.

Pixel → world XY transform uses calibrated camera intrinsics (K matrix +
distortion coefficients loaded from overhead_camera.json) plus camera extrinsics
(position and rotation in world frame, sourced from MuJoCo via cam_pos /
cam_xmat parameters).  Each detected bounding-box centre is undistorted,
back-projected into a ray in the camera frame, converted from OpenCV convention
to MuJoCo convention (y and z flipped), rotated to world frame with cam_rot, and
intersected with the table plane (z = pick_z) to give a metric (x, y) coordinate
in the robot base frame.

Topics published
----------------
/pick_pose          geometry_msgs/PoseStamped   best pick target (gripper down)
/detected_class     std_msgs/String             label of the picked object
/detections_image   sensor_msgs/Image           annotated debug frame

Parameters
----------
camera_topic    str     "/camera/image_raw"   ROS image topic (set to "" to use camera_id)
camera_id       int     0                     OpenCV camera index when not using ROS topic
model           str     "yolo11m.pt"          detection model (see object_classification.py)
confidence      float   0.25                  minimum detection confidence
target_class    str     ""                    class name to pick; empty = highest-confidence object
intrinsics_path str     ""                    path to overhead_camera.json; empty = repo default
                                              (config/camera_intrinsics/overhead_camera.json)
cam_pos         list    [0.0, 0.0, 1.0]       camera position in world frame [x, y, z]
                                              set from MuJoCo data.cam_xpos[camera_id]
cam_xmat        list    identity              9 floats, row-major 3×3 camera rotation in world frame
                                              set from MuJoCo data.cam_xmat[camera_id] (after mj_forward)
pick_z          float   0.0                   table surface height in world frame (ray-plane intersection)
frame_id        str     "base_link"           world / robot base frame name
publish_hz      float   10.0                  max publish rate (frames are processed as fast as possible)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import Image
from std_msgs.msg import String

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
    OBJECTS365V2_CLASS_NAMES,
)

try:
    import cv2
    from cv_bridge import CvBridge
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]
    CvBridge = None  # type: ignore[assignment]

# Quaternion for gripper pointing straight down (180° around X in ROS convention).
_GRIPPER_DOWN = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)

# Default intrinsics JSON relative to the repo root (two levels above this file).
_DEFAULT_INTRINSICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "camera_intrinsics"
    / "overhead_camera.json"
)


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load K matrix, distortion coefficients, and calibration resolution from JSON.

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

    Pipeline (mirrors cube_detection.py / cube_pose_to_world):
      1. cv2.undistortPoints removes lens distortion and divides by focal length
         → normalised camera coordinates (x_n, y_n) with z_cv = 1.
      2. Flip to MuJoCo camera convention (x right, y up, z backward):
         ray_mj = [x_n, -y_n, -1].
      3. Rotate ray to world frame: ray_world = cam_rot @ ray_mj.
      4. Ray-plane intersection at z = pick_z:
         t = (pick_z - cam_pos[2]) / ray_world[2]
         (x, y) = cam_pos[:2] + t * ray_world[:2].
    """
    pts = np.array([[[u, v]]], dtype=np.float32)
    # undistortPoints with no P returns normalised undistorted coords.
    nd = cv2.undistortPoints(pts, K, dist)
    x_n, y_n = float(nd[0, 0, 0]), float(nd[0, 0, 1])

    # OpenCV ray [x_n, y_n, 1] → MuJoCo convention [x_n, -y_n, -1].
    ray_mj = np.array([x_n, -y_n, -1.0])

    # Rotate ray from MuJoCo camera frame to world frame.
    ray_world = cam_rot @ ray_mj

    if abs(ray_world[2]) < 1e-6:
        raise ValueError(
            f"Back-projected ray is parallel to the table plane for pixel ({u:.1f}, {v:.1f})."
        )

    t = (pick_z - cam_pos[2]) / ray_world[2]
    return float(cam_pos[0] + t * ray_world[0]), float(cam_pos[1] + t * ray_world[1])


def _collect_yolo_detections(
    results: list[Any],
    target_classes: list[str],
    class_name_lookup: list[str] | None,
) -> list[dict]:
    """Return list of {label, confidence, cx, cy, xyxy} from YOLO results."""
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
    """Return list of {label, confidence, cx, cy, xyxy} from RF-DETR detections."""
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


class PickPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("pick_pose_node")

        # --- declare parameters ---
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("model", "yolo11m.pt")
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("target_class", "")
        self.declare_parameter("intrinsics_path", "")
        # Camera position in world frame [x, y, z] — copy from MuJoCo
        # data.cam_xpos[camera_id] after mj_forward.
        self.declare_parameter("cam_pos", [0.0, 0.0, 1.0])
        # Camera rotation matrix in world frame — 9 floats, row-major 3×3.
        # Copy from MuJoCo data.cam_xmat[camera_id].flatten().tolist().
        self.declare_parameter(
            "cam_xmat",
            [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0],
        )
        self.declare_parameter("pick_z", 0.0)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("publish_hz", 10.0)

        # --- read parameters ---
        self._camera_topic: str = self.get_parameter("camera_topic").value
        self._camera_id: int = self.get_parameter("camera_id").value
        model_name: str = self.get_parameter("model").value
        self._confidence: float = self.get_parameter("confidence").value
        target_class_raw: str = self.get_parameter("target_class").value
        self._target_classes: list[str] = (
            [normalize_label(target_class_raw)] if target_class_raw.strip() else []
        )

        intrinsics_path_raw: str = self.get_parameter("intrinsics_path").value
        intrinsics_path = (
            Path(intrinsics_path_raw) if intrinsics_path_raw.strip() else _DEFAULT_INTRINSICS_PATH
        )
        if not intrinsics_path.is_file():
            raise RuntimeError(
                f"Camera intrinsics file not found: {intrinsics_path}\n"
                "Set the 'intrinsics_path' parameter or place overhead_camera.json at "
                "config/camera_intrinsics/overhead_camera.json."
            )
        self._K, self._dist, self._calib_w, self._calib_h = _load_intrinsics(intrinsics_path)
        self.get_logger().info(f"Loaded camera intrinsics from '{intrinsics_path}'.")

        # K scaled to the live frame resolution (set on first frame).
        self._K_scaled: np.ndarray | None = None
        self._frame_size: tuple[int, int] | None = None  # (width, height)

        cam_pos_flat: list[float] = self.get_parameter("cam_pos").value
        self._cam_pos: np.ndarray = np.array(cam_pos_flat, dtype=float)
        cam_xmat_flat: list[float] = self.get_parameter("cam_xmat").value
        self._cam_rot: np.ndarray = np.array(cam_xmat_flat, dtype=float).reshape(3, 3)

        self._pick_z: float = self.get_parameter("pick_z").value
        self._frame_id: str = self.get_parameter("frame_id").value
        publish_hz: float = self.get_parameter("publish_hz").value

        # --- load detection model ---
        configure_model_environment()
        self.get_logger().info(f"Loading model '{model_name}' …")
        self._loaded_model = load_detection_model(
            model_name,
            target_classes=self._target_classes or None,
        )
        model_type, _, _ = self._loaded_model
        self._model_type = model_type
        self.get_logger().info(f"Model loaded (type={model_type}).")

        # --- publishers ---
        self._pub_pose = self.create_publisher(PoseStamped, "/pick_pose", 10)
        self._pub_class = self.create_publisher(String, "/detected_class", 10)
        self._pub_image = self.create_publisher(Image, "/detections_image", 10)

        # --- camera source ---
        self._bridge = CvBridge() if CvBridge is not None else None
        self._cap = None  # OpenCV VideoCapture, used when not subscribing to a topic

        if self._camera_topic:
            self.create_subscription(Image, self._camera_topic, self._image_callback, 10)
            self.get_logger().info(f"Subscribing to camera topic '{self._camera_topic}'.")
        else:
            if cv2 is None:
                raise RuntimeError("cv2 not found; install opencv-python.")
            self._cap = cv2.VideoCapture(self._camera_id)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open camera {self._camera_id}.")
            self.get_logger().info(f"Opened local camera id={self._camera_id}.")
            interval = 1.0 / max(publish_hz, 1.0)
            self.create_timer(interval, self._timer_callback)

        self._last_publish = 0.0
        self._min_interval = 1.0 / max(publish_hz, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(self, pil_image: Any) -> list[dict]:
        """Run model on a PIL Image; return detections sorted best-first."""
        model_type, model, class_name_lookup = self._loaded_model

        if model_type == "rf-detr":
            raw = model.predict(pil_image, threshold=self._confidence)
            dets = _collect_rfdetr_detections(raw, self._target_classes, class_name_lookup)
        else:
            results = list(model.predict(source=pil_image, conf=self._confidence, verbose=False))
            dets = _collect_yolo_detections(results, self._target_classes, class_name_lookup)

        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return dets

    def _publish_pick(self, best: dict, stamp: Any) -> None:
        try:
            world_x, world_y = _pixel_to_world_xy(
                best["cx"], best["cy"],
                self._K_scaled,
                self._dist,
                self._cam_pos,
                self._cam_rot,
                self._pick_z,
            )
        except ValueError as exc:
            self.get_logger().warn(f"Skipping pick: {exc}")
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self._frame_id
        pose_msg.pose.position.x = world_x
        pose_msg.pose.position.y = world_y
        pose_msg.pose.position.z = self._pick_z
        pose_msg.pose.orientation = _GRIPPER_DOWN

        self._pub_pose.publish(pose_msg)

        class_msg = String()
        class_msg.data = best["label"]
        self._pub_class.publish(class_msg)

        self.get_logger().info(
            f"Pick → label={best['label']} conf={best['confidence']:.2f} "
            f"pixel=({best['cx']:.1f}, {best['cy']:.1f}) "
            f"world=({world_x:.4f}, {world_y:.4f}, {self._pick_z:.4f})"
        )

    def _annotate_and_publish_image(self, frame: Any, detections: list[dict], stamp: Any) -> None:
        if cv2 is None or self._bridge is None or frame is None:
            return
        for d in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
            color = (0, 255, 0) if d is detections[0] else (0, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{d['label']} {d['confidence']:.2f}"
            cv2.putText(frame, text, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        img_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = stamp
        self._pub_image.publish(img_msg)

    def _process_frame(self, frame: Any, stamp: Any) -> None:
        now = time.monotonic()
        if now - self._last_publish < self._min_interval:
            return
        self._last_publish = now

        # Scale K to the live frame resolution on first frame (or if it changes).
        frame_h, frame_w = frame.shape[:2]
        if self._frame_size != (frame_w, frame_h):
            self._frame_size = (frame_w, frame_h)
            self._K_scaled = _scale_K(self._K, self._calib_w, self._calib_h, frame_w, frame_h)
            self.get_logger().info(
                f"Frame resolution {frame_w}×{frame_h} "
                f"(calibration was {self._calib_w}×{self._calib_h}); K scaled."
            )

        pil_image = camera_frame_to_image(frame)
        detections = self._detect(pil_image)

        if detections:
            self._publish_pick(detections[0], stamp)
        else:
            label = (
                f"target: {', '.join(self._target_classes)}" if self._target_classes else "any object"
            )
            self.get_logger().debug(f"No detections for {label}.")

        self._annotate_and_publish_image(frame, detections, stamp)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        if self._bridge is None or cv2 is None:
            self.get_logger().error("cv_bridge or cv2 not available; cannot process ROS Image.")
            return
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._process_frame(frame, msg.header.stamp)

    def _timer_callback(self) -> None:
        ok, frame = self._cap.read()
        if not ok:
            self.get_logger().warn("Failed to read from camera.")
            return
        self._process_frame(frame, self.get_clock().now().to_msg())

    def destroy_node(self) -> None:
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PickPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
