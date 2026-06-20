#!/usr/bin/env python3
"""ROS2 node: overhead camera object detection → robot pick pose publisher.

Detects objects in an overhead camera feed and publishes the best match as a
geometry_msgs/PoseStamped in the robot base frame, so a LeRobot arm can reach
and pick the object.

Pixel → robot XY transform uses a 3×3 homography matrix supplied via the
`homography` parameter (9 floats, row-major).  Z is a fixed table-height
parameter (`pick_z`).  Run camera–robot calibration once (e.g. with a
checkerboard or point-click tool) and paste the resulting matrix as a parameter.

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
homography      list    identity (pixels)     9 floats for the 3×3 pixel→robot-XY homography
pick_z          float   0.0                   fixed Z in robot base frame (table surface height)
frame_id        str     "base_link"           robot base frame name
publish_hz      float   10.0                  max publish rate (frames are processed as fast as possible)
"""

from __future__ import annotations

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


def _pixel_to_robot_xy(u: float, v: float, H: np.ndarray) -> tuple[float, float]:
    """Apply a 3×3 homography to map pixel (u, v) → robot (x, y)."""
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _collect_yolo_detections(
    results: list[Any],
    target_classes: list[str],
    class_name_lookup: list[str] | None,
) -> list[dict]:
    """Return list of {label, confidence, cx, cy} from YOLO results."""
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
    """Return list of {label, confidence, cx, cy} from RF-DETR detections."""
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
        # Row-major 3×3 homography (pixel → robot XY).  Default = identity so
        # outputs are raw pixel coordinates until the user calibrates.
        self.declare_parameter(
            "homography",
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
        h_flat: list[float] = self.get_parameter("homography").value
        self._H: np.ndarray = np.array(h_flat, dtype=float).reshape(3, 3)
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
            # Open camera directly and drive with a timer.
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
        """Run model on a PIL Image; return sorted detections (best first)."""
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
        robot_x, robot_y = _pixel_to_robot_xy(best["cx"], best["cy"], self._H)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self._frame_id
        pose_msg.pose.position.x = robot_x
        pose_msg.pose.position.y = robot_y
        pose_msg.pose.position.z = self._pick_z
        pose_msg.pose.orientation = _GRIPPER_DOWN

        self._pub_pose.publish(pose_msg)

        class_msg = String()
        class_msg.data = best["label"]
        self._pub_class.publish(class_msg)

        self.get_logger().info(
            f"Pick → label={best['label']} conf={best['confidence']:.2f} "
            f"pixel=({best['cx']:.1f}, {best['cy']:.1f}) "
            f"robot=({robot_x:.4f}, {robot_y:.4f}, {self._pick_z:.4f})"
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
