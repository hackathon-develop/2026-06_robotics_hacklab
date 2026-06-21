#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record SO-101 leader/follower teleoperation into a LeRobotDataset.

This is the manual-data counterpart to ``pick_and_place/real.py`` and
``pick_and_place/record_sim.py``. A physical SO-101 leader provides the per-tick
``action``. The follower is driven with that action, its encoder readback becomes
``observation.state``, and the wrist/overhead cameras are recorded at the same
control cadence. The resulting dataset uses the same feature schema as the
analytic trajectory recorders, so downstream conversion/training code can treat
teleop and analytic episodes the same way.

Episodes are operator-delimited: press Enter to start recording, press Enter
again to save that episode, or press ``r`` while recording to discard it and
retry the same episode number.
"""

from __future__ import annotations

import argparse
import datetime
import math
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

from pick_and_place import build_scene
from pick_and_place.cam_align_solve import parse_index_or_path
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.executor import (
    CONTROL_HZ,
    RecordingSession,
    clamp_and_warn,
    follower_clamp_limits,
)
from pick_and_place.follower import (
    action_to_joints,
    joints_to_action,
    make_so101_follower,
    make_so101_leader,
)
from pick_and_place.image_rectify import SQUARE_SIZE, build_undistort_map, transform_frame
from pick_and_place.kinematics import derive_kinematics


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


class CameraReader:
    """Background OpenCV reader with a single latest-frame buffer."""

    def __init__(
        self,
        source: str,
        *,
        label: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        import cv2

        backend = (
            cv2.CAP_AVFOUNDATION
            if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION")
            else cv2.CAP_ANY
        )
        camera = parse_index_or_path(source)
        self._cap = cv2.VideoCapture(camera, backend)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(camera)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open {label} camera {source!r}; check the OpenCV camera index "
                "with scripts/hackathon/show_cam_ids.py"
            )

        self._label = label
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, float(fps))
        actual_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = (
            float(actual_fps)
            if actual_fps and not math.isnan(actual_fps) and actual_fps > 0.0
            else float(fps)
        )
        print(
            f"{label} camera opened: requested {width}x{height}@{fps:g}, "
            f"got {actual_width:g}x{actual_height:g}@{actual_fps:g}"
        )
        if actual_fps and not math.isnan(actual_fps) and not math.isclose(
            actual_fps, fps, rel_tol=1e-2
        ):
            print(
                f"warning: {label} camera reports {actual_fps:g} fps, not requested "
                f"{fps:g}; recording still logs one latest frame per control tick"
            )

        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_for_first(self, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.001)
        raise RuntimeError(f"timed out waiting for the {self._label} camera stream")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


def _teleop_tick(
    leader, follower, previous: np.ndarray, low, high, warned
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror one leader sample to the follower; return ``(action, state)``."""
    leader_joints = action_to_joints(leader.get_action(), previous)
    commanded = clamp_and_warn(leader_joints, low, high, warned)
    follower.send_action(joints_to_action(commanded))
    measured = action_to_joints(follower.get_observation(), commanded)
    return commanded, measured


def _ramp_follower_to_leader(
    leader,
    follower,
    current: np.ndarray,
    low,
    high,
    warned,
    *,
    duration: float,
    fps: float,
) -> np.ndarray:
    """Smoothly bring the follower from its current pose onto the leader pose."""
    if duration <= 0.0:
        commanded, _ = _teleop_tick(leader, follower, current, low, high, warned)
        return commanded

    steps = max(1, round(duration * fps))
    period = 1.0 / fps
    target = action_to_joints(leader.get_action(), current)
    commanded = current
    for i in range(1, steps + 1):
        step_start = time.monotonic()
        target = action_to_joints(leader.get_action(), target)
        interp = current + _smoothstep(i / steps) * (target - current)
        commanded = clamp_and_warn(interp, low, high, warned)
        follower.send_action(joints_to_action(commanded))
        remaining = period - (time.monotonic() - step_start)
        if remaining > 0:
            time.sleep(remaining)
    return commanded


def _drain_or_fail(record_queue: queue.Queue) -> None:
    """Wait for the writer thread to consume all frames queued for an episode."""
    record_queue.join()


class StdinKeyReader:
    """Nonblocking single-key reader that restores terminal settings on exit."""

    def __init__(self) -> None:
        self._fd = None
        self._old_settings = None

    def __enter__(self) -> "StdinKeyReader":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *args) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read(self) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)


def _enter_pressed() -> bool:
    """Return true once the operator has pressed Enter without blocking teleop."""
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return False
    sys.stdin.readline()
    return True


def _nonempty_files(root: Path, pattern: str) -> list[Path]:
    return [path for path in root.glob(pattern) if path.is_file() and path.stat().st_size > 0]


def _verify_lerobot_layout(root: Path) -> None:
    """Fail loudly if the saved dataset is missing the LeRobot on-disk layout."""
    episode_meta = _nonempty_files(root, "meta/episodes/**/*.parquet")
    data_files = _nonempty_files(root, "data/**/*.parquet")
    video_files = _nonempty_files(root, "videos/**/*")
    missing = []
    if not (root / "meta" / "info.json").is_file():
        missing.append("meta/info.json")
    if not episode_meta:
        missing.append("meta/episodes/**/*.parquet")
    if not data_files:
        missing.append("data/**/*.parquet")
    if not video_files:
        missing.append("videos/**/*")

    if missing:
        raise RuntimeError(
            "LeRobot dataset layout is incomplete after finalize; missing "
            + ", ".join(missing)
            + f" under {root}"
        )
    print(
        "LeRobot layout OK: "
        f"{len(episode_meta)} episode metadata file(s), "
        f"{len(data_files)} data file(s), {len(video_files)} video file(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leader-port", required=True, help="serial port of the SO-101 leader"
    )
    parser.add_argument("--leader-id", default="liddy", help="leader calibration id")
    parser.add_argument(
        "--follower-port", required=True, help="serial port of the SO-101 follower"
    )
    parser.add_argument("--follower-id", default="folly", help="follower calibration id")
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera")
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera")
    parser.add_argument("--camera-width", type=int, default=1280, help="overhead capture width")
    parser.add_argument("--camera-height", type=int, default=720, help="overhead capture height")
    parser.add_argument("--wrist-width", type=int, default=1280, help="wrist capture width")
    parser.add_argument("--wrist-height", type=int, default=720, help="wrist capture height")
    parser.add_argument(
        "--camera-start-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for first frames after opening cameras (default: 10)",
    )
    parser.add_argument(
        "--camera-name",
        default="overhead_camera",
        help="intrinsics name for the overhead camera (default: overhead_camera)",
    )
    parser.add_argument(
        "--wrist-camera-name",
        default="wrist_camera",
        help="intrinsics name for the wrist camera (default: wrist_camera)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to record")
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=0.0,
        help="deprecated; episodes are now stopped manually by pressing Enter again",
    )
    parser.add_argument(
        "--reset-duration",
        type=float,
        default=0.0,
        help="deprecated; teleop now stays active until Enter starts the next episode",
    )
    parser.add_argument(
        "--ramp-duration",
        type=float,
        default=2.0,
        help="unrecorded smooth ramp from follower pose to leader pose before recording",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=CONTROL_HZ,
        help=f"control/recording rate in Hz (default: {CONTROL_HZ:g})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="output dir for the LeRobotDataset (default: py/datasets/<timestamp>)",
    )
    parser.add_argument(
        "--repo-id",
        default="local/pick-and-place-so101-teleop",
        help="dataset repo id stored in metadata",
    )
    parser.add_argument(
        "--task",
        default="Pick up the cube and place it at the target.",
        help="natural-language task instruction saved with every frame",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="LeRobot video codec (default: h264 for portable mp4 files)",
    )
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode video during capture; --no-streaming-encoding falls back to PNG-then-encode",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="background image-writer threads for PNG-then-encode mode",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="deprecated; manual Enter start/stop is always used",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.episode_duration < 0.0:
        parser.error("--episode-duration cannot be negative")
    if args.reset_duration < 0.0:
        parser.error("--reset-duration cannot be negative")
    if args.ramp_duration < 0.0:
        parser.error("--ramp-duration cannot be negative")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("--camera-width/--camera-height must be positive")
    if args.wrist_width <= 0 or args.wrist_height <= 0:
        parser.error("--wrist-width/--wrist-height must be positive")
    if args.camera_start_timeout <= 0.0:
        parser.error("--camera-start-timeout must be positive")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(__file__).resolve().parents[1] / "datasets" / timestamp
    )

    print("Building joint-limit model...")
    model = build_scene(include_environment=True).compile()
    kinematics = derive_kinematics(model)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()

    leader = None
    follower = None
    wrist = overhead = None
    recording = None
    record_queue: queue.Queue = queue.Queue()
    writer_error: list[BaseException] = []
    writer_done = threading.Event()
    wrist_undistort_map = None
    overhead_undistort_map = None

    def record_writer() -> None:
        import cv2

        try:
            while True:
                item = record_queue.get()
                try:
                    if item is None:
                        return
                    state, action, wrist_bgr, overhead_bgr = item
                    assert recording is not None
                    assert recording.dataset is not None
                    assert wrist_undistort_map is not None
                    assert overhead_undistort_map is not None
                    wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
                    overhead_rgb = cv2.cvtColor(overhead_bgr, cv2.COLOR_BGR2RGB)
                    recording.dataset.add_frame(
                        {
                            "observation.state": state,
                            "action": action,
                            "observation.images.wrist": transform_frame(
                                wrist_rgb, wrist_undistort_map, SQUARE_SIZE, cv2
                            ),
                            "observation.images.overhead": transform_frame(
                                overhead_rgb, overhead_undistort_map, SQUARE_SIZE, cv2
                            ),
                            "task": recording.task,
                        }
                    )
                finally:
                    record_queue.task_done()
        except BaseException as exc:  # noqa: BLE001 - surface writer failures in main thread
            writer_error.append(exc)
        finally:
            writer_done.set()

    writer_thread = None
    current = np.zeros(6, dtype=float)
    current_episode_has_frames = False
    interrupted = False
    layout_error = None
    saved_episodes = 0
    try:
        print(f"Connecting to leader on {args.leader_port}...")
        leader = make_so101_leader(args.leader_port, args.leader_id)
        leader.connect(calibrate=True)

        print(f"Connecting to follower on {args.follower_port}...")
        follower = make_so101_follower(
            args.follower_port,
            args.follower_id,
            disable_torque_on_disconnect=False,
        )
        follower.connect(calibrate=True)

        print("Opening cameras...")
        wrist = CameraReader(
            args.wrist_camera,
            label="wrist",
            width=args.wrist_width,
            height=args.wrist_height,
            fps=args.fps,
        )
        effective_fps = wrist.fps
        print(f"Recording/control FPS set from wrist camera: {effective_fps:g}")
        overhead = CameraReader(
            args.camera,
            label="overhead",
            width=args.camera_width,
            height=args.camera_height,
            fps=effective_fps,
        )
        if not math.isclose(overhead.fps, effective_fps, rel_tol=1e-2):
            print(
                f"warning: overhead camera reports {overhead.fps:g} fps while wrist reports "
                f"{effective_fps:g}; overhead frames will be sampled onto the wrist clock"
            )
        print(f"Waiting up to {args.camera_start_timeout:g}s for camera frames...")
        first_overhead = overhead.wait_for_first(args.camera_start_timeout)
        first_wrist = wrist.wait_for_first(args.camera_start_timeout)

        import cv2

        intrinsics_by_camera = load_local_camera_intrinsics()
        missing = [
            name
            for name in (args.camera_name, args.wrist_camera_name)
            if name not in intrinsics_by_camera
        ]
        if missing:
            raise RuntimeError(
                f"missing calibrated intrinsics for {missing}; cannot record rectified images"
            )
        overhead_undistort_map = build_undistort_map(
            intrinsics_by_camera[args.camera_name],
            first_overhead.shape[1],
            first_overhead.shape[0],
            cv2,
        )
        wrist_undistort_map = build_undistort_map(
            intrinsics_by_camera[args.wrist_camera_name],
            first_wrist.shape[1],
            first_wrist.shape[0],
            cv2,
        )

        image_shape = (SQUARE_SIZE, SQUARE_SIZE, 3)
        recording = RecordingSession(
            repo_id=args.repo_id,
            root=dataset_root,
            task=args.task,
            fps=effective_fps,
            vcodec=args.vcodec,
            streaming_encoding=args.streaming_encoding,
            image_writer_threads=args.image_writer_threads,
        )
        recording.create_dataset(image_shape, image_shape)
        writer_thread = threading.Thread(target=record_writer, daemon=True)
        writer_thread.start()

        current = action_to_joints(follower.get_observation(), current)
        print("Ramping follower onto leader pose...")
        current = _ramp_follower_to_leader(
            leader,
            follower,
            current,
            clamp_low,
            clamp_high,
            clip_warned,
            duration=args.ramp_duration,
            fps=effective_fps,
        )

        period = 1.0 / effective_fps
        recording_active = False
        episode_start = 0.0
        frame_count = 0
        next_tick = time.monotonic()
        print(
            "Teleop active. Press Enter to start recording "
            f"episode 1/{args.episodes}."
        )
        with StdinKeyReader() as keys:
            while saved_episodes < args.episodes:
                key = keys.read()
                if key in ("\n", "\r"):
                    if not recording_active:
                        recording_active = True
                        current_episode_has_frames = True
                        frame_count = 0
                        episode_start = time.monotonic()
                        episode_number = saved_episodes + 1
                        print(
                            f"Recording episode {episode_number}/{args.episodes}. "
                            "Press Enter to save, or r to discard and retry."
                        )
                    else:
                        print("Stopping recording and saving episode...")
                        _drain_or_fail(record_queue)
                        if writer_error:
                            raise RuntimeError("record writer failed") from writer_error[0]
                        if frame_count == 0:
                            if recording.dataset.has_pending_frames():
                                recording.dataset.clear_episode_buffer()
                            current_episode_has_frames = False
                            recording_active = False
                            print(
                                "No frames captured; episode was not saved. "
                                "Press Enter to retry."
                            )
                        else:
                            dropped = recording.dropped_frame_count()
                            if dropped:
                                raise RuntimeError(
                                    f"Streaming video encoder dropped {dropped} frame(s); "
                                    "refusing to save a desynchronized episode. Try "
                                    "--no-streaming-encoding or another --vcodec."
                                )
                            recording.dataset.save_episode()
                            current_episode_has_frames = False
                            recording_active = False
                            saved_episodes += 1
                            duration = time.monotonic() - episode_start
                            print(
                                f"Saved episode {saved_episodes}/{args.episodes}: "
                                f"{frame_count} frames over {duration:.1f}s."
                            )
                            if saved_episodes < args.episodes:
                                print(
                                    "Teleop active. Press Enter to start recording "
                                    f"episode {saved_episodes + 1}/{args.episodes}."
                                )
                    next_tick = time.monotonic()
                    continue
                if key is not None and key.lower() == "r":
                    if recording_active:
                        print("Discarding current episode. Teleop stays active...")
                        _drain_or_fail(record_queue)
                        if writer_error:
                            raise RuntimeError("record writer failed") from writer_error[0]
                        if recording.dataset.has_pending_frames():
                            recording.dataset.clear_episode_buffer()
                        current_episode_has_frames = False
                        recording_active = False
                        frame_count = 0
                        print(
                            "Discarded episode. Press Enter to record "
                            f"episode {saved_episodes + 1}/{args.episodes} again."
                        )
                    else:
                        print("Not recording. Press Enter to start an episode.")
                    next_tick = time.monotonic()
                    continue

                current, actual = _teleop_tick(
                    leader, follower, current, clamp_low, clamp_high, clip_warned
                )
                if recording_active:
                    wrist_frame = wrist.latest()
                    overhead_frame = overhead.latest()
                    if wrist_frame is not None and overhead_frame is not None:
                        record_queue.put(
                            (
                                actual.astype(np.float32),
                                current.astype(np.float32),
                                wrist_frame,
                                overhead_frame,
                            )
                        )
                        frame_count += 1

                if writer_error:
                    raise RuntimeError("record writer failed") from writer_error[0]

                next_tick += period
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -period:
                    next_tick = time.monotonic()
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted.")
    finally:
        if writer_thread is not None and not writer_error:
            record_queue.put(None)
            record_queue.join()
            writer_done.wait(timeout=30.0)
            writer_thread.join(timeout=1.0)
        elif writer_thread is not None:
            writer_done.wait(timeout=1.0)
            writer_thread.join(timeout=1.0)
        if recording is not None and recording.dataset is not None:
            if current_episode_has_frames and recording.dataset.has_pending_frames():
                recording.dataset.clear_episode_buffer()
                reason = "interrupted" if interrupted else "failed"
                print(f"Discarded {reason} partial episode.")
            print("Finalizing dataset...")
            recording.finalize()
            print(f"Dataset written to {dataset_root}")
            if saved_episodes > 0:
                try:
                    _verify_lerobot_layout(dataset_root)
                except RuntimeError as exc:
                    layout_error = exc
        if wrist is not None:
            wrist.close()
        if overhead is not None:
            overhead.close()
        print("Disconnecting hardware...")
        if follower is not None:
            follower.disconnect()
        if leader is not None:
            leader.disconnect()
        if layout_error is not None:
            raise layout_error


if __name__ == "__main__":
    main()
