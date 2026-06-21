#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD
"""Move the physical SO-101 to a detected object's hover pose."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_SRC = REPO_ROOT / "py" / "src"
sys.path.insert(0, str(PY_SRC))

ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_INDEX = 5


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def vertical_hover_matrix(
    x: float,
    y: float,
    z: float,
    rz_deg: float = 0.0,
    *,
    z_reference: str = "gripper_body",
) -> np.ndarray:
    """Build a world-from-gripper matrix for the requested hover pose.

    By default, ``z`` is the gripper body/end-effector frame height. Use
    ``z_reference='ik_target'`` to treat ``z`` as the repo's internal jaw/tip IK
    target height instead.
    """
    from pick_and_place import transforms as tf
    from pick_and_place.geometry import GRIPPER_TARGET_POSITION

    yaw = math.radians(rz_deg)
    x_axis = np.array((math.cos(yaw), math.sin(yaw), 0.0), dtype=float)
    z_axis = np.array((0.0, 0.0, 1.0), dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    matrix = tf.make_basis(x_axis, y_axis, z_axis)
    requested = np.array((x, y, z), dtype=float)
    if z_reference == "ik_target":
        matrix[:3, 3] = requested - matrix[:3, :3] @ GRIPPER_TARGET_POSITION
    elif z_reference == "gripper_body":
        matrix[:3, 3] = requested
    else:
        raise ValueError("z_reference must be 'gripper_body' or 'ik_target'")
    return matrix


def _select_branch(
    branches: list[Any],
    current_sim_joints: dict[str, float] | None,
) -> Any:
    if not branches:
        raise ValueError("hover pose is unreachable: no IK branch within joint limits")
    if current_sim_joints is None:
        return branches[0]
    return min(
        branches,
        key=lambda branch: sum(
            abs(branch.joints[name] - current_sim_joints[name]) for name in ARM_JOINT_NAMES
        ),
    )


def solve_hover_joints(
    kinematics: Any,
    hover_pose: dict[str, Any],
    current_sim_joints: dict[str, float] | None = None,
) -> Any:
    from pick_and_place.ik import solve_simple_pregrasp_ik

    matrix = vertical_hover_matrix(
        float(hover_pose["x"]),
        float(hover_pose["y"]),
        float(hover_pose["z"]),
        float(hover_pose.get("rz", 0.0)),
        z_reference=str(hover_pose.get("z_reference", "gripper_body")),
    )
    branches = solve_simple_pregrasp_ik(kinematics, matrix)
    return _select_branch(branches, current_sim_joints)


def _ramp_follower(
    follower,
    start_real: np.ndarray,
    target_real: np.ndarray,
    *,
    duration: float,
    control_hz: float,
    max_joint_speed: float,
) -> None:
    from pick_and_place.follower import joints_to_action

    delta = target_real - start_real
    arm_travel = float(np.max(np.abs(delta[:GRIPPER_INDEX]))) if GRIPPER_INDEX else 0.0
    speed_duration = arm_travel / max_joint_speed if max_joint_speed > 0 else 0.0
    duration = max(float(duration), speed_duration)
    steps = max(1, round(duration * control_hz))
    period = 1.0 / control_hz
    for index in range(1, steps + 1):
        tick_start = time.monotonic()
        target = start_real + _smoothstep(index / steps) * delta
        follower.send_action(joints_to_action(target))
        remaining = period - (time.monotonic() - tick_start)
        if remaining > 0:
            time.sleep(remaining)


def move_to_hover_payload(
    payload: dict[str, Any],
    *,
    follower_port: str = "",
    follower_id: str = "folly",
    offsets_path: str | Path | None = None,
    duration: float = 2.0,
    control_hz: float = 30.0,
    max_joint_speed: float = 10.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Move through detected hover poses, returning home between each one."""
    from pick_and_place import build_scene
    from pick_and_place.executor import clamp_and_warn, follower_clamp_limits
    from pick_and_place.follower import (
        action_to_joints,
        load_follower_joint_offsets,
        make_so101_follower,
        real_frame_to_sim,
        sim_frame_to_real,
    )
    from pick_and_place.kinematics import derive_kinematics
    from pick_and_place.trajectory import GRIPPER_OPEN, NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER

    hover_poses = payload.get("arm_hover_poses")
    if hover_poses is None:
        hover_pose = payload.get("arm_hover_pose")
        if not isinstance(hover_pose, dict):
            raise ValueError("payload must contain arm_hover_pose or arm_hover_poses")
        hover_poses = [hover_pose]
    if not isinstance(hover_poses, list) or not all(isinstance(pose, dict) for pose in hover_poses):
        raise ValueError("arm_hover_poses must be a list of objects")
    if not dry_run and not follower_port:
        raise ValueError("follower_port is required unless dry_run is enabled")

    model = build_scene(include_environment=False).compile()
    kinematics = derive_kinematics(model)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    offsets = load_follower_joint_offsets(offsets_path)
    clip_warned: set[str] = set()
    home_real = clamp_and_warn(
        sim_frame_to_real(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, offsets),
        clamp_low,
        clamp_high,
        clip_warned,
    )

    follower = None
    current_real = None
    if not dry_run:
        follower = make_so101_follower(
            follower_port,
            follower_id,
            disable_torque_on_disconnect=False,
        )
        follower.connect()
        current_real = action_to_joints(follower.get_observation(), clamp_low)

    home_sim_joints, _ = real_frame_to_sim(home_real, offsets)
    targets = []
    for index, hover_pose in enumerate(hover_poses, start=1):
        branch = solve_hover_joints(kinematics, hover_pose, home_sim_joints)
        target_real = clamp_and_warn(
            sim_frame_to_real(branch.joints, GRIPPER_OPEN, offsets),
            clamp_low,
            clamp_high,
            clip_warned,
        )
        targets.append(
            {
                "index": index,
                "label": hover_pose.get("label"),
                "confidence": hover_pose.get("confidence"),
                "hover_pose": hover_pose,
                "ik_elbow": branch.elbow,
                "target_sim_joints_rad": {
                    name: float(value) for name, value in branch.joints.items()
                },
                "target_real": target_real.tolist(),
                "_target_real_array": target_real,
            }
        )

    diagnostics = {
        "home_real": home_real.tolist(),
        "num_targets": len(targets),
        "targets": [
            {key: value for key, value in target.items() if key != "_target_real_array"}
            for target in targets
        ],
    }

    if dry_run:
        return diagnostics

    assert follower is not None and current_real is not None
    try:
        print("[arm] homing before object sequence", file=sys.stderr)
        _ramp_follower(
            follower,
            current_real,
            home_real,
            duration=duration,
            control_hz=control_hz,
            max_joint_speed=max_joint_speed,
        )
        current_real = home_real
        for target in targets:
            hover_pose = target["hover_pose"]
            target_real = target["_target_real_array"]
            print(
                "[arm] moving to object "
                f"#{target['index']} label={target['label']!r} conf={target['confidence']} "
                f"x={float(hover_pose['x']):.3f} y={float(hover_pose['y']):.3f} "
                f"z={float(hover_pose['z']):.3f} elbow={target['ik_elbow']}",
                file=sys.stderr,
            )
            _ramp_follower(
                follower,
                current_real,
                target_real,
                duration=duration,
                control_hz=control_hz,
                max_joint_speed=max_joint_speed,
            )
            print(f"[arm] returning home after object #{target['index']}", file=sys.stderr)
            _ramp_follower(
                follower,
                target_real,
                home_real,
                duration=duration,
                control_hz=control_hz,
                max_joint_speed=max_joint_speed,
            )
            current_real = home_real
    finally:
        if follower is not None:
            try:
                measured = action_to_joints(follower.get_observation(), home_real)
                print("[arm] final homing before disconnect", file=sys.stderr)
                _ramp_follower(
                    follower,
                    measured,
                    home_real,
                    duration=duration,
                    control_hz=control_hz,
                    max_joint_speed=max_joint_speed,
                )
            finally:
                follower.disconnect()
    return diagnostics


def _load_payload(path: str) -> dict[str, Any]:
    if path == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the real arm to a first-sequence hover pose.")
    parser.add_argument("payload", help="Path to first_pick_sequence JSON, or '-' for stdin.")
    parser.add_argument("--follower-port", default="", help="Serial port of the SO-101 follower.")
    parser.add_argument("--follower-id", default="folly", help="Follower calibration id.")
    parser.add_argument("--offsets-path", default=None, help="Optional sim->real joint offsets JSON.")
    parser.add_argument("--duration", type=float, default=2.0, help="Minimum ramp duration in seconds.")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Command streaming rate.")
    parser.add_argument("--max-joint-speed", type=float, default=10.0, help="Arm joint speed cap in deg/s.")
    parser.add_argument("--dry-run", action="store_true", help="Solve and print targets without moving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = move_to_hover_payload(
        _load_payload(args.payload),
        follower_port=args.follower_port,
        follower_id=args.follower_id,
        offsets_path=args.offsets_path,
        duration=args.duration,
        control_hz=args.control_hz,
        max_joint_speed=args.max_joint_speed,
        dry_run=args.dry_run,
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
