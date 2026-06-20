#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Smoke-test the multi-arm MuJoCo scene composition."""

from __future__ import annotations

import argparse

import mujoco

from pick_and_place.arm_config import names_for_arm
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.multiarm_scene import build_multiarm_scene


def _require_body(model: mujoco.MjModel, name: str) -> None:
    model.body(name)
    print(f"body   {name}: ok")


def _require_joint(model: mujoco.MjModel, name: str) -> None:
    model.joint(name)
    print(f"joint  {name}: ok")


def _require_camera(model: mujoco.MjModel, name: str) -> None:
    model.camera(name)
    print(f"camera {name}: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-environment",
        action="store_true",
        help="include workspace frame, overhead camera, and AprilTag textures",
    )
    args = parser.parse_args()

    print("building multi-arm scene...")
    model = build_multiarm_scene(
        include_environment=args.full_environment,
        apriltag_cube=args.full_environment,
    ).compile()
    print("compiled")
    print(f"nbody={model.nbody} njnt={model.njnt} nu={model.nu} ngeom={model.ngeom}")

    for arm_name in ("arm1", "arm2"):
        _require_body(model, f"{arm_name}_base")
        _require_body(model, f"{arm_name}_gripper")
        _require_joint(model, f"{arm_name}_shoulder_pan")
        _require_joint(model, f"{arm_name}_shoulder_lift")
        _require_joint(model, f"{arm_name}_elbow_flex")
        _require_joint(model, f"{arm_name}_wrist_flex")
        _require_joint(model, f"{arm_name}_wrist_roll")

    _require_body(model, "pick_cube")
    if args.full_environment:
        _require_camera(model, "overhead_camera")

    print("\nkinematics:")
    for arm_name in ("arm1", "arm2"):
        kinematics = derive_kinematics(model, names_for_arm(arm_name))
        limit = kinematics.joint_limits["shoulder_pan"]
        print(
            f"{arm_name}: pan_axis={kinematics.pan_axis} "
            f"shoulder_pan=[{limit.min:.3f}, {limit.max:.3f}]"
        )


if __name__ == "__main__":
    main()
