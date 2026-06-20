#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Open or export the perimeter-based multi-arm SO-101 scene.

The default two-arm placement follows ``perimeter_scene/perimeter_plates.xml``:
two north-side arm mounts at +/-0.116 m around the original single-arm base.
Toggle geom group 4 in the viewer (key '4') to show or hide workspace overlays.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import mujoco
import mujoco.viewer

from pick_and_place.arm_config import ArmConfig, DEFAULT_TWO_ARM_CONFIGS
from pick_and_place.multiarm_scene import build_multiarm_scene


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


def _with_wrist_camera(arms: tuple[ArmConfig, ...], wrist_camera: bool) -> tuple[ArmConfig, ...]:
    return tuple(replace(arm, wrist_camera=wrist_camera) for arm in arms)


def _export_multiarm_scene(
    output: Path,
    *,
    arms: tuple[ArmConfig, ...],
    include_environment: bool,
    apriltag_cube: bool | None,
) -> Path:
    spec = build_multiarm_scene(
        arms,
        include_environment=include_environment,
        apriltag_cube=apriltag_cube,
    )
    spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    return output


def _set_wrist_roll_neutral(model: mujoco.MjModel, data: mujoco.MjData, arms: tuple[ArmConfig, ...]) -> None:
    # Match view_scene.py: compensate for the physical 2.8 deg arm twist.
    wrist_roll = math.radians(2.8 - 90)
    for arm in arms:
        joint_name = f"{arm.name}_wrist_roll"
        try:
            data.joint(joint_name).qpos = wrist_roll
            data.actuator(joint_name).ctrl = wrist_roll
        except KeyError:
            print(f"Warning: skipping missing wrist roll joint/actuator {joint_name!r}")
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        help="arm placement as name,x,y,yaw_deg; repeat for 2-4 arms",
    )
    parser.add_argument(
        "--wrist-camera",
        action="store_true",
        help="include wrist-camera mounts/modules on every arm",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="XML",
        help="write the composed scene to XML before opening the viewer",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="write the XML without opening the viewer (requires --export)",
    )
    parser.add_argument(
        "--environment",
        action="store_true",
        help="include the workspace frame and overhead camera mount in the scene",
    )
    parser.add_argument(
        "--apriltag-cube",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "use the AprilTag-stickered pick cube instead of the plain red cube; "
            "defaults to on with --environment, off otherwise"
        ),
    )
    args = parser.parse_args()

    if args.export_only and args.export is None:
        parser.error("--export-only requires --export")

    arms = tuple(args.arm) if args.arm else DEFAULT_TWO_ARM_CONFIGS
    arms = _with_wrist_camera(arms, args.wrist_camera)

    if args.export is not None:
        output = _export_multiarm_scene(
            args.export,
            arms=arms,
            include_environment=args.environment,
            apriltag_cube=args.apriltag_cube,
        )
        print(f"Wrote {output}")

    if not args.export_only:
        model = build_multiarm_scene(
            arms,
            include_environment=args.environment,
            apriltag_cube=args.apriltag_cube,
        ).compile()
        data = mujoco.MjData(model)
        _set_wrist_roll_neutral(model, data, arms)
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
