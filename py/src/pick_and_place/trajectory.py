# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place trajectory, built phase by phase.

This drives the MuJoCo model through position actuators under real physics, so
each phase emits joint *set points* that a servo tracks rather than poses written
straight to ``qpos``.

Implemented so far: phase 1, neutral -> hover above the source cube.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pick_and_place.geometry import CubePose, VERTICAL_FACES, pregrasp_matrix
from pick_and_place.ik import IkBranch, solve_simple_pregrasp_ik
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics

# Tip-contact height of the source hover above the floor (clears the 3 cm cube
# top by 1 cm). At the grasp the tip sits at the cube-center height, so the
# world-z offset applied to the pregrasp is ``tip_z - pose.z``.
SOURCE_HOVER_TIP_Z = 0.04

# Gripper joint angle at the hover pregrasp: 40 deg open.
GRIPPER_OPEN = math.radians(40.0)

NEUTRAL_ARM_JOINTS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": -math.pi / 2,
}
NEUTRAL_GRIPPER = 0.0

# Phase 1: neutral -> hover pregrasp above the source cube.
STAGE1_DURATION = 2.0


@dataclass(frozen=True)
class Frame:
    """One trajectory sample: arm joint set points plus the gripper set point."""

    joints: dict[str, float]
    gripper: float


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def _lerp_joints(a: dict[str, float], b: dict[str, float], alpha: float) -> dict[str, float]:
    return {name: a[name] + (b[name] - a[name]) * alpha for name in ARM_JOINT_NAMES}


def select_hover(k: So101Kinematics, source: CubePose) -> tuple[str, IkBranch]:
    """Pick the grasp face and hover branch for the source cube.

    Simplified placeholder: tries the vertical faces in preference order and
    takes the first reachable hover, preferring the elbow-up branch. The full
    selection (which must also satisfy the carry and drop) arrives with the
    later phases.
    """
    hover_offset = SOURCE_HOVER_TIP_Z - source.z
    for face in VERTICAL_FACES:
        matrix = pregrasp_matrix(face, source, hover_offset)
        if matrix is None:
            continue
        branches = solve_simple_pregrasp_ik(k, matrix)
        if not branches:
            continue
        branch = next((b for b in branches if b.elbow == "up"), branches[0])
        return face, branch
    raise ValueError("No reachable hover pose for the source cube")


@dataclass(frozen=True)
class HoverApproach:
    """Phase 1 trajectory: neutral -> hover, holding the hover at the end."""

    hover_joints: dict[str, float]
    duration: float = STAGE1_DURATION

    def evaluate(self, t: float) -> Frame:
        alpha = _smoothstep(t / self.duration) if self.duration > 0 else 1.0
        joints = _lerp_joints(NEUTRAL_ARM_JOINTS, self.hover_joints, alpha)
        gripper = NEUTRAL_GRIPPER + (GRIPPER_OPEN - NEUTRAL_GRIPPER) * alpha
        return Frame(joints=joints, gripper=gripper)


def hover_approach(k: So101Kinematics, source: CubePose) -> HoverApproach:
    _, branch = select_hover(k, source)
    return HoverApproach(hover_joints=branch.joints)
