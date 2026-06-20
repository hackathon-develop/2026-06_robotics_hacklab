# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Configuration and MuJoCo names for one SO-101 arm instance."""

from __future__ import annotations

from dataclasses import dataclass


ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


@dataclass(frozen=True)
class ArmConfig:
    """Placement/configuration for one arm in a shared world frame."""

    name: str
    base_pos: tuple[float, float, float]
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    wrist_camera: bool = False

    @property
    def prefix(self) -> str:
        """Prefix applied to all MuJoCo object names for this arm."""
        return f"{self.name}_"


@dataclass(frozen=True)
class ArmNames:
    """Namespaced MuJoCo object names for one arm instance."""

    prefix: str = ""

    @property
    def base(self) -> str:
        return self.prefix + "base"

    @property
    def shoulder(self) -> str:
        return self.prefix + "shoulder"

    @property
    def gripper(self) -> str:
        return self.prefix + "gripper"

    @property
    def wrist_camera(self) -> str:
        return self.prefix + "wrist_camera"

    @property
    def shoulder_pan(self) -> str:
        return self.prefix + "shoulder_pan"

    @property
    def shoulder_lift(self) -> str:
        return self.prefix + "shoulder_lift"

    @property
    def elbow_flex(self) -> str:
        return self.prefix + "elbow_flex"

    @property
    def wrist_flex(self) -> str:
        return self.prefix + "wrist_flex"

    @property
    def wrist_roll(self) -> str:
        return self.prefix + "wrist_roll"

    @property
    def joints(self) -> dict[str, str]:
        """Logical joint name -> MuJoCo joint name."""
        return {logical: self.prefix + logical for logical in ARM_JOINT_NAMES}


def names_for_arm(name: str) -> ArmNames:
    """Return names for an arm configured with ``ArmConfig(name=...)``."""
    return ArmNames(prefix=f"{name}_")


# Match perimeter_scene/perimeter_plates.xml north_02/north_04 arm-base mounts.
# In the current world frame those frame-local +/-0.116 m x offsets map to world
# y offsets around the original single-arm base position. Both mounts share the
# same printed-base orientation, so neither arm is yaw-flipped by default.
DEFAULT_TWO_ARM_CONFIGS = (
    ArmConfig(name="arm1", base_pos=(0.0, -0.116, 0.0072)),
    ArmConfig(name="arm2", base_pos=(0.0, 0.116, 0.0072)),
)
