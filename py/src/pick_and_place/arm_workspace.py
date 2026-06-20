# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Per-arm reachability helpers for shared-world task selection."""

from __future__ import annotations

import math

import numpy as np

from pick_and_place.geometry import CubePose as PlanningCubePose
from pick_and_place.kinematics import So101Kinematics
from pick_and_place.trajectory import grasp_candidates
from pick_and_place.workspace_overlays import CUBE_PLACEMENT_OVERLAY, WORKSPACE_OVERLAYS


def _angle_in_limits(angle: float, lower: float, upper: float) -> bool:
    return lower <= angle <= upper


def rough_xy_allowed(
    kinematics: So101Kinematics,
    xy: tuple[float, float] | np.ndarray,
    *,
    inner_radius: float,
    outer_radius: float,
) -> bool:
    """Fast arm-local floor reach check for a world XY point."""
    x, y = float(xy[0]), float(xy[1])
    dx = x - float(kinematics.pan_axis[0])
    dy = y - float(kinematics.pan_axis[1])
    radius = math.hypot(dx, dy)
    if not inner_radius <= radius <= outer_radius:
        return False

    # ``tip_position`` uses azimuth = -shoulder_pan, so invert the world angle.
    shoulder_pan = -math.atan2(dy, dx)
    limit = kinematics.joint_limits["shoulder_pan"]
    return _angle_in_limits(shoulder_pan, limit.min, limit.max)


def rough_pickup_allowed(kinematics: So101Kinematics, xy: tuple[float, float] | np.ndarray) -> bool:
    """Return whether ``xy`` is inside this arm's rough floor-pickup workspace."""
    overlay = WORKSPACE_OVERLAYS[-1]
    return rough_xy_allowed(
        kinematics,
        xy,
        inner_radius=overlay.inner_radius,
        outer_radius=overlay.outer_radius,
    )


def rough_drop_allowed(kinematics: So101Kinematics, xy: tuple[float, float] | np.ndarray) -> bool:
    """Return whether ``xy`` is inside this arm's rough drop workspace."""
    return rough_xy_allowed(
        kinematics,
        xy,
        inner_radius=CUBE_PLACEMENT_OVERLAY.inner_radius,
        outer_radius=CUBE_PLACEMENT_OVERLAY.outer_radius,
    )


def _yaw_from_rotation(rotation: np.ndarray) -> float:
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def planning_cube_pose(position: np.ndarray, rotation: np.ndarray | None = None) -> PlanningCubePose:
    """Convert a tracked world pose into the planner's compact cube pose."""
    yaw = 0.0 if rotation is None else _yaw_from_rotation(np.asarray(rotation, dtype=float))
    return PlanningCubePose(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        yaw=yaw,
    )


def arm_can_pick(
    kinematics: So101Kinematics,
    position: np.ndarray,
    rotation: np.ndarray | None = None,
) -> bool:
    """Return whether this arm has at least one IK-valid grasp candidate."""
    if not rough_pickup_allowed(kinematics, position[:2]):
        return False
    cube = planning_cube_pose(position, rotation)
    return next(grasp_candidates(kinematics, cube), None) is not None
