# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Decision-making pipeline for multi-robot task assignment."""

from __future__ import annotations

import math
from typing import Literal, TypedDict


class RobotState(TypedDict):
    id: str
    position: tuple[float, float, float]
    status: str  # 'free' | 'busy'


def select_nearest_free_robot(
    robots: list[RobotState],
    object_pos: tuple[float, float, float],
) -> str | None:
    """Return the id of the free robot closest to ``object_pos``.

    Returns ``None`` if no robot is free.
    """
    ox, oy, oz = object_pos
    min_dist = float("inf")
    selected_id: str | None = None
    for robot in robots:
        if robot["status"] != "free":
            continue
        rx, ry, rz = robot["position"]
        dist = math.sqrt((rx - ox) ** 2 + (ry - oy) ** 2 + (rz - oz) ** 2)
        if dist < min_dist:
            min_dist = dist
            selected_id = robot["id"]
    return selected_id


def can_robot_reach_cube(
    robot: RobotState,
    cube_pos: tuple[float, float, float],
) -> bool:
    """Return whether ``robot`` can reach the floor cube at ``cube_pos``.

    Translates the cube into the robot's local frame (as if the robot sits at
    world origin) before delegating to the workspace pickup-allowed check.
    """
    from pick_and_place.workspace_overlays import is_cube_pickup_allowed

    cx, cy, _ = cube_pos
    rx, ry, _ = robot["position"]
    return is_cube_pickup_allowed(cx - rx, cy - ry)


def robot_states_for_scene(
    reference_side: Literal["left", "right"] = "left",
    *,
    status: str = "free",
) -> list[RobotState]:
    """Return world-frame RobotState dicts for both robots in the two-robot scene.

    The controlled robot (``reference_side``) is always placed at world origin by
    the scene builder; the passive robot sits at ``second_robot_offset_y`` along Y.
    Both are returned as ``status`` (default ``'free'``) so the caller can mark
    busy robots before passing the list to ``select_nearest_free_robot``.
    """
    from pick_and_place.scene import ROBOT_BASE_HEIGHT, second_robot_offset_y

    other_y = second_robot_offset_y(reference_side)
    h = ROBOT_BASE_HEIGHT
    if reference_side == "left":
        return [
            {"id": "left",  "position": (0.0, 0.0,     h), "status": status},
            {"id": "right", "position": (0.0, other_y, h), "status": status},
        ]
    return [
        {"id": "right", "position": (0.0, 0.0,     h), "status": status},
        {"id": "left",  "position": (0.0, other_y, h), "status": status},
    ]


def select_robot_for_cube(
    cube_pos: tuple[float, float, float],
    *,
    busy_ids: set[str] | None = None,
    reference_side: Literal["left", "right"] = "left",
) -> Literal["left", "right"] | None:
    """Pick the best free, reachable robot for a cube at ``cube_pos``.

    Pipeline:
    1. Occupancy    – exclude robots in ``busy_ids``.
    2. Reachability – exclude robots whose workspace does not cover the cube.
    3. Distance     – choose the closest remaining robot.
    4. Tie-break    – visual clarity / approach angle (not yet implemented).

    Returns ``'left'`` or ``'right'``, or ``None`` if no candidate remains.
    """
    robots = robot_states_for_scene(reference_side)
    if busy_ids:
        for r in robots:
            if r["id"] in busy_ids:
                r["status"] = "busy"

    free_robots = [r for r in robots if r["status"] == "free"]
    reachable = [r for r in free_robots if can_robot_reach_cube(r, cube_pos)]

    if not reachable:
        return None

    # Step 4 (TODO): if distances are within threshold, apply visual-clarity tie-breaker.
    chosen = select_nearest_free_robot(reachable, cube_pos)
    return chosen  # type: ignore[return-value]
