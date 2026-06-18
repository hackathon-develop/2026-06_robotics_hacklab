# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import numpy as np

import pytest

from pick_and_place.episodes import EpisodeSamplingError, _build_model, prepare_episode
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose, WORLD_UP
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.trajectory import (
    _carry_geometry_matrix,
    grasp_candidates,
    plan_carry_candidates,
)


def _drop_gripper_z(carry: object) -> np.ndarray:
    drop_cube = _carry_geometry_matrix(carry, 1.0)
    return drop_cube[:3, :3] @ carry.cube_from_gripper[:3, :3] @ np.array((0.0, 0.0, 1.0))


def test_vertical_zone_prefers_near_vertical_drops():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    target = CubePose(x=0.20, y=-0.05, z=CUBE_HALF_SIZE)
    model, _ = _build_model(source)
    kinematics = derive_kinematics(model)
    grasp = next(grasp_candidates(kinematics, source))

    carries = list(plan_carry_candidates(kinematics, grasp, source, target))

    assert carries
    drop_z_axes = [_drop_gripper_z(carry) for carry in carries]
    max_angle = np.deg2rad(15.0)
    assert any(not np.allclose(axis, WORLD_UP, atol=1e-7) for axis in drop_z_axes)
    for axis in drop_z_axes:
        assert float(np.dot(axis, WORLD_UP)) >= np.cos(max_angle) - 1e-7


def test_fixed_target_must_be_in_allowed_drop_zone():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    target_on_apriltag_exclusion = CubePose(x=0.10, y=0.20, z=CUBE_HALF_SIZE)

    with pytest.raises(EpisodeSamplingError, match="outside the allowed drop zone"):
        prepare_episode(
            np.random.default_rng(0),
            source,
            target_on_apriltag_exclusion,
            max_attempts=1,
        )
