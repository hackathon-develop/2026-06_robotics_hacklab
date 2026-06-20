# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compose a shared scene containing multiple namespaced SO-101 arms."""

from __future__ import annotations

import mujoco

from pick_and_place.arm_config import ArmConfig, DEFAULT_TWO_ARM_CONFIGS
from pick_and_place.builder import STOCK_ASSETS_DIR, build_robot
from pick_and_place.environment import (
    add_overhead_camera_mount,
    add_workspace_frame,
    add_workspace_frame_apriltags,
)
from pick_and_place.materials import MaterialConfig, apply_materials
from pick_and_place.scene import _add_groundplane, _add_pick_cube, _add_pick_cube_apriltags
from pick_and_place.workspace_overlays import add_workspace_overlays


def build_multiarm_scene(
    arms: tuple[ArmConfig, ...] | list[ArmConfig] = DEFAULT_TWO_ARM_CONFIGS,
    *,
    materials: MaterialConfig | None = None,
    include_environment: bool = True,
    apriltag_cube: bool | None = None,
) -> mujoco.MjSpec:
    """Return a scene with one shared environment and N prefixed robot arms.

    Each arm is loaded from the stock SO-101 model, placed by its ``ArmConfig``,
    and attached with ``MjSpec.attach(prefix=...)``. MuJoCo applies the prefix to
    bodies, joints, actuators, geoms, sites, cameras, assets, and their internal
    references, so multiple robot instances can coexist in one compiled model.
    """
    if apriltag_cube is None:
        apriltag_cube = include_environment

    spec = mujoco.MjSpec()
    spec.modelname = "so101_multiarm_with_cube"
    spec.meshdir = str(STOCK_ASSETS_DIR)
    spec.worldbody.add_light(
        name="scene_light",
        pos=(0.0, 0.0, 1.0),
        dir=(0.0, 0.0, -1.0),
    )

    add_workspace_overlays(spec, spec.worldbody)
    _add_pick_cube(spec, apriltag=apriltag_cube)

    if include_environment:
        add_workspace_frame(spec)
        add_overhead_camera_mount(spec)

    for arm in arms:
        robot = build_robot(wrist_camera=arm.wrist_camera, materials=materials)
        base = robot.body("base")
        base.pos = arm.base_pos
        base.quat = arm.base_quat
        frame = spec.worldbody.add_frame(name=f"{arm.name}_attach")
        spec.attach(robot, prefix=arm.prefix, frame=frame)

    apply_materials(spec, materials or MaterialConfig())
    if apriltag_cube:
        _add_pick_cube_apriltags(spec)
    if include_environment:
        add_workspace_frame_apriltags(spec)
    _add_groundplane(spec)
    return spec
