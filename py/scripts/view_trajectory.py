#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the SO-101 through the pick-and-place trajectory under real physics.

The arm is controlled through the model's position-servo actuators: each frame
the trajectory's joint set points are written to ``data.ctrl`` and the simulation
is stepped, so gravity and contact are live. The cube gets a free joint and rests
on the floor as a genuine rigid body.

Phases 1-2 so far: neutral -> hover -> pregrasp at the source cube center.
"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.trajectory import NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, pick_approach
from pick_and_place.workspace_overlays import (
    AZIMUTH_MAX,
    AZIMUTH_MIN,
    PAN_AXIS,
    WORKSPACE_OVERLAYS,
)

_CLEARANCE_OVERLAY = next(o for o in WORKSPACE_OVERLAYS if o.name == "workspace_clearance_pregrasp")


def _random_source() -> CubePose:
    """Sample a cube pose uniformly inside the clearance-pregrasp annular sector."""
    r_inner, r_outer = _CLEARANCE_OVERLAY.inner_radius, _CLEARANCE_OVERLAY.outer_radius
    # Uniform area sampling: draw r² uniformly so density is flat in 2-D.
    r = math.sqrt(np.random.uniform(r_inner**2, r_outer**2))
    theta = np.random.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
    yaw = np.random.uniform(0.0, 2 * math.pi)
    return CubePose(
        x=PAN_AXIS[0] + r * math.cos(theta),
        y=PAN_AXIS[1] + r * math.sin(theta),
        z=CUBE_HALF_SIZE,
        yaw=yaw,
    )


def _set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="source cube (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    args = parser.parse_args()

    if args.source is not None:
        source = CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
    else:
        source = _random_source()
        print(f"cube: x={source.x:.4f}  y={source.y:.4f}  yaw={math.degrees(source.yaw):.1f}°")

    spec = build_scene()
    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    half_yaw = source.yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()  # make the cube a real dynamic body
    model = spec.compile()
    data = mujoco.MjData(model)

    kinematics = derive_kinematics(model)
    trajectory = pick_approach(kinematics, source)
    g = trajectory.grasp
    print(f"grasp: face={g.face}  elbow={g.elbow}")

    # Start at the neutral pose, gripper closed, holding via the servos.
    for name, value in NEUTRAL_ARM_JOINTS.items():
        _set_joint(model, data, name, value)
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }
    for name, value in NEUTRAL_ARM_JOINTS.items():
        data.ctrl[actuator_id[name]] = value
    data.ctrl[actuator_id["gripper"]] = NEUTRAL_GRIPPER
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            frame = trajectory.evaluate(data.time)
            for name, value in frame.joints.items():
                data.ctrl[actuator_id[name]] = value
            data.ctrl[actuator_id["gripper"]] = frame.gripper
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
