#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Splice a VLA grasp segment into the classical pick-and-place pipeline, in sim.

The classical planner drives the arm from neutral to a hover above the cube
(``ApproachPhase``); from there a SmolVLA policy takes over and runs closed-loop
until it has descended, grasped, and lifted the cube; then the classical planner
resumes from the measured post-lift state and plays Carry/Release/Retreat.

The hand-back point is detected purely from the tool-center-point's world height
(``DescentAscentDetector``): the end-effector dips below the hover to reach the
cube, bottoms out, then rises again as it lifts. That descent-then-ascent profile
is object-agnostic, so the splice never reads AprilTags or a cube model.

Everything inside the segment is shared with the real arm via
``pick_and_place.grasp_segment``: this script only supplies the sim ``env``
adapter (MuJoCo rendering, the sim<->real frame conversions, and TCP forward
kinematics) and the surrounding classical playback.

The policy speaks the real (hardware) frame the dataset was recorded in -- arm
joints in degrees, gripper as a 0-100 position -- while MuJoCo speaks radians,
so the adapter converts at both boundaries with zero calibration offsets (the
real frame is sim degrees, which is what an uncalibrated fine-tune trains on).
"""

from __future__ import annotations

import argparse
import math
import os
import time

# Some SmolVLM backbone ops are not implemented for Apple MPS; fall back to CPU
# for just those ops instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.episodes import (
    EpisodeSamplingError,
    _preflight,
    is_unexpected,
    prepare_episode,
)
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    GRIPPER_INDEX,
    JOINT_NAMES,
    load_follower_joint_offsets,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.grasp_segment import (
    ASCENT_MARGIN,
    DESCENT_MARGIN,
    DescentAscentDetector,
    run_grasp_segment,
)
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    add_paper_target_marker,
    place_paper_target_marker,
)
from pick_and_place.trajectory import Trajectory, replan_remaining_candidates
from pick_and_place.vla import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INSTRUCTION,
    make_policy,
    select_device,
)
from pick_and_place.workspace_overlays import is_cube_drop_allowed

# Control/render rate of the VLA segment. One policy query and one camera render
# happen per control tick; the sim integrates ``substeps`` model steps per tick.
CONTROL_HZ = 10.0


def _build_model(
    source: CubePose,
    target_xy: tuple[float, float],
    target_yaw: float,
    render_size: int,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the standard (AprilTag, calibrated-camera) scene with the pick cube
    as a free rigid body at ``source``, ready to be handed to ``prepare_episode``.

    The black drop-zone square is rendered at ``target_xy``/``target_yaw`` so the
    frames match a real recording, where a physical paper square on the table marks
    where the cube must be placed; without it the policy sees no target.
    ``render_size`` enlarges the offscreen framebuffer so the camera renders fed to
    the policy fit (MuJoCo defaults to 640x480, too small for a 512 square)."""
    spec = build_scene(include_environment=True)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, render_size)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, render_size)
    apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
    intrinsics = load_local_camera_intrinsics()
    for camera in spec.cameras:
        if camera.name in intrinsics and "fovy_deg" in intrinsics[camera.name]:
            camera.fovy = float(intrinsics[camera.name]["fovy_deg"])

    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    half_yaw = source.yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()

    add_paper_target_marker(spec)

    model = spec.compile()
    place_paper_target_marker(
        model,
        target_xy,
        target_yaw,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=is_cube_drop_allowed(target_xy[0], target_xy[1]),
        alpha=1.0,
    )
    return model, mujoco.MjData(model)


def _joint_qpos_adr(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    ]


def _sim_state_to_real(qpos_rad: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sim joint positions (radians, ``JOINT_NAMES`` order) -> real-frame state
    vector (arm degrees + gripper 0-100), matching the dataset convention."""
    arm = {name: float(qpos_rad[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
    return sim_frame_to_real(arm, float(qpos_rad[GRIPPER_INDEX]), offsets).astype(np.float32)


def _real_action_to_ctrl(action_real: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Real-frame action vector from the policy -> sim ctrl (radians,
    ``JOINT_NAMES`` order, which the actuators follow)."""
    arm_rad, gripper_rad = real_frame_to_sim(action_real, offsets)
    return np.array([arm_rad[name] for name in ARM_JOINT_NAMES] + [gripper_rad])


def _play_trajectory(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: Trajectory,
    actuator_id: dict[str, int],
    viewer,
) -> None:
    """Step the sim through a (sub)trajectory, writing its set points to the
    position-servo actuators. Paced to the model timestep when a viewer is open."""
    start = data.time
    while data.time - start < trajectory.duration:
        step_start = time.time()
        frame = trajectory.evaluate(data.time - start)
        for name, value in frame.joints.items():
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


class _SimGraspEnv:
    """Sim adapter implementing the ``grasp_segment.GraspEnv`` protocol.

    Renders the two cameras, reports proprioceptive state and TCP height, and
    applies policy actions as position targets, stepping physics ``substeps``
    times per control tick so the sim advances one control period."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        kinematics,
        actuator_id: dict[str, int],
        offsets: np.ndarray,
        render_size: int,
        viewer,
    ) -> None:
        self._model = model
        self._data = data
        self._kinematics = kinematics
        self._actuator_id = actuator_id
        self._offsets = offsets
        self._viewer = viewer
        self._joint_adr = _joint_qpos_adr(model)
        self._ctrl_low = model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = model.actuator_ctrlrange[:, 1].copy()
        self._substeps = max(1, round((1.0 / CONTROL_HZ) / model.opt.timestep))
        self._renderer = mujoco.Renderer(model, height=render_size, width=render_size)

    def _render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()  # (H, W, 3) uint8 RGB

    def images(self) -> tuple[np.ndarray, np.ndarray]:
        return self._render("wrist_camera"), self._render("overhead_camera")

    def observation_state(self) -> np.ndarray:
        return _sim_state_to_real(self._data.qpos[self._joint_adr], self._offsets)

    def tcp_z(self) -> float:
        arm = {
            name: float(self._data.qpos[self._joint_adr[i]])
            for i, name in enumerate(ARM_JOINT_NAMES)
        }
        return float(self._kinematics.tip_position(arm)[2])

    def apply_action(self, action_real: np.ndarray) -> None:
        ctrl = _real_action_to_ctrl(action_real, self._offsets)
        self._data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_step(self._model, self._data, nstep=self._substeps)
        if self._viewer is not None:
            self._viewer.sync()

    def close(self) -> None:
        self._renderer.close()


def _select_handback_trajectory(
    episode,
    measured_joints: dict[str, float],
    measured_gripper: float,
) -> Trajectory | None:
    """Replan Carry/Release/Retreat from the measured post-lift state, returning
    the first candidate that passes the collision preflight (mirrors the executor's
    replan-after-lift path)."""
    for candidate in replan_remaining_candidates(
        episode.kinematics,
        measured_joints,
        measured_gripper,
        "lift",
        episode.source,
        episode.target,
        episode.grasp,
        episode.end_joints,
        episode.end_gripper,
        drop_orientation=episode.trajectory.drop_orientation,
    ):
        events = _preflight(
            episode.model,
            candidate,
            episode.actuator_id,
            episode.robot_geom_ids,
            episode.env_geom_ids,
        )
        if not any(is_unexpected(n1, n2) for _, n1, n2 in events):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="language task string")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF policy checkpoint")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument(
        "--image-size", type=int, default=512, help="square render size fed to the VLA"
    )
    parser.add_argument("--source", type=float, nargs=2, metavar=("X", "Y"), default=(0.22, 0.0))
    parser.add_argument("--source-yaw", type=float, default=0.0, help="cube yaw (radians)")
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the drop-zone center (x, y); omit to sample one randomly like the recording",
    )
    parser.add_argument("--target-yaw", type=float, default=0.0, help="drop-zone yaw (radians)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for target sampling")
    parser.add_argument("--headless", action="store_true", help="no viewer; render only for the policy")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "cap the VLA segment to this many control ticks then hand back regardless "
            "(0 = use --grasp-max-ticks); a small value is a fast plumbing smoke test"
        ),
    )
    parser.add_argument(
        "--descent-margin",
        type=float,
        default=DESCENT_MARGIN,
        help="TCP drop below the hover (m) that starts the descent (hand-back detector)",
    )
    parser.add_argument(
        "--ascent-margin",
        type=float,
        default=ASCENT_MARGIN,
        help="TCP rise above the bottom (m) that fires the hand-back (detector)",
    )
    parser.add_argument(
        "--grasp-max-ticks",
        type=int,
        default=600,
        help="safety budget of control ticks for the VLA segment before it times out",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Loading {args.checkpoint} on {device} (first run downloads the weights)...")

    source = CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE, yaw=args.source_yaw)
    if args.target is not None:
        target_xy, target_yaw = tuple(args.target), args.target_yaw
        target = CubePose(x=target_xy[0], y=target_xy[1], z=CUBE_HALF_SIZE)
    else:
        from pick_and_place.episodes import sample_target

        target = sample_target(np.random.default_rng(args.seed))
        target_xy, target_yaw = (target.x, target.y), args.target_yaw
    print(f"Cube at ({source.x:.4f}, {source.y:.4f}), yaw {source.yaw:.3f} rad")
    print(f"Drop zone at ({target_xy[0]:.4f}, {target_xy[1]:.4f}), yaw {target_yaw:.3f} rad")

    model, data = _build_model(source, target_xy, target_yaw, args.image_size)

    # Plan with the very same model so the classical phases (Approach, then the
    # post-lift Carry/Release/Retreat) share its kinematics and cube pose.
    try:
        episode = prepare_episode(
            np.random.default_rng(args.seed),
            source,
            target,
            model=model,
            data=data,
            verbose=True,
        )
    except EpisodeSamplingError as exc:
        raise SystemExit(str(exc)) from exc

    # Zero sim->real offsets: the real frame is sim degrees with no calibration
    # bias, which is what an uncalibrated fine-tune is trained against.
    offsets = load_follower_joint_offsets(None)
    hw = (args.image_size, args.image_size)
    policy_bundle = make_policy(args.checkpoint, hw, hw, device)
    policy_bundle[0].reset()

    viewer_ctx = None if args.headless else mujoco.viewer.launch_passive(model, data)
    viewer = viewer_ctx.__enter__() if viewer_ctx is not None else None

    print(f"Instruction: {args.instruction!r}")
    if args.checkpoint == DEFAULT_CHECKPOINT:
        print("Running with the un-finetuned base: VLA actions are NOT task-calibrated.")

    try:
        # 1. Classical approach: neutral -> hover above the cube.
        approach = Trajectory((episode.trajectory.phases[0],))
        print(f"Playing classical {episode.trajectory.phases[0].name} phase...")
        _play_trajectory(model, data, approach, episode.actuator_id, viewer)

        # 2. VLA segment: descend -> grasp -> lift, closed-loop.
        env = _SimGraspEnv(
            model, data, episode.kinematics, episode.actuator_id, offsets, args.image_size, viewer
        )
        detector = DescentAscentDetector(
            descent_margin=args.descent_margin, ascent_margin=args.ascent_margin
        )
        max_ticks = args.steps if args.steps > 0 else args.grasp_max_ticks
        print(f"Running VLA grasp segment (max {max_ticks} ticks)...")
        result = run_grasp_segment(
            env, policy_bundle, args.instruction, detector, device, max_ticks, CONTROL_HZ
        )
        env.close()
        if result.done:
            print(f"Hand-back: descent-then-ascent detected after {result.ticks} ticks.")
        else:
            print(f"VLA segment timed out after {result.ticks} ticks; handing back anyway.")

        # 3. Hand back: replan Carry/Release/Retreat from the measured post-lift state.
        joint_adr = _joint_qpos_adr(model)
        measured_joints = {
            name: float(data.qpos[joint_adr[i]]) for i, name in enumerate(ARM_JOINT_NAMES)
        }
        measured_gripper = float(data.qpos[joint_adr[GRIPPER_INDEX]])
        handback = _select_handback_trajectory(episode, measured_joints, measured_gripper)
        if handback is None:
            raise SystemExit("no collision-free Carry/Release/Retreat from the post-lift state")

        # 4. Classical carry/release/retreat.
        print("Playing classical carry/release/retreat...")
        _play_trajectory(model, data, handback, episode.actuator_id, viewer)
        print("Done.")

        # Hold the final pose in the viewer until it is closed.
        while viewer is not None and viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
