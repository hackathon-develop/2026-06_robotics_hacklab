# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Shared core for the VLA-owned descend->grasp->lift segment.

The classical pipeline drives the arm to a hover above the cube; from there a
SmolVLA policy takes over and runs closed-loop until it has grasped and lifted
the object, after which the classical planner resumes from the measured
post-lift state. This module holds the parts that are identical sim and real:

- ``DescentAscentDetector`` decides when the segment is finished purely from the
  tool-center-point's world height. The end-effector dips below the hover to
  reach the object, bottoms out, then rises again as it lifts; that
  descent-then-ascent profile is object-agnostic, so the hand-back trigger does
  not depend on AprilTags or a cube model.
- ``GraspEnv`` is the narrow interface the loop drives. Sim and real each supply
  an adapter; the loop itself never imports MuJoCo or the hardware.
- ``run_grasp_segment`` is that loop: render -> build the LeRobot observation ->
  ``predict_action`` -> apply -> check the detector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pick_and_place.follower import GRIPPER_INDEX, JOINT_NAMES
from pick_and_place.vla import OVERHEAD_FEATURE, WRIST_FEATURE

# TCP world-z swing, in metres, that distinguishes the descent and the lift from
# servo jitter at the hover. The descent only has to clear the noise floor; the
# ascent must exceed the descent so a wobble at the bottom never reads as a lift.
DESCENT_MARGIN = 0.015
ASCENT_MARGIN = 0.03


class DescentAscentDetector:
    """Fire ``done`` once the TCP has descended below the hover and risen again.

    The baseline ``z0`` is captured from the first sample (the hover height).
    Once the TCP drops more than ``descent_margin`` below it the detector starts
    tracking the running minimum; once it climbs back ``ascent_margin`` above
    that minimum the grasp-and-lift is complete.

    The optional ``gripper_closed_max`` soft guard refuses to accept the ascent
    until the gripper position has dropped to (at or below) that 0-100 value, so
    a lift attempted with open jaws does not end the segment prematurely. It is
    object-agnostic and can be disabled by leaving it ``None``.
    """

    def __init__(
        self,
        descent_margin: float = DESCENT_MARGIN,
        ascent_margin: float = ASCENT_MARGIN,
        gripper_closed_max: float | None = None,
    ) -> None:
        self.descent_margin = descent_margin
        self.ascent_margin = ascent_margin
        self.gripper_closed_max = gripper_closed_max
        self.z0: float | None = None
        self.z_min: float | None = None
        self.descending = False
        self.gripper_was_closed = gripper_closed_max is None

    def update(self, z: float, gripper: float | None = None) -> bool:
        """Feed one TCP-z sample (and optional gripper position); return whether
        the descend-then-ascent has completed."""
        if self.z0 is None:
            self.z0 = z
            self.z_min = z
            return False

        if gripper is not None and self.gripper_closed_max is not None:
            if gripper <= self.gripper_closed_max:
                self.gripper_was_closed = True

        if not self.descending:
            if z < self.z0 - self.descent_margin:
                self.descending = True
                self.z_min = z
            return False

        assert self.z_min is not None
        self.z_min = min(self.z_min, z)
        if z > self.z_min + self.ascent_margin and self.gripper_was_closed:
            return True
        return False


class GraspEnv(Protocol):
    """Everything ``run_grasp_segment`` needs from its environment.

    State and action are in the real (hardware) frame the policy speaks: arm
    joints in degrees, gripper as a 0-100 position. Images are ``(H, W, 3)``
    uint8 RGB. ``tcp_z`` is the gripper target's world height in metres.
    """

    def observation_state(self) -> np.ndarray: ...

    def images(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(wrist_rgb, overhead_rgb)``."""
        ...

    def tcp_z(self) -> float: ...

    def apply_action(self, action_real: np.ndarray) -> None: ...


@dataclass
class GraspSegmentResult:
    """Outcome of one ``run_grasp_segment`` call."""

    done: bool
    timed_out: bool
    ticks: int


def run_grasp_segment(
    env: GraspEnv,
    policy_bundle: tuple,
    instruction: str,
    detector: DescentAscentDetector,
    device,
    max_ticks: int,
    control_hz: float,
) -> GraspSegmentResult:
    """Drive the policy closed-loop until the detector fires or the tick budget runs out.

    ``policy_bundle`` is the ``(policy, preprocessor, postprocessor)`` triple from
    ``vla.make_policy``. Each tick renders the two cameras, builds the observation
    a LeRobot policy expects, queries one action, applies it in the real frame, and
    feeds the new TCP height to ``detector``. Paced to ``control_hz``.
    """
    from lerobot.utils.control_utils import predict_action

    policy, preprocessor, postprocessor = policy_bundle
    period = 1.0 / control_hz

    for tick in range(max_ticks):
        tick_start = time.time()

        wrist_frame, overhead_frame = env.images()
        observation = {
            "observation.state": env.observation_state(),
            WRIST_FEATURE: wrist_frame,
            OVERHEAD_FEATURE: overhead_frame,
        }
        action = predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=False,
            task=instruction,
            robot_type="so101",
        )
        action_real = action.to("cpu").numpy().reshape(-1)[: len(JOINT_NAMES)]
        env.apply_action(action_real)

        if detector.update(env.tcp_z(), float(action_real[GRIPPER_INDEX])):
            return GraspSegmentResult(done=True, timed_out=False, ticks=tick + 1)

        remaining = period - (time.time() - tick_start)
        if remaining > 0:
            time.sleep(remaining)

    return GraspSegmentResult(done=False, timed_out=True, ticks=max_ticks)
