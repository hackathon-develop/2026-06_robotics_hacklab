#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run the analytic (closed-form planner) pick-and-place on the physical SO-101.

Prepares a collision-free episode from the closed-form ``pick_and_carry`` planner
(the same sampler ``view_trajectory`` uses) and runs it on the real arm via
``pick_and_place.executor``. Today this is open-loop feedforward — the sim is the
source of truth: it integrates physics in a live viewer while set points feed the
follower at ``CONTROL_HZ`` and motor readback is logged. The executor is where the
phase state machine and checkpoint replanning will grow (see
``docs/realworld-execution-roadmap.md``). With zero offsets the per-joint tracking
report doubles as a sim→real calibration measurement.

This is the analytic hardware path. For sim-only playback (no arm) use
``view_trajectory``; learned policies live under ``pick_and_place.il`` / ``.rl``.
"""

from __future__ import annotations

import argparse

import numpy as np

from pick_and_place.episodes import prepare_episode
from pick_and_place.executor import REAL_ARM_DEFAULT_SPEED, execute_episode
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose


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
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="target (x, y) on the floor; omit for a random pose in the clearance annulus",
    )
    parser.add_argument(
        "--follower-port",
        required=True,
        help="serial port of the SO-101 follower",
    )
    parser.add_argument(
        "--follower-id",
        default="folly",
        help="follower calibration id used by lerobot (default: folly)",
    )
    parser.add_argument(
        "--offsets-path",
        default=None,
        help="JSON of per-joint sim→real degree offsets (default: zero offsets)",
    )
    parser.add_argument(
        "--record-path",
        default=None,
        help="CSV path for the per-tick desired-vs-actual motor log",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="playback speed multiplier of the nominal trajectory pace "
        f"(1.0 = nominal; default {REAL_ARM_DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--environment",
        action="store_true",
        help="include the calibration workspace_frame and overhead camera mount in the scene",
    )
    args = parser.parse_args()

    source = (
        CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
        if args.source is not None
        else None
    )
    target = (
        CubePose(x=args.target[0], y=args.target[1], z=CUBE_HALF_SIZE)
        if args.target is not None
        else None
    )

    episode = prepare_episode(
        np.random.default_rng(),
        source,
        target,
        verbose=True,
        include_environment=args.environment,
    )

    execute_episode(
        episode,
        follower_port=args.follower_port,
        follower_id=args.follower_id,
        offsets_path=args.offsets_path,
        record_path=args.record_path,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
