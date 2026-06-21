# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Shared plumbing for running a SmolVLA policy on this robot, sim or real.

A LeRobot policy expects a fixed observation contract: a proprioceptive state
vector, one image per camera keyed by name, and a language instruction. The
state and action are in the *real (hardware) frame* the dataset was recorded in
— arm joints in degrees, gripper as a 0-100 position — which is why a sim run
converts at its boundaries while a hardware run feeds the follower's readings
straight through.

SmolVLA keys cameras by their name in ``input_features``, so the training
``--rename_map`` and eval must agree on which physical camera fills each slot.
Following SmolVLA's convention that the main/overview camera comes first, the
overhead view is ``camera1`` and the wrist is ``camera2``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from pick_and_place.follower import JOINT_NAMES

OVERHEAD_FEATURE = "observation.images.camera1"
WRIST_FEATURE = "observation.images.camera2"
DEFAULT_CHECKPOINT = "lerobot/smolvla_base"
# DEFAULT_INSTRUCTION = "Pick up the cube and place it on the target."
DEFAULT_INSTRUCTION = "Pick and grab the object."


def _install_lerobot_policies_stub() -> None:
    """Bypass broken eager imports in ``lerobot.policies.__init__``.

    Some LeRobot builds import every policy from ``lerobot.policies`` package
    initialization. A broken unrelated policy can then prevent SmolVLA-only
    runtime imports. Installing a minimal package module preserves normal
    submodule resolution while avoiding that eager initializer.
    """
    existing = sys.modules.get("lerobot.policies")
    if existing is not None and hasattr(existing, "__path__"):
        return

    import lerobot

    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("could not locate the lerobot package")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    policies_path = package_root / "policies"
    if not policies_path.is_dir():
        raise ImportError(f"could not locate lerobot policies directory: {policies_path}")

    module = types.ModuleType("lerobot.policies")
    module.__file__ = str(policies_path / "__init__.py")
    module.__path__ = [str(policies_path)]
    module.__package__ = "lerobot"
    sys.modules["lerobot.policies"] = module
    setattr(lerobot, "policies", module)


def select_device(requested: str):
    """Resolve ``auto`` to the best available torch device, or honor an explicit one."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_policy(
    checkpoint: str,
    wrist_hw: tuple[int, int],
    overhead_hw: tuple[int, int],
    device,
):
    """Load a SmolVLA checkpoint with feature specs for our 6-DOF arm and two
    cameras, plus its pre/post-processors.

    SmolVLA pads state/action to a fixed internal width and resizes every camera
    image to its own square input, so the base weights load against any robot
    whose dims fit — no finetuning needed to run a forward pass — and the
    declared image shapes need only name the cameras, not match a fixed size.
    The normalization stats come from the checkpoint's own saved processor (the
    base ships its pretraining stats; a fine-tune saves the project dataset's),
    which is why the dataset stays in raw physical units.
    """
    _install_lerobot_policies_stub()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    n_joints = len(JOINT_NAMES)
    checkpoint_path = Path(checkpoint)
    if (checkpoint_path / "config.json").is_file():
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"checkpoint {checkpoint!r} is {config.type!r}, expected 'smolvla'")
        config.device = str(device)
    else:
        config = SmolVLAConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(n_joints,)),
                WRIST_FEATURE: PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, wrist_hw[0], wrist_hw[1])
                ),
                OVERHEAD_FEATURE: PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, overhead_hw[0], overhead_hw[1])
                ),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(n_joints,)),
            },
            device=str(device),
        )
    action_shape = config.output_features["action"].shape
    if action_shape[0] != n_joints:
        raise ValueError(
            f"checkpoint action shape {action_shape} does not match robot joints ({n_joints})"
        )
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": str(device)}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor
