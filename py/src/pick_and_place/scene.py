# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compose the SO-101 robot with a floor, workspace overlays, light, and cube."""

from __future__ import annotations

from pathlib import Path

import mujoco

from pick_and_place.builder import STOCK_ASSETS_DIR, build_robot
from pick_and_place.environment import (
    APRILTAG_TEXTURE_DIR,
    TABLE_CENTER,
    add_overhead_camera_mount,
    add_workspace_frame,
    add_workspace_frame_apriltags,
)
from pick_and_place.materials import MaterialConfig, apply_materials
from pick_and_place.workspace_overlays import add_workspace_overlays

# Tag IDs stickered onto the pick cube's six faces, in MuJoCo cube-texture order
# (right, left, up, down, front, back). With the cube unrotated those map to the
# world directions -X, +X, -Y, +Y, +Z, -Z respectively.
PICK_CUBE_APRILTAG_IDS: tuple[int, int, int, int, int, int] = (0, 1, 2, 3, 4, 5)

# Half-edge of the pick cube; the 30 mm faces carry 30 mm AprilTag stickers.
PICK_CUBE_HALF_SIZE = 0.015

# Plain pick cube colour, used when the AprilTag faces are not requested.
PICK_CUBE_RGBA = (0.82, 0.12, 0.08, 1.0)

# Robot mounts sit 7.2 mm above the floor, on top of the workspace-frame plates.
ROBOT_BASE_HEIGHT = 0.0072

# Each north arm-base plate is 0.116 m (local x) either side of the frame centre.
# Local +x maps to world +y under WORKSPACE_FRAME_QUAT, so the "left" plate is on
# world +y. With the world rooted at the table centre, both robot bases sit at
# x = ROBOT_BASE_X (the plate edge offset, -TABLE_CENTER_x in legacy coords) and
# symmetric in y on the two plates.
ROBOT_PLATE_HALF_SPACING = 0.116
ROBOT_BASE_X = -TABLE_CENTER[0]

# The robots in the scene, each a (actuator prefix, base position) pair. The
# controlled robot is unprefixed on the +y ("left") plate; the passive robot is
# ``other_`` prefixed on the -y plate.
ROBOT_BASE_POSITIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("", (ROBOT_BASE_X, ROBOT_PLATE_HALF_SPACING, ROBOT_BASE_HEIGHT)),
    ("other_", (ROBOT_BASE_X, -ROBOT_PLATE_HALF_SPACING, ROBOT_BASE_HEIGHT)),
)


def build_scene(
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    include_environment: bool = True,
    apriltag_cube: bool | None = None,
) -> mujoco.MjSpec:
    """Return the composed robots with a floor, workspace overlays, light, and cube.

    The world is rooted at the table-frame centre: the workspace frame sits at the
    origin and both robots sit on their fixed, symmetric north plates (the
    unprefixed controlled robot on +y, the passive ``other_`` robot on -y).

    ``apriltag_cube`` selects the pick cube's appearance: the plain red cube for
    the simple scene, or the AprilTag-stickered cube (a perception target) for
    the standard scene. When left ``None`` it follows ``include_environment``, so
    the simple scene gets the red cube and the standard scene the tagged one.
    """
    if apriltag_cube is None:
        apriltag_cube = include_environment

    spec = build_robot(wrist_camera=wrist_camera, materials=materials)
    spec.modelname = "so101_two_robots"
    spec.worldbody.add_light(
        name="scene_light",
        pos=(0.0, 0.0, 1.0),
        dir=(0.0, 0.0, -1.0),
    )

    # The robots are mounted on the workspace-frame plates, elevating their bases
    # by the frame's thickness (7.2 mm). This is critical for IK solving because
    # the floor (where the cube rests) is at Z=0.
    base = spec.body("base")
    base.pos = ROBOT_BASE_POSITIONS[0][1]

    # Attach overlays to worldbody so they stay on the floor.
    add_workspace_overlays(spec, spec.worldbody)
    _add_pick_cube(spec, apriltag=apriltag_cube)

    if include_environment:
        collision_default = spec.find_default("collision")
        add_workspace_frame(spec, collision_default=collision_default)
        add_overhead_camera_mount(spec, collision_default=collision_default)

    apply_materials(spec, materials or MaterialConfig())
    if apriltag_cube:
        _add_pick_cube_apriltags(spec)
    if include_environment:
        add_workspace_frame_apriltags(spec)
    _add_groundplane(spec)

    for prefix, pos in ROBOT_BASE_POSITIONS[1:]:
        _attach_robot(spec, prefix, pos, wrist_camera=wrist_camera, materials=materials)

    return spec


def _attach_robot(
    spec: mujoco.MjSpec,
    prefix: str,
    pos: tuple[float, float, float],
    *,
    wrist_camera: bool,
    materials: MaterialConfig | None,
) -> None:
    """Attach an additional robot at ``pos`` with the given actuator-name prefix."""
    other = build_robot(wrist_camera=wrist_camera, materials=materials)
    frame = spec.worldbody.add_frame(pos=pos)
    spec.attach(other, prefix=prefix, frame=frame)


def build_environment(
    *,
    materials: MaterialConfig | None = None,
    apriltag_cube: bool = True,
) -> mujoco.MjSpec:
    """Return only the environment, with no robot.

    Contains the floor, pick cube, calibration workspace frame, and overhead
    camera mount, all attached to the worldbody. The web viewer loads this on
    top of the standalone ``so101`` model so the robot is defined exactly once
    instead of being baked into the scene a second time. This is the standard
    scene, so the pick cube carries AprilTag faces by default.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "pick_and_place_environment"
    _add_pick_cube(spec, apriltag=apriltag_cube)
    add_workspace_frame(spec)
    add_overhead_camera_mount(spec)
    apply_materials(spec, materials or MaterialConfig())
    if apriltag_cube:
        _add_pick_cube_apriltags(spec)
    add_workspace_frame_apriltags(spec)
    _add_groundplane(spec)
    return spec


def _add_groundplane(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.2, 0.3),
        markrgb=(0.8, 0.8, 0.8),
        width=300,
        height=300,
    )
    groundplane = spec.add_material(
        name="groundplane",
        texuniform=True,
        texrepeat=(5.0, 5.0),
        reflectance=0.2,
    )
    groundplane.textures[1] = "groundplane"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05),
        material="groundplane",
    )


def export_scene(
    output: Path,
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    include_environment: bool = True,
    apriltag_cube: bool | None = None,
) -> Path:
    """Write a standalone, machine-local XML file for the composed scene."""
    spec = build_scene(
        wrist_camera=wrist_camera,
        materials=materials,
        include_environment=include_environment,
        apriltag_cube=apriltag_cube,
    )
    spec.meshdir = str(STOCK_ASSETS_DIR)
    spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    return output


def _add_pick_cube(spec: mujoco.MjSpec, *, apriltag: bool) -> None:
    cube = spec.worldbody.add_body(name="pick_cube", pos=(-0.079579, -0.004, PICK_CUBE_HALF_SIZE))
    half = PICK_CUBE_HALF_SIZE
    # The AprilTag stickers are white-backed; the material (added after
    # apply_materials) carries the per-face textures and tints them with this
    # white base so the tags render at full contrast. A plain cube keeps its
    # solid colour and no material.
    rgba = (1.0, 1.0, 1.0, 1.0) if apriltag else PICK_CUBE_RGBA
    cube.add_geom(
        name="pick_cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half, half, half),
        rgba=rgba,
        material="pick_cube_apriltags" if apriltag else "",
        mass=0.0137,
        priority=1,
        solref=(0.002, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    )


def _add_pick_cube_apriltags(spec: mujoco.MjSpec) -> None:
    """Texture the pick cube's six faces with their AprilTag stickers.

    Called after :func:`apply_materials`, which clears the spec's materials, so
    the cube texture and material survive into the compiled model (mirroring how
    the workspace-frame tags are added).
    """
    texture = spec.add_texture(
        name="pick_cube_apriltags",
        type=mujoco.mjtTexture.mjTEXTURE_CUBE,
    )
    texture.cubefiles = [
        str(APRILTAG_TEXTURE_DIR / f"tagStandard41h12_{tag_id:05d}_30x30mm_tag20mm.png")
        for tag_id in PICK_CUBE_APRILTAG_IDS
    ]
    material = spec.add_material(name="pick_cube_apriltags")
    material.textures[1] = "pick_cube_apriltags"
