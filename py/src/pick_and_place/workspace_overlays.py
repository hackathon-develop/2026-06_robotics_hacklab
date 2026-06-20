# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fixed MuJoCo meshes for the standard SO-101 workspace overlays."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from pick_and_place.environment import (
    WORKSPACE_FRAME_APRILTAG_PLATES,
    WORKSPACE_FRAME_POS,
    WORKSPACE_FRAME_QUAT,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose

WORKSPACE_FRAME_INNER_HALF_EXTENT = 0.2813 - 0.0187

# World (x, y) of the controlled arm's shoulder-pan axis. The arm sits on the +y
# north plate in the table-centred world, so this is the stock pan-axis offset
# (0.0388353, ~0) shifted onto that base. Recompute when the robot kinematics or
# the base placement change.
PAN_AXIS = (-0.2407437, 0.1159999910234)
AZIMUTH_MIN = -1.9198621771937634
AZIMUTH_MAX = AZIMUTH_MIN + 3.839724354387525
_PAN_AXIS = PAN_AXIS
_AZIMUTH_MIN = AZIMUTH_MIN
_AZIMUTH_LENGTH = AZIMUTH_MAX - AZIMUTH_MIN

# Shoulder-pan axis offset relative to an (unrotated) robot base, read off the
# stock SO-101 at its reference pose. A robot's world pan axis is its base XY
# plus this offset; both arms face world +x, so they share the azimuth sector.
PAN_AXIS_BASE_OFFSET = (0.0388353, -8.97657e-09)


def pan_axis_for_base(base_xy: tuple[float, float]) -> tuple[float, float]:
    """World (x, y) of the shoulder-pan axis for a robot based at ``base_xy``."""
    return (base_xy[0] + PAN_AXIS_BASE_OFFSET[0], base_xy[1] + PAN_AXIS_BASE_OFFSET[1])

WORKSPACE_OVERLAY_GROUP = 4
_SEGMENTS = 96
_HALF_THICKNESS = 0.00001
_RGBA = (1.0, 0.4667, 0.0, 0.22)
_CUBE_PLACEMENT_RGBA = (0.1333, 0.7725, 0.3686, 0.42)
_CUBE_EXCLUSION_RGBA = (0.9373, 0.2667, 0.2667, 0.62)


@dataclass(frozen=True)
class WorkspaceOverlay:
    """A fixed annular sector projected onto the floor."""

    name: str
    inner_radius: float
    outer_radius: float
    z: float


WORKSPACE_OVERLAYS = (
    WorkspaceOverlay("workspace_global", 0.0, 0.4418431804405771, 0.0002),
    WorkspaceOverlay("workspace_ground_height_arm", 0.0, 0.42911855464836285, 0.0004),
    WorkspaceOverlay(
        "workspace_ground_pregrasp",
        0.07330432949931037,
        0.25536116910332146,
        0.0006,
    ),
    WorkspaceOverlay(
        "workspace_clearance_pregrasp",
        0.08671592519116689,
        0.24194957341146492,
        0.0008,
    ),
)

# Conservatively inset the frame interior by the cube's circumradius. This
# keeps every corner clear of the rails for every sampled yaw.
_CUBE_FRAME_MARGIN = math.sqrt(2.0) * CUBE_HALF_SIZE
_CUBE_FRAME_HALF_EXTENT = WORKSPACE_FRAME_INNER_HALF_EXTENT - _CUBE_FRAME_MARGIN
_APRILTAG_PLATE_HALF_SIZE = 0.03
CUBE_APRILTAG_EXCLUSION_HALF_EXTENT = _APRILTAG_PLATE_HALF_SIZE + _CUBE_FRAME_MARGIN
CUBE_PLACEMENT_BOUNDS = (
    WORKSPACE_FRAME_POS[0] - _CUBE_FRAME_HALF_EXTENT,
    WORKSPACE_FRAME_POS[0] + _CUBE_FRAME_HALF_EXTENT,
    WORKSPACE_FRAME_POS[1] - _CUBE_FRAME_HALF_EXTENT,
    WORKSPACE_FRAME_POS[1] + _CUBE_FRAME_HALF_EXTENT,
)
CUBE_PLACEMENT_OVERLAY = WorkspaceOverlay(
    "workspace_cube_placement",
    WORKSPACE_OVERLAYS[-1].inner_radius,
    0.4310255047641903,
    0.0010,
)


def _is_cube_center_allowed(
    x: float,
    y: float,
    overlay: WorkspaceOverlay,
    pan_axis: tuple[float, float] = PAN_AXIS,
) -> bool:
    """Whether a cube center is in ``overlay``'s reachable zone for one arm.

    The annular-sector + azimuth test is taken relative to ``pan_axis`` (so it
    recentres per arm), while the table placement rectangle and AprilTag-plate
    exclusions are world-fixed and shared across all arms.
    """
    x_min, x_max, y_min, y_max = CUBE_PLACEMENT_BOUNDS
    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        return False
    dx = x - pan_axis[0]
    dy = y - pan_axis[1]
    radius = math.hypot(dx, dy)
    azimuth = math.atan2(dy, dx)
    in_clearance_sector = (
        overlay.inner_radius <= radius <= overlay.outer_radius
        and AZIMUTH_MIN <= azimuth <= AZIMUTH_MAX
    )
    if not in_clearance_sector:
        return False

    local_x, local_y = _world_to_frame_xy(x, y)
    return not any(
        abs(local_x - tag_pos[0]) <= CUBE_APRILTAG_EXCLUSION_HALF_EXTENT
        and abs(local_y - tag_pos[1]) <= CUBE_APRILTAG_EXCLUSION_HALF_EXTENT
        for _, _, tag_pos in WORKSPACE_FRAME_APRILTAG_PLATES
    )


def is_cube_pickup_allowed(
    x: float, y: float, pan_axis: tuple[float, float] = PAN_AXIS
) -> bool:
    """Return whether a floor cube can use the vertical pickup pose."""
    return _is_cube_center_allowed(x, y, WORKSPACE_OVERLAYS[-1], pan_axis)


def is_vertical_grip_allowed(
    x: float, y: float, pan_axis: tuple[float, float] = PAN_AXIS
) -> bool:
    """Return whether a floor cube can use the vertical gripper pose."""
    return is_cube_pickup_allowed(x, y, pan_axis)


def is_cube_drop_allowed(
    x: float, y: float, pan_axis: tuple[float, float] = PAN_AXIS
) -> bool:
    """Return whether a cube-center drop target is in the broad arm workspace."""
    return _is_cube_center_allowed(x, y, CUBE_PLACEMENT_OVERLAY, pan_axis)


# Backward-compatible name for callers that mean pickup placement.
is_cube_placement_allowed = is_cube_pickup_allowed


def sample_handover_pose(
    rng: np.random.Generator,
    pan_drop: tuple[float, float],
    pan_pick: tuple[float, float],
    source: CubePose,
    target: CubePose,
    *,
    bias_sigma: float = 0.06,
    max_attempts: int = 20000,
) -> CubePose:
    """Sample a floor cube pose in the zone shared by two arms.

    The returned pose is drop-allowed for the arm at ``pan_drop`` and
    pickup-allowed for the arm at ``pan_pick`` (their overlapping reachable
    zone). Candidates are biased toward the straight ``source``->``target`` line
    via a Gaussian on the perpendicular distance, so the handover spot stays
    roughly on the way. Raises ``ValueError`` if no shared pose is found within
    ``max_attempts``.
    """
    line_dx = target.x - source.x
    line_dy = target.y - source.y
    line_len = math.hypot(line_dx, line_dy)
    r_inner = CUBE_PLACEMENT_OVERLAY.inner_radius
    r_outer = CUBE_PLACEMENT_OVERLAY.outer_radius
    for _ in range(max_attempts):
        # Sample inside the dropping arm's broad sector, then filter to the spot
        # the picking arm can also reach with a vertical grasp.
        r = rng.uniform(r_inner, r_outer)
        theta = rng.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
        x = pan_drop[0] + r * math.cos(theta)
        y = pan_drop[1] + r * math.sin(theta)
        if not is_cube_drop_allowed(x, y, pan_drop):
            continue
        if not is_cube_pickup_allowed(x, y, pan_pick):
            continue
        if line_len > 0.0:
            perp = abs(line_dx * (y - source.y) - line_dy * (x - source.x)) / line_len
            if rng.random() > math.exp(-((perp / bias_sigma) ** 2)):
                continue
        yaw = float(rng.uniform(0.0, 2 * math.pi))
        return CubePose(x=x, y=y, z=CUBE_HALF_SIZE, yaw=yaw)
    raise ValueError("no handover pose found in the zone shared by both arms")


def workspace_interior_corners_world() -> np.ndarray:
    """World-space corners of the workspace-frame interior.

    Used to mask overhead detections to the table surface, excluding off-table
    clutter (keyboard, cables, the shadowed table border).
    """
    h = WORKSPACE_FRAME_INNER_HALF_EXTENT
    z = WORKSPACE_FRAME_POS[2]
    return np.array(
        [(*_frame_to_world_xy(fx, fy), z) for fx, fy in ((-h, -h), (h, -h), (h, h), (-h, h))],
        dtype=float,
    )


def _world_to_frame_xy(x: float, y: float) -> tuple[float, float]:
    """Transform a world XY point into the workspace-frame coordinate system."""
    w, qx, qy, qz = WORKSPACE_FRAME_QUAT
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * w)
    r10 = 2.0 * (qx * qy + qz * w)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    world_dx = x - WORKSPACE_FRAME_POS[0]
    world_dy = y - WORKSPACE_FRAME_POS[1]
    return (
        r00 * world_dx + r10 * world_dy,
        r01 * world_dx + r11 * world_dy,
    )


def _frame_to_world_xy(x: float, y: float) -> tuple[float, float]:
    """Transform a workspace-frame XY point into world coordinates."""
    w, qx, qy, qz = WORKSPACE_FRAME_QUAT
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * w)
    r10 = 2.0 * (qx * qy + qz * w)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    return (
        WORKSPACE_FRAME_POS[0] + r00 * x + r01 * y,
        WORKSPACE_FRAME_POS[1] + r10 * x + r11 * y,
    )


def add_workspace_overlays(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    *,
    pan_axis: tuple[float, float] = PAN_AXIS,
    prefix: str = "",
) -> None:
    """Add one arm's non-colliding reachable-sector overlays in ``parent`` coords.

    The annular sectors recentre on ``pan_axis`` (so each arm draws its own
    fan); ``prefix`` namespaces the geoms/meshes so several arms can coexist. The
    world-fixed AprilTag exclusion boxes are added once via
    :func:`add_cube_exclusion_overlays`.
    """
    for overlay in WORKSPACE_OVERLAYS:
        name = f"{prefix}{overlay.name}"
        vertices, faces = _annular_sector_mesh(
            overlay.inner_radius,
            overlay.outer_radius,
        )
        mesh = spec.add_mesh(name=name)
        mesh.uservert = vertices.flatten()
        mesh.userface = faces.flatten()
        parent.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=(*pan_axis, overlay.z),
            rgba=_RGBA,
            contype=0,
            conaffinity=0,
            group=WORKSPACE_OVERLAY_GROUP,
        )

    placement_name = f"{prefix}{CUBE_PLACEMENT_OVERLAY.name}"
    vertices, faces = _clipped_annular_sector_mesh(
        CUBE_PLACEMENT_OVERLAY.inner_radius,
        CUBE_PLACEMENT_OVERLAY.outer_radius,
        CUBE_PLACEMENT_BOUNDS,
        pan_axis,
    )
    mesh = spec.add_mesh(name=placement_name)
    mesh.uservert = vertices.flatten()
    mesh.userface = faces.flatten()
    parent.add_geom(
        name=placement_name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        pos=(*pan_axis, CUBE_PLACEMENT_OVERLAY.z),
        rgba=_CUBE_PLACEMENT_RGBA,
        contype=0,
        conaffinity=0,
        group=WORKSPACE_OVERLAY_GROUP,
    )


def add_cube_exclusion_overlays(spec: mujoco.MjSpec, parent: mujoco.MjsBody) -> None:
    """Add the world-fixed AprilTag-plate exclusion boxes (shared by all arms)."""
    del spec  # geoms only; no meshes to register
    for _, corner_name, tag_pos in WORKSPACE_FRAME_APRILTAG_PLATES:
        tag_x, tag_y = _frame_to_world_xy(tag_pos[0], tag_pos[1])
        parent.add_geom(
            name=f"workspace_cube_exclusion_tag_{corner_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                CUBE_APRILTAG_EXCLUSION_HALF_EXTENT,
                CUBE_APRILTAG_EXCLUSION_HALF_EXTENT,
                _HALF_THICKNESS,
            ),
            pos=(tag_x, tag_y, CUBE_PLACEMENT_OVERLAY.z + 0.00004),
            rgba=_CUBE_EXCLUSION_RGBA,
            contype=0,
            conaffinity=0,
            group=WORKSPACE_OVERLAY_GROUP,
        )


def _clipped_annular_sector_mesh(
    inner_radius: float,
    outer_radius: float,
    bounds: tuple[float, float, float, float],
    pan_axis: tuple[float, float] = _PAN_AXIS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the annular sector clipped to an axis-aligned center rectangle.

    The mesh is centred at the pan axis (geoms place it at ``pos=pan_axis``), so
    the rectangle bounds are taken relative to ``pan_axis``.
    """
    x_min, x_max, y_min, y_max = bounds
    local_bounds = (
        x_min - pan_axis[0],
        x_max - pan_axis[0],
        y_min - pan_axis[1],
        y_max - pan_axis[1],
    )
    sections: list[tuple[float, float, float]] = []
    for angle in np.linspace(_AZIMUTH_MIN, _AZIMUTH_MIN + _AZIMUTH_LENGTH, _SEGMENTS + 1):
        dx, dy = math.cos(angle), math.sin(angle)
        exits = []
        if dx > 0.0:
            exits.append(local_bounds[1] / dx)
        elif dx < 0.0:
            exits.append(local_bounds[0] / dx)
        if dy > 0.0:
            exits.append(local_bounds[3] / dy)
        elif dy < 0.0:
            exits.append(local_bounds[2] / dy)
        clipped_outer = min(outer_radius, *exits)
        if clipped_outer >= inner_radius:
            sections.append((float(angle), inner_radius, clipped_outer))

    if len(sections) < 2:
        raise ValueError("cube placement bounds do not intersect the workspace")

    vertices: list[tuple[float, float, float]] = []
    for z in (-_HALF_THICKNESS, _HALF_THICKNESS):
        for angle, inner, outer in sections:
            vertices.append((inner * math.cos(angle), inner * math.sin(angle), z))
            vertices.append((outer * math.cos(angle), outer * math.sin(angle), z))

    count = len(sections)
    layer = 2 * count
    faces: list[tuple[int, int, int]] = []
    for i in range(count - 1):
        bi, bo = 2 * i, 2 * i + 1
        ni, no = bi + 2, bo + 2
        faces.extend(((bi, bo, ni), (bo, no, ni)))
        faces.extend(((bi + layer, ni + layer, bo + layer), (bo + layer, ni + layer, no + layer)))
        faces.extend(((bi, ni, bi + layer), (ni, ni + layer, bi + layer)))
        faces.extend(((bo, bo + layer, no), (no, bo + layer, no + layer)))

    for inner_index, outer_index in ((0, 1), (2 * (count - 1), 2 * (count - 1) + 1)):
        faces.extend(
            (
                (inner_index, inner_index + layer, outer_index),
                (outer_index, inner_index + layer, outer_index + layer),
            )
        )
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def _annular_sector_mesh(
    inner_radius: float,
    outer_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed, thin annular-sector mesh centered at the origin."""
    angles = np.linspace(_AZIMUTH_MIN, _AZIMUTH_MIN + _AZIMUTH_LENGTH, _SEGMENTS + 1)
    outer_xy = np.column_stack((outer_radius * np.cos(angles), outer_radius * np.sin(angles)))
    if math.isclose(inner_radius, 0.0):
        contour_xy = np.vstack((outer_xy, (0.0, 0.0)))
    else:
        inner_xy = np.column_stack((inner_radius * np.cos(angles), inner_radius * np.sin(angles)))
        contour_xy = np.vstack((outer_xy, inner_xy[::-1]))

    count = len(contour_xy)
    bottom = np.column_stack((contour_xy, np.full(count, -_HALF_THICKNESS)))
    top = np.column_stack((contour_xy, np.full(count, _HALF_THICKNESS)))
    vertices = np.vstack((bottom, top))

    faces: list[tuple[int, int, int]] = []
    if math.isclose(inner_radius, 0.0):
        center = count - 1
        for i in range(_SEGMENTS):
            faces.extend(((center, i + 1, i), (center + count, i + count, i + 1 + count)))
    else:
        for i in range(_SEGMENTS):
            outer_a = i
            outer_b = i + 1
            inner_a = count - 1 - i
            inner_b = count - 2 - i
            faces.extend(
                (
                    (outer_a, inner_a, outer_b),
                    (outer_b, inner_a, inner_b),
                    (outer_a + count, outer_b + count, inner_a + count),
                    (outer_b + count, inner_b + count, inner_a + count),
                )
            )

    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, j, i + count), (j, j + count, i + count)))

    return vertices, np.asarray(faces, dtype=int)
