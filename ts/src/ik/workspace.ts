// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  DEFAULT_CUBE_POSE,
  GRIPPER_TARGET_POSITION
} from '../visualizations/pregrasp-pose-shared/body-factories';
import { createSimplePregraspMatrix } from '../visualizations/simple-pregrasp-pose/pose';
import { type So101Kinematics } from './kinematics';
import { solveSimplePregraspIk } from './simple-ik';

// Reachable workspace for the *simple* vertical pregrasp with a cube resting on
// the ground. Because the cube is on the floor and the pose keeps the gripper
// vertical, the IK target (the jaw contact projected onto the roll axis) sits at
// a fixed height, with the wrist straight above it. The reachable set therefore
// collapses to a closed-form annular SECTOR around the shoulder_pan axis:
//
//   * radial band  — set purely by the planar 2R reach plus the
//     shoulder_lift / elbow_flex / wrist_flex limits. The wrist height is
//     constant, so this band is azimuth-independent.
//   * azimuth band — exactly the shoulder_pan range (shoulder_pan = -azimuth).
//
// wrist_roll never enters the region boundary: rollAngle = (faceNormalAzimuth -
// targetAzimuth) - twist, and its forbidden arc (the gap past the asymmetric
// roll limits) is narrower than the 90 deg spacing of a cube's four vertical
// faces. So at most one face is ever roll-blocked and at least three remain
// graspable -- hence for *any* cube yaw at least one face solves, everywhere in
// the sector. `anyYawReachable` records that this holds for the loaded model.

export interface SimpleWorkspaceSector {
  // World location of the vertical shoulder_pan axis.
  panAxis: THREE.Vector2;
  // Horizontal distance from the pan axis to the IK target, in metres. The IK
  // target is the cube-center offset outward along the grasped face normal by
  // `faceOffset`, so the graspable cube-center band is this band +/- faceOffset.
  radial: { min: number; max: number };
  // World azimuth of the target, measured from the pan axis, in radians.
  azimuth: { min: number; max: number };
  // Height of the IK target above the floor, in metres (constant: cube on
  // ground, vertical pose).
  targetHeight: number;
  // Radial distance from the cube center to the IK target along the grasped
  // face normal, in metres.
  faceOffset: number;
  // True when every cube yaw has at least one graspable face throughout the
  // sector (the wrist_roll argument above).
  anyYawReachable: boolean;
}

// Build the world-from-gripper matrix for a vertical grasp whose IK target lands
// at `target`, with the jaws closing along the horizontal `closingAzimuth`.
function verticalGraspMatrix(
  target: THREE.Vector3,
  closingAzimuth: number
): THREE.Matrix4 {
  const x = new THREE.Vector3(Math.cos(closingAzimuth), Math.sin(closingAzimuth), 0);
  const z = new THREE.Vector3(0, 0, 1);
  const y = new THREE.Vector3().crossVectors(z, x);
  const matrix = new THREE.Matrix4().makeBasis(x, y, z);
  // target = origin + R * GRIPPER_TARGET_POSITION, and R maps (0,0,z) -> (0,0,z),
  // so the gripper origin is simply the target lifted by -GRIPPER_TARGET z.
  matrix.setPosition(
    target.x,
    target.y,
    target.z - GRIPPER_TARGET_POSITION.z
  );
  return matrix;
}

// Does a vertical grasp at horizontal distance `radial` along azimuth 0 solve?
// Closing along the radial keeps wrist_roll near zero and pan at zero, isolating
// the radial reach + flex limits.
function radialReachable(
  k: So101Kinematics,
  radial: number,
  targetHeight: number
): boolean {
  const target = new THREE.Vector3(
    k.panAxis.x + radial,
    k.panAxis.y,
    targetHeight
  );
  return solveSimplePregraspIk(k, verticalGraspMatrix(target, 0)).type === 'success';
}

// Bisect the radial reach boundary between a known-reachable and a known-
// unreachable distance.
function bisectRadial(
  k: So101Kinematics,
  reachable: number,
  unreachable: number,
  targetHeight: number
): number {
  let lo = reachable;
  let hi = unreachable;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (radialReachable(k, mid, targetHeight)) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

export function computeSimpleWorkspace(k: So101Kinematics): SimpleWorkspaceSector {
  // Derive the target height and the cube-center -> target offset from the real
  // simple-pose math rather than hard-coding. Grasp the +x face of the default
  // (ground) cube: the target is the jaw contact projected onto the roll axis.
  const worldFromGripper = createSimplePregraspMatrix('+x', DEFAULT_CUBE_POSE);
  if (worldFromGripper === undefined) {
    throw new Error('Simple pregrasp matrix undefined for the default ground cube');
  }
  const sampleTarget = GRIPPER_TARGET_POSITION.clone().applyMatrix4(worldFromGripper);
  const height = sampleTarget.z;
  const faceOffset = Math.hypot(
    sampleTarget.x - DEFAULT_CUBE_POSE.x,
    sampleTarget.y - DEFAULT_CUBE_POSE.y
  );

  // Locate the feasible radial interval: coarse scan, then bisect both edges.
  const step = 0.002;
  let firstHit = NaN;
  let lastHit = NaN;
  for (let r = 0.01; r <= 0.32 + 1e-9; r += step) {
    if (radialReachable(k, r, height)) {
      if (Number.isNaN(firstHit)) { firstHit = r; }
      lastHit = r;
    }
  }
  if (Number.isNaN(firstHit)) {
    throw new Error('No reachable radial distance found for the simple workspace');
  }
  const radialMin = bisectRadial(k, firstHit, firstHit - step, height);
  const radialMax = bisectRadial(k, lastHit, lastHit + step, height);

  // Azimuth band = shoulder_pan range, negated (shoulder_pan = -azimuth).
  const pan = k.jointLimits.shoulder_pan;
  const azimuth = { min: -pan.max, max: -pan.min };

  // wrist_roll forbidden-arc width vs. the 90 deg face spacing.
  const roll = k.jointLimits.wrist_roll;
  const forbiddenArc = 2 * Math.PI - (roll.max - roll.min);
  const anyYawReachable = forbiddenArc < Math.PI / 2;

  return {
    panAxis: k.panAxis.clone(),
    radial: { min: radialMin, max: radialMax },
    azimuth,
    targetHeight: height,
    faceOffset,
    anyYawReachable
  };
}

// The cube-center radial band graspable for *any* yaw: the target band shrunk by
// the face offset on both ends (the worst-case grasped face can sit a face
// offset either side of the center).
export function anyYawCubeCenterBand(
  sector: SimpleWorkspaceSector
): { min: number; max: number } {
  return {
    min: sector.radial.min + sector.faceOffset,
    max: sector.radial.max - sector.faceOffset
  };
}

// Axis-aligned world bounding box (metres) of the any-yaw cube-center sector.
// Used to bound the X/Y placement sliders to the usable workspace.
export function sectorBoundingBox(
  sector: SimpleWorkspaceSector
): { x: { min: number; max: number }; y: { min: number; max: number } } {
  const band = anyYawCubeCenterBand(sector);
  const { min: azMin, max: azMax } = sector.azimuth;
  // Angles where cos/sin reach extrema, kept only when inside the swept sector.
  const angles = [azMin, azMax];
  for (const a of [0, Math.PI / 2, Math.PI, -Math.PI / 2, -Math.PI]) {
    if (a >= azMin && a <= azMax) { angles.push(a); }
  }
  const xs: number[] = [];
  const ys: number[] = [];
  for (const radius of [band.min, band.max]) {
    for (const a of angles) {
      xs.push(sector.panAxis.x + radius * Math.cos(a));
      ys.push(sector.panAxis.y + radius * Math.sin(a));
    }
  }
  return {
    x: { min: Math.min(...xs), max: Math.max(...xs) },
    y: { min: Math.min(...ys), max: Math.max(...ys) }
  };
}
