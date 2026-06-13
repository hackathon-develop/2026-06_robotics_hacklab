// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  CUBE_HALF_SIZE,
  type CubeFace,
  type CubePose
} from '../visualizations/pregrasp-pose-shared/body-factories';
import { createSimplePregraspMatrix } from '../visualizations/simple-pregrasp-pose/pose';
import type { WebModel } from '../web-model';
import { deriveSo101Kinematics } from './kinematics';
import { solveSimplePregraspIk } from './simple-ik';
import {
  anyYawCubeCenterBand,
  computeSimpleWorkspace,
  sectorBoundingBox
} from './workspace';

const model = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../public/so101.json', import.meta.url)),
    'utf8'
  )
) as WebModel;

const VERTICAL_FACES: CubeFace[] = ['+x', '-x', '+y', '-y'];

// At a cube center (x, y) with the given yaw, can at least one vertical face be
// grasped by the simple pregrasp IK?
function anyFaceSolves(
  k: ReturnType<typeof deriveSo101Kinematics>,
  x: number,
  y: number,
  yaw: number
): boolean {
  const pose: CubePose = { x, y, z: CUBE_HALF_SIZE, roll: 0, pitch: 0, yaw };
  return VERTICAL_FACES.some(face => {
    const matrix = createSimplePregraspMatrix(face, pose);
    if (matrix === undefined) { return false; }
    return solveSimplePregraspIk(k, matrix).type === 'success';
  });
}

describe('computeSimpleWorkspace', () => {
  const k = deriveSo101Kinematics(model);
  const sector = computeSimpleWorkspace(k);

  it('reports the closed-form sector for the loaded model', () => {
    console.log(
      '[simple workspace] radial %s..%s m, azimuth +/-%s deg, ' +
      'targetHeight %s m, faceOffset %s m, anyYawReachable=%s',
      sector.radial.min.toFixed(4),
      sector.radial.max.toFixed(4),
      ((sector.azimuth.max * 180) / Math.PI).toFixed(1),
      sector.targetHeight.toFixed(4),
      sector.faceOffset.toFixed(4),
      sector.anyYawReachable
    );

    // Pin the computed numbers so changes to the kinematics are visible here.
    expect(sector.targetHeight).toBeCloseTo(CUBE_HALF_SIZE, 4);
    expect(sector.radial.min).toBeCloseTo(0.0562, 3);
    expect(sector.radial.max).toBeCloseTo(0.2725, 3);
    expect(sector.azimuth.min).toBeCloseTo(-1.91986, 4);
    expect(sector.azimuth.max).toBeCloseTo(1.91986, 4);
    expect(sector.anyYawReachable).toBe(true);
  });

  it('validates the sector: every interior cube center solves for all yaws', () => {
    // Shrink the radial band by the face offset (target vs. cube center) and the
    // azimuth band slightly, then brute-check that a worst-case yaw sweep always
    // finds a graspable face -- confirming the closed-form claim.
    const rLo = sector.radial.min + sector.faceOffset;
    const rHi = sector.radial.max - sector.faceOffset;
    const azPad = 0.05;

    for (let r = rLo; r <= rHi; r += (rHi - rLo) / 8) {
      for (let az = sector.azimuth.min + azPad; az <= sector.azimuth.max - azPad;
        az += (sector.azimuth.max - sector.azimuth.min - 2 * azPad) / 10) {
        const x = sector.panAxis.x + r * Math.cos(az);
        const y = sector.panAxis.y + r * Math.sin(az);
        for (let yaw = 0; yaw < Math.PI / 2; yaw += Math.PI / 2 / 18) {
          expect(anyFaceSolves(k, x, y, yaw)).toBe(true);
        }
      }
    }
  });

  it('shrinks the target band by the face offset for cube centers', () => {
    const band = anyYawCubeCenterBand(sector);
    expect(band.min).toBeCloseTo(sector.radial.min + sector.faceOffset, 6);
    expect(band.max).toBeCloseTo(sector.radial.max - sector.faceOffset, 6);
  });

  it('bounds the cube-center sector (drives the X/Y slider ranges)', () => {
    const bbox = sectorBoundingBox(sector);
    const band = anyYawCubeCenterBand(sector);
    // Pan range exceeds +/-90 deg, so the outer radius sets both Y extremes and
    // the +X extreme, measured from the pan axis.
    expect(bbox.x.max).toBeCloseTo(sector.panAxis.x + band.max, 4);
    expect(bbox.y.max).toBeCloseTo(sector.panAxis.y + band.max, 4);
    expect(bbox.y.min).toBeCloseTo(sector.panAxis.y - band.max, 4);
    expect(bbox.x.min).toBeLessThan(sector.panAxis.x);
  });

  it('confirms wrist_roll cannot veto every cube yaw', () => {
    const roll = k.jointLimits.wrist_roll;
    const forbiddenArc = 2 * Math.PI - (roll.max - roll.min);
    expect(forbiddenArc).toBeLessThan(Math.PI / 2);
  });
});
