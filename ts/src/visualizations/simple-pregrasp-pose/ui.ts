// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendCubePoseInputs,
  appendFaceInputs,
  createPane,
  type CubePoseInputs,
  type PregraspPosePane,
  SIDE_FACES } from '../pregrasp-pose-shared/ui';

export interface SimplePregraspPoseDom extends CubePoseInputs {
  root: HTMLDivElement;
  pane: PregraspPosePane;
  faceInputs: HTMLInputElement[];
  resetButton: HTMLButtonElement;
  status: HTMLOutputElement;
}

export function buildUi(parent: HTMLElement): SimplePregraspPoseDom {
  const root = document.createElement('div');
  root.className = 'visualization pregrasp-pose-viz-root';

  const controls = document.createElement('div');
  controls.className = 'pregrasp-pose-breakdown-viz-controls';
  const faceInputs = appendFaceInputs(
    controls, 'simple-pregrasp-pose-cube-face', SIDE_FACES
  );
  const cubePoseInputs = appendCubePoseInputs(controls);
  const resetButton = document.createElement('button');
  resetButton.className = 'simple-pregrasp-pose-viz-reset';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  controls.appendChild(resetButton);
  root.appendChild(controls);

  const status = document.createElement('output');
  status.className = 'simple-pregrasp-pose-viz-status';
  root.appendChild(status);

  const pane = createPane('Simple pregrasp pose', 'combined', 'final', true);
  root.appendChild(pane.element);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return { root, pane, faceInputs, resetButton, status, ...cubePoseInputs };
}
