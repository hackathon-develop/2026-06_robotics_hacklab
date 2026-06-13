// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendCubePoseInputs,
  appendDegreeSliderGroup,
  appendFaceInputs,
  appendFloorModeInput,
  createPane,
  FLOOR_FACES,
  type PregraspPosePane
} from '../pregrasp-pose-shared/ui';

export { FLOOR_FACES } from '../pregrasp-pose-shared/ui';

export interface PregraspPoseDom {
  root: HTMLDivElement;
  pane: PregraspPosePane;
  faceInputs: HTMLInputElement[];
  floorModeInput: HTMLInputElement;
  hingeInput: HTMLInputElement;
  xInput: HTMLInputElement;
  yInput: HTMLInputElement;
  zInput: HTMLInputElement;
  yawInput: HTMLInputElement;
  pitchInput: HTMLInputElement;
  rollInput: HTMLInputElement;
}

export function buildUi(parent: HTMLElement): PregraspPoseDom {
  const root = document.createElement('div');
  root.className = 'visualization pregrasp-pose-viz-root';

  const controls = document.createElement('div');
  controls.className = 'pregrasp-pose-breakdown-viz-controls';

  const floorModeInput = appendFloorModeInput(controls);
  const faceInputs = appendFaceInputs(
    controls, 'pregrasp-pose-cube-face', undefined, FLOOR_FACES
  );
  const {
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  } = appendCubePoseInputs(controls, true);
  const hingeInput = appendDegreeSliderGroup(controls, 'Hinge', 0, 360, 0);

  root.appendChild(controls);

  const pane = createPane('Pregrasp pose', 'combined', 'final', true);
  root.appendChild(pane.element);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root, pane, faceInputs, floorModeInput, hingeInput,
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  };
}
