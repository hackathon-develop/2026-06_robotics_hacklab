// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { appendSliderGroup } from '../pregrasp-pose-shared/ui';
import { MAX_ROBOTS } from './scene';

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 420;

// Default table footprint and robot count.
export const DEFAULT_TABLE_WIDTH_CM = 80;
export const DEFAULT_TABLE_LENGTH_CM = 60;
export const DEFAULT_ROBOT_COUNT = 2;

export interface TableLayoutDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  widthInput: HTMLInputElement;
  lengthInput: HTMLInputElement;
  countInputs: HTMLInputElement[];
}

function appendRobotCountInputs(parent: HTMLElement): HTMLInputElement[] {
  const group = document.createElement('div');
  group.className = 'pregrasp-pose-breakdown-viz-controls-group';
  const groupLabel = document.createElement('span');
  groupLabel.textContent = 'Robots';
  const options = document.createElement('div');
  options.className = 'pregrasp-pose-breakdown-viz-face-options';
  const inputs: HTMLInputElement[] = [];
  for (let count = 1; count <= MAX_ROBOTS; count++) {
    const wrapper = document.createElement('label');
    wrapper.className = 'pregrasp-pose-breakdown-viz-face-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'table-layout-robot-count';
    input.value = String(count);
    input.checked = count === DEFAULT_ROBOT_COUNT;
    const label = document.createElement('span');
    label.textContent = String(count);
    wrapper.append(input, label);
    options.appendChild(wrapper);
    inputs.push(input);
  }
  group.append(groupLabel, options);
  parent.appendChild(group);
  return inputs;
}

export function buildUi(parent: HTMLElement): TableLayoutDom {
  const root = document.createElement('div');
  root.className = 'visualization table-layout-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'table-layout-viz-viewport';

  const controls = document.createElement('div');
  controls.className = 'table-layout-viz-controls';

  const countInputs = appendRobotCountInputs(controls);
  const widthInput = appendSliderGroup(
    controls, 'Width', 30, 250, DEFAULT_TABLE_WIDTH_CM, 1, ' cm'
  );
  const lengthInput = appendSliderGroup(
    controls, 'Length', 30, 400, DEFAULT_TABLE_LENGTH_CM, 1, ' cm'
  );

  const hint = document.createElement('p');
  hint.className = 'table-layout-viz-hint';
  hint.textContent =
    'Drag a robot to slide it along the table edges; it always snaps to the ' +
    'nearest side. The colored rings show each arm’s reachable workspace.';
  controls.appendChild(hint);

  const layout = document.createElement('div');
  layout.className = 'table-layout-viz-layout';
  layout.append(viewport, controls);
  root.appendChild(layout);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return { root, viewport, widthInput, lengthInput, countInputs };
}
