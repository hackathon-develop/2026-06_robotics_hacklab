// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 520;

export interface JointControl {
  input: HTMLInputElement;
  value: HTMLOutputElement;
}

export interface RobotVizDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  controls: Map<string, JointControl>;
  resetButton: HTMLButtonElement;
}

export interface JointControlDefinition {
  name: string;
  label: string;
  lower: number;
  upper: number;
  value: number;
}

export function buildUi(
  parent: HTMLElement,
  joints: JointControlDefinition[]
): RobotVizDom {
  const root = document.createElement('div');
  root.className = 'visualization robot-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'robot-viz-viewport';
  root.appendChild(viewport);

  const panel = document.createElement('div');
  panel.className = 'robot-viz-controls';

  const header = document.createElement('div');
  header.className = 'robot-viz-controls-header';
  const title = document.createElement('strong');
  title.textContent = 'Joint angles';
  const resetButton = document.createElement('button');
  resetButton.type = 'button';
  resetButton.textContent = 'Reset pose';
  header.append(title, resetButton);
  panel.appendChild(header);

  const controls = new Map<string, JointControl>();
  for (const joint of joints) {
    const row = document.createElement('label');
    row.className = 'robot-viz-joint';

    const label = document.createElement('span');
    label.textContent = joint.label;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(joint.lower);
    input.max = String(joint.upper);
    input.step = '0.01';
    input.value = String(joint.value);

    const value = document.createElement('output');
    value.textContent = formatDegrees(joint.value);

    row.append(label, input, value);
    panel.appendChild(row);
    controls.set(joint.name, { input, value });
  }

  root.appendChild(panel);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return { root, viewport, controls, resetButton };
}

export function formatDegrees(radians: number): string {
  return `${Math.round(radians * 180 / Math.PI)}°`;
}
