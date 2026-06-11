// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { createRobotScene, JOINTS } from './scene';
import { buildUi, formatDegrees } from './ui';

export interface RobotVisualization {
  destroy(): void;
}

export interface RobotVisualizationOptions {
  modelBasePath?: string;
}

export function initializeRobotVisualization(
  parent: HTMLElement,
  options: RobotVisualizationOptions = {}
): Promise<RobotVisualization> {
  const ui = buildUi(parent, [...JOINTS]);
  const vizScene = createRobotScene(ui.viewport, options.modelBasePath);
  const { renderer, camera, scene, orbitControls } = vizScene;
  const listeners: (() => void)[] = [];

  for (const joint of JOINTS) {
    const control = ui.controls.get(joint.name);
    if (!control) {continue;}
    const update = (): void => {
      const value = Number(control.input.value);
      control.value.textContent = formatDegrees(value);
      vizScene.setJoint(joint.name, value);
    };
    control.input.addEventListener('input', update);
    listeners.push(() => { control.input.removeEventListener('input', update); });
  }

  const reset = (): void => {
    for (const joint of JOINTS) {
      const control = ui.controls.get(joint.name);
      if (!control) {continue;}
      control.input.value = String(joint.value);
      control.input.dispatchEvent(new Event('input'));
    }
  };
  ui.resetButton.addEventListener('click', reset);
  listeners.push(() => { ui.resetButton.removeEventListener('click', reset); });

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) {return;}
    animationFrameId = window.requestAnimationFrame(animate);
    orbitControls.update();
    renderer.render(scene, camera);
  }
  animate();

  return Promise.resolve({
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      for (const removeListener of listeners) {removeListener();}
      vizScene.destroy();
      ui.root.remove();
    }
  });
}
