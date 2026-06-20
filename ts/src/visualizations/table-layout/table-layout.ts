// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { deriveSo101Kinematics } from '../../ik/kinematics';
import { loadWebModel } from '../../web-model';
import { createTableLayoutScene } from './scene';
import {
  buildUi,
  DEFAULT_ROBOT_COUNT,
  DEFAULT_TABLE_LENGTH_CM,
  DEFAULT_TABLE_WIDTH_CM
} from './ui';

export interface TableLayoutVisualization {
  destroy(): void;
}

export interface TableLayoutOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

export async function initializeTableLayoutVisualization(
  parent: HTMLElement,
  options: TableLayoutOptions = {}
): Promise<TableLayoutVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const kinematics = deriveSo101Kinematics(model);

  const ui = buildUi(parent);
  const vizScene = createTableLayoutScene(
    ui.viewport, model, kinematics, options.modelBasePath
  );

  vizScene.setTableSize(DEFAULT_TABLE_WIDTH_CM / 100, DEFAULT_TABLE_LENGTH_CM / 100);
  vizScene.setRobotCount(DEFAULT_ROBOT_COUNT);

  const applyTableSize = (): void => {
    vizScene.setTableSize(
      Number(ui.widthInput.value) / 100,
      Number(ui.lengthInput.value) / 100
    );
  };
  ui.widthInput.addEventListener('input', applyTableSize);
  ui.lengthInput.addEventListener('input', applyTableSize);

  const countListeners = ui.countInputs.map(input => {
    const listener = (): void => {
      if (input.checked) { vizScene.setRobotCount(Number(input.value)); }
    };
    input.addEventListener('change', listener);
    return listener;
  });

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    vizScene.orbitControls.update();
    vizScene.renderer.render(vizScene.scene, vizScene.camera);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      ui.widthInput.removeEventListener('input', applyTableSize);
      ui.lengthInput.removeEventListener('input', applyTableSize);
      for (const [index, input] of ui.countInputs.entries()) {
        input.removeEventListener('change', countListeners[index]);
      }
      vizScene.destroy();
      ui.root.remove();
    }
  };
}
