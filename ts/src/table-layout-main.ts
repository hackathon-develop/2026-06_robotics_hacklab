// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Standalone entry point: mounts only the table-layout visualization. Built as
// an isolated bundle (see vite.config.table-layout.ts) for static hosting.

import './style.css';

import {
  initTableLayoutVisualization,
  type TableLayoutVisualization
} from './visualizations/table-layout';

let visualization: TableLayoutVisualization | null = null;

function initialize(): void {
  const panel = document.getElementById('table-layout-visualization');
  if (!panel) { return; }
  visualization?.destroy();
  visualization = null;
  void initTableLayoutVisualization(panel).then(viz => {
    visualization = viz;
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
