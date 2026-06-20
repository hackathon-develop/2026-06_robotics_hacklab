// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Isolated build for the standalone table-layout page. Outputs to
// dist-table-layout/ (robot assets in public/ are copied automatically) so the
// folder can be deployed on its own as a static site.

import { resolve } from 'node:path';

import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist-table-layout',
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(__dirname, 'table-layout.html')
    }
  }
});
