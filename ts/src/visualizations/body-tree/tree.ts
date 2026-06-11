// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { glbName, loadMesh } from '../../mesh-loader';
import { LINKS, type VisualDefinition } from '../robot/scene';

export interface BodyTreeVisualization {
  destroy(): void;
}

interface MeshMetrics {
  bytes: number;
  triangles: number;
  vertices: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) { return `${bytes} B`; }
  if (bytes < 1024 ** 2) { return `${(bytes / 1024).toFixed(1)} KB`; }
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

function formatCount(count: number): string {
  return new Intl.NumberFormat('en', { notation: 'compact' }).format(count);
}

function metricsLabel(metrics: MeshMetrics): string {
  return `${formatCount(metrics.triangles)} tris · ${formatBytes(metrics.bytes)}`;
}

function requiredElement(root: ParentNode, selector: string): HTMLElement {
  const element = root.querySelector<HTMLElement>(selector);
  if (!element) { throw new Error(`Missing required element ${selector}`); }
  return element;
}

export function initializeBodyTreeVisualization(
  parent: HTMLElement,
  modelBasePath = '/so101_assets'
): Promise<BodyTreeVisualization> {
  const root = document.createElement('div');
  root.className = 'body-tree-root';
  root.innerHTML = `
    <aside class="body-tree-sidebar">
      <header>
        <strong>Mesh complexity</strong>
        <span>${LINKS.length} bodies ·
          ${LINKS.reduce((n, link) => n + link.visuals.length, 0)} geoms</span>
      </header>
      <input class="body-tree-search" type="search"
        placeholder="Filter bodies and geoms…" aria-label="Filter bodies and geoms">
      <div class="body-tree-list"></div>
    </aside>
    <section class="body-tree-inspector">
      <div class="body-tree-viewport"></div>
      <div class="body-tree-info">
        <strong>Loading mesh diagnostics…</strong>
        <span>File size and polygon counts are calculated from the served Meshopt GLBs.</span>
      </div>
    </section>`;
  parent.querySelector('.placeholder')?.replaceWith(root);
  if (!root.parentElement) { parent.appendChild(root); }

  const viewport = requiredElement(root, '.body-tree-viewport');
  const list = requiredElement(root, '.body-tree-list');
  const info = requiredElement(root, '.body-tree-info');
  const search = requiredElement(root, '.body-tree-search') as HTMLInputElement;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(0.48, 0.48, 0.38);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0.1);
  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const light = new THREE.DirectionalLight(0xfff2d6, 3);
  light.position.set(2, 2, 5);
  scene.add(light);
  const grid = new THREE.GridHelper(1, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const printed = new THREE.MeshStandardMaterial({ color: 0xffb000, roughness: 0.55 });
  const motor = new THREE.MeshStandardMaterial({ color: 0x292d32, roughness: 0.7 });
  const selected = new THREE.MeshStandardMaterial({
    color: 0x38bdf8, emissive: 0x075985, roughness: 0.35
  });
  const basePath = modelBasePath.replace(/\/$/, '');
  const bodies = new Map<string, THREE.Group>();
  const meshes = new Map<string, THREE.Mesh>();
  const meshMetrics = new Map<string, MeshMetrics>();
  const bodyMetrics = new Map<string, MeshMetrics>();
  const rows = new Map<string, HTMLElement>();
  const hiddenMeshes = new Set<string>();
  const hiddenBodies = new Set<string>();
  let selectedId = '';

  const select = (id: string, title: string, details: string, object: THREE.Object3D): void => {
    selectedId = id;
    for (const [meshId, mesh] of meshes) {
      const visual = mesh.userData.visual as VisualDefinition;
      mesh.material = meshId === id ? selected : visual.motor === true ? motor : printed;
    }
    for (const [rowId, row] of rows) { row.classList.toggle('selected', rowId === id); }
    info.innerHTML = `<strong>${title}</strong><span>${details}</span>`;
    const box = new THREE.Box3().setFromObject(object);
    if (!box.isEmpty()) {
      const center = box.getCenter(new THREE.Vector3());
      const size = Math.max(box.getSize(new THREE.Vector3()).length(), 0.08);
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size, size, size * 0.75));
      controls.update();
    }
  };

  for (const link of LINKS) {
    const group = new THREE.Group();
    bodies.set(link.name, group);
    if (link.joint) {
      const origin = new THREE.Group();
      origin.position.set(...link.joint.position);
      origin.rotation.set(...link.joint.rotation, 'ZYX');
      origin.add(group);
      bodies.get(link.joint.parent)?.add(origin);
    } else { scene.add(group); }

    const bodyRow = document.createElement('details');
    bodyRow.className = 'body-tree-body';
    bodyRow.open = true;
    const summary = document.createElement('summary');
    summary.innerHTML = `<span class="body-tree-icon">B</span>
      <span>${link.name}</span><small>loading…</small>
      <button class="body-tree-toggle" type="button" title="Hide body geoms"
        aria-label="Toggle ${link.name} geoms" aria-pressed="false">●</button>`;
    bodyRow.appendChild(summary);
    rows.set(link.name, summary);
    summary.addEventListener('click', () => {
      const joint = link.joint;
      const metrics = bodyMetrics.get(link.name);
      const hierarchy = joint ? `joint ${joint.name} · parent ${joint.parent}` : 'root body';
      select(link.name, link.name, metrics
        ? `${hierarchy} · ${metricsLabel(metrics)} · ${formatCount(metrics.vertices)} vertices`
        : `${hierarchy} · loading mesh diagnostics…`, group);
    });
    const bodyToggle = requiredElement(summary, '.body-tree-toggle') as HTMLButtonElement;
    bodyToggle.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const visible = hiddenBodies.has(link.name);
      if (visible) { hiddenBodies.delete(link.name); } else { hiddenBodies.add(link.name); }
      for (const [meshId, mesh] of meshes) {
        if (meshId.startsWith(`${link.name}/`)) {
          mesh.visible = visible && !hiddenMeshes.has(meshId);
        }
      }
      bodyToggle.classList.toggle('off', !visible);
      bodyToggle.ariaPressed = String(!visible);
      bodyToggle.title = visible ? 'Hide body geoms' : 'Show body geoms';
    });

    link.visuals.forEach((visual, index) => {
      const id = `${link.name}/${index}`;
      const row = document.createElement('button');
      row.className = 'body-tree-geom';
      row.type = 'button';
      row.innerHTML = `<span class="body-tree-icon">G</span>
        <span>${visual.mesh}</span>
        <small>loading…</small>
        <span class="body-tree-toggle" role="switch" tabindex="0"
          title="Hide geom" aria-label="Hide ${visual.mesh}" aria-checked="true">●</span>`;
      bodyRow.appendChild(row);
      rows.set(id, row);
      loadMesh(`${basePath}/${glbName(visual.mesh)}`).then(({ bytes, geometry }) => {
        const metrics = {
          bytes,
          triangles: geometry.index
            ? geometry.index.count / 3
            : geometry.getAttribute('position').count / 3,
          vertices: geometry.getAttribute('position').count
        };
        meshMetrics.set(id, metrics);
        const total = bodyMetrics.get(link.name) ?? { bytes: 0, triangles: 0, vertices: 0 };
        total.bytes += metrics.bytes;
        total.triangles += metrics.triangles;
        total.vertices += metrics.vertices;
        bodyMetrics.set(link.name, total);
        requiredElement(row, 'small').textContent = metricsLabel(metrics);
        requiredElement(summary, 'small').textContent = metricsLabel(total);
        const mesh = new THREE.Mesh(geometry, visual.motor === true ? motor : printed);
        mesh.position.set(...visual.position);
        mesh.rotation.set(...visual.rotation, 'ZYX');
        mesh.userData.visual = visual;
        group.add(mesh);
        meshes.set(id, mesh);
        mesh.visible = !hiddenBodies.has(link.name) && !hiddenMeshes.has(id);
        if (selectedId === id) { mesh.material = selected; }
        if (meshMetrics.size === LINKS.reduce((n, item) => n + item.visuals.length, 0)) {
          const all = [...meshMetrics.values()].reduce((sum, item) => ({
            bytes: sum.bytes + item.bytes,
            triangles: sum.triangles + item.triangles,
            vertices: sum.vertices + item.vertices
          }), { bytes: 0, triangles: 0, vertices: 0 });
          info.innerHTML = `<strong>Full robot mesh set</strong>
            <span>${metricsLabel(all)} · ${formatCount(all.vertices)} vertices ·
              ${meshMetrics.size} GLB instances</span>`;
        }
      }).catch((error: unknown) => {
        requiredElement(row, 'small').textContent = 'load error';
        console.error(error);
      });
      row.addEventListener('click', () => {
        const mesh = meshes.get(id) ?? group;
        const type = visual.motor === true ? 'motor' : 'printed part';
        const metrics = meshMetrics.get(id);
        select(id, visual.mesh, metrics
          ? `${type} · ${metricsLabel(metrics)} · ${formatCount(metrics.vertices)} vertices`
          : `${type} · loading mesh diagnostics…`, mesh);
      });
      const geomToggle = requiredElement(row, '.body-tree-toggle');
      const toggleGeom = (event: Event): void => {
        event.preventDefault();
        event.stopPropagation();
        const visible = hiddenMeshes.has(id);
        if (visible) { hiddenMeshes.delete(id); } else { hiddenMeshes.add(id); }
        const mesh = meshes.get(id);
        if (mesh) { mesh.visible = visible && !hiddenBodies.has(link.name); }
        geomToggle.classList.toggle('off', !visible);
        geomToggle.setAttribute('aria-checked', String(visible));
        geomToggle.title = visible ? 'Hide geom' : 'Show geom';
      };
      geomToggle.addEventListener('click', toggleGeom);
      geomToggle.addEventListener('keydown', event => {
        if (event.key === ' ' || event.key === 'Enter') { toggleGeom(event); }
      });
    });
    list.appendChild(bodyRow);
  }

  const filter = (): void => {
    const query = search.value.trim().toLowerCase();
    for (const details of list.querySelectorAll<HTMLElement>('.body-tree-body')) {
      details.hidden = query !== '' && !details.textContent.toLowerCase().includes(query);
    }
  };
  search.addEventListener('input', filter);
  const resize = (): void => {
    const width = viewport.clientWidth || 600;
    const height = viewport.clientHeight || 520;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(viewport);
  resize();
  let frame = 0;
  let destroyed = false;
  const animate = (): void => {
    if (destroyed) { return; }
    frame = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

  return Promise.resolve({ destroy(): void {
    destroyed = true;
    cancelAnimationFrame(frame);
    observer.disconnect();
    controls.dispose();
    renderer.dispose();
    printed.dispose();
    motor.dispose();
    selected.dispose();
    root.remove();
  } });
}
