// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import type { So101Kinematics } from '../../ik/kinematics';
import { buildWebModel, type WebModel } from '../../web-model';
import {
  addWorkspaceOverlaysToScene,
  buildWorkspaceOverlaySpecs
} from '../workspace-overlay';
import {
  createXyMultiDragControls,
  type XyDragControls,
  type XyDragTarget
} from '../xy-drag-controls';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

export const MAX_ROBOTS = 4;

// Distinct annulus colors so each robot's reachable workspace is identifiable.
const ROBOT_COLORS = [0xff7700, 0x2288ff, 0x1faa5a, 0xb247d6];

// A robot clamped to one of the four table edges. `edge` selects the side and
// `frac` is the normalized position (0..1) along that edge. The robot always
// faces inward (its +x reach axis points toward the table interior).
type Edge = 0 | 1 | 2 | 3; // 0:+X, 1:-X, 2:+Y, 3:-Y
interface Placement {
  edge: Edge;
  frac: number;
}

// Default placements: spread the first robots across different edges.
const DEFAULT_PLACEMENTS: Placement[] = [
  { edge: 3, frac: 0.5 }, // near edge (−Y)
  { edge: 2, frac: 0.5 }, // far edge (+Y)
  { edge: 1, frac: 0.5 }, // left edge (−X)
  { edge: 0, frac: 0.5 } // right edge (+X)
];

interface RobotSlot {
  group: THREE.Group;
  placement: Placement;
  disposeModel(): void;
  disposeOverlays(): void;
}

// Inward normal of each edge (pointing toward the table interior).
function edgeNormal(edge: Edge): THREE.Vector2 {
  switch (edge) {
  case 0: return new THREE.Vector2(-1, 0);
  case 1: return new THREE.Vector2(1, 0);
  case 2: return new THREE.Vector2(0, -1);
  case 3: return new THREE.Vector2(0, 1);
  }
}

// World point on `edge` at fraction `frac`, given half-extents (hx, hy).
function edgePoint(edge: Edge, frac: number, hx: number, hy: number): THREE.Vector2 {
  switch (edge) {
  case 0: return new THREE.Vector2(hx, -hy + frac * 2 * hy);
  case 1: return new THREE.Vector2(-hx, -hy + frac * 2 * hy);
  case 2: return new THREE.Vector2(-hx + frac * 2 * hx, hy);
  case 3: return new THREE.Vector2(-hx + frac * 2 * hx, -hy);
  }
}

// Snap an arbitrary point to the nearest table edge, returning the edge and the
// clamped fraction along it.
function snapToEdge(x: number, y: number, hx: number, hy: number): Placement {
  const candidates: Placement[] = [
    { edge: 0, frac: (y + hy) / (2 * hy) },
    { edge: 1, frac: (y + hy) / (2 * hy) },
    { edge: 2, frac: (x + hx) / (2 * hx) },
    { edge: 3, frac: (x + hx) / (2 * hx) }
  ];
  let best: Placement | null = null;
  let bestDist = Infinity;
  for (const candidate of candidates) {
    const frac = Math.min(1, Math.max(0, candidate.frac));
    const point = edgePoint(candidate.edge, frac, hx, hy);
    const dist = (point.x - x) ** 2 + (point.y - y) ** 2;
    if (dist < bestDist) {
      bestDist = dist;
      best = { edge: candidate.edge, frac };
    }
  }
  return best ?? { edge: 0, frac: 0.5 };
}

export interface TableLayoutScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  setTableSize(width: number, length: number): void;
  setRobotCount(count: number): void;
  resize(): void;
  destroy(): void;
}

export function createTableLayoutScene(
  viewport: HTMLElement,
  model: WebModel,
  kinematics: So101Kinematics,
  modelBasePath = '/so101_assets'
): TableLayoutScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(
    42, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100
  );
  camera.up.set(0, 0, 1);
  camera.position.set(0.9, -0.9, 0.95);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0, 0, 0);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  scene.add(directionalLight);

  // Table surface: top face at z = 0 so robots stand on it. Rebuilt on resize.
  let halfWidth = 0.4;
  let halfLength = 0.3;
  const tableThickness = 0.02;
  const tableMaterial = new THREE.MeshStandardMaterial({
    color: 0xcdb892, roughness: 0.85, metalness: 0
  });
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x7a6a4a });
  let tableMesh: THREE.Mesh | null = null;
  let tableEdges: THREE.LineSegments | null = null;

  function rebuildTable(): void {
    if (tableMesh) {
      tableMesh.removeFromParent();
      tableMesh.geometry.dispose();
    }
    if (tableEdges) {
      tableEdges.removeFromParent();
      tableEdges.geometry.dispose();
    }
    const geo = new THREE.BoxGeometry(
      halfWidth * 2, halfLength * 2, tableThickness
    );
    tableMesh = new THREE.Mesh(geo, tableMaterial);
    tableMesh.position.set(0, 0, -tableThickness / 2);
    scene.add(tableMesh);
    // Highlight the top rectangle outline: that perimeter is where robots clamp.
    const top = new THREE.PlaneGeometry(halfWidth * 2, halfLength * 2);
    tableEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(top), edgeMaterial
    );
    tableEdges.position.set(0, 0, 0.0005);
    scene.add(tableEdges);
    top.dispose();
  }
  rebuildTable();

  const overlaySpecs = buildWorkspaceOverlaySpecs(kinematics);

  // Robot pool: built lazily and reused. `setRobotCount` toggles visibility.
  const slots: RobotSlot[] = [];

  function buildSlot(index: number): RobotSlot {
    const group = new THREE.Group();
    const built = buildWebModel(model, modelBasePath);
    group.add(built.root);

    const color = ROBOT_COLORS[index % ROBOT_COLORS.length];
    const tinted = overlaySpecs.map(spec => ({ ...spec, color }));
    const disposeOverlays = addWorkspaceOverlaysToScene(group, tinted);

    scene.add(group);
    return {
      group,
      placement: { ...DEFAULT_PLACEMENTS[index] },
      disposeModel(): void {
        for (const mats of built.materialsByName.values()) {
          for (const mat of mats) { mat.dispose(); }
        }
      },
      disposeOverlays
    };
  }

  // Position a robot group on its edge, facing the table interior.
  function applyPlacement(slot: RobotSlot): void {
    const { edge, frac } = slot.placement;
    const point = edgePoint(edge, frac, halfWidth, halfLength);
    const normal = edgeNormal(edge);
    slot.group.position.set(point.x, point.y, 0);
    // Model +x is the reach axis; aim it along the inward normal.
    slot.group.rotation.z = Math.atan2(normal.y, normal.x);
  }

  // Dimension annotations along each table edge: how far each robot sits from
  // the two ends of the side it is clamped to. Rebuilt whenever a placement or
  // the table changes.
  const EDGE_DIM_COLOR = 0x1e293b;
  const measurementsGroup = new THREE.Group();
  scene.add(measurementsGroup);
  const edgeLineMaterial = new THREE.LineBasicMaterial({ color: EDGE_DIM_COLOR });
  const dimGeometries: THREE.BufferGeometry[] = [];
  const dimTextures: THREE.Texture[] = [];
  const dimSpriteMaterials: THREE.SpriteMaterial[] = [];

  let activeCount = 0;

  function clearMeasurements(): void {
    measurementsGroup.clear();
    for (const geo of dimGeometries) { geo.dispose(); }
    for (const tex of dimTextures) { tex.dispose(); }
    for (const mat of dimSpriteMaterials) { mat.dispose(); }
    dimGeometries.length = 0;
    dimTextures.length = 0;
    dimSpriteMaterials.length = 0;
  }

  function addDimLine(
    a: THREE.Vector2, b: THREE.Vector2, material: THREE.LineBasicMaterial
  ): void {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(a.x, a.y, 0.006),
      new THREE.Vector3(b.x, b.y, 0.006)
    ]);
    dimGeometries.push(geo);
    measurementsGroup.add(new THREE.Line(geo, material));
  }

  // A camera-facing text label rendered to a canvas texture.
  function addDimLabel(at: THREE.Vector2, text: string): void {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.font = 'bold 52px sans-serif';
      const metrics = ctx.measureText(text);
      const padding = 18;
      const boxW = metrics.width + padding * 2;
      const boxH = 72;
      const x0 = (canvas.width - boxW) / 2;
      const y0 = (canvas.height - boxH) / 2;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.fillRect(x0, y0, boxW, boxH);
      ctx.strokeStyle = 'rgba(30, 41, 59, 0.35)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, boxW, boxH);
      ctx.fillStyle = '#0f172a';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    }
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture, depthTest: false, transparent: true
    });
    dimTextures.push(texture);
    dimSpriteMaterials.push(material);
    const sprite = new THREE.Sprite(material);
    // World height of the label; width keeps the canvas aspect ratio.
    const height = 0.05;
    sprite.scale.set(height * (canvas.width / canvas.height), height, 1);
    sprite.position.set(at.x, at.y, 0.03);
    sprite.renderOrder = 10;
    measurementsGroup.add(sprite);
  }

  const cm = (meters: number): string => `${Math.round(meters * 100)} cm`;
  const mid = (a: THREE.Vector2, b: THREE.Vector2): THREE.Vector2 =>
    new THREE.Vector2((a.x + b.x) / 2, (a.y + b.y) / 2);

  function rebuildMeasurements(): void {
    clearMeasurements();
    const active = slots.slice(0, activeCount);

    // For every side that has robots, walk along the edge from one corner to
    // the other through the robots (sorted by position), dimensioning each
    // gap: corner→robot, robot→robot, robot→corner. All segments are collinear
    // with the edge, so nothing crosses the table.
    for (let edge = 0 as Edge; edge < 4; edge = (edge + 1) as Edge) {
      const fracs = active
        .filter(slot => slot.placement.edge === edge)
        .map(slot => slot.placement.frac)
        .sort((a, b) => a - b);
      if (fracs.length === 0) { continue; }

      const inset = edgeNormal(edge).multiplyScalar(0.04);
      const edgeLength = edge === 0 || edge === 1 ? halfLength * 2 : halfWidth * 2;
      const stops = [0, ...fracs, 1];
      for (let i = 0; i < stops.length - 1; i++) {
        const f0 = stops[i];
        const f1 = stops[i + 1];
        const a = edgePoint(edge, f0, halfWidth, halfLength).add(inset);
        const b = edgePoint(edge, f1, halfWidth, halfLength).add(inset);
        addDimLine(a, b, edgeLineMaterial);
        addDimLabel(mid(a, b), cm((f1 - f0) * edgeLength));
      }
    }
  }

  // Coalesce rapid placement changes (e.g. during a drag) into one rebuild per
  // frame.
  let destroyed = false;
  let measurementsPending = false;
  function requestMeasurements(): void {
    if (measurementsPending) { return; }
    measurementsPending = true;
    requestAnimationFrame(() => {
      measurementsPending = false;
      if (!destroyed) { rebuildMeasurements(); }
    });
  }

  // Dolly the camera so the whole table fits, preserving the current orbit
  // angle.
  function fitCamera(): void {
    const maxDim = Math.max(halfWidth * 2, halfLength * 2);
    const distance = Math.max(0.6, maxDim * 1.25);
    const direction = camera.position.clone().sub(orbitControls.target);
    if (direction.lengthSq() < 1e-9) { direction.set(0.9, -0.9, 0.95); }
    direction.normalize().multiplyScalar(distance);
    camera.position.copy(orbitControls.target).add(direction);
    orbitControls.update();
  }

  let dragControls: XyDragControls | null = null;

  function rebuildDragControls(count: number): void {
    dragControls?.destroy();
    const targets: XyDragTarget[] = slots.slice(0, count).map(slot => ({
      object: slot.group,
      onDrag(x: number, y: number): void {
        slot.placement = snapToEdge(x, y, halfWidth, halfLength);
        applyPlacement(slot);
        requestMeasurements();
      }
    }));
    dragControls = createXyMultiDragControls({
      camera, domElement: renderer.domElement, targets, orbitControls
    });
  }

  function setRobotCount(count: number): void {
    const clamped = Math.min(MAX_ROBOTS, Math.max(1, count));
    while (slots.length < clamped) {
      const slot = buildSlot(slots.length);
      applyPlacement(slot);
      slots.push(slot);
    }
    for (const [index, slot] of slots.entries()) {
      slot.group.visible = index < clamped;
    }
    activeCount = clamped;
    rebuildDragControls(clamped);
    rebuildMeasurements();
  }

  function setTableSize(width: number, length: number): void {
    halfWidth = width / 2;
    halfLength = length / 2;
    rebuildTable();
    for (const slot of slots) { applyPlacement(slot); }
    fitCamera();
    rebuildMeasurements();
  }

  function resize(): void {
    const width = viewport.clientWidth || CANVAS_WIDTH;
    const height = viewport.clientHeight || CANVAS_HEIGHT;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  resize();

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    setTableSize,
    setRobotCount,
    resize,
    destroy(): void {
      destroyed = true;
      dragControls?.destroy();
      orbitControls.dispose();
      renderer.dispose();
      for (const slot of slots) {
        slot.disposeOverlays();
        slot.disposeModel();
      }
      clearMeasurements();
      edgeLineMaterial.dispose();
      tableMesh?.geometry.dispose();
      tableEdges?.geometry.dispose();
      tableMaterial.dispose();
      edgeMaterial.dispose();
    }
  };
}
