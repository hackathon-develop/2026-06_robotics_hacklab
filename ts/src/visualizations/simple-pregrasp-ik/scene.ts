// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import {
  buildWebModel,
  setJointAngle,
  type WebModel
} from '../../web-model';
import {
  createCubeBody,
  createWorldFromCubeMatrix,
  type CubePose
} from '../pregrasp-pose-shared/body-factories';
import { createBodyMaterials } from '../pregrasp-pose-shared/materials';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from '../pregrasp-pose-shared/ui';

// Open the fixed/moving jaws to the same angle the pregrasp-pose
// visualizations use, so the rendered gripper matches the grasp geometry.
const GRIPPER_OPEN_ANGLE = Math.PI / 3;

// Annular sector (on the ground) marking where a cube center is graspable for
// any yaw. Built from the closed-form workspace; see src/ik/workspace.ts.
export interface WorkspaceOverlaySpec {
  center: THREE.Vector2;
  innerRadius: number;
  outerRadius: number;
  thetaStart: number;
  thetaLength: number;
}

export interface SimplePregraspIkScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  setJoint(name: string, radians: number): void;
  updateCubePose(pose: CubePose): void;
  resize(): void;
  destroy(): void;
}

export function createSimplePregraspIkScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets',
  workspace?: WorkspaceOverlaySpec
): SimplePregraspIkScene {
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
  camera.position.set(0.55, 0.45, 0.42);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0.2, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(0.8, 16, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(0.2, 0, 0);
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.05));

  // Ground overlay marking the any-yaw graspable cube-center region.
  let workspaceGeometry: THREE.RingGeometry | undefined;
  let workspaceMaterial: THREE.MeshBasicMaterial | undefined;
  if (workspace) {
    workspaceGeometry = new THREE.RingGeometry(
      workspace.innerRadius, workspace.outerRadius, 96, 1,
      workspace.thetaStart, workspace.thetaLength
    );
    workspaceMaterial = new THREE.MeshBasicMaterial({
      color: 0x2fa84f,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const workspaceMesh = new THREE.Mesh(workspaceGeometry, workspaceMaterial);
    // RingGeometry lies in the XY plane; lift slightly to avoid z-fighting.
    workspaceMesh.position.set(workspace.center.x, workspace.center.y, 0.0006);
    workspaceMesh.renderOrder = -1;
    scene.add(workspaceMesh);
  }

  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);
  setJointAngle(model, builtModel.jointPivots, 'gripper', GRIPPER_OPEN_ANGLE);

  const materials = createBodyMaterials();
  const cubePart = createCubeBody(materials);
  scene.add(cubePart.body);

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
    setJoint(name: string, radians: number): void {
      setJointAngle(model, builtModel.jointPivots, name, radians);
    },
    updateCubePose(pose: CubePose): void {
      createWorldFromCubeMatrix(pose).decompose(
        cubePart.body.position, cubePart.body.quaternion, cubePart.body.scale
      );
    },
    resize,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      for (const mats of builtModel.materialsByName.values()) {
        for (const mat of mats) { mat.dispose(); }
      }
      cubePart.destroy();
      materials.destroy();
      workspaceGeometry?.dispose();
      workspaceMaterial?.dispose();
    }
  };
}
