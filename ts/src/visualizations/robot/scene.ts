// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { glbName, loadMesh } from '../../mesh-loader';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

export interface VisualDefinition {
  mesh: string;
  position: [number, number, number];
  rotation: [number, number, number];
  motor?: boolean;
}

export interface LinkDefinition {
  name: string;
  visuals: VisualDefinition[];
  joint?: {
    name: string;
    parent: string;
    position: [number, number, number];
    rotation: [number, number, number];
  };
}

export const JOINTS = [
  { name: 'shoulder_pan', label: 'Shoulder pan', lower: -1.91986, upper: 1.91986, value: 0 },
  { name: 'shoulder_lift', label: 'Shoulder lift', lower: -1.74533, upper: 1.74533, value: 0 },
  { name: 'elbow_flex', label: 'Elbow flex', lower: -1.69, upper: 1.69, value: 0 },
  { name: 'wrist_flex', label: 'Wrist flex', lower: -1.65806, upper: 1.65806, value: 0 },
  { name: 'wrist_roll', label: 'Wrist roll', lower: -2.74385, upper: 2.84121, value: 0 },
  { name: 'gripper', label: 'Gripper', lower: -0.174533, upper: 1.74533, value: 0 }
] as const;

export const LINKS: LinkDefinition[] = [
  /* eslint-disable max-len -- Keep calibrated URDF transforms readable as single records. */
  {
    name: 'base_link',
    visuals: [
      { mesh: 'base_motor_holder_so101_v1.stl', position: [-0.00636471, -0.0000994414, -0.0024], rotation: [1.5708, 0, 1.5708] },
      { mesh: 'base_so101_v2.stl', position: [-0.00636471, 0, -0.0024], rotation: [1.5708, 0, 1.5708] },
      { mesh: 'sts3215_03a_v1.stl', position: [0.0263353, 0, 0.0437], rotation: [0, 0, 0], motor: true },
      { mesh: 'waveshare_mounting_plate_so101_v2.stl', position: [-0.0309827, -0.000199441, 0.0474], rotation: [1.5708, 0, 1.5708] }
    ]
  },
  {
    name: 'shoulder_link',
    joint: { name: 'shoulder_pan', parent: 'base_link', position: [0.0388353, 0, 0.0624], rotation: [3.14159, 0, -3.14159] },
    visuals: [
      { mesh: 'sts3215_03a_v1.stl', position: [-0.0303992, 0.000422241, -0.0417], rotation: [1.5708, 1.5708, 0], motor: true },
      { mesh: 'motor_holder_so101_base_v1.stl', position: [-0.0675992, -0.000177759, 0.0158499], rotation: [1.5708, -1.5708, 0] },
      { mesh: 'rotation_pitch_so101_v1.stl', position: [0.0122008, 0.0000222413, 0.0464], rotation: [-1.5708, 0, 0] }
    ]
  },
  {
    name: 'upper_arm_link',
    joint: { name: 'shoulder_lift', parent: 'shoulder_link', position: [-0.0303992, -0.0182778, -0.0542], rotation: [-1.5708, -1.5708, 0] },
    visuals: [
      { mesh: 'sts3215_03a_v1.stl', position: [-0.11257, -0.0155, 0.0187], rotation: [-3.14159, 0, -1.5708], motor: true },
      { mesh: 'upper_arm_so101_v1.stl', position: [-0.065085, 0.012, 0.0182], rotation: [3.14159, 0, 0] }
    ]
  },
  {
    name: 'lower_arm_link',
    joint: { name: 'elbow_flex', parent: 'upper_arm_link', position: [-0.11257, -0.028, 0], rotation: [0, 0, 1.5708] },
    visuals: [
      { mesh: 'under_arm_so101_v1.stl', position: [-0.0648499, -0.032, 0.0182], rotation: [3.14159, 0, 0] },
      { mesh: 'motor_holder_so101_wrist_v1.stl', position: [-0.0648499, -0.032, 0.018], rotation: [-3.14159, 0, 0] },
      { mesh: 'sts3215_03a_v1.stl', position: [-0.1224, 0.0052, 0.0187], rotation: [-3.14159, 0, -3.14159], motor: true }
    ]
  },
  {
    name: 'wrist_link',
    joint: { name: 'wrist_flex', parent: 'lower_arm_link', position: [-0.1349, 0.0052, 0], rotation: [0, 0, -1.5708] },
    visuals: [
      { mesh: 'sts3215_03a_no_horn_v1.stl', position: [0, -0.0424, 0.0306], rotation: [1.5708, 1.5708, 0], motor: true },
      { mesh: 'wrist_roll_pitch_so101_v2.stl', position: [0, -0.028, 0.0181], rotation: [-1.5708, -1.5708, 0] }
    ]
  },
  {
    name: 'gripper_link',
    joint: { name: 'wrist_roll', parent: 'wrist_link', position: [0, -0.0611, 0.0181], rotation: [1.5708, 0.0486795, 3.14159] },
    visuals: [
      { mesh: 'sts3215_03a_v1.stl', position: [0.0077, 0.0001, -0.0234], rotation: [-1.5708, 0, 0], motor: true },
      { mesh: 'wrist_roll_follower_so101_v1.stl', position: [0, -0.000218214, 0.000949706], rotation: [-3.14159, 0, 0] }
    ]
  },
  {
    name: 'moving_jaw_link',
    joint: { name: 'gripper', parent: 'gripper_link', position: [0.0202, 0.0188, -0.0234], rotation: [1.5708, 0, 0] },
    visuals: [
      { mesh: 'moving_jaw_so101_v1.stl', position: [0, 0, 0.0189], rotation: [0, 0, 0] }
    ]
  }
  /* eslint-enable max-len */
];

export interface RobotScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  setJoint(name: string, radians: number): void;
  resize(): void;
  destroy(): void;
}

export function createRobotScene(
  viewport: HTMLElement,
  modelBasePath = '/so101_assets'
): RobotScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(42, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(0.48, 0.48, 0.38);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(1, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const printedMaterial = new THREE.MeshStandardMaterial({ color: 0xffb000, roughness: 0.55 });
  const motorMaterial = new THREE.MeshStandardMaterial({ color: 0x292d32, roughness: 0.7 });
  const basePath = modelBasePath.endsWith('/') ? modelBasePath.slice(0, -1) : modelBasePath;
  const links = new Map<string, THREE.Group>();
  const jointPivots = new Map<string, THREE.Group>();

  for (const link of LINKS) {
    const linkGroup = new THREE.Group();
    links.set(link.name, linkGroup);

    if (link.joint) {
      const parent = links.get(link.joint.parent);
      if (!parent) {
        throw new Error(`Missing parent link ${link.joint.parent}`);
      }
      const origin = new THREE.Group();
      origin.position.set(...link.joint.position);
      origin.rotation.set(...link.joint.rotation, 'ZYX');
      const pivot = new THREE.Group();
      origin.add(pivot);
      pivot.add(linkGroup);
      parent.add(origin);
      jointPivots.set(link.joint.name, pivot);
    } else {
      scene.add(linkGroup);
    }

    for (const visual of link.visuals) {
      loadMesh(`${basePath}/${glbName(visual.mesh)}`).then(({ geometry }) => {
        const material = visual.motor === true ? motorMaterial : printedMaterial;
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...visual.position);
        mesh.rotation.set(...visual.rotation, 'ZYX');
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        linkGroup.add(mesh);
      }).catch(console.error);
    }
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
    setJoint(name: string, radians: number): void {
      const pivot = jointPivots.get(name);
      if (pivot) {
        pivot.rotation.z = radians;
      }
    },
    resize,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      printedMaterial.dispose();
      motorMaterial.dispose();
    }
  };
}
