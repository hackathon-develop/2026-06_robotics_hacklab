// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { MeshoptDecoder } from 'meshoptimizer';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export interface LoadedMesh {
  bytes: number;
  geometry: THREE.BufferGeometry;
}

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const cache = new Map<string, Promise<LoadedMesh>>();

export function glbName(mesh: string): string {
  return mesh.replace(/\.stl$/i, '.glb');
}

export function loadMesh(url: string): Promise<LoadedMesh> {
  const cached = cache.get(url);
  if (cached) { return cached; }

  const promise = fetch(url).then(async response => {
    if (!response.ok) { throw new Error(`Unable to load ${url}: ${response.status}`); }
    const buffer = await response.arrayBuffer();
    const gltf = await loader.parseAsync(buffer, '');
    const geometries: THREE.BufferGeometry[] = [];
    gltf.scene.updateMatrixWorld(true);
    gltf.scene.traverse(object => {
      if (object instanceof THREE.Mesh) {
        const geometry = (object.geometry as THREE.BufferGeometry).clone();
        geometry.applyMatrix4(object.matrixWorld);
        geometries.push(geometry);
      }
    });
    if (geometries.length !== 1) {
      throw new Error(`Expected one mesh in ${url}, found ${geometries.length}`);
    }
    return { bytes: buffer.byteLength, geometry: geometries[0] };
  });
  cache.set(url, promise);
  return promise;
}
