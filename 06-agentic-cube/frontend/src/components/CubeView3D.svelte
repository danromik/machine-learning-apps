<!--
  A real 3D Rubik's Cube rendered with three.js.

  Builds the cube from individual cubie groups (a black plastic body + a colored
  sticker plane per outward face). `animateMove` physically rotates the affected
  layer 90° about its axis — grouping the layer's cubies under a temporary pivot,
  tweening the pivot, then baking the transforms back — so the user watches the
  cube actually turn, not snap between states. OrbitControls give free
  drag-to-rotate / scroll-to-zoom of the viewport.

  Parent contract (via bind:this):
    reset(frame)              rebuild the cube from a CubeFrame (scramble start)
    animateMove(move, ms)     returns a Promise that resolves when the turn ends
    isAnimating()
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import type { CubeFrame, MoveMeta } from '../api';
  import { cubeStyle } from '../state.svelte';
  import { cubeStyleById } from '../cubeStyles';

  let { size = 3 }: { size?: number } = $props();

  // Lattice coords step by 2 (e.g. a 3x3 uses -2,0,2), so a position scale of
  // 0.5 puts adjacent cubie centers 1.0 apart. With a cubie body just under 1.0
  // that leaves only a thin seam — a solid-looking cube, not spaced-out cuboids.
  const SPACING = 0.5;      // lattice-coord -> world scale (center spacing = 2*SPACING)
  const CUBIE = 0.97;       // plastic body size (slightly < center spacing → thin seam)
  const STICKER = 0.86;     // sticker plane size

  let container: HTMLDivElement;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls | null = null;
  let raf = 0;
  let resizeObs: ResizeObserver | null = null;

  // One THREE.Group per cubie; userData.coord holds its integer lattice coord.
  let cubies: THREE.Group[] = [];
  let pendingFrame: CubeFrame | null = null; // reset() called before scene ready
  let lastFrame: CubeFrame | null = null; // last frame built, for restyle rebuilds
  let bodyGeo: THREE.BoxGeometry | null = null;
  let stickerGeo: THREE.PlaneGeometry | null = null;
  let animating = $state(false);

  export function isAnimating() {
    return animating;
  }

  // ── Integer 90° rotation, matching the backend's right-hand-rule _rot ─────
  function rot(v: [number, number, number], axis: number, dir: number): [number, number, number] {
    const [x, y, z] = v;
    if (dir === 1) {
      if (axis === 0) return [x, -z, y];
      if (axis === 1) return [z, y, -x];
      return [-y, x, z];
    }
    if (axis === 0) return [x, z, -y];
    if (axis === 1) return [-z, y, x];
    return [y, -x, z];
  }

  function orientSticker(plane: THREE.Mesh, normal: number[]) {
    const [nx, ny, nz] = normal;
    if (nx === 1) plane.rotation.y = Math.PI / 2;
    else if (nx === -1) plane.rotation.y = -Math.PI / 2;
    else if (ny === 1) plane.rotation.x = -Math.PI / 2;
    else if (ny === -1) plane.rotation.x = Math.PI / 2;
    else if (nz === -1) plane.rotation.y = Math.PI;
    // nz === 1 → default orientation
    plane.position.set(nx * (CUBIE / 2 + 0.01), ny * (CUBIE / 2 + 0.01), nz * (CUBIE / 2 + 0.01));
  }

  function clearCubies() {
    // Dedupe materials before disposing (the body material is shared across
    // cubies within a build).
    const mats = new Set<THREE.Material>();
    for (const g of cubies) {
      scene.remove(g);
      g.traverse((o) => {
        const m = (o as THREE.Mesh).material;
        if (Array.isArray(m)) m.forEach((mm) => mats.add(mm));
        else if (m) mats.add(m as THREE.Material);
      });
    }
    mats.forEach((m) => m.dispose());
    cubies = [];
  }

  export function reset(frame: CubeFrame) {
    if (!scene) {
      pendingFrame = frame; // applied once the scene is ready (onMount)
      return;
    }
    lastFrame = frame;
    clearCubies();
    if (!bodyGeo) bodyGeo = new THREE.BoxGeometry(CUBIE, CUBIE, CUBIE);
    if (!stickerGeo) stickerGeo = new THREE.PlaneGeometry(STICKER, STICKER);

    const style = cubeStyleById(cubeStyle.id);
    const glass = style.transparent;
    // Solid cubes get an opaque black plastic body (one shared material). Glass
    // cubes drop the body entirely so you can see straight through to the far
    // stickers — cleaner than a tinted translucent box.
    const bodyMat = glass
      ? null
      : new THREE.MeshStandardMaterial({ color: 0x101012, roughness: 0.55 });

    for (const cubie of frame.cubies) {
      const g = new THREE.Group();
      if (bodyMat) g.add(new THREE.Mesh(bodyGeo, bodyMat));
      for (const st of cubie.stickers) {
        const mat = new THREE.MeshStandardMaterial({
          color: style.colors[st.color] ?? 0x808080,
          roughness: glass ? 0.2 : 0.4,
          transparent: glass,
          opacity: glass ? style.opacity : 1,
          depthWrite: !glass, // let translucent stickers blend with what's behind
          side: glass ? THREE.DoubleSide : THREE.FrontSide,
        });
        const plane = new THREE.Mesh(stickerGeo, mat);
        orientSticker(plane, st.normal);
        g.add(plane);
      }
      g.position.set(
        cubie.pos[0] * SPACING,
        cubie.pos[1] * SPACING,
        cubie.pos[2] * SPACING
      );
      g.userData.coord = [...cubie.pos];
      scene.add(g);
      cubies.push(g);
    }
  }

  export function animateMove(move: MoveMeta, durationMs = 320): Promise<void> {
    return new Promise((resolve) => {
      if (!scene || animating) {
        resolve();
        return;
      }
      const { axis, sign, dir } = move;
      const layerCoord = sign * (size - 1);
      const layer = cubies.filter((g) => Math.round(g.userData.coord[axis]) === layerCoord);
      if (layer.length === 0) {
        resolve();
        return;
      }
      animating = true;
      const pivot = new THREE.Group();
      scene.add(pivot);
      for (const g of layer) pivot.attach(g);

      const target = dir * (Math.PI / 2);
      const start = performance.now();
      const axisName = axis === 0 ? 'x' : axis === 1 ? 'y' : 'z';

      const tick = (now: number) => {
        const t = Math.min(1, (now - start) / durationMs);
        const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
        pivot.rotation[axisName] = target * eased;
        if (t < 1) {
          requestAnimationFrame(tick);
        } else {
          // Bake: reparent cubies back to the scene (keeps world transform) and
          // update their integer lattice coord for the next layer selection.
          for (const g of layer) {
            scene.attach(g);
            g.userData.coord = rot(g.userData.coord, axis, dir);
          }
          scene.remove(pivot);
          animating = false;
          resolve();
        }
      };
      requestAnimationFrame(tick);
    });
  }

  // ── three.js lifecycle ────────────────────────────────────────────────────
  onMount(() => {
    scene = new THREE.Scene();
    const w = container.clientWidth || 480;
    const h = container.clientHeight || 360;
    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
    const d = size === 2 ? 2.9 : 4.3;
    camera.position.set(d, d * 0.85, d);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const key = new THREE.DirectionalLight(0xffffff, 0.6);
    key.position.set(5, 8, 6);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.3);
    fill.position.set(-6, -3, -4);
    scene.add(fill);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.enablePan = false;
    controls.minDistance = 1.8;
    controls.maxDistance = 12;

    const loop = () => {
      raf = requestAnimationFrame(loop);
      controls?.update();
      if (renderer && scene && camera) renderer.render(scene, camera);
    };
    loop();

    resizeObs = new ResizeObserver(() => {
      if (!renderer || !camera || !container) return;
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      if (cw === 0 || ch === 0) return;
      camera.aspect = cw / ch;
      camera.updateProjectionMatrix();
      renderer.setSize(cw, ch);
    });
    resizeObs.observe(container);

    // Apply any frame requested before the scene was ready.
    if (pendingFrame) {
      reset(pendingFrame);
      pendingFrame = null;
    }
  });

  // Restyle the current cube live when the selected style changes (re-applies
  // materials by rebuilding from the last frame).
  $effect(() => {
    void cubeStyle.id;
    if (scene && lastFrame) reset(lastFrame);
  });

  onDestroy(() => {
    cancelAnimationFrame(raf);
    resizeObs?.disconnect();
    clearCubies();
    bodyGeo?.dispose();
    stickerGeo?.dispose();
    controls?.dispose();
    renderer?.dispose();
    if (renderer?.domElement && renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
    renderer = null;
  });
</script>

<div bind:this={container} class="w-full h-full min-h-0"></div>
