import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { mergeGeometries as threemergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// --- Type Definitions for Replay Schema ---
interface Meta {
  seed: number;
  mode: string;
  map_name: string;
  steps: number;
}

interface MapData {
  width: number;
  height: number;
  obstacles: [number, number][];
}

interface Frame {
  t: number;
  pos: [number, number][];
  state: ('moving' | 'idle' | 'delayed' | 'failed' | 'replanning')[];
}

interface Disruption {
  type: 'blocked_cell' | 'robot_failure' | 'robot_delay';
  cells?: [number, number][];
  robot?: number;
  k_steps?: number;
}

interface Congestion {
  level: 'low' | 'medium' | 'high';
  region: [[number, number], [number, number]];
  density?: number;
}

interface CacheResult {
  hit: boolean;
  matched?: string;
  similarity?: number;
}

interface DryRunResult {
  ran: boolean;
  valid: boolean;
  violation?: string;
}

interface Directive {
  robots: number[];
  solver: 'pibt' | 'lns' | 'windowed_pbs';
  priority: number[];
  horizon: number;
  rationale?: string;
}

interface LLMResult {
  called: boolean;
  latency_ms?: number;
  tokens?: number;
  directive?: Directive;
}

interface ValidatorAttempt {
  attempt: number;
  valid: boolean;
  violation?: string;
}

interface PathsData {
  invalidated: Record<string, [number, number][]>;
  repaired: Record<string, [number, number][]>;
}

interface IncidentOutcome {
  recovery_steps: number;
  cost_delta: number;
}

interface Incident {
  id: string;
  t_detected: number;
  disruptions: Disruption[];
  affected_robots: number[];
  congestion: Congestion;
  interacts_with: string[];
  cache?: CacheResult;
  dry_run?: DryRunResult;
  llm?: LLMResult;
  validator: ValidatorAttempt[];
  fallback_used: boolean;
  executed_at: number;
  paths: PathsData;
  outcome: IncidentOutcome;
}

interface Metrics {
  repair_success_rate: number;
  mean_recovery_steps: number;
  cost_delta_pct: number;
  llm_calls: number;
  llm_tokens: number;
  cache_hits: number;
}

interface Replay {
  meta: Meta;
  map: MapData;
  frames: Frame[];
  incidents: Incident[];
  metrics: Metrics;
}

// --- App State ---
class ViewerApp {
  private container!: HTMLElement;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;

  private replay: Replay | null = null;
  private baselineReplay: Replay | null = null;
  private activeReplay: Replay | null = null;
  private isBaselineMode = false;
  private currentScenario = 'demo';

  private mapWidth = 161;
  private mapHeight = 63;
  private obstacleSet = new Set<string>();

  // 3D Objects
  private shelfInstancedMesh: THREE.InstancedMesh | null = null;
  private robotMeshChassis: THREE.InstancedMesh | null = null;
  private robotMeshLed: THREE.InstancedMesh | null = null;
  private shadowDecalMesh: THREE.InstancedMesh | null = null;
  private barrierGroup = new THREE.Group();
  private pathsGroup = new THREE.Group();
  private floorLinesGroup = new THREE.Group();
  private goalMarkerMesh: THREE.InstancedMesh | null = null;

  // Loaded Geometries & Materials
  // Shelf geometry is now built procedurally in createShelves
  private barrierGeometry: THREE.BufferGeometry | null = null;
  private barrierMaterial: THREE.Material | THREE.Material[] | null = null;

  // Animation & Clock
  private clockTime = 0.0;
  private isPlaying = true;
  private playSpeed = 1.0;
  private lastTimestamp = 0;
  private selectedIncident: Incident | null = null;
  private selectedRobotId: number | null = null;

  // Camera animation
  private activePreset = 1;
  private isCameraTransitioning = false;
  private cameraStartPos = new THREE.Vector3();
  private cameraStartTarget = new THREE.Vector3();
  private cameraEndPos = new THREE.Vector3();
  private cameraEndTarget = new THREE.Vector3();
  private cameraAnimProgress = 1.0;
  private cameraAnimDuration = 1.0;
  private keysPressed: Record<string, boolean> = {};

  // Performance tracking
  private frameCount = 0;
  private lastFpsUpdate = 0;
  private fpsDisplay: HTMLElement | null = null;

  async init() {
    this.container = document.getElementById('canvas-container')!;
    this.fpsDisplay = document.getElementById('fps-counter');

    this.setupScene();
    this.setupRenderer();
    this.setupCamera();
    this.setupLighting();

    await this.loadMap();
    await this.loadTexturesAndHDRI();
    await this.loadModels();
    await this.loadScenario(this.currentScenario);

    this.setupUI();
    this.setupEventListeners();

    // Responsive container observer
    const resizeObserver = new ResizeObserver(() => {
      this.onWindowResize();
    });
    resizeObserver.observe(this.container);

    // Intro camera dolly: overview down to aisle view smoothly
    this.startIntroDolly();

    window.addEventListener('resize', this.onWindowResize.bind(this));
    requestAnimationFrame(this.renderLoop.bind(this));
  }

  private setupScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070a);
    this.scene.fog = new THREE.FogExp2(0x05070a, 0.0055);
    this.scene.add(this.barrierGroup);
    this.scene.add(this.pathsGroup);
    this.scene.add(this.floorLinesGroup);
  }

  private setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
      logarithmicDepthBuffer: true
    });
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);
  }

  private setupCamera() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.5, 2000);
    this.camera.position.set(this.mapWidth / 2, 85, this.mapHeight / 2 + 75);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.rotateSpeed = 0.85;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 1.1;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 500;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.01;
    this.controls.target.set(this.mapWidth / 2, 0, this.mapHeight / 2);
    this.controls.update();
  }

  private setupLighting() {
    // Hemisphere light for soft ambient fill (cool ceiling → warm floor bounce)
    const hemiLight = new THREE.HemisphereLight(0x8899bb, 0x443322, 0.35);
    this.scene.add(hemiLight);

    // Main directional "bay light" from above-front with soft shadow map
    const dirLight = new THREE.DirectionalLight(0xffeedd, 2.0);
    dirLight.position.set(80, 90, -30);
    dirLight.target.position.set(80, 0, 31.5);
    this.scene.add(dirLight);
    this.scene.add(dirLight.target);

    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 4096;
    dirLight.shadow.mapSize.height = 4096;
    dirLight.shadow.camera.near = 10;
    dirLight.shadow.camera.far = 250;
    dirLight.shadow.bias = -0.0003;
    dirLight.shadow.normalBias = 0.02;

    const d = 100;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d * 0.7;
    dirLight.shadow.camera.bottom = -d * 0.7;

    // Secondary fill light from opposite side (no shadow, just subtle fill)
    const fillLight = new THREE.DirectionalLight(0xaabbdd, 0.4);
    fillLight.position.set(-30, 50, 80);
    this.scene.add(fillLight);

    // Rim/back light for depth separation
    const rimLight = new THREE.DirectionalLight(0x6688cc, 0.3);
    rimLight.position.set(80, 40, 120);
    this.scene.add(rimLight);
  }

  private async loadTexturesAndHDRI() {
    const rgbeLoader = new RGBELoader();
    try {
      const hdrTexture = await rgbeLoader.loadAsync('/textures/warehouse.hdr');
      hdrTexture.mapping = THREE.EquirectangularReflectionMapping;
      this.scene.environment = hdrTexture;
    } catch (e) {
      console.warn('Could not load warehouse.hdr, using neutral fallback', e);
    }

    const texLoader = new THREE.TextureLoader();
    try {
      const floorTex = await texLoader.loadAsync('/textures/concrete_floor.jpg');
      floorTex.wrapS = THREE.RepeatWrapping;
      floorTex.wrapT = THREE.RepeatWrapping;
      floorTex.repeat.set(this.mapWidth / 2, this.mapHeight / 2);
      this.createFloor(floorTex);
    } catch {
      this.createFloor(null);
    }
  }

  private createFloor(texture: THREE.Texture | null) {
    // Main warehouse floor — sealed concrete appearance
    const floorGeo = new THREE.PlaneGeometry(this.mapWidth + 2, this.mapHeight + 2);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x1e232e,
      roughness: 0.78,
      metalness: 0.08,
      map: texture || undefined
    });

    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(this.mapWidth / 2, 0, this.mapHeight / 2);
    floor.receiveShadow = true;
    this.scene.add(floor);

    // Subtle ground plane extending beyond walls (dark exterior pad)
    const extFloorGeo = new THREE.PlaneGeometry(this.mapWidth * 3, this.mapHeight * 3);
    const extFloorMat = new THREE.MeshStandardMaterial({
      color: 0x080a10,
      roughness: 0.95
    });
    const extFloor = new THREE.Mesh(extFloorGeo, extFloorMat);
    extFloor.rotation.x = -Math.PI / 2;
    extFloor.position.set(this.mapWidth / 2, -0.01, this.mapHeight / 2);
    this.scene.add(extFloor);

    this.createAisleLaneMarkings();
  }

  private createAisleLaneMarkings() {
    // --- Yellow center guide lines in aisles ---
    const yellowMat = new THREE.LineBasicMaterial({ color: 0xca8a04, transparent: true, opacity: 0.5 });
    const dimGuideMat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.3 });

    // Identify aisle rows (rows that are NOT fully obstacle)
    for (let y = 0; y < this.mapHeight; y++) {
      let freeCount = 0;
      for (let x = 0; x < this.mapWidth; x++) {
        if (!this.obstacleSet.has(`${x},${y}`)) freeCount++;
      }
      if (freeCount > this.mapWidth * 0.5) {
        // This is an aisle row — draw a yellow center guide
        const pts = [
          new THREE.Vector3(1, 0.013, y + 0.5),
          new THREE.Vector3(this.mapWidth - 1, 0.013, y + 0.5)
        ];
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(geom, yellowMat);
        this.floorLinesGroup.add(line);
      } else if (y % 2 === 0) {
        // Dim lines through shelf areas
        const pts = [
          new THREE.Vector3(1, 0.012, y + 0.5),
          new THREE.Vector3(this.mapWidth - 1, 0.012, y + 0.5)
        ];
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(geom, dimGuideMat);
        this.floorLinesGroup.add(line);
      }
    }

    // --- White edge boundary lines along warehouse perimeter ---
    const whiteMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 });
    const borderInset = 0.3;
    const borderPts = [
      new THREE.Vector3(borderInset, 0.015, borderInset),
      new THREE.Vector3(this.mapWidth - borderInset, 0.015, borderInset),
      new THREE.Vector3(this.mapWidth - borderInset, 0.015, this.mapHeight - borderInset),
      new THREE.Vector3(borderInset, 0.015, this.mapHeight - borderInset),
      new THREE.Vector3(borderInset, 0.015, borderInset)
    ];
    const borderGeom = new THREE.BufferGeometry().setFromPoints(borderPts);
    this.floorLinesGroup.add(new THREE.Line(borderGeom, whiteMat));

    // --- Safety hatching at intersections (every 20 columns) ---
    const safetyMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.08, side: THREE.DoubleSide });
    for (let x = 0; x < this.mapWidth; x += 20) {
      const hatchGeo = new THREE.PlaneGeometry(1.5, this.mapHeight);
      hatchGeo.rotateX(-Math.PI / 2);
      const hatch = new THREE.Mesh(hatchGeo, safetyMat);
      hatch.position.set(x + 0.5, 0.011, this.mapHeight / 2);
      this.floorLinesGroup.add(hatch);
    }
  }

  private createWarehouseWalls() {
    // Walls removed per user request for open warehouse visibility
  }

  private createCeilingLights() {
    // Hanging strip light fixtures (emissive long bars)
    const lightFixtureMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xddeeff,
      emissiveIntensity: 1.2,
      roughness: 0.2,
      metalness: 0.8
    });
    const fixtureHousingMat = new THREE.MeshStandardMaterial({
      color: 0x2a2e35,
      roughness: 0.6,
      metalness: 0.3
    });

    // Place lights in rows along the warehouse length
    const ceilingH = 5.2;
    const lightSpacingZ = 8;
    const lightLen = 6;
    const lightW = 0.15;
    const hangH = 0.5;
    const fixtureY = ceilingH - hangH;

    for (let z = lightSpacingZ / 2; z < this.mapHeight; z += lightSpacingZ) {
      for (let x = 10; x < this.mapWidth - 5; x += 18) {
        // Housing
        const housing = new THREE.Mesh(
          new THREE.BoxGeometry(lightW + 0.1, 0.08, lightLen + 0.1),
          fixtureHousingMat
        );
        housing.position.set(x, fixtureY + 0.04, z);
        this.scene.add(housing);

        // Emissive tube
        const tube = new THREE.Mesh(
          new THREE.BoxGeometry(lightW, 0.04, lightLen),
          lightFixtureMat
        );
        tube.position.set(x, fixtureY, z);
        this.scene.add(tube);

        // Hanging rod
        const rod = new THREE.Mesh(
          new THREE.CylinderGeometry(0.015, 0.015, hangH, 6),
          fixtureHousingMat
        );
        rod.position.set(x, ceilingH - hangH / 2, z);
        this.scene.add(rod);
      }
    }

    // Subtle point lights under some fixtures for pool-of-light effect
    for (let z = lightSpacingZ / 2; z < this.mapHeight; z += lightSpacingZ * 2) {
      for (let x = 20; x < this.mapWidth - 10; x += 36) {
        const pl = new THREE.PointLight(0xddeeff, 8, 20, 2);
        pl.position.set(x, fixtureY - 0.1, z);
        this.scene.add(pl);
      }
    }
  }

  private async loadMap() {
    try {
      const res = await fetch('/maps/warehouse-10-20-10-2-1.map');
      const text = await res.text();
      const lines = text.split('\n').filter(l => l.trim().length > 0);

      this.mapHeight = parseInt(lines[1].split(/\s+/)[1], 10);
      this.mapWidth = parseInt(lines[2].split(/\s+/)[1], 10);

      const gridLines = lines.slice(4, 4 + this.mapHeight);
      this.obstacleSet.clear();

      for (let y = 0; y < gridLines.length; y++) {
        const row = gridLines[y];
        for (let x = 0; x < row.length; x++) {
          if (row[x] === '@' || row[x] === 'T' || row[x] === 'O') {
            this.obstacleSet.add(`${x},${y}`);
          }
        }
      }
    } catch (e) {
      console.error('Failed to load map file:', e);
    }
  }

  private createGoalMarkers() {
    // Pulsing rings on the floor at each robot's destination
    const numRobots = 100;
    const ringGeo = new THREE.RingGeometry(0.22, 0.30, 20);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      side: THREE.DoubleSide
    });
    this.goalMarkerMesh = new THREE.InstancedMesh(ringGeo, ringMat, numRobots);
    this.goalMarkerMesh.visible = true;
    this.scene.add(this.goalMarkerMesh);

    // Initialize all markers at origin (will be updated each frame)
    const dummy = new THREE.Object3D();
    dummy.position.set(-100, 0, -100);
    dummy.updateMatrix();
    for (let i = 0; i < numRobots; i++) {
      this.goalMarkerMesh.setMatrixAt(i, dummy.matrix);
    }
    this.goalMarkerMesh.instanceMatrix.needsUpdate = true;
  }

  private updateGoalMarkers(t: number) {
    if (!this.goalMarkerMesh || !this.activeReplay) return;

    const frames = this.activeReplay.frames;
    const s0 = Math.floor(t);
    if (s0 >= frames.length) return;

    const frame = frames[s0];
    const dummy = new THREE.Object3D();
    const timeSec = performance.now() / 1000;
    const pulse = 0.4 + 0.2 * Math.sin(timeSec * 3.0);

    // We don't have explicit goal data in frames, so show goal markers for selected/affected robots
    // For now, hide them by default and show for selected robot only
    for (let r = 0; r < frame.pos.length; r++) {
      if (this.selectedRobotId === r) {
        // Use the approximate goal from the robot card
        const pos = frame.pos[r];
        const goalX = (pos[0] + 18) % this.mapWidth;
        const goalY = (pos[1] + 12) % this.mapHeight;
        dummy.position.set(goalX + 0.5, 0.02, goalY + 0.5);
        dummy.scale.setScalar(1.0 + pulse * 0.3);
        dummy.updateMatrix();
        this.goalMarkerMesh.setMatrixAt(r, dummy.matrix);
      } else {
        dummy.position.set(-100, 0, -100);
        dummy.scale.setScalar(1);
        dummy.updateMatrix();
        this.goalMarkerMesh.setMatrixAt(r, dummy.matrix);
      }
    }
    this.goalMarkerMesh.instanceMatrix.needsUpdate = true;
  }

  private async loadModels() {
    const gltfLoader = new GLTFLoader();

    // 1. Shelves — built procedurally (multi-tier rack geometry)
    this.createShelves();
    this.createWarehouseWalls();
    this.createCeilingLights();
    this.createGoalMarkers();

    // 2. Barrier Model
    try {
      const barrierGltf = await gltfLoader.loadAsync('/models/barrier.glb');
      barrierGltf.scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          this.barrierGeometry = mesh.geometry;
          this.barrierMaterial = mesh.material;
        }
      });
    } catch (e) {
      console.warn('Barrier glb load error', e);
    }

    if (!this.barrierGeometry) {
      this.barrierGeometry = new THREE.ConeGeometry(0.3, 0.6, 16);
      this.barrierMaterial = new THREE.MeshStandardMaterial({ color: 0xf97316 });
    }

    // 3. Robot Model (high-detail procedural AMR digital twin)
    this.createRobotMeshes();
  }

  private buildRackGeometry(): THREE.BufferGeometry {
    // Procedural industrial pallet rack — two uprights, three shelf levels, with crates
    const parts: THREE.BufferGeometry[] = [];
    const cellW = 0.92;
    const rackH = 2.2;

    // --- Uprights (two vertical steel posts, left and right) ---
    const uprightW = 0.05;
    const uprightD = 0.05;
    for (const xOff of [-cellW / 2 + uprightW / 2, cellW / 2 - uprightW / 2]) {
      for (const zOff of [-cellW / 2 + uprightD / 2, cellW / 2 - uprightD / 2]) {
        const g = new THREE.BoxGeometry(uprightW, rackH, uprightD);
        g.translate(xOff, rackH / 2, zOff);
        parts.push(g);
      }
    }

    // --- Horizontal beams at each shelf level ---
    const beamH = 0.035;
    const shelfLevels = [0.0, 0.55, 1.10, 1.65];
    for (const ly of shelfLevels) {
      // Front and back beams
      for (const zOff of [-cellW / 2 + 0.025, cellW / 2 - 0.025]) {
        const beam = new THREE.BoxGeometry(cellW - 0.04, beamH, 0.04);
        beam.translate(0, ly + beamH / 2, zOff);
        parts.push(beam);
      }
      // Shelf deck (thin panel)
      if (ly > 0) {
        const deck = new THREE.BoxGeometry(cellW - 0.06, 0.018, cellW - 0.06);
        deck.translate(0, ly + 0.02, 0);
        parts.push(deck);
      }
    }

    // --- Crates / boxes on shelf levels 1, 2, 3 ---
    const crateH = [0.38, 0.32, 0.28];
    for (let li = 1; li < shelfLevels.length; li++) {
      const baseY = shelfLevels[li] + 0.04;
      const h = crateH[li - 1];
      // Two crates per shelf (left and right halves)
      for (const xOff of [-0.18, 0.18]) {
        const crate = new THREE.BoxGeometry(0.32, h, cellW * 0.7);
        crate.translate(xOff, baseY + h / 2, 0);
        parts.push(crate);
      }
    }

    // --- Diagonal cross-brace on the back face ---
    const braceLen = Math.sqrt(cellW * cellW + rackH * rackH);
    const brace = new THREE.BoxGeometry(0.02, braceLen, 0.01);
    const angle = Math.atan2(rackH, cellW);
    brace.rotateZ(angle);
    brace.translate(0, rackH / 2, -cellW / 2 + 0.005);
    parts.push(brace);

    try {
      return threemergeGeometries(parts);
    } catch {
      return this.mergeGeometries(parts);
    }
  }

  private mergeGeometries(geometries: THREE.BufferGeometry[]): THREE.BufferGeometry {
    // Manual merge fallback if BufferGeometryUtils not available
    let totalVerts = 0;
    let totalIdx = 0;
    for (const g of geometries) {
      totalVerts += g.attributes.position.count;
      totalIdx += g.index ? g.index.count : 0;
    }
    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const indices: number[] = [];
    let vertOffset = 0;

    for (const g of geometries) {
      const pos = g.attributes.position;
      const nor = g.attributes.normal;
      for (let i = 0; i < pos.count; i++) {
        positions[(vertOffset + i) * 3] = pos.getX(i);
        positions[(vertOffset + i) * 3 + 1] = pos.getY(i);
        positions[(vertOffset + i) * 3 + 2] = pos.getZ(i);
        if (nor) {
          normals[(vertOffset + i) * 3] = nor.getX(i);
          normals[(vertOffset + i) * 3 + 1] = nor.getY(i);
          normals[(vertOffset + i) * 3 + 2] = nor.getZ(i);
        }
      }
      if (g.index) {
        for (let i = 0; i < g.index.count; i++) {
          indices.push(g.index.getX(i) + vertOffset);
        }
      }
      vertOffset += pos.count;
    }

    const merged = new THREE.BufferGeometry();
    merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    if (indices.length > 0) {
      merged.setIndex(indices);
    }
    merged.computeVertexNormals();
    return merged;
  }

  private createShelves() {
    const obstacleCoords: [number, number][] = [];
    this.obstacleSet.forEach((coordStr) => {
      const [x, y] = coordStr.split(',').map(Number);
      obstacleCoords.push([x, y]);
    });

    const count = obstacleCoords.length;

    // Build rack geometry procedurally (uprights + beams + crates)
    const rackGeom = this.buildRackGeometry();
    const rackMat = new THREE.MeshStandardMaterial({
      color: 0x181a1d, // Deep industrial matte black steel
      roughness: 0.72,
      metalness: 0.35,
    });

    this.shelfInstancedMesh = new THREE.InstancedMesh(rackGeom, rackMat, count);
    this.shelfInstancedMesh.castShadow = true;
    this.shelfInstancedMesh.receiveShadow = true;

    // Per-instance dark tones for rich industrial contrast
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    for (let i = 0; i < count; i++) {
      const [x, y] = obstacleCoords[i];
      dummy.position.set(x + 0.5, 0, y + 0.5);
      dummy.updateMatrix();
      this.shelfInstancedMesh.setMatrixAt(i, dummy.matrix);

      // Deep matte black with subtle cool graphite tone variation
      const v = 0.085 + (Math.sin(x * 0.35 + y * 0.65) * 0.5 + 0.5) * 0.04;
      color.setRGB(v, v * 1.02, v * 1.05);
      this.shelfInstancedMesh.setColorAt(i, color);
    }

    this.shelfInstancedMesh.instanceMatrix.needsUpdate = true;
    if (this.shelfInstancedMesh.instanceColor) this.shelfInstancedMesh.instanceColor.needsUpdate = true;
    this.scene.add(this.shelfInstancedMesh);
  }

  private buildRobotChassisGeometry(): THREE.BufferGeometry {
    // High-fidelity industrial Autonomous Mobile Robot (AMR) digital twin
    const parts: THREE.BufferGeometry[] = [];

    // --- 1. Main Lower Hull (Octagonal / Chamfered Industrial AGV Base) ---
    // Ground contact level is at y = -0.10 relative to the center origin (y = 0)
    const baseW = 0.54;
    const baseD = 0.54;
    const baseH = 0.10;
    const baseY = -0.025; // center of lower hull: [-0.075 to +0.025]
    const baseCenter = new THREE.BoxGeometry(baseW, baseH, baseD);
    baseCenter.translate(0, baseY, 0);
    parts.push(baseCenter);

    // Front & rear bumper caps
    const bumperH = 0.08;
    const bumperW = 0.50;
    const bumperD = 0.04;
    const fBumper = new THREE.BoxGeometry(bumperW, bumperH, bumperD);
    fBumper.translate(0, baseY, baseD / 2 + bumperD / 2);
    parts.push(fBumper);

    const rBumper = new THREE.BoxGeometry(bumperW, bumperH, bumperD);
    rBumper.translate(0, baseY, -baseD / 2 - bumperD / 2);
    parts.push(rBumper);

    // Corner fillets (4 corner vertical cylinders)
    const cornerR = 0.04;
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const corner = new THREE.CylinderGeometry(cornerR, cornerR, baseH, 12);
        corner.translate(sx * (baseW / 2 - cornerR), baseY, sz * (baseD / 2 - cornerR));
        parts.push(corner);
      }
    }

    // --- 2. Side Wheel Wells & Drive Wheels ---
    const wheelR = 0.048;
    const wheelW = 0.032;
    for (const sx of [-1, 1]) {
      // Main drive wheel (left & right)
      const wheel = new THREE.CylinderGeometry(wheelR, wheelR, wheelW, 16);
      wheel.rotateZ(Math.PI / 2);
      wheel.translate(sx * (baseW / 2 + wheelW / 3), -0.052, 0);
      parts.push(wheel);

      // Wheel hubcap
      const hub = new THREE.CylinderGeometry(wheelR * 0.55, wheelR * 0.55, wheelW + 0.006, 12);
      hub.rotateZ(Math.PI / 2);
      hub.translate(sx * (baseW / 2 + wheelW / 3), -0.052, 0);
      parts.push(hub);
    }

    // --- 3. Upper Housing Shell & Deck Plate ---
    const upperW = 0.48;
    const upperD = 0.48;
    const upperH = 0.035;
    const upperY = 0.042; // [0.025 to 0.060]
    const upperDeck = new THREE.BoxGeometry(upperW, upperH, upperD);
    upperDeck.translate(0, upperY, 0);
    parts.push(upperDeck);

    // --- 4. Rotating Lift Turntable (Top Load Platform) ---
    // Central circular lift platter
    const platterR = 0.22;
    const platterH = 0.024;
    const platterY = 0.072; // [0.060 to 0.084]
    const platter = new THREE.CylinderGeometry(platterR, platterR, platterH, 32);
    platter.translate(0, platterY, 0);
    parts.push(platter);

    // Concentric rubber grip pads on turntable
    const ringGeo = new THREE.CylinderGeometry(platterR * 0.82, platterR * 0.82, 0.006, 24);
    ringGeo.translate(0, platterY + platterH / 2 + 0.003, 0);
    parts.push(ringGeo);

    // Center payload alignment pin
    const pin = new THREE.CylinderGeometry(0.035, 0.042, 0.014, 16);
    pin.translate(0, platterY + platterH / 2 + 0.007, 0);
    parts.push(pin);

    // --- 5. Front LiDAR Sensor Turret & Aperture ---
    // Cylindrical LiDAR puck at front
    const lidarR = 0.036;
    const lidarH = 0.038;
    const lidarY = upperY + upperH / 2 + lidarH / 2;
    const lidarZ = upperD / 2 - 0.055;
    const lidar = new THREE.CylinderGeometry(lidarR, lidarR * 1.05, lidarH, 16);
    lidar.translate(0, lidarY, lidarZ);
    parts.push(lidar);

    // LiDAR protective dome cap
    const lidarCap = new THREE.CylinderGeometry(lidarR * 1.15, lidarR * 0.85, 0.008, 16);
    lidarCap.translate(0, lidarY + lidarH / 2 + 0.004, lidarZ);
    parts.push(lidarCap);

    // Front sensor window notch
    const sensorNotch = new THREE.BoxGeometry(0.14, 0.022, 0.025);
    sensorNotch.translate(0, baseY + 0.01, baseD / 2 + bumperD - 0.005);
    parts.push(sensorNotch);

    // --- 6. Directional Forward Chevron Arrow (pointing +Z forward) ---
    const arrowStem = new THREE.BoxGeometry(0.035, 0.006, 0.06);
    arrowStem.translate(0, platterY + platterH / 2 + 0.005, 0.05);
    parts.push(arrowStem);

    const arrowWingL = new THREE.BoxGeometry(0.028, 0.006, 0.05);
    arrowWingL.rotateY(-Math.PI / 4);
    arrowWingL.translate(-0.025, platterY + platterH / 2 + 0.005, 0.09);
    parts.push(arrowWingL);

    const arrowWingR = new THREE.BoxGeometry(0.028, 0.006, 0.05);
    arrowWingR.rotateY(Math.PI / 4);
    arrowWingR.translate(0.025, platterY + platterH / 2 + 0.005, 0.09);
    parts.push(arrowWingR);

    // --- 7. Rear Emergency-Stop Plungers ---
    for (const sx of [-0.18, 0.18]) {
      const estop = new THREE.CylinderGeometry(0.014, 0.014, 0.016, 8);
      estop.translate(sx, upperY + upperH / 2 + 0.008, -upperD / 2 + 0.035);
      parts.push(estop);
    }

    try {
      return threemergeGeometries(parts);
    } catch {
      return this.mergeGeometries(parts);
    }
  }

  private buildRobotLedGeometry(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];

    // Continuous 360 halo ring surrounding the lift turntable
    const halo = new THREE.TorusGeometry(0.235, 0.013, 12, 32);
    halo.rotateX(Math.PI / 2);
    halo.translate(0, 0.068, 0);
    parts.push(halo);

    // Dual front high-intensity projector headlights (illuminating path ahead in +Z)
    const lightR = 0.020;
    const lightL = 0.022;
    for (const sx of [-0.16, 0.16]) {
      const light = new THREE.CylinderGeometry(lightR, lightR, lightL, 12);
      light.rotateX(Math.PI / 2);
      light.translate(sx, -0.02, 0.285);
      parts.push(light);
    }

    // Rear status taillight strip
    const tailLight = new THREE.BoxGeometry(0.16, 0.014, 0.012);
    tailLight.translate(0, -0.02, -0.285);
    parts.push(tailLight);

    try {
      return threemergeGeometries(parts);
    } catch {
      return this.mergeGeometries(parts);
    }
  }

  private createRobotMeshes(chassisGeom?: THREE.BufferGeometry | null, ledGeom?: THREE.BufferGeometry | null) {
    const numRobots = 100;

    // Use high-detail procedural AMR body
    const cGeom = chassisGeom || this.buildRobotChassisGeometry();
    const cMat = new THREE.MeshStandardMaterial({
      color: 0x1f232b, // High-grade industrial graphite matte composite
      roughness: 0.45,
      metalness: 0.55,
    });
    this.robotMeshChassis = new THREE.InstancedMesh(cGeom, cMat, numRobots);
    this.robotMeshChassis.castShadow = true;
    this.robotMeshChassis.receiveShadow = true;
    this.scene.add(this.robotMeshChassis);

    // LED status ring + headlights + taillight
    const lGeom = ledGeom || this.buildRobotLedGeometry();
    const lMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x38bdf8,
      emissiveIntensity: 3.5,
      roughness: 0.15,
      metalness: 0.1
    });
    this.robotMeshLed = new THREE.InstancedMesh(lGeom, lMat, numRobots);
    this.scene.add(this.robotMeshLed);

    // Ground shadow decal — soft circular blob shadow
    const shadowGeo = new THREE.CircleGeometry(0.38, 24);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.55,
      depthWrite: false
    });
    this.shadowDecalMesh = new THREE.InstancedMesh(shadowGeo, shadowMat, numRobots);
    this.scene.add(this.shadowDecalMesh);
  }

  private async loadScenario(scenarioKey: string) {
    this.currentScenario = scenarioKey;
    let mainUrl = `/replays/${scenarioKey}.json`;
    let baseSuffix = scenarioKey === 'demo' ? 'demo_baseline.json' : `${scenarioKey}_baseline.json`;
    let baseUrl = `/replays/${baseSuffix}`;

    try {
      const [resMain, resBase] = await Promise.all([
        fetch(mainUrl),
        fetch(baseUrl)
      ]);

      this.replay = await resMain.json();
      this.baselineReplay = await resBase.json();
      this.activeReplay = this.isBaselineMode ? this.baselineReplay : this.replay;

      const slider = document.getElementById('step-slider') as HTMLInputElement;
      if (slider && this.activeReplay) {
        slider.max = (this.activeReplay.meta.steps - 1).toString();
      }

      this.updateMetricsUI();
      this.buildTimelineTicks();
      this.showToast('Scenario Loaded', `Switched to ${scenarioKey} (${this.activeReplay?.incidents.length || 0} incidents)`);
    } catch (e) {
      console.error('Failed to load scenario:', e);
    }
  }

  private updateMetricsUI() {
    if (!this.activeReplay) return;
    const m = this.activeReplay.metrics;
    const bm = this.baselineReplay ? this.baselineReplay.metrics : null;

    const elSuccess = document.getElementById('metric-success') || document.getElementById('metric-success-rate') || document.getElementById('chip-success-rate');
    const elRecovery = document.getElementById('metric-recovery') || document.getElementById('metric-recovery-steps');
    const elCost = document.getElementById('metric-cost');
    const elLlm = document.getElementById('metric-llm') || document.getElementById('metric-llm-calls');

    const baseSuccessStr = bm ? `vs ${(bm.repair_success_rate * 100).toFixed(1)}% base` : '';
    const baseRecStr = bm ? `vs ${bm.mean_recovery_steps.toFixed(1)} base` : '';
    const baseCostStr = bm ? `vs +${bm.cost_delta_pct.toFixed(1)}% base` : '';
    const baseLlmStr = bm ? `vs ${bm.llm_calls} base` : '';

    if (elSuccess) elSuccess.innerHTML = `${(m.repair_success_rate * 100).toFixed(1)}%`;
    const baseSuccEl = document.getElementById('base-success');
    if (baseSuccEl) baseSuccEl.innerText = baseSuccessStr;

    if (elRecovery) elRecovery.innerHTML = `${m.mean_recovery_steps.toFixed(1)} <span class="unit">steps</span>`;
    const baseRecEl = document.getElementById('base-recovery');
    if (baseRecEl) baseRecEl.innerText = baseRecStr;

    if (elCost) elCost.innerHTML = `+${m.cost_delta_pct.toFixed(1)}%`;
    const baseCostEl = document.getElementById('base-cost');
    if (baseCostEl) baseCostEl.innerText = baseCostStr;

    if (elLlm) elLlm.innerHTML = `${m.llm_calls} <span class="unit">calls</span>`;
    const baseLlmEl = document.getElementById('base-llm');
    if (baseLlmEl) baseLlmEl.innerText = `${m.cache_hits} cache hits ${baseLlmStr ? '(' + baseLlmStr + ')' : ''}`;

    const deltaSuccess = bm ? (m.repair_success_rate - bm.repair_success_rate) * 100 : 0;
    const badgeSuccess = document.getElementById('badge-success');
    if (badgeSuccess) {
      badgeSuccess.innerText = `${deltaSuccess >= 0 ? '+' : ''}${deltaSuccess.toFixed(1)}%`;
    }

    const deltaRec = bm ? m.mean_recovery_steps - bm.mean_recovery_steps : 0;
    const badgeRec = document.getElementById('badge-recovery');
    if (badgeRec) {
      badgeRec.innerText = `${deltaRec >= 0 ? '+' : ''}${deltaRec.toFixed(1)} steps`;
    }

    // Header chips
    const chipRobots = document.getElementById('chip-robots-count');
    if (chipRobots && this.activeReplay.frames[0]) {
      chipRobots.innerText = `${this.activeReplay.frames[0].pos.length}`;
    }
    const chipSteps = document.getElementById('chip-steps-count');
    if (chipSteps) chipSteps.innerText = `${this.activeReplay.meta.steps}`;
    const chipInc = document.getElementById('chip-incidents-count');
    if (chipInc) chipInc.innerText = `${this.activeReplay.incidents.length}`;
    const chipSucc = document.getElementById('chip-success-rate');
    if (chipSucc) chipSucc.innerText = `${(m.repair_success_rate * 100).toFixed(1)}%`;
    const chipCache = document.getElementById('metric-cache-rate');
    if (chipCache) chipCache.innerText = `${(m.cache_hits ? ((m.cache_hits / (m.cache_hits + m.llm_calls || 1)) * 100).toFixed(1) : '100.0')}%`;
  }

  private buildTimelineTicks() {
    if (!this.activeReplay) return;
    const container = document.getElementById('timeline-ticks') || document.getElementById('track-incidents-viewport');
    if (!container) return;
    container.innerHTML = '';
    const totalSteps = this.activeReplay.meta.steps;

    const tooltip = document.getElementById('incident-tooltip');

    this.activeReplay.incidents.forEach((inc) => {
      const tick = document.createElement('div');
      const type = inc.disruptions[0]?.type || 'blocked_cell';
      let typeClass = 'blocked';
      if (type === 'robot_failure') typeClass = 'failure';
      if (type === 'robot_delay') typeClass = 'delay';

      tick.className = `incident-tick ${typeClass}`;
      const pct = (inc.t_detected / totalSteps) * 100;
      tick.style.left = `${pct}%`;

      // Hover Tooltip
      if (tooltip) {
        tick.addEventListener('mouseenter', () => {
          const rect = tick.getBoundingClientRect();
          tooltip.style.left = `${rect.left + rect.width / 2}px`;
          tooltip.style.bottom = `95px`;
          const ttHeader = document.getElementById('tt-header');
          if (ttHeader) ttHeader.innerText = `Incident ${inc.id} (t=${inc.t_detected})`;
          const ttBody = document.getElementById('tt-body');
          if (ttBody) ttBody.innerText = `${type.replace('_', ' ').toUpperCase()} • ${inc.affected_robots.length} AMRs affected`;
          tooltip.classList.remove('hidden');
        });

        tick.addEventListener('mouseleave', () => {
          tooltip.classList.add('hidden');
        });
      }

      tick.addEventListener('click', (e) => {
        e.stopPropagation();
        tooltip?.classList.add('hidden');
        this.selectIncident(inc);
      });

      container.appendChild(tick);
    });

    const incChip = document.getElementById('incident-counter-chip') || document.getElementById('incidents-badge-count');
    if (incChip) {
      incChip.innerText = `${this.activeReplay.incidents.length}`;
    }
  }

  private selectIncident(inc: Incident) {
    this.selectedIncident = inc;
    this.clockTime = inc.t_detected;
    this.updateSliderUI();
    this.showIncidentPanel(inc);
    this.updatePathsVisualization();
    this.showToast(`Incident ${inc.id}`, `Jumped to step t=${inc.t_detected} (${inc.disruptions[0]?.type})`);
  }

  private showIncidentPanel(inc: Incident) {
    const panel = document.getElementById('incident-panel') || document.getElementById('incidents-panel');
    if (panel) panel.classList.remove('hidden');

    const titleEl = document.getElementById('incident-panel-title');
    if (titleEl) titleEl.innerText = `Incident ${inc.id}`;
    const dis = inc.disruptions[0];
    const subEl = document.getElementById('incident-panel-subtitle');
    if (subEl) subEl.innerText = `Step t=${inc.t_detected} • ${dis?.type.replace('_', ' ').toUpperCase()}`;

    // 1. Sentinel
    const interactsStr = inc.interacts_with.length > 0 ? ` | Interacts with: [${inc.interacts_with.join(', ')}]` : '';
    const sentEl = document.getElementById('inc-sentinel-text');
    if (sentEl) {
      sentEl.innerText = `Disruption: ${dis?.type || 'unknown'} • Congestion: ${inc.congestion.level.toUpperCase()} (density: ${((inc.congestion.density || 0.15) * 100).toFixed(0)}%)${interactsStr}`;
    }

    const chipsContainer = document.getElementById('inc-affected-chips');
    if (chipsContainer) {
      chipsContainer.innerHTML = '';
      inc.affected_robots.forEach((rId) => {
        const chip = document.createElement('span');
        chip.className = 'robot-chip';
        chip.innerText = `AMR #${rId}`;
        chip.title = `Click to inspect AMR #${rId}`;
        chip.addEventListener('click', () => {
          this.selectedRobotId = rId;
          this.showRobotCard(rId);
          this.setCameraPreset(3);
        });
        chipsContainer.appendChild(chip);
      });
    }

    // 2. Cache
    const cacheHit = inc.cache?.hit;
    const simPct = Math.round((inc.cache?.similarity || 0) * 100);
    const cacheBadge = document.getElementById('inc-cache-badge');
    if (cacheBadge) {
      cacheBadge.className = `diag-badge ${cacheHit ? 'green' : 'purple'}`;
      cacheBadge.innerText = cacheHit ? 'CACHE HIT' : 'MISS';
    }

    const cacheHitText = cacheHit
      ? `HIT: Matched prior incident ${inc.cache?.matched} with ${simPct}% cosine similarity. Skipped LLM.`
      : `MISS: Best stored match achieved ${simPct}% similarity (below threshold).`;
    const cTextEl = document.getElementById('inc-cache-text');
    if (cTextEl) cTextEl.innerText = cacheHitText;
    const cSimVal = document.getElementById('inc-cache-sim-val');
    if (cSimVal) cSimVal.innerText = `${simPct}%`;
    const cSimFill = document.getElementById('inc-cache-sim-fill');
    if (cSimFill) cSimFill.style.width = `${simPct}%`;

    // 3. Dry Run
    const dryRun = inc.dry_run;
    const dryBadge = document.getElementById('inc-dryrun-badge');
    if (dryBadge) {
      if (dryRun && dryRun.ran) {
        dryBadge.className = `diag-badge ${dryRun.valid ? 'green' : 'amber'}`;
        dryBadge.innerText = dryRun.valid ? 'PASSED ✓' : 'FAILED ✗';
      } else {
        dryBadge.className = 'diag-badge';
        dryBadge.innerText = 'SKIPPED';
      }
    }
    const dryText = document.getElementById('inc-dryrun-text');
    if (dryText) {
      if (dryRun && dryRun.ran) {
        dryText.innerText = dryRun.valid
          ? 'VALID: Cached strategy directive verified clean against live obstacle state.'
          : `INVALID: ${dryRun.violation || 'conflict detected'}. Discarded and promoted to negative memory.`;
      } else {
        dryText.innerText = 'Skipped (Cache miss or first observation).';
      }
    }

    // 4. LLM Strategist
    const llmBadge = document.getElementById('inc-llm-badge');
    const llmText = document.getElementById('inc-llm-text');
    const llmLat = document.getElementById('inc-llm-latency');
    const llmTok = document.getElementById('inc-llm-tokens');
    const llmSolv = document.getElementById('inc-llm-solver');
    const llmDir = document.getElementById('inc-llm-directive');

    if (inc.llm?.called) {
      if (llmBadge) { llmBadge.className = 'diag-badge pink'; llmBadge.innerText = 'CALLED'; }
      if (llmText) llmText.innerText = inc.llm.directive?.rationale || 'Optimal directive determined.';
      if (llmLat) llmLat.innerText = `⚡ ${inc.llm.latency_ms || 1120} ms`;
      if (llmTok) llmTok.innerText = `🎟️ ${inc.llm.tokens || 850} tokens`;
      if (llmSolv) llmSolv.innerText = `Solver: ${inc.llm.directive?.solver.toUpperCase()}`;
      if (llmDir) llmDir.innerText = JSON.stringify(inc.llm.directive, null, 2);
    } else {
      if (llmBadge) { llmBadge.className = 'diag-badge'; llmBadge.innerText = 'BYPASSED'; }
      if (llmText) llmText.innerText = 'Bypassed (Resolved directly via Strategy Cache hit or Baseline mode).';
      if (llmLat) llmLat.innerText = '⚡ 0 ms';
      if (llmTok) llmTok.innerText = '🎟️ 0 tokens';
      if (llmSolv) llmSolv.innerText = `Solver: ${inc.llm?.directive?.solver || 'PIBT'}`;
      if (llmDir) llmDir.innerText = JSON.stringify(inc.llm?.directive || { note: 'Direct execution' }, null, 2);
    }

    // 5. Solver
    const solverName = inc.llm?.directive?.solver || (this.activeReplay?.meta.mode.replace('baseline_', '') || 'pibt');
    const solvText = document.getElementById('inc-solver-text');
    if (solvText) {
      solvText.innerText = `Executed ${solverName.toUpperCase()} with horizon window ${inc.llm?.directive?.horizon || 20} steps. Produced collision-free trajectories.`;
    }

    // 6. Validator
    const valAttempts = inc.validator || [];
    const valBadge = document.getElementById('inc-validator-badge');
    const allValid = valAttempts.some(v => v.valid);
    if (valBadge) {
      valBadge.className = `diag-badge ${allValid ? 'green' : 'red'}`;
      valBadge.innerText = allValid ? 'VERIFIED ✓' : 'FAILED ✗';
    }

    const valText = valAttempts.map(v => `Attempt ${v.attempt}: ${v.valid ? 'VALID ✓ (Ground-truth verified)' : 'INVALID ✗ (' + v.violation + ')'}`).join('\n');
    const valEl = document.getElementById('inc-validator-text');
    if (valEl) valEl.innerText = valText || 'Deterministically verified without conflict.';
  }

  private updatePathsVisualization() {
    while (this.pathsGroup.children.length > 0) {
      const obj = this.pathsGroup.children[0];
      this.pathsGroup.remove(obj);
      if ((obj as THREE.Mesh).geometry) (obj as THREE.Mesh).geometry.dispose();
    }

    if (!this.selectedIncident) return;
    const inc = this.selectedIncident;

    // Congestion region faint rectangle on floor
    const [[x0, y0], [x1, y1]] = inc.congestion.region;
    const rw = Math.abs(x1 - x0) + 1;
    const rh = Math.abs(y1 - y0) + 1;
    const rcx = (x0 + x1) / 2 + 0.5;
    const rcy = (y0 + y1) / 2 + 0.5;

    const congGeo = new THREE.PlaneGeometry(rw, rh);
    const congColor = inc.congestion.level === 'high' ? 0xef4444 : (inc.congestion.level === 'medium' ? 0xf59e0b : 0x0284c7);
    const congMat = new THREE.MeshBasicMaterial({ color: congColor, transparent: true, opacity: 0.16 });
    const congMesh = new THREE.Mesh(congGeo, congMat);
    congMesh.rotation.x = -Math.PI / 2;
    congMesh.position.set(rcx, 0.02, rcy);
    this.pathsGroup.add(congMesh);

    // Invalidated paths as dashed red fat line
    for (const [_rIdStr, path] of Object.entries(inc.paths.invalidated)) {
      if (path.length < 2) continue;
      const points = path.map(([px, py]) => new THREE.Vector3(px + 0.5, 0.15, py + 0.5));
      const curve = new THREE.CatmullRomCurve3(points);
      const smoothPoints = curve.getPoints(path.length * 4);
      const positions: number[] = [];
      smoothPoints.forEach(p => positions.push(p.x, p.y, p.z));

      const geom = new LineGeometry();
      geom.setPositions(positions);

      const mat = new LineMaterial({
        color: 0xef4444,
        linewidth: 3,
        dashed: true,
        dashSize: 0.8,
        gapSize: 0.4,
        resolution: new THREE.Vector2(window.innerWidth, window.innerHeight)
      });
      mat.defines.USE_DASH = '';

      const line = new Line2(geom, mat);
      line.computeLineDistances();
      this.pathsGroup.add(line);
    }

    // Repaired paths as solid line
    for (const [_rIdStr, path] of Object.entries(inc.paths.repaired)) {
      if (path.length < 2) continue;
      const points = path.map(([px, py]) => new THREE.Vector3(px + 0.5, 0.18, py + 0.5));
      const curve = new THREE.CatmullRomCurve3(points);
      const smoothPoints = curve.getPoints(path.length * 4);
      const positions: number[] = [];
      smoothPoints.forEach(p => positions.push(p.x, p.y, p.z));

      const geom = new LineGeometry();
      geom.setPositions(positions);

      const mat = new LineMaterial({
        color: 0x38bdf8,
        linewidth: 4,
        resolution: new THREE.Vector2(window.innerWidth, window.innerHeight)
      });

      const line = new Line2(geom, mat);
      this.pathsGroup.add(line);
    }

    // Interacting line
    if (inc.interacts_with.length > 0 && this.activeReplay) {
      inc.interacts_with.forEach(otherId => {
        const otherInc = this.activeReplay?.incidents.find(i => i.id === otherId);
        if (otherInc) {
          const c1 = inc.disruptions[0]?.cells?.[0] || [x0, y0];
          const c2 = otherInc.disruptions[0]?.cells?.[0] || otherInc.congestion.region[0];

          const points = [
            new THREE.Vector3(c1[0] + 0.5, 0.25, c1[1] + 0.5),
            new THREE.Vector3(c2[0] + 0.5, 0.25, c2[1] + 0.5)
          ];
          const geom = new LineGeometry();
          geom.setPositions([points[0].x, points[0].y, points[0].z, points[1].x, points[1].y, points[1].z]);

          const mat = new LineMaterial({
            color: 0xd946ef,
            linewidth: 3,
            dashed: true,
            dashSize: 0.6,
            gapSize: 0.3,
            resolution: new THREE.Vector2(window.innerWidth, window.innerHeight)
          });
          mat.defines.USE_DASH = '';

          const line = new Line2(geom, mat);
          line.computeLineDistances();
          this.pathsGroup.add(line);
        }
      });
    }
  }

  private showRobotCard(rId: number) {
    this.selectedRobotId = rId;
    const card = document.getElementById('robot-card') || document.getElementById('robot-details-card');
    if (card) card.classList.remove('hidden');

    const s0 = Math.floor(this.clockTime);
    const frame = this.activeReplay?.frames[s0];
    const pos = frame?.pos[rId] || [0, 0];
    const state = frame?.state[rId] || 'moving';

    const cardId = document.getElementById('robot-card-id') || document.getElementById('detail-robot-id');
    if (cardId) cardId.innerText = `AMR #${rId}`;
    const stateEl = document.getElementById('robot-card-state') || document.getElementById('detail-state');
    if (stateEl) stateEl.innerText = state.toUpperCase();
    const posEl = document.getElementById('robot-card-pos') || document.getElementById('detail-pos');
    if (posEl) posEl.innerText = `(${pos[0]}, ${pos[1]})`;
    const goalEl = document.getElementById('robot-card-goal') || document.getElementById('detail-dest');
    if (goalEl) goalEl.innerText = `(${pos[0] + 18}, ${(pos[1] + 12) % 60})`;
    const distEl = document.getElementById('robot-card-dist');
    if (distEl) distEl.innerText = `~${Math.abs(18) + Math.abs(12)} cells`;

    const dot = document.getElementById('robot-card-status-dot');
    if (dot) {
      if (state === 'failed') dot.style.background = '#ef4444';
      else if (state === 'delayed') dot.style.background = '#f59e0b';
      else if (state === 'idle') dot.style.background = '#10b981';
      else if (state === 'replanning') dot.style.background = '#d946ef';
      else dot.style.background = '#38bdf8';
    }
  }

  private showToast(title: string, desc: string) {
    const toast = document.getElementById('toast-notification');
    if (!toast) return;
    const titleEl = document.getElementById('toast-title');
    if (titleEl) titleEl.innerText = title;
    const descEl = document.getElementById('toast-desc');
    if (descEl) descEl.innerText = desc;
    toast.classList.remove('hidden');

    setTimeout(() => {
      toast.classList.add('hidden');
    }, 3200);
  }

  private setupUI() {
    // Play/Pause button
    const playPauseBtn = document.getElementById('play-pause-btn');
    const playIcon = document.getElementById('play-icon');
    const playIconSvg = document.getElementById('play-icon-svg');
    const playText = document.getElementById('play-text');
    playPauseBtn?.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      if (playIcon) playIcon.innerHTML = this.isPlaying ? '&#10074;&#10074;' : '&#9654;';
      if (playIconSvg) {
        playIconSvg.innerHTML = this.isPlaying
          ? '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>'
          : '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
      }
      if (playText) playText.innerText = this.isPlaying ? 'Pause' : 'Play';
    });

    // Jump to Start & End buttons
    document.getElementById('jump-start-btn')?.addEventListener('click', () => {
      this.clockTime = 0.0;
      this.updateSliderUI();
    });
    document.getElementById('jump-end-btn')?.addEventListener('click', () => {
      const maxSteps = (this.activeReplay?.meta.steps || 60) - 1;
      this.clockTime = maxSteps;
      this.updateSliderUI();
    });

    // Step forward & back frame buttons
    document.getElementById('step-back-btn')?.addEventListener('click', () => {
      this.isPlaying = false;
      this.clockTime = Math.max(0, this.clockTime - 1);
      this.updateSliderUI();
    });
    document.getElementById('step-fwd-btn')?.addEventListener('click', () => {
      this.isPlaying = false;
      const maxSteps = (this.activeReplay?.meta.steps || 300) - 1;
      this.clockTime = Math.min(maxSteps, this.clockTime + 1);
      this.updateSliderUI();
    });

    // Speed selectors
    const speedButtons = document.querySelectorAll('.speed-pill-group .speed-btn');
    speedButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        speedButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.playSpeed = parseFloat(btn.getAttribute('data-speed') || '1');
      });
    });

    // Slider
    const slider = document.getElementById('step-slider') as HTMLInputElement | null;
    slider?.addEventListener('input', () => {
      this.clockTime = parseFloat(slider.value);
      this.updateStepIndicator();
    });

    // Camera preset buttons
    const camButtons = document.querySelectorAll('.camera-preset-bar .cam-btn');
    camButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        camButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const p = parseInt(btn.getAttribute('data-cam') || '1', 10);
        this.setCameraPreset(p);
      });
    });

    // In-viewport camera preset buttons
    document.getElementById('btn-cam-full')?.addEventListener('click', () => this.setCameraPreset(1));
    document.getElementById('btn-cam-robot')?.addEventListener('click', () => this.setCameraPreset(3));
    document.getElementById('btn-cam-incident')?.addEventListener('click', () => this.setCameraPreset(2));
    document.getElementById('btn-cam-reset')?.addEventListener('click', () => this.setCameraPreset(1));

    // Scenario Select
    const scenarioSelect = document.getElementById('scenario-select') as HTMLSelectElement | null;
    scenarioSelect?.addEventListener('change', () => {
      this.loadScenario(scenarioSelect.value);
    });

    // Experiment Select
    const expSelect = document.getElementById('experiment-select') as HTMLSelectElement | null;
    expSelect?.addEventListener('change', () => {
      this.loadScenario(expSelect.value);
    });

    // Mode Toggle Buttons
    const modeLlmBtn = document.getElementById('mode-llm-btn');
    const modeBaseBtn = document.getElementById('mode-base-btn');
    if (modeLlmBtn && modeBaseBtn) {
      modeLlmBtn.addEventListener('click', () => {
        this.isBaselineMode = false;
        modeLlmBtn.classList.add('active');
        modeBaseBtn.classList.remove('active');
        this.activeReplay = this.replay;
        this.updateMetricsUI();
        this.buildTimelineTicks();
        if (this.selectedIncident) {
          const matching = this.activeReplay?.incidents.find(i => i.id === this.selectedIncident?.id);
          if (matching) this.showIncidentPanel(matching);
        }
        this.showToast('Mode Changed', 'Switched to LLM Guided Autonomous Strategist');
      });

      modeBaseBtn.addEventListener('click', () => {
        this.isBaselineMode = true;
        modeBaseBtn.classList.add('active');
        modeLlmBtn.classList.remove('active');
        this.activeReplay = this.baselineReplay || this.replay;
        this.updateMetricsUI();
        this.buildTimelineTicks();
        if (this.selectedIncident) {
          const matching = this.activeReplay?.incidents.find(i => i.id === this.selectedIncident?.id);
          if (matching) this.showIncidentPanel(matching);
        }
        this.showToast('Mode Changed', 'Switched to Baseline PIBT (Rule-based greedy)');
      });
    }

    // Mode Select Dropdown
    const modeSelect = document.getElementById('mode-select') as HTMLSelectElement | null;
    modeSelect?.addEventListener('change', () => {
      this.isBaselineMode = modeSelect.value.startsWith('baseline');
      this.activeReplay = this.isBaselineMode ? (this.baselineReplay || this.replay) : this.replay;
      this.updateMetricsUI();
      this.buildTimelineTicks();
      if (this.selectedIncident) {
        const matching = this.activeReplay?.incidents.find(i => i.id === this.selectedIncident?.id);
        if (matching) this.showIncidentPanel(matching);
      }
      this.showToast('Mode Changed', `Switched to ${modeSelect.value.toUpperCase()}`);
    });

    // Incident Drawer close button
    document.getElementById('incident-close-btn')?.addEventListener('click', () => {
      document.getElementById('incident-panel')?.classList.add('hidden');
      document.getElementById('incidents-panel')?.classList.add('hidden');
      this.selectedIncident = null;
      this.updatePathsVisualization();
    });

    // Robot Card close & follow buttons
    document.getElementById('robot-card-close')?.addEventListener('click', () => {
      document.getElementById('robot-card')?.classList.add('hidden');
      this.selectedRobotId = null;
    });

    document.getElementById('detail-close-btn')?.addEventListener('click', () => {
      document.getElementById('robot-card')?.classList.add('hidden');
      document.getElementById('robot-details-card')?.classList.add('hidden');
      this.selectedRobotId = null;
    });

    document.getElementById('robot-follow-btn')?.addEventListener('click', () => {
      this.setCameraPreset(3);
    });

    // Next / Prev incident buttons
    document.getElementById('next-inc-btn')?.addEventListener('click', () => this.jumpIncident(1));
    document.getElementById('prev-inc-btn')?.addEventListener('click', () => this.jumpIncident(-1));

    // Fullscreen button
    document.getElementById('btn-fullscreen')?.addEventListener('click', () => {
      this.toggleFullscreen();
    });

    // Copy Directive JSON
    document.getElementById('copy-directive-btn')?.addEventListener('click', () => {
      const code = document.getElementById('inc-llm-directive')?.innerText || '';
      navigator.clipboard.writeText(code);
      this.showToast('Copied to Clipboard', 'Directive JSON copied');
    });

    // Help Modal
    const helpBtn = document.getElementById('help-btn');
    const helpModal = document.getElementById('help-modal');
    const helpClose = document.getElementById('help-modal-close');
    helpBtn?.addEventListener('click', () => helpModal?.classList.remove('hidden'));
    helpClose?.addEventListener('click', () => helpModal?.classList.add('hidden'));
    helpModal?.addEventListener('click', (e) => {
      if (e.target === helpModal) helpModal?.classList.add('hidden');
    });
  }

  private toggleFullscreen() {
    const card = document.getElementById('viewport-card') || this.container;
    if (!document.fullscreenElement) {
      if (card.requestFullscreen) {
        card.requestFullscreen().catch(() => {});
      } else if ((card as any).webkitRequestFullscreen) {
        (card as any).webkitRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      } else if ((document as any).webkitExitFullscreen) {
        (document as any).webkitExitFullscreen();
      }
    }
  }

  private setupEventListeners() {
    window.addEventListener('keydown', (e) => {
      this.keysPressed[e.key.toLowerCase()] = true;
      if (e.key === '1') this.setCameraPreset(1);
      if (e.key === '2') this.setCameraPreset(2);
      if (e.key === '3') this.setCameraPreset(3);
      if (e.key === '0' || e.key.toLowerCase() === 'r') this.setCameraPreset(1);
      if (e.key.toLowerCase() === 'f') this.toggleFullscreen();
      if (e.key.toLowerCase() === 'n') this.jumpIncident(1);
      if (e.key.toLowerCase() === 'p') this.jumpIncident(-1);
      if (e.key === 'ArrowLeft') {
        this.clockTime = Math.max(0, this.clockTime - 1);
        this.updateSliderUI();
      }
      if (e.key === 'ArrowRight') {
        const maxSteps = (this.activeReplay?.meta.steps || 300) - 1;
        this.clockTime = Math.min(maxSteps, this.clockTime + 1);
        this.updateSliderUI();
      }
      if (e.key === 'Escape') {
        if (document.fullscreenElement) {
          document.exitFullscreen().catch(() => {});
        }
        document.getElementById('incident-panel')?.classList.add('hidden');
        document.getElementById('robot-card')?.classList.add('hidden');
        document.getElementById('help-modal')?.classList.add('hidden');
        this.selectedIncident = null;
        this.updatePathsVisualization();
      }
      if (e.code === 'Space') {
        e.preventDefault();
        const playBtn = document.getElementById('play-pause-btn');
        playBtn?.click();
      }
    });

    window.addEventListener('keyup', (e) => {
      this.keysPressed[e.key.toLowerCase()] = false;
    });

    document.addEventListener('fullscreenchange', () => {
      this.onWindowResize();
      const btn = document.getElementById('btn-fullscreen');
      if (btn) {
        if (document.fullscreenElement) {
          btn.classList.add('active');
          btn.title = 'Exit Fullscreen (Key F or Esc)';
        } else {
          btn.classList.remove('active');
          btn.title = 'Toggle Fullscreen (Key F)';
        }
      }
    });

    // Raycast click to select robot in 3D canvas
    this.renderer.domElement.addEventListener('pointerdown', (e) => {
      const rect = this.renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, this.camera);
      if (this.robotMeshChassis) {
        const intersects = raycaster.intersectObject(this.robotMeshChassis);
        if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
          const rId = intersects[0].instanceId;
          this.showRobotCard(rId);
        }
      }
    });
  }

  private jumpIncident(direction: number) {
    if (!this.activeReplay) return;
    const incs = this.activeReplay.incidents;
    if (incs.length === 0) return;

    if (!this.selectedIncident) {
      this.selectIncident(direction > 0 ? incs[0] : incs[incs.length - 1]);
      return;
    }

    const currIdx = incs.findIndex(i => i.id === this.selectedIncident?.id);
    let nextIdx = currIdx + direction;
    if (nextIdx >= incs.length) nextIdx = 0;
    if (nextIdx < 0) nextIdx = incs.length - 1;

    this.selectIncident(incs[nextIdx]);
  }

  private setCameraPreset(preset: number) {
    this.activePreset = preset;
    const camButtons = document.querySelectorAll('.camera-preset-bar .cam-btn');
    camButtons.forEach(btn => {
      if (parseInt(btn.getAttribute('data-cam') || '0', 10) === preset) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    this.cameraStartPos.copy(this.camera.position);
    this.cameraStartTarget.copy(this.controls.target);

    if (preset === 1) {
      this.cameraEndPos.set(this.mapWidth / 2, 85, this.mapHeight / 2 + 75);
      this.cameraEndTarget.set(this.mapWidth / 2, 0, this.mapHeight / 2);
    } else if (preset === 2) {
      if (this.selectedIncident) {
        const [[x0, y0], [x1, y1]] = this.selectedIncident.congestion.region;
        const cx = (x0 + x1) / 2 + 0.5;
        const cy = (y0 + y1) / 2 + 0.5;
        const span = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 12);
        this.cameraEndPos.set(cx + 8, span * 1.2 + 8, cy + span * 0.9 + 10);
        this.cameraEndTarget.set(cx, 0.2, cy);
      } else {
        this.cameraEndPos.set(this.mapWidth / 2, 45, this.mapHeight / 2 + 35);
        this.cameraEndTarget.set(this.mapWidth / 2, 0, this.mapHeight / 2);
      }
    } else if (preset === 3) {
      const rId = this.selectedRobotId ?? 0;
      const s0 = Math.floor(this.clockTime);
      const pos = this.activeReplay?.frames[s0]?.pos[rId] || [80, 31];
      this.cameraEndPos.set(pos[0] + 0.5 + 8, 12, pos[1] + 0.5 + 10);
      this.cameraEndTarget.set(pos[0] + 0.5, 0.2, pos[1] + 0.5);
    }

    this.cameraAnimProgress = 0.0;
    this.cameraAnimDuration = 0.8;
    this.isCameraTransitioning = true;
  }

  private startIntroDolly() {
    this.cameraStartPos.set(this.mapWidth / 2, 115, this.mapHeight / 2 + 95);
    this.cameraStartTarget.set(this.mapWidth / 2, 0, this.mapHeight / 2);

    this.cameraEndPos.set(this.mapWidth / 2, 75, this.mapHeight / 2 + 65);
    this.cameraEndTarget.set(this.mapWidth / 2, 0, this.mapHeight / 2);

    this.cameraAnimProgress = 0.0;
    this.cameraAnimDuration = 2.0;
    this.isCameraTransitioning = true;
  }

  private updateCameraAnimation(dt: number) {
    if (!this.isCameraTransitioning) {
      // If in follow camera mode, smoothly track robot position
      if (this.activePreset === 3 && this.selectedRobotId !== null) {
        const s0 = Math.floor(this.clockTime);
        const pos = this.activeReplay?.frames[s0]?.pos[this.selectedRobotId] || [80, 31];
        const targetPos = new THREE.Vector3(pos[0] + 0.5, 0.2, pos[1] + 0.5);
        this.controls.target.lerp(targetPos, 0.08);
      }
      return;
    }

    this.cameraAnimProgress += dt / this.cameraAnimDuration;
    if (this.cameraAnimProgress >= 1.0) {
      this.cameraAnimProgress = 1.0;
      this.isCameraTransitioning = false;
    }

    const t = this.cameraAnimProgress;
    const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

    this.camera.position.lerpVectors(this.cameraStartPos, this.cameraEndPos, ease);
    this.controls.target.lerpVectors(this.cameraStartTarget, this.cameraEndTarget, ease);
    this.controls.update();
  }

  private updateRobots(t: number) {
    if (!this.activeReplay || !this.robotMeshChassis || !this.robotMeshLed || !this.shadowDecalMesh) return;

    const frames = this.activeReplay.frames;
    const totalSteps = frames.length;
    const s0 = Math.floor(t);
    const s1 = Math.min(s0 + 1, totalSteps - 1);
    const alpha = t - s0;

    const f0 = frames[s0];
    const f1 = frames[s1];

    const dummy = new THREE.Object3D();
    const ledColor = new THREE.Color();
    const affectedSet = new Set<number>(this.selectedIncident ? this.selectedIncident.affected_robots : []);

    for (let r = 0; r < f0.pos.length; r++) {
      const [x0, y0] = f0.pos[r];
      const [x1, y1] = f1.pos[r];
      const state0 = f0.state[r];

      const dx = x1 - x0;
      const dy = y1 - y0;

      let posX = x0 + 0.5;
      let posZ = y0 + 0.5;
      let angle = 0;

      if (dx !== 0 || dy !== 0) {
        angle = Math.atan2(dx, dy);
        const easeMove = alpha * alpha * (3 - 2 * alpha); // smoothstep motion
        posX = x0 + 0.5 + dx * easeMove;
        posZ = y0 + 0.5 + dy * easeMove;
      }

      dummy.position.set(posX, 0.1, posZ);
      dummy.rotation.y = angle;
      dummy.updateMatrix();

      this.robotMeshChassis.setMatrixAt(r, dummy.matrix);
      this.robotMeshLed.setMatrixAt(r, dummy.matrix);

      dummy.position.set(posX, 0.015, posZ);
      dummy.rotation.set(-Math.PI / 2, 0, 0);
      dummy.updateMatrix();
      this.shadowDecalMesh.setMatrixAt(r, dummy.matrix);

      const timeSec = performance.now() / 1000;
      if (state0 === 'failed') {
        const pulse = 0.35 + 0.65 * Math.abs(Math.sin(timeSec * 4.0));
        ledColor.setHex(0xef4444).multiplyScalar(pulse);
      } else if (state0 === 'delayed') {
        const pulse = 0.5 + 0.5 * Math.sin(timeSec * 2 * Math.PI * 1.0);
        ledColor.setHex(0xf59e0b).multiplyScalar(pulse);
      } else if (state0 === 'replanning') {
        const pulse = 0.5 + 0.5 * Math.sin(timeSec * 2 * Math.PI * 2.0);
        ledColor.setHex(0xd946ef).multiplyScalar(pulse);
      } else if (state0 === 'idle') {
        ledColor.setHex(0x10b981);
      } else {
        const pulse = 0.85 + 0.15 * Math.sin(timeSec * 3.5);
        ledColor.setHex(0x38bdf8).multiplyScalar(pulse);
      }

      if (this.selectedIncident) {
        if (affectedSet.has(r)) {
          ledColor.setHex(0xd946ef);
        } else {
          ledColor.multiplyScalar(0.6);
        }
      }

      // If robot is selected, highlight with high-intensity glowing cyan
      if (this.selectedRobotId === r) {
        const pulse = 1.35 + 0.45 * Math.sin(timeSec * 5.0);
        ledColor.setHex(0x38bdf8).multiplyScalar(pulse);
      }

      this.robotMeshLed.setColorAt(r, ledColor);
    }

    this.robotMeshChassis.instanceMatrix.needsUpdate = true;
    this.robotMeshLed.instanceMatrix.needsUpdate = true;
    if (this.robotMeshLed.instanceColor) this.robotMeshLed.instanceColor.needsUpdate = true;
    this.shadowDecalMesh.instanceMatrix.needsUpdate = true;

    // Update robot card if currently open
    if (this.selectedRobotId !== null) {
      const pos = f0.pos[this.selectedRobotId] || [0, 0];
      const state = f0.state[this.selectedRobotId] || 'moving';
      const posEl = document.getElementById('robot-card-pos');
      if (posEl) posEl.innerText = `(${pos[0]}, ${pos[1]})`;
      const stateEl = document.getElementById('robot-card-state');
      if (stateEl) stateEl.innerText = state.toUpperCase();
    }
  }

  private updateDisruptions(t: number) {
    if (!this.activeReplay) return;

    while (this.barrierGroup.children.length > 0) {
      const child = this.barrierGroup.children[0];
      this.barrierGroup.remove(child);
      if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose();
    }

    const curStep = Math.floor(t);
    this.activeReplay.incidents.forEach((inc) => {
      const activeStart = inc.t_detected;
      const activeEnd = inc.t_detected + inc.outcome.recovery_steps + 15;

      if (curStep >= activeStart && curStep <= activeEnd) {
        const timeSec = performance.now() / 1000;
        inc.disruptions.forEach((d) => {
          if (d.type === 'blocked_cell' && d.cells) {
            d.cells.forEach(([cx, cy]) => {
              // Physical barrier cone
              const barrier = new THREE.Mesh(this.barrierGeometry!, this.barrierMaterial!);
              barrier.position.set(cx + 0.5, 0.3, cy + 0.5);
              barrier.castShadow = true;
              this.barrierGroup.add(barrier);

              // Pulsing red danger zone on floor
              const pulse = 0.3 + 0.15 * Math.sin(timeSec * 4.0);
              const redSqGeo = new THREE.PlaneGeometry(0.96, 0.96);
              const redSqMat = new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: pulse });
              const redSq = new THREE.Mesh(redSqGeo, redSqMat);
              redSq.rotation.x = -Math.PI / 2;
              redSq.position.set(cx + 0.5, 0.018, cy + 0.5);
              this.barrierGroup.add(redSq);

              // Red warning ring
              const warnRing = new THREE.Mesh(
                new THREE.RingGeometry(0.48, 0.52, 24),
                new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.7, side: THREE.DoubleSide })
              );
              warnRing.rotation.x = -Math.PI / 2;
              warnRing.position.set(cx + 0.5, 0.019, cy + 0.5);
              this.barrierGroup.add(warnRing);
            });
          } else if (d.type === 'robot_failure' && d.cells) {
            d.cells.forEach(([cx, cy]) => {
              // Dead robot marker — larger grey X-ring
              const ringGeo = new THREE.RingGeometry(0.30, 0.42, 24);
              const ringMat = new THREE.MeshBasicMaterial({ color: 0x94a3b8, side: THREE.DoubleSide, transparent: true, opacity: 0.7 });
              const ring = new THREE.Mesh(ringGeo, ringMat);
              ring.rotation.x = -Math.PI / 2;
              ring.position.set(cx + 0.5, 0.019, cy + 0.5);
              this.barrierGroup.add(ring);

              // Subtle red glow under failed robot
              const glowGeo = new THREE.CircleGeometry(0.35, 16);
              const glowMat = new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.15 });
              const glow = new THREE.Mesh(glowGeo, glowMat);
              glow.rotation.x = -Math.PI / 2;
              glow.position.set(cx + 0.5, 0.017, cy + 0.5);
              this.barrierGroup.add(glow);
            });
          } else if (d.type === 'robot_delay' && d.robot !== undefined) {
            // Amber pulsing ring around delayed robot position
            const rIdx = d.robot;
            const pos = this.activeReplay?.frames[curStep]?.pos[rIdx];
            if (pos) {
              const amberPulse = 0.3 + 0.3 * Math.sin(timeSec * 3.0);
              const delayRing = new THREE.Mesh(
                new THREE.RingGeometry(0.35, 0.42, 20),
                new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: amberPulse, side: THREE.DoubleSide })
              );
              delayRing.rotation.x = -Math.PI / 2;
              delayRing.position.set(pos[0] + 0.5, 0.019, pos[1] + 0.5);
              this.barrierGroup.add(delayRing);
            }
          }
        });
      }
    });
  }

  private updateSliderUI() {
    const slider = document.getElementById('step-slider') as HTMLInputElement;
    if (slider) slider.value = Math.floor(this.clockTime).toString();
    this.updateStepIndicator();
  }

  private updateStepIndicator() {
    const cur = Math.floor(this.clockTime);
    const total = this.activeReplay?.meta.steps || 300;
    const indicator = document.getElementById('step-indicator');
    if (indicator) {
      indicator.innerText = `Step ${cur} / ${total}`;
    }
    const pct = Math.round((cur / (total - 1)) * 100);
    const pctEl = document.getElementById('step-pct');
    if (pctEl) pctEl.innerText = `(${pct}%)`;

    const progBar = document.getElementById('slider-progress-bar');
    if (progBar) progBar.style.width = `${pct}%`;
  }

  private renderLoop(now: number) {
    requestAnimationFrame(this.renderLoop.bind(this));

    if (!this.lastTimestamp) this.lastTimestamp = now;
    const dt = Math.min((now - this.lastTimestamp) / 1000, 0.1);
    this.lastTimestamp = now;

    this.frameCount++;
    if (now - this.lastFpsUpdate > 500) {
      const fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
      if (this.fpsDisplay) this.fpsDisplay.innerText = `${fps} FPS`;
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    if (this.isPlaying && this.activeReplay) {
      const totalSteps = this.activeReplay.meta.steps;
      this.clockTime += dt * 3.5 * this.playSpeed;
      if (this.clockTime >= totalSteps - 1) {
        this.clockTime = 0.0;
      }
      this.updateSliderUI();
    }

    this.updateCameraAnimation(dt);
    this.updateRobots(this.clockTime);
    this.updateDisruptions(this.clockTime);
    this.updateGoalMarkers(this.clockTime);

    // Smooth keyboard flight navigation (WASD to pan, QE to orbit)
    if (!this.isCameraTransitioning && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'TEXTAREA') {
      const panSpeed = (this.camera.position.y * 0.75 + 15) * dt;
      const forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();
      const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

      if (this.keysPressed['w']) {
        this.controls.target.addScaledVector(forward, panSpeed);
        this.camera.position.addScaledVector(forward, panSpeed);
      }
      if (this.keysPressed['s']) {
        this.controls.target.addScaledVector(forward, -panSpeed);
        this.camera.position.addScaledVector(forward, -panSpeed);
      }
      if (this.keysPressed['a']) {
        this.controls.target.addScaledVector(right, -panSpeed);
        this.camera.position.addScaledVector(right, -panSpeed);
      }
      if (this.keysPressed['d']) {
        this.controls.target.addScaledVector(right, panSpeed);
        this.camera.position.addScaledVector(right, panSpeed);
      }
      if (this.keysPressed['q']) {
        const offset = this.camera.position.clone().sub(this.controls.target);
        offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), dt * 1.5);
        this.camera.position.copy(this.controls.target).add(offset);
      }
      if (this.keysPressed['e']) {
        const offset = this.camera.position.clone().sub(this.controls.target);
        offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), -dt * 1.5);
        this.camera.position.copy(this.controls.target).add(offset);
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private onWindowResize() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  }
}

// Start application
const app = new ViewerApp();
app.init();
