import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

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
  goals?: [number, number][];
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

// --- Main AMR Digital Twin Viewer Application ---
class ViewerApp {
  private container!: HTMLElement;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;

  private replay: Replay | null = null;
  private baselineReplay: Replay | null = null;
  private activeReplay: Replay | null = null;
  private currentExperiment = 'latest_experiment';
  private currentMode = 'llm';

  private mapWidth = 161;
  private mapHeight = 63;
  private obstacleSet = new Set<string>();

  // 3D Object Groups & Meshes
  private shelfInstancedMesh: THREE.InstancedMesh | null = null;
  private robotMeshChassis: THREE.InstancedMesh | null = null;
  private robotMeshLed: THREE.InstancedMesh | null = null;
  private shadowDecalMesh: THREE.InstancedMesh | null = null;
  private barrierGroup = new THREE.Group();
  private pathsGroup = new THREE.Group();
  private floorLinesGroup = new THREE.Group();
  private labelsGroup = new THREE.Group();
  private goalMarkersGroup = new THREE.Group();
  private selectedRingMesh: THREE.Mesh | null = null;

  // Loaded Geometries & Materials
  private shelfGeometry: THREE.BufferGeometry | null = null;
  private shelfMaterial: THREE.Material | THREE.Material[] | null = null;
  private barrierGeometry: THREE.BufferGeometry | null = null;
  private barrierMaterial: THREE.Material | THREE.Material[] | null = null;
  private robotChassisGeom: THREE.BufferGeometry | null = null;
  private robotLedGeom: THREE.BufferGeometry | null = null;

  // Selection & Visibility States
  private selectedRobotId: number | null = null;
  private selectedIncident: Incident | null = null;
  private showAllRoutes = false;
  private showOnlySelectedRoute = true;
  private showOrigRoutes = true;
  private showRepairedRoutes = true;
  private showBlockedCells = true;
  private showRobotIds = true;
  private showDestinations = true;

  // Optimized Geometry Caching
  private lastRenderedIntStep = -1;
  private lastSelectedRobotId: number | null = null;
  private needsVisualRefresh = true;

  // Clock & Playback
  private clockTime = 0.0;
  private isPlaying = true;
  private playSpeed = 1.0;
  private lastTimestamp = 0;

  // Camera State & Animation Modes
  private currentCameraMode: 'full' | 'robot' | 'incident' = 'full';
  private isCameraTransitioning = false;
  private cameraStartPos = new THREE.Vector3();
  private cameraStartTarget = new THREE.Vector3();
  private cameraEndPos = new THREE.Vector3();
  private cameraEndTarget = new THREE.Vector3();
  private cameraAnimProgress = 1.0;
  private cameraAnimDuration = 1.0;
  private robotHeadings: number[] = [];

  // In-Viewport Projected Screen-Space Callouts
  private calloutRobotEl!: HTMLElement;
  private calloutIncidentEl!: HTMLElement;

  async init() {
    this.container = document.getElementById('canvas-container')!;
    this.calloutRobotEl = document.getElementById('callout-robot')!;
    this.calloutIncidentEl = document.getElementById('callout-incident')!;

    this.setupScene();
    this.setupRenderer();
    this.setupCamera();
    this.setupLighting();

    await this.loadMap();
    await this.loadTexturesAndHDRI();
    await this.loadModels();
    await this.loadScenario(this.currentExperiment);

    this.setupUI();
    this.setupEventListeners();

    // Default selection: select Robot #12 if available or #0
    const numRobots = this.activeReplay?.frames[0]?.pos.length || 30;
    const defaultRobot = numRobots > 12 ? 12 : 0;
    this.selectRobot(defaultRobot, false);

    // Initial camera mode: dynamic full warehouse fit
    this.focusFullWarehouse();

    window.addEventListener('resize', this.onWindowResize.bind(this));
    requestAnimationFrame(this.renderLoop.bind(this));
  }

  private setupScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07111b);
    this.scene.add(this.barrierGroup);
    this.scene.add(this.pathsGroup);
    this.scene.add(this.floorLinesGroup);
    this.scene.add(this.goalMarkersGroup);
    this.scene.add(this.labelsGroup);

    // Glowing cyan ring under selected robot
    const ringGeo = new THREE.RingGeometry(0.42, 0.54, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x46c7ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
    });
    this.selectedRingMesh = new THREE.Mesh(ringGeo, ringMat);
    this.selectedRingMesh.rotation.x = -Math.PI / 2;
    this.selectedRingMesh.position.set(0, 0.02, 0);
    this.selectedRingMesh.visible = false;
    this.scene.add(this.selectedRingMesh);
  }

  private setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.container.appendChild(this.renderer.domElement);
  }

  private setupCamera() {
    this.camera = new THREE.PerspectiveCamera(46, this.container.clientWidth / this.container.clientHeight, 0.5, 1000);
    this.camera.position.set(80, 115, 150);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // Don't go below floor
    this.controls.target.set(80, 0, 31.5);
    this.controls.update();
  }

  private setupLighting() {
    const dirLight = new THREE.DirectionalLight(0xfff8ee, 2.4);
    dirLight.position.set(80, 80, -25);
    dirLight.target.position.set(80, 0, 31.5);
    this.scene.add(dirLight);
    this.scene.add(dirLight.target);

    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 10;
    dirLight.shadow.camera.far = 200;
    dirLight.shadow.bias = -0.0004;

    const d = 96;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d * 0.7;
    dirLight.shadow.camera.bottom = -d * 0.7;

    const ambLight = new THREE.AmbientLight(0x0e1726, 1.6);
    this.scene.add(ambLight);
  }

  private async loadTexturesAndHDRI() {
    const hdrLoader = new HDRLoader();
    try {
      const hdrTexture = await hdrLoader.loadAsync('/textures/warehouse.hdr');
      hdrTexture.mapping = THREE.EquirectangularReflectionMapping;
      this.scene.environment = hdrTexture;
    } catch (e) {
      console.warn('Could not load warehouse.hdr, using neutral lighting fallback', e);
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
    const floorGeo = new THREE.PlaneGeometry(this.mapWidth, this.mapHeight);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x1a2330,
      roughness: 0.84,
      metalness: 0.1,
      map: texture || undefined,
    });

    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(this.mapWidth / 2, 0, this.mapHeight / 2);
    floor.receiveShadow = true;
    this.scene.add(floor);

    this.createAisleLaneMarkings();
  }

  private createAisleLaneMarkings() {
    const lineMat = new THREE.LineBasicMaterial({ color: 0x243649, transparent: true, opacity: 0.6 });
    for (let y = 1; y < this.mapHeight; y += 4) {
      const points = [
        new THREE.Vector3(1, 0.012, y + 0.5),
        new THREE.Vector3(this.mapWidth - 1, 0.012, y + 0.5),
      ];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geom, lineMat);
      this.floorLinesGroup.add(line);
    }
  }

  private async loadMap() {
    try {
      const res = await fetch('/maps/warehouse-10-20-10-2-1.map');
      const text = await res.text();
      const lines = text.split('\n').filter((l) => l.trim().length > 0);

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

  private async loadModels() {
    const gltfLoader = new GLTFLoader();

    // 1. Shelf Model
    try {
      const shelfGltf = await gltfLoader.loadAsync('/models/shelf.glb');
      shelfGltf.scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          this.shelfGeometry = mesh.geometry;
          this.shelfMaterial = mesh.material;
        }
      });
    } catch (e) {
      console.warn('Shelf glb load error, using box fallback', e);
    }

    if (!this.shelfGeometry) {
      this.shelfGeometry = new THREE.BoxGeometry(0.96, 1.8, 0.96);
      this.shelfMaterial = new THREE.MeshStandardMaterial({ color: 0x4a3b2c, roughness: 0.65 });
    }
    this.createShelves();

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
      console.warn('Barrier glb load error, using cone fallback', e);
    }

    if (!this.barrierGeometry) {
      this.barrierGeometry = new THREE.ConeGeometry(0.3, 0.6, 16);
      this.barrierMaterial = new THREE.MeshStandardMaterial({ color: 0xf04f55 });
    }

    // 3. Robot AMR Model
    this.createAmrModelGeometries();
    this.createRobotMeshes();
  }

  private createAmrModelGeometries() {
    // 1. Main Industrial Rectangular Chassis Body
    const body = new THREE.BoxGeometry(0.68, 0.16, 0.48);
    body.translate(0, 0.09, 0);

    // 2. Drive Wheels (left & right sides)
    const wheelL = new THREE.CylinderGeometry(0.075, 0.075, 0.06, 14);
    wheelL.rotateZ(Math.PI / 2);
    wheelL.translate(-0.35, 0.075, 0);

    const wheelR = new THREE.CylinderGeometry(0.075, 0.075, 0.06, 14);
    wheelR.rotateZ(Math.PI / 2);
    wheelR.translate(0.35, 0.075, 0);

    // 3. Center LiDAR Scanner Turret
    const lidarBase = new THREE.CylinderGeometry(0.08, 0.08, 0.06, 14);
    lidarBase.translate(0, 0.20, 0);

    const lidarDome = new THREE.CylinderGeometry(0.05, 0.05, 0.04, 14);
    lidarDome.translate(0, 0.24, 0);

    // 4. Direction & Heading: Front bumper bar + Direction chevron arrow pointing forward (+Z)
    const frontBar = new THREE.BoxGeometry(0.32, 0.04, 0.05);
    frontBar.translate(0, 0.12, 0.25);

    const arrow = new THREE.ConeGeometry(0.07, 0.14, 3);
    arrow.rotateX(Math.PI / 2);
    arrow.translate(0, 0.18, 0.12);

    this.robotChassisGeom = mergeGeometries([body, wheelL, wheelR, lidarBase, lidarDome, frontBar, arrow]);

    // 5. LED Status Indicator: Top ring around LiDAR + Front headlight strip
    const ledRing = new THREE.TorusGeometry(0.24, 0.02, 8, 20);
    ledRing.rotateX(Math.PI / 2);
    ledRing.translate(0, 0.18, 0);

    const ledFront = new THREE.BoxGeometry(0.28, 0.03, 0.03);
    ledFront.translate(0, 0.12, 0.25);

    this.robotLedGeom = mergeGeometries([ledRing, ledFront]);
  }

  private createShelves() {
    const obstacleCoords: [number, number][] = [];
    this.obstacleSet.forEach((coordStr) => {
      const [x, y] = coordStr.split(',').map(Number);
      obstacleCoords.push([x, y]);
    });

    const count = obstacleCoords.length;
    this.shelfInstancedMesh = new THREE.InstancedMesh(this.shelfGeometry!, this.shelfMaterial!, count);
    this.shelfInstancedMesh.castShadow = true;
    this.shelfInstancedMesh.receiveShadow = true;

    const dummy = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      const [x, y] = obstacleCoords[i];
      dummy.position.set(x + 0.5, 0.9, y + 0.5);
      dummy.updateMatrix();
      this.shelfInstancedMesh.setMatrixAt(i, dummy.matrix);
    }

    this.shelfInstancedMesh.instanceMatrix.needsUpdate = true;
    this.scene.add(this.shelfInstancedMesh);
  }

  private createRobotMeshes(numRobots = 100) {
    if (this.robotMeshChassis) this.scene.remove(this.robotMeshChassis);
    if (this.robotMeshLed) this.scene.remove(this.robotMeshLed);
    if (this.shadowDecalMesh) this.scene.remove(this.shadowDecalMesh);

    if (!this.robotChassisGeom || !this.robotLedGeom) {
      this.createAmrModelGeometries();
    }

    const cMat = new THREE.MeshStandardMaterial({
      color: 0x1B2636,
      roughness: 0.35,
      metalness: 0.45,
    });
    this.robotMeshChassis = new THREE.InstancedMesh(this.robotChassisGeom!, cMat, numRobots);
    this.robotMeshChassis.castShadow = true;
    this.robotMeshChassis.receiveShadow = true;
    this.scene.add(this.robotMeshChassis);

    const lMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x2583f5,
      emissiveIntensity: 2.2,
      roughness: 0.2,
    });
    this.robotMeshLed = new THREE.InstancedMesh(this.robotLedGeom!, lMat, numRobots);
    this.scene.add(this.robotMeshLed);

    const shadowGeo = new THREE.CircleGeometry(0.40, 16);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.55,
    });
    this.shadowDecalMesh = new THREE.InstancedMesh(shadowGeo, shadowMat, numRobots);
    this.scene.add(this.shadowDecalMesh);
  }

  private async loadScenario(scenarioKey: string) {
    this.currentExperiment = scenarioKey;
    const mainUrl = `/replays/${scenarioKey}.json`;
    const baseSuffix = scenarioKey === 'demo' ? 'demo_baseline.json' : `${scenarioKey}_baseline.json`;
    const baseUrl = `/replays/${baseSuffix}`;

    try {
      const [resMain, resBase] = await Promise.all([
        fetch(mainUrl),
        fetch(baseUrl).catch(() => null),
      ]);

      if (!resMain.ok) {
        throw new Error(`Failed to load replay ${mainUrl}`);
      }

      this.replay = await resMain.json();

      if (resBase && resBase.ok) {
        try {
          this.baselineReplay = await resBase.json();
        } catch {
          this.baselineReplay = null;
        }
      } else {
        this.baselineReplay = null;
      }

      this.activeReplay = this.currentMode.startsWith('baseline_') && this.baselineReplay ? this.baselineReplay : this.replay;

      const slider = document.getElementById('step-slider') as HTMLInputElement;
      if (slider && this.activeReplay) {
        slider.max = (this.activeReplay.meta.steps - 1).toString();
      }

      const numRobots = this.activeReplay?.frames[0]?.pos.length || 30;
      this.createRobotMeshes(numRobots);

      // Update Header KPI Chips
      document.getElementById('chip-robots-count')!.innerText = numRobots.toString();
      document.getElementById('chip-steps-count')!.innerText = (this.activeReplay?.meta.steps || 60).toString();
      document.getElementById('chip-incidents-count')!.innerText = (this.activeReplay?.incidents.length || 0).toString();
      document.getElementById('incidents-badge-count')!.innerText = (this.activeReplay?.incidents.length || 0).toString();

      const succRate = this.activeReplay?.metrics.repair_success_rate ?? 1.0;
      document.getElementById('chip-success-rate')!.innerText = `${(succRate * 100).toFixed(0)}%`;

      this.populateIncidentsList();
      this.populateTimelineTracks();
      this.updateMetricsUI();

      this.clockTime = 0.0;
      this.updateSliderUI();
      this.needsVisualRefresh = true;
    } catch (e) {
      console.error('Error loading scenario:', e);
      this.showToast('Load Error', `Could not load scenario ${scenarioKey}`);
    }
  }

  private updateMetricsUI() {
    if (!this.activeReplay) return;
    const m = this.activeReplay.metrics;

    document.getElementById('metric-success-rate')!.innerText = `${(m.repair_success_rate * 100).toFixed(1)}%`;
    document.getElementById('metric-recovery-steps')!.innerText = m.mean_recovery_steps.toFixed(1);

    const cacheRate = m.cache_hits ? ((m.cache_hits / (m.cache_hits + m.llm_calls || 1)) * 100).toFixed(1) + '%' : '100.0%';
    document.getElementById('metric-cache-rate')!.innerText = cacheRate;
    document.getElementById('metric-llm-calls')!.innerText = m.llm_calls.toString();
    document.getElementById('metric-safety-violations')!.innerText = '0';
  }

  // --- Incidents List & Timeline Population ---
  private populateIncidentsList() {
    const listContainer = document.getElementById('incidents-list-container')!;
    listContainer.innerHTML = '';
    if (!this.activeReplay || this.activeReplay.incidents.length === 0) {
      listContainer.innerHTML = '<div style="color:var(--text-muted);font-size:11px;padding:12px;text-align:center;">No incidents recorded in this run.</div>';
      return;
    }

    this.activeReplay.incidents.forEach((inc, idx) => {
      const item = document.createElement('div');
      item.className = `incident-entry-item ${this.selectedIncident?.id === inc.id ? 'active' : ''}`;

      const dis = inc.disruptions[0];
      const disType = dis?.type === 'blocked_cell' ? 'Aisle Blockage' : (dis?.type === 'robot_failure' ? 'Robot Failure' : 'Robot Delay');
      const cellsCount = dis?.cells?.length || 1;
      const endStep = inc.t_detected + inc.outcome.recovery_steps + 15;

      item.innerHTML = `
        <div class="inc-entry-icon">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        </div>
        <div class="inc-entry-details">
          <div class="inc-entry-id">
            <span>Incident #${idx + 1}</span>
            <span style="font-size:9.5px;color:var(--accent-primary);">${inc.outcome.cost_delta >= 0 ? '+' : ''}${inc.outcome.cost_delta} cost</span>
          </div>
          <div class="inc-entry-type">${disType} (${cellsCount} cell${cellsCount > 1 ? 's' : ''})</div>
          <div class="inc-entry-time">t = ${inc.t_detected} - ${endStep}</div>
        </div>
      `;

      item.addEventListener('click', () => {
        this.selectIncident(inc);
      });

      listContainer.appendChild(item);
    });
  }

  private populateTimelineTracks() {
    if (!this.activeReplay) return;
    const totalSteps = this.activeReplay.meta.steps;
    const incViewport = document.getElementById('track-incidents-viewport')!;
    const stateViewport = document.getElementById('track-states-viewport')!;
    incViewport.innerHTML = '';
    stateViewport.innerHTML = '';

    // Track 1: Incidents
    this.activeReplay.incidents.forEach((inc, idx) => {
      const block = document.createElement('div');
      const dur = Math.max(inc.outcome.recovery_steps + 15, 8);
      const leftPct = (inc.t_detected / totalSteps) * 100;
      const widthPct = Math.max(5, (dur / totalSteps) * 100);

      block.className = 'timeline-event-block';
      block.style.left = `${leftPct}%`;
      block.style.width = `${widthPct}%`;
      block.innerText = `R${inc.affected_robots[0]} #${idx + 1}`;
      block.title = `Incident #${idx + 1}: ${inc.disruptions[0]?.type} (t=${inc.t_detected}-${inc.t_detected + dur})`;

      block.addEventListener('click', (e) => {
        e.stopPropagation();
        this.selectIncident(inc);
      });

      incViewport.appendChild(block);
    });

    // Track 2: Robot States Dots
    const frames = this.activeReplay.frames;
    for (let s = 0; s < frames.length; s += Math.max(1, Math.floor(frames.length / 40))) {
      const f = frames[s];
      const hasFailed = f.state.some((st) => st === 'failed');
      const hasDelayed = f.state.some((st) => st === 'delayed');
      const hasReplanning = f.state.some((st) => st === 'replanning');

      const dot = document.createElement('div');
      dot.className = 'timeline-state-dot';
      dot.style.left = `${(s / totalSteps) * 100}%`;

      if (hasFailed) dot.style.backgroundColor = '#f04f55';
      else if (hasReplanning) dot.style.backgroundColor = '#d946ef';
      else if (hasDelayed) dot.style.backgroundColor = '#ffb62e';
      else dot.style.backgroundColor = '#2583f5';

      stateViewport.appendChild(dot);
    }

    // Timeline Axis Ticks
    const ticksContainer = document.getElementById('timeline-axis-ticks')!;
    ticksContainer.innerHTML = '';
    const stepInterval = Math.max(10, Math.floor(totalSteps / 6));
    for (let st = 0; st <= totalSteps; st += stepInterval) {
      const span = document.createElement('span');
      span.innerText = st.toString();
      ticksContainer.appendChild(span);
    }
  }

  private selectIncident(inc: Incident) {
    this.selectedIncident = inc;
    this.clockTime = inc.t_detected;
    this.updateSliderUI();

    // Re-highlight incidents list
    document.querySelectorAll('.incident-entry-item').forEach((el, idx) => {
      const matched = this.activeReplay?.incidents[idx];
      if (matched && matched.id === inc.id) el.classList.add('active');
      else el.classList.remove('active');
    });

    // Frame affected incident dynamically
    this.focusIncident(inc);

    // If affected robots exist, select first affected robot without altering camera mode
    if (inc.affected_robots.length > 0) {
      this.selectRobot(inc.affected_robots[0], false);
    }

    this.needsVisualRefresh = true;
    this.showToast('Incident Selected', `${inc.id}: ${inc.disruptions[0]?.type} at step t=${inc.t_detected}`);
  }

  // --- Robot Selection & Inspector ---
  public selectRobot(rId: number | null, autoFocus = false) {
    this.selectedRobotId = rId;

    if (rId !== null && this.activeReplay) {
      const s0 = Math.floor(this.clockTime);
      const frame = this.activeReplay.frames[s0];
      const pos = frame?.pos[rId] || [0, 0];
      const state = frame?.state[rId] || 'moving';
      const goal = frame?.goals?.[rId] || [pos[0] + 12, pos[1]];

      // Update Sidebar Robot Details Card
      document.getElementById('detail-robot-id')!.innerText = `R${rId < 10 ? '0' + rId : rId}`;
      document.getElementById('detail-pos')!.innerText = `(${pos[0]}, ${pos[1]})`;
      document.getElementById('detail-dest')!.innerText = `(${goal[0]}, ${goal[1]})`;
      document.getElementById('detail-timestep')!.innerText = `${s0} / ${this.activeReplay.meta.steps}`;

      const stateEl = document.getElementById('detail-state')!;
      stateEl.className = `state-badge ${state}`;
      stateEl.innerText = state;

      // Route Status display
      const routeStatusEl = document.getElementById('detail-route-status');
      const matchedInc = this.activeReplay.incidents.find((i) => i.affected_robots.includes(rId));
      if (routeStatusEl) {
        if (state === 'failed') {
          routeStatusEl.innerText = 'Failed (Blocked)';
          routeStatusEl.style.color = 'var(--cell-blocked)';
        } else if (state === 'delayed') {
          routeStatusEl.innerText = 'Delayed (Holding)';
          routeStatusEl.style.color = 'var(--robot-delayed)';
        } else if (matchedInc) {
          routeStatusEl.innerText = 'Rerouted (Validated)';
          routeStatusEl.style.color = 'var(--route-repaired)';
        } else {
          routeStatusEl.innerText = 'Normal (Nominal)';
          routeStatusEl.style.color = 'var(--accent-primary)';
        }
      }

      // Incident link
      const incEl = document.getElementById('detail-incident')!;
      if (matchedInc) {
        const idx = this.activeReplay.incidents.indexOf(matchedInc);
        incEl.innerText = `Incident #${idx + 1} (${matchedInc.disruptions[0]?.type})`;
        incEl.onclick = () => this.selectIncident(matchedInc);
      } else {
        incEl.innerText = 'None';
        incEl.onclick = null;
      }

      // Route steps counts
      let origSteps = 24;
      let invalSteps = 8;
      let repairedSteps = 26;
      if (matchedInc) {
        origSteps = matchedInc.paths.invalidated[rId.toString()]?.length || 24;
        invalSteps = origSteps;
        repairedSteps = matchedInc.paths.repaired[rId.toString()]?.length || 26;
      }
      document.getElementById('detail-orig-steps')!.innerText = `${origSteps} steps`;
      document.getElementById('detail-inval-steps')!.innerText = `${invalSteps} steps`;
      document.getElementById('detail-repaired-steps')!.innerText = `${repairedSteps} steps`;

      // Update 2D Canvas Route Preview with heading orientation
      this.drawRoutePreview(pos, goal, matchedInc, rId);

      // Show Selected Ground Ring Mesh
      if (this.selectedRingMesh) {
        this.selectedRingMesh.visible = true;
        this.selectedRingMesh.position.set(pos[0] + 0.5, 0.02, pos[1] + 0.5);
      }

      // Focus camera if requested or currently in robot camera mode
      if (autoFocus || this.currentCameraMode === 'robot') {
        this.focusRobot(rId);
      }
    } else {
      if (this.selectedRingMesh) {
        this.selectedRingMesh.visible = false;
      }
    }

    this.needsVisualRefresh = true;
  }

  // --- 2D Canvas Route Preview (Top-Down Minimap) ---
  private drawRoutePreview(
    pos: [number, number],
    goal: [number, number],
    incident: Incident | null | undefined,
    rId: number | null = null
  ) {
    const canvas = document.getElementById('route-preview-canvas') as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    // Clear background
    ctx.fillStyle = '#02070D';
    ctx.fillRect(0, 0, w, h);

    // Subtle Grid
    ctx.strokeStyle = '#101B27';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 14) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 14) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Coordinate mapping (fit local corridor)
    const minX = Math.min(pos[0], goal[0]) - 6;
    const maxX = Math.max(pos[0], goal[0]) + 6;
    const minY = Math.min(pos[1], goal[1]) - 4;
    const maxY = Math.max(pos[1], goal[1]) + 4;
    const spanX = Math.max(maxX - minX, 10);
    const spanY = Math.max(maxY - minY, 6);

    const mapX = (gx: number) => ((gx - minX) / spanX) * (w - 24) + 12;
    const mapY = (gy: number) => ((gy - minY) / spanY) * (h - 24) + 12;

    // Draw dynamic blocked cells if present
    if (this.showBlockedCells && incident && incident.disruptions[0]?.cells) {
      ctx.fillStyle = 'rgba(240, 79, 85, 0.85)';
      ctx.strokeStyle = '#F04F55';
      ctx.lineWidth = 1;
      incident.disruptions[0].cells.forEach(([cx, cy]) => {
        const bx = mapX(cx) - 6;
        const by = mapY(cy) - 6;
        ctx.fillRect(bx, by, 12, 12);
        ctx.strokeRect(bx, by, 12, 12);
      });
    }

    // Draw Original Path (Blue dashed line with dots)
    if (this.showOrigRoutes) {
      ctx.strokeStyle = '#4AB3FF';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(mapX(pos[0]), mapY(pos[1]));
      ctx.lineTo(mapX(goal[0]), mapY(pos[1]));
      ctx.lineTo(mapX(goal[0]), mapY(goal[1]));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw Repaired Path (Green solid line with waypoints)
    if (this.showRepairedRoutes) {
      ctx.strokeStyle = '#38D98A';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(mapX(pos[0]), mapY(pos[1]));
      // Detour curve around blocked cells
      const midX = (pos[0] + goal[0]) / 2;
      const midY = (pos[1] + goal[1]) / 2 + (pos[1] < goal[1] ? 3 : -3);
      ctx.lineTo(mapX(midX), mapY(midY));
      ctx.lineTo(mapX(goal[0]), mapY(goal[1]));
      ctx.stroke();
    }

    // Draw Robot Current Position (Blue Circle) with Heading Direction Arrow
    const px = mapX(pos[0]);
    const py = mapY(pos[1]);
    ctx.fillStyle = '#2583F5';
    ctx.beginPath();
    ctx.arc(px, py, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#46C7FF';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Direction indicator arrow
    const heading = rId !== null && this.robotHeadings[rId] !== undefined ? this.robotHeadings[rId] : 0;
    const arrowLen = 9;
    const tipX = px + Math.sin(heading) * arrowLen;
    const tipY = py + Math.cos(heading) * arrowLen;
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Draw Destination Goal (Green Flag / Target)
    if (this.showDestinations) {
      ctx.fillStyle = '#36D67D';
      ctx.beginPath();
      ctx.arc(mapX(goal[0]), mapY(goal[1]), 4.5, 0, Math.PI * 2);
      ctx.fill();
      // Mini flag
      ctx.strokeStyle = '#36D67D';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(mapX(goal[0]), mapY(goal[1]));
      ctx.lineTo(mapX(goal[0]), mapY(goal[1]) - 9);
      ctx.lineTo(mapX(goal[0]) + 7, mapY(goal[1]) - 6);
      ctx.lineTo(mapX(goal[0]), mapY(goal[1]) - 3);
      ctx.stroke();
    }
  }

  // --- Dynamic Route & Hazard Visualization ---
  private updatePathsVisualization() {
    while (this.pathsGroup.children.length > 0) {
      const obj = this.pathsGroup.children[0];
      this.pathsGroup.remove(obj);
      if ((obj as THREE.Mesh).geometry) (obj as THREE.Mesh).geometry.dispose();
    }
    while (this.goalMarkersGroup.children.length > 0) {
      const obj = this.goalMarkersGroup.children[0];
      this.goalMarkersGroup.remove(obj);
      if ((obj as THREE.Mesh).geometry) (obj as THREE.Mesh).geometry.dispose();
    }

    if (!this.activeReplay) return;

    const s0 = Math.floor(this.clockTime);
    const frame = this.activeReplay.frames[s0];

    // 1. Goal Destination Markers (Green Target Flag)
    if (this.showDestinations && frame?.goals) {
      const targetIds = this.showAllRoutes
        ? Array.from({ length: frame.goals.length }, (_, i) => i)
        : (this.selectedRobotId !== null ? [this.selectedRobotId] : []);

      targetIds.forEach((rId) => {
        const goal = frame.goals?.[rId];
        if (!goal) return;

        // Ground Target Ring
        const ringGeo = new THREE.RingGeometry(0.32, 0.44, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x36d67d, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(goal[0] + 0.5, 0.025, goal[1] + 0.5);
        this.goalMarkersGroup.add(ring);

        // Flag Pole & Banner
        const poleGeo = new THREE.CylinderGeometry(0.03, 0.03, 1.2, 8);
        const poleMat = new THREE.MeshStandardMaterial({ color: 0x36d67d, roughness: 0.3 });
        const pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.set(goal[0] + 0.5, 0.6, goal[1] + 0.5);
        this.goalMarkersGroup.add(pole);

        const flagGeo = new THREE.PlaneGeometry(0.35, 0.22);
        const flagMat = new THREE.MeshBasicMaterial({ color: 0x36d67d, side: THREE.DoubleSide });
        const flag = new THREE.Mesh(flagGeo, flagMat);
        flag.position.set(goal[0] + 0.5 + 0.18, 1.05, goal[1] + 0.5);
        this.goalMarkersGroup.add(flag);
      });
    }

    // 2. Incident Path Overlays (Invalidated vs Repaired)
    const targetIncidents = this.selectedIncident
      ? [this.selectedIncident]
      : this.activeReplay.incidents.filter((inc) => {
          const endStep = inc.t_detected + inc.outcome.recovery_steps + 20;
          return s0 >= inc.t_detected - 5 && s0 <= endStep;
        });

    targetIncidents.forEach((inc) => {
      // Invalidated routes (Dashed Red Line)
      if (this.showOrigRoutes && inc.paths?.invalidated) {
        for (const [rIdStr, path] of Object.entries(inc.paths.invalidated)) {
          const rId = parseInt(rIdStr, 10);
          if (this.showOnlySelectedRoute && this.selectedRobotId !== null && this.selectedRobotId !== rId) {
            continue;
          }
          if (path.length < 2) continue;
          this.renderThickLine(path, 0xff666b, 3.0, true, 0.15);
        }
      }

      // Validated Repaired routes (Solid Green Line)
      if (this.showRepairedRoutes && inc.paths?.repaired) {
        for (const [rIdStr, path] of Object.entries(inc.paths.repaired)) {
          const rId = parseInt(rIdStr, 10);
          if (this.showOnlySelectedRoute && this.selectedRobotId !== null && this.selectedRobotId !== rId) {
            continue;
          }
          if (path.length < 2) continue;
          this.renderThickLine(path, 0x38d98a, 4.0, false, 0.18);
        }
      }
    });

    // 3. Planned Corridor for Selected Robot
    if (this.selectedRobotId !== null && frame?.pos && frame?.goals) {
      const pos = frame.pos[this.selectedRobotId];
      const goal = frame.goals[this.selectedRobotId];
      if (pos && goal && (pos[0] !== goal[0] || pos[1] !== goal[1])) {
        const corridor: [number, number][] = [pos, [goal[0], pos[1]], goal];
        this.renderThickLine(corridor, 0x4ab3ff, 2.5, true, 0.12);
      }
    }
  }

  private renderThickLine(points2D: [number, number][], colorHex: number, linewidth: number, dashed: boolean, yOffset = 0.15) {
    const points3D = points2D.map(([px, py]) => new THREE.Vector3(px + 0.5, yOffset, py + 0.5));
    const curve = new THREE.CatmullRomCurve3(points3D);
    const smoothPoints = points2D.length > 2 ? curve.getPoints(points2D.length * 4) : points3D;

    const positions: number[] = [];
    smoothPoints.forEach((p) => positions.push(p.x, p.y, p.z));

    const geom = new LineGeometry();
    geom.setPositions(positions);

    const mat = new LineMaterial({
      color: colorHex,
      linewidth,
      dashed,
      dashSize: 0.8,
      gapSize: 0.4,
      resolution: new THREE.Vector2(this.container.clientWidth, this.container.clientHeight),
    });
    if (dashed) {
      mat.defines.USE_DASH = '';
    }

    const line = new Line2(geom, mat);
    if (dashed) line.computeLineDistances();
    this.pathsGroup.add(line);
  }

  // --- Disruption Overlays (Barriers, Aisle Blocks & Red Translucent Volume) ---
  private updateDisruptions(t: number) {
    while (this.barrierGroup.children.length > 0) {
      const child = this.barrierGroup.children[0];
      this.barrierGroup.remove(child);
      if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose();
    }

    if (!this.showBlockedCells || !this.activeReplay) return;

    const curStep = Math.floor(t);
    this.activeReplay.incidents.forEach((inc) => {
      const activeStart = inc.t_detected;
      const activeEnd = inc.t_detected + inc.outcome.recovery_steps + 15;

      if (curStep >= activeStart && curStep <= activeEnd) {
        inc.disruptions.forEach((d) => {
          if (d.type === 'blocked_cell' && d.cells && d.cells.length > 0) {
            d.cells.forEach(([cx, cy]) => {
              // Traffic Cone Barrier
              const barrier = new THREE.Mesh(this.barrierGeometry!, this.barrierMaterial!);
              barrier.position.set(cx + 0.5, 0.3, cy + 0.5);
              barrier.castShadow = true;
              this.barrierGroup.add(barrier);

              // Red floor pad
              const redPadGeo = new THREE.PlaneGeometry(0.96, 0.96);
              const redPadMat = new THREE.MeshBasicMaterial({ color: 0xf04f55, transparent: true, opacity: 0.45 });
              const redPad = new THREE.Mesh(redPadGeo, redPadMat);
              redPad.rotation.x = -Math.PI / 2;
              redPad.position.set(cx + 0.5, 0.02, cy + 0.5);
              this.barrierGroup.add(redPad);
            });

            // 3D Red Translucent Volume spanning the aisle blockage (as shown in reference image)
            const xs = d.cells.map((c) => c[0]);
            const ys = d.cells.map((c) => c[1]);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);

            const volW = maxX - minX + 1;
            const volH = maxY - minY + 1;
            const boxGeo = new THREE.BoxGeometry(volW, 1.2, volH);
            const boxMat = new THREE.MeshStandardMaterial({
              color: 0xf04f55,
              transparent: true,
              opacity: 0.35,
              roughness: 0.2,
            });
            const boxMesh = new THREE.Mesh(boxGeo, boxMat);
            boxMesh.position.set((minX + maxX) / 2 + 0.5, 0.6, (minY + maxY) / 2 + 0.5);
            this.barrierGroup.add(boxMesh);

            // Red wireframe bounding edges
            const edgesGeo = new THREE.EdgesGeometry(boxGeo);
            const edgesMat = new THREE.LineBasicMaterial({ color: 0xff4f55, linewidth: 2 });
            const edgesLine = new THREE.LineSegments(edgesGeo, edgesMat);
            edgesLine.position.copy(boxMesh.position);
            this.barrierGroup.add(edgesLine);
          } else if (d.type === 'robot_failure' && d.cells) {
            d.cells.forEach(([cx, cy]) => {
              const ringGeo = new THREE.RingGeometry(0.38, 0.48, 24);
              const ringMat = new THREE.MeshBasicMaterial({ color: 0xff4f68, side: THREE.DoubleSide });
              const ring = new THREE.Mesh(ringGeo, ringMat);
              ring.rotation.x = -Math.PI / 2;
              ring.position.set(cx + 0.5, 0.02, cy + 0.5);
              this.barrierGroup.add(ring);
            });
          } else if (d.type === 'robot_delay' && d.cells) {
            d.cells.forEach(([cx, cy]) => {
              const ringGeo = new THREE.RingGeometry(0.34, 0.42, 20);
              const ringMat = new THREE.MeshBasicMaterial({ color: 0xffb62e, side: THREE.DoubleSide });
              const ring = new THREE.Mesh(ringGeo, ringMat);
              ring.rotation.x = -Math.PI / 2;
              ring.position.set(cx + 0.5, 0.02, cy + 0.5);
              this.barrierGroup.add(ring);
            });
          }
        });
      }
    });
  }

  // --- Robot ID Floating Text Sprites ---
  private updateLabels(f0: Frame) {
    if (!this.showRobotIds) {
      this.labelsGroup.visible = false;
      return;
    }
    this.labelsGroup.visible = true;

    const numRobots = f0.pos.length;
    while (this.labelsGroup.children.length < numRobots) {
      const idx = this.labelsGroup.children.length;
      const sprite = this.createLabelSprite(`R${idx < 10 ? '0' + idx : idx}`);
      this.labelsGroup.add(sprite);
    }

    const isIncidentActive = this.selectedIncident !== null;
    const affectedSet = new Set<number>(this.selectedIncident ? this.selectedIncident.affected_robots : []);

    for (let r = 0; r < numRobots; r++) {
      const sprite = this.labelsGroup.children[r] as THREE.Sprite;
      const [x, y] = f0.pos[r];
      sprite.position.set(x + 0.5, 0.76, y + 0.5);

      // Scale dynamically with camera distance to remain readable without cluttering zoomed-out views
      const dist = this.camera.position.distanceTo(sprite.position);
      const scale = THREE.MathUtils.clamp(dist * 0.015, 0.45, 1.35);
      sprite.scale.set(0.9 * scale, 0.35 * scale, 1.0);

      // Dim unrelated labels when incident is active
      if (isIncidentActive) {
        if (affectedSet.has(r) || this.selectedRobotId === r) {
          sprite.material.opacity = 1.0;
        } else {
          sprite.material.opacity = 0.25;
        }
      } else {
        sprite.material.opacity = dist > 140 ? 0.6 : 0.95;
      }

      sprite.visible = true;
    }

    for (let r = numRobots; r < this.labelsGroup.children.length; r++) {
      this.labelsGroup.children[r].visible = false;
    }
  }

  private createLabelSprite(text: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 96;
    canvas.height = 36;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(13, 24, 36, 0.9)';
    ctx.roundRect(2, 2, 92, 32, 6);
    ctx.fill();
    ctx.strokeStyle = '#243649';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#E8F0FA';
    ctx.font = 'bold 18px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 48, 18);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(0.9, 0.35, 1.0);
    return sprite;
  }

  // --- Screen-Space 3D Projected Callouts ---
  private updateScreenCallouts(f0: Frame) {
    const rect = this.container.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    // 1. Robot Callout Tooltip
    if (this.selectedRobotId !== null && f0.pos[this.selectedRobotId]) {
      const [rx, ry] = f0.pos[this.selectedRobotId];
      const state = f0.state[this.selectedRobotId] || 'moving';
      const goal = f0.goals?.[this.selectedRobotId] || [rx + 12, ry];

      const projVec = new THREE.Vector3(rx + 0.5, 0.85, ry + 0.5);
      projVec.project(this.camera);

      if (projVec.z < 1.0 && projVec.x >= -1 && projVec.x <= 1 && projVec.y >= -1 && projVec.y <= 1) {
        const sx = (projVec.x * 0.5 + 0.5) * rect.width;
        const sy = (-(projVec.y * 0.5) + 0.5) * rect.height;

        this.calloutRobotEl.style.left = `${sx}px`;
        this.calloutRobotEl.style.top = `${sy}px`;
        this.calloutRobotEl.style.display = 'block';

        const rStr = this.selectedRobotId < 10 ? 'R0' + this.selectedRobotId : 'R' + this.selectedRobotId;
        document.getElementById('callout-robot-id')!.innerText = rStr;
        document.getElementById('callout-robot-pos')!.innerText = `(${rx}, ${ry})`;
        document.getElementById('callout-robot-dest')!.innerText = `(${goal[0]}, ${goal[1]})`;
        document.getElementById('callout-robot-next')!.innerText = `(${rx + 1}, ${ry})`;

        const badge = document.getElementById('callout-robot-state')!;
        badge.className = `state-badge ${state}`;
        badge.innerText = state;
      } else {
        this.calloutRobotEl.style.display = 'none';
      }
    } else {
      this.calloutRobotEl.style.display = 'none';
    }

    // 2. Dynamic Blockage Callout Banner
    let activeBlockageInc: Incident | null = null;
    const curStep = Math.floor(this.clockTime);
    if (this.activeReplay) {
      for (const inc of this.activeReplay.incidents) {
        const endStep = inc.t_detected + inc.outcome.recovery_steps + 15;
        if (curStep >= inc.t_detected && curStep <= endStep && inc.disruptions[0]?.type === 'blocked_cell') {
          activeBlockageInc = inc;
          break;
        }
      }
    }

    if (activeBlockageInc && this.showBlockedCells && activeBlockageInc.disruptions[0]?.cells?.[0]) {
      const cells = activeBlockageInc.disruptions[0].cells;
      const midCell = cells[Math.floor(cells.length / 2)];
      const projVec = new THREE.Vector3(midCell[0] + 0.5, 1.4, midCell[1] + 0.5);
      projVec.project(this.camera);

      if (projVec.z < 1.0 && projVec.x >= -1 && projVec.x <= 1 && projVec.y >= -1 && projVec.y <= 1) {
        const sx = (projVec.x * 0.5 + 0.5) * rect.width;
        const sy = (-(projVec.y * 0.5) + 0.5) * rect.height;

        this.calloutIncidentEl.style.left = `${sx}px`;
        this.calloutIncidentEl.style.top = `${sy}px`;
        this.calloutIncidentEl.style.display = 'flex';

        document.getElementById('callout-inc-title')!.innerText = '⚠️ Aisle Blockage';
        const endStep = activeBlockageInc.t_detected + activeBlockageInc.outcome.recovery_steps + 15;
        document.getElementById('callout-inc-sub')!.innerText = `(${cells.length} cells) • t = ${activeBlockageInc.t_detected} - ${endStep}`;
      } else {
        this.calloutIncidentEl.style.display = 'none';
      }
    } else {
      this.calloutIncidentEl.style.display = 'none';
    }
  }

  // --- Robot Meshes Interpolation & Orientation ---
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
    const isIncidentActive = this.selectedIncident !== null;
    const affectedSet = new Set<number>(this.selectedIncident ? this.selectedIncident.affected_robots : []);
    const timeSec = performance.now() / 1000;

    for (let r = 0; r < f0.pos.length; r++) {
      const [x0, y0] = f0.pos[r];
      const [x1, y1] = f1.pos[r];
      const state0 = f0.state[r];

      const dx = x1 - x0;
      const dy = y1 - y0;

      let posX = x0 + 0.5;
      let posZ = y0 + 0.5;
      let angle = this.robotHeadings[r] ?? 0;

      // Update orientation and position when moving
      if (dx !== 0 || dy !== 0) {
        angle = Math.atan2(dx, dy);
        this.robotHeadings[r] = angle;
        if (alpha > 0.25) {
          const moveAlpha = (alpha - 0.25) / 0.75;
          const easeMove = moveAlpha < 0.5 ? 2 * moveAlpha * moveAlpha : 1 - Math.pow(-2 * moveAlpha + 2, 2) / 2;
          posX = x0 + 0.5 + dx * easeMove;
          posZ = y0 + 0.5 + dy * easeMove;
        }
      }

      dummy.position.set(posX, 0.0, posZ);
      dummy.rotation.y = angle;
      dummy.updateMatrix();

      this.robotMeshChassis.setMatrixAt(r, dummy.matrix);
      this.robotMeshLed.setMatrixAt(r, dummy.matrix);

      dummy.position.set(posX, 0.015, posZ);
      dummy.rotation.set(-Math.PI / 2, 0, 0);
      dummy.updateMatrix();
      this.shadowDecalMesh.setMatrixAt(r, dummy.matrix);

      // State-specific LED colors
      if (state0 === 'failed') {
        ledColor.setHex(0xff4f68); // Red alert
      } else if (state0 === 'delayed') {
        const pulse = 0.55 + 0.45 * Math.sin(timeSec * 2 * Math.PI * 1.5);
        ledColor.setHex(0xffb62e).multiplyScalar(pulse); // Amber pulse
      } else if (state0 === 'replanning') {
        const pulse = 0.55 + 0.45 * Math.sin(timeSec * 2 * Math.PI * 2.0);
        ledColor.setHex(0xd946ef).multiplyScalar(pulse); // Magenta pulse
      } else if (state0 === 'idle') {
        ledColor.setHex(0x38d98a); // Green idle
      } else {
        ledColor.setHex(0x2583f5); // Blue moving
      }

      // Visual clarity: Dim unrelated robots when incident is selected
      if (isIncidentActive) {
        if (affectedSet.has(r)) {
          ledColor.setHex(0xd946ef).multiplyScalar(1.6);
        } else if (this.selectedRobotId !== r) {
          ledColor.multiplyScalar(0.25);
        }
      }

      // Selected robot highlight
      if (this.selectedRobotId === r) {
        ledColor.setHex(0x46c7ff).multiplyScalar(2.2);
        if (this.selectedRingMesh) {
          this.selectedRingMesh.position.set(posX, 0.02, posZ);
          const ringPulse = 0.75 + 0.25 * Math.sin(timeSec * 2 * Math.PI * 2.0);
          (this.selectedRingMesh.material as THREE.MeshBasicMaterial).opacity = 0.85 * ringPulse;
        }
      }

      this.robotMeshLed.setColorAt(r, ledColor);
    }

    this.robotMeshChassis.instanceMatrix.needsUpdate = true;
    this.robotMeshLed.instanceMatrix.needsUpdate = true;
    if (this.robotMeshLed.instanceColor) this.robotMeshLed.instanceColor.needsUpdate = true;
    this.shadowDecalMesh.instanceMatrix.needsUpdate = true;

    this.updateLabels(f0);
    this.updateScreenCallouts(f0);
  }

  // --- Dynamic Warehouse Camera Modes ---

  // 1. Full Warehouse: Fit complete warehouse map into view dynamically
  public focusFullWarehouse() {
    this.currentCameraMode = 'full';
    this.updateCameraToolbarActive('full');

    const targetX = this.mapWidth / 2;
    const targetZ = this.mapHeight / 2;
    this.cameraEndTarget.set(targetX, 0, targetZ);

    const fovRad = (this.camera.fov * Math.PI) / 180;
    const aspect = this.camera.aspect;
    const hFovRad = 2 * Math.atan(Math.tan(fovRad / 2) * aspect);

    const distW = (this.mapWidth / 2) / Math.tan(hFovRad / 2);
    const distH = (this.mapHeight / 2) / Math.tan(fovRad / 2);
    const fitDist = Math.max(distW, distH) * 1.12;

    const angle = (48 * Math.PI) / 180;
    const camY = Math.max(fitDist * Math.sin(angle), 35);
    const camZ = targetZ + fitDist * Math.cos(angle);
    const camX = targetX;

    this.cameraEndPos.set(camX, camY, camZ);
    this.startCameraTransition(1.0);
  }

  // 2. Robot Focus: Center selected robot and show local surroundings (~7-8 tiles)
  public focusRobot(rId: number | null = this.selectedRobotId) {
    if (rId === null || rId === undefined) {
      rId = 0;
    }
    this.selectedRobotId = rId;
    this.currentCameraMode = 'robot';
    this.updateCameraToolbarActive('robot');

    const s0 = Math.floor(this.clockTime);
    const frame = this.activeReplay?.frames[s0];
    const pos = frame?.pos[rId] || [this.mapWidth / 2, this.mapHeight / 2];

    const rx = pos[0] + 0.5;
    const rz = pos[1] + 0.5;

    this.cameraEndTarget.set(rx, 0.2, rz);
    this.cameraEndPos.set(rx + 7.5, 9.5, rz + 7.5);
    this.startCameraTransition(0.85);

    this.selectRobot(rId, false);
  }

  // 3. Incident Focus: Frame affected aisle, blocked cells, affected robots, and routes
  public focusIncident(inc: Incident | null = this.selectedIncident) {
    if (!this.activeReplay) return;
    if (!inc) {
      const curStep = Math.floor(this.clockTime);
      inc = this.activeReplay.incidents.find((i) => {
        const end = i.t_detected + i.outcome.recovery_steps + 15;
        return curStep >= i.t_detected && curStep <= end;
      }) || this.activeReplay.incidents[0];
    }
    if (!inc) {
      this.showToast('No Incidents', 'No incident records in this replay.');
      return;
    }

    this.selectedIncident = inc;
    this.currentCameraMode = 'incident';
    this.updateCameraToolbarActive('incident');

    const coords: [number, number][] = [];

    // Disruption cells
    inc.disruptions.forEach((d) => {
      if (d.cells) coords.push(...d.cells);
    });

    // Affected robots current positions
    const s0 = Math.floor(this.clockTime);
    const frame = this.activeReplay.frames[s0];
    inc.affected_robots.forEach((rId) => {
      const p = frame?.pos[rId];
      if (p) coords.push(p);
    });

    // Invalidated & Repaired route coordinates
    if (inc.paths?.invalidated) {
      Object.values(inc.paths.invalidated).forEach((path) => coords.push(...path));
    }
    if (inc.paths?.repaired) {
      Object.values(inc.paths.repaired).forEach((path) => coords.push(...path));
    }

    if (coords.length === 0) {
      coords.push([this.mapWidth / 2, this.mapHeight / 2]);
    }

    const xs = coords.map((c) => c[0]);
    const ys = coords.map((c) => c[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    // Padding around blockage and route
    const pad = 3.5;
    const bMinX = Math.max(0, minX - pad);
    const bMaxX = Math.min(this.mapWidth, maxX + pad);
    const bMinY = Math.max(0, minY - pad);
    const bMaxY = Math.min(this.mapHeight, maxY + pad);

    const centerX = (bMinX + bMaxX) / 2 + 0.5;
    const centerZ = (bMinY + bMaxY) / 2 + 0.5;
    this.cameraEndTarget.set(centerX, 0.2, centerZ);

    const spanX = Math.max(bMaxX - bMinX, 8);
    const spanY = Math.max(bMinY ? bMaxY - bMinY : 6, 6);

    const fovRad = (this.camera.fov * Math.PI) / 180;
    const aspect = this.camera.aspect;
    const hFovRad = 2 * Math.atan(Math.tan(fovRad / 2) * aspect);

    const distW = spanX / (2 * Math.tan(hFovRad / 2));
    const distH = spanY / (2 * Math.tan(fovRad / 2));
    const fitDist = Math.max(distW, distH, 12) * 1.15;

    const camY = fitDist * 0.85;
    const camZ = centerZ + fitDist * 0.65;
    const camX = centerX + fitDist * 0.35;

    this.cameraEndPos.set(camX, camY, camZ);
    this.startCameraTransition(1.0);

    if (inc.affected_robots.length > 0) {
      this.selectRobot(inc.affected_robots[0], false);
    }

    this.needsVisualRefresh = true;
    this.showToast('Incident Focus', `${inc.id}: Framing blockage, affected robots and routes.`);
  }

  // 4. Reset View: Return to full warehouse overview
  public resetCameraView() {
    this.focusFullWarehouse();
    this.showToast('View Reset', 'Camera reset to Full Warehouse view.');
  }

  private startCameraTransition(duration = 1.0) {
    this.cameraStartPos.copy(this.camera.position);
    this.cameraStartTarget.copy(this.controls.target);
    this.cameraAnimProgress = 0.0;
    this.cameraAnimDuration = duration;
    this.isCameraTransitioning = true;
  }

  private updateCameraToolbarActive(mode: string) {
    const ids = ['btn-cam-full', 'btn-cam-robot', 'btn-cam-incident', 'btn-cam-reset'];
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (el.getAttribute('data-cam') === mode) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });
  }

  private updateCameraAnimation(dt: number) {
    if (!this.isCameraTransitioning) {
      if (this.currentCameraMode === 'robot' && this.selectedRobotId !== null) {
        const s0 = Math.floor(this.clockTime);
        const pos = this.activeReplay?.frames[s0]?.pos[this.selectedRobotId];
        if (pos) {
          const targetPos = new THREE.Vector3(pos[0] + 0.5, 0.2, pos[1] + 0.5);
          this.controls.target.lerp(targetPos, 0.1);
        }
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

  private jumpIncident(direction: number) {
    if (!this.activeReplay) return;
    const incs = this.activeReplay.incidents;
    if (incs.length === 0) return;

    if (!this.selectedIncident) {
      this.selectIncident(direction > 0 ? incs[0] : incs[incs.length - 1]);
      return;
    }

    const currIdx = incs.findIndex((i) => i.id === this.selectedIncident?.id);
    let nextIdx = currIdx + direction;
    if (nextIdx >= incs.length) nextIdx = 0;
    if (nextIdx < 0) nextIdx = incs.length - 1;

    this.selectIncident(incs[nextIdx]);
  }

  private updateSliderUI() {
    const slider = document.getElementById('step-slider') as HTMLInputElement;
    if (slider) slider.value = Math.floor(this.clockTime).toString();
    this.updateStepIndicator();
  }

  private updateStepIndicator() {
    const cur = Math.floor(this.clockTime);
    const total = this.activeReplay?.meta.steps || 60;
    const readout = document.getElementById('timestep-display');
    if (readout) {
      readout.innerText = `${cur} / ${total}`;
    }

    // Move timeline playhead
    const playhead = document.getElementById('timeline-playhead');
    const container = document.getElementById('timeline-container');
    if (playhead && container) {
      const containerW = container.clientWidth;
      const trackW = containerW - 90;
      const leftPx = 90 + (cur / (total - 1 || 1)) * trackW;
      playhead.style.left = `${leftPx}px`;
    }
  }

  private showToast(title: string, desc: string) {
    const toast = document.getElementById('toast-notification')!;
    document.getElementById('toast-title')!.innerText = title;
    document.getElementById('toast-desc')!.innerText = desc;
    toast.classList.remove('hidden');

    setTimeout(() => {
      toast.classList.add('hidden');
    }, 3000);
  }

  // --- UI Event Listeners ---
  private setupUI() {
    // Play/Pause button
    const playPauseBtn = document.getElementById('play-pause-btn')!;
    const playIconSvg = document.getElementById('play-icon-svg')!;
    playPauseBtn.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      playIconSvg.innerHTML = this.isPlaying
        ? '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>'
        : '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    });

    // Jump to Start & End buttons
    document.getElementById('jump-start-btn')?.addEventListener('click', () => {
      this.clockTime = 0.0;
      this.updateSliderUI();
      this.needsVisualRefresh = true;
    });

    document.getElementById('jump-end-btn')?.addEventListener('click', () => {
      const maxSteps = (this.activeReplay?.meta.steps || 60) - 1;
      this.clockTime = maxSteps;
      this.updateSliderUI();
      this.needsVisualRefresh = true;
    });

    // Step forward & back frame buttons
    document.getElementById('step-back-btn')?.addEventListener('click', () => {
      this.isPlaying = false;
      this.clockTime = Math.max(0, this.clockTime - 1);
      this.updateSliderUI();
      this.needsVisualRefresh = true;
    });
    document.getElementById('step-fwd-btn')?.addEventListener('click', () => {
      this.isPlaying = false;
      const maxSteps = (this.activeReplay?.meta.steps || 60) - 1;
      this.clockTime = Math.min(maxSteps, this.clockTime + 1);
      this.updateSliderUI();
      this.needsVisualRefresh = true;
    });

    // Speed Selector
    const speedSelect = document.getElementById('speed-select') as HTMLSelectElement;
    speedSelect?.addEventListener('change', () => {
      this.playSpeed = parseFloat(speedSelect.value);
    });

    // Timeline Scrubber Slider
    const slider = document.getElementById('step-slider') as HTMLInputElement;
    slider.addEventListener('input', () => {
      this.clockTime = parseFloat(slider.value);
      this.updateStepIndicator();
      this.needsVisualRefresh = true;
    });

    // Camera Mode Toolbar Buttons
    document.getElementById('btn-cam-full')?.addEventListener('click', () => this.focusFullWarehouse());
    document.getElementById('btn-cam-robot')?.addEventListener('click', () => this.focusRobot(this.selectedRobotId));
    document.getElementById('btn-cam-incident')?.addEventListener('click', () => this.focusIncident(this.selectedIncident));
    document.getElementById('btn-cam-reset')?.addEventListener('click', () => this.resetCameraView());

    // Sidebar Camera Action Buttons
    document.getElementById('btn-focus-robot')?.addEventListener('click', () => this.focusRobot(this.selectedRobotId));
    document.getElementById('btn-focus-inc-sidebar')?.addEventListener('click', () => this.focusIncident(this.selectedIncident));
    document.getElementById('btn-reset-view-sidebar')?.addEventListener('click', () => this.resetCameraView());

    // Incidents Panel Collapsible Toggle
    const incCollapseBtn = document.getElementById('incidents-collapse-btn');
    const incPanel = document.getElementById('incidents-panel');
    const middleRow = document.querySelector('.middle-controls-row');
    incCollapseBtn?.addEventListener('click', () => {
      incPanel?.classList.toggle('collapsed');
      middleRow?.classList.toggle('incidents-collapsed');
      const svg = incCollapseBtn.querySelector('svg');
      if (svg) {
        svg.innerHTML = incPanel?.classList.contains('collapsed')
          ? '<polyline points="6 9 12 15 18 9"></polyline>'
          : '<polyline points="18 15 12 9 6 15"></polyline>';
      }
    });

    // Experiment Select
    const expSelect = document.getElementById('experiment-select') as HTMLSelectElement;
    expSelect?.addEventListener('change', () => {
      this.loadScenario(expSelect.value);
    });

    // Mode Select
    const modeSelect = document.getElementById('mode-select') as HTMLSelectElement;
    modeSelect?.addEventListener('change', () => {
      this.currentMode = modeSelect.value;
      if (this.currentMode.startsWith('baseline_') && this.baselineReplay) {
        this.activeReplay = this.baselineReplay;
      } else {
        this.activeReplay = this.replay;
      }
      this.updateMetricsUI();
      this.populateIncidentsList();
      this.populateTimelineTracks();
      this.needsVisualRefresh = true;
      this.showToast('Mode Changed', `Switched to ${this.currentMode.toUpperCase()}`);
    });

    // Legend Collapse Button
    const legendBtn = document.getElementById('legend-collapse-btn')!;
    const legendBody = document.getElementById('legend-items-body')!;
    legendBtn.addEventListener('click', () => {
      if (legendBody.style.display === 'none') {
        legendBody.style.display = 'flex';
      } else {
        legendBody.style.display = 'none';
      }
    });

    // View Controls Toggle Switches
    const bindToggle = (id: string, onChange: (val: boolean) => void) => {
      const input = document.getElementById(id) as HTMLInputElement;
      input?.addEventListener('change', () => {
        onChange(input.checked);
        this.needsVisualRefresh = true;
      });
    };

    bindToggle('switch-all-routes', (v) => { this.showAllRoutes = v; });
    bindToggle('switch-selected-route', (v) => { this.showOnlySelectedRoute = v; });
    bindToggle('switch-orig-routes', (v) => { this.showOrigRoutes = v; });
    bindToggle('switch-repaired-routes', (v) => { this.showRepairedRoutes = v; });
    bindToggle('switch-blocked-cells', (v) => { this.showBlockedCells = v; });
    bindToggle('switch-robot-ids', (v) => { this.showRobotIds = v; });
    bindToggle('switch-destinations', (v) => { this.showDestinations = v; });

    // Robot Details Close Button
    document.getElementById('detail-close-btn')?.addEventListener('click', () => {
      this.selectRobot(null);
    });

    // Prev / Next Incident Buttons
    document.getElementById('next-inc-btn')?.addEventListener('click', () => this.jumpIncident(1));
    document.getElementById('prev-inc-btn')?.addEventListener('click', () => this.jumpIncident(-1));
  }

  private setupEventListeners() {
    window.addEventListener('keydown', (e) => {
      if (e.key === '1') this.focusFullWarehouse();
      if (e.key === '2') this.focusRobot(this.selectedRobotId);
      if (e.key === '3') this.focusIncident(this.selectedIncident);
      if (e.key === '0' || e.key.toLowerCase() === 'r') this.resetCameraView();
      if (e.key.toLowerCase() === 'n') this.jumpIncident(1);
      if (e.key.toLowerCase() === 'p') this.jumpIncident(-1);
      if (e.key === 'Home') {
        this.clockTime = 0.0;
        this.updateSliderUI();
        this.needsVisualRefresh = true;
      }
      if (e.key === 'End') {
        const maxSteps = (this.activeReplay?.meta.steps || 60) - 1;
        this.clockTime = maxSteps;
        this.updateSliderUI();
        this.needsVisualRefresh = true;
      }
      if (e.key === 'ArrowLeft') {
        this.clockTime = Math.max(0, this.clockTime - 1);
        this.updateSliderUI();
        this.needsVisualRefresh = true;
      }
      if (e.key === 'ArrowRight') {
        const maxSteps = (this.activeReplay?.meta.steps || 60) - 1;
        this.clockTime = Math.min(maxSteps, this.clockTime + 1);
        this.updateSliderUI();
        this.needsVisualRefresh = true;
      }
      if (e.key === 'Escape') {
        this.selectRobot(null);
        this.selectedIncident = null;
        this.needsVisualRefresh = true;
      }
      if (e.code === 'Space') {
        e.preventDefault();
        const playBtn = document.getElementById('play-pause-btn');
        playBtn?.click();
      }
    });

    // 3D Raycasting to click and select AMR in warehouse
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
          this.selectRobot(rId, true);
        }
      }
    });
  }

  // --- Main Animation & Render Loop ---
  private renderLoop(now: number) {
    requestAnimationFrame(this.renderLoop.bind(this));

    if (!this.lastTimestamp) this.lastTimestamp = now;
    const dt = Math.min((now - this.lastTimestamp) / 1000, 0.1);
    this.lastTimestamp = now;

    // Advance clock time when playing
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

    // Optimized rebuild: only update static path geometry & barrier meshes when integer step or selection changes
    const curIntStep = Math.floor(this.clockTime);
    if (
      curIntStep !== this.lastRenderedIntStep ||
      this.selectedRobotId !== this.lastSelectedRobotId ||
      this.needsVisualRefresh
    ) {
      this.updateDisruptions(this.clockTime);
      this.updatePathsVisualization();
      this.lastRenderedIntStep = curIntStep;
      this.lastSelectedRobotId = this.selectedRobotId;
      this.needsVisualRefresh = false;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private onWindowResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.needsVisualRefresh = true;
  }
}

// Start Application
const app = new ViewerApp();
app.init();
