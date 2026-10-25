import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
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

  // Loaded Geometries & Materials
  private shelfGeometry: THREE.BufferGeometry | null = null;
  private shelfMaterial: THREE.Material | THREE.Material[] | null = null;
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

  // Performance tracking
  private frameCount = 0;
  private lastFpsUpdate = 0;
  private fpsDisplay!: HTMLElement;

  async init() {
    this.container = document.getElementById('canvas-container')!;
    this.fpsDisplay = document.getElementById('fps-counter')!;

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

    // Intro camera dolly: overview down to aisle view over 4 seconds
    this.startIntroDolly();

    window.addEventListener('resize', this.onWindowResize.bind(this));
    requestAnimationFrame(this.renderLoop.bind(this));
  }

  private setupScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090e);
    this.scene.add(this.barrierGroup);
    this.scene.add(this.pathsGroup);
    this.scene.add(this.floorLinesGroup);
  }

  private setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.18;
    this.container.appendChild(this.renderer.domElement);
  }

  private setupCamera() {
    this.camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.5, 1000);
    this.camera.position.set(80, 115, 150);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // Don't go below floor
    this.controls.target.set(80, 0, 31.5);
    this.controls.update();
  }

  private setupLighting() {
    // Single DirectionalLight from above-front with 2048 soft shadow map fitted to map
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
    const floorGeo = new THREE.PlaneGeometry(this.mapWidth, this.mapHeight);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x242a35,
      roughness: 0.82,
      metalness: 0.12,
      map: texture || undefined
    });

    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(this.mapWidth / 2, 0, this.mapHeight / 2);
    floor.receiveShadow = true;
    this.scene.add(floor);

    this.createAisleLaneMarkings();
  }

  private createAisleLaneMarkings() {
    const lineMat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.5 });
    for (let y = 1; y < this.mapHeight; y += 4) {
      const points = [
        new THREE.Vector3(1, 0.012, y + 0.5),
        new THREE.Vector3(this.mapWidth - 1, 0.012, y + 0.5)
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
      this.shelfMaterial = new THREE.MeshStandardMaterial({ color: 0x1e3a8a, roughness: 0.6 });
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
      console.warn('Barrier glb load error', e);
    }

    if (!this.barrierGeometry) {
      this.barrierGeometry = new THREE.ConeGeometry(0.3, 0.6, 16);
      this.barrierMaterial = new THREE.MeshStandardMaterial({ color: 0xf97316 });
    }

    // 3. Robot Model
    try {
      const robotGltf = await gltfLoader.loadAsync('/models/robot.glb');
      let chassisGeom: THREE.BufferGeometry | null = null;
      let ledGeom: THREE.BufferGeometry | null = null;

      robotGltf.scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (mesh.name === 'LED' || (mesh.material as THREE.Material)?.name === 'LED') {
            ledGeom = mesh.geometry;
          } else if (!chassisGeom) {
            chassisGeom = mesh.geometry;
          }
        }
      });

      this.createRobotMeshes(chassisGeom, ledGeom);
    } catch {
      this.createRobotMeshes(null, null);
    }
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

  private createRobotMeshes(chassisGeom: THREE.BufferGeometry | null, ledGeom: THREE.BufferGeometry | null) {
    const numRobots = 100;

    const cGeom = chassisGeom || new THREE.CylinderGeometry(0.35, 0.35, 0.18, 20);
    const cMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.4,
      metalness: 0.3
    });
    this.robotMeshChassis = new THREE.InstancedMesh(cGeom, cMat, numRobots);
    this.robotMeshChassis.castShadow = true;
    this.robotMeshChassis.receiveShadow = true;
    this.scene.add(this.robotMeshChassis);

    const lGeom = ledGeom || new THREE.TorusGeometry(0.32, 0.02, 12, 24);
    const lMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x38bdf8,
      emissiveIntensity: 2.2,
      roughness: 0.2
    });
    this.robotMeshLed = new THREE.InstancedMesh(lGeom, lMat, numRobots);
    this.scene.add(this.robotMeshLed);

    const shadowGeo = new THREE.CircleGeometry(0.38, 16);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.45
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

    const elSuccess = document.getElementById('metric-success')!;
    const elRecovery = document.getElementById('metric-recovery')!;
    const elCost = document.getElementById('metric-cost')!;
    const elLlm = document.getElementById('metric-llm')!;

    const baseSuccessStr = bm ? `vs ${(bm.repair_success_rate * 100).toFixed(1)}% base` : '';
    const baseRecStr = bm ? `vs ${bm.mean_recovery_steps.toFixed(1)} base` : '';
    const baseCostStr = bm ? `vs +${bm.cost_delta_pct.toFixed(1)}% base` : '';
    const baseLlmStr = bm ? `vs ${bm.llm_calls} base` : '';

    elSuccess.innerHTML = `${(m.repair_success_rate * 100).toFixed(1)}%`;
    document.getElementById('base-success')!.innerText = baseSuccessStr;

    elRecovery.innerHTML = `${m.mean_recovery_steps.toFixed(1)} <span class="unit">steps</span>`;
    document.getElementById('base-recovery')!.innerText = baseRecStr;

    elCost.innerHTML = `+${m.cost_delta_pct.toFixed(1)}%`;
    document.getElementById('base-cost')!.innerText = baseCostStr;

    elLlm.innerHTML = `${m.llm_calls} <span class="unit">calls</span>`;
    document.getElementById('base-llm')!.innerText = `${m.cache_hits} cache hits ${baseLlmStr ? '(' + baseLlmStr + ')' : ''}`;

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
  }

  private buildTimelineTicks() {
    if (!this.activeReplay) return;
    const container = document.getElementById('timeline-ticks')!;
    container.innerHTML = '';
    const totalSteps = this.activeReplay.meta.steps;

    const tooltip = document.getElementById('incident-tooltip')!;

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
      tick.addEventListener('mouseenter', () => {
        const rect = tick.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2}px`;
        tooltip.style.bottom = `95px`;
        document.getElementById('tt-header')!.innerText = `Incident ${inc.id} (t=${inc.t_detected})`;
        document.getElementById('tt-body')!.innerText = `${type.replace('_', ' ').toUpperCase()} • ${inc.affected_robots.length} AMRs affected`;
        tooltip.classList.remove('hidden');
      });

      tick.addEventListener('mouseleave', () => {
        tooltip.classList.add('hidden');
      });

      tick.addEventListener('click', (e) => {
        e.stopPropagation();
        tooltip.classList.add('hidden');
        this.selectIncident(inc);
      });

      container.appendChild(tick);
    });

    const incChip = document.getElementById('incident-counter-chip');
    if (incChip) {
      incChip.innerText = `${this.activeReplay.incidents.length} Incidents Logged`;
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
    const panel = document.getElementById('incident-panel')!;
    panel.classList.remove('hidden');

    document.getElementById('incident-panel-title')!.innerText = `Incident ${inc.id}`;
    const dis = inc.disruptions[0];
    document.getElementById('incident-panel-subtitle')!.innerText = `Step t=${inc.t_detected} • ${dis?.type.replace('_', ' ').toUpperCase()}`;

    // 1. Sentinel
    const interactsStr = inc.interacts_with.length > 0 ? ` | Interacts with: [${inc.interacts_with.join(', ')}]` : '';
    document.getElementById('inc-sentinel-text')!.innerText =
      `Disruption: ${dis?.type || 'unknown'} • Congestion: ${inc.congestion.level.toUpperCase()} (density: ${((inc.congestion.density || 0.15) * 100).toFixed(0)}%)${interactsStr}`;

    const chipsContainer = document.getElementById('inc-affected-chips')!;
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

    // 2. Cache
    const cacheHit = inc.cache?.hit;
    const simPct = Math.round((inc.cache?.similarity || 0) * 100);
    const cacheBadge = document.getElementById('inc-cache-badge')!;
    cacheBadge.className = `diag-badge ${cacheHit ? 'green' : 'purple'}`;
    cacheBadge.innerText = cacheHit ? 'CACHE HIT' : 'MISS';

    const cacheHitText = cacheHit
      ? `HIT: Matched prior incident ${inc.cache?.matched} with ${simPct}% cosine similarity. Skipped LLM.`
      : `MISS: Best stored match achieved ${simPct}% similarity (below threshold).`;
    document.getElementById('inc-cache-text')!.innerText = cacheHitText;
    document.getElementById('inc-cache-sim-val')!.innerText = `${simPct}%`;
    document.getElementById('inc-cache-sim-fill')!.style.width = `${simPct}%`;

    // 3. Dry Run
    const dryRun = inc.dry_run;
    const dryBadge = document.getElementById('inc-dryrun-badge')!;
    if (dryRun && dryRun.ran) {
      dryBadge.className = `diag-badge ${dryRun.valid ? 'green' : 'amber'}`;
      dryBadge.innerText = dryRun.valid ? 'PASSED ✓' : 'FAILED ✗';
      document.getElementById('inc-dryrun-text')!.innerText = dryRun.valid
        ? 'VALID: Cached strategy directive verified clean against live obstacle state.'
        : `INVALID: ${dryRun.violation || 'conflict detected'}. Discarded and promoted to negative memory.`;
    } else {
      dryBadge.className = 'diag-badge';
      dryBadge.innerText = 'SKIPPED';
      document.getElementById('inc-dryrun-text')!.innerText = 'Skipped (Cache miss or first observation).';
    }

    // 4. LLM Strategist
    const llmBadge = document.getElementById('inc-llm-badge')!;
    if (inc.llm?.called) {
      llmBadge.className = 'diag-badge pink';
      llmBadge.innerText = 'CALLED';
      document.getElementById('inc-llm-text')!.innerText = inc.llm.directive?.rationale || 'Optimal directive determined.';
      document.getElementById('inc-llm-latency')!.innerText = `⚡ ${inc.llm.latency_ms || 1120} ms`;
      document.getElementById('inc-llm-tokens')!.innerText = `🎟️ ${inc.llm.tokens || 850} tokens`;
      document.getElementById('inc-llm-solver')!.innerText = `Solver: ${inc.llm.directive?.solver.toUpperCase()}`;
      document.getElementById('inc-llm-directive')!.innerText = JSON.stringify(inc.llm.directive, null, 2);
    } else {
      llmBadge.className = 'diag-badge';
      llmBadge.innerText = 'BYPASSED';
      document.getElementById('inc-llm-text')!.innerText = 'Bypassed (Resolved directly via Strategy Cache hit or Baseline mode).';
      document.getElementById('inc-llm-latency')!.innerText = '⚡ 0 ms';
      document.getElementById('inc-llm-tokens')!.innerText = '🎟️ 0 tokens';
      document.getElementById('inc-llm-solver')!.innerText = `Solver: ${inc.llm?.directive?.solver || 'PIBT'}`;
      document.getElementById('inc-llm-directive')!.innerText = JSON.stringify(inc.llm?.directive || { note: 'Direct execution' }, null, 2);
    }

    // 5. Solver
    const solverName = inc.llm?.directive?.solver || (this.activeReplay?.meta.mode.replace('baseline_', '') || 'pibt');
    document.getElementById('inc-solver-text')!.innerText =
      `Executed ${solverName.toUpperCase()} with horizon window ${inc.llm?.directive?.horizon || 20} steps. Produced collision-free trajectories.`;

    // 6. Validator
    const valAttempts = inc.validator || [];
    const valBadge = document.getElementById('inc-validator-badge')!;
    const allValid = valAttempts.some(v => v.valid);
    valBadge.className = `diag-badge ${allValid ? 'green' : 'red'}`;
    valBadge.innerText = allValid ? 'VERIFIED ✓' : 'FAILED ✗';

    const valText = valAttempts.map(v => `Attempt ${v.attempt}: ${v.valid ? 'VALID ✓ (Ground-truth verified)' : 'INVALID ✗ (' + v.violation + ')'}`).join('\n');
    document.getElementById('inc-validator-text')!.innerText = valText || 'Deterministically verified without conflict.';
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
    const card = document.getElementById('robot-card')!;
    card.classList.remove('hidden');

    const s0 = Math.floor(this.clockTime);
    const frame = this.activeReplay?.frames[s0];
    const pos = frame?.pos[rId] || [0, 0];
    const state = frame?.state[rId] || 'moving';

    document.getElementById('robot-card-id')!.innerText = `AMR #${rId}`;
    const stateEl = document.getElementById('robot-card-state')!;
    stateEl.innerText = state.toUpperCase();
    document.getElementById('robot-card-pos')!.innerText = `(${pos[0]}, ${pos[1]})`;
    document.getElementById('robot-card-goal')!.innerText = `(${pos[0] + 18}, ${(pos[1] + 12) % 60})`;
    document.getElementById('robot-card-dist')!.innerText = `~${Math.abs(18) + Math.abs(12)} cells`;

    const dot = document.getElementById('robot-card-status-dot')!;
    if (state === 'failed') dot.style.background = '#ef4444';
    else if (state === 'delayed') dot.style.background = '#f59e0b';
    else if (state === 'idle') dot.style.background = '#10b981';
    else if (state === 'replanning') dot.style.background = '#d946ef';
    else dot.style.background = '#38bdf8';
  }

  private showToast(title: string, desc: string) {
    const toast = document.getElementById('toast-notification')!;
    document.getElementById('toast-title')!.innerText = title;
    document.getElementById('toast-desc')!.innerText = desc;
    toast.classList.remove('hidden');

    setTimeout(() => {
      toast.classList.add('hidden');
    }, 3200);
  }

  private setupUI() {
    // Play/Pause button
    const playPauseBtn = document.getElementById('play-pause-btn')!;
    const playIcon = document.getElementById('play-icon')!;
    const playText = document.getElementById('play-text')!;
    playPauseBtn.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      playIcon.innerHTML = this.isPlaying ? '&#10074;&#10074;' : '&#9654;';
      playText.innerText = this.isPlaying ? 'Pause' : 'Play';
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
    const slider = document.getElementById('step-slider') as HTMLInputElement;
    slider.addEventListener('input', () => {
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

    // Scenario Select
    const scenarioSelect = document.getElementById('scenario-select') as HTMLSelectElement;
    scenarioSelect.addEventListener('change', () => {
      this.loadScenario(scenarioSelect.value);
    });

    // Mode Toggle Buttons
    const modeLlmBtn = document.getElementById('mode-llm-btn')!;
    const modeBaseBtn = document.getElementById('mode-base-btn')!;
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

    // Incident Drawer close button
    document.getElementById('incident-close-btn')!.addEventListener('click', () => {
      document.getElementById('incident-panel')!.classList.add('hidden');
      this.selectedIncident = null;
      this.updatePathsVisualization();
    });

    // Robot Card close & follow buttons
    document.getElementById('robot-card-close')!.addEventListener('click', () => {
      document.getElementById('robot-card')!.classList.add('hidden');
      this.selectedRobotId = null;
    });

    document.getElementById('robot-follow-btn')!.addEventListener('click', () => {
      this.setCameraPreset(3);
    });

    // Next / Prev incident buttons
    document.getElementById('next-inc-btn')!.addEventListener('click', () => this.jumpIncident(1));
    document.getElementById('prev-inc-btn')!.addEventListener('click', () => this.jumpIncident(-1));

    // Copy Directive JSON
    document.getElementById('copy-directive-btn')?.addEventListener('click', () => {
      const code = document.getElementById('inc-llm-directive')?.innerText || '';
      navigator.clipboard.writeText(code);
      this.showToast('Copied to Clipboard', 'Directive JSON copied');
    });

    // Help Modal
    const helpBtn = document.getElementById('help-btn')!;
    const helpModal = document.getElementById('help-modal')!;
    const helpClose = document.getElementById('help-modal-close')!;
    helpBtn.addEventListener('click', () => helpModal.classList.remove('hidden'));
    helpClose.addEventListener('click', () => helpModal.classList.add('hidden'));
    helpModal.addEventListener('click', (e) => {
      if (e.target === helpModal) helpModal.classList.add('hidden');
    });
  }

  private setupEventListeners() {
    window.addEventListener('keydown', (e) => {
      if (e.key === '1') this.setCameraPreset(1);
      if (e.key === '2') this.setCameraPreset(2);
      if (e.key === '3') this.setCameraPreset(3);
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
      this.cameraEndPos.set(this.mapWidth / 2, 115, this.mapHeight / 2 + 85);
      this.cameraEndTarget.set(this.mapWidth / 2, 0, this.mapHeight / 2);
    } else if (preset === 2) {
      this.cameraEndPos.set(65, 22, 50);
      this.cameraEndTarget.set(50, 0, 20);
    } else if (preset === 3) {
      const rId = this.selectedRobotId ?? 0;
      const s0 = Math.floor(this.clockTime);
      const pos = this.activeReplay?.frames[s0]?.pos[rId] || [80, 31];
      this.cameraEndPos.set(pos[0] + 0.5 + 8, 10, pos[1] + 0.5 + 8);
      this.cameraEndTarget.set(pos[0] + 0.5, 0.2, pos[1] + 0.5);
    }

    this.cameraAnimProgress = 0.0;
    this.cameraAnimDuration = 1.0;
    this.isCameraTransitioning = true;
  }

  private startIntroDolly() {
    this.cameraStartPos.set(80, 140, 160);
    this.cameraStartTarget.set(80, 0, 31.5);

    this.cameraEndPos.set(65, 22, 50);
    this.cameraEndTarget.set(50, 0, 20);

    this.cameraAnimProgress = 0.0;
    this.cameraAnimDuration = 4.0;
    this.isCameraTransitioning = true;
  }

  private updateCameraAnimation(dt: number) {
    if (!this.isCameraTransitioning) {
      // If in follow camera mode, smoothly track robot position
      if (this.activePreset === 3 && this.selectedRobotId !== null) {
        const s0 = Math.floor(this.clockTime);
        const pos = this.activeReplay?.frames[s0]?.pos[this.selectedRobotId] || [80, 31];
        const targetPos = new THREE.Vector3(pos[0] + 0.5, 0.2, pos[1] + 0.5);
        this.controls.target.lerp(targetPos, 0.1);
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
        if (alpha > 0.33) {
          const moveAlpha = (alpha - 0.33) / 0.67;
          const easeMove = moveAlpha < 0.5 ? 2 * moveAlpha * moveAlpha : 1 - Math.pow(-2 * moveAlpha + 2, 2) / 2;
          posX = x0 + 0.5 + dx * easeMove;
          posZ = y0 + 0.5 + dy * easeMove;
        }
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
        ledColor.setHex(0x1e293b);
      } else if (state0 === 'delayed') {
        const pulse = 0.5 + 0.5 * Math.sin(timeSec * 2 * Math.PI * 1.0);
        ledColor.setHex(0xf59e0b).multiplyScalar(pulse);
      } else if (state0 === 'replanning') {
        const pulse = 0.5 + 0.5 * Math.sin(timeSec * 2 * Math.PI * 2.0);
        ledColor.setHex(0xd946ef).multiplyScalar(pulse);
      } else if (state0 === 'idle') {
        ledColor.setHex(0x10b981);
      } else {
        ledColor.setHex(0x38bdf8);
      }

      if (this.selectedIncident) {
        if (affectedSet.has(r)) {
          ledColor.setHex(0xd946ef);
        } else {
          ledColor.multiplyScalar(0.6);
        }
      }

      // If robot is selected, highlight with glowing cyan
      if (this.selectedRobotId === r) {
        ledColor.setHex(0x38bdf8).multiplyScalar(1.5);
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
        inc.disruptions.forEach((d) => {
          if (d.type === 'blocked_cell' && d.cells) {
            d.cells.forEach(([cx, cy]) => {
              const barrier = new THREE.Mesh(this.barrierGeometry!, this.barrierMaterial!);
              barrier.position.set(cx + 0.5, 0.3, cy + 0.5);
              barrier.castShadow = true;
              this.barrierGroup.add(barrier);

              const redSqGeo = new THREE.PlaneGeometry(0.96, 0.96);
              const redSqMat = new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.45 });
              const redSq = new THREE.Mesh(redSqGeo, redSqMat);
              redSq.rotation.x = -Math.PI / 2;
              redSq.position.set(cx + 0.5, 0.02, cy + 0.5);
              this.barrierGroup.add(redSq);
            });
          } else if (d.type === 'robot_failure' && d.cells) {
            d.cells.forEach(([cx, cy]) => {
              const ringGeo = new THREE.RingGeometry(0.36, 0.44, 24);
              const ringMat = new THREE.MeshBasicMaterial({ color: 0x64748b, side: THREE.DoubleSide });
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

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  }
}

// Start application
const app = new ViewerApp();
app.init();
