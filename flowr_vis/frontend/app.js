/* ==========================================================================
   FLOWR Visualization – Application JavaScript
   ========================================================================== */

const API_BASE = '';

// ── Color constants matching CSS palette ──
const COLORS = {
    lavender: '#DFC6F6',
    lavenderDark: '#9c8aac',
    chartreuse: '#D5D653',
    chartreuseDark: '#aaab42',
    amber: '#785E01',
    amberLight: '#ae9e66',
    bgViewer: '#E9E3DC',
    labelDefault: '#6f637b',
    generated: '#9c8aac',
    replaced: '#cc3333',       // red – atoms being REPLACED/regenerated
    replacedGlow: 'rgba(204, 51, 51, 0.6)',
    kept: '#2288cc',           // blue – atoms being KEPT/fixed
    keptGlow: 'rgba(34, 136, 204, 0.6)',
    // Legacy alias (fixedAtoms = atoms to replace, see naming note)
    fixed: '#cc3333',
    fixedGlow: 'rgba(204, 51, 51, 0.6)',
};

const INTERACTION_COLORS = {
    HBond: '#239fcd',
    SaltBridge: '#e35959',
};

// ===== State =====
let state = {
    workflowType: 'sbdd',           // 'sbdd' or 'lbdd'
    jobId: null,
    proteinData: null,
    ligandData: null,
    genMode: 'denovo',           // current generation mode
    fixedAtoms: new Set(),       // H-inclusive RDKit indices of atoms to REPLACE
    heavyAtomMap: new Map(),     // maps H-inclusive idx → heavy-atom-only idx
    viewer: null,
    surfaceOn: false,
    viewMode: 'complex',
    ligandModel: null,
    proteinModel: null,
    generatedResults: [],
    activeResultIdx: -1,
    generatedModel: null,
    atomLabels: [],
    selectionSpheres: [],
    selectionSphereMap: new Map(),   // idx → sphere handle for fast add/remove
    interactionShapes: [],           // 3D interaction cylinders/labels
    showingInteractions: null,       // 'both' | null
    refHasExplicitHs: false,         // whether uploaded ref ligand had Hs
    autoHighlightSpheres: [],        // spheres for auto-highlighted inpainting mask
    refLigandVisible: true,          // whether the reference ligand is shown
    genHsVisible: false,             // whether Hs are shown on generated ligands
    genUsedOptimization: false,      // whether optimize_gen_ligs or optimize_gen_ligs_hs was used
    numHeavyAtoms: null,              // user-specified or auto-detected number of heavy atoms
    pocketOnlyMode: false,           // SBDD: true when user proceeds without ligand (pocket-only de novo)
    lbddScratchMode: false,          // LBDD: true when generating from scratch (no reference)
    priorCloud: null,                // prior cloud data {center, points, ...}
    priorCloudSpheres: [],           // 3Dmol sphere handles for the cloud
    priorCloudVisible: true,         // visibility toggle for the prior cloud
    priorCloudPlacing: false,        // whether the prior cloud is in placement/drag mode
    _priorDrag: null,                // transient drag state {startX, startY, startCenter, startPoints}
    _priorPlacedCenter: null,        // user-placed center coordinates {x, y, z} (overrides file)
    _uploadedPriorCenterFilename: null, // filename after early upload of prior center
    _generating: false,                  // concurrency guard for startGeneration
    anisotropicPrior: false,                 // anisotropic (directional) Gaussian prior
    refLigandComPrior: false,            // shift prior center to variable fragment CoM
    proteinFullAtom: false,              // toggle: show entire protein as sticks
    bindingSiteVisible: false,           // toggle: show pocket residues within 3.5Å as sticks
};

// ── RDKit.js module (loaded asynchronously) ──
let RDKitModule = null;

async function initRDKitLib() {
    if (typeof initRDKitModule === 'undefined') {
        console.warn('RDKit.js not loaded from CDN');
        return;
    }
    try {
        RDKitModule = await initRDKitModule();
        console.log('RDKit.js initialized');
    } catch (e) {
        console.warn('RDKit.js initialization failed:', e);
    }
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    // Task 1: Ensure a completely clean slate on every page load / reload.
    // Clear any session-level caches that might leak across reloads.
    try { sessionStorage.clear(); } catch (_) { }

    // Start with landing page – main app init happens after launch
    initLandingPage();
    initRDKitLib();
});

function initMainApp() {
    initViewer();
    checkBackend();
    initTooltips();
    initResizeHandle();
    initFilterDiversityToggle();
    initAddNoiseToggle();
    initPropertyFilterToggle();
    initAdmeFilterToggle();
    // Task 2: Hide sidebar sections until both protein and ligand are uploaded
    _updateSidebarVisibility();
}

// =========================================================================
// Landing Page – Checkpoint Selection
// =========================================================================

let _ckptData = { base: [], project: [] };
let _selectedCkptPath = null;
let _selectedWorkflow = 'sbdd';

function onWorkflowSelect(wf) {
    _selectedWorkflow = wf;
    document.querySelectorAll('.workflow-card').forEach(c => c.classList.remove('selected'));
    const card = document.getElementById('wf-' + wf);
    if (card) card.classList.add('selected');
    // Re-fetch checkpoints for the selected workflow (sbdd → ckpts/sbdd, lbdd → ckpts/lbdd)
    _fetchCheckpoints(wf);
    // Update checkpoint directory text
    const ckptDirText = document.getElementById('ckpt-dir-text');
    if (ckptDirText) {
        const dir = wf === 'lbdd' ? 'ckpts/lbdd' : 'ckpts/sbdd';
        ckptDirText.innerHTML = `Checkpoint files are loaded from the <code>${dir}</code> directory.`;
    }
}

async function _fetchCheckpoints(workflow) {
    try {
        const resp = await fetch(`${API_BASE}/checkpoints?workflow=${workflow}`);
        _ckptData = await resp.json();
    } catch (e) {
        console.error('Failed to fetch checkpoints', e);
        _ckptData = { base: [], project: [] };
    }
    populateBaseSelect();
    populateProjectSelect();
    onCkptTypeChange(document.querySelector('input[name="ckpt-type"]:checked')?.value || 'base');
}

async function initLandingPage() {
    // Default workflow is sbdd
    _selectedWorkflow = 'sbdd';
    await _fetchCheckpoints('sbdd');
}

function populateBaseSelect() {
    const sel = document.getElementById('base-ckpt-select');
    sel.innerHTML = '';

    if (_ckptData.base.length === 0) {
        sel.innerHTML = '<option value="">No base checkpoints found</option>';
        sel.disabled = true;
        return;
    }

    _ckptData.base.forEach((ckpt, i) => {
        const opt = document.createElement('option');
        opt.value = ckpt.path;
        opt.textContent = ckpt.name;
        sel.appendChild(opt);
    });
    sel.disabled = false;
    sel.addEventListener('change', onCkptSelectChange);
    onCkptSelectChange(); // trigger initial selection
}

function populateProjectSelect() {
    const sel = document.getElementById('project-ckpt-select');
    sel.innerHTML = '';

    if (_ckptData.project.length === 0) {
        sel.innerHTML = '<option value="">No project checkpoints found</option>';
        sel.disabled = true;
        return;
    }

    _ckptData.project.forEach(proj => {
        // If a project has multiple ckpts, list them individually with project prefix
        proj.checkpoints.forEach(ckpt => {
            const opt = document.createElement('option');
            opt.value = ckpt.path;
            opt.textContent = proj.checkpoints.length > 1
                ? `${proj.project_id} / ${ckpt.name}`
                : proj.project_id;
            sel.appendChild(opt);
        });
    });
    sel.disabled = false;
    sel.addEventListener('change', onCkptSelectChange);
}

function onCkptTypeChange(type) {
    const baseGroup = document.getElementById('base-select-group');
    const projGroup = document.getElementById('project-select-group');

    if (type === 'base') {
        baseGroup.classList.remove('hidden');
        projGroup.classList.add('hidden');
    } else {
        baseGroup.classList.add('hidden');
        projGroup.classList.remove('hidden');
    }
    onCkptSelectChange();
}

function onCkptSelectChange() {
    const type = document.querySelector('input[name="ckpt-type"]:checked')?.value || 'base';
    const sel = type === 'base'
        ? document.getElementById('base-ckpt-select')
        : document.getElementById('project-ckpt-select');

    _selectedCkptPath = sel.value || null;
    const btn = document.getElementById('launch-btn');
    btn.disabled = !_selectedCkptPath;
}

async function launchApp() {
    if (!_selectedCkptPath) return;

    const btn = document.getElementById('launch-btn');
    const status = document.getElementById('landing-status');

    btn.disabled = true;
    btn.innerHTML = '⏳ Preparing…';
    status.className = 'landing-status status-loading';
    status.classList.remove('hidden');
    status.textContent = '⏳ Registering checkpoint…';

    try {
        // Register the checkpoint selection on the server.
        // Model loading is DEFERRED until the user clicks "Generate".
        const resp = await fetch(`${API_BASE}/register-checkpoint`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ckpt_path: _selectedCkptPath, workflow_type: _selectedWorkflow }),
        });
        const data = await resp.json();

        state.workflowType = _selectedWorkflow;

        status.className = 'landing-status status-success';
        status.textContent = 'Checkpoint registered. Entering app…';
        setTimeout(() => transitionToApp(), 500);

    } catch (e) {
        // Let the user in anyway — generation will fail gracefully.
        status.className = 'landing-status status-error';
        status.textContent = `Warning: ${e.message}. Entering app anyway…`;
        setTimeout(() => transitionToApp(), 1200);
    }
}



/**
 * Helper to update the top-right compute badge.
 * @param {string} label  – e.g. 'MPS', 'CUDA', 'CPU', 'WORKER OFFLINE'
 * @param {'idle'|'loading'|'loaded'|'error'} state
 */
function _updateDeviceBadge(label, state) {
    let badge = document.getElementById('device-badge');
    if (!badge) return;
    badge.textContent = label.toUpperCase();
    const cls = {
        idle: 'badge badge-idle',
        loading: 'badge badge-loading',
        loaded: 'badge badge-success',
        error: 'badge badge-error',
    };
    badge.className = cls[state] || 'badge badge-idle';
}

function transitionToApp() {
    const landing = document.getElementById('landing-page');
    const mainApp = document.getElementById('main-app');

    landing.classList.add('fade-out');
    setTimeout(() => {
        landing.classList.add('hidden');
        mainApp.classList.remove('hidden');
        initMainApp();
        _applyWorkflowMode(state.workflowType);
    }, 450);
}

/**
 * Return to the landing / checkpoint-selection page, fully resetting
 * all application state so the user can start from scratch.
 */
function returnToLanding() {
    // If a generation is in progress, cancel it first
    if (state._generating && state.jobId) {
        fetch(`${API_BASE}/cancel/${state.jobId}`, { method: 'POST' }).catch(() => { });
    }

    // ── Tear down 3D viewer contents ──
    if (state.generatedModel !== null) {
        state.viewer.removeModel(state.generatedModel);
        state.generatedModel = null;
    }
    if (state.ligandModel) {
        state.viewer.removeModel(state.ligandModel);
        state.ligandModel = null;
    }
    if (state.proteinModel) {
        state.viewer.removeModel(state.proteinModel);
        state.proteinModel = null;
    }
    clearInteractions3D();
    clearAutoHighlight();
    clearPriorCloud();
    closeMol2DOverlay();
    // Remove all labels / spheres the viewer may still hold
    state.atomLabels.forEach(l => state.viewer.removeLabel(l));
    state.atomLabels = [];
    state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
    state.selectionSpheres = [];
    state.selectionSphereMap.clear();
    if (state.viewer) {
        state.viewer.removeAllSurfaces();
        state.viewer.render();
    }

    // ── Reset JS state to factory defaults ──
    state.jobId = null;
    state.proteinData = null;
    state.ligandData = null;
    state.genMode = 'denovo';
    state.fixedAtoms.clear();
    state.heavyAtomMap.clear();
    state.generatedResults = [];
    state.activeResultIdx = -1;
    state.surfaceOn = false;
    state.viewMode = 'complex';
    state.refHasExplicitHs = false;
    state.autoHighlightSpheres = [];
    state.refLigandVisible = true;
    state.genHsVisible = false;
    state.genUsedOptimization = false;
    state.priorCloud = null;
    state.priorCloudSpheres = [];
    state.priorCloudVisible = true;
    state.priorCloudPlacing = false;
    state._priorDrag = null;
    state._priorPlacedCenter = null;
    state._uploadedPriorCenterFilename = null;
    state._generating = false;
    state.anisotropicPrior = false;
    state.refLigandComPrior = false;
    state.proteinFullAtom = false;
    state.bindingSiteVisible = false;
    state.interactionShapes = [];
    state.showingInteractions = null;
    state.workflowType = 'sbdd';
    state._conformerJobId = null;

    // De novo mode state
    state.numHeavyAtoms = null;
    state.pocketOnlyMode = false;
    state.lbddScratchMode = false;

    // ── Reset all UI elements ──
    // Upload placeholders
    document.getElementById('protein-placeholder')?.classList.remove('hidden');
    document.getElementById('protein-success')?.classList.add('hidden');
    document.getElementById('ligand-placeholder')?.classList.remove('hidden');
    document.getElementById('ligand-success')?.classList.add('hidden');
    document.getElementById('ligand-info')?.classList.add('hidden');
    // Hide de novo UI
    document.getElementById('pocket-only-toggle')?.classList.add('hidden');
    document.getElementById('num-heavy-atoms-group')?.classList.add('hidden');
    document.getElementById('lbdd-scratch-group')?.classList.add('hidden');
    document.getElementById('pocket-only-btn')?.classList.remove('active');
    document.getElementById('lbdd-scratch-btn')?.classList.remove('active');

    // Restore all upload options and guidance banners (hidden by dynamic UI logic)
    document.getElementById('sbdd-guidance')?.classList.remove('hidden');
    document.getElementById('lbdd-guidance')?.classList.remove('hidden');
    document.getElementById('ligand-upload-group')?.classList.remove('hidden');
    document.getElementById('smiles-input-group')?.classList.remove('hidden');

    // Reset LBDD-specific UI
    const confPicker = document.getElementById('conformer-picker');
    if (confPicker) { confPicker.classList.add('hidden'); }
    const confGrid = document.getElementById('conformer-grid');
    if (confGrid) confGrid.innerHTML = '';
    const smilesInput = document.getElementById('smiles-input');
    if (smilesInput) smilesInput.value = '';
    const smilesFileName = document.getElementById('smiles-file-name');
    if (smilesFileName) smilesFileName.textContent = '';
    // Viewer overlay
    document.getElementById('viewer-overlay')?.classList.remove('hidden');
    // Results
    const list = document.getElementById('results-list');
    if (list) { list.classList.add('hidden'); list.innerHTML = ''; }
    document.getElementById('results-placeholder')?.classList.remove('hidden');
    document.getElementById('gen-summary')?.classList.add('hidden');
    document.getElementById('clear-gen-btn')?.classList.add('hidden');
    document.getElementById('gen-hs-controls')?.classList.add('hidden');
    document.getElementById('save-section')?.classList.add('hidden');
    document.getElementById('viz-controls')?.classList.add('hidden');
    // Metrics
    document.getElementById('metrics-panel')?.classList.add('hidden');
    const ml = document.getElementById('metrics-log');
    if (ml) ml.innerHTML = '';
    // Progress bar
    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) progressContainer.classList.add('hidden');
    const pf = document.getElementById('progress-fill');
    if (pf) pf.style.width = '0%';
    const pt = document.getElementById('progress-text');
    if (pt) pt.textContent = 'Generating...';
    // Rank controls
    document.getElementById('rank-select-controls')?.classList.add('hidden');
    document.getElementById('rank-reset-btn')?.classList.add('hidden');
    // Affinity panel
    document.getElementById('affinity-panel-body')?.classList.add('hidden');
    document.getElementById('affinity-panel-chevron')?.classList.remove('expanded');
    // Generate button
    const btn = document.getElementById('generate-btn');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶</span> Generate Ligands';
        btn.onclick = startGeneration;
    }
    // Hide Ligand button
    const hideLigBtn = document.getElementById('btn-hide-ligand');
    if (hideLigBtn) { hideLigBtn.textContent = '👁 Hide Ligand'; hideLigBtn.classList.remove('active'); }
    // Status badge & device badge
    setBadge('idle', 'Ready');
    _updateDeviceBadge('NO GPU', 'idle');
    // Reset generation settings (mode, filters, etc.)
    _resetGenerationSettings();
    // Hide sidebar sections that need inputs
    _updateSidebarVisibility();
    // Unfreeze sidebar if it was frozen
    _freezeSidebar(false);

    // ── Transition back to landing page ──
    const mainApp = document.getElementById('main-app');
    const landing = document.getElementById('landing-page');
    mainApp.classList.add('hidden');
    landing.classList.remove('hidden');
    landing.classList.remove('fade-out');

    // Reset landing page status text and launch button
    const status = document.getElementById('landing-status');
    if (status) { status.classList.add('hidden'); status.textContent = ''; }
    const launchBtn = document.getElementById('launch-btn');
    if (launchBtn) {
        launchBtn.disabled = !_selectedCkptPath;
        launchBtn.innerHTML = '🚀 Launch';
    }
}

/**
 * Reset the current session without returning to the landing page.
 * Clears uploads, viewer, ligand/protein state, conformer picker,
 * SMILES input, and results so the user can re-upload fresh.
 */
function resetCurrentSession() {
    // Cancel any running generation
    if (state._generating && state.jobId) {
        fetch(`${API_BASE}/cancel/${state.jobId}`, { method: 'POST' }).catch(() => { });
    }

    // ── Tear down 3D viewer contents ──
    if (state.generatedModel !== null) {
        state.viewer.removeModel(state.generatedModel);
        state.generatedModel = null;
    }
    if (state.ligandModel) {
        state.viewer.removeModel(state.ligandModel);
        state.ligandModel = null;
    }
    if (state.proteinModel) {
        state.viewer.removeModel(state.proteinModel);
        state.proteinModel = null;
    }
    clearInteractions3D();
    clearAutoHighlight();
    clearPriorCloud();
    closeMol2DOverlay();
    state.atomLabels.forEach(l => state.viewer.removeLabel(l));
    state.atomLabels = [];
    state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
    state.selectionSpheres = [];
    state.selectionSphereMap.clear();
    if (state.viewer) {
        state.viewer.removeAllSurfaces();
        state.viewer.render();
    }

    // ── Reset JS state ──
    state.jobId = null;
    state.proteinData = null;
    state.ligandData = null;
    state.genMode = 'denovo';
    state.fixedAtoms.clear();
    state.heavyAtomMap.clear();
    state.generatedResults = [];
    state.activeResultIdx = -1;
    state.surfaceOn = false;
    state.viewMode = state.workflowType === 'lbdd' ? 'ligand' : 'complex';
    state.refHasExplicitHs = false;
    state.autoHighlightSpheres = [];
    state.refLigandVisible = true;
    state.genHsVisible = false;
    state.genUsedOptimization = false;
    state.priorCloud = null;
    state.priorCloudSpheres = [];
    state.priorCloudVisible = true;
    state.priorCloudPlacing = false;
    state._priorDrag = null;
    state._priorPlacedCenter = null;
    state._uploadedPriorCenterFilename = null;
    state._generating = false;
    state.anisotropicPrior = false;
    state.refLigandComPrior = false;
    state.proteinFullAtom = false;
    state.bindingSiteVisible = false;
    state._conformerJobId = null;

    // De novo mode state
    state.numHeavyAtoms = null;
    state.pocketOnlyMode = false;
    state.lbddScratchMode = false;

    // ── Reset upload UI ──
    document.getElementById('protein-placeholder')?.classList.remove('hidden');
    document.getElementById('protein-success')?.classList.add('hidden');
    document.getElementById('ligand-placeholder')?.classList.remove('hidden');
    document.getElementById('ligand-success')?.classList.add('hidden');
    document.getElementById('ligand-info')?.classList.add('hidden');
    // Hide de novo UI
    document.getElementById('pocket-only-toggle')?.classList.add('hidden');
    document.getElementById('num-heavy-atoms-group')?.classList.add('hidden');
    document.getElementById('lbdd-scratch-group')?.classList.add('hidden');
    document.getElementById('pocket-only-btn')?.classList.remove('active');
    document.getElementById('lbdd-scratch-btn')?.classList.remove('active');

    // Restore all upload options and guidance banners (hidden by dynamic UI logic)
    document.getElementById('sbdd-guidance')?.classList.remove('hidden');
    document.getElementById('lbdd-guidance')?.classList.remove('hidden');
    document.getElementById('ligand-upload-group')?.classList.remove('hidden');
    document.getElementById('smiles-input-group')?.classList.remove('hidden');

    const protFile = document.getElementById('protein-file');
    if (protFile) protFile.value = '';
    const ligFile = document.getElementById('ligand-file');
    if (ligFile) ligFile.value = '';

    // Reset LBDD-specific UI
    const confPicker = document.getElementById('conformer-picker');
    if (confPicker) confPicker.classList.add('hidden');
    const confGrid = document.getElementById('conformer-grid');
    if (confGrid) confGrid.innerHTML = '';
    const smilesInput = document.getElementById('smiles-input');
    if (smilesInput) smilesInput.value = '';
    const smilesFileName = document.getElementById('smiles-file-name');
    if (smilesFileName) smilesFileName.textContent = '';

    // Viewer overlay
    document.getElementById('viewer-overlay')?.classList.remove('hidden');

    // Results
    const list = document.getElementById('results-list');
    if (list) { list.classList.add('hidden'); list.innerHTML = ''; }
    document.getElementById('results-placeholder')?.classList.remove('hidden');
    document.getElementById('gen-summary')?.classList.add('hidden');
    document.getElementById('clear-gen-btn')?.classList.add('hidden');
    document.getElementById('gen-hs-controls')?.classList.add('hidden');
    document.getElementById('save-section')?.classList.add('hidden');
    document.getElementById('viz-controls')?.classList.add('hidden');

    // Metrics
    document.getElementById('metrics-panel')?.classList.add('hidden');
    const ml = document.getElementById('metrics-log');
    if (ml) ml.innerHTML = '';

    // Progress bar
    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) progressContainer.classList.add('hidden');
    const pf = document.getElementById('progress-fill');
    if (pf) pf.style.width = '0%';
    const pt = document.getElementById('progress-text');
    if (pt) pt.textContent = 'Generating...';

    // Rank controls
    document.getElementById('rank-select-controls')?.classList.add('hidden');
    document.getElementById('rank-reset-btn')?.classList.add('hidden');

    // Affinity panel
    document.getElementById('affinity-panel-body')?.classList.add('hidden');
    document.getElementById('affinity-panel-chevron')?.classList.remove('expanded');

    // Reset generation mode to de novo
    const denovoRadio = document.querySelector('input[name="gen-mode"][value="denovo"]');
    if (denovoRadio) denovoRadio.checked = true;
    document.getElementById('atom-selection-section')?.classList.add('hidden');
    document.getElementById('fragment-growing-opts')?.classList.add('hidden');

    // Generate button
    const btn = document.getElementById('generate-btn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">▶</span> Generate Ligands';
        btn.onclick = startGeneration;
    }

    // Hide Ligand button
    const hideLigBtn = document.getElementById('btn-hide-ligand');
    if (hideLigBtn) { hideLigBtn.textContent = '👁 Hide Ligand'; hideLigBtn.classList.remove('active'); }

    // Status badge
    setBadge('idle', 'Ready');
    _updateDeviceBadge('NO GPU', 'idle');

    // Reset generation settings
    _resetGenerationSettings();

    // Re-apply workflow mode to restore SBDD/LBDD-specific elements
    // (e.g. the LBDD scratch option that was explicitly hidden above)
    _applyWorkflowMode(state.workflowType);

    // Unfreeze sidebar
    _freezeSidebar(false);

    showToast('Session reset — upload new files to start', 'info');
}

function initViewer() {
    const container = document.getElementById('viewer3d');
    state.viewer = $3Dmol.createViewer(container, {
        backgroundColor: COLORS.bgViewer,
        antialias: true,
        id: 'mol-viewer',
    });
    // NOTE: Do NOT call setClickable here — no models exist yet.
    // setClickable is applied in renderLigand() and reapplied after every
    // setStyle call in reapplyClickable().
    state.viewer.render();
}

async function checkBackend() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        const dot = document.getElementById('backend-status');
        dot.style.color = data.status === 'ok' ? COLORS.chartreuse : '#e05555';
        dot.title = `Server: ${data.status} | RDKit: ${data.rdkit}`;

        // Create the compute badge (starts idle — no GPU allocated yet)
        let deviceBadge = document.getElementById('device-badge');
        if (!deviceBadge) {
            deviceBadge = document.createElement('span');
            deviceBadge.id = 'device-badge';
            deviceBadge.style.marginRight = '6px';
            deviceBadge.style.fontSize = '11px';
            dot.parentElement.insertBefore(deviceBadge, dot);
        }
        // Badge stays idle until Generate is clicked
        _updateDeviceBadge('NO GPU', 'idle');
    } catch (e) {
        document.getElementById('backend-status').style.color = '#e05555';
    }
}

// ===== File Upload =====
function handleProteinDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) {
        const input = document.getElementById('protein-file');
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        uploadProtein(input);
    }
}

function handleLigandDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) {
        const input = document.getElementById('ligand-file');
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        uploadLigand(input);
    }
}

async function uploadProtein(input) {
    const file = input.files[0];
    if (!file) return;

    setBadge('loading', 'Uploading…');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch(`${API_BASE}/upload/protein`, {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Reset all state from previous job when re-uploading a protein
        if (state.generatedModel !== null) {
            state.viewer.removeModel(state.generatedModel);
            state.generatedModel = null;
        }
        // Remove previous ligand model, labels, and selection spheres
        if (state.ligandModel !== null) {
            state.viewer.removeModel(state.ligandModel);
            state.ligandModel = null;
        }
        state.atomLabels.forEach(l => state.viewer.removeLabel(l));
        state.atomLabels = [];
        state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
        state.selectionSpheres = [];
        state.selectionSphereMap.clear();
        // Remove any surfaces
        if (state.surfaceOn) {
            state.viewer.removeAllSurfaces();
            state.surfaceOn = false;
            const surfBtn = document.getElementById('btn-surface');
            if (surfBtn) surfBtn.classList.remove('active');
        }
        state.generatedResults = [];
        state.activeResultIdx = -1;
        state.ligandData = null;
        state.fixedAtoms.clear();
        state.heavyAtomMap.clear();
        state.refLigandVisible = true;
        state.genHsVisible = false;
        state.genUsedOptimization = false;
        // Reset Hide Ligand button
        const hideLigBtn = document.getElementById('btn-hide-ligand');
        if (hideLigBtn) { hideLigBtn.textContent = '👁 Hide Ligand'; hideLigBtn.classList.remove('active'); }
        clearInteractions3D();
        clearAutoHighlight();
        clearPriorCloud();
        state._uploadedPriorCenterFilename = null;
        closeMol2DOverlay();
        // Reset ligand upload UI
        document.getElementById('ligand-placeholder').classList.remove('hidden');
        document.getElementById('ligand-success').classList.add('hidden');
        document.getElementById('ligand-info')?.classList.add('hidden');
        // Reset results UI
        const list = document.getElementById('results-list');
        list.classList.add('hidden');
        list.innerHTML = '';
        document.getElementById('results-placeholder').classList.remove('hidden');
        document.getElementById('gen-summary').classList.add('hidden');
        document.getElementById('clear-gen-btn')?.classList.add('hidden');
        document.getElementById('gen-hs-controls')?.classList.add('hidden');
        document.getElementById('save-section')?.classList.add('hidden');
        // Reset generation mode to de novo
        state.genMode = 'denovo';
        const denovoRadio = document.querySelector('input[name="gen-mode"][value="denovo"]');
        if (denovoRadio) denovoRadio.checked = true;
        document.getElementById('atom-selection-section').classList.add('hidden');
        document.getElementById('fragment-growing-opts')?.classList.add('hidden');
        // Reset generation settings & filter options
        _resetGenerationSettings();
        // Reset metrics
        const metricsPanel = document.getElementById('metrics-panel');
        if (metricsPanel) metricsPanel.classList.add('hidden');
        const metricsLog = document.getElementById('metrics-log');
        if (metricsLog) metricsLog.innerHTML = '';

        state.jobId = data.job_id;
        state.proteinData = data;

        document.getElementById('protein-placeholder').classList.add('hidden');
        document.getElementById('protein-success').classList.remove('hidden');
        document.getElementById('protein-filename').textContent = data.filename;
        document.getElementById('viewer-overlay').classList.add('hidden');

        // Show the pocket-only toggle (SBDD)
        const pocketToggle = document.getElementById('pocket-only-toggle');
        if (pocketToggle) pocketToggle.classList.remove('hidden');

        renderProtein(data.pdb_data, data.format);
        setBadge('success', 'Protein loaded');
        showToast(`Protein loaded: ${data.filename}`, 'success');
        updateGenerateBtn();
    } catch (e) {
        console.error(e);
        setBadge('error', 'Upload failed');
        showToast('Failed to upload protein file', 'error');
    }
}

async function uploadLigand(input) {
    const file = input.files[0];
    if (!file) return;

    const isLBDD = state.workflowType === 'lbdd';

    // In SBDD mode, protein must be uploaded first (creates the job)
    if (!isLBDD && !state.jobId) {
        showToast('Upload a protein first', 'error');
        return;
    }

    setBadge('loading', 'Uploading…');
    const formData = new FormData();
    formData.append('file', file);

    try {
        let resp, data;
        if (isLBDD && !state.jobId) {
            // LBDD: use /upload/molecule which creates a job without protein
            resp = await fetch(`${API_BASE}/upload/molecule`, {
                method: 'POST',
                body: formData,
            });
        } else {
            resp = await fetch(`${API_BASE}/upload/ligand/${state.jobId}`, {
                method: 'POST',
                body: formData,
            });
        }
        if (!resp.ok) throw new Error(await resp.text());
        data = await resp.json();

        // For LBDD, the molecule upload returns a job_id
        if (isLBDD && data.job_id) {
            state.jobId = data.job_id;
        }

        // ── Reset ligand-dependent state from previous ligand ──
        if (state.generatedModel !== null) {
            state.viewer.removeModel(state.generatedModel);
            state.generatedModel = null;
        }
        state.generatedResults = [];
        state.activeResultIdx = -1;
        state.fixedAtoms.clear();
        state.heavyAtomMap.clear();
        state.genHsVisible = false;
        state.genUsedOptimization = false;
        clearInteractions3D();
        clearAutoHighlight();
        clearPriorCloud();
        closeMol2DOverlay();
        // Reset results UI
        const resList = document.getElementById('results-list');
        if (resList) { resList.classList.add('hidden'); resList.innerHTML = ''; }
        document.getElementById('results-placeholder')?.classList.remove('hidden');
        document.getElementById('gen-summary')?.classList.add('hidden');
        document.getElementById('clear-gen-btn')?.classList.add('hidden');
        document.getElementById('gen-hs-controls')?.classList.add('hidden');
        document.getElementById('save-section')?.classList.add('hidden');
        document.getElementById('viz-controls')?.classList.add('hidden');
        // Reset metrics
        const mp = document.getElementById('metrics-panel');
        if (mp) mp.classList.add('hidden');
        const ml = document.getElementById('metrics-log');
        if (ml) ml.innerHTML = '';

        state.ligandData = data;

        // Build atom index map for O(1) lookups by idx
        state.ligandData._atomByIdx = new Map();
        data.atoms.forEach(a => state.ligandData._atomByIdx.set(a.idx, a));

        // Build mapping from H-inclusive RDKit indices → heavy-atom-only indices.
        // FLOWR removes hydrogens, so atom numbering changes. We must send
        // heavy-atom-only indices to the server for substructure inpainting.
        state.heavyAtomMap = new Map();
        let heavyIdx = 0;
        data.atoms.forEach(a => {
            if (a.atomicNum !== 1) {
                state.heavyAtomMap.set(a.idx, heavyIdx);
                heavyIdx++;
            }
        });

        document.getElementById('ligand-placeholder').classList.add('hidden');
        document.getElementById('ligand-success').classList.remove('hidden');
        document.getElementById('ligand-filename').textContent = data.filename;

        const infoCard = document.getElementById('ligand-info');
        infoCard.classList.remove('hidden');
        document.getElementById('ligand-smiles').textContent = data.smiles_noH || data.smiles || 'N/A';
        document.getElementById('ligand-heavy-atoms').textContent = data.num_heavy_atoms;
        document.getElementById('ligand-total-atoms').textContent = data.num_atoms;

        // Track whether the original file had explicit Hs
        state.refHasExplicitHs = data.has_explicit_hs || false;

        // Show detected reference affinity
        const refAffEl = document.getElementById('ligand-ref-affinity');
        if (refAffEl && data.ref_affinity) {
            const ra = data.ref_affinity;
            refAffEl.textContent = `${ra.p_label}: ${ra.p_value.toFixed(2)}`;
            refAffEl.title = `From SDF tag "${ra.raw_tag}"` + (ra.unit ? ` (${ra.assay_type} = ${ra.raw_value} ${ra.unit})` : '');
            refAffEl.closest('.info-row')?.classList.remove('hidden');
        } else if (refAffEl) {
            refAffEl.closest('.info-row')?.classList.add('hidden');
        }

        document.getElementById('viewer-overlay')?.classList.add('hidden');
        renderLigand(data.sdf_data);
        setBadge('success', 'Ligand loaded');
        showToast(`Ligand loaded: ${data.filename} (${data.num_heavy_atoms} heavy atoms)`, 'success');

        // ── De novo helpers: populate num_heavy_atoms from reference ligand ──
        state.numHeavyAtoms = data.num_heavy_atoms;
        _showNumHeavyAtomsField(data.num_heavy_atoms, 'from reference — editable');

        // ── Populate property filter panel with reference-based defaults ──
        if (data.ref_properties) {
            populatePropertyFilterPanel(data.ref_properties);
        }

        // If pocket-only / scratch mode was previously active, deactivate it
        // now that a real ligand has been provided.
        if (state.pocketOnlyMode) {
            state.pocketOnlyMode = false;
            document.getElementById('pocket-only-btn')?.classList.remove('active');
        }
        if (state.lbddScratchMode) {
            state.lbddScratchMode = false;
            document.getElementById('lbdd-scratch-btn')?.classList.remove('active');
        }

        // Hide unselected upload options and guidance text after successful upload
        if (!isLBDD) {
            // SBDD: hide guidance and pocket-only toggle
            document.getElementById('sbdd-guidance')?.classList.add('hidden');
            document.getElementById('pocket-only-toggle')?.classList.add('hidden');
        } else {
            // LBDD: hide guidance, SMILES input, and scratch button
            document.getElementById('lbdd-guidance')?.classList.add('hidden');
            document.getElementById('smiles-input-group')?.classList.add('hidden');
            document.getElementById('lbdd-scratch-group')?.classList.add('hidden');
        }

        // Re-enable conditional generation modes now that a reference ligand exists
        document.querySelectorAll('input[name="gen-mode"]').forEach(r => {
            r.disabled = false;
            r.closest('.mode-option')?.classList.remove('hidden');
        });
        // Re-show filter-substructure checkbox
        document.getElementById('filter-cond-substructure')?.closest('.setting-group')?.classList.remove('hidden');

        updateGenerateBtn();
        _updateSidebarVisibility();

        // Show prior cloud for the currently selected mode now that inputs
        // are available (de novo is the default).
        const hasSbddInputs = !isLBDD && state.proteinData && state.ligandData && state.jobId;
        const hasLbddInputs = isLBDD && state.ligandData && state.jobId;
        if (hasSbddInputs || hasLbddInputs) {
            if (state.genMode === 'substructure_inpainting') {
                _updateSubstructurePriorCloud();
            } else {
                _fetchAndRenderPriorCloudPreview();
            }
        }
    } catch (e) {
        console.error(e);
        setBadge('error', 'Upload failed');
        showToast('Failed to upload ligand file', 'error');
    }
}

// =========================================================================
// Pocket-Only Mode (SBDD) & Scratch Mode (LBDD) + Heavy Atoms Control
// =========================================================================

/**
 * Enable pocket-only de novo mode (SBDD).
 * The user uploaded a protein pocket file but no ligand.
 * Shows the num_heavy_atoms field and enables generation controls.
 */
async function enablePocketOnlyMode() {
    state.pocketOnlyMode = true;
    state.genMode = 'denovo';

    // Show the num heavy atoms field
    _showNumHeavyAtomsField(25, 'Pocket-only de novo (no reference ligand)');

    // Hide unselected upload options and guidance text
    document.getElementById('sbdd-guidance')?.classList.add('hidden');
    document.getElementById('ligand-upload-group')?.classList.add('hidden');
    document.getElementById('pocket-only-toggle')?.classList.add('hidden');

    // Hide viewer overlay
    document.getElementById('viewer-overlay')?.classList.add('hidden');

    // Force de novo mode and disable non-applicable modes
    const denovoRadio = document.querySelector('input[name="gen-mode"][value="denovo"]');
    if (denovoRadio) denovoRadio.checked = true;
    // Hide conditional modes that require a reference ligand
    document.querySelectorAll('input[name="gen-mode"]:not([value="denovo"])').forEach(r => {
        r.disabled = true;
        r.closest('.mode-option')?.classList.add('hidden');
    });
    // Hide the filter-substructure checkbox (no substructure to filter against)
    document.getElementById('filter-cond-substructure')?.closest('.setting-group')?.classList.add('hidden');

    // Update UI
    updateGenerateBtn();
    _updateSidebarVisibility();

    // Create a de novo job on the server (no ligand needed)
    try {
        const resp = await fetch(`${API_BASE}/create-denovo-job`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: state.jobId,
                num_heavy_atoms: state.numHeavyAtoms,
                workflow_type: 'sbdd',
            }),
        });
        if (resp.ok) {
            const data = await resp.json();
            // job_id stays the same; job now has de novo metadata
        }
    } catch (e) {
        console.warn('Failed to create de novo job:', e);
    }

    // Fetch and render prior cloud preview at pocket COM
    _fetchAndRenderPriorCloudPreview();

    showToast('Pocket-only de novo mode enabled — specify the number of heavy atoms', 'info');
}

/**
 * Enable LBDD scratch mode — generate molecules without a reference.
 * Shows the num_heavy_atoms field and enables generation.
 */
async function enableLbddScratchMode() {
    state.lbddScratchMode = true;
    state.genMode = 'denovo';

    // Show the num heavy atoms field
    _showNumHeavyAtomsField(25, 'De novo from scratch (no reference molecule)');

    // Hide unselected upload options and guidance text
    document.getElementById('lbdd-guidance')?.classList.add('hidden');
    document.getElementById('ligand-upload-group')?.classList.add('hidden');
    document.getElementById('smiles-input-group')?.classList.add('hidden');
    document.getElementById('lbdd-scratch-group')?.classList.add('hidden');

    // Hide viewer overlay
    document.getElementById('viewer-overlay')?.classList.add('hidden');

    // Force de novo mode and disable non-applicable modes
    const denovoRadio = document.querySelector('input[name="gen-mode"][value="denovo"]');
    if (denovoRadio) denovoRadio.checked = true;
    document.querySelectorAll('input[name="gen-mode"]:not([value="denovo"])').forEach(r => {
        r.disabled = true;
        r.closest('.mode-option')?.classList.add('hidden');
    });
    // Hide the filter-substructure checkbox (no substructure to filter against)
    document.getElementById('filter-cond-substructure')?.closest('.setting-group')?.classList.add('hidden');

    // Create a scratch job on the server
    try {
        const resp = await fetch(`${API_BASE}/create-denovo-job`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_heavy_atoms: state.numHeavyAtoms,
                workflow_type: 'lbdd',
            }),
        });
        if (resp.ok) {
            const data = await resp.json();
            state.jobId = data.job_id;
        }
    } catch (e) {
        console.warn('Failed to create scratch job:', e);
    }

    updateGenerateBtn();
    _updateSidebarVisibility();

    // Render a prior cloud (centered at origin for LBDD scratch)
    _fetchAndRenderPriorCloudPreview();

    showToast('Scratch mode enabled — specify the number of heavy atoms to generate', 'info');
}

/**
 * Handle changes to the Number of Heavy Atoms input.
 * Dynamically updates the prior cloud preview.
 */
let _numHeavyAtomsDebounce = null;
function onNumHeavyAtomsChange() {
    const input = document.getElementById('num-heavy-atoms');
    const val = parseInt(input?.value);
    if (!val || val < 1 || val > 200) return;
    state.numHeavyAtoms = val;

    // Debounce the prior cloud update to avoid excessive requests
    clearTimeout(_numHeavyAtomsDebounce);
    _numHeavyAtomsDebounce = setTimeout(() => {
        if (state.genMode === 'denovo' && state.jobId) {
            _fetchAndRenderPriorCloudPreview();
        }
    }, 250);

    updateGenerateBtn();
}

/**
 * Show the num-heavy-atoms field with optional reference count.
 */
function _showNumHeavyAtomsField(refCount, hintText) {
    const group = document.getElementById('num-heavy-atoms-group');
    // Only show the field visually when de novo mode is active
    if (group && state.genMode === 'denovo') group.classList.remove('hidden');
    const input = document.getElementById('num-heavy-atoms');
    if (input) {
        if (refCount != null) {
            input.value = refCount;
            state.numHeavyAtoms = refCount;
        }
    }
    const hint = document.getElementById('num-heavy-atoms-hint');
    if (hint && hintText) hint.textContent = hintText;
}

/**
 * Hide the num-heavy-atoms field.
 */
function _hideNumHeavyAtomsField() {
    const group = document.getElementById('num-heavy-atoms-group');
    if (group) group.classList.add('hidden');
    state.numHeavyAtoms = null;
}

// =========================================================================
// SMILES Input & Conformer Picker (LBDD)
// =========================================================================

async function submitSmiles() {
    const inp = document.getElementById('smiles-input');
    const smiles = inp?.value?.trim();
    if (!smiles) { showToast('Enter a SMILES string', 'error'); return; }

    const btn = document.getElementById('smiles-submit-btn');
    btn.disabled = true;
    btn.textContent = '⏳…';
    setBadge('loading', 'Generating conformers…');

    try {
        const resp = await fetch(`${API_BASE}/generate-conformers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles, max_confs: 10 }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        if (!data.conformers || data.conformers.length === 0) {
            showToast('No conformers could be generated', 'error');
            return;
        }

        // Store job_id from conformer generation
        state._conformerJobId = data.job_id;

        // Show conformer picker
        _renderConformerGrid(data.conformers, data.job_id);
        setBadge('success', `${data.conformers.length} conformers`);
        showToast(`Generated ${data.conformers.length} conformers`, 'success');
    } catch (e) {
        console.error(e);
        showToast('Conformer generation failed: ' + e.message, 'error');
        setBadge('error', 'Failed');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate 3D';
    }
}

function uploadSmilesFile(input) {
    const file = input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
        const text = e.target.result.trim();
        // Take first line, first column (space/tab separated)
        const firstSmiles = text.split('\n')[0].split(/[\s\t]/)[0].trim();
        if (firstSmiles) {
            document.getElementById('smiles-input').value = firstSmiles;
            document.getElementById('smiles-file-name').textContent = file.name;
        }
    };
    reader.readAsText(file);
}

function _renderConformerGrid(conformers, jobId) {
    const grid = document.getElementById('conformer-grid');
    const picker = document.getElementById('conformer-picker');
    grid.innerHTML = '';
    picker.classList.remove('hidden');

    // Hide the SMILES input and other upload options once conformers are shown
    document.getElementById('smiles-input-group')?.classList.add('hidden');
    document.getElementById('lbdd-guidance')?.classList.add('hidden');
    document.getElementById('ligand-upload-group')?.classList.add('hidden');
    document.getElementById('lbdd-scratch-group')?.classList.add('hidden');

    // Sort by energy (lowest first) — server already sorts, but ensure here.
    const sorted = conformers
        .map((c, i) => ({ ...c, _origIdx: i }))
        .sort((a, b) => (a.energy ?? Infinity) - (b.energy ?? Infinity));

    // Build a dropdown <select>
    const sel = document.createElement('select');
    sel.id = 'conformer-select';
    sel.className = 'conformer-select';
    sorted.forEach((conf, rank) => {
        const opt = document.createElement('option');
        opt.value = conf._origIdx;
        const eStr = conf.energy != null ? conf.energy.toFixed(2) + ' kcal/mol' : '—';
        opt.textContent = `Conformer #${rank + 1}  —  ${eStr}`;
        sel.appendChild(opt);
    });
    sel.onchange = () => _selectConformer(jobId, parseInt(sel.value), conformers);
    grid.appendChild(sel);

    // Auto-select lowest-energy conformer (first in sorted list)
    if (sorted.length > 0) {
        _selectConformer(jobId, sorted[0]._origIdx, conformers);
    }
}

async function _selectConformer(jobId, confIdx, conformers) {
    setBadge('loading', 'Loading conformer…');

    try {
        const resp = await fetch(`${API_BASE}/select-conformer/${jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conformer_idx: confIdx }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Now the SDF is saved as ligand for this job — update state as if uploaded
        state.jobId = data.job_id;
        state.ligandData = data;

        // Build atom index map for O(1) lookups by idx
        state.ligandData._atomByIdx = new Map();
        (data.atoms || []).forEach(a => state.ligandData._atomByIdx.set(a.idx, a));

        // Build heavy atom map
        state.heavyAtomMap = new Map();
        let heavyIdx = 0;
        (data.atoms || []).forEach(a => {
            if (a.atomicNum !== 1) {
                state.heavyAtomMap.set(a.idx, heavyIdx);
                heavyIdx++;
            }
        });

        // Update UI
        document.getElementById('ligand-placeholder')?.classList.add('hidden');
        document.getElementById('ligand-success')?.classList.remove('hidden');
        document.getElementById('ligand-filename').textContent = data.filename || 'conformer.sdf';

        const infoCard = document.getElementById('ligand-info');
        if (infoCard) {
            infoCard.classList.remove('hidden');
            document.getElementById('ligand-smiles').textContent = data.smiles_noH || data.smiles || '';
            document.getElementById('ligand-heavy-atoms').textContent = data.num_heavy_atoms || '';
            document.getElementById('ligand-total-atoms').textContent = data.num_atoms || '';
        }

        document.getElementById('viewer-overlay')?.classList.add('hidden');
        if (data.sdf_data) renderLigand(data.sdf_data);

        setBadge('success', 'Conformer loaded');
        showToast(`Conformer #${confIdx + 1} selected`, 'success');

        // Populate num_heavy_atoms from conformer data (same as uploadLigand)
        state.numHeavyAtoms = data.num_heavy_atoms;
        _showNumHeavyAtomsField(data.num_heavy_atoms, 'from reference — editable');

        // Hide unselected upload options and guidance text (LBDD conformer path)
        document.getElementById('lbdd-guidance')?.classList.add('hidden');
        document.getElementById('ligand-upload-group')?.classList.add('hidden');
        document.getElementById('lbdd-scratch-group')?.classList.add('hidden');

        updateGenerateBtn();
        _updateSidebarVisibility();

        // Trigger prior cloud for the current mode (de novo is default)
        const isLBDD = state.workflowType === 'lbdd';
        const hasInputs = isLBDD
            ? (state.ligandData && state.jobId)
            : (state.proteinData && state.ligandData && state.jobId);
        if (hasInputs) {
            if (state.genMode === 'substructure_inpainting') {
                _updateSubstructurePriorCloud();
            } else {
                _fetchAndRenderPriorCloudPreview();
            }
        }
    } catch (e) {
        console.error(e);
        showToast('Failed to select conformer', 'error');
        setBadge('error', 'Failed');
    }
}

// =========================================================================
// 3D Rendering
// =========================================================================

function renderProtein(pdbData, format) {
    // Remove previous protein model if re-uploading
    if (state.proteinModel !== null) {
        state.viewer.removeModel(state.proteinModel);
        state.proteinModel = null;
    }
    // Reset full-atom / binding-site toggles
    state.proteinFullAtom = false;
    state.bindingSiteVisible = false;
    const fullAtomBtn = document.getElementById('btn-full-atom');
    if (fullAtomBtn) fullAtomBtn.classList.remove('active');
    const bindingSiteBtn = document.getElementById('btn-binding-site');
    if (bindingSiteBtn) bindingSiteBtn.classList.remove('active');

    const fmt = (format === '.cif' || format === '.mmcif') ? 'cif' : 'pdb';
    state.proteinModel = state.viewer.addModel(pdbData, fmt);
    applyProteinStyle();
    state.viewer.zoomTo();
    state.viewer.render();
}

/**
 * Return the best available ligand model for binding-site/interaction
 * calculations — prefers the reference ligand, falls back to the
 * currently selected generated ligand.
 */
function _getActiveLigandModel() {
    return state.ligandModel || state.generatedModel;
}

function applyProteinStyle() {
    // Base: always cartoon
    state.viewer.setStyle(
        { model: state.proteinModel },
        {
            cartoon: {
                color: 'spectrum',
                opacity: 0.85,
                thickness: 0.2,
            }
        }
    );

    // Full atom overlay: sticks on the entire protein
    if (state.proteinFullAtom) {
        state.viewer.addStyle(
            { model: state.proteinModel },
            {
                stick: { radius: 0.1, colorscheme: 'Jmol' },
            }
        );
    }

    // Binding site overlay: sticks on whole residues within 3.5Å of ligand
    if (state.bindingSiteVisible && _getActiveLigandModel()) {
        const bsSerials = _getBindingSiteSerials(3.5);
        if (bsSerials.length > 0) {
            state.viewer.addStyle(
                { model: state.proteinModel, serial: bsSerials },
                { stick: { radius: 0.15, colorscheme: 'Jmol' } }
            );
        }
    }
}

/**
 * Compute atom serial numbers of protein residues within `cutoff` Å of the
 * ligand. Selects whole residues (if any atom of a residue is within cutoff,
 * all atoms of that residue are included).
 */
function _getBindingSiteSerials(cutoff) {
    const activeLigand = _getActiveLigandModel();
    if (!state.proteinModel || !activeLigand) return [];
    const ligAtoms = activeLigand.selectedAtoms({});
    const protAtoms = state.proteinModel.selectedAtoms({});
    if (!ligAtoms.length || !protAtoms.length) return [];
    const cutoffSq = cutoff * cutoff;

    // Find residues with at least one atom within cutoff of any ligand atom
    const nearbyResKeys = new Set();
    for (const pAtom of protAtoms) {
        for (const lAtom of ligAtoms) {
            const dx = pAtom.x - lAtom.x;
            const dy = pAtom.y - lAtom.y;
            const dz = pAtom.z - lAtom.z;
            if (dx * dx + dy * dy + dz * dz < cutoffSq) {
                nearbyResKeys.add(`${pAtom.chain}_${pAtom.resi}`);
                break;
            }
        }
    }

    // Collect all atom serial numbers from those residues
    const serials = [];
    for (const pAtom of protAtoms) {
        if (nearbyResKeys.has(`${pAtom.chain}_${pAtom.resi}`)) {
            serials.push(pAtom.serial);
        }
    }
    return serials;
}

function renderLigand(sdfData) {
    // Remove previous ligand model if re-uploading
    if (state.ligandModel !== null) {
        state.viewer.removeModel(state.ligandModel);
        state.ligandModel = null;
    }
    state.ligandModel = state.viewer.addModel(sdfData, 'sdf');

    // 1. Apply visual style
    applyLigandStyleBase();

    // 2. Mark atoms clickable AFTER the model is added + styled
    reapplyClickable();

    state.viewer.zoomTo({ model: state.ligandModel }, 300);
    state.viewer.render();
    addAtomLabels();
}

/** Base ligand style (sticks + spheres, Jmol colors) */
function applyLigandStyleBase() {
    state.viewer.setStyle(
        { model: state.ligandModel },
        {
            stick: { radius: 0.15, colorscheme: 'Jmol' },
            sphere: { radius: 0.3, colorscheme: 'Jmol' },
        }
    );
}

/**
 * (Re-)apply setClickable to the ligand model.
 * Must be called AFTER every setStyle that touches the ligand model,
 * because setStyle resets the internal intersection shapes used for picking.
 */
function reapplyClickable() {
    if (!state.ligandModel) return;

    state.viewer.setClickable(
        { model: state.ligandModel },
        true,
        function (atom, viewer, event) {
            if (!state.ligandData) return;
            // Only allow atom selection in substructure_inpainting mode
            if (state.genMode !== 'substructure_inpainting') return;
            // Require Shift+click for atom selection
            if (!event || !event.shiftKey) return;
            // Only allow selecting heavy atoms (skip H)
            if (atom.elem === 'H') return;
            const idx = findLigandAtomIdx(atom);
            if (idx !== null) {
                // Double-check it's not H
                const atomInfo = state.ligandData._atomByIdx.get(idx);
                if (atomInfo && atomInfo.atomicNum === 1) return;
                toggleAtom(idx);
            }
        }
    );
}

// ── Selection blob tuning ──
const SELECTION_BLOB_OPACITY = 0.65;   // 0 = invisible, 1 = fully opaque
const SELECTION_BLOB_RADIUS = 0.38;   // slightly larger than base atom sphere (0.3)

function addAtomLabels() {
    state.atomLabels.forEach(l => state.viewer.removeLabel(l));
    state.atomLabels = [];

    // Remove old selection blobs
    state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
    state.selectionSpheres = [];
    state.selectionSphereMap.clear();

    if (!state.ligandData) return;
    // Don't add labels/spheres when the ref ligand is hidden
    if (!state.refLigandVisible) return;

    state.ligandData.atoms.forEach((atom) => {
        if (atom.atomicNum === 1) return; // skip H

        // Plain label – no color changes for selected atoms
        const label = state.viewer.addLabel(
            `${atom.symbol}${atom.idx}`,
            {
                position: { x: atom.x, y: atom.y, z: atom.z },
                fontSize: 11,
                fontColor: COLORS.labelDefault,
                backgroundColor: 'transparent',
                backgroundOpacity: 0,
                showBackground: false,
                alignment: 'center',
            }
        );
        state.atomLabels.push(label);

        // Translucent red blob around selected atoms
        if (state.fixedAtoms.has(atom.idx)) {
            const blob = state.viewer.addSphere({
                center: { x: atom.x, y: atom.y, z: atom.z },
                radius: SELECTION_BLOB_RADIUS,
                color: COLORS.fixed,
                opacity: SELECTION_BLOB_OPACITY,
            });
            state.selectionSpheres.push(blob);
            state.selectionSphereMap.set(atom.idx, blob);
        }
    });
    state.viewer.render();
}

/**
 * Update the visual highlight of fixed vs. unfixed atoms,
 * then re-apply clickable (because setStyle nukes intersection shapes).
 */
function updateAtomHighlights() {
    if (!state.ligandModel || !state.ligandData) return;

    // 1. Reset base style (Jmol colors for all atoms – no color changes)
    applyLigandStyleBase();

    // 2. Restore clickable after the style reset
    reapplyClickable();

    state.viewer.render();

    // 3. Redraw labels + selection blobs
    addAtomLabels();
}

function findLigandAtomIdx(clickedAtom) {
    if (!state.ligandData) return null;

    // The atom object from 3Dmol already carries `index` and coordinates.
    // Match by 3D proximity to our server-side atom list (robust to any
    // index mismatch between 3Dmol's internal serial and RDKit's idx).
    let bestIdx = null;
    let bestDist = Infinity;

    state.ligandData.atoms.forEach((a) => {
        const dx = a.x - clickedAtom.x;
        const dy = a.y - clickedAtom.y;
        const dz = a.z - clickedAtom.z;
        const dist = dx * dx + dy * dy + dz * dz; // squared is fine for comparison
        if (dist < bestDist && dist < 0.25) {      // 0.5 Å threshold → 0.25 squared
            bestDist = dist;
            bestIdx = a.idx;
        }
    });

    return bestIdx;
}

// =========================================================================
// Generation Mode
// =========================================================================

const MODE_LABELS = {
    denovo: 'De Novo',
    substructure_inpainting: 'Substructure Inpainting',
    scaffold_hopping: 'Scaffold Hopping',
    scaffold_elaboration: 'Scaffold Elaboration',
    linker_inpainting: 'Linker Inpainting',
    core_growing: 'Core Growing',
    fragment_growing: 'Fragment Growing',
};

function onModeChange(mode) {
    state.genMode = mode;

    // Show/hide number of heavy atoms field (only for de novo)
    const numHAGroup = document.getElementById('num-heavy-atoms-group');
    if (numHAGroup) {
        if (mode === 'denovo') {
            numHAGroup.classList.remove('hidden');
            // Pre-populate from reference ligand if available
            if (state.ligandData && state.ligandData.num_heavy_atoms && !state.numHeavyAtoms) {
                const input = document.getElementById('num-heavy-atoms');
                if (input) input.value = state.ligandData.num_heavy_atoms;
                state.numHeavyAtoms = state.ligandData.num_heavy_atoms;
            }
        } else {
            numHAGroup.classList.add('hidden');
        }
    }

    // Show/hide atom selection (only for substructure_inpainting)
    const atomSection = document.getElementById('atom-selection-section');
    atomSection.classList.toggle('hidden', mode !== 'substructure_inpainting');

    // Show/hide fragment growing sub-options
    const fragOpts = document.getElementById('fragment-growing-opts');
    if (mode === 'fragment_growing') {
        fragOpts.classList.remove('hidden');
    } else {
        fragOpts.classList.add('hidden');
    }

    // Show/hide core growing sub-options
    const coreOpts = document.getElementById('core-growing-opts');
    if (coreOpts) {
        if (mode === 'core_growing') {
            coreOpts.classList.remove('hidden');
            _fetchRingSystemCount();
        } else {
            coreOpts.classList.add('hidden');
        }
    }

    // Show/hide anisotropic prior checkbox (applicable for conditional modes)
    const anisoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
    const anisoOpt = document.getElementById('anisotropic-prior-opt');
    if (anisoOpt) {
        if (anisoModes.includes(mode)) {
            anisoOpt.classList.remove('hidden');
        } else {
            anisoOpt.classList.add('hidden');
            // Reset when switching to non-applicable mode
            const cb = document.getElementById('anisotropic-prior-cb');
            if (cb) cb.checked = false;
            state.anisotropicPrior = false;
        }
    }

    // Show/hide ref-ligand CoM prior checkbox (applicable for scaffold_hopping, scaffold_elaboration, linker_inpainting, core_growing)
    const refComModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing'];
    const refComOpt = document.getElementById('ref-ligand-com-prior-opt');
    if (refComOpt) {
        if (refComModes.includes(mode)) {
            refComOpt.classList.remove('hidden');
        } else {
            refComOpt.classList.add('hidden');
            const rcCb = document.getElementById('ref-ligand-com-prior-cb');
            if (rcCb) rcCb.checked = false;
            state.refLigandComPrior = false;
        }
    }

    // Clear manual atom selection when leaving substructure mode
    if (mode !== 'substructure_inpainting') {
        state.fixedAtoms.clear();
        updateSelectionUI();
        // Remove stale selection spheres (fixedAtoms is empty → removes all)
        _rebuildAllSpheres();
    }

    // Clear previous auto-highlight spheres
    clearAutoHighlight();

    // Auto-highlight atoms for inpainting modes
    const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
    if (autoModes.includes(mode) && state.ligandData && state.jobId) {
        fetchAndHighlightInpaintingMask(mode);
    } else {
        // Rebuild spheres for substructure mode (or clear for denovo)
        _rebuildAllSpheres();
    }

    // Auto-select filter substructure for non-denovo modes
    const filterSubCb = document.getElementById('filter-cond-substructure');
    if (filterSubCb) {
        filterSubCb.checked = (mode !== 'denovo');
    }

    // Prior cloud preview: show for all conditional modes, clear for denovo.
    // Also update if anisotropic prior is active (to show directional cloud shape).
    // Clear any user-placed center when switching modes so the new mode
    // starts with the default (pocket COM) position.
    state._priorPlacedCenter = null;
    _updatePriorCoordsDisplay(null);
    const resetBtn = document.getElementById('reset-prior-pos-btn');
    if (resetBtn) resetBtn.classList.add('hidden');

    if (mode !== 'denovo' && mode !== 'substructure_inpainting' && state.ligandData && state.jobId) {
        _fetchAndRenderPriorCloudPreview();
    } else if (mode === 'substructure_inpainting') {
        // Prior cloud will appear dynamically once atoms are selected
        clearPriorCloud();
    } else if (mode === 'denovo' && state.jobId && (state.ligandData || state.pocketOnlyMode || state.lbddScratchMode)) {
        // Show prior cloud at pocket COM for de novo mode (including pocket-only / scratch)
        _fetchAndRenderPriorCloudPreview();
    } else {
        _exitPriorPlacement();
        clearPriorCloud();
    }
}

/**
 * Handle anisotropic prior checkbox toggle.
 * Updates the prior cloud preview to show directional/shape-based cloud
 * instead of spherical when checked.
 */
function onAnisotropicPriorChange(checked) {
    state.anisotropicPrior = checked;

    // Re-fetch and render prior cloud preview with new anisotropic setting
    if (state.genMode !== 'denovo' && state.ligandData && state.jobId) {
        _fetchAndRenderPriorCloudPreview();
    }
}

/**
 * Handle ref-ligand CoM prior checkbox toggle.
 * Shifts the prior cloud center to the variable fragment CoM of the
 * reference ligand for applicable conditional modes.
 */
function onRefLigandComPriorChange(checked) {
    state.refLigandComPrior = checked;

    // Re-fetch and render prior cloud preview with new ref_ligand_com_prior setting
    if (state.genMode !== 'denovo' && state.ligandData && state.jobId) {
        _fetchAndRenderPriorCloudPreview();
    }
}

// =========================================================================
// Atom Selection
// =========================================================================

function toggleAtom(idx) {
    const wasSelected = state.fixedAtoms.has(idx);
    if (wasSelected) {
        state.fixedAtoms.delete(idx);
    } else {
        state.fixedAtoms.add(idx);
    }
    updateSelectionUI();
    // Fast path: only add/remove the single sphere — no style/clickable reset
    _updateSingleSphere(idx, !wasSelected);

    // Update prior cloud to reflect the selected fragment
    if (state.genMode === 'substructure_inpainting') {
        _updateSubstructurePriorCloud();
    }
}

/**
 * Update the prior cloud for substructure_inpainting mode.
 * Centers the cloud at the COM of the selected (to-be-replaced) atoms
 * so the user can see where new atoms will be generated.
 * If no atoms are selected, clears the prior cloud.
 */
function _updateSubstructurePriorCloud() {
    if (state.fixedAtoms.size === 0) {
        clearPriorCloud();
        return;
    }
    // Compute COM of selected (to-be-replaced) atoms
    let sx = 0, sy = 0, sz = 0, n = 0;
    state.fixedAtoms.forEach(idx => {
        const atom = state.ligandData?._atomByIdx?.get(idx);
        if (atom && atom.atomicNum !== 1) {
            sx += atom.x; sy += atom.y; sz += atom.z;
            n++;
        }
    });
    if (n === 0) {
        clearPriorCloud();
        return;
    }
    // Set the placed center to the selected fragment COM
    state._priorPlacedCenter = { x: sx / n, y: sy / n, z: sz / n };
    _fetchAndRenderPriorCloudPreview();
}

/** Add or remove a single selection sphere without touching styles/clickable. */
function _updateSingleSphere(idx, selected) {
    const atom = state.ligandData?._atomByIdx?.get(idx);
    if (!atom || atom.atomicNum === 1) return;

    if (selected) {
        const blob = state.viewer.addSphere({
            center: { x: atom.x, y: atom.y, z: atom.z },
            radius: SELECTION_BLOB_RADIUS,
            color: COLORS.fixed,
            opacity: SELECTION_BLOB_OPACITY,
        });
        state.selectionSphereMap.set(idx, blob);
        state.selectionSpheres.push(blob);
    } else {
        const blob = state.selectionSphereMap.get(idx);
        if (blob) {
            state.viewer.removeShape(blob);
            state.selectionSphereMap.delete(idx);
            const i = state.selectionSpheres.indexOf(blob);
            if (i >= 0) state.selectionSpheres.splice(i, 1);
        }
    }
    state.viewer.render();
}

function clearSelection() {
    state.fixedAtoms.clear();
    updateSelectionUI();
    _rebuildAllSpheres();
    if (state.genMode === 'substructure_inpainting') {
        _updateSubstructurePriorCloud();
    }
}

function selectAllAtoms() {
    if (!state.ligandData) return;
    state.fixedAtoms.clear();
    state.ligandData.atoms.forEach(a => {
        if (a.atomicNum !== 1) state.fixedAtoms.add(a.idx);
    });
    updateSelectionUI();
    _rebuildAllSpheres();
    if (state.genMode === 'substructure_inpainting') {
        _updateSubstructurePriorCloud();
    }
}

function selectHeavyAtoms() {
    selectAllAtoms();
}

/** Rebuild all selection spheres (batch operation). No style/clickable reset. */
function _rebuildAllSpheres() {
    // Remove existing selection spheres
    state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
    state.selectionSpheres = [];
    state.selectionSphereMap.clear();

    // Don't add spheres when the ref ligand is hidden
    if (!state.refLigandVisible) return;

    if (state.ligandData) {
        state.ligandData.atoms.forEach(atom => {
            if (atom.atomicNum === 1) return;
            if (state.fixedAtoms.has(atom.idx)) {
                const blob = state.viewer.addSphere({
                    center: { x: atom.x, y: atom.y, z: atom.z },
                    radius: SELECTION_BLOB_RADIUS,
                    color: COLORS.fixed,
                    opacity: SELECTION_BLOB_OPACITY,
                });
                state.selectionSpheres.push(blob);
                state.selectionSphereMap.set(atom.idx, blob);
            }
        });
    }
    state.viewer.render();
}

function updateSelectionUI() {
    const container = document.getElementById('selected-atoms-list');

    if (state.fixedAtoms.size === 0) {
        container.innerHTML = '<span class="muted-text">No atoms marked for replacement</span>';
        _hideInpaintingLegend();
        return;
    }

    const sorted = Array.from(state.fixedAtoms).sort((a, b) => a - b);
    const chips = sorted.map(idx => {
        const atom = state.ligandData?._atomByIdx?.get(idx);
        const label = atom ? `${atom.symbol}${idx}` : `#${idx}`;
        return `<span class="atom-chip" onclick="toggleAtom(${idx})" title="Click to deselect">
            ${label} <span class="chip-x">✕</span>
        </span>`;
    }).join('');

    container.innerHTML = chips;

    // Show legend for substructure_inpainting matching other inpainting modes
    if (state.genMode === 'substructure_inpainting') {
        const nSelected = state.fixedAtoms.size;
        const totalHeavy = state.ligandData?.num_heavy_atoms || 0;
        _showInpaintingLegend('substructure_inpainting', totalHeavy - nSelected, nSelected, false);
    }
}

// =========================================================================
// View Controls
// =========================================================================

function setViewMode(mode) {
    state.viewMode = mode;
    // Only toggle active class on view-mode buttons, not all toolbar buttons
    document.querySelectorAll('#view-complex, #view-protein, #view-ligand').forEach(
        b => b.classList.remove('active')
    );
    document.getElementById(`view-${mode}`).classList.add('active');

    if (mode === 'complex') {
        if (state.proteinModel) applyProteinStyle();
        if (state.ligandModel) {
            if (state.generatedModel !== null && state.refLigandVisible) {
                // Keep reference ligand dimmed when a generated ligand is showing
                state.viewer.setStyle(
                    { model: state.ligandModel },
                    {
                        stick: { radius: 0.1, colorscheme: 'Jmol', opacity: 0.3 },
                        sphere: { radius: 0.15, colorscheme: 'Jmol', opacity: 0.3 },
                    }
                );
                reapplyClickable();
                addAtomLabels();
            } else if (!state.refLigandVisible) {
                state.viewer.setStyle({ model: state.ligandModel }, {});
            } else {
                updateAtomHighlights();
            }
        }
        // Re-show generated ligand if it exists
        if (state.generatedModel !== null) {
            state.viewer.setStyle(
                { model: state.generatedModel },
                {
                    stick: { radius: 0.18, colorscheme: 'Jmol' },
                    sphere: { radius: 0.35, colorscheme: 'Jmol' },
                }
            );
        }
        // Restore auto-highlight spheres if needed (e.g. after returning from protein view)
        _restoreAutoHighlight();
        state.viewer.zoomTo();
    } else if (mode === 'protein') {
        if (state.proteinModel) {
            state.viewer.setStyle({ model: state.proteinModel },
                { cartoon: { color: 'spectrum', opacity: 1.0, thickness: 0.3 } });
            // Respect full-atom toggle in protein-only view
            if (state.proteinFullAtom) {
                state.viewer.addStyle(
                    { model: state.proteinModel },
                    { stick: { radius: 0.1, colorscheme: 'Jmol' } }
                );
            }
            // Respect binding-site toggle (ligand model data still exists even if hidden)
            if (state.bindingSiteVisible && state.ligandModel) {
                const bsSerials = _getBindingSiteSerials(3.5);
                if (bsSerials.length > 0) {
                    state.viewer.addStyle(
                        { model: state.proteinModel, serial: bsSerials },
                        { stick: { radius: 0.15, colorscheme: 'Jmol' } }
                    );
                }
            }
        }
        if (state.ligandModel) state.viewer.setStyle({ model: state.ligandModel }, {});
        // Hide generated ligand in protein-only view
        if (state.generatedModel !== null) {
            state.viewer.setStyle({ model: state.generatedModel }, {});
        }
        // Remove atom labels & selection spheres (they're independent of the model)
        state.atomLabels.forEach(l => state.viewer.removeLabel(l));
        state.atomLabels = [];
        state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
        state.selectionSpheres = [];
        state.selectionSphereMap.clear();
        // Also hide auto-highlight spheres
        state.autoHighlightSpheres.forEach(s => {
            try { state.viewer.removeShape(s); } catch (_) { }
        });
        state.autoHighlightSpheres = [];
        _hideInpaintingLegend();
        if (state.proteinModel) state.viewer.zoomTo({ model: state.proteinModel });
    } else if (mode === 'ligand') {
        if (state.proteinModel) state.viewer.setStyle({ model: state.proteinModel }, {});
        if (state.ligandModel) {
            if (state.generatedModel !== null && state.refLigandVisible) {
                // Keep reference ligand dimmed when a generated ligand is showing
                state.viewer.setStyle(
                    { model: state.ligandModel },
                    {
                        stick: { radius: 0.1, colorscheme: 'Jmol', opacity: 0.3 },
                        sphere: { radius: 0.15, colorscheme: 'Jmol', opacity: 0.3 },
                    }
                );
                reapplyClickable();
                addAtomLabels();
            } else if (!state.refLigandVisible) {
                state.viewer.setStyle({ model: state.ligandModel }, {});
            } else {
                updateAtomHighlights();
            }
        }
        // Re-show generated ligand if it exists
        if (state.generatedModel !== null) {
            state.viewer.setStyle(
                { model: state.generatedModel },
                {
                    stick: { radius: 0.18, colorscheme: 'Jmol' },
                    sphere: { radius: 0.35, colorscheme: 'Jmol' },
                }
            );
        }
        // Restore auto-highlight spheres if needed (e.g. after returning from protein view)
        _restoreAutoHighlight();
        if (state.ligandModel) state.viewer.zoomTo({ model: state.ligandModel }, 200);
        else if (state.generatedModel) state.viewer.zoomTo({ model: state.generatedModel }, 200);
    }
    state.viewer.render();
}

function resetView() {
    state.viewer.zoomTo();
    state.viewer.render();
}

function toggleSurface() {
    state.surfaceOn = !state.surfaceOn;
    const btn = document.getElementById('btn-surface');

    if (state.surfaceOn) {
        state.viewer.addSurface(
            $3Dmol.SurfaceType.VDW,
            { opacity: 0.6, color: 'white' },
            { model: state.proteinModel }
        );
        btn.classList.add('active');
    } else {
        state.viewer.removeAllSurfaces();
        btn.classList.remove('active');
    }
    state.viewer.render();
}

/**
 * Toggle full-atom stick representation of the entire protein.
 */
function toggleFullAtom() {
    if (!state.proteinModel) return;
    state.proteinFullAtom = !state.proteinFullAtom;
    document.getElementById('btn-full-atom')?.classList.toggle('active', state.proteinFullAtom);
    applyProteinStyle();
    state.viewer.render();
}

/**
 * Toggle binding-site view: show protein residues within 3.5 Å of the
 * reference ligand as sticks on top of the cartoon.
 */
function toggleBindingSite() {
    if (!state.proteinModel) return;
    if (!_getActiveLigandModel()) {
        showToast('Upload or generate a ligand first to show the binding site', 'error');
        return;
    }
    state.bindingSiteVisible = !state.bindingSiteVisible;
    document.getElementById('btn-binding-site')?.classList.toggle('active', state.bindingSiteVisible);
    applyProteinStyle();
    state.viewer.render();
}

function centerOnLigand() {
    if (state.ligandModel) {
        state.viewer.zoomTo({ model: state.ligandModel }, 300);
        state.viewer.render();
    }
}

/**
 * Remove all generated ligands from the viewer and reset results panel,
 * going back to the reference ligand view.
 */
function clearGeneratedLigands() {
    // Remove generated model from viewer
    if (state.generatedModel !== null) {
        state.viewer.removeModel(state.generatedModel);
        state.generatedModel = null;
    }

    // Clear state
    state.generatedResults = [];
    state.activeResultIdx = -1;

    // Reset results panel UI
    const list = document.getElementById('results-list');
    list.classList.add('hidden');
    list.innerHTML = '';
    document.getElementById('results-placeholder').classList.remove('hidden');
    document.getElementById('gen-summary').classList.add('hidden');
    const clearBtn = document.getElementById('clear-gen-btn');
    if (clearBtn) clearBtn.classList.add('hidden');

    // Clear metrics panel
    const metricsPanel = document.getElementById('metrics-panel');
    if (metricsPanel) metricsPanel.classList.add('hidden');
    const metricsLog = document.getElementById('metrics-log');
    if (metricsLog) metricsLog.innerHTML = '';

    // Hide generated H controls
    const hsControls = document.getElementById('gen-hs-controls');
    if (hsControls) hsControls.classList.add('hidden');
    state.genHsVisible = false;
    state.genUsedOptimization = false;

    // Hide rank-select controls and reset button
    const rankControls = document.getElementById('rank-select-controls');
    if (rankControls) rankControls.classList.add('hidden');
    const rankResetBtn = document.getElementById('rank-reset-btn');
    if (rankResetBtn) rankResetBtn.classList.add('hidden');

    // Collapse affinity panel body
    const affinityBody = document.getElementById('affinity-panel-body');
    if (affinityBody) affinityBody.classList.add('hidden');
    const affinityChevron = document.getElementById('affinity-panel-chevron');
    if (affinityChevron) affinityChevron.classList.remove('expanded');

    // Close 2D structure overlay
    closeMol2DOverlay();

    // Clear 3D interactions
    clearInteractions3D();

    // Hide save section
    const saveSection = document.getElementById('save-section');
    if (saveSection) saveSection.classList.add('hidden');

    // Hide visualization buttons
    const vizControls = document.getElementById('viz-controls');
    if (vizControls) vizControls.classList.add('hidden');

    // Reset progress bar
    const progressContainer = document.getElementById('progress-container');
    progressContainer.classList.add('hidden');
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = 'Generating...';

    // ── Reset generation settings & filtering options to defaults ──
    _resetGenerationSettings();

    // Restore reference ligand to full opacity
    if (state.ligandModel) {
        updateAtomHighlights();
    }

    setBadge('idle', 'Ready');
    showToast('Generated ligands cleared', 'success');
    state.viewer.render();
}

/**
 * Reset all generation settings, generation mode, and filtering/post-processing
 * options back to their default values. Called when clearing generated ligands.
 */
function _resetGenerationSettings() {
    // ── Reset generation mode to De Novo ──
    state.genMode = 'denovo';
    const denovoRadio = document.querySelector('input[name="gen-mode"][value="denovo"]');
    if (denovoRadio) denovoRadio.checked = true;

    // Hide mode-dependent sections
    document.getElementById('atom-selection-section').classList.add('hidden');
    const fragOpts = document.getElementById('fragment-growing-opts');
    if (fragOpts) fragOpts.classList.add('hidden');

    // Hide and reset anisotropic prior checkbox
    const anisoOpt = document.getElementById('anisotropic-prior-opt');
    if (anisoOpt) anisoOpt.classList.add('hidden');
    const anisoCb = document.getElementById('anisotropic-prior-cb');
    if (anisoCb) anisoCb.checked = false;
    state.anisotropicPrior = false;

    // Hide and reset ref-ligand CoM prior checkbox
    const refComOpt = document.getElementById('ref-ligand-com-prior-opt');
    if (refComOpt) refComOpt.classList.add('hidden');
    const refComCb = document.getElementById('ref-ligand-com-prior-cb');
    if (refComCb) refComCb.checked = false;
    state.refLigandComPrior = false;

    // Clear atom selection state
    state.fixedAtoms.clear();
    updateSelectionUI();

    // Reset fragment-growing inputs
    const growSizeInput = document.getElementById('grow-size');
    if (growSizeInput) growSizeInput.value = '5';
    const priorFileInput = document.getElementById('prior-center-file');
    if (priorFileInput) priorFileInput.value = '';
    const priorFilename = document.getElementById('prior-center-filename');
    if (priorFilename) priorFilename.textContent = '';
    state._uploadedPriorCenterFilename = null;
    state._priorPlacedCenter = null;
    _updatePriorCoordsDisplay(null);
    const resetPriorBtn = document.getElementById('reset-prior-pos-btn');
    if (resetPriorBtn) resetPriorBtn.classList.add('hidden');

    // Clear auto-highlight spheres and inpainting legend
    clearAutoHighlight();

    // Clear prior cloud (also exits placement mode internally)
    clearPriorCloud();

    // Reset de novo pocket-only / scratch state
    state.pocketOnlyMode = false;
    state.lbddScratchMode = false;
    state.numHeavyAtoms = null;
    _hideNumHeavyAtomsField();
    document.getElementById('pocket-only-btn')?.classList.remove('active');
    document.getElementById('lbdd-scratch-btn')?.classList.remove('active');
    // Re-enable all generation mode radio buttons and show their labels
    document.querySelectorAll('input[name="gen-mode"]').forEach(r => {
        r.disabled = false;
        r.closest('.mode-option')?.classList.remove('hidden');
    });
    // Re-show filter-substructure checkbox
    document.getElementById('filter-cond-substructure')?.closest('.setting-group')?.classList.remove('hidden');

    // Mode is now de novo — show the default prior cloud preview
    _fetchAndRenderPriorCloudPreview();

    // ── Filtering & post-processing checkboxes ──
    const _set = (id, val) => { const el = document.getElementById(id); if (el) el.checked = val; };

    _set('filter-valid-unique', true);        // default: checked
    _set('filter-cond-substructure', false);  // default: unchecked
    _set('filter-diversity', false);          // default: unchecked
    _set('sample-mol-sizes', false);          // default: unchecked
    _set('optimize-gen-ligs', false);         // default: unchecked
    _set('optimize-gen-ligs-hs', false);      // default: unchecked
    _set('calculate-pb-valid', false);        // default: unchecked
    _set('filter-pb-valid', false);           // default: unchecked
    _set('calculate-strain', false);          // default: unchecked

    // ── Add noise: default = checked with scale 0.1 ──
    _set('add-noise', true);
    const noiseSlider = document.getElementById('noise-scale');
    if (noiseSlider) noiseSlider.value = '0.1';
    const noiseVal = document.getElementById('noise-scale-val');
    if (noiseVal) noiseVal.textContent = '0.1';
    const noiseGroup = document.getElementById('noise-scale-group');
    if (noiseGroup) noiseGroup.classList.remove('hidden');

    // ── Diversity threshold: reset to 0.9, hide sub-option ──
    const divSlider = document.getElementById('diversity-threshold');
    if (divSlider) divSlider.value = '0.9';
    const divVal = document.getElementById('div-thresh-val');
    if (divVal) divVal.textContent = '0.9';
    const divGroup = document.getElementById('diversity-threshold-group');
    if (divGroup) divGroup.classList.add('hidden');

    // ── Generation settings sliders ──
    const _setSlider = (id, valId, val) => {
        const sl = document.getElementById(id);
        const sp = document.getElementById(valId);
        if (sl) sl.value = val;
        if (sp) sp.textContent = val;
    };
    _setSlider('n-samples', 'n-samples-val', '10');
    _setSlider('batch-size', 'batch-size-val', '25');
    _setSlider('integration-steps', 'steps-val', '100');
    _setSlider('pocket-cutoff', 'cutoff-val', '6.0');
}

/**
 * Toggle the visibility of the reference ligand in the 3D viewer.
 */
function toggleRefLigandVisibility() {
    if (!state.ligandModel) return;
    const btn = document.getElementById('btn-hide-ligand');

    state.refLigandVisible = !state.refLigandVisible;

    if (state.refLigandVisible) {
        // Restore: dimmed if gen ligand is showing, full otherwise
        if (state.generatedModel !== null) {
            state.viewer.setStyle(
                { model: state.ligandModel },
                {
                    stick: { radius: 0.1, colorscheme: 'Jmol', opacity: 0.3 },
                    sphere: { radius: 0.15, colorscheme: 'Jmol', opacity: 0.3 },
                }
            );
        } else {
            applyLigandStyleBase();
        }
        reapplyClickable();
        // Re-show atom labels and selection spheres
        addAtomLabels();
        // Re-show auto-highlight if an inpainting mode is active
        const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
        if (autoModes.includes(state.genMode) && state.ligandData && state.jobId) {
            fetchAndHighlightInpaintingMask(state.genMode);
        }
        // Refresh 3D interactions to include ref ligand interactions again
        if (state.showingInteractions) {
            clearInteractions3D();
            showInteractions3D('both');
        }
        if (btn) {
            btn.textContent = '👁 Hide Ligand';
            btn.classList.remove('active');
        }
    } else {
        // Hide: make transparent
        state.viewer.setStyle({ model: state.ligandModel }, {});
        // Remove atom labels, selection spheres, and auto-highlight spheres
        state.atomLabels.forEach(l => state.viewer.removeLabel(l));
        state.atomLabels = [];
        state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
        state.selectionSpheres = [];
        state.selectionSphereMap.clear();
        state.autoHighlightSpheres.forEach(s => {
            try { state.viewer.removeShape(s); } catch (_) { }
        });
        state.autoHighlightSpheres = [];
        _hideInpaintingLegend();
        // Refresh 3D interactions to hide ref ligand interactions
        if (state.showingInteractions) {
            clearInteractions3D();
            showInteractions3D('both');
        }
        if (btn) {
            btn.textContent = '👁 Show Ligand';
            btn.classList.add('active');
        }
    }
    state.viewer.render();
}

// =========================================================================
// Generation
// =========================================================================

function updateGenerateBtn() {
    const btn = document.getElementById('generate-btn');
    const isLBDD = state.workflowType === 'lbdd';

    // Standard path: need protein+ligand (SBDD) or ligand (LBDD).
    // De novo path: pocket-only (SBDD) or scratch (LBDD) need numHeavyAtoms + jobId.
    const standardReady = isLBDD ? !!state.ligandData : !!(state.proteinData && state.ligandData);
    const denovoReady = isLBDD
        ? (state.lbddScratchMode && state.numHeavyAtoms > 0 && !!state.jobId)
        : (state.pocketOnlyMode && !!state.proteinData && state.numHeavyAtoms > 0 && !!state.jobId);

    btn.disabled = !(standardReady || denovoReady);
    _updateSidebarVisibility();
}

/**
 * Task 2: Show/hide sidebar sections that should only be visible
 * when both a protein and a ligand have been uploaded.
 */
/**
 * Apply SBDD/LBDD mode: show/hide elements tagged with .sbdd-only / .lbdd-only.
 * Also updates subtitle and viewer overlay text.
 */
function _applyWorkflowMode(wf) {
    const isLBDD = wf === 'lbdd';

    // Show/hide tagged elements
    document.querySelectorAll('.sbdd-only').forEach(el => {
        if (isLBDD) el.classList.add('hidden');
        else el.classList.remove('hidden');
    });
    document.querySelectorAll('.lbdd-only').forEach(el => {
        if (isLBDD) el.classList.remove('hidden');
        else el.classList.add('hidden');
    });

    // Update topbar subtitle
    const sub = document.getElementById('topbar-subtitle');
    if (sub) sub.textContent = isLBDD ? 'Ligand-Based Molecule Generation' : 'Structure-Based Ligand Generation';

    // Update viewer overlay instructions
    const overlayTitle = document.getElementById('viewer-overlay-title');
    const overlayDesc = document.getElementById('viewer-overlay-desc');
    if (overlayTitle) overlayTitle.textContent = isLBDD ? 'Upload a Molecule' : 'Upload a Protein & Ligand';
    if (overlayDesc) overlayDesc.textContent = isLBDD
        ? 'Upload an SDF/MOL file, enter a SMILES string, or generate from scratch using the left panel.'
        : 'Upload a protein structure (PDB/CIF) and optionally a ligand (SDF/MOL), or use pocket-only mode for de novo generation.';

    // Update ligand upload label
    const ligLabel = document.getElementById('ligand-upload-label');
    if (ligLabel) ligLabel.textContent = isLBDD ? 'Reference Molecule' : 'Ligand';

    // Update sidebar visibility rules
    _updateSidebarVisibility();
}

function _updateSidebarVisibility() {
    const isLBDD = state.workflowType === 'lbdd';
    // SBDD needs both; LBDD only needs a ligand (molecule).
    // De novo pocket-only / scratch modes also count as ready.
    const standardReady = isLBDD ? !!state.ligandData : !!(state.proteinData && state.ligandData);
    const denovoReady = isLBDD
        ? (state.lbddScratchMode && state.numHeavyAtoms > 0 && !!state.jobId)
        : (state.pocketOnlyMode && !!state.proteinData && state.numHeavyAtoms > 0 && !!state.jobId);
    const ready = standardReady || denovoReady;
    const sections = document.querySelectorAll('.panel-left .panel-section.needs-inputs');
    sections.forEach(s => {
        if (ready) {
            s.classList.remove('hidden');
        } else {
            s.classList.add('hidden');
        }
    });
    // Re-hide workflow-specific sections that shouldn't be visible
    // (avoids infinite recursion with _applyWorkflowMode)
    if (ready) {
        document.querySelectorAll('.panel-left .panel-section.needs-inputs.sbdd-only').forEach(el => {
            if (isLBDD) el.classList.add('hidden');
        });
        document.querySelectorAll('.panel-left .panel-section.needs-inputs.lbdd-only').forEach(el => {
            if (!isLBDD) el.classList.add('hidden');
        });
    }
}

async function startGeneration() {
    // Standard path requires ligandData; de novo pocket-only / scratch path
    // only needs numHeavyAtoms (and protein for SBDD).
    const hasDenovoInputs = state.pocketOnlyMode || state.lbddScratchMode;
    if (!state.jobId) return;
    if (!state.ligandData && !hasDenovoInputs) return;
    // SBDD also requires protein
    if (state.workflowType !== 'lbdd' && !state.proteinData) return;
    if (state._generating) return;
    state._generating = true;

    const btn = document.getElementById('generate-btn');
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">✕</span> Cancel Generation';
    btn.onclick = cancelGeneration;
    setBadge('loading', 'Allocating…');

    // Freeze sidebar inputs (only cancel button stays interactive)
    _freezeSidebar(true);

    // Clear previous metrics/mol-detail on new generation
    const mp = document.getElementById('metrics-panel');
    if (mp) mp.classList.add('hidden');
    const ml = document.getElementById('metrics-log');
    if (ml) ml.innerHTML = '';
    closeMol2DOverlay();

    const progressContainer = document.getElementById('progress-container');
    progressContainer.classList.remove('hidden');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    progressFill.style.width = '5%';
    progressText.textContent = 'Submitting…';

    const nSamples = parseInt(document.getElementById('n-samples').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const steps = parseInt(document.getElementById('integration-steps').value);
    const cutoff = parseFloat(document.getElementById('pocket-cutoff').value);
    const isLBDD = state.workflowType === 'lbdd';

    // Build mode-specific payload — LBDD supports all the same gen modes as SBDD
    const mode = state.genMode;
    const payload = {
        job_id: state.jobId,
        workflow_type: state.workflowType,
        protein_path: isLBDD ? null : (state.proteinData?.filename || null),
        ligand_path: state.ligandData ? state.ligandData.filename : null,
        gen_mode: mode,
        fixed_atoms: (mode === 'substructure_inpainting')
            ? Array.from(state.fixedAtoms).map(
                idx => state.heavyAtomMap.get(idx)
            ).filter(idx => idx !== undefined)
            : [],
        n_samples: nSamples,
        batch_size: batchSize,
        integration_steps: steps,
        pocket_cutoff: isLBDD ? 6.0 : cutoff,

        // De novo: num_heavy_atoms for placeholder ligand generation
        num_heavy_atoms: state.numHeavyAtoms || null,

        // Noise
        coord_noise_scale: document.getElementById('add-noise')?.checked
            ? parseFloat(document.getElementById('noise-scale')?.value || '0.1')
            : 0.0,

        // Post-processing options
        sample_mol_sizes: document.getElementById('sample-mol-sizes')?.checked || false,
        filter_valid_unique: document.getElementById('filter-valid-unique')?.checked || false,
        filter_cond_substructure: document.getElementById('filter-cond-substructure')?.checked || false,
        filter_diversity: document.getElementById('filter-diversity')?.checked || false,
        diversity_threshold: parseFloat(document.getElementById('diversity-threshold')?.value || '0.9'),
        optimize_gen_ligs: (!isLBDD && document.getElementById('optimize-gen-ligs')?.checked) || false,
        optimize_gen_ligs_hs: (!isLBDD && document.getElementById('optimize-gen-ligs-hs')?.checked) || false,
        calculate_pb_valid: (!isLBDD && document.getElementById('calculate-pb-valid')?.checked) || false,
        filter_pb_valid: (!isLBDD && document.getElementById('filter-pb-valid')?.checked) || false,
        calculate_strain_energies: document.getElementById('calculate-strain')?.checked || false,

        // Anisotropic prior (applicable for conditional modes)
        anisotropic_prior: state.anisotropicPrior,
        ref_ligand_com_prior: state.refLigandComPrior,

        // Core growing ring system selection
        ring_system_index: state._ringSystemIndex || 0,

        // Property & ADMET filters
        property_filter: buildPropertyFilterPayload(),
        adme_filter: buildAdmeFilterPayload(),
    };

    // LBDD-specific fields
    if (isLBDD) {
        payload.optimize_method = document.getElementById('lbdd-optimize-method')?.value || 'none';
    }

    // Fragment growing extras
    if (mode === 'fragment_growing') {
        payload.grow_size = parseInt(document.getElementById('grow-size').value) || 5;

        // Priority: user-placed center coordinates > uploaded prior center file
        if (state._priorPlacedCenter) {
            // Send the exact 3D coordinates the user positioned interactively
            payload.prior_center_coords = {
                x: state._priorPlacedCenter.x,
                y: state._priorPlacedCenter.y,
                z: state._priorPlacedCenter.z,
            };
        } else if (state._uploadedPriorCenterFilename) {
            // Use the already-uploaded prior center filename (uploaded in onPriorCenterFileChange)
            payload.prior_center_filename = state._uploadedPriorCenterFilename;
        } else {
            // Fallback: upload now if a file is selected but wasn't uploaded yet
            const priorFileInput = document.getElementById('prior-center-file');
            if (priorFileInput && priorFileInput.files.length > 0) {
                const formData = new FormData();
                formData.append('file', priorFileInput.files[0]);
                const upResp = await fetch(`${API_BASE}/upload_prior_center/${state.jobId}`, {
                    method: 'POST',
                    body: formData,
                });
                if (upResp.ok) {
                    const upData = await upResp.json();
                    payload.prior_center_filename = upData.filename;
                    state._uploadedPriorCenterFilename = upData.filename;
                }
            }
        }
    }

    try {
        // 1. Submit generation request — this triggers GPU allocation
        _updateDeviceBadge('ALLOCATING…', 'loading');
        progressText.textContent = 'Allocating GPU…';
        progressFill.style.width = '2%';

        const resp = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(await resp.text());

        // Server accepted the job — switch text to Starting while we wait
        // for the first poll to determine the actual phase.
        progressText.textContent = 'Starting…';
        progressFill.style.width = '5%';

        // 2. Poll for completion — progress text is driven by job status
        //    (allocating_gpu → Allocating GPU…, generating → Generating…)
        const data = await pollGeneration(state.jobId, progressFill, progressText);

        progressFill.style.width = '100%';
        progressText.textContent = `Done! ${data.n_generated} ligands in ${data.elapsed_time}s`;

        state.generatedResults = data.results;
        await renderResults(data);

        const modeLabel = data.mode === 'flowr' ? 'FLOWR' : (data.mode || 'Mock');
        setBadge('success', `${data.n_generated} generated`);
        showToast(`Generated ${data.n_generated} ligands in ${data.elapsed_time}s (${modeLabel})`, 'success');

        document.getElementById('gen-summary').classList.remove('hidden');
        document.getElementById('summary-count').textContent = data.n_generated;
        document.getElementById('summary-time').textContent = `${data.elapsed_time}s`;
        document.getElementById('summary-mode').textContent = modeLabel;
        document.getElementById('summary-gen-mode').textContent =
            MODE_LABELS[mode] || mode;
        const fixedRow = document.getElementById('summary-fixed-row');
        if (mode === 'substructure_inpainting') {
            fixedRow.classList.remove('hidden');
            document.getElementById('summary-fixed').textContent = state.fixedAtoms.size;
        } else {
            fixedRow.classList.add('hidden');
        }

        if (data.error) {
            showToast(`Note: ${data.error}`, 'error');
        }

    } catch (e) {
        console.error(e);
        progressFill.style.width = '0%';
        progressText.textContent = 'Generation failed';
        progressContainer.classList.add('hidden');
        setBadge('error', 'Failed');
        showToast('Generation failed: ' + e.message, 'error');
    }

    // GPU released — revert badge to idle
    _updateDeviceBadge('NO GPU', 'idle');

    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">▶</span> Generate Ligands';
    btn.onclick = startGeneration;
    state._generating = false;

    // Unfreeze sidebar
    _freezeSidebar(false);
}

/**
 * Cancel a running generation by notifying the server.
 * The pollGeneration loop will pick up the 'cancelled' status.
 */
async function cancelGeneration() {
    if (!state.jobId || !state._generating) return;
    const btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span> Cancelling…';
    try {
        await fetch(`${API_BASE}/cancel/${state.jobId}`, { method: 'POST' });
    } catch (e) {
        console.warn('Cancel request failed:', e);
    }
}

/**
 * Freeze / unfreeze sidebar inputs during generation.
 * The cancel button and 3D viewer remain interactive.
 */
function _freezeSidebar(freeze) {
    const panel = document.querySelector('.panel');
    if (!panel) return;
    const inputs = panel.querySelectorAll(
        'input:not(#generate-btn), select, button:not(#generate-btn), .slider'
    );
    inputs.forEach(el => {
        if (freeze) {
            el.dataset.wasFrozen = el.disabled ? '1' : '0';
            el.disabled = true;
        } else {
            // Restore only those that were not already disabled
            if (el.dataset.wasFrozen === '0') el.disabled = false;
            delete el.dataset.wasFrozen;
        }
    });
    if (freeze) {
        panel.classList.add('sidebar-frozen');
    } else {
        panel.classList.remove('sidebar-frozen');
    }
}

/**
 * Poll /job/{id} until status is 'completed', 'failed', or 'cancelled'.
 * Uses recursive setTimeout to avoid request stacking on slow networks.
 */
function pollGeneration(jobId, progressFill, progressText) {
    const btn = document.getElementById('generate-btn');
    return new Promise((resolve, reject) => {
        let elapsed = 0;
        let currentPhase = null; // determined by first poll response
        let _allocationToast = null; // ref to dismiss early on fast transition

        const poll = async () => {
            elapsed += 1;
            try {
                const resp = await fetch(`${API_BASE}/job/${jobId}`);
                const data = await resp.json();

                // First poll: determine initial phase
                if (currentPhase === null) {
                    currentPhase = data.status;
                    if (currentPhase === 'generating') {
                        // No real allocation phase (local mode) —
                        // combine into one toast to avoid overlap.
                        try {
                            const msResp = await fetch(`${API_BASE}/model-status`);
                            const msData = await msResp.json();
                            const device = (msData.device || 'GPU').toUpperCase();
                            _updateDeviceBadge(device, 'loaded');
                            showToast(`GPU allocated on ${device} — starting generation…`, 'success');
                        } catch (_) {
                            _updateDeviceBadge('GPU', 'loaded');
                            showToast('GPU allocated — starting generation…', 'success');
                        }
                    } else if (currentPhase === 'allocating_gpu') {
                        // Real allocation phase (SLURM) — show a persistent
                        // toast that we'll dismiss when allocation finishes.
                        _allocationToast = showToast('Allocating GPU…', 'success');
                    }
                }

                // Detect phase transition: allocating_gpu → generating
                // (fires when there was a real allocation phase, e.g. SLURM)
                if (currentPhase === 'allocating_gpu' && data.status === 'generating') {
                    currentPhase = 'generating';
                    elapsed = 0; // reset timer for generation phase

                    // Dismiss the allocation toast early, then show the new one
                    if (_allocationToast) { _allocationToast.dismiss(); _allocationToast = null; }

                    try {
                        const msResp = await fetch(`${API_BASE}/model-status`);
                        const msData = await msResp.json();
                        const device = (msData.device || 'GPU').toUpperCase();
                        _updateDeviceBadge(device, 'loaded');
                        showToast(`GPU allocated on ${device} — starting generation…`, 'success');
                    } catch (_) {
                        _updateDeviceBadge('GPU', 'loaded');
                        showToast('GPU allocated — starting generation…', 'success');
                    }
                }

                // Update progress bar and text based on job status
                if (data.status === 'allocating_gpu') {
                    progressFill.style.width = '3%';
                    progressText.textContent = `Allocating GPU… (${elapsed}s)`;
                } else if (data.status === 'generating') {
                    const pct = Math.max(data.progress || 0, 10);
                    progressFill.style.width = `${Math.min(pct, 95)}%`;
                    progressText.textContent = `Generating… (${elapsed}s)`;
                }

                if (data.status === 'completed') {
                    resolve(data);
                } else if (data.status === 'cancelled') {
                    reject(new Error('Generation cancelled'));
                } else if (data.status === 'failed') {
                    reject(new Error(data.error || 'Generation failed'));
                } else if (elapsed > 1800) {
                    reject(new Error('Generation timed out after 30 minutes'));
                } else {
                    setTimeout(poll, 1000);
                }
            } catch (e) {
                reject(e);
            }
        };
        setTimeout(poll, 1000);
    });
}

// =========================================================================
// Results
// =========================================================================

async function renderResults(data) {
    const placeholder = document.getElementById('results-placeholder');
    const list = document.getElementById('results-list');

    placeholder.classList.add('hidden');
    list.classList.remove('hidden');
    list.innerHTML = '';

    const clearBtn = document.getElementById('clear-gen-btn');
    if (clearBtn) clearBtn.classList.remove('hidden');

    // Show visualization buttons
    const vizControls = document.getElementById('viz-controls');
    if (vizControls) vizControls.classList.remove('hidden');

    // Track whether optimization was used (affects default H display)
    state.genUsedOptimization = data.used_optimization || false;
    state.genHsVisible = state.genUsedOptimization; // show Hs by default only if optimized

    // Show the hydrogen controls for generated ligands
    const hsControls = document.getElementById('gen-hs-controls');
    if (hsControls) hsControls.classList.remove('hidden');

    data.results.forEach((result, i) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.onclick = () => showGeneratedLigand(i);

        const mw = result.properties?.mol_weight || '–';
        const ha = result.properties?.num_heavy_atoms || '–';

        card.innerHTML = `
            <div class="result-header">
                <span class="result-title">Ligand #${i + 1}</span>
                <span class="result-badge badge badge-idle">${ha} HA</span>
            </div>
            <div class="result-smiles">${escapeHtml(result.smiles || 'N/A')}</div>
            <div class="result-props">
                <span>MW: ${mw}</span>
            </div>
        `;
        list.appendChild(card);
    });

    // Show rank-select controls if any result has affinity data
    const hasAffinity = data.results.some(r => {
        const p = r.properties || {};
        return p.pic50 != null || p.pki != null || p.pkd != null || p.pec50 != null;
    });
    const rankControls = document.getElementById('rank-select-controls');
    if (rankControls) {
        rankControls.classList.toggle('hidden', !hasAffinity);
    }

    // Render metrics if available
    renderMetrics(data.metrics);

    // ── Restore prior cloud (awaited so it finishes before we dim the
    //    reference ligand — avoids race conditions with viewer.render()). ──
    // Substructure_inpainting: derive from selected atoms.
    // All other modes (incl. de novo): prefer worker-returned cloud, then
    //   fall back to server re-fetch.
    if (state.genMode === 'substructure_inpainting') {
        if (state.fixedAtoms.size > 0) {
            _updateSubstructurePriorCloud();
        } else {
            clearPriorCloud();
        }
    } else if (data.prior_cloud) {
        // Worker returned a cloud (all modes including de novo)
        renderPriorCloud(data.prior_cloud);
    } else {
        // Fallback: worker didn't return a cloud — re-fetch from server.
        // Await so the viewer.render() inside finishes before we dim the ref.
        await _fetchAndRenderPriorCloudPreview();
    }

    // Update save section
    updateSaveSection();

    // ── Show first generated ligand (this also dims the reference) ──
    if (data.results.length > 0) showGeneratedLigand(0);
}

function showGeneratedLigand(idx) {
    if (idx < 0 || idx >= state.generatedResults.length) return;

    state.activeResultIdx = idx;
    document.querySelectorAll('.result-card').forEach((card, i) => {
        card.classList.toggle('active', i === idx);
    });

    const result = state.generatedResults[idx];

    // Remove previous generated model
    if (state.generatedModel !== null) {
        state.viewer.removeModel(state.generatedModel);
        state.generatedModel = null;
    }

    // Dim the reference ligand (respecting visibility toggle)
    if (state.ligandModel) {
        if (state.refLigandVisible) {
            state.viewer.setStyle(
                { model: state.ligandModel },
                {
                    stick: { radius: 0.1, colorscheme: 'Jmol', opacity: 0.3 },
                    sphere: { radius: 0.15, colorscheme: 'Jmol', opacity: 0.3 },
                }
            );
        } else {
            state.viewer.setStyle({ model: state.ligandModel }, {});
        }
        // Restore clickable after setStyle (which resets intersection shapes)
        reapplyClickable();
    }

    // Add generated ligand – pick the correct SDF based on H visibility preference
    const sdfToUse = _getGenSdf(result);
    if (sdfToUse) {
        state.generatedModel = state.viewer.addModel(sdfToUse, 'sdf');
        state.viewer.setStyle(
            { model: state.generatedModel },
            {
                stick: { radius: 0.18, colorscheme: 'Jmol' },
                sphere: { radius: 0.35, colorscheme: 'Jmol' },
            }
        );
    }

    // Protein visibility
    if (state.proteinModel && state.viewMode !== 'ligand') {
        applyProteinStyle();
    }

    state.viewer.render();

    if (state.generatedModel) {
        state.viewer.zoomTo({ model: state.generatedModel }, 300);
        state.viewer.render();
    }

    // Show 2D structure overlay (hide Interactions tab for LBDD)
    showMol2D(result.smiles, `Ligand #${idx + 1}`, result.properties);
    const tabInteractionsBtn = document.getElementById('tab-interactions');
    if (tabInteractionsBtn) {
        tabInteractionsBtn.classList.toggle('hidden', state.workflowType === 'lbdd');
    }

    // Reset interaction diagram cache so it reloads for new ligand
    const intCanvas = document.getElementById('mol-2d-interaction');
    if (intCanvas) intCanvas.dataset.loaded = '';

    // Reset properties cache so it reloads for new ligand
    const propsCanvas = document.getElementById('mol-2d-properties');
    if (propsCanvas) propsCanvas.dataset.loaded = '';

    // Clear 3D interactions if showing (will re-show for new ligand if toggled)
    // Only for SBDD — LBDD has no protein so no protein-ligand interactions.
    if (state.showingInteractions && state.workflowType !== 'lbdd') {
        clearInteractions3D();
        // Re-show interactions for the newly selected ligand
        showInteractions3D('both');
    } else if (state.showingInteractions && state.workflowType === 'lbdd') {
        clearInteractions3D();
        state.showingInteractions = false;
    }
}

// =========================================================================
// UI Helpers
// =========================================================================

function setBadge(type, text) {
    const badge = document.getElementById('status-badge');
    badge.className = `badge badge-${type}`;
    badge.textContent = text;
}

/**
 * Show a toast notification at the bottom of the screen.
 * Multiple simultaneous toasts stack upward automatically.
 * Returns the toast element (has a .dismiss() helper to remove early).
 */
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.dataset.toastId = String(Date.now()) + Math.random();
    document.body.appendChild(toast);

    // Stack above any existing toasts
    _repositionToasts();

    const removeToast = () => {
        toast.remove();
        _repositionToasts();
    };
    toast.dismiss = removeToast;
    const timer = setTimeout(removeToast, 3500);
    toast._timer = timer;
    return toast;
}

/** Reposition all visible toasts so they stack upward from the bottom. */
function _repositionToasts() {
    const toasts = Array.from(document.querySelectorAll('.toast'));
    const BASE_BOTTOM = 50;
    const GAP = 8;
    let offset = BASE_BOTTOM;
    // Oldest at the bottom, newest stacked above
    toasts.forEach(t => {
        t.style.bottom = offset + 'px';
        offset += t.offsetHeight + GAP;
    });
}

// =========================================================================
// Tooltips (custom popup from data-tooltip attribute)
// =========================================================================

function initTooltips() {
    document.querySelectorAll('.help-tooltip[data-tooltip]').forEach(el => {
        let popup = null;
        const show = () => {
            if (popup) return;
            popup = document.createElement('div');
            popup.className = 'tooltip-popup';
            popup.textContent = el.getAttribute('data-tooltip');
            document.body.appendChild(popup);
            // Position using fixed coords relative to viewport
            const rect = el.getBoundingClientRect();
            const pw = popup.offsetWidth;
            const ph = popup.offsetHeight;
            // Place above the ? icon
            let top = rect.top - ph - 10;
            let left = rect.left + rect.width / 2 - pw / 2;
            // Clamp to viewport edges
            if (left < 8) left = 8;
            if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
            if (top < 8) { top = rect.bottom + 8; } // flip below if no room above
            popup.style.top = top + 'px';
            popup.style.left = left + 'px';
            // Adjust arrow to point at the ? icon
            const arrowLeft = Math.max(10, Math.min(pw - 10, rect.left + rect.width / 2 - left));
            popup.style.setProperty('--arrow-left', arrowLeft + 'px');
        };
        const hide = () => {
            if (popup) { popup.remove(); popup = null; }
        };
        el.addEventListener('mouseenter', show);
        el.addEventListener('mouseleave', hide);
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            popup ? hide() : show();
        });
    });
}

// =========================================================================
// Filter-diversity threshold toggle
// =========================================================================

function initFilterDiversityToggle() {
    const cb = document.getElementById('filter-diversity');
    const group = document.getElementById('diversity-threshold-group');
    if (cb && group) {
        cb.addEventListener('change', () => {
            group.classList.toggle('hidden', !cb.checked);
        });
    }
}

// =========================================================================
// Property filter panel toggle & population
// =========================================================================

/** All properties available for filtering, grouped by type. */
const PROPERTY_DEFS = [
    // Continuous
    { name: 'MolWt', label: 'Mol. Weight', type: 'cont', step: 1 },
    { name: 'LogP', label: 'LogP', type: 'cont', step: 0.1 },
    { name: 'TPSA', label: 'TPSA', type: 'cont', step: 1 },
    { name: 'FractionCSP3', label: 'Frac. CSP3', type: 'cont', step: 0.01 },
    // Discrete
    { name: 'NumHAcceptors', label: 'H Acceptors', type: 'disc', step: 1 },
    { name: 'NumHDonors', label: 'H Donors', type: 'disc', step: 1 },
    { name: 'NumRotatableBonds', label: 'Rot. Bonds', type: 'disc', step: 1 },
    { name: 'NumHeavyAtoms', label: 'Heavy Atoms', type: 'disc', step: 1 },
    { name: 'RingCount', label: 'Ring Count', type: 'disc', step: 1 },
    { name: 'NumAromaticRings', label: 'Arom. Rings', type: 'disc', step: 1 },
    { name: 'NumChiralCenters', label: 'Chiral Ctrs', type: 'disc', step: 1 },
    { name: 'NumHeteroatoms', label: 'Heteroatoms', type: 'disc', step: 1 },
    { name: 'NumAliphaticRings', label: 'Aliph. Rings', type: 'disc', step: 1 },
    { name: 'NumAromaticCarbocycles', label: 'Arom. Carbo.', type: 'disc', step: 1 },
    { name: 'NumAromaticHeterocycles', label: 'Arom. Hetero.', type: 'disc', step: 1 },
    { name: 'NumSaturatedCarbocycles', label: 'Sat. Carbo.', type: 'disc', step: 1 },
    { name: 'NumSaturatedHeterocycles', label: 'Sat. Hetero.', type: 'disc', step: 1 },
    { name: 'NumAliphaticCarbocycles', label: 'Aliph. Carbo.', type: 'disc', step: 1 },
    { name: 'NumAliphaticHeterocycles', label: 'Aliph. Hetero.', type: 'disc', step: 1 },
];

function initPropertyFilterToggle() {
    const cb = document.getElementById('filter-properties');
    const panel = document.getElementById('property-filter-panel');
    if (cb && panel) {
        cb.addEventListener('change', () => {
            panel.classList.toggle('hidden', !cb.checked);
        });
    }
}

/**
 * Populate the property filter panel with reference-based default ranges.
 * Called after ligand upload when ref_properties are available.
 */
function populatePropertyFilterPanel(refProps) {
    state.refProperties = refProps || {};
    const list = document.getElementById('property-filter-list');
    if (!list) return;
    list.innerHTML = '';

    for (const def of PROPERTY_DEFS) {
        const refVal = refProps[def.name];
        // Compute default ±20% range
        let defaultMin, defaultMax;
        if (refVal == null) {
            defaultMin = 0;
            defaultMax = 1;
        } else if (def.type === 'disc') {
            const delta = Math.max(1, Math.round(Math.abs(refVal) * 0.2));
            defaultMin = Math.max(0, refVal - delta);
            defaultMax = refVal + delta;
        } else {
            // continuous
            const delta = Math.abs(refVal) * 0.2;
            const minDelta = def.step * 2; // ensure minimum range
            const effectiveDelta = Math.max(delta, minDelta);
            defaultMin = Math.round((refVal - effectiveDelta) / def.step) * def.step;
            defaultMax = Math.round((refVal + effectiveDelta) / def.step) * def.step;
            // Round to reasonable precision
            defaultMin = parseFloat(defaultMin.toFixed(3));
            defaultMax = parseFloat(defaultMax.toFixed(3));
        }

        const row = document.createElement('div');
        row.className = 'prop-filter-row';
        row.dataset.propName = def.name;
        row.innerHTML = `
            <input type="checkbox" class="prop-cb" title="Enable ${def.label} filter">
            <span class="prop-name" title="${def.name} (ref: ${refVal != null ? refVal : 'N/A'})">${def.label}</span>
            <div class="prop-range-inputs">
                <input type="number" class="input-number prop-min" step="${def.step}" value="${defaultMin}" title="Minimum">
                <span class="prop-range-sep">–</span>
                <input type="number" class="input-number prop-max" step="${def.step}" value="${defaultMax}" title="Maximum">
            </div>
        `;

        // Toggle active state for visual feedback
        const cb = row.querySelector('.prop-cb');
        cb.addEventListener('change', () => {
            row.classList.toggle('active', cb.checked);
        });

        // Clicking the label toggles the checkbox
        const nameSpan = row.querySelector('.prop-name');
        nameSpan.addEventListener('click', () => {
            cb.checked = !cb.checked;
            cb.dispatchEvent(new Event('change'));
        });

        list.appendChild(row);
    }
}

/**
 * Build the property_filter payload array from active property filter rows.
 * Returns null if no properties are selected, otherwise array of {name, min, max}.
 */
function buildPropertyFilterPayload() {
    const cb = document.getElementById('filter-properties');
    if (!cb || !cb.checked) return null;

    const rows = document.querySelectorAll('#property-filter-list .prop-filter-row');
    const filters = [];
    for (const row of rows) {
        const propCb = row.querySelector('.prop-cb');
        if (!propCb || !propCb.checked) continue;
        const name = row.dataset.propName;
        const minVal = row.querySelector('.prop-min')?.value;
        const maxVal = row.querySelector('.prop-max')?.value;
        filters.push({
            name,
            min: minVal !== '' ? parseFloat(minVal) : null,
            max: maxVal !== '' ? parseFloat(maxVal) : null,
        });
    }
    return filters.length > 0 ? filters : null;
}

// =========================================================================
// ADMET filter panel toggle & management
// =========================================================================

let _admeEntryCounter = 1;

function initAdmeFilterToggle() {
    const cb = document.getElementById('filter-adme');
    const panel = document.getElementById('adme-filter-panel');
    if (cb && panel) {
        cb.addEventListener('change', () => {
            panel.classList.toggle('hidden', !cb.checked);
        });
    }
    // Wire up model file input on the initial entry
    _wireAdmeFileInput(document.getElementById('adme-entry-0'));
}

function addAdmeEntry() {
    const container = document.getElementById('adme-filter-entries');
    if (!container) return;
    const idx = _admeEntryCounter++;
    const entry = document.createElement('div');
    entry.className = 'adme-entry';
    entry.id = `adme-entry-${idx}`;
    entry.innerHTML = `
        <div class="adme-entry-row">
            <input type="text" class="input-text adme-name" placeholder="Property name"
                title="Name for this ADMET property (e.g. clearance, hERG)">
            <label class="btn btn-sm btn-outline adme-upload-btn">
                <span>📁 Model</span>
                <input type="file" class="adme-model-file" accept=".pt,.pth,.pkl,.joblib,.ckpt,.onnx,.bin" hidden>
            </label>
            <button type="button" class="btn btn-sm btn-outline" onclick="this.closest('.adme-entry').remove()"
                title="Remove this filter">✕</button>
        </div>
        <div class="adme-entry-row">
            <label class="adme-range-label">Min</label>
            <input type="number" class="input-number adme-min" step="any" placeholder="—">
            <label class="adme-range-label">Max</label>
            <input type="number" class="input-number adme-max" step="any" placeholder="—">
        </div>
        <span class="adme-model-name" title="">No model selected</span>
    `;
    container.appendChild(entry);
    _wireAdmeFileInput(entry);
}

function _wireAdmeFileInput(entry) {
    if (!entry) return;
    const fileInput = entry.querySelector('.adme-model-file');
    const nameSpan = entry.querySelector('.adme-model-name');
    if (!fileInput) return;
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;
        if (!state.jobId) {
            showToast('Upload a ligand first to create a job', 'error');
            return;
        }
        nameSpan.textContent = 'Uploading…';
        try {
            const formData = new FormData();
            formData.append('file', file);
            const resp = await fetch(`${API_BASE}/upload/adme-model/${state.jobId}`, {
                method: 'POST',
                body: formData,
            });
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();
            // Store the model URL on the entry element for payload construction
            entry.dataset.modelUrl = data.model_url || '';
            entry.dataset.modelFilename = data.filename || file.name;
            nameSpan.textContent = data.filename || file.name;
            nameSpan.title = data.model_url || '';
        } catch (err) {
            nameSpan.textContent = 'Upload failed';
            console.error('ADMET model upload failed:', err);
            showToast('ADMET model upload failed', 'error');
        }
    });
}

/**
 * Build the adme_filter payload array from ADMET filter entries.
 * Returns null if ADMET filtering is disabled or no entries configured.
 */
function buildAdmeFilterPayload() {
    const cb = document.getElementById('filter-adme');
    if (!cb || !cb.checked) return null;

    const entries = document.querySelectorAll('#adme-filter-entries .adme-entry');
    const filters = [];
    for (const entry of entries) {
        const name = entry.querySelector('.adme-name')?.value?.trim();
        if (!name) continue;
        const minVal = entry.querySelector('.adme-min')?.value;
        const maxVal = entry.querySelector('.adme-max')?.value;
        const modelFile = entry.dataset.modelFilename || null;
        filters.push({
            name,
            min: minVal !== '' ? parseFloat(minVal) : null,
            max: maxVal !== '' ? parseFloat(maxVal) : null,
            model_file: modelFile,
            model_url: entry.dataset.modelUrl || null,
        });
    }
    return filters.length > 0 ? filters : null;
}

// =========================================================================
// Add-noise slider toggle
// =========================================================================

function initAddNoiseToggle() {
    const cb = document.getElementById('add-noise');
    const group = document.getElementById('noise-scale-group');
    if (cb && group) {
        cb.addEventListener('change', () => {
            group.classList.toggle('hidden', !cb.checked);
        });
    }
}

// =========================================================================
// Prior center file pick handler
// =========================================================================

async function onPriorCenterFileChange(input) {
    const nameSpan = document.getElementById('prior-center-filename');
    if (nameSpan) {
        nameSpan.textContent = input.files.length > 0 ? input.files[0].name : '';
    }

    // Upload the file immediately so the server can compute a prior-cloud preview
    if (input.files.length > 0 && state.jobId) {
        const formData = new FormData();
        formData.append('file', input.files[0]);
        try {
            const resp = await fetch(`${API_BASE}/upload_prior_center/${state.jobId}`, {
                method: 'POST',
                body: formData,
            });
            if (resp.ok) {
                const data = await resp.json();
                // Store the uploaded filename for later use in generation payload
                state._uploadedPriorCenterFilename = data.filename;
                // Clear any previously placed center (file takes precedence)
                state._priorPlacedCenter = null;
                _updatePriorCoordsDisplay(null);
                const resetBtn = document.getElementById('reset-prior-pos-btn');
                if (resetBtn) resetBtn.classList.add('hidden');
                if (data.prior_cloud) {
                    renderPriorCloud(data.prior_cloud);
                }
            }
        } catch (e) {
            console.warn('Prior center upload/preview failed:', e);
        }
    } else if (input.files.length === 0) {
        // File was cleared — remove prior cloud & re-fetch without prior center
        state._uploadedPriorCenterFilename = null;
        _fetchAndRenderPriorCloudPreview();
    }
}

/**
 * Called when the grow-size input changes.
 * Re-fetches the prior-cloud preview with the new size and updates the legend.
 * Debounced to avoid server request per keystroke/spinner click.
 */
let _growSizeDebounceTimer = null;
function onGrowSizeChange() {
    // Update the inpainting legend text immediately (no server call needed)
    _updateGrowSizeLegend();

    clearTimeout(_growSizeDebounceTimer);
    _growSizeDebounceTimer = setTimeout(() => {
        if (state.genMode !== 'denovo' && state.jobId && state.ligandData) {
            _fetchAndRenderPriorCloudPreview();
        }
    }, 300);
}

/** Update the "To-be-generated" legend row with the current grow-size value. */
function _updateGrowSizeLegend() {
    if (state.genMode !== 'fragment_growing') return;
    const legend = document.getElementById('inpainting-legend');
    if (!legend) return;
    const growSize = parseInt(document.getElementById('grow-size')?.value) || 5;
    // Find the second .legend-row (the "To-be-generated" row)
    const rows = legend.querySelectorAll('.legend-row');
    if (rows.length >= 2) {
        const span = rows[1].querySelector('span:last-child');
        if (span) span.textContent = `To-be-generated (${growSize} atoms)`;
    }
}

/**
 * Fetch the number of ring systems from the server and populate the dropdown.
 */
async function _fetchRingSystemCount() {
    if (!state.jobId || !state.ligandData) return;
    try {
        const resp = await fetch(`${API_BASE}/ring-systems/${state.jobId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        const n = data.num_ring_systems || 0;
        const select = document.getElementById('ring-system-index');
        if (!select) return;
        select.innerHTML = '';
        for (let i = 0; i < Math.max(n, 1); i++) {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `Ring system ${i}`;
            select.appendChild(opt);
        }
        select.value = '0';
        state._ringSystemIndex = 0;
    } catch (e) {
        console.warn('Failed to fetch ring system count:', e);
    }
}

/**
 * Called when the ring system dropdown changes.
 * Re-fetches the inpainting mask with the selected ring system index.
 */
function onRingSystemIndexChange(value) {
    state._ringSystemIndex = parseInt(value) || 0;
    if (state.genMode === 'core_growing' && state.ligandData && state.jobId) {
        fetchAndHighlightInpaintingMask('core_growing');
        // Update prior cloud to match the new ring system's replaced atom count
        _fetchAndRenderPriorCloudPreview();
    }
}

/**
 * Fetch a prior-cloud preview from the server and render it.
 * For fragment_growing, uses the grow_size slider value.
 * For other conditional modes, uses the reference ligand heavy atom count.
 * If the user has manually placed the cloud, re-centers the fetched cloud
 * at the user's chosen position.
 */
async function _fetchAndRenderPriorCloudPreview() {
    // Allow preview when we have a ligand OR are in pocket-only / scratch mode
    if (!state.jobId) return;
    if (!state.ligandData && !state.pocketOnlyMode && !state.lbddScratchMode) return;
    let cloudSize;
    if (state.genMode === 'fragment_growing') {
        cloudSize = parseInt(document.getElementById('grow-size')?.value) || 5;
    } else if (state.genMode === 'substructure_inpainting') {
        // For substructure mode: cloud size = number of atoms being replaced
        cloudSize = Math.max(1, state.fixedAtoms.size);
    } else {
        // For de novo / other modes use state.numHeavyAtoms (may come from
        // reference ligand or manual entry) or fall back to ligandData.
        cloudSize = state.numHeavyAtoms
            || state.ligandData?.num_heavy_atoms
            || parseInt(document.getElementById('grow-size')?.value) || 20;
    }
    const pocketCutoff = parseFloat(document.getElementById('pocket-cutoff')?.value) || 6;
    const genMode = state.genMode || 'fragment_growing';
    const ringIdx = state._ringSystemIndex || 0;
    const placedCenter = state._priorPlacedCenter;
    try {
        const resp = await fetch(`${API_BASE}/prior-cloud-preview/${state.jobId}?grow_size=${cloudSize}&pocket_cutoff=${pocketCutoff}&anisotropic=${state.anisotropicPrior}&gen_mode=${genMode}&ring_system_index=${ringIdx}&ref_ligand_com_prior=${state.refLigandComPrior}`);
        if (resp.ok) {
            const cloud = await resp.json();

            // If user has manually placed the cloud, shift to their chosen center
            if (placedCenter) {
                const dx = placedCenter.x - cloud.center.x;
                const dy = placedCenter.y - cloud.center.y;
                const dz = placedCenter.z - cloud.center.z;
                cloud.center = { ...placedCenter };
                cloud.points = cloud.points.map(p => ({
                    x: p.x + dx,
                    y: p.y + dy,
                    z: p.z + dz,
                }));
            }

            renderPriorCloud(cloud);

            // Restore placement UI state if center was placed
            if (placedCenter) {
                _updatePriorCoordsDisplay(placedCenter);
                const resetBtn = document.getElementById('reset-prior-pos-btn');
                if (resetBtn) resetBtn.classList.remove('hidden');
            }

        } else {
            clearPriorCloud();
        }
    } catch (e) {
        console.warn('Prior cloud preview fetch failed:', e);
    }
}

// =========================================================================
// Resizable right sidebar
// =========================================================================

function initResizeHandle() {
    // Right panel resize handle
    const handle = document.getElementById('panel-resize-handle');
    const panel = document.getElementById('results-panel');
    if (handle && panel) {
        let startX, startWidth;
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startX = e.clientX;
            startWidth = panel.offsetWidth;
            handle.classList.add('active');

            const onMove = (ev) => {
                const delta = startX - ev.clientX;
                const newW = Math.max(280, Math.min(800, startWidth + delta));
                panel.style.width = newW + 'px';
                panel.style.minWidth = newW + 'px';
            };
            const onUp = () => {
                handle.classList.remove('active');
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    }

    // Left panel resize handle
    const handleLeft = document.getElementById('panel-resize-handle-left');
    const panelLeft = document.querySelector('.panel:not(.panel-right)');
    if (handleLeft && panelLeft) {
        let startX, startWidth;
        handleLeft.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startX = e.clientX;
            startWidth = panelLeft.offsetWidth;
            handleLeft.classList.add('active');

            const onMove = (ev) => {
                const delta = ev.clientX - startX;
                const newW = Math.max(280, Math.min(700, startWidth + delta));
                panelLeft.style.width = newW + 'px';
                panelLeft.style.minWidth = newW + 'px';
            };
            const onUp = () => {
                handleLeft.classList.remove('active');
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    }
}

// =========================================================================
// Metrics rendering
// =========================================================================

function renderMetrics(metrics) {
    const panel = document.getElementById('metrics-panel');
    const log = document.getElementById('metrics-log');
    if (!panel || !log) return;

    if (!metrics || metrics.length === 0) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');
    log.innerHTML = metrics.map(m =>
        `<div class="metric-line">${escapeHtml(m)}</div>`
    ).join('');
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

// =========================================================================
// 2D molecule structure overlay (using RDKit.js client-side rendering)
// =========================================================================

function showMol2D(smiles, name, properties) {
    const overlay = document.getElementById('mol-2d-overlay');
    const canvas = document.getElementById('mol-2d-canvas');
    const smilesEl = document.getElementById('mol-2d-smiles');
    const propsEl = document.getElementById('mol-2d-props');
    const nameEl = document.getElementById('mol-2d-name');
    if (!overlay || !canvas) return;

    if (!smiles) {
        overlay.classList.add('hidden');
        return;
    }

    overlay.classList.remove('hidden');
    if (nameEl) nameEl.textContent = name || '2D Structure';

    // Reset to structure tab when showing a new molecule
    switchMol2DTab('structure');

    // Render using RDKit.js (client-side, instant)
    if (RDKitModule) {
        let mol = null;
        try {
            mol = RDKitModule.get_mol(smiles);
            if (mol) {
                const svg = mol.get_svg_with_highlights(JSON.stringify({ width: 380, height: 260 }));
                canvas.innerHTML = svg;
            } else {
                canvas.innerHTML = '<span class="muted-text">Invalid molecule</span>';
            }
        } catch (e) {
            _fallbackServerRender(smiles, canvas);
        } finally {
            if (mol) try { mol.delete(); } catch (_) { }
        }
    } else {
        _fallbackServerRender(smiles, canvas);
    }

    // SMILES
    if (smilesEl) smilesEl.textContent = smiles;

    // Properties
    if (propsEl && properties) {
        const mw = properties.mol_weight || '–';
        const tpsa = properties.tpsa != null ? properties.tpsa : '–';
        const logp = properties.logp != null ? properties.logp : '–';
        propsEl.innerHTML = `<span>MW ${mw}</span><span>TPSA ${tpsa}</span><span>logP ${logp}</span>`;
    } else if (propsEl) {
        propsEl.innerHTML = '';
    }
}

function _fallbackServerRender(smiles, container) {
    container.innerHTML = '<span class="muted-text">Loading…</span>';
    fetch(`${API_BASE}/mol-image?smiles=${encodeURIComponent(smiles)}&width=380&height=260`)
        .then(resp => {
            if (!resp.ok) throw new Error('Failed');
            return resp.text();
        })
        .then(svg => { container.innerHTML = svg; })
        .catch(() => { container.innerHTML = '<span class="muted-text">Could not render</span>'; });
}

function closeMol2DOverlay() {
    const overlay = document.getElementById('mol-2d-overlay');
    if (overlay) overlay.classList.add('hidden');
}

// =========================================================================
// 2D Overlay Tab Switching (Structure / Interactions / Properties)
// =========================================================================

function switchMol2DTab(tab) {
    const structureCanvas = document.getElementById('mol-2d-canvas');
    const interactionCanvas = document.getElementById('mol-2d-interaction');
    const propertiesCanvas = document.getElementById('mol-2d-properties');
    const tabStructure = document.getElementById('tab-structure');
    const tabInteractions = document.getElementById('tab-interactions');
    const tabProperties = document.getElementById('tab-properties');
    const overlay = document.getElementById('mol-2d-overlay');

    // Hide all canvases, deactivate all tabs
    structureCanvas.classList.add('hidden');
    interactionCanvas.classList.add('hidden');
    if (propertiesCanvas) propertiesCanvas.classList.add('hidden');
    tabStructure.classList.remove('active');
    tabInteractions.classList.remove('active');
    if (tabProperties) tabProperties.classList.remove('active');
    if (overlay) overlay.classList.remove('interactions-active', 'properties-active');

    if (tab === 'structure') {
        structureCanvas.classList.remove('hidden');
        tabStructure.classList.add('active');
    } else if (tab === 'interactions') {
        interactionCanvas.classList.remove('hidden');
        tabInteractions.classList.add('active');
        if (overlay) overlay.classList.add('interactions-active');
        _loadInteractionDiagram();
    } else if (tab === 'properties') {
        if (propertiesCanvas) propertiesCanvas.classList.remove('hidden');
        if (tabProperties) tabProperties.classList.add('active');
        if (overlay) overlay.classList.add('properties-active');
        _loadLigandProperties();
    }
}

function _loadInteractionDiagram() {
    const container = document.getElementById('mol-2d-interaction');
    if (!container || !state.jobId) return;
    if (container.dataset.loaded === 'true' || container.dataset.loaded === 'unavailable') return;

    const ligIdx = state.activeResultIdx >= 0 ? state.activeResultIdx : -1;
    container.innerHTML = '<span class="muted-text">Loading interactions…</span>';

    fetch(`${API_BASE}/interaction-diagram/${state.jobId}?ligand_idx=${ligIdx}`)
        .then(resp => {
            if (resp.status === 501) {
                container.innerHTML = '<span class="muted-text">OpenEye not available for interaction diagrams.</span>';
                container.dataset.loaded = 'unavailable';
                return;
            }
            if (!resp.ok) throw new Error('Failed');
            return resp.text();
        })
        .then(svg => {
            if (svg) {
                container.innerHTML = svg;
                container.dataset.loaded = 'true';
            }
        })
        .catch(() => {
            container.innerHTML = '<span class="muted-text">Could not load interactions</span>';
        });
}

// =========================================================================
// Save Ligands
// =========================================================================

function updateSaveSection() {
    const section = document.getElementById('save-section');
    const checkboxList = document.getElementById('save-checkbox-list');
    if (!section || !checkboxList) return;

    if (state.generatedResults.length === 0) {
        section.classList.add('hidden');
        return;
    }
    section.classList.remove('hidden');

    // Populate checkbox list
    checkboxList.innerHTML = '';
    state.generatedResults.forEach((r, i) => {
        const label = document.createElement('label');
        label.className = 'save-checkbox-item';
        label.innerHTML = `
            <input type="checkbox" class="save-ligand-cb" data-idx="${i}" onchange="updateSaveSelectedCount()">
            <span class="save-cb-label">Ligand #${i + 1}</span>
            <span class="save-cb-smiles">${escapeHtml((r.smiles || 'N/A').slice(0, 30))}</span>
        `;
        checkboxList.appendChild(label);
    });

    updateSaveSelectedCount();
}

function toggleAllSaveCheckboxes(checked) {
    document.querySelectorAll('.save-ligand-cb').forEach(cb => cb.checked = checked);
    updateSaveSelectedCount();
}

function updateSaveSelectedCount() {
    const count = document.querySelectorAll('.save-ligand-cb:checked').length;
    const el = document.getElementById('save-selected-count');
    if (el) el.textContent = `${count} selected`;
}

function getSelectedSaveIndices() {
    const indices = [];
    document.querySelectorAll('.save-ligand-cb:checked').forEach(cb => {
        indices.push(parseInt(cb.dataset.idx));
    });
    return indices;
}

async function saveSelectedLigands() {
    const indices = getSelectedSaveIndices();
    if (indices.length === 0) {
        showToast('No ligands selected for saving', 'error');
        return;
    }
    if (!state.jobId) return;

    try {
        const resp = await fetch(`${API_BASE}/save-selected-ligands/${state.jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ indices }),
        });
        const data = await resp.json();
        showToast(`Saved ${data.saved_count} ligands to output/`, 'success');
        _showSaveStatus(`Saved ${data.saved_count} files to output/`);
    } catch (e) {
        showToast('Error saving ligands: ' + e.message, 'error');
    }
}

async function saveAllLigands() {
    if (!state.jobId) return;
    try {
        const resp = await fetch(`${API_BASE}/save-all-ligands/${state.jobId}`, { method: 'POST' });
        const data = await resp.json();
        showToast(`Saved ${data.saved_count} ligands to output/`, 'success');
        _showSaveStatus(`Saved ${data.saved_count} files to output/`);
    } catch (e) {
        showToast('Error saving ligands: ' + e.message, 'error');
    }
}

function _showSaveStatus(msg) {
    const el = document.getElementById('save-status');
    if (!el) return;
    el.classList.remove('hidden');
    el.textContent = msg;
    setTimeout(() => el.classList.add('hidden'), 4000);
}

// =========================================================================
// 3D Interaction Visualization
// =========================================================================

async function showInteractions3D(target) {
    if (!state.jobId) return;

    const btnRef = document.getElementById('btn-int-ref');

    // Toggle off if already showing
    if (state.showingInteractions === 'both') {
        clearInteractions3D();
        return;
    }

    clearInteractions3D();
    state.showingInteractions = 'both';

    const hasGenSelected = state.activeResultIdx >= 0;
    const hasRefLigand = !!state.ligandData && !state.pocketOnlyMode && !state.lbddScratchMode;
    let totalInteractions = 0;

    // Show interactions for reference ligand (only if it exists and is visible)
    if (hasRefLigand && state.refLigandVisible) {
        try {
            const refResp = await fetch(`${API_BASE}/compute-interactions/${state.jobId}?ligand_idx=-1`);
            if (refResp.ok) {
                const refData = await refResp.json();
                totalInteractions += (refData.interactions || []).length;
                // Only dim ref interactions if a generated ligand is also selected
                renderInteractions3D(refData.interactions, 'ref', hasGenSelected);
            }
        } catch (e) {
            console.warn('Failed to compute ref interactions:', e);
        }
    }

    // Also show for generated ligand if one is selected
    if (hasGenSelected) {
        try {
            const genResp = await fetch(`${API_BASE}/compute-interactions/${state.jobId}?ligand_idx=${state.activeResultIdx}`);
            if (genResp.ok) {
                const genData = await genResp.json();
                totalInteractions += (genData.interactions || []).length;
                renderInteractions3D(genData.interactions, 'gen', false);
            }
        } catch (e) {
            console.warn('Failed to compute gen interactions:', e);
        }
    }

    // Provide user feedback
    if (!hasRefLigand && !hasGenSelected) {
        showToast('Select a generated ligand first to show interactions', 'warning');
        clearInteractions3D();
        return;
    }

    if (btnRef) btnRef.classList.add('active');
    if (totalInteractions > 0) {
        showToast(`Showing ${totalInteractions} interaction(s)`, 'success');
    } else {
        showToast('No interactions found for the selected ligand(s)', 'info');
    }
}

function renderInteractions3D(interactions, source, dimmed) {
    // Collect unique interacting residues for stick display
    const interactingResidues = new Set();
    // dimmed=true → transparent thin ref interactions alongside a generated ligand
    const opacity = dimmed ? 0.3 : 1.0;
    const cylRadius = dimmed ? 0.04 : 0.06;

    interactions.forEach(inter => {
        const color = INTERACTION_COLORS[inter.type] || '#808080';
        const p = inter.protein;
        const l = inter.ligand;

        // Track interacting residues
        interactingResidues.add(`${p.chain}:${p.res_seq}`);

        // Dashed cylinder for the interaction
        const cyl = state.viewer.addCylinder({
            start: { x: p.x, y: p.y, z: p.z },
            end: { x: l.x, y: l.y, z: l.z },
            radius: cylRadius,
            color: color,
            opacity: opacity,
            dashed: true,
            dashLength: 0.15,
            gapLength: 0.08,
            fromCap: 1,
            toCap: 1,
        });
        state.interactionShapes.push(cyl);
    });

    // Show interacting residues as sticks on top of cartoon
    if (state.proteinModel && interactingResidues.size > 0) {
        interactingResidues.forEach(resKey => {
            const [chain, resSeq] = resKey.split(':');
            const sel = { model: state.proteinModel, chain: chain, resi: parseInt(resSeq) };
            state.viewer.addStyle(sel, {
                stick: { radius: 0.12, colorscheme: 'Jmol' },
            });
        });
    }

    state.viewer.render();
}

function clearInteractions3D() {
    state.interactionShapes.forEach(s => {
        try { state.viewer.removeShape(s); } catch (_) { }
    });
    state.interactionShapes = [];
    state.showingInteractions = null;

    // Reset protein style to remove residue sticks
    if (state.proteinModel) {
        applyProteinStyle();
    }

    const btnRef = document.getElementById('btn-int-ref');
    if (btnRef) btnRef.classList.remove('active');

    state.viewer.render();
}

// =========================================================================
// Prior Cloud Visualisation (Fragment Growing)
// =========================================================================

const PRIOR_CLOUD_COLOR = 0xce93d8;  // soft lavender
const PRIOR_CENTER_COLOR = 0xb388ff;  // brighter purple for the centre marker
const PRIOR_CLOUD_OPACITY = 0.35;
const PRIOR_CLOUD_RADIUS = 0.35;
const PRIOR_CENTER_RADIUS = 0.55;

/**
 * Render the prior point cloud in the 3Dmol viewer.
 * Called after generation results arrive for fragment_growing mode.
 *
 * @param {Object} cloud  – {center, points, n_atoms, has_prior_center}
 */
function renderPriorCloud(cloud) {
    _clearPriorCloudSpheres();
    if (!cloud || !cloud.points || cloud.points.length === 0) return;

    state.priorCloud = cloud;
    state.priorCloudVisible = true;

    // Centre marker (slightly larger, brighter)
    const cs = state.viewer.addSphere({
        center: { x: cloud.center.x, y: cloud.center.y, z: cloud.center.z },
        radius: PRIOR_CENTER_RADIUS,
        color: PRIOR_CENTER_COLOR,
        opacity: 0.5,
    });
    state.priorCloudSpheres.push(cs);

    // Cloud points
    cloud.points.forEach(pt => {
        const s = state.viewer.addSphere({
            center: { x: pt.x, y: pt.y, z: pt.z },
            radius: PRIOR_CLOUD_RADIUS,
            color: PRIOR_CLOUD_COLOR,
            opacity: PRIOR_CLOUD_OPACITY,
        });
        state.priorCloudSpheres.push(s);
    });

    // Show the toggle button
    const btn = document.getElementById('toggle-prior-cloud');
    if (btn) {
        btn.classList.remove('hidden');
        btn.classList.add('active');
    }

    state.viewer.render();
}

/** Remove all prior-cloud spheres and hide the toggle button. */
function clearPriorCloud() {
    _exitPriorPlacement();
    _clearPriorCloudSpheres();
    state.priorCloud = null;
    state.priorCloudVisible = false;
    state._priorPlacedCenter = null;
    state._priorDrag = null;
    _updatePriorCoordsDisplay(null);

    const btn = document.getElementById('toggle-prior-cloud');
    if (btn) {
        btn.classList.add('hidden');
        btn.classList.remove('active');
    }
    const resetBtn = document.getElementById('reset-prior-pos-btn');
    if (resetBtn) resetBtn.classList.add('hidden');
}

/** Remove only the visual spheres (preserves placement state). */
function _clearPriorCloudSpheres() {
    state.priorCloudSpheres.forEach(s => {
        try { state.viewer.removeShape(s); } catch (_) { }
    });
    state.priorCloudSpheres = [];
    if (state.viewer) state.viewer.render();
}

/** Toggle prior-cloud visibility. */
function togglePriorCloud() {
    if (!state.priorCloud) return;

    if (state.priorCloudVisible) {
        // Hide spheres but keep data
        state.priorCloudSpheres.forEach(s => {
            try { state.viewer.removeShape(s); } catch (_) { }
        });
        state.priorCloudSpheres = [];
        state.priorCloudVisible = false;
    } else {
        // Re-render from stored data
        state.priorCloudVisible = true;
        const cloud = state.priorCloud;

        const cs = state.viewer.addSphere({
            center: { x: cloud.center.x, y: cloud.center.y, z: cloud.center.z },
            radius: PRIOR_CENTER_RADIUS,
            color: PRIOR_CENTER_COLOR,
            opacity: 0.5,
        });
        state.priorCloudSpheres.push(cs);

        cloud.points.forEach(pt => {
            const s = state.viewer.addSphere({
                center: { x: pt.x, y: pt.y, z: pt.z },
                radius: PRIOR_CLOUD_RADIUS,
                color: PRIOR_CLOUD_COLOR,
                opacity: PRIOR_CLOUD_OPACITY,
            });
            state.priorCloudSpheres.push(s);
        });
    }

    const btn = document.getElementById('toggle-prior-cloud');
    if (btn) btn.classList.toggle('active', state.priorCloudVisible);

    state.viewer.render();
}

// =========================================================================
// Prior Cloud Interactive Placement (Fragment Growing)
// =========================================================================

/**
 * Toggle the "Place Prior Cloud" mode.
 * When active, Shift+click places the cloud center at the clicked 3D position,
 * and Shift+drag moves the cloud in the screen plane.
 * Normal rotation/zoom works without Shift.
 */
function togglePriorPlacement() {
    if (!state.priorCloud) {
        showToast('No prior cloud available — upload a ligand first', 'error');
        return;
    }
    state.priorCloudPlacing = !state.priorCloudPlacing;
    const btn = document.getElementById('place-prior-btn');
    const canvas = document.getElementById('viewer3d');

    if (state.priorCloudPlacing) {
        btn.classList.add('placing');
        btn.innerHTML = '<span id="place-prior-icon">📌</span> Placing…';
        canvas.classList.add('prior-placing');
        _attachPriorPlacementListeners();
        showToast('Placement mode ON — Shift+click to place, Shift+drag to move the cloud. Release Shift to rotate/zoom normally.', 'success');
    } else {
        btn.classList.remove('placing');
        btn.innerHTML = '<span id="place-prior-icon">📌</span> Place';
        canvas.classList.remove('prior-placing');
        canvas.classList.remove('prior-dragging');
        _detachPriorPlacementListeners();
    }
}

/** Exit placement mode cleanly (e.g. when switching generation modes). */
function _exitPriorPlacement() {
    if (!state.priorCloudPlacing) return;
    state.priorCloudPlacing = false;
    const btn = document.getElementById('place-prior-btn');
    if (btn) {
        btn.classList.remove('placing');
        btn.innerHTML = '<span id="place-prior-icon">📌</span> Place';
    }
    const canvas = document.getElementById('viewer3d');
    if (canvas) {
        canvas.classList.remove('prior-placing');
        canvas.classList.remove('prior-dragging');
    }
    _detachPriorPlacementListeners();
}

/**
 * Reset user-placed prior cloud position back to the default
 * (ligand center of mass or prior center file).
 */
function resetPriorPosition() {
    state._priorPlacedCenter = null;
    _updatePriorCoordsDisplay(null);
    document.getElementById('reset-prior-pos-btn').classList.add('hidden');
    // Re-fetch from server (default position)
    _fetchAndRenderPriorCloudPreview();
    showToast('Prior cloud position reset to default', 'success');
}

/** Update the coordinate display next to the Place button. */
function _updatePriorCoordsDisplay(center) {
    const el = document.getElementById('prior-coords-display');
    if (!el) return;
    if (center) {
        el.textContent = `(${center.x.toFixed(1)}, ${center.y.toFixed(1)}, ${center.z.toFixed(1)})`;
    } else {
        el.textContent = '';
    }
}

// ── Placement event handlers ──

let _priorPlacementHandlers = null;

function _attachPriorPlacementListeners() {
    if (_priorPlacementHandlers) return;
    const canvas = document.getElementById('viewer3d');

    const onMouseDown = (e) => {
        if (!state.priorCloudPlacing || !state.priorCloud) return;
        if (!e.shiftKey) return; // Only intercept when Shift is held

        e.preventDefault();
        e.stopPropagation();

        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        // Store drag start state
        state._priorDrag = {
            startX: e.clientX,
            startY: e.clientY,
            screenX: sx,
            screenY: sy,
            startCenter: { ...state.priorCloud.center },
            startPoints: state.priorCloud.points.map(p => ({ ...p })),
            moved: false,
        };
        canvas.classList.add('prior-dragging');
    };

    const onMouseMove = (e) => {
        if (!state._priorDrag) return;

        e.preventDefault();
        e.stopPropagation();

        const dx = e.clientX - state._priorDrag.startX;
        const dy = e.clientY - state._priorDrag.startY;

        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
            state._priorDrag.moved = true;
        }

        // Convert screen-space delta to world-space delta using the viewer's
        // model-view-projection setup. 3Dmol.js viewer exposes the rotation
        // matrix and scale. We use the inverse of the rotation to map screen
        // movement to world movement.
        const worldDelta = _screenDeltaToWorld(dx, dy, state._priorDrag.startCenter);

        // Apply delta to all cloud points
        const sc = state._priorDrag.startCenter;
        const newCenter = {
            x: sc.x + worldDelta.x,
            y: sc.y + worldDelta.y,
            z: sc.z + worldDelta.z,
        };

        const newPoints = state._priorDrag.startPoints.map(p => ({
            x: p.x + worldDelta.x,
            y: p.y + worldDelta.y,
            z: p.z + worldDelta.z,
        }));

        // Update cloud data and re-render spheres
        state.priorCloud.center = newCenter;
        state.priorCloud.points = newPoints;
        _rerenderPriorCloudSpheresDebounced();
    };

    const onMouseUp = (e) => {
        if (!state._priorDrag) return;

        e.preventDefault();
        e.stopPropagation();

        canvas.classList.remove('prior-dragging');

        if (!state._priorDrag.moved) {
            // It was a Shift+click (not a drag) — place the cloud center
            // at the clicked 3D position (finding the closest depth reference)
            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            _placePriorAtScreenPos(sx, sy);
        }

        // Commit placement
        _commitPriorPlacement();
        state._priorDrag = null;
    };

    const onWheel = (e) => {
        if (!state.priorCloudPlacing || !state.priorCloud || !e.shiftKey) return;

        e.preventDefault();
        e.stopPropagation();

        // Shift+scroll moves the cloud along the camera's depth (Z) axis
        // Scale speed with zoom level for consistent feel
        const viewer = state.viewer;
        const c = state.priorCloud.center;
        const screenC = viewer.modelToScreen(c);
        const probe = viewer.modelToScreen({ x: c.x + 1, y: c.y, z: c.z });
        const pixPerAng = Math.hypot(probe.x - screenC.x, probe.y - screenC.y);
        const speed = Math.max(0.05, 50.0 / Math.max(pixPerAng, 1));
        const delta = e.deltaY > 0 ? -speed : speed;

        const depthDelta = _screenDepthToWorld(delta);

        state.priorCloud.center.x += depthDelta.x;
        state.priorCloud.center.y += depthDelta.y;
        state.priorCloud.center.z += depthDelta.z;

        state.priorCloud.points.forEach(p => {
            p.x += depthDelta.x;
            p.y += depthDelta.y;
            p.z += depthDelta.z;
        });

        _rerenderPriorCloudSpheres();
        _commitPriorPlacement();
    };

    // Use capture phase to intercept before 3Dmol.js
    canvas.addEventListener('mousedown', onMouseDown, true);
    document.addEventListener('mousemove', onMouseMove, true);
    document.addEventListener('mouseup', onMouseUp, true);
    canvas.addEventListener('wheel', onWheel, { capture: true, passive: false });

    const onKeyDown = (e) => {
        if (e.key === 'Escape' && state.priorCloudPlacing) {
            togglePriorPlacement();
        }
    };
    document.addEventListener('keydown', onKeyDown);

    const onBlur = () => {
        if (state._priorDrag) {
            const canvasEl = document.getElementById('viewer3d');
            if (canvasEl) canvasEl.classList.remove('prior-dragging');
            _commitPriorPlacement();
            state._priorDrag = null;
        }
    };
    window.addEventListener('blur', onBlur);

    _priorPlacementHandlers = { onMouseDown, onMouseMove, onMouseUp, onWheel, onKeyDown, onBlur };
}

function _detachPriorPlacementListeners() {
    if (!_priorPlacementHandlers) return;
    const canvas = document.getElementById('viewer3d');
    const h = _priorPlacementHandlers;

    canvas.removeEventListener('mousedown', h.onMouseDown, true);
    document.removeEventListener('mousemove', h.onMouseMove, true);
    document.removeEventListener('mouseup', h.onMouseUp, true);
    canvas.removeEventListener('wheel', h.onWheel, true);
    document.removeEventListener('keydown', h.onKeyDown);
    window.removeEventListener('blur', h.onBlur);

    _priorPlacementHandlers = null;
    state._priorDrag = null;
}

/**
 * Convert a screen-space pixel delta (dx, dy) to a world-space delta vector.
 *
 * Uses 3Dmol.js's internal rotation matrix to compute the mapping between
 * screen axes and world axes, accounting for the current zoom level.
 */
function _screenDeltaToWorld(dx, dy, refCenter) {
    // Convert screen-space pixel delta to world-space delta via differential
    // projection. Uses the viewer's modelToScreen to build a local Jacobian,
    // then computes the pseudoinverse to map 2D screen displacement back to 3D.
    const viewer = state.viewer;
    const c = refCenter || state.priorCloud?.center;
    if (!c) return { x: 0, y: 0, z: 0 };

    // Get screen position of the reference center
    const screenCenter = viewer.modelToScreen({ x: c.x, y: c.y, z: c.z });

    // Project small world-space offsets to screen, compute the
    // screen-to-world ratio, then scale.
    const eps = 1.0;  // 1 Angstrom probe
    const probeX = viewer.modelToScreen({ x: c.x + eps, y: c.y, z: c.z });
    const probeY = viewer.modelToScreen({ x: c.x, y: c.y + eps, z: c.z });
    const probeZ = viewer.modelToScreen({ x: c.x, y: c.y, z: c.z + eps });

    // Screen-space displacements per 1Å along each world axis
    const dxWx = probeX.x - screenCenter.x;
    const dyWx = probeX.y - screenCenter.y;
    const dxWy = probeY.x - screenCenter.x;
    const dyWy = probeY.y - screenCenter.y;
    const dxWz = probeZ.x - screenCenter.x;
    const dyWz = probeZ.y - screenCenter.y;

    // Solve 2D linear system: [dxWx dxWy dxWz] [wx]   [dx]
    //                         [dyWx dyWy dyWz] [wy] = [dy]
    // Since this is under-determined (3 unknowns, 2 equations), we pick
    // the minimum-norm solution in the screen plane (ignore depth movement
    // for XY drag — depth is handled by Shift+scroll).
    // We use the pseudoinverse approach: A^T (A A^T)^-1 b
    const A = [
        [dxWx, dxWy, dxWz],
        [dyWx, dyWy, dyWz],
    ];

    // A * A^T  (2x2 matrix)
    const AAT = [
        [A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2],
        A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2]],
        [A[1][0] * A[0][0] + A[1][1] * A[0][1] + A[1][2] * A[0][2],
        A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2]],
    ];

    const det = AAT[0][0] * AAT[1][1] - AAT[0][1] * AAT[1][0];
    if (Math.abs(det) < 1e-12) return { x: 0, y: 0, z: 0 };

    // (A A^T)^-1
    const AATinv = [
        [AAT[1][1] / det, -AAT[0][1] / det],
        [-AAT[1][0] / det, AAT[0][0] / det],
    ];

    // (A A^T)^-1 * b
    const tmp = [
        AATinv[0][0] * dx + AATinv[0][1] * dy,
        AATinv[1][0] * dx + AATinv[1][1] * dy,
    ];

    // A^T * tmp = world delta
    return {
        x: A[0][0] * tmp[0] + A[1][0] * tmp[1],
        y: A[0][1] * tmp[0] + A[1][1] * tmp[1],
        z: A[0][2] * tmp[0] + A[1][2] * tmp[1],
    };
}

/**
 * Convert a depth delta (in arbitrary units) to a world-space displacement
 * along the camera's viewing direction.
 */
function _screenDepthToWorld(delta) {
    const viewer = state.viewer;
    const c = state.priorCloud?.center;
    if (!c) return { x: 0, y: 0, z: 0 };

    const screenC = viewer.modelToScreen({ x: c.x, y: c.y, z: c.z });
    const eps = 1.0;

    // Probe along each axis to find which world direction corresponds to screen depth
    const probeX = viewer.modelToScreen({ x: c.x + eps, y: c.y, z: c.z });
    const probeY = viewer.modelToScreen({ x: c.x, y: c.y + eps, z: c.z });
    const probeZ = viewer.modelToScreen({ x: c.x, y: c.y, z: c.z + eps });

    // Screen Z component per world axis
    const dzWx = probeX.z - screenC.z;
    const dzWy = probeY.z - screenC.z;
    const dzWz = probeZ.z - screenC.z;

    // Normalize to get the world-space direction of screen depth
    const norm = Math.sqrt(dzWx * dzWx + dzWy * dzWy + dzWz * dzWz);
    if (norm < 1e-12) return { x: 0, y: 0, z: 0 };

    return {
        x: (dzWx / norm) * delta,
        y: (dzWy / norm) * delta,
        z: (dzWz / norm) * delta,
    };
}

/**
 * Place the prior cloud center at a 3D position derived from the screen
 * click. Uses the ligand center of mass depth as the reference depth plane.
 */
function _placePriorAtScreenPos(sx, sy) {
    if (!state.priorCloud || !state.ligandData) return;
    const viewer = state.viewer;

    // Use the current cloud center's screen position for depth reference
    const c = state.priorCloud.center;
    const screenC = viewer.modelToScreen(c);

    // Project back: find the world point at (sx, sy) at the same depth as the cloud center
    // Using the same differential approach as _screenDeltaToWorld
    const dx = sx - screenC.x;
    const dy = sy - screenC.y;

    const worldDelta = _screenDeltaToWorld(dx, dy);

    // Shift all cloud points by this delta
    const oldCenter = { ...state.priorCloud.center };
    state.priorCloud.center = {
        x: oldCenter.x + worldDelta.x,
        y: oldCenter.y + worldDelta.y,
        z: oldCenter.z + worldDelta.z,
    };

    state.priorCloud.points = state.priorCloud.points.map(p => ({
        x: p.x - oldCenter.x + state.priorCloud.center.x,
        y: p.y - oldCenter.y + state.priorCloud.center.y,
        z: p.z - oldCenter.z + state.priorCloud.center.z,
    }));

    _rerenderPriorCloudSpheres();
}

/** Re-draw all prior cloud spheres at their current positions. */
function _rerenderPriorCloudSpheres() {
    // Remove existing spheres
    state.priorCloudSpheres.forEach(s => {
        try { state.viewer.removeShape(s); } catch (_) { }
    });
    state.priorCloudSpheres = [];

    if (!state.priorCloud || !state.priorCloudVisible) return;

    const cloud = state.priorCloud;

    // Centre marker
    const cs = state.viewer.addSphere({
        center: { x: cloud.center.x, y: cloud.center.y, z: cloud.center.z },
        radius: PRIOR_CENTER_RADIUS,
        color: PRIOR_CENTER_COLOR,
        opacity: 0.5,
    });
    state.priorCloudSpheres.push(cs);

    // Cloud points
    cloud.points.forEach(pt => {
        const s = state.viewer.addSphere({
            center: { x: pt.x, y: pt.y, z: pt.z },
            radius: PRIOR_CLOUD_RADIUS,
            color: PRIOR_CLOUD_COLOR,
            opacity: PRIOR_CLOUD_OPACITY,
        });
        state.priorCloudSpheres.push(s);
    });

    state.viewer.render();
}

/** Debounced version of _rerenderPriorCloudSpheres for use during drag. */
let _priorRenderRAF = null;
function _rerenderPriorCloudSpheresDebounced() {
    if (_priorRenderRAF) return;
    _priorRenderRAF = requestAnimationFrame(() => {
        _rerenderPriorCloudSpheres();
        _priorRenderRAF = null;
    });
}

/**
 * Commit the current cloud center as the user's chosen placement.
 * Stores coordinates locally and updates the display.
 */
function _commitPriorPlacement() {
    if (!state.priorCloud) return;

    const center = state.priorCloud.center;
    state._priorPlacedCenter = { ...center };
    _updatePriorCoordsDisplay(center);

    const resetBtn = document.getElementById('reset-prior-pos-btn');
    if (resetBtn) resetBtn.classList.remove('hidden');
}

// =========================================================================
// Auto-Highlight Inpainting Mask
// =========================================================================

function clearAutoHighlight() {
    state.autoHighlightSpheres.forEach(s => {
        try { state.viewer.removeShape(s); } catch (_) { }
    });
    state.autoHighlightSpheres = [];
    _hideInpaintingLegend();
}

/**
 * Restore auto-highlight spheres if an inpainting mode is active and
 * the ref ligand is visible. Called when switching back from protein view.
 */
function _restoreAutoHighlight() {
    if (!state.refLigandVisible) return;
    const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
    if (autoModes.includes(state.genMode) && state.ligandData && state.jobId && state.autoHighlightSpheres.length === 0) {
        fetchAndHighlightInpaintingMask(state.genMode);
    }
}

async function fetchAndHighlightInpaintingMask(mode) {
    if (!state.jobId || !state.ligandData) return;
    // Don't add highlight spheres when the ref ligand is hidden
    if (!state.refLigandVisible) return;
    clearAutoHighlight();

    try {
        let maskUrl = `${API_BASE}/inpainting-mask/${state.jobId}?mode=${mode}`;
        if (mode === 'core_growing' && state._ringSystemIndex !== undefined) {
            maskUrl += `&ring_system_index=${state._ringSystemIndex}`;
        }
        const resp = await fetch(maskUrl);
        if (!resp.ok) {
            if (resp.status === 501) {
                console.warn('Inpainting mask not available');
                return;
            }
            throw new Error(await resp.text());
        }
        const data = await resp.json();

        // Re-check after async fetch: user may have changed mode or hidden the ligand
        if (!state.refLigandVisible) return;
        if (state.genMode !== mode) return;

        // Modes where selected atoms are KEPT fixed (show blue blobs)
        const fixedModes = ['core_growing', 'fragment_growing'];
        const isFixedMode = fixedModes.includes(mode);

        // For fixed-atom modes, highlight BOTH fixed (blue) and replaced (red).
        // For replaced-atom modes, highlight the replaced atoms in red.
        const fixedIndices = data.fixed || [];
        const replacedIndices = data.replaced || [];

        // Draw fixed atoms (blue) – only for modes that keep atoms fixed
        if (isFixedMode) {
            fixedIndices.forEach(idx => {
                const atom = state.ligandData._atomByIdx.get(idx);
                if (!atom || atom.atomicNum === 1) return;
                const blob = state.viewer.addSphere({
                    center: { x: atom.x, y: atom.y, z: atom.z },
                    radius: SELECTION_BLOB_RADIUS,
                    color: COLORS.kept,
                    opacity: SELECTION_BLOB_OPACITY,
                });
                state.autoHighlightSpheres.push(blob);
            });
            // Also draw replaced atoms (red) so the user sees the full picture
            replacedIndices.forEach(idx => {
                const atom = state.ligandData._atomByIdx.get(idx);
                if (!atom || atom.atomicNum === 1) return;
                const blob = state.viewer.addSphere({
                    center: { x: atom.x, y: atom.y, z: atom.z },
                    radius: SELECTION_BLOB_RADIUS,
                    color: COLORS.replaced,
                    opacity: SELECTION_BLOB_OPACITY,
                });
                state.autoHighlightSpheres.push(blob);
            });
        } else {
            // For scaffold_hopping/scaffold_elaboration/linker_inpainting: show replaced atoms in red
            replacedIndices.forEach(idx => {
                const atom = state.ligandData._atomByIdx.get(idx);
                if (!atom || atom.atomicNum === 1) return;
                const blob = state.viewer.addSphere({
                    center: { x: atom.x, y: atom.y, z: atom.z },
                    radius: SELECTION_BLOB_RADIUS,
                    color: COLORS.replaced,
                    opacity: SELECTION_BLOB_OPACITY,
                });
                state.autoHighlightSpheres.push(blob);
            });
        }

        state.viewer.render();

        // Count only heavy atoms (skip H) for accurate legend labels
        const nFixedHeavy = fixedIndices.filter(idx => {
            const a = state.ligandData._atomByIdx.get(idx);
            return a && a.atomicNum !== 1;
        }).length;
        const nReplacedHeavy = replacedIndices.filter(idx => {
            const a = state.ligandData._atomByIdx.get(idx);
            return a && a.atomicNum !== 1;
        }).length;

        // Show inline legend
        _showInpaintingLegend(mode, nFixedHeavy, nReplacedHeavy, isFixedMode);
    } catch (e) {
        console.warn('Failed to fetch inpainting mask:', e);
    }
}

/** Show a small legend below the mode selector explaining the blob colors. */
function _showInpaintingLegend(mode, nFixed, nReplaced, isFixedMode) {
    let legend = document.getElementById('inpainting-legend');
    if (!legend) {
        legend = document.createElement('div');
        legend.id = 'inpainting-legend';
        legend.className = 'inpainting-legend';
        // Insert after the mode-selector container
        const modeSelector = document.querySelector('.mode-selector');
        if (modeSelector) {
            modeSelector.insertAdjacentElement('afterend', legend);
        } else {
            return;
        }
    }

    if (mode === 'substructure_inpainting') {
        // Substructure mode: user-selected atoms will be replaced/regenerated
        const nSelected = nReplaced;
        const nKept = nFixed;
        legend.innerHTML = `
            <div class="legend-row">
                <span class="legend-dot" style="background:${COLORS.kept}"></span>
                <span>Fixed (${nKept} atoms)</span>
            </div>
            <div class="legend-row">
                <span class="legend-dot" style="background:${COLORS.replaced}"></span>
                <span>To-be-generated (${nSelected} atoms)</span>
            </div>
        `;
    } else if (isFixedMode) {
        // For fragment_growing, the "replaced" count is 0 — show grow_size instead
        const isFragGrowing = (mode === 'fragment_growing');
        const growSize = isFragGrowing
            ? (parseInt(document.getElementById('grow-size')?.value) || 5)
            : nReplaced;
        const replacedLabel = isFragGrowing
            ? `To-be-generated (${growSize} atoms)`
            : `To-be-replaced (${nReplaced} atoms)`;

        legend.innerHTML = `
            <div class="legend-row">
                <span class="legend-dot" style="background:${COLORS.kept}"></span>
                <span>To-be-fixed (${nFixed} atoms)</span>
            </div>
            <div class="legend-row">
                <span class="legend-dot" style="background:${COLORS.replaced}"></span>
                <span>${replacedLabel}</span>
            </div>
        `;
    } else {
        legend.innerHTML = `
            <div class="legend-row">
                <span class="legend-dot" style="background:${COLORS.replaced}"></span>
                <span>To-be-replaced (${nReplaced} atoms)</span>
            </div>
        `;
    }
    legend.classList.remove('hidden');
}

function _hideInpaintingLegend() {
    const legend = document.getElementById('inpainting-legend');
    if (legend) legend.classList.add('hidden');
}

// =========================================================================
// Modal Utilities
// =========================================================================

function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.remove('hidden');
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        // Purge any Plotly plots to free WebGL/canvas resources
        modal.querySelectorAll('[id^="propspace-plot-"], #chemspace-plot, [id^="affinity-plot-"]').forEach(el => {
            try { Plotly.purge(el); } catch (_) { }
        });
        modal.classList.add('hidden');
    }
}

function onModalOverlayClick(event, modalId) {
    if (event.target.classList.contains('modal-overlay')) {
        closeModal(modalId);
    }
}

// Close modals on ESC key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay:not(.hidden)').forEach(m => {
            closeModal(m.id);
        });
    }
});

// =========================================================================
// Chemical Space Visualization
// =========================================================================

let _chemSpaceCurrentMethod = 'pca';

async function showChemicalSpace() {
    if (!state.jobId || state.generatedResults.length === 0) {
        showToast('Generate ligands first', 'error');
        return;
    }
    openModal('chemspace-modal');
    await _fetchAndRenderChemSpace('pca');
}

function switchChemSpaceMethod(method) {
    // Update toggle buttons
    document.querySelectorAll('#chemspace-method-toggle .method-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.method === method);
    });
    _fetchAndRenderChemSpace(method);
}

async function _fetchAndRenderChemSpace(method) {
    _chemSpaceCurrentMethod = method;
    const loading = document.getElementById('chemspace-loading');
    const plotDiv = document.getElementById('chemspace-plot');
    loading.classList.remove('hidden');
    plotDiv.innerHTML = '';

    try {
        const resp = await fetch(`${API_BASE}/chemical-space/${state.jobId}?method=${method}`);
        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(errText);
        }
        const data = await resp.json();
        loading.classList.add('hidden');
        _renderChemSpacePlot(data);
    } catch (e) {
        loading.classList.add('hidden');
        plotDiv.innerHTML = `<div class="modal-error">Failed to compute chemical space: ${escapeHtml(e.message)}</div>`;
    }
}

function _renderChemSpacePlot(data) {
    const plotDiv = document.getElementById('chemspace-plot');
    if (typeof Plotly === 'undefined') {
        plotDiv.innerHTML = '<div class="modal-error">Plotly.js failed to load. Check your internet connection.</div>';
        return;
    }
    const method = data.method;

    // Axis labels
    const axisLabels = {
        pca: ['PC 1', 'PC 2'],
        tsne: ['t-SNE 1', 't-SNE 2'],
        umap: ['UMAP 1', 'UMAP 2'],
    };
    const [xLabel, yLabel] = axisLabels[method] || ['Dim 1', 'Dim 2'];

    // Generated ligands trace
    const genX = data.ligands.map(l => l.x);
    const genY = data.ligands.map(l => l.y);
    const genText = data.ligands.map(l => {
        const p = l.properties || {};
        return `<b>Ligand #${l.idx + 1}</b><br>` +
            `SMILES: ${(l.smiles || '').slice(0, 45)}${(l.smiles || '').length > 45 ? '…' : ''}<br>` +
            `MW: ${p.MolWt ?? '–'}<br>` +
            `LogP: ${p.LogP ?? '–'}<br>` +
            `TPSA: ${p.TPSA ?? '–'}<br>` +
            `Fsp3: ${p.FractionCSP3 ?? '–'}<br>` +
            `HBA: ${p.NumHAcceptors ?? '–'}<br>` +
            `RotBonds: ${p.NumRotatableBonds ?? '–'}`;
    });
    const genIndices = data.ligands.map(l => l.idx);

    const genTrace = {
        x: genX,
        y: genY,
        mode: 'markers',
        type: data.ligands.length > 500 ? 'scattergl' : 'scatter',
        name: 'Generated',
        text: genText,
        hovertemplate: '%{text}<extra></extra>',
        customdata: genIndices,
        marker: {
            size: 9,
            color: '#DFC6F6',
            line: { width: 1.5, color: '#9c8aac' },
            opacity: 0.85,
        },
    };

    // Reference ligand trace
    const refP = data.reference.properties || {};
    const refText = `<b>Reference Ligand</b><br>` +
        `MW: ${refP.MolWt ?? '–'}<br>` +
        `LogP: ${refP.LogP ?? '–'}<br>` +
        `TPSA: ${refP.TPSA ?? '–'}<br>` +
        `Fsp3: ${refP.FractionCSP3 ?? '–'}<br>` +
        `HBA: ${refP.NumHAcceptors ?? '–'}<br>` +
        `RotBonds: ${refP.NumRotatableBonds ?? '–'}`;

    const refTrace = {
        x: [data.reference.x],
        y: [data.reference.y],
        mode: 'markers',
        type: 'scatter',
        name: 'Reference',
        text: [refText],
        hovertemplate: '%{text}<extra></extra>',
        marker: {
            size: 16,
            color: '#cc3333',
            symbol: 'x',
            line: { width: 2.5, color: '#8b0000' },
            opacity: 1.0,
        },
    };

    const layout = {
        xaxis: {
            title: xLabel,
            gridcolor: 'rgba(120,94,1,0.1)',
            zerolinecolor: 'rgba(120,94,1,0.15)',
        },
        yaxis: {
            title: yLabel,
            gridcolor: 'rgba(120,94,1,0.1)',
            zerolinecolor: 'rgba(120,94,1,0.15)',
        },
        plot_bgcolor: 'rgba(255,255,255,0.4)',
        paper_bgcolor: 'transparent',
        margin: { l: 60, r: 30, t: 30, b: 60 },
        font: { family: 'Inter, sans-serif', size: 13, color: '#3a3238' },
        legend: {
            x: 1, y: 1, xanchor: 'right',
            bgcolor: 'rgba(255,255,255,0.7)',
            bordercolor: 'rgba(120,94,1,0.2)',
            borderwidth: 1,
        },
        hovermode: 'closest',
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
    };

    Plotly.purge(plotDiv);
    Plotly.newPlot(plotDiv, [genTrace, refTrace], layout, config);

    // Click on generated ligand dot → show that ligand in the sidebar
    plotDiv.on('plotly_click', (eventData) => {
        const point = eventData.points[0];
        if (point && point.data.name === 'Generated' && point.customdata !== undefined) {
            showGeneratedLigand(point.customdata);
        }
    });
}

// =========================================================================
// Property Space Visualization (Violin Plots)
// =========================================================================

async function showPropertySpace() {
    if (!state.jobId || state.generatedResults.length === 0) {
        showToast('Generate ligands first', 'error');
        return;
    }
    openModal('propspace-modal');

    const loading = document.getElementById('propspace-loading');
    const grid = document.getElementById('propspace-grid');
    loading.classList.remove('hidden');
    grid.innerHTML = '';

    try {
        const resp = await fetch(`${API_BASE}/property-space/${state.jobId}`);
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        loading.classList.add('hidden');
        _renderPropertySpacePlots(data);
    } catch (e) {
        loading.classList.add('hidden');
        grid.innerHTML = `<div class="modal-error">Failed to compute properties: ${escapeHtml(e.message)}</div>`;
    }
}

function _renderPropertySpacePlots(data) {
    const grid = document.getElementById('propspace-grid');
    grid.innerHTML = '';
    if (typeof Plotly === 'undefined') {
        grid.innerHTML = '<div class="modal-error">Plotly.js failed to load. Check your internet connection.</div>';
        return;
    }

    const refProps = data.reference || {};
    const allProps = data.property_names || [];
    const contProps = new Set(data.continuous_properties || []);

    allProps.forEach(propName => {
        // Collect values for this property across all generated ligands
        const values = [];
        data.ligands.forEach(lig => {
            const v = lig.properties?.[propName];
            if (v !== null && v !== undefined) values.push(v);
        });
        if (values.length === 0) return;

        const refVal = refProps[propName];
        const isContinuous = contProps.has(propName);

        // Create container div
        const div = document.createElement('div');
        div.className = 'propspace-cell';
        const plotId = `propspace-plot-${propName}`;
        div.innerHTML = `<div id="${plotId}" class="propspace-violin"></div>`;
        grid.appendChild(div);

        // Format display name
        const displayName = propName.replace(/([A-Z])/g, ' $1').trim();

        if (isContinuous) {
            // Violin plot for continuous properties
            const violinTrace = {
                type: 'violin',
                y: values,
                name: 'Generated',
                box: { visible: true },
                meanline: { visible: true },
                fillcolor: 'rgba(223,198,246,0.5)',
                line: { color: '#9c8aac', width: 1.5 },
                points: values.length <= 50 ? 'all' : false,
                jitter: 0.3,
                pointpos: -1.5,
                marker: { color: '#9c8aac', size: 4, opacity: 0.6 },
                hoverinfo: 'y',
                spanmode: 'hard',
            };

            const traces = [violinTrace];
            const annotations = [];

            if (refVal !== null && refVal !== undefined) {
                // Add reference line as a shape + annotation
                annotations.push({
                    y: refVal,
                    x: 0,
                    xref: 'x',
                    yref: 'y',
                    text: `Ref: ${typeof refVal === 'number' ? refVal.toFixed(2) : refVal}`,
                    showarrow: true,
                    arrowhead: 0,
                    arrowwidth: 1.5,
                    arrowcolor: '#cc3333',
                    ax: 65,
                    ay: 0,
                    font: { color: '#cc3333', size: 11, family: 'JetBrains Mono, monospace' },
                    bgcolor: 'rgba(255,255,255,0.85)',
                    bordercolor: '#cc3333',
                    borderwidth: 1,
                    borderpad: 3,
                });
            }

            Plotly.newPlot(plotId, traces, {
                yaxis: { gridcolor: 'rgba(120,94,1,0.1)', zeroline: false },
                xaxis: { title: { text: displayName, font: { size: 12 } }, showticklabels: false, zeroline: false },
                plot_bgcolor: 'rgba(255,255,255,0.3)',
                paper_bgcolor: 'transparent',
                margin: { l: 55, r: 20, t: 20, b: 45 },
                font: { family: 'Inter, sans-serif', size: 11 },
                showlegend: false,
                annotations: annotations,
                shapes: refVal !== null && refVal !== undefined ? [{
                    type: 'line', xref: 'paper', x0: 0.05, x1: 0.65,
                    y0: refVal, y1: refVal, yref: 'y',
                    line: { color: '#cc3333', width: 2, dash: 'dash' },
                }] : [],
            }, { responsive: true, displayModeBar: false });
        } else {
            // Histogram for discrete properties
            const histTrace = {
                type: 'histogram',
                x: values,
                name: 'Generated',
                marker: {
                    color: 'rgba(223,198,246,0.65)',
                    line: { color: '#9c8aac', width: 1 },
                },
                hoverinfo: 'x+y',
                xbins: { size: 1 },
            };

            const annotations = [];
            const shapes = [];
            if (refVal !== null && refVal !== undefined) {
                shapes.push({
                    type: 'line', yref: 'paper', y0: 0, y1: 1,
                    x0: refVal, x1: refVal, xref: 'x',
                    line: { color: '#cc3333', width: 2.5, dash: 'dash' },
                });
                annotations.push({
                    x: refVal,
                    y: 1,
                    xref: 'x',
                    yref: 'paper',
                    text: `Ref: ${refVal}`,
                    showarrow: false,
                    font: { color: '#cc3333', size: 11, family: 'JetBrains Mono, monospace' },
                    bgcolor: 'rgba(255,255,255,0.85)',
                    bordercolor: '#cc3333',
                    borderwidth: 1,
                    borderpad: 3,
                    yanchor: 'bottom',
                });
            }

            Plotly.newPlot(plotId, [histTrace], {
                xaxis: {
                    title: displayName,
                    dtick: 1,
                    gridcolor: 'rgba(120,94,1,0.1)',
                },
                yaxis: {
                    title: 'Count',
                    gridcolor: 'rgba(120,94,1,0.1)',
                },
                plot_bgcolor: 'rgba(255,255,255,0.3)',
                paper_bgcolor: 'transparent',
                margin: { l: 50, r: 20, t: 20, b: 45 },
                font: { family: 'Inter, sans-serif', size: 11 },
                showlegend: false,
                bargap: 0.08,
                annotations: annotations,
                shapes: shapes,
            }, { responsive: true, displayModeBar: false });
        }
    });
}

// =========================================================================
// Affinity Distribution Visualization
// =========================================================================

const _AFFINITY_LABELS = {
    pic50: 'pIC50',
    pki: 'pKi',
    pkd: 'pKd',
    pec50: 'pEC50',
};

const _AFFINITY_COLORS = {
    pic50: { fill: 'rgba(223,198,246,0.45)', line: '#9c8aac', marker: '#7b6b8a' },
    pki: { fill: 'rgba(213,214,83,0.40)', line: '#aaab42', marker: '#8a8b32' },
    pkd: { fill: 'rgba(120,94,1,0.25)', line: '#ae9e66', marker: '#785E01' },
    pec50: { fill: 'rgba(156,138,172,0.40)', line: '#6f637b', marker: '#594f62' },
};

async function showAffinityDistribution() {
    if (!state.jobId || state.generatedResults.length === 0) {
        showToast('Generate ligands first', 'error');
        return;
    }
    openModal('affinity-modal');

    const loading = document.getElementById('affinity-loading');
    const container = document.getElementById('affinity-plot-container');
    const refInfo = document.getElementById('affinity-ref-info');
    loading.classList.remove('hidden');
    container.innerHTML = '';
    refInfo.classList.add('hidden');
    refInfo.innerHTML = '';

    try {
        const resp = await fetch(`${API_BASE}/affinity-distribution/${state.jobId}`);
        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(errText);
        }
        const data = await resp.json();
        loading.classList.add('hidden');
        // Show only the selected affinity type
        const selectedType = document.getElementById('rank-affinity-type')?.value || 'pic50';
        if (data.affinity_types && data.affinity_types.includes(selectedType)) {
            data.affinity_types = [selectedType];
        }
        _renderAffinityPlots(data);
    } catch (e) {
        loading.classList.add('hidden');
        container.innerHTML = `<div class="modal-error">Failed to load affinity data: ${escapeHtml(e.message)}</div>`;
    }
}

function _renderAffinityPlots(data) {
    const container = document.getElementById('affinity-plot-container');
    const refInfo = document.getElementById('affinity-ref-info');
    container.innerHTML = '';

    if (typeof Plotly === 'undefined') {
        container.innerHTML = '<div class="modal-error">Plotly.js failed to load.</div>';
        return;
    }

    const refAff = data.ref_affinity;

    // Show reference affinity info banner if detected
    if (refAff) {
        refInfo.classList.remove('hidden');
        const rawDesc = refAff.unit
            ? `${refAff.assay_type} = ${refAff.raw_value} ${refAff.unit}`
            : `${refAff.assay_type} = ${refAff.raw_value}`;
        refInfo.innerHTML = `
            <div class="affinity-ref-banner">
                <span class="affinity-ref-icon">📌</span>
                <div class="affinity-ref-text">
                    <strong>Reference ligand:</strong> ${escapeHtml(refAff.p_label)} = ${refAff.p_value.toFixed(2)}
                    <span class="affinity-ref-raw">(from SDF tag <code>${escapeHtml(refAff.raw_tag)}</code>: ${escapeHtml(rawDesc)})</span>
                </div>
            </div>
        `;
    }

    // Create one plot per affinity type
    data.affinity_types.forEach((affType, idx) => {
        const dist = data.distributions[affType];
        if (!dist || dist.values.length === 0) return;

        const label = _AFFINITY_LABELS[affType] || affType;
        const colors = _AFFINITY_COLORS[affType] || _AFFINITY_COLORS.pic50;
        const plotId = `affinity-plot-${affType}`;

        const plotWrapper = document.createElement('div');
        plotWrapper.className = 'affinity-plot-wrapper';
        plotWrapper.innerHTML = `
            <h3 class="affinity-plot-title">${escapeHtml(label)} Distribution</h3>
            <div id="${plotId}" class="affinity-plot"></div>
        `;
        container.appendChild(plotWrapper);

        const values = dist.values;
        const n = values.length;
        const mean = values.reduce((a, b) => a + b, 0) / n;
        const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
        const sorted = [...values].sort((a, b) => a - b);
        const median = n % 2 === 0
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
            : sorted[Math.floor(n / 2)];

        const traces = [];
        const annotations = [];
        const shapes = [];

        // Main trace: violin + box + strip (Plotly's combined violin)
        traces.push({
            type: 'violin',
            y: values,
            name: 'Generated',
            box: { visible: true, fillcolor: 'rgba(255,255,255,0.5)', width: 0.12 },
            meanline: { visible: true, color: colors.line, width: 2 },
            fillcolor: colors.fill,
            line: { color: colors.line, width: 1.5 },
            points: n <= 80 ? 'all' : 'outliers',
            jitter: 0.45,
            pointpos: -1.8,
            marker: { color: colors.marker, size: 5, opacity: 0.7 },
            hovertemplate: `${label}: %{y:.2f}<extra>%{text}</extra>`,
            text: dist.labels,
            customdata: dist.indices || dist.labels.map((_, idx) => idx),
            spanmode: 'hard',
            bandwidth: n > 5 ? undefined : 0.3,
            scalemode: 'width',
            width: 0.7,
        });

        // Reference line (if same affinity type matches)
        let refVal = null;
        if (refAff) {
            // Direct mapping: refAff.p_label "pIC50"→"pic50", "pKi"→"pki", "pKd"→"pkd", "pEC50"→"pec50"
            const refKeyMap = { 'pIC50': 'pic50', 'pKi': 'pki', 'pKd': 'pkd', 'pEC50': 'pec50' };
            const mappedKey = refKeyMap[refAff.p_label];

            if (mappedKey === affType) {
                refVal = refAff.p_value;

                // Dashed reference line across the full violin area
                shapes.push({
                    type: 'line',
                    xref: 'paper',
                    x0: 0.0,
                    x1: 0.95,
                    y0: refVal,
                    y1: refVal,
                    yref: 'y',
                    line: { color: '#cc3333', width: 2, dash: 'dash' },
                });

                // Annotation label
                annotations.push({
                    y: refVal,
                    x: 0.97,
                    xref: 'paper',
                    yref: 'y',
                    text: `Ref: ${refVal.toFixed(2)}`,
                    showarrow: false,
                    font: { color: '#cc3333', size: 12, family: 'JetBrains Mono, monospace' },
                    bgcolor: 'rgba(255,255,255,0.9)',
                    bordercolor: '#cc3333',
                    borderwidth: 1,
                    borderpad: 4,
                    xanchor: 'right',
                });
            }
        }

        // Stats annotation (bottom right)
        const statsText = [
            `n = ${n}`,
            `mean = ${mean.toFixed(2)}`,
            `median = ${median.toFixed(2)}`,
            `std = ${std.toFixed(2)}`,
        ].join('<br>');

        annotations.push({
            x: 1.0,
            y: 0.0,
            xref: 'paper',
            yref: 'paper',
            text: statsText,
            showarrow: false,
            font: { size: 11, family: 'JetBrains Mono, monospace', color: '#594f62' },
            bgcolor: 'rgba(255,255,255,0.85)',
            bordercolor: 'rgba(120,94,1,0.2)',
            borderwidth: 1,
            borderpad: 6,
            xanchor: 'right',
            yanchor: 'bottom',
            align: 'left',
        });

        const layout = {
            yaxis: {
                title: { text: label, font: { size: 14 } },
                gridcolor: 'rgba(120,94,1,0.1)',
                zeroline: false,
            },
            xaxis: {
                showticklabels: false,
                zeroline: false,
            },
            plot_bgcolor: 'rgba(255,255,255,0.3)',
            paper_bgcolor: 'transparent',
            margin: { l: 65, r: 85, t: 15, b: 25 },
            font: { family: 'Inter, sans-serif', size: 12 },
            showlegend: false,
            annotations: annotations,
            shapes: shapes,
            height: 380,
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d', 'zoom2d', 'pan2d'],
            displaylogo: false,
        };

        Plotly.purge(plotId);
        Plotly.newPlot(plotId, traces, layout, config);

        // Click → show ligand in sidebar (uses customdata for robustness)
        document.getElementById(plotId).on('plotly_click', (eventData) => {
            const point = eventData.points[0];
            if (point && typeof point.customdata === 'number') {
                showGeneratedLigand(point.customdata);
            }
        });
    });

    // If no plots were rendered
    if (container.children.length === 0) {
        container.innerHTML = '<div class="modal-error">No affinity data to display.</div>';
    }
}

// =========================================================================
// Per-Ligand Properties Panel (in 2D Overlay)
// =========================================================================

async function _loadLigandProperties() {
    const container = document.getElementById('mol-2d-properties');
    if (!container || !state.jobId) return;
    if (container.dataset.loaded === 'true') return;

    const ligIdx = state.activeResultIdx >= 0 ? state.activeResultIdx : -1;
    container.innerHTML = '<span class="muted-text">Loading properties…</span>';

    try {
        const resp = await fetch(`${API_BASE}/ligand-properties/${state.jobId}/${ligIdx}`);
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        const props = data.properties || {};
        const propNames = data.property_names || Object.keys(props);

        // Separate into groups
        const physicalProps = ['MolWt', 'LogP', 'FractionCSP3', 'TPSA'];
        const structuralProps = propNames.filter(p => !physicalProps.includes(p) && p !== 'NumAlerts');
        const alertProps = propNames.filter(p => p === 'NumAlerts');

        let html = '<div class="prop-table-container">';
        html += '<table class="prop-table">';

        // Physical Properties group
        html += '<tr class="prop-group-header"><td colspan="2">Physical Properties</td></tr>';
        physicalProps.forEach(name => {
            if (props[name] !== undefined) {
                const displayName = name.replace(/([A-Z])/g, ' $1').trim();
                const val = typeof props[name] === 'number' ? props[name].toFixed(3) : props[name];
                html += `<tr><td class="prop-name">${escapeHtml(displayName)}</td><td class="prop-value">${val ?? '–'}</td></tr>`;
            }
        });

        // Structural Properties group
        html += '<tr class="prop-group-header"><td colspan="2">Structural Properties</td></tr>';
        structuralProps.forEach(name => {
            if (props[name] !== undefined) {
                const displayName = name.replace(/([A-Z])/g, ' $1').trim();
                const val = props[name];
                html += `<tr><td class="prop-name">${escapeHtml(displayName)}</td><td class="prop-value">${val ?? '–'}</td></tr>`;
            }
        });

        // Alerts group
        if (alertProps.length > 0) {
            html += '<tr class="prop-group-header"><td colspan="2">Alerts</td></tr>';
            alertProps.forEach(name => {
                if (props[name] !== undefined) {
                    const displayName = name.replace(/([A-Z])/g, ' $1').trim();
                    const val = props[name];
                    const cls = val > 0 ? 'prop-value prop-alert' : 'prop-value';
                    html += `<tr><td class="prop-name">${escapeHtml(displayName)}</td><td class="${cls}">${val ?? '–'}</td></tr>`;
                }
            });
        }

        html += '</table></div>';
        container.innerHTML = html;
        container.dataset.loaded = 'true';
    } catch (e) {
        container.innerHTML = '<span class="muted-text">Could not load properties</span>';
    }
}

// Drag-over visual effects
document.querySelectorAll('.upload-area').forEach(area => {
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('drag-over');
    });
    area.addEventListener('dragleave', () => {
        area.classList.remove('drag-over');
    });
});

// =========================================================================
// Reference Ligand Hydrogens (Add / Remove)
// =========================================================================

async function removeRefHydrogens() {
    if (!state.jobId) return;
    try {
        const resp = await fetch(`${API_BASE}/ligand-remove-hs/${state.jobId}`, { method: 'POST' });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Update ligand data
        state.ligandData.atoms = data.atoms;
        state.ligandData._atomByIdx = new Map(data.atoms.map(a => [a.idx, a]));
        state.ligandData.bonds = data.bonds;
        state.ligandData.smiles = data.smiles;
        state.ligandData.sdf_data = data.sdf_data;
        state.ligandData.num_atoms = data.num_atoms;
        state.ligandData.num_heavy_atoms = data.num_heavy_atoms;
        state.ligandData.has_explicit_hs = false;

        // Rebuild heavy-atom index mapping (atom indices changed)
        state.heavyAtomMap = new Map();
        let heavyIdx = 0;
        data.atoms.forEach(a => {
            if (a.atomicNum !== 1) {
                state.heavyAtomMap.set(a.idx, heavyIdx);
                heavyIdx++;
            }
        });

        // Clear stale atom selections and auto-highlights
        state.fixedAtoms.clear();
        updateSelectionUI();
        clearAutoHighlight();

        // Update info card – always show SMILES without H
        document.getElementById('ligand-smiles').textContent = data.smiles_noH || data.smiles || 'N/A';
        document.getElementById('ligand-heavy-atoms').textContent = data.num_heavy_atoms;
        document.getElementById('ligand-total-atoms').textContent = data.num_atoms;

        // Re-render ligand in 3D
        if (state.ligandModel !== null) {
            state.viewer.removeModel(state.ligandModel);
            state.ligandModel = null;
        }
        renderLigand(data.sdf_data);

        // Re-trigger auto-highlight for inpainting modes after H change
        const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
        if (autoModes.includes(state.genMode)) {
            fetchAndHighlightInpaintingMask(state.genMode);
        }
        if (state.genMode === 'fragment_growing') {
            _fetchAndRenderPriorCloudPreview();
        }

        showToast('Hydrogens removed', 'success');
    } catch (e) {
        showToast('Failed to remove hydrogens: ' + e.message, 'error');
    }
}

async function addRefHydrogens() {
    if (!state.jobId) return;
    try {
        const resp = await fetch(`${API_BASE}/ligand-add-hs/${state.jobId}`, { method: 'POST' });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Update ligand data
        state.ligandData.atoms = data.atoms;
        state.ligandData._atomByIdx = new Map(data.atoms.map(a => [a.idx, a]));
        state.ligandData.bonds = data.bonds;
        state.ligandData.smiles = data.smiles;
        state.ligandData.sdf_data = data.sdf_data;
        state.ligandData.num_atoms = data.num_atoms;
        state.ligandData.num_heavy_atoms = data.num_heavy_atoms;
        state.ligandData.has_explicit_hs = true;

        // Rebuild heavy-atom index mapping (atom indices changed)
        state.heavyAtomMap = new Map();
        let heavyIdx = 0;
        data.atoms.forEach(a => {
            if (a.atomicNum !== 1) {
                state.heavyAtomMap.set(a.idx, heavyIdx);
                heavyIdx++;
            }
        });

        // Clear stale atom selections and auto-highlights
        state.fixedAtoms.clear();
        updateSelectionUI();
        clearAutoHighlight();

        // Update info card – always show SMILES without H
        document.getElementById('ligand-smiles').textContent = data.smiles_noH || data.smiles || 'N/A';
        document.getElementById('ligand-heavy-atoms').textContent = data.num_heavy_atoms;
        document.getElementById('ligand-total-atoms').textContent = data.num_atoms;

        // Re-render ligand in 3D
        if (state.ligandModel !== null) {
            state.viewer.removeModel(state.ligandModel);
            state.ligandModel = null;
        }
        renderLigand(data.sdf_data);

        // Re-trigger auto-highlight for inpainting modes after H change
        const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
        if (autoModes.includes(state.genMode)) {
            fetchAndHighlightInpaintingMask(state.genMode);
        }
        if (state.genMode === 'fragment_growing') {
            _fetchAndRenderPriorCloudPreview();
        }

        showToast('Hydrogens added', 'success');
    } catch (e) {
        showToast('Failed to add hydrogens: ' + e.message, 'error');
    }
}

// =========================================================================
// Generated Ligand Hydrogens (Add / Remove)
// =========================================================================

/**
 * Return the appropriate SDF string for the active generated ligand
 * based on the current H visibility preference.
 *
 * When optimization was used and Hs are removed then re-added, we fall
 * back to sdf_with_hs (which contains the optimized H positions).
 */
function _getGenSdf(result) {
    if (!result) return null;
    if (state.genHsVisible) {
        // Prefer the optimized SDF with Hs (preserves optimized H coordinates)
        return result.sdf_with_hs || result.sdf;
    } else {
        return result.sdf_no_hs || result.sdf;
    }
}

/**
 * Remove hydrogens from the currently displayed generated ligand.
 */
function removeGenHydrogens() {
    state.genHsVisible = false;
    _refreshGenLigandDisplay();
    showToast('Generated ligand Hs removed', 'success');
}

/**
 * Add hydrogens to the currently displayed generated ligand.
 * If optimization was used, this restores the optimized H positions
 * (from sdf_with_hs stored on the result).
 */
function addGenHydrogens() {
    state.genHsVisible = true;
    _refreshGenLigandDisplay();
    showToast('Generated ligand Hs added', 'success');
}

/**
 * Re-render the active generated ligand with the current H preference.
 */
function _refreshGenLigandDisplay() {
    if (state.activeResultIdx < 0 || state.activeResultIdx >= state.generatedResults.length) return;

    const result = state.generatedResults[state.activeResultIdx];

    // Remove previous generated model
    if (state.generatedModel !== null) {
        state.viewer.removeModel(state.generatedModel);
        state.generatedModel = null;
    }

    const sdfToUse = _getGenSdf(result);
    if (sdfToUse) {
        state.generatedModel = state.viewer.addModel(sdfToUse, 'sdf');
        state.viewer.setStyle(
            { model: state.generatedModel },
            {
                stick: { radius: 0.18, colorscheme: 'Jmol' },
                sphere: { radius: 0.35, colorscheme: 'Jmol' },
            }
        );
    }

    state.viewer.render();
}

// ---------------------------------------------------------------------------
// Predicted Affinity Panel
// ---------------------------------------------------------------------------

function toggleAffinityPanel() {
    const body = document.getElementById('affinity-panel-body');
    const chevron = document.getElementById('affinity-panel-chevron');
    if (!body) return;
    const isHidden = body.classList.contains('hidden');
    body.classList.toggle('hidden', !isHidden);
    if (chevron) chevron.classList.toggle('expanded', isHidden);
}

function onAffinityTypeChange() {
    // No-op for now; the affinity type selection is read when
    // "Show Distribution" or "Rank" is clicked.
}

function _rerenderResultsList() {
    const list = document.getElementById('results-list');
    list.innerHTML = '';

    state.generatedResults.forEach((result, i) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.onclick = () => showGeneratedLigand(i);

        const mw = result.properties?.mol_weight || '–';
        const ha = result.properties?.num_heavy_atoms || '–';

        card.innerHTML = `
            <div class="result-header">
                <span class="result-title">Ligand #${i + 1}</span>
                <span class="result-badge badge badge-idle">${ha} HA</span>
            </div>
            <div class="result-smiles">${escapeHtml(result.smiles || 'N/A')}</div>
            <div class="result-props">
                <span>MW: ${mw}</span>
            </div>
        `;
        list.appendChild(card);
    });

    if (state.generatedResults.length > 0) showGeneratedLigand(0);
}

async function applyRankSelect() {
    if (!state.jobId || state.generatedResults.length === 0) return;

    const affinityType = document.getElementById('rank-affinity-type').value;
    const topNInput = document.getElementById('rank-top-n').value;
    const topN = topNInput ? parseInt(topNInput) : null;

    try {
        const resp = await fetch(`${API_BASE}/rank-select/${state.jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ affinity_type: affinityType, top_n: topN }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Update state and re-render
        state.generatedResults = data.results;
        // Remove current generated model
        if (state.generatedModel !== null) {
            state.viewer.removeModel(state.generatedModel);
            state.generatedModel = null;
        }
        state.activeResultIdx = -1;

        _rerenderResultsList();
        updateSaveSection();

        // Show reset button
        document.getElementById('rank-reset-btn').classList.remove('hidden');

        const label = { pic50: 'pIC50', pki: 'pKi', pkd: 'pKd', pec50: 'pEC50' }[affinityType];
        showToast(`Ranked by ${label}` + (topN ? ` (top ${topN})` : ''), 'success');
    } catch (e) {
        showToast('Rank & select failed: ' + e.message, 'error');
    }
}

async function resetRankSelect() {
    if (!state.jobId) return;

    try {
        const resp = await fetch(`${API_BASE}/reset-rank/${state.jobId}`, {
            method: 'POST',
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        state.generatedResults = data.results;
        if (state.generatedModel !== null) {
            state.viewer.removeModel(state.generatedModel);
            state.generatedModel = null;
        }
        state.activeResultIdx = -1;

        _rerenderResultsList();
        updateSaveSection();

        document.getElementById('rank-reset-btn').classList.add('hidden');
        showToast('Rank selection reset', 'success');
    } catch (e) {
        showToast('Reset failed: ' + e.message, 'error');
    }
}
