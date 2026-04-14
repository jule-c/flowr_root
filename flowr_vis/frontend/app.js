/* ==========================================================================
   FLOWR Visualization – Application JavaScript
   ========================================================================== */

const API_BASE = '';

async function authFetch(url, options = {}) {
    options.credentials = 'include';
    return fetch(url, options);
}

// ── Color constants matching CSS palette ──
const COLORS = {
    lavender: '#DFC6F6',
    lavenderDark: '#9c8aac',
    chartreuse: '#D5D653',
    chartreuseDark: '#aaab42',
    amber: '#785E01',
    amberLight: '#ae9e66',
    bgViewer: '#F7F3ED',
    labelDefault: '#6f637b',
    generated: '#9c8aac',
    replaced: '#CE93D8',       // soft purple – atoms being REPLACED/regenerated
    replacedGlow: 'rgba(206, 147, 216, 0.22)',
    kept: '#9c8aac',           // lavender – atoms being KEPT/fixed
    keptGlow: 'rgba(156, 138, 172, 0.18)',
    hoverResidue: '#c8a2e8',   // medium lavender – residue hover highlight
    // Legacy alias (fixedAtoms = atoms to replace, see naming note)
    fixed: '#CE93D8',
    fixedGlow: 'rgba(206, 147, 216, 0.22)',
};

const INTERACTION_COLORS = {
    HBond: '#239fcd',
    SaltBridge: '#e35959',
};

// ── Round / iteration color palette for multi-round generation ──
// Each generation round gets a distinct color; cycles if >N rounds.
const ROUND_COLORS = [
    { fill: '#DFC6F6', line: '#9c8aac', fillAlpha: 'rgba(223,198,246,0.5)' },  // 0 lavender
    { fill: '#D5D68C', line: '#aaab42', fillAlpha: 'rgba(213,214,140,0.5)' },  // 1 chartreuse
    { fill: '#F2C78A', line: '#c9943c', fillAlpha: 'rgba(242,199,138,0.5)' },  // 2 amber/gold
    { fill: '#8EC8E8', line: '#4a9bc7', fillAlpha: 'rgba(142,200,232,0.5)' },  // 3 sky blue
    { fill: '#F5A3B5', line: '#c46b80', fillAlpha: 'rgba(245,163,181,0.5)' },  // 4 rose
    { fill: '#A8E6CF', line: '#5bbf94', fillAlpha: 'rgba(168,230,207,0.5)' },  // 5 mint
    { fill: '#D4B8E0', line: '#8a6b9e', fillAlpha: 'rgba(212,184,224,0.5)' },  // 6 orchid
    { fill: '#FFD6A5', line: '#c9943c', fillAlpha: 'rgba(255,214,165,0.5)' },  // 7 peach
];

function _roundColor(iteration) {
    return ROUND_COLORS[iteration % ROUND_COLORS.length];
}

function _hexToRgb(hex) {
    const h = hex.replace('#', '');
    return [Number.parseInt(h.substring(0, 2), 16), Number.parseInt(h.substring(2, 4), 16), Number.parseInt(h.substring(4, 6), 16)];
}

function _buildKdeColorscale(hexColor) {
    const [r, g, b] = _hexToRgb(hexColor);
    return [
        [0, `rgba(${r},${g},${b},0)`],
        [0.25, `rgba(${r},${g},${b},0.04)`],
        [0.5, `rgba(${r},${g},${b},0.12)`],
        [0.75, `rgba(${r},${g},${b},0.22)`],
        [1, `rgba(${r},${g},${b},0.35)`],
    ];
}

// ===== State =====
let state = {
    _currentUser: 'anonymous',
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
    iterationIdx: 0,                     // current generation round (increments each generation)
    allGeneratedResults: [],             // accumulated [{iteration, results}] across rounds
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
    _hoveredResidue: null,               // {resi, chain} of hovered protein residue (or null)
    _cachedBindingSiteSerials: null,     // cached result of _getBindingSiteSerials (invalidated on protein/ligand change)
    activeView: 'ref',                   // 'ref' | 'gen' — which ligand is shown full-atom
    selectedOverlayModels: [],           // 3Dmol models for transparent selected-ligand overlays
    refProperties: null,                 // reference ligand properties from upload
    originalLigand: null,                // original ligand data (before first set-as-reference swap)
    molstarOpen: false,                  // whether the Molstar overlay is currently visible
    molstarReady: false,                 // whether the Molstar iframe has signalled 'ready'
};

// ── RDKit.js module (loaded asynchronously) ──
let RDKitModule = null;

// ── AbortController for active generation polling ──
let _activeGenerationController = null;

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
document.addEventListener('DOMContentLoaded', async () => {
    // Task 1: Ensure a completely clean slate on every page load / reload.
    // Clear any session-level caches that might leak across reloads.
    try { sessionStorage.clear(); } catch (e) { console.debug('sessionStorage unavailable:', e.message); }

    // Start with landing page
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

    // Molstar postMessage listener
    window.addEventListener('message', function (event) {
        if (event.origin !== window.location.origin) return;
        const data = event.data;
        if (!data || data.source !== 'molstar-embed') return;

        if (data.status === 'ready') {
            state.molstarReady = true;
            sendStructureToMolstar();
        } else if (data.status === 'error') {
            console.warn('Molstar error:', data.message);
        }
    });
    // Task 2: Hide sidebar sections until both protein and ligand are uploaded
    _updateSidebarVisibility();

    // ── Main-app session file input ──
    const mainSessionFile = document.getElementById('main-session-file');
    if (mainSessionFile) {
        mainSessionFile.addEventListener('change', () => {
            if (mainSessionFile.files.length) {
                _handleSessionFileSelect(mainSessionFile.files[0], false);
            }
            mainSessionFile.value = '';
        });
    }
}

// =========================================================================
// Landing Page – Checkpoint Selection
// =========================================================================

let _ckptData = { base: [], project: [] };
let _selectedCkptPath = null;
let _initialCkptPath = null;
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
        const resp = await authFetch(`${API_BASE}/checkpoints?workflow=${workflow}`);
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
    // Show the landing page (starts hidden to prevent flash before auth check)
    document.getElementById('landing-page')?.classList.remove('hidden');
    // Default workflow is sbdd
    _selectedWorkflow = 'sbdd';
    await _fetchCheckpoints('sbdd');

    // ── Session restore dropzone ──
    const dropzone = document.getElementById('landing-restore-dropzone');
    const fileInput = document.getElementById('landing-session-file');
    if (dropzone) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        });
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('drag-over');
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                _handleSessionFileSelect(e.dataTransfer.files[0], true);
            }
        });
    }
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                _handleSessionFileSelect(fileInput.files[0], true);
            }
            fileInput.value = '';
        });
    }
}

function populateBaseSelect() {
    const sel = document.getElementById('base-ckpt-select');
    sel.innerHTML = '';

    if (_ckptData.base.length === 0) {
        sel.innerHTML = '<option value="">No base checkpoints found</option>';
        sel.disabled = true;
        return;
    }

    const frag = document.createDocumentFragment();
    _ckptData.base.forEach((ckpt, i) => {
        const opt = document.createElement('option');
        opt.value = ckpt.path;
        opt.textContent = ckpt.name;
        frag.appendChild(opt);
    });
    sel.textContent = '';
    sel.appendChild(frag);
    sel.disabled = false;
    sel.removeEventListener('change', onCkptSelectChange);
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

    const frag = document.createDocumentFragment();
    _ckptData.project.forEach(proj => {
        // If a project has multiple ckpts, list them individually with project prefix
        proj.checkpoints.forEach(ckpt => {
            const opt = document.createElement('option');
            opt.value = ckpt.path;
            opt.textContent = proj.checkpoints.length > 1
                ? `${proj.project_id} / ${ckpt.name}`
                : proj.project_id;
            frag.appendChild(opt);
        });
    });
    sel.textContent = '';
    sel.appendChild(frag);
    sel.disabled = false;
    sel.removeEventListener('change', onCkptSelectChange);
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
    btn.innerHTML = 'Preparing…';
    status.className = 'landing-status status-loading';
    status.classList.remove('hidden');
    status.textContent = 'Registering checkpoint…';

    try {
        // Register the checkpoint selection on the server.
        // Model loading is DEFERRED until the user clicks "Generate".
        const resp = await authFetch(`${API_BASE}/register-checkpoint`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ckpt_path: _selectedCkptPath, workflow_type: _selectedWorkflow }),
        });
        await resp.json();

        state.workflowType = _selectedWorkflow;
        _initialCkptPath = _selectedCkptPath;

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
        _updateModelCard();
    }, 450);
}

/**
 * Shared state/UI teardown used by both returnToLanding() and resetCurrentSession().
 * @param {Object} opts
 * @param {boolean} [opts.generateBtnDisabled=false] - Disable the generate button after reset.
 * @param {boolean} [opts.resetWorkflow=false]        - Reset workflowType to 'sbdd'.
 * @param {boolean} [opts.clearFileInputs=false]      - Clear protein/ligand file inputs.
 */
function _resetAppState(opts = {}) {
    // ── Abort any in-flight generation polling ──
    if (_activeGenerationController) {
        _activeGenerationController.abort();
        _activeGenerationController = null;
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
    const hoverCard = document.getElementById('residue-hover-card');
    if (hoverCard) hoverCard.style.display = 'none';
    state.atomLabels.forEach(l => state.viewer.removeLabel(l));
    state.atomLabels = [];
    state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
    state.selectionSpheres = [];
    state.selectionSphereMap.clear();
    if (state.viewer) {
        state.viewer.removeAllSurfaces();
        state.viewer.render();
    }

    // Clean up overlay models
    _clearOverlayModels();

    // ── Reset JS state to factory defaults ──
    state.jobId = null;
    state.proteinData = null;
    state.ligandData = null;
    state.genMode = 'denovo';
    state.fixedAtoms.clear();
    state.heavyAtomMap.clear();
    state.generatedResults = [];
    state.activeResultIdx = -1;
    state.allGeneratedResults = [];
    state.iterationIdx = 0;
    state.surfaceOn = false;
    if (opts.resetWorkflow) {
        state.workflowType = 'sbdd';
        state.viewMode = 'complex';
    } else {
        state.viewMode = state.workflowType === 'lbdd' ? 'ligand' : 'complex';
    }
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
    state._hoveredResidue = null;
    state._cachedBindingSiteSerials = null;
    state.interactionShapes = [];
    state.showingInteractions = null;
    state._conformerJobId = null;
    state.activeView = 'ref';
    state.refProperties = null;

    // Active learning state
    state._alFinetuned = false;
    state._alCkptPath = null;
    state._alCheckpointSaved = false;
    state._alRound = 0;
    state._alCancelling = false;
    _selectedCkptPath = _initialCkptPath;
    _updateModelCard();

    // Original ligand (clear stale cross-session data)
    state.originalLigand = null;

    // De novo mode state
    state.numHeavyAtoms = null;
    state.pocketOnlyMode = false;
    state.lbddScratchMode = false;

    // ── Close AL modals if open ──
    document.getElementById('al-explain-modal')?.classList.add('hidden');
    document.getElementById('al-progress-modal')?.classList.add('hidden');

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
    // Reset RCSB fetch UI
    const rcsbInput = document.getElementById('rcsb-pdb-input');
    if (rcsbInput) rcsbInput.value = '';
    document.getElementById('rcsb-success')?.classList.add('hidden');
    const rcsbBtn = document.getElementById('rcsb-fetch-btn');
    if (rcsbBtn) rcsbBtn.disabled = false;
    // Restore all upload options and guidance banners (hidden by dynamic UI logic)
    document.getElementById('sbdd-guidance')?.classList.remove('hidden');
    document.getElementById('lbdd-guidance')?.classList.remove('hidden');
    document.getElementById('ligand-upload-group')?.classList.remove('hidden');
    document.getElementById('smiles-input-group')?.classList.remove('hidden');

    // Clear file inputs if requested
    if (opts.clearFileInputs) {
        const protFile = document.getElementById('protein-file');
        if (protFile) { protFile.value = ''; protFile.disabled = false; delete protFile.dataset.wasFrozen; }
        const ligFile = document.getElementById('ligand-file');
        if (ligFile) { ligFile.value = ''; ligFile.disabled = false; delete ligFile.dataset.wasFrozen; }
    }

    // Release stuck focus/drag state on upload areas
    document.querySelectorAll('.upload-area').forEach(a => a.classList.remove('drag-over'));
    if (document.activeElement) document.activeElement.blur();

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
    _hideFailureCard();
    const list = document.getElementById('results-list');
    if (list) { list.classList.add('hidden'); list.innerHTML = ''; }
    document.getElementById('results-placeholder')?.classList.remove('hidden');
    document.getElementById('gen-summary')?.classList.add('hidden');
    document.getElementById('bulk-actions')?.classList.add('hidden');
    document.getElementById('gen-hs-controls')?.classList.add('hidden');
    document.getElementById('save-section')?.classList.add('hidden');
    document.getElementById('analysis-section')?.classList.add('hidden');
    document.getElementById('ref-ligand-card-section')?.classList.add('hidden');
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
        btn.disabled = !!opts.generateBtnDisabled;
        btn.innerHTML = '<span class="btn-icon"></span> Generate Ligands';
        btn.onclick = startGeneration;
    }
    // Hide Ligand button
    const hideLigBtn = document.getElementById('btn-hide-ligand');
    if (hideLigBtn) { hideLigBtn.textContent = '\u25CB Hide Ligand'; hideLigBtn.classList.remove('active'); }
    // Status badge & device badge
    setBadge('idle', 'Ready');
    _updateDeviceBadge('NO GPU', 'idle');
    // Reset generation settings (mode, filters, etc.)
    _resetGenerationSettings();
    // Hide sidebar sections that need inputs
    _updateSidebarVisibility();
    // Unfreeze sidebar if it was frozen
    _freezeSidebar(false);
}

/**
 * Return to the landing / checkpoint-selection page, fully resetting
 * all application state so the user can start from scratch.
 */
function returnToLanding() { // NOSONAR
    if (state._generating && state.jobId) {
        authFetch(`${API_BASE}/cancel/${state.jobId}`, { method: 'POST' }).catch(() => { /* fire-and-forget cancel */ });
    }
    _resetAppState({ generateBtnDisabled: false, resetWorkflow: true });

    // ── Transition back to landing page ──
    document.getElementById('main-app').classList.add('hidden');
    const landing = document.getElementById('landing-page');
    landing.classList.remove('hidden', 'fade-out');

    // Reset landing page status text and launch button
    const status = document.getElementById('landing-status');
    if (status) { status.classList.add('hidden'); status.textContent = ''; }
    const launchBtn = document.getElementById('launch-btn');
    if (launchBtn) {
        launchBtn.disabled = !_selectedCkptPath;
        launchBtn.innerHTML = 'Launch';
    }
}

/**
 * Reset the current session without returning to the landing page.
 * Clears uploads, viewer, ligand/protein state, conformer picker,
 * SMILES input, and results so the user can re-upload fresh.
 */
function resetCurrentSession(silent = false) { // NOSONAR
    if (state._generating && state.jobId) {
        authFetch(`${API_BASE}/cancel/${state.jobId}`, { method: 'POST' }).catch(() => { /* fire-and-forget cancel */ });
    }
    _resetAppState({ generateBtnDisabled: true, clearFileInputs: true });
    _applyWorkflowMode(state.workflowType);
    if (!silent) showToast('Session reset — upload new files to start', 'info');
}

function initViewer() {
    const container = document.getElementById('viewer3d');
    state.viewer = $3Dmol.createViewer(container, {
        backgroundColor: COLORS.bgViewer,
        antialias: true,
        id: 'mol-viewer',
        hoverDuration: 250,
    });
    // NOTE: Do NOT call setClickable here — no models exist yet.
    // setClickable is applied in renderLigand() and reapplied after every
    // setStyle call in reapplyClickable().
    state.viewer.render();
    _attachLassoListeners();

    // Track whether mouse is inside the viewer canvas
    state._mouseInViewer = false;
    container.addEventListener('mouseenter', function () {
        state._mouseInViewer = true;
    });

    // Clear residue hover when cursor leaves the viewer canvas
    container.addEventListener('mouseleave', function () {
        state._mouseInViewer = false;
        // Cancel 3Dmol's pending hoverTimeout to prevent stale hover-in callbacks
        if (state.viewer && state.viewer.hoverTimeout) {
            clearTimeout(state.viewer.hoverTimeout);
            state.viewer.hoverTimeout = null;
        }
        const card = document.getElementById('residue-hover-card');
        if (card) card.style.display = 'none';
        _clearHoverHighlight();
    });
}

async function checkBackend() {
    const controller = new AbortController();
    const tid = setTimeout(() => controller.abort(), 5000);
    try {
        const resp = await authFetch(`${API_BASE}/health`, { signal: controller.signal });
        clearTimeout(tid);
        if (!resp.ok) throw new Error('Server error');
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
            deviceBadge.style.fontSize = '13px';
            dot.parentElement.insertBefore(deviceBadge, dot);
        }
        // Badge stays idle until Generate is clicked
        _updateDeviceBadge('NO GPU', 'idle');
    } catch (e) {
        clearTimeout(tid);
        console.debug('Status check failed:', e.message);
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

async function uploadProtein(input) { // NOSONAR
    const file = input.files[0];
    if (!file) return;

    setBadge('loading', 'Uploading…');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await authFetch(`${API_BASE}/upload/protein`, {
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
        state.allGeneratedResults = [];
        state.iterationIdx = 0;
        state.ligandData = null;
        state.fixedAtoms.clear();
        state.heavyAtomMap.clear();
        state.refLigandVisible = true;
        state.genHsVisible = false;
        state.genUsedOptimization = false;
        // Reset Hide Ligand button
        const hideLigBtn = document.getElementById('btn-hide-ligand');
        if (hideLigBtn) { hideLigBtn.textContent = '\u25CB Hide Ligand'; hideLigBtn.classList.remove('active'); }
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
        document.getElementById('bulk-actions')?.classList.add('hidden');
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
        // Clear stale RCSB UI if user previously used RCSB fetch
        document.getElementById('rcsb-success')?.classList.add('hidden');
        const rcsbIn = document.getElementById('rcsb-pdb-input');
        if (rcsbIn) rcsbIn.value = '';
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
        let errMsg = 'Failed to upload protein file';
        try {
            const parsed = JSON.parse(e.message);
            if (parsed.detail) errMsg = parsed.detail;
        } catch (parseErr) {
            console.debug('Error parse failed:', parseErr.message);
            if (e.message && e.message.length < 300) errMsg = e.message;
        }
        showToast(errMsg, 'error');
    }
}


/**
 * Fetch a PDB structure from RCSB by PDB ID.
 * Downloads protein + ligand via the server, then loads them
 * into the viewer (same behavior as manual upload).
 */
async function fetchFromRCSB() {
    const input = document.getElementById('rcsb-pdb-input');
    const pdbId = (input.value || '').trim().toUpperCase();
    if (!pdbId || pdbId.length !== 4 || !/^[A-Za-z0-9]{4}$/.test(pdbId)) {
        showToast('Enter a valid 4-character PDB ID', 'error');
        return;
    }

    const fetchBtn = document.getElementById('rcsb-fetch-btn');
    fetchBtn.disabled = true;
    setBadge('loading', 'Fetching from RCSB…');

    try {
        const resp = await authFetch(`${API_BASE}/fetch-rcsb`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pdb_id: pdbId }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // ── Reset state from any previous job (same as uploadProtein) ──
        if (state.generatedModel !== null) {
            state.viewer.removeModel(state.generatedModel);
            state.generatedModel = null;
        }
        if (state.ligandModel !== null) {
            state.viewer.removeModel(state.ligandModel);
            state.ligandModel = null;
        }
        state.atomLabels.forEach(l => state.viewer.removeLabel(l));
        state.atomLabels = [];
        state.selectionSpheres.forEach(s => state.viewer.removeShape(s));
        state.selectionSpheres = [];
        state.selectionSphereMap.clear();
        if (state.surfaceOn) {
            state.viewer.removeAllSurfaces();
            state.surfaceOn = false;
            const surfBtn = document.getElementById('btn-surface');
            if (surfBtn) surfBtn.classList.remove('active');
        }
        state.generatedResults = [];
        state.activeResultIdx = -1;
        state.allGeneratedResults = [];
        state.iterationIdx = 0;
        state.ligandData = null;
        state.fixedAtoms.clear();
        state.heavyAtomMap.clear();
        state.refLigandVisible = true;
        state.genHsVisible = false;
        state.genUsedOptimization = false;
        state.pocketOnlyMode = false;
        state.lbddScratchMode = false;
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
        const rl = document.getElementById('results-list');
        if (rl) { rl.classList.add('hidden'); rl.innerHTML = ''; }
        document.getElementById('results-placeholder')?.classList.remove('hidden');
        document.getElementById('gen-summary')?.classList.add('hidden');
        document.getElementById('bulk-actions')?.classList.add('hidden');
        document.getElementById('gen-hs-controls')?.classList.add('hidden');
        document.getElementById('save-section')?.classList.add('hidden');
        // Reset generation mode
        state.genMode = 'denovo';
        const denovoRadio = document.querySelector('input[name="gen-mode"][value="denovo"]');
        if (denovoRadio) denovoRadio.checked = true;
        document.getElementById('atom-selection-section').classList.add('hidden');
        document.getElementById('fragment-growing-opts')?.classList.add('hidden');
        _resetGenerationSettings();
        // Reset metrics
        const mp = document.getElementById('metrics-panel');
        if (mp) mp.classList.add('hidden');
        const ml = document.getElementById('metrics-log');
        if (ml) ml.innerHTML = '';

        // ── Store protein data ──
        state.jobId = data.job_id;
        state.proteinData = {
            job_id: data.job_id,
            filename: data.protein_filename,
            format: '.pdb',
            pdb_data: data.pdb_data,
        };

        // Update protein upload UI to show success
        document.getElementById('protein-placeholder').classList.add('hidden');
        document.getElementById('protein-success').classList.remove('hidden');
        document.getElementById('protein-filename').textContent = data.protein_filename;
        document.getElementById('viewer-overlay').classList.add('hidden');

        // Update RCSB success indicator
        document.getElementById('rcsb-success').classList.remove('hidden');

        // Render protein
        renderProtein(data.pdb_data, '.pdb');

        if (data.ligand_found && data.ligand) {
            // ── Ligand was found — load it ──
            state.ligandData = data.ligand;
            state.ligandData._atomByIdx = new Map();
            data.ligand.atoms.forEach(a => state.ligandData._atomByIdx.set(a.idx, a));

            // Build heavy-atom index map
            state.heavyAtomMap = new Map();
            let heavyIdx = 0;
            data.ligand.atoms.forEach(a => {
                if (a.atomicNum !== 1) {
                    state.heavyAtomMap.set(a.idx, heavyIdx);
                    heavyIdx++;
                }
            });

            // Update ligand upload UI
            document.getElementById('ligand-placeholder').classList.add('hidden');
            document.getElementById('ligand-success').classList.remove('hidden');
            document.getElementById('ligand-filename').textContent = data.ligand.filename;

            // Populate ligand info card
            const infoCard = document.getElementById('ligand-info');
            infoCard.classList.remove('hidden');
            document.getElementById('ligand-smiles').textContent = data.ligand.smiles_noH || data.ligand.smiles || 'N/A';
            document.getElementById('ligand-heavy-atoms').textContent = data.ligand.num_heavy_atoms;
            document.getElementById('ligand-total-atoms').textContent = data.ligand.num_atoms;
            state.refHasExplicitHs = data.ligand.has_explicit_hs || false;
            state.numHeavyAtoms = data.ligand.num_heavy_atoms;
            _showNumHeavyAtomsField(data.ligand.num_heavy_atoms, 'from RCSB — editable');

            // Ref affinity (not applicable for RCSB-fetched)
            const refAffEl = document.getElementById('ligand-ref-affinity');
            if (refAffEl) refAffEl.closest('.info-row')?.classList.add('hidden');

            // Populate property filter panel
            if (data.ligand.ref_properties) {
                populatePropertyFilterPanel(data.ligand.ref_properties);
            }
            state.refProperties = data.ligand.ref_properties || null;
            _showRefLigandCard();

            renderLigand(data.ligand.sdf_data);

            // Hide SBDD guidance and pocket-only toggle
            document.getElementById('sbdd-guidance')?.classList.add('hidden');
            document.getElementById('pocket-only-toggle')?.classList.add('hidden');

            // Enable all generation modes
            document.querySelectorAll('input[name="gen-mode"]').forEach(r => {
                r.disabled = false;
                r.closest('.mode-option')?.classList.remove('hidden');
            });
            document.getElementById('filter-cond-substructure')?.closest('.setting-group')?.classList.remove('hidden');

            document.getElementById('rcsb-loaded-label').textContent =
                `${pdbId} loaded (protein + ligand: ${data.ligand.ligand_id})`;
            setBadge('success', 'Complex loaded');
            showToast(`Loaded ${pdbId} from RCSB: protein + ligand ${data.ligand.ligand_id}`, 'success');

            _fetchAndRenderPriorCloudPreview();
        } else {
            // ── No ligand found — show protein, let user choose next step ──
            document.getElementById('rcsb-loaded-label').textContent =
                `${pdbId} loaded (protein only — no ligand found)`;

            // Show pocket-only toggle so user can choose de novo
            const pocketToggle = document.getElementById('pocket-only-toggle');
            if (pocketToggle) pocketToggle.classList.remove('hidden');

            setBadge('success', 'Protein loaded');
            showToast(`No ligand found in ${pdbId} — upload a ligand or continue in pocket-only de novo mode`, 'info');
        }

        updateGenerateBtn();
        _updateSidebarVisibility();
    } catch (e) {
        console.error(e);
        setBadge('error', 'RCSB fetch failed');
        let errMsg = `Failed to fetch PDB ${pdbId} from RCSB`;
        try {
            const parsed = JSON.parse(e.message);
            if (parsed.detail) errMsg = parsed.detail;
        } catch (_parseErr) {
            if (e.message && e.message.length < 300) errMsg = e.message;
        }
        showToast(errMsg, 'error');
    } finally {
        fetchBtn.disabled = false;
    }
}


async function uploadLigand(input) { // NOSONAR
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
            resp = await authFetch(`${API_BASE}/upload/molecule`, {
                method: 'POST',
                body: formData,
            });
        } else {
            resp = await authFetch(`${API_BASE}/upload/ligand/${state.jobId}`, {
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
        state.allGeneratedResults = [];
        state.iterationIdx = 0;
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
        document.getElementById('bulk-actions')?.classList.add('hidden');
        document.getElementById('gen-hs-controls')?.classList.add('hidden');
        document.getElementById('save-section')?.classList.add('hidden');
        document.getElementById('analysis-section')?.classList.add('hidden');
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
        setBadge('success', state.proteinData ? 'Complex loaded' : 'Ligand loaded');
        showToast(`Ligand loaded: ${data.filename} (${data.num_heavy_atoms} heavy atoms)`, 'success');

        // ── De novo helpers: populate num_heavy_atoms from reference ligand ──
        state.numHeavyAtoms = data.num_heavy_atoms;
        _showNumHeavyAtomsField(data.num_heavy_atoms, 'from reference — editable');

        // ── Populate property filter panel with reference-based defaults ──
        if (data.ref_properties) {
            populatePropertyFilterPanel(data.ref_properties);
        }

        // Store ref properties and show the reference ligand card in right panel
        state.refProperties = data.ref_properties || null;
        _showRefLigandCard();

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
        if (isLBDD) {
            // LBDD: hide guidance, SMILES input, and scratch button
            document.getElementById('lbdd-guidance')?.classList.add('hidden');
            document.getElementById('smiles-input-group')?.classList.add('hidden');
            document.getElementById('lbdd-scratch-group')?.classList.add('hidden');
        } else {
            // SBDD: hide guidance and pocket-only toggle
            document.getElementById('sbdd-guidance')?.classList.add('hidden');
            document.getElementById('pocket-only-toggle')?.classList.add('hidden');
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
        // Extract server error message (FastAPI returns {"detail": "..."})
        let errMsg = 'Failed to upload ligand file';
        try {
            const parsed = JSON.parse(e.message);
            if (parsed.detail) errMsg = parsed.detail;
        } catch (parseErr) {
            console.debug('Error parse failed:', parseErr.message);
            if (e.message && e.message.length < 300) errMsg = e.message;
        }
        showToast(errMsg, 'error');
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
        const resp = await authFetch(`${API_BASE}/create-denovo-job`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: state.jobId,
                num_heavy_atoms: state.numHeavyAtoms,
                workflow_type: 'sbdd',
            }),
        });
        if (resp.ok) {
            await resp.json();
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
        const resp = await authFetch(`${API_BASE}/create-denovo-job`, {
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
    const val = Number.parseInt(input?.value);
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
    btn.textContent = '…';
    setBadge('loading', 'Generating conformers…');

    try {
        const resp = await authFetch(`${API_BASE}/generate-conformers`, {
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
    file.text().then(raw => {
        const text = raw.trim();
        // Take first line, first column (space/tab separated)
        const firstSmiles = text.split('\n')[0].split(/\s/)[0].trim();
        if (firstSmiles) {
            document.getElementById('smiles-input').value = firstSmiles;
            document.getElementById('smiles-file-name').textContent = file.name;
        }
    });
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
        const eStr = conf.energy == null ? '—' : conf.energy.toFixed(2) + ' kcal/mol';
        opt.textContent = `Conformer #${rank + 1}  —  ${eStr}`;
        sel.appendChild(opt);
    });
    sel.onchange = () => _selectConformer(jobId, Number.parseInt(sel.value), conformers);
    grid.appendChild(sel);

    // Auto-select lowest-energy conformer (first in sorted list)
    if (sorted.length > 0) {
        _selectConformer(jobId, sorted[0]._origIdx, conformers);
    }
}

async function _selectConformer(jobId, confIdx, conformers) {
    setBadge('loading', 'Loading conformer…');

    try {
        const resp = await authFetch(`${API_BASE}/select-conformer/${jobId}`, {
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
    state._cachedBindingSiteSerials = null;
    applyProteinStyle();
    reapplyHoverable();
    if (!state._restoringSession) state.viewer.zoomTo();
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
    // Base cartoon — use bolder style in protein-only view
    const isProteinView = state.viewMode === 'protein';
    state.viewer.setStyle(
        { model: state.proteinModel },
        {
            cartoon: {
                color: 'spectrum',
                opacity: isProteinView ? 1 : 0.85,
                thickness: isProteinView ? 0.3 : 0.2,
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
    // Return cached result if available
    if (state._cachedBindingSiteSerials) return state._cachedBindingSiteSerials;

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
    state._cachedBindingSiteSerials = serials;
    return serials;
}

function renderLigand(sdfData) {
    // Remove previous ligand model if re-uploading
    if (state.ligandModel !== null) {
        state.viewer.removeModel(state.ligandModel);
        state.ligandModel = null;
    }
    state.ligandModel = state.viewer.addModel(sdfData, 'sdf');
    state._cachedBindingSiteSerials = null;

    // 1. Apply visual style
    applyLigandStyleBase();

    // 2. Mark atoms clickable AFTER the model is added + styled
    reapplyClickable();

    if (!state._restoringSession) state.viewer.zoomTo({ model: state.ligandModel }, 300);
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
// Modes that support shift+click atom selection
const _ATOM_SELECT_MODES = new Set([
    'substructure_inpainting',
    'scaffold_hopping',
    'scaffold_elaboration',
    'linker_inpainting',
    'core_growing',
]);

function reapplyClickable() {
    if (!state.ligandModel) return;

    state.viewer.setClickable(
        { model: state.ligandModel },
        true,
        function (atom, viewer, event) {
            if (!state.ligandData) return;
            // Allow atom selection in all conditional modes
            if (!_ATOM_SELECT_MODES.has(state.genMode)) return;
            // Require Shift+click for atom selection
            if (!event?.shiftKey) return;
            // Only allow selecting heavy atoms (skip H)
            if (atom.elem === 'H') return;
            const idx = findLigandAtomIdx(atom);
            if (idx !== null) {
                // Double-check it's not H
                const atomInfo = state.ligandData._atomByIdx.get(idx);
                if (atomInfo?.atomicNum === 1) return;
                toggleAtom(idx);
            }
        }
    );
}

// ── RAF-batched render for hover — prevents multiple render() calls per frame ──
let _hoverRafId = null;
function _scheduleHoverRender() {
    if (_hoverRafId) return;
    _hoverRafId = requestAnimationFrame(() => {
        _hoverRafId = null;
        if (state.viewer) state.viewer.render();
    });
}

function _setHoverHighlight(resi, chain) {
    // Re-apply base protein styles (fast with binding-site caching)
    applyProteinStyle();
    // Add hover overlay on top (addStyle doesn't clear existing)
    const sel = { model: state.proteinModel, resi: resi, chain: chain };
    // Semi-transparent lavender cartoon highlight (visible but subtle)
    state.viewer.addStyle(sel, {
        cartoon: { color: COLORS.hoverResidue, opacity: 0.35 },
    });
    // Show full atom sticks with proper atom-type colors (Jmol scheme)
    state.viewer.addStyle(sel, {
        stick: { colorscheme: 'Jmol', radius: 0.15 },
    });
    _scheduleHoverRender();
}

function _clearHoverHighlight() {
    state._hoveredResidue = null;
    applyProteinStyle();  // resets to base styles (fast with cache)
    _scheduleHoverRender();
}

function reapplyHoverable() {
    if (!state.proteinModel) return;
    const card = document.getElementById('residue-hover-card');
    if (!card) return;
    state.viewer.setHoverable(
        { model: state.proteinModel },
        true,
        function (atom) {
            if (!atom) return;
            // Ignore stale hover callbacks fired after mouse left viewer
            if (!state._mouseInViewer) return;
            // Update structured card content
            const nameEl = document.getElementById('residue-hover-name');
            const chainEl = document.getElementById('residue-hover-chain');
            if (nameEl) nameEl.textContent = atom.resn + ' ' + atom.resi;
            if (chainEl) chainEl.textContent = 'Chain ' + atom.chain;
            card.style.display = 'flex';

            // Skip re-render if same residue is already highlighted
            if (
                state._hoveredResidue &&
                state._hoveredResidue.resi === atom.resi &&
                state._hoveredResidue.chain === atom.chain
            ) return;

            // Highlight new residue
            state._hoveredResidue = { resi: atom.resi, chain: atom.chain };
            _setHoverHighlight(atom.resi, atom.chain);
        },
        function () {
            card.style.display = 'none';
            if (!state._hoveredResidue) return;
            _clearHoverHighlight();
        }
    );
}

// ── Dual-sphere selection tuning ──
const SELECTION_SPHERE = {
    inner: { radius: 0.36, opacity: 0.3 },
    outer: { radius: 0.48, opacity: 0.55, wireframe: true, linewidth: 1.5 },
};

function _addDualSphere(viewer, pos, color) {
    const inner = viewer.addSphere({
        center: pos,
        radius: SELECTION_SPHERE.inner.radius,
        color: color,
        opacity: SELECTION_SPHERE.inner.opacity,
    });
    const outer = viewer.addSphere({
        center: pos,
        radius: SELECTION_SPHERE.outer.radius,
        color: color,
        opacity: SELECTION_SPHERE.outer.opacity,
        wireframe: SELECTION_SPHERE.outer.wireframe,
        linewidth: SELECTION_SPHERE.outer.linewidth,
    });
    return { inner, outer };
}

function _removeDualSphere(viewer, handle) {
    if (!handle) return;
    if (handle.inner) viewer.removeShape(handle.inner);
    if (handle.outer) viewer.removeShape(handle.outer);
}

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
                fontSize: 13,
                fontColor: COLORS.labelDefault,
                backgroundColor: 'transparent',
                backgroundOpacity: 0,
                showBackground: false,
                alignment: 'center',
            }
        );
        state.atomLabels.push(label);

        // Dual-sphere glow around selected atoms
        if (state.fixedAtoms.has(atom.idx)) {
            const pos = { x: atom.x, y: atom.y, z: atom.z };
            const pair = _addDualSphere(state.viewer, pos, COLORS.fixed);
            state.selectionSpheres.push(pair.inner, pair.outer);
            state.selectionSphereMap.set(atom.idx, pair);
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

function onModeChange(mode) { // NOSONAR
    state.genMode = mode;

    // Show/hide number of heavy atoms field (only for de novo)
    const numHAGroup = document.getElementById('num-heavy-atoms-group');
    if (numHAGroup) {
        if (mode === 'denovo') {
            numHAGroup.classList.remove('hidden');
            // Pre-populate from reference ligand if available
            if (state.ligandData?.num_heavy_atoms && !state.numHeavyAtoms) {
                const input = document.getElementById('num-heavy-atoms');
                if (input) input.value = state.ligandData.num_heavy_atoms;
                state.numHeavyAtoms = state.ligandData.num_heavy_atoms;
            }
        } else {
            numHAGroup.classList.add('hidden');
        }
    }

    // Show/hide atom selection for all conditional modes that support shift+click
    const atomSection = document.getElementById('atom-selection-section');
    atomSection.classList.toggle('hidden', !_ATOM_SELECT_MODES.has(mode));

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

    // Clear manual atom selection when switching modes
    state.fixedAtoms.clear();
    updateSelectionUI();
    // Remove stale selection spheres (fixedAtoms is empty → removes all)
    _rebuildAllSpheres();

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

    // For conditional modes with auto-highlight, also update the auto highlight
    // spheres to reflect the user's manual override
    if (state.genMode !== 'substructure_inpainting' && _ATOM_SELECT_MODES.has(state.genMode)) {
        _syncAutoHighlightWithSelection();
    }

    // Update prior cloud to reflect the selected fragment.
    // Only substructure_inpainting auto-shifts the prior to selected atoms' COM.
    // Other conditional modes use the standard prior cloud positioning
    // (controlled by anisotropic prior / ref ligand COM prior checkboxes).
    if (state.genMode === 'substructure_inpainting') {
        _debouncedSubstructurePriorCloud();
    } else if (_ATOM_SELECT_MODES.has(state.genMode)) {
        // For scaffold/linker/core modes: re-fetch the cloud preview
        // The server computes the correct grow_size from the molecule's mask
        // which may change based on the editing context
        _debouncedCloudPreview();
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
        const pos = { x: atom.x, y: atom.y, z: atom.z };
        const pair = _addDualSphere(state.viewer, pos, COLORS.fixed);
        state.selectionSphereMap.set(idx, pair);
        state.selectionSpheres.push(pair.inner, pair.outer);
    } else {
        const pair = state.selectionSphereMap.get(idx);
        if (pair) {
            _removeDualSphere(state.viewer, pair);
            state.selectionSphereMap.delete(idx);
            for (const s of [pair.inner, pair.outer]) {
                const i = state.selectionSpheres.indexOf(s);
                if (i >= 0) state.selectionSpheres.splice(i, 1);
            }
        }
    }
    state.viewer.render();
}

// ── Lasso Tool ──────────────────────────────────────────────────────────────

let _lassoListenersAttached = false;

const _lassoState = {
    active: false,
    pending: false,
    startX: 0, startY: 0,
    points: [],
    canvas: null,
    ctx: null,
};

/** Convert a mouse event to coords local to #viewer3d container (immune to e.target changes). */
function _containerXY(e) {
    const rect = document.getElementById('viewer3d').getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

/** Convert 3D model position to coords local to #viewer3d container. */
function _modelToContainerXY(atom) {
    const p = state.viewer.modelToScreen({ x: atom.x, y: atom.y, z: atom.z });
    if (!p) return null;
    const rect = document.getElementById('viewer3d').getBoundingClientRect();
    const docEl = document.documentElement;
    return {
        x: p.x - (rect.left + window.scrollX - docEl.clientLeft),
        y: p.y - (rect.top + window.scrollY - docEl.clientTop)
    };
}

function _pointInPolygon(px, py, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].x, yi = polygon[i].y;
        const xj = polygon[j].x, yj = polygon[j].y;
        if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

function _ensureLassoCanvas() {
    if (_lassoState.canvas) return;
    const container = document.getElementById('viewer3d');
    if (!container) return;
    const c = document.createElement('canvas');
    c.className = 'lasso-canvas';
    c.width = container.clientWidth;
    c.height = container.clientHeight;
    container.appendChild(c);
    _lassoState.canvas = c;
    _lassoState.ctx = c.getContext('2d');
}

function _removeLassoCanvas() {
    if (_lassoState.canvas) {
        _lassoState.canvas.remove();
        _lassoState.canvas = null;
        _lassoState.ctx = null;
    }
    const container = document.getElementById('viewer3d');
    if (container) container.classList.remove('lasso-active');
}

function _drawLasso() {
    const { ctx, canvas, points } = _lassoState;
    if (!ctx || points.length < 2) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Fill
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    ctx.closePath();
    ctx.fillStyle = 'rgba(159,138,172,0.12)';
    ctx.fill();

    // Stroke
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    ctx.strokeStyle = '#9c8aac';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Closing dashed line
    const last = points.at(-1);
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(points[0].x, points[0].y);
    ctx.strokeStyle = '#9c8aac';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);
}

function _completeLasso() {
    const { points } = _lassoState;
    _lassoState.active = false;
    _lassoState.pending = false;
    _removeLassoCanvas();

    if (points.length < 3) return;

    const viewer = state.viewer;
    if (!viewer || !state.ligandData) return;

    let changed = false;
    state.ligandData.atoms.forEach(atom => {
        if (atom.atomicNum === 1) return;
        if (state.fixedAtoms.has(atom.idx)) return;
        const screen = _modelToContainerXY(atom);
        if (!screen) return;
        if (_pointInPolygon(screen.x, screen.y, points)) {
            state.fixedAtoms.add(atom.idx);
            changed = true;
        }
    });

    if (changed) {
        _rebuildAllSpheres();
        updateSelectionUI();
        if (state.genMode !== 'substructure_inpainting' && _ATOM_SELECT_MODES.has(state.genMode)) {
            _syncAutoHighlightWithSelection();
        }
        if (state.genMode === 'substructure_inpainting') {
            _updateSubstructurePriorCloud();
        } else if (_ATOM_SELECT_MODES.has(state.genMode)) {
            _debouncedCloudPreview();
        }
        viewer.render();
    }
}

function _cancelLasso() {
    _lassoState.active = false;
    _lassoState.pending = false;
    _lassoState.points = [];
    _removeLassoCanvas();
}

/** Find nearest heavy atom to screen coords and toggle it (manual shift+click). */
function _handleShiftClick(screenX, screenY) {
    if (!state.viewer || !state.ligandData) return;
    const CLICK_RADIUS = 20;
    let bestIdx = null;
    let bestDist = CLICK_RADIUS;
    state.ligandData.atoms.forEach(atom => {
        if (atom.atomicNum === 1) return;
        const screen = _modelToContainerXY(atom);
        if (!screen) return;
        const dist = Math.hypot(screen.x - screenX, screen.y - screenY);
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = atom.idx;
        }
    });
    if (bestIdx !== null) {
        toggleAtom(bestIdx);
    }
}

function _attachLassoListeners() {
    if (_lassoListenersAttached) return;
    _lassoListenersAttached = true;
    const container = document.getElementById('viewer3d');
    if (!container) return;

    const DRAG_THRESHOLD = 5;
    const MIN_POINT_DIST = 3;

    container.addEventListener('mousedown', function (e) {
        if (!e.shiftKey) return;
        if (state.priorCloudPlacing) return;
        if (!_ATOM_SELECT_MODES.has(state.genMode)) return;
        if (e.button !== 0) return;

        const pos = _containerXY(e);
        _lassoState.startX = pos.x;
        _lassoState.startY = pos.y;
        _lassoState.points = [];
        _lassoState.active = false;
        _lassoState.pending = true;

        // Block 3Dmol.js from starting its Shift+drag zoom
        e.preventDefault();
        e.stopPropagation();
    }, true);

    container.addEventListener('mousemove', function (e) {
        if (!e.shiftKey) {
            if (_lassoState.active || _lassoState.pending) _cancelLasso();
            return;
        }
        if (state.priorCloudPlacing) return;
        if (!_lassoState.pending && !_lassoState.active) return;

        const pos = _containerXY(e);

        if (!_lassoState.active && _lassoState.pending && (e.buttons & 1)) {
            const dx = pos.x - _lassoState.startX;
            const dy = pos.y - _lassoState.startY;
            if (Math.hypot(dx, dy) >= DRAG_THRESHOLD) {
                _lassoState.active = true;
                _lassoState.points = [{ x: _lassoState.startX, y: _lassoState.startY }];
                _ensureLassoCanvas();
                container.classList.add('lasso-active');
            }
        }

        if (_lassoState.active) {
            const last = _lassoState.points.at(-1);
            const dx = pos.x - last.x;
            const dy = pos.y - last.y;
            if (Math.hypot(dx, dy) >= MIN_POINT_DIST) {
                _lassoState.points.push({ x: pos.x, y: pos.y });
                _drawLasso();
            }
        }

        e.preventDefault();
        e.stopPropagation();
    }, true);

    container.addEventListener('mouseup', function (e) {
        if (_lassoState.active) {
            _completeLasso();
            e.preventDefault();
            e.stopPropagation();
        } else if (_lassoState.pending) {
            // Didn't cross drag threshold → treat as shift+click
            _lassoState.pending = false;
            const pos = _containerXY(e);
            _handleShiftClick(pos.x, pos.y);
            e.preventDefault();
            e.stopPropagation();
        }
    }, true);

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && _lassoState.active) _cancelLasso();
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && state.molstarOpen) closeMolstar();
    });

    container.addEventListener('click', function (e) {
        if (!e.shiftKey) return;
        if (state.priorCloudPlacing) return;
        if (!_ATOM_SELECT_MODES.has(state.genMode)) return;
        e.stopPropagation();
    }, true);

    window.addEventListener('blur', _cancelLasso);
    window.addEventListener('resize', _cancelLasso);
}

function clearSelection() {
    state.fixedAtoms.clear();
    updateSelectionUI();
    _rebuildAllSpheres();
    if (state.genMode === 'substructure_inpainting') {
        _debouncedSubstructurePriorCloud();
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
        _debouncedSubstructurePriorCloud();
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
                const pos = { x: atom.x, y: atom.y, z: atom.z };
                const pair = _addDualSphere(state.viewer, pos, COLORS.fixed);
                state.selectionSpheres.push(pair.inner, pair.outer);
                state.selectionSphereMap.set(atom.idx, pair);
            }
        });
    }
    state.viewer.render();
}

function updateSelectionUI() {
    const container = document.getElementById('selected-atoms-list');

    if (state.fixedAtoms.size === 0) {
        container.innerHTML = '<span class="muted-text">No atoms marked for replacement</span>';
        // Only hide legend if no auto-highlight is showing either
        if (state.genMode === 'substructure_inpainting') {
            _hideInpaintingLegend();
        }
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

    // Show legend for all modes that support atom selection
    if (_ATOM_SELECT_MODES.has(state.genMode)) {
        const nSelected = state.fixedAtoms.size;
        const totalHeavy = state.ligandData?.num_heavy_atoms || 0;
        const fixedModes = ['core_growing', 'fragment_growing'];
        const isFixedMode = fixedModes.includes(state.genMode);
        _showInpaintingLegend(state.genMode, totalHeavy - nSelected, nSelected, isFixedMode);
    }
}

// =========================================================================
// View Controls
// =========================================================================

function setViewMode(mode) { // NOSONAR
    state.viewMode = mode;
    const hoverCard = document.getElementById('residue-hover-card');
    if (hoverCard) hoverCard.style.display = 'none';
    state._hoveredResidue = null;
    // Only toggle active class on view-mode buttons, not all toolbar buttons
    document.querySelectorAll('#view-complex, #view-protein, #view-ligand').forEach(
        b => b.classList.remove('active')
    );
    document.getElementById(`view-${mode}`).classList.add('active');

    if (mode === 'complex') {
        if (state.proteinModel) { applyProteinStyle(); reapplyHoverable(); }
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
            } else if (state.refLigandVisible) {
                updateAtomHighlights();
            } else {
                state.viewer.setStyle({ model: state.ligandModel }, {});
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
                { cartoon: { color: 'spectrum', opacity: 1, thickness: 0.3 } });
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
            reapplyHoverable();
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
            try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
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
            } else if (state.refLigandVisible) {
                updateAtomHighlights();
            } else {
                state.viewer.setStyle({ model: state.ligandModel }, {});
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
    if (!state.proteinModel) {
        showToast('Upload a protein first to toggle surface', 'error');
        return;
    }
    state.surfaceOn = !state.surfaceOn;
    const btn = document.getElementById('btn-surface');

    if (state.surfaceOn) {
        btn.classList.add('active');
        $3Dmol.setSyncSurface(true);
        state.viewer.addSurface(
            $3Dmol.SurfaceType.VDW,
            { opacity: 0.65, color: COLORS.lavender },
            { model: state.proteinModel }
        );
        state.viewer.render();
    } else {
        state.viewer.removeAllSurfaces();
        btn.classList.remove('active');
        state.viewer.render();
    }
}

// ===== Molstar Explorer =====

function toggleMolstar() {
    if (state.molstarOpen) {
        closeMolstar();
    } else {
        openMolstar();
    }
}

function openMolstar() {
    if (!state.proteinData?.pdb_data) {
        showToast('Load a protein structure first', 'warning');
        return;
    }
    const overlay = document.getElementById('molstar-overlay');
    const iframe = document.getElementById('molstar-frame');
    const btn = document.getElementById('btn-molstar');
    state.molstarOpen = true;
    state.molstarReady = false;
    overlay.style.display = 'flex';
    btn.classList.add('molstar-active');
    document.querySelector('.viewer-container')?.classList.add('molstar-freeze');
    iframe.src = '/static/molstar-embed.html';
}

function closeMolstar() {
    const overlay = document.getElementById('molstar-overlay');
    const iframe = document.getElementById('molstar-frame');
    const btn = document.getElementById('btn-molstar');
    state.molstarOpen = false;
    state.molstarReady = false;
    overlay.style.display = 'none';
    btn.classList.remove('molstar-active');
    document.querySelector('.viewer-container')?.classList.remove('molstar-freeze');
    iframe.src = '';
}

function sendStructureToMolstar() {
    const iframe = document.getElementById('molstar-frame');
    if (!iframe?.contentWindow || !state.molstarReady) return;

    let sdfData = null;
    let ligandLabel = 'Ligand';

    if (state.activeView === 'gen' && state.generatedResults?.length > 0) {
        const activeResult = state.generatedResults[state.activeResultIdx];
        sdfData = activeResult?.sdf_hs || activeResult?.sdf || null;
        ligandLabel = `Generated Ligand ${state.activeResultIdx + 1}`;
    }
    if (!sdfData && state.ligandData?.sdf_data) {
        sdfData = state.ligandData.sdf_data;
        if (state.ligandData.filename) {
            ligandLabel = state.ligandData.filename.replace(/\.(sdf|mol)$/i, '');
        }
    }

    // Patch SDF first line with ligand name for Mol* entity labeling
    if (sdfData) {
        const lines = sdfData.split('\n');
        if (lines.length > 0 && lines[0].trim() === '') {
            lines[0] = ligandLabel;
            sdfData = lines.join('\n');
        }
    }

    iframe.contentWindow.postMessage({
        type: 'load-structure',
        pdb: state.proteinData.pdb_data,
        sdf: sdfData,
        proteinFormat: state.proteinData.format || 'pdb',
        label: state.proteinData.pdb_id || 'Protein',
        ligandLabel: ligandLabel
    }, window.location.origin);
}

/**
 * Toggle full-atom stick representation of the entire protein.
 */
function toggleFullAtom() {
    if (!state.proteinModel) return;
    state.proteinFullAtom = !state.proteinFullAtom;
    document.getElementById('btn-full-atom')?.classList.toggle('active', state.proteinFullAtom);
    applyProteinStyle();
    reapplyHoverable();
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
    state._cachedBindingSiteSerials = null;
    applyProteinStyle();
    reapplyHoverable();
    state.viewer.render();
}

function centerOnLigand() {
    if (state.ligandModel) {
        state.viewer.zoomTo({ model: state.ligandModel }, 300);
        state.viewer.render();
    }
}

/**
 * Show confirmation dialog before clearing all generated ligands.
 * If there are accumulated rounds, warns about losing all history.
 */
function clearGeneratedLigands() {
    if (state.generatedResults.length === 0 && state.allGeneratedResults.length === 0) {
        showToast('No generated ligands to clear', 'warning');
        return;
    }
    const nRounds = state.allGeneratedResults.length;
    const totalLigs = state.allGeneratedResults.reduce(
        (sum, r) => sum + r.results.length, 0
    );
    const msg = document.getElementById('clear-confirm-msg');
    if (msg) {
        if (nRounds > 1) {
            msg.textContent = `This will permanently delete all ${totalLigs} generated ligands from ${nRounds} generation rounds.`;
        } else {
            msg.textContent = `This will permanently delete all ${totalLigs} generated ligand${totalLigs === 1 ? '' : 's'} and reset the generation history.`;
        }
    }
    openModal('clear-confirm-modal');
}

/**
 * Remove all overlay models from the viewer and clear the tracking array.
 */
function _clearOverlayModels() {
    if (!state.viewer) return;
    state.selectedOverlayModels.forEach(m => {
        try { state.viewer.removeModel(m); } catch (e) { console.debug('Model already removed:', e.message); }
    });
    state.selectedOverlayModels = [];
}

/**
 * Actually clear all generated ligands (called after user confirms).
 */
function _executeClearAll() { // NOSONAR
    closeModal('clear-confirm-modal');

    // Remove generated model from viewer
    if (state.generatedModel !== null) {
        state.viewer.removeModel(state.generatedModel);
        state.generatedModel = null;
    }

    // Clear state
    state.generatedResults = [];
    state.activeResultIdx = -1;
    state.allGeneratedResults = [];
    state.iterationIdx = 0;
    state.activeView = 'ref';
    state.originalLigand = null;

    // Hide original ligand card
    const origSection = document.getElementById('original-ligand-card-section');
    if (origSection) origSection.classList.add('hidden');

    // Clean up overlay models
    _clearOverlayModels();

    // Clear backend history (fire-and-forget)
    if (state.jobId) {
        authFetch(`${API_BASE}/clear-history/${state.jobId}`, { method: 'POST' })
            .catch(() => { /* best effort */ });
    }

    // Reset results panel UI
    const list = document.getElementById('results-list');
    list.classList.add('hidden');
    list.innerHTML = '';
    document.getElementById('results-placeholder').classList.remove('hidden');
    document.getElementById('gen-summary').classList.add('hidden');

    // Keep reference ligand card visible if ligand is still loaded;
    // just remove active highlight since there are no generated ligands
    const refCard = document.getElementById('ref-ligand-card');
    if (refCard) refCard.classList.remove('active');

    // Hide bulk-actions section
    const bulkActions = document.getElementById('bulk-actions');
    if (bulkActions) bulkActions.classList.add('hidden');

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

    // Hide analysis section (affinity + viz controls)
    const analysisSection = document.getElementById('analysis-section');
    if (analysisSection) analysisSection.classList.add('hidden');

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
    // Restore input-state badge if inputs are still loaded
    const inputBadge = _getInputBadgeText();
    if (inputBadge) setBadge('success', inputBadge);
    state.viewer.render();

    // Sidebar is unlocked when back in reference view with no results
    _syncSidebarLock();
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
    document.getElementById('fragment-growing-opts')?.classList.add('hidden');

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
    _set('filter-diversity', true);           // default: checked
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

    // ── Diversity threshold: reset to 0.8, show sub-option ──
    const divSlider = document.getElementById('diversity-threshold');
    if (divSlider) divSlider.value = '0.8';
    const divVal = document.getElementById('div-thresh-val');
    if (divVal) divVal.textContent = '0.8';
    const divGroup = document.getElementById('diversity-threshold-group');
    if (divGroup) divGroup.classList.remove('hidden');

    // ── Generation settings sliders ──
    const _setSlider = (id, valId, val) => {
        const sl = document.getElementById(id);
        const sp = document.getElementById(valId);
        if (sl) sl.value = val;
        if (sp) sp.textContent = val;
    };
    _setSlider('n-samples', 'n-samples-val', '100');
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
        // Restore: full if reference is active view or no gen model, dimmed otherwise
        if (state.activeView === 'ref' || state.generatedModel === null) {
            applyLigandStyleBase();
        } else {
            state.viewer.setStyle(
                { model: state.ligandModel },
                {
                    stick: { radius: 0.1, colorscheme: 'Jmol', opacity: 0.3 },
                    sphere: { radius: 0.15, colorscheme: 'Jmol', opacity: 0.3 },
                }
            );
        }
        reapplyClickable();
        // Re-show atom labels and selection spheres
        addAtomLabels();
        // Re-show auto-highlight if an inpainting mode is active
        const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
        if (autoModes.includes(state.genMode) && state.ligandData && state.jobId) {
            fetchAndHighlightInpaintingMask(state.genMode);
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
            try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
        });
        state.autoHighlightSpheres = [];
        _hideInpaintingLegend();
    }

    // Refresh 3D interactions (both show/hide paths need this)
    if (state.showingInteractions) {
        clearInteractions3D();
        showInteractions3D('both');
    }
    if (btn) {
        btn.textContent = state.refLigandVisible ? '\u25CB Hide Ligand' : '\u2299 Show Ligand';
        btn.classList.toggle('active', !state.refLigandVisible);
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

async function startGeneration() { // NOSONAR
    // Standard path requires ligandData; de novo pocket-only / scratch path
    // only needs numHeavyAtoms (and protein for SBDD).
    const hasDenovoInputs = state.pocketOnlyMode || state.lbddScratchMode;
    if (!state.jobId) return;
    if (!state.ligandData && !hasDenovoInputs) return;
    // SBDD also requires protein
    if (state.workflowType !== 'lbdd' && !state.proteinData) return;
    if (state._generating) return;
    _hideFailureCard();
    state._generating = true;

    const btn = document.getElementById('generate-btn');
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon"></span> Cancel Generation';
    btn.onclick = cancelGeneration;

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

    const nSamples = Number.parseInt(document.getElementById('n-samples').value);
    const batchSize = Number.parseInt(document.getElementById('batch-size').value);
    const steps = Number.parseInt(document.getElementById('integration-steps').value);
    const cutoff = Number.parseFloat(document.getElementById('pocket-cutoff').value);
    const isLBDD = state.workflowType === 'lbdd';

    // Build mode-specific payload — LBDD supports all the same gen modes as SBDD
    const mode = state.genMode;
    const payload = {
        job_id: state.jobId,
        workflow_type: state.workflowType,
        protein_path: isLBDD ? null : (state.proteinData?.filename || null),
        ligand_path: state.ligandData ? state.ligandData.filename : null,
        gen_mode: mode,
        fixed_atoms: (_ATOM_SELECT_MODES.has(mode) && state.fixedAtoms.size > 0)
            ? Array.from(state.fixedAtoms).map(
                idx => state.heavyAtomMap.get(idx)
            ).filter(idx => idx !== undefined)
            : [],
        n_samples: nSamples,
        batch_size: batchSize,
        integration_steps: steps,
        pocket_cutoff: isLBDD ? 6 : cutoff,

        // Random seed for reproducibility
        seed: (() => { const v = Number.parseInt(document.getElementById('gen-seed').value, 10); return Number.isNaN(v) ? 42 : v; })(),

        // De novo: num_heavy_atoms for placeholder ligand generation
        num_heavy_atoms: state.numHeavyAtoms || null,

        // Noise
        coord_noise_scale: document.getElementById('add-noise')?.checked
            ? Number.parseFloat(document.getElementById('noise-scale')?.value || '0.1')
            : 0,

        // Post-processing options
        sample_mol_sizes: document.getElementById('sample-mol-sizes')?.checked || false,
        filter_valid_unique: document.getElementById('filter-valid-unique')?.checked || false,
        filter_cond_substructure: document.getElementById('filter-cond-substructure')?.checked || false,
        filter_diversity: document.getElementById('filter-diversity')?.checked || false,
        diversity_threshold: Number.parseFloat(document.getElementById('diversity-threshold')?.value || '0.8'),
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
        payload.grow_size = Number.parseInt(document.getElementById('grow-size').value) || 5;

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
                const upResp = await authFetch(`${API_BASE}/upload_prior_center/${state.jobId}`, {
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

        const resp = await authFetch(`${API_BASE}/generate`, {
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
        _activeGenerationController = new AbortController();
        const data = await pollGeneration(state.jobId, progressFill, progressText, _activeGenerationController.signal);

        progressFill.style.width = '100%';
        progressText.textContent = `Done! ${data.n_generated} ligands in ${data.elapsed_time}s`;

        state.generatedResults = data.results;
        // Accumulate results for multi-round tracking
        state.allGeneratedResults.push({
            iteration: state.iterationIdx,
            results: data.results,
        });
        state.iterationIdx++;
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

        // Show generation warnings (e.g. fallback to de novo, sanitization issues)
        if (data.warnings && data.warnings.length > 0) {
            data.warnings.forEach(w => showToast(w, 'error'));
        }

    } catch (e) {
        if (e.name === 'AbortError') {
            console.debug('Generation polling aborted');
            // Clean up progress UI after cancel
            progressFill.style.width = '0%';
            progressText.textContent = '';
            progressContainer.classList.add('hidden');
            // Restore badge to input state
            const inputBadge = _getInputBadgeText();
            if (inputBadge) setBadge('success', inputBadge);
            else setBadge('idle', 'Ready');
            showToast('Generation cancelled', 'info');
        } else {
            console.error(e);
            progressFill.style.width = '0%';
            progressText.textContent = 'Generation failed';
            progressContainer.classList.add('hidden');
            setBadge('error', 'Failed');
            let errMsg = e.message || String(e);
            if (errMsg.includes('CUDA out of memory') || errMsg.includes('OutOfMemoryError')) {
                errMsg = 'CUDA out of memory \u2013 reduce batch size!';
            }
            showToast('Generation failed: ' + errMsg, 'error');
            _showFailureCard(errMsg, e.healthCheckAdvice || null);
        }
    } finally {
        _activeGenerationController = null;
    }

    // GPU released — revert badge to idle
    _updateDeviceBadge('NO GPU', 'idle');

    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon"></span> Generate Ligands';
    btn.onclick = startGeneration;
    state._generating = false;

    // Sync sidebar lock — stays locked if a generated ligand is active
    _syncSidebarLock();
}

/**
 * Cancel a running generation by notifying the server.
 * The pollGeneration loop will pick up the 'cancelled' status.
 */
async function cancelGeneration() {
    if (!state.jobId || !state._generating) return;
    const btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon"></span> Cancelling…';
    if (_activeGenerationController) {
        _activeGenerationController.abort();
        _activeGenerationController = null;
    }
    try {
        await authFetch(`${API_BASE}/cancel/${state.jobId}`, { method: 'POST' });
    } catch (e) {
        console.warn('Cancel request failed:', e);
    }
}

/**
 * Synchronise sidebar lock state with the current view.
 * After generation, the sidebar stays locked until the user is viewing
 * the reference ligand (activeView === 'ref'). This prevents ambiguity
 * about which ligand serves as the generation basis.
 *
 * During active generation (state._generating) this is a no-op —
 * the hard freeze from _freezeSidebar(true) takes precedence.
 */
function _syncSidebarLock() {
    if (state._generating) return;                       // hard-freeze active
    const hasResults = state.generatedResults.length > 0;
    const locked = hasResults && state.activeView !== 'ref';
    _freezeSidebar(locked);
    // Show / hide the hover hint overlay (only post-generation, not during)
    const overlay = document.getElementById('sidebar-lock-overlay');
    if (overlay) overlay.classList.toggle('hidden', !locked);
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
            // Only save original state on the first freeze; don't overwrite
            // if already frozen (e.g. generation freeze → post-gen lock)
            if (!('wasFrozen' in el.dataset)) {
                el.dataset.wasFrozen = el.disabled ? '1' : '0';
            }
            el.disabled = true;
        } else {
            // Restore only those that were not already disabled
            if (el.dataset.wasFrozen === '0') el.disabled = false;
            delete el.dataset.wasFrozen;
        }
    });
    if (freeze) {
        panel.classList.add('sidebar-frozen');
        // During a hard-freeze (generation), always hide the lock overlay
        // — it is only shown for the soft post-generation lock via _syncSidebarLock
        if (state._generating) {
            const overlay = document.getElementById('sidebar-lock-overlay');
            if (overlay) overlay.classList.add('hidden');
        }
    } else {
        panel.classList.remove('sidebar-frozen');
    }
}

/**
 * Show the standalone failure card in the right panel.
 * Does NOT touch #results-list — existing round cards are preserved.
 */
function _showFailureCard(errMsg, advice) {
    const card = document.getElementById('generation-failure-card');
    if (!card) return;
    const msgEl = document.getElementById('failure-card-message');
    if (msgEl) {
        msgEl.textContent = errMsg;
        msgEl.title = errMsg;
    }
    // Show/hide advice section
    let adviceEl = document.getElementById('failure-card-advice');
    if (!adviceEl && advice) {
        adviceEl = document.createElement('p');
        adviceEl.id = 'failure-card-advice';
        adviceEl.className = 'failure-card-advice';
        if (msgEl?.parentNode) {
            msgEl.parentNode.insertBefore(adviceEl, msgEl.nextSibling);
        }
    }
    if (adviceEl) {
        if (advice) {
            adviceEl.textContent = advice;
            adviceEl.classList.remove('hidden');
        } else {
            adviceEl.textContent = '';
            adviceEl.classList.add('hidden');
        }
    }
    card.classList.remove('hidden');
    // Wire retry button (clone-and-replace to avoid listener stacking)
    const retryBtn = document.getElementById('failure-card-retry-btn');
    if (retryBtn) {
        const fresh = retryBtn.cloneNode(true);
        retryBtn.parentNode.replaceChild(fresh, retryBtn);
        fresh.addEventListener('click', () => {
            _hideFailureCard();
            startGeneration();
        });
    }
}

/**
 * Hide the failure card. Called at generation start and on full reset.
 */
function _hideFailureCard() {
    const card = document.getElementById('generation-failure-card');
    if (card) card.classList.add('hidden');
}

/**
 * Poll /job/{id} until status is 'completed', 'failed', or 'cancelled'.
 * Uses recursive setTimeout to avoid request stacking on slow networks.
 * Retries up to 3 times on transient network errors with exponential backoff.
 */
function pollGeneration(jobId, progressFill, progressText, signal) {
    return new Promise((resolve, reject) => {
        let currentPhase = null; // determined by first poll response
        let _allocationToast = null; // ref to dismiss early on fast transition
        let _pollFailures = 0;
        const MAX_POLL_FAILURES = 3;

        // Independent wall-clock timer that updates every second
        let _phaseStart = Date.now();
        let _currentPhaseLabel = 'Allocating GPU…';
        const _timerInterval = setInterval(() => {
            const secs = Math.floor((Date.now() - _phaseStart) / 1000);
            if (progressText) {
                progressText.textContent = `${_currentPhaseLabel} (${secs}s)`;
            }
        }, 1000);
        const _totalStart = Date.now();

        // Abort handling: if the signal fires, reject immediately
        if (signal) {
            signal.addEventListener('abort', () => {
                clearInterval(_timerInterval);
                reject(new DOMException('Generation polling aborted', 'AbortError'));
            }, { once: true });
        }

        const poll = async () => { // NOSONAR
            try {
                if (signal?.aborted) return;
                const resp = await authFetch(`${API_BASE}/job/${jobId}`, { signal });
                if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
                _pollFailures = 0;
                const data = await resp.json();

                // First poll: determine initial phase
                if (currentPhase === null) {
                    currentPhase = data.status;
                    if (currentPhase !== 'allocating_gpu') {
                        // No real allocation phase (local mode) —
                        // combine into one toast to avoid overlap.
                        try {
                            const msResp = await authFetch(`${API_BASE}/model-status`, { signal });
                            if (msResp.ok) {
                                const msData = await msResp.json();
                                const device = (msData.device || 'GPU').toUpperCase();
                                _updateDeviceBadge(device, 'loaded');
                                showToast(`GPU allocated on ${device}`, 'success');
                            } else {
                                _updateDeviceBadge('GPU', 'loaded');
                                showToast('GPU allocated', 'success');
                            }
                        } catch (e) {
                            console.debug('Status parse error:', e.message);
                            _updateDeviceBadge('GPU', 'loaded');
                            showToast('GPU allocated', 'success');
                        }
                        _phaseStart = Date.now();
                        _currentPhaseLabel = 'Starting…';
                    } else {
                        // Real allocation phase (SLURM) — show a persistent
                        // toast that we'll dismiss when allocation finishes.
                        _allocationToast = showToast('Allocating GPU…', 'success');
                    }
                }

                // Detect phase transition: allocating_gpu → any active status
                // (fires when there was a real allocation phase, e.g. SLURM)
                if (currentPhase === 'allocating_gpu' && data.status !== 'allocating_gpu') {
                    currentPhase = data.status;
                    _phaseStart = Date.now();
                    _currentPhaseLabel = 'Starting…';

                    // Dismiss the allocation toast early, then show the new one
                    if (_allocationToast) { _allocationToast.dismiss(); _allocationToast = null; }

                    try {
                        const msResp = await authFetch(`${API_BASE}/model-status`, { signal });
                        if (msResp.ok) {
                            const msData = await msResp.json();
                            const device = (msData.device || 'GPU').toUpperCase();
                            _updateDeviceBadge(device, 'loaded');
                            showToast(`GPU allocated on ${device}`, 'success');
                        } else {
                            _updateDeviceBadge('GPU', 'loaded');
                            showToast('GPU allocated', 'success');
                        }
                    } catch (e) {
                        if (e.name === 'AbortError') throw e;
                        console.debug('Status parse error:', e.message);
                        _updateDeviceBadge('GPU', 'loaded');
                        showToast('GPU allocated', 'success');
                    }
                }

                // Update progress bar and phase label based on job status
                if (data.status === 'allocating_gpu') {
                    progressFill.style.width = '3%';
                    if (_currentPhaseLabel !== 'Allocating GPU…') {
                        _currentPhaseLabel = 'Allocating GPU…';
                    }
                } else if (data.status === 'starting') {
                    progressFill.style.width = '5%';
                    if (_currentPhaseLabel !== 'Starting…') {
                        _phaseStart = Date.now();
                        _currentPhaseLabel = 'Starting…';
                    }
                } else if (data.status === 'loading_model') {
                    progressFill.style.width = '8%';
                    if (_currentPhaseLabel !== 'Loading model…') {
                        _phaseStart = Date.now();
                        _currentPhaseLabel = 'Loading model…';
                    }
                } else if (data.status === 'generating') {
                    const pct = Math.max(data.progress || 0, 10);
                    progressFill.style.width = `${Math.min(pct, 95)}%`;
                    if (_currentPhaseLabel !== 'Generating…') {
                        _phaseStart = Date.now();
                        _currentPhaseLabel = 'Generating…';
                    }
                }

                if (data.status === 'completed') {
                    clearInterval(_timerInterval);
                    resolve(data);
                } else if (data.status === 'cancelled') {
                    clearInterval(_timerInterval);
                    reject(new Error('Generation cancelled'));
                } else if (data.status === 'failed') {
                    clearInterval(_timerInterval);
                    const err = new Error(data.error || 'Generation failed');
                    err.healthCheckType = data.health_check_type || null;
                    err.healthCheckAdvice = data.health_check_advice || null;
                    reject(err);
                } else if (Math.floor((Date.now() - _totalStart) / 1000) > 1800) {
                    clearInterval(_timerInterval);
                    reject(new Error('Generation timed out after 30 minutes'));
                } else {
                    const pollMs = (() => {
                        if (!data) return 2000;
                        const s = data.status;
                        if (s === 'generating') return 1000;
                        if (s === 'loading_model') return 3000;
                        if (s === 'allocating_gpu' || s === 'starting' || s === 'queued') return 5000;
                        return 2000;
                    })();
                    setTimeout(poll, pollMs);
                }
            } catch (e) {
                if (e.name === 'AbortError') return; // handled by signal listener
                _pollFailures++;
                if (_pollFailures <= MAX_POLL_FAILURES) {
                    console.debug(`Poll retry ${_pollFailures}/${MAX_POLL_FAILURES}:`, e.message);
                    setTimeout(poll, 2000 * _pollFailures);
                    return;
                }
                clearInterval(_timerInterval);
                reject(e);
            }
        };
        setTimeout(poll, 1000);
    });
}

// =========================================================================
// Results
// =========================================================================

async function renderResults(data) { // NOSONAR
    const placeholder = document.getElementById('results-placeholder');
    const list = document.getElementById('results-list');

    placeholder.classList.add('hidden');
    list.classList.remove('hidden');
    list.innerHTML = '';

    // Show bulk-actions section (select/save/clear controls)
    const bulkActions = document.getElementById('bulk-actions');
    if (bulkActions) bulkActions.classList.remove('hidden');

    // Show analysis section (affinity + viz buttons)
    const analysisSection = document.getElementById('analysis-section');
    if (analysisSection) analysisSection.classList.remove('hidden');

    // Refresh reference ligand card (un-highlight — generated ligand will be active)
    _showRefLigandCard();
    const refCard = document.getElementById('ref-ligand-card');
    if (refCard) refCard.classList.remove('active');

    // Track whether optimization was used (affects default H display)
    state.genUsedOptimization = data.used_optimization || false;
    state.genHsVisible = state.genUsedOptimization; // show Hs by default only if optimized

    // Show the hydrogen controls for generated ligands
    const hsControls = document.getElementById('gen-hs-controls');
    if (hsControls) hsControls.classList.remove('hidden');

    const currentIter = state.iterationIdx - 1; // just incremented
    const hasPrevRounds = state.allGeneratedResults.length > 1;

    // ── Current round header (collapsible) ──
    const currentSection = document.createElement('div');
    currentSection.className = 'current-round-section';
    {
        const hdr = document.createElement('div');
        hdr.className = 'round-header round-header-collapsible';
        const rc = _roundColor(currentIter);
        hdr.innerHTML = `<span class="round-dot" style="background:${rc.fill};border-color:${rc.line}"></span>` +
            `<span class="round-label">Round ${currentIter + 1}</span>` +
            `<span class="round-count">${data.results.length} ligand${data.results.length === 1 ? '' : 's'}</span>` +
            `<span class="round-chevron">&#x25B8;</span>`;
        const body = document.createElement('div');
        body.className = 'current-round-body';
        hdr.onclick = () => {
            const isCollapsed = hdr.classList.toggle('collapsed');
            body.classList.toggle('hidden', isCollapsed);
        };
        currentSection.appendChild(hdr);
        currentSection._body = body;
    }
    list.appendChild(currentSection);

    // ── Current round cards (interactive — for 3D viewer) ──
    const currentBody = currentSection._body;
    data.results.forEach((result, i) => {
        try {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.onclick = (e) => {
                if (e.target.classList.contains('result-card-checkbox')) return;
                if (e.target.closest('.set-ref-btn')) return;
                showGeneratedLigand(i);
            };

            const mw = result.properties?.mol_weight || '–';
            const tpsa = result.properties?.tpsa ?? result.properties?.TPSA ?? '–';
            const logp = result.properties?.logp ?? result.properties?.LogP ?? '–';
            const ha = result.properties?.num_heavy_atoms || '–';

            card.innerHTML = `
                <input type="checkbox" class="result-card-checkbox save-ligand-cb" data-idx="${i}"
                       onchange="event.stopPropagation(); updateSaveSelectedCount()">
                <div class="result-card-content">
                    <div class="result-header">
                        <span class="result-title">Ligand #${i + 1}</span>
                        <span class="result-badge badge badge-idle">${escapeHtml(String(ha))} HA</span>
                    </div>
                    <div class="result-smiles">${escapeHtml(smilesWithoutH(result.smiles) || 'N/A')}</div>
                    <div class="result-props">
                        <span>MW: ${escapeHtml(String(mw))}</span>
                        <span>TPSA: ${escapeHtml(String(tpsa))}</span>
                        <span>logP: ${escapeHtml(String(logp))}</span>
                    </div>
                    <button class="set-ref-btn" onclick="event.stopPropagation(); setAsReference(${i})" title="Use this ligand as the new reference">Set as Reference</button>
                </div>
            `;
            currentBody.appendChild(card);
        } catch (e) {
            console.error(`Failed to render result card ${i}:`, e);
            const errCard = document.createElement('div');
            errCard.className = 'result-card result-card-error';
            errCard.innerHTML = `<div class="result-card-content"><span class="result-title">Ligand #${i + 1}</span><span class="muted-text">Render error</span></div>`;
            currentBody.appendChild(errCard);
        }
    });
    currentSection.appendChild(currentBody);

    // ── Previous rounds (collapsed, expandable) ──
    if (hasPrevRounds) {
        // Insert previous rounds in reverse chronological order (newest first)
        for (let ri = state.allGeneratedResults.length - 2; ri >= 0; ri--) {
            const round = state.allGeneratedResults[ri];
            const iter = round.iteration;
            const rc = _roundColor(iter);
            const nLigs = round.results.length;

            const section = document.createElement('div');
            section.className = 'prev-round-section';

            const hdr = document.createElement('div');
            hdr.className = 'round-header round-header-collapsible collapsed';
            hdr.innerHTML = `<span class="round-dot" style="background:${rc.fill};border-color:${rc.line}"></span>` +
                `<span class="round-label">Round ${iter + 1}</span>` +
                `<span class="round-count">${nLigs} ligand${nLigs === 1 ? '' : 's'}</span>` +
                `<span class="round-chevron">&#x25B8;</span>`;
            hdr.onclick = () => {
                const body = section.querySelector('.prev-round-body');
                const isCollapsed = hdr.classList.toggle('collapsed');
                body.classList.toggle('hidden', isCollapsed);
            };
            section.appendChild(hdr);

            const body = document.createElement('div');
            body.className = 'prev-round-body hidden';
            round.results.forEach((result, j) => {
                try {
                    const card = document.createElement('div');
                    card.className = 'result-card result-card-prev';
                    const mw = result.properties?.mol_weight || '–';
                    const tpsa = result.properties?.tpsa ?? result.properties?.TPSA ?? '–';
                    const logp = result.properties?.logp ?? result.properties?.LogP ?? '–';
                    const ha = result.properties?.num_heavy_atoms || '–';
                    card.innerHTML = `
                        <div class="result-card-content">
                            <div class="result-header">
                                <span class="result-title">Ligand #${j + 1}</span>
                                <span class="result-badge badge badge-idle">${escapeHtml(String(ha))} HA</span>
                            </div>
                            <div class="result-smiles">${escapeHtml(smilesWithoutH(result.smiles) || 'N/A')}</div>
                            <div class="result-props">
                                <span>MW: ${escapeHtml(String(mw))}</span>
                                <span>TPSA: ${escapeHtml(String(tpsa))}</span>
                                <span>logP: ${escapeHtml(String(logp))}</span>
                            </div>
                        </div>
                    `;
                    body.appendChild(card);
                } catch (e) {
                    console.error(`Failed to render prev-round card ${j}:`, e);
                    const errCard = document.createElement('div');
                    errCard.className = 'result-card result-card-prev result-card-error';
                    errCard.innerHTML = `<div class="result-card-content"><span class="result-title">Ligand #${j + 1}</span><span class="muted-text">Render error</span></div>`;
                    body.appendChild(errCard);
                }
            });
            section.appendChild(body);
            list.appendChild(section);
        }
    }

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

    // Update save section + sync Select All toggle with fresh checkboxes
    updateSaveSection();
    updateSaveSelectedCount();

    // ── Show first generated ligand (this also dims the reference) ──
    if (data.results.length > 0) showGeneratedLigand(0);
}

function showGeneratedLigand(idx) { // NOSONAR
    if (idx < 0 || idx >= state.generatedResults.length) return;

    state.activeResultIdx = idx;
    state.activeView = 'gen';

    // Un-highlight reference card, highlight only the clicked generated card
    const refCard = document.getElementById('ref-ligand-card');
    if (refCard) refCard.classList.remove('active');
    const resultsList = document.getElementById('results-list');
    if (resultsList) {
        resultsList.querySelectorAll('.result-card:not(.result-card-prev)').forEach((card, i) => {
            card.classList.toggle('active', i === idx);
        });
    }

    const result = state.generatedResults[idx];

    try {
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
            // Invalidate binding-site cache when fallback ligand changes
            if (!state.ligandModel) state._cachedBindingSiteSerials = null;
            applyProteinStyle();
            reapplyHoverable();
        }

        // Clean up any overlay models from reference view
        _clearOverlayModels();

        state.viewer.render();

        if (state.generatedModel && !state._restoringSession) {
            state.viewer.zoomTo({ model: state.generatedModel }, 300);
            state.viewer.render();
        }
    } catch (e) {
        console.error('Error displaying generated ligand:', e);
        showToast('Error displaying ligand: ' + e.message, 'error');
    }

    // Show 2D structure overlay (hide Interactions tab for LBDD)
    showMol2D(result.smiles, `Ligand #${idx + 1}`, result.properties);
    document.getElementById('tab-interactions')?.classList.toggle('hidden', state.workflowType === 'lbdd');

    // Reset interaction diagram cache so it reloads for new ligand
    document.getElementById('mol-2d-interaction')?.setAttribute('data-loaded', '');

    // Reset properties cache so it reloads for new ligand
    document.getElementById('mol-2d-properties')?.setAttribute('data-loaded', '');

    // Clear 3D interactions if showing (will re-show for new ligand if toggled)
    // Only for SBDD — LBDD has no protein so no protein-ligand interactions.
    if (state.showingInteractions) {
        clearInteractions3D();
        if (state.workflowType === 'lbdd') {
            state.showingInteractions = false;
        } else {
            showInteractions3D('both');
        }
    }

    // Lock sidebar while a generated ligand is active
    _syncSidebarLock();

}

// =========================================================================
// Reference ↔ Generated Ligand Switching
// =========================================================================

/**
 * Populate and show the reference ligand card in the right panel.
 * Called after ligand upload and after generation results.
 */
function _showRefLigandCard() { // NOSONAR
    const section = document.getElementById('ref-ligand-card-section');
    if (!section || !state.ligandData) return;
    if (state.pocketOnlyMode || state.lbddScratchMode) return;

    section.classList.remove('hidden');

    const titleEl = document.getElementById('ref-card-title');
    const badgeEl = document.getElementById('ref-card-badge');
    const smilesEl = document.getElementById('ref-card-smiles');
    const propsEl = document.getElementById('ref-card-props');
    const canvas = document.getElementById('ref-card-2d');

    // Use uploaded filename (without extension) as card title
    const rawName = document.getElementById('ligand-filename')?.textContent || '';
    const baseName = rawName.replace(/\.[^.]+$/, '') || 'Reference';
    if (titleEl) titleEl.textContent = baseName;
    const ha = document.getElementById('ligand-heavy-atoms')?.textContent || '';
    if (badgeEl) badgeEl.textContent = ha ? `${ha} HA` : '';

    // SMILES — always without explicit hydrogens
    const smiles = smilesWithoutH(document.getElementById('ligand-smiles')?.textContent) || '';
    if (smilesEl) smilesEl.textContent = smiles || '';

    // 2D structure preview (small inline SVG via RDKit.js)
    if (canvas && smiles && RDKitModule) {
        let mol = null;
        try {
            mol = RDKitModule.get_mol(smiles);
            if (mol) {
                canvas.innerHTML = mol.get_svg_with_highlights(JSON.stringify({ width: 320, height: 180 }));
            }
        } catch (e) { console.debug('Render failed:', e.message); }
        finally { if (mol) try { mol.delete(); } catch (e) { console.debug('Mol cleanup:', e.message); } }
    } else if (canvas) {
        canvas.innerHTML = '';
    }

    // Properties
    if (propsEl && state.refProperties) {
        const mw = state.refProperties.MolWt ?? state.refProperties.mol_weight ?? '–';
        const tpsa = state.refProperties.TPSA ?? state.refProperties.tpsa ?? '–';
        const logp = state.refProperties.LogP ?? state.refProperties.logp ?? '–';
        const fmtN = v => typeof v === 'number' ? v.toFixed(1) : v;
        propsEl.innerHTML = `<span>MW: ${escapeHtml(String(fmtN(mw)))}</span><span>TPSA: ${escapeHtml(String(fmtN(tpsa)))}</span><span>logP: ${escapeHtml(String(fmtN(logp)))}</span>`;
    } else if (propsEl) {
        propsEl.innerHTML = '';
    }
}

/**
 * Show or update the "Original Ligand" card in the right panel.
 * This appears above the Reference Ligand card once the user has swapped
 * the reference via "Set as Reference" on a generated ligand.
 */
function _showOriginalLigandCard() { // NOSONAR
    const section = document.getElementById('original-ligand-card-section');
    if (!section) return;

    const orig = state.originalLigand;
    if (!orig) {
        section.classList.add('hidden');
        return;
    }
    section.classList.remove('hidden');

    const titleEl = document.getElementById('orig-card-title');
    const badgeEl = document.getElementById('orig-card-badge');
    const smilesEl = document.getElementById('orig-card-smiles');
    const propsEl = document.getElementById('orig-card-props');
    const canvas = document.getElementById('orig-card-2d');

    const baseName = (orig.filename || '').replace(/\.[^.]+$/, '') || 'Original';
    if (titleEl) titleEl.textContent = baseName;
    if (badgeEl) badgeEl.textContent = orig.num_heavy_atoms ? `${orig.num_heavy_atoms} HA` : '';

    const smiles = smilesWithoutH(orig.smiles_noH || orig.smiles) || '';
    if (smilesEl) smilesEl.textContent = smiles;

    // 2D preview
    if (canvas && smiles && RDKitModule) {
        let mol = null;
        try {
            mol = RDKitModule.get_mol(smiles);
            if (mol) canvas.innerHTML = mol.get_svg_with_highlights(JSON.stringify({ width: 320, height: 180 }));
        } catch (e) { console.debug('Render failed:', e.message); }
        finally { if (mol) try { mol.delete(); } catch (e) { console.debug('Mol cleanup:', e.message); } }
    } else if (canvas) {
        canvas.innerHTML = '';
    }

    // Properties
    if (propsEl && orig.properties) {
        const mw = orig.properties.MolWt ?? orig.properties.mol_weight ?? '–';
        const tpsa = orig.properties.TPSA ?? orig.properties.tpsa ?? '–';
        const logp = orig.properties.LogP ?? orig.properties.logp ?? '–';
        const fmtN = v => typeof v === 'number' ? v.toFixed(1) : v;
        propsEl.innerHTML = `<span>MW: ${escapeHtml(String(fmtN(mw)))}</span><span>TPSA: ${escapeHtml(String(fmtN(tpsa)))}</span><span>logP: ${escapeHtml(String(fmtN(logp)))}</span>`;
    } else if (propsEl) {
        propsEl.innerHTML = '';
    }
}

/** Open 2D overlay for the original ligand. */
function showOriginalLigand2D() {
    const orig = state.originalLigand;
    if (!orig) return;
    showMol2D(orig.smiles_noH || orig.smiles, 'Original Ligand', orig.properties);
}

/**
 * Restore the original ligand as the active reference.
 * Sends the original SDF to the /set-reference endpoint.
 */
async function setOriginalAsReference() {
    const orig = state.originalLigand;
    if (!orig?.sdf_data || !state.jobId) return;

    const allBtns = document.querySelectorAll('.set-ref-btn');
    allBtns.forEach(b => { b.disabled = true; });

    try {
        const resp = await authFetch(`${API_BASE}/set-reference/${state.jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdf_data: orig.sdf_data }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Update state exactly as setAsReference does
        state.ligandData = data;
        state.ligandData._atomByIdx = new Map();
        data.atoms.forEach(a => state.ligandData._atomByIdx.set(a.idx, a));

        state.heavyAtomMap = new Map();
        let heavyIdx = 0;
        data.atoms.forEach(a => {
            if (a.atomicNum !== 1) {
                state.heavyAtomMap.set(a.idx, heavyIdx);
                heavyIdx++;
            }
        });

        state.numHeavyAtoms = data.num_heavy_atoms;
        state.refHasExplicitHs = data.has_explicit_hs || false;
        state.refProperties = data.ref_properties || null;

        // Update ligand info card
        document.getElementById('ligand-smiles').textContent = data.smiles_noH || data.smiles || '';
        document.getElementById('ligand-heavy-atoms').textContent = data.num_heavy_atoms;
        document.getElementById('ligand-total-atoms').textContent = data.num_atoms;
        document.getElementById('ligand-filename').textContent = orig.filename || 'Original';

        // Reference affinity
        const refAffEl = document.getElementById('ligand-ref-affinity');
        if (refAffEl && data.ref_affinity) {
            const ra = data.ref_affinity;
            refAffEl.textContent = `${ra.p_label}: ${ra.p_value.toFixed(2)}`;
            refAffEl.closest('.info-row')?.classList.remove('hidden');
        } else if (refAffEl) {
            refAffEl.closest('.info-row')?.classList.add('hidden');
        }

        _showNumHeavyAtomsField(data.num_heavy_atoms, 'from reference — editable');
        if (data.ref_properties) populatePropertyFilterPanel(data.ref_properties);

        // Remove existing 3D models
        if (state.generatedModel !== null) {
            state.viewer.removeModel(state.generatedModel);
            state.generatedModel = null;
        }
        _clearOverlayModels();
        if (state.ligandModel !== null) {
            state.viewer.removeModel(state.ligandModel);
            state.ligandModel = null;
        }

        renderLigand(data.sdf_data);
        state.fixedAtoms.clear();
        updateSelectionUI();
        clearAutoHighlight();

        // Switch to reference view
        state.activeView = 'ref';
        state.activeResultIdx = -1;
        _showRefLigandCard();
        const refCard = document.getElementById('ref-ligand-card');
        if (refCard) refCard.classList.add('active');
        document.querySelectorAll('.result-card:not(.ref-ligand-card)').forEach(c => c.classList.remove('active'));

        if (state.genMode === 'substructure_inpainting') {
            _updateSubstructurePriorCloud();
        } else {
            await _fetchAndRenderPriorCloudPreview();
        }

        const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
        if (autoModes.includes(state.genMode)) fetchAndHighlightInpaintingMask(state.genMode);

        const refSmiles = document.getElementById('ligand-smiles')?.textContent || '';
        if (refSmiles) showMol2D(refSmiles, '', state.refProperties);

        _syncSidebarLock();
        showToast('Original ligand restored as reference', 'success');
    } catch (e) {
        console.error('Failed to restore original as reference:', e);
        showToast('Failed to restore original ligand', 'error');
    } finally {
        allBtns.forEach(b => { b.disabled = false; });
    }
}

/**
 * Switch the 3D viewer to show the reference ligand as full-atom.
 * Selected (checked) generated ligands are shown as transparent overlays.
 */
function showReferenceLigandView() {
    if (!state.ligandModel) return;

    state.activeView = 'ref';

    // Highlight the reference card, un-highlight all generated cards
    const refCard = document.getElementById('ref-ligand-card');
    if (refCard) refCard.classList.add('active');
    document.querySelectorAll('.result-card:not(.ref-ligand-card)').forEach(c => c.classList.remove('active'));

    // Remove the currently displayed generated model
    if (state.generatedModel !== null) {
        state.viewer.removeModel(state.generatedModel);
        state.generatedModel = null;
    }
    state.activeResultIdx = -1;

    // Show reference ligand full-atom
    state.refLigandVisible = true;
    applyLigandStyleBase();
    reapplyClickable();

    // Show selected generated ligands as transparent overlays
    _updateOverlayModels();

    // Protein
    if (state.proteinModel && state.viewMode !== 'ligand') {
        applyProteinStyle();
        reapplyHoverable();
    }

    state.viewer.zoomTo({ model: state.ligandModel }, 300);
    state.viewer.render();

    // Update 2D overlay: show reference SMILES
    const refSmiles = document.getElementById('ligand-smiles')?.textContent || '';
    if (refSmiles) {
        showMol2D(refSmiles, '', state.refProperties);
    } else {
        closeMol2DOverlay();
    }

    // Clear/re-show interactions in reference-active mode
    if (state.showingInteractions) {
        clearInteractions3D();
        showInteractions3D('both');
    }

    // Update the eye-toggle button text
    const btn = document.getElementById('btn-hide-ligand');
    if (btn) {
        btn.textContent = '\u25CB Hide Ligand';
        btn.classList.remove('active');
    }

    // Unlock sidebar when reference is active
    _syncSidebarLock();
}

/**
 * Add/remove transparent overlay models for selected (checked) generated ligands.
 * Only shown when activeView === 'ref'.
 */
function _updateOverlayModels() {
    // Always clean up existing overlays first
    _clearOverlayModels();

    // Only show overlays when reference is the active full-atom view
    if (state.activeView !== 'ref') {
        state.viewer.render();
        return;
    }

    // Gather selected indices from checkboxes
    const selectedIndices = getSelectedSaveIndices();

    selectedIndices.forEach(idx => {
        const result = state.generatedResults[idx];
        if (!result) return;
        const sdf = _getGenSdf(result);
        if (!sdf) return;
        const model = state.viewer.addModel(sdf, 'sdf');
        state.viewer.setStyle(
            { model },
            {
                stick: { radius: 0.1, colorscheme: 'Jmol', opacity: 0.3 },
                sphere: { radius: 0.15, colorscheme: 'Jmol', opacity: 0.3 },
            }
        );
        state.selectedOverlayModels.push(model);
    });

    state.viewer.render();
}

// =========================================================================
// Set Generated Ligand as New Reference
// =========================================================================

/**
 * Replace the current reference ligand with a generated ligand.
 * Sends its SDF to the server, receives full ligand metadata,
 * then updates all client state exactly as a fresh upload would.
 * Previous rounds and finetuning history are preserved.
 */
async function setAsReference(idx) { // NOSONAR
    const result = state.generatedResults[idx];
    if (!result || !state.jobId) return;

    const sdf = result.sdf_no_hs || result.sdf;
    if (!sdf) {
        showToast('No SDF data for this ligand', 'error');
        return;
    }

    // Disable button while loading
    const allBtns = document.querySelectorAll('.set-ref-btn');
    allBtns.forEach(b => { b.disabled = true; });

    try {
        const resp = await authFetch(`${API_BASE}/set-reference/${state.jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdf_data: sdf }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Persist original ligand on first reference swap
        if (!state.originalLigand && data.original_ligand) {
            state.originalLigand = data.original_ligand;
        }

        // ── Update state exactly as uploadLigand does ──
        state.ligandData = data;
        state.ligandData._atomByIdx = new Map();
        data.atoms.forEach(a => state.ligandData._atomByIdx.set(a.idx, a));

        state.heavyAtomMap = new Map();
        let heavyIdx = 0;
        data.atoms.forEach(a => {
            if (a.atomicNum !== 1) {
                state.heavyAtomMap.set(a.idx, heavyIdx);
                heavyIdx++;
            }
        });

        state.numHeavyAtoms = data.num_heavy_atoms;
        state.refHasExplicitHs = data.has_explicit_hs || false;
        state.refProperties = data.ref_properties || null;

        // Update ligand info card
        document.getElementById('ligand-smiles').textContent = data.smiles_noH || data.smiles || '';
        document.getElementById('ligand-heavy-atoms').textContent = data.num_heavy_atoms;
        document.getElementById('ligand-total-atoms').textContent = data.num_atoms;
        document.getElementById('ligand-filename').textContent = `Ligand #${idx + 1}`;

        // Reference affinity
        const refAffEl = document.getElementById('ligand-ref-affinity');
        if (refAffEl && data.ref_affinity) {
            const ra = data.ref_affinity;
            refAffEl.textContent = `${ra.p_label}: ${ra.p_value.toFixed(2)}`;
            refAffEl.closest('.info-row')?.classList.remove('hidden');
        } else if (refAffEl) {
            refAffEl.closest('.info-row')?.classList.add('hidden');
        }

        // De novo heavy atoms field
        _showNumHeavyAtomsField(data.num_heavy_atoms, 'from reference — editable');

        // Property filter panel
        if (data.ref_properties) {
            populatePropertyFilterPanel(data.ref_properties);
        }

        // Remove existing 3D ligand, generated model, and overlays
        if (state.generatedModel !== null) {
            state.viewer.removeModel(state.generatedModel);
            state.generatedModel = null;
        }
        _clearOverlayModels();
        if (state.ligandModel !== null) {
            state.viewer.removeModel(state.ligandModel);
            state.ligandModel = null;
        }

        // Render new reference in 3D
        renderLigand(data.sdf_data);

        // Reset atom selections (indices changed)
        state.fixedAtoms.clear();
        updateSelectionUI();
        clearAutoHighlight();

        // Switch to reference view
        state.activeView = 'ref';
        state.activeResultIdx = -1;
        _showOriginalLigandCard();
        _showRefLigandCard();
        const refCard = document.getElementById('ref-ligand-card');
        if (refCard) refCard.classList.add('active');
        document.querySelectorAll('.result-card:not(.ref-ligand-card):not(.original-ligand-card)').forEach(c => c.classList.remove('active'));

        // Refresh prior cloud for current gen mode
        if (state.genMode === 'substructure_inpainting') {
            _updateSubstructurePriorCloud();
        } else {
            await _fetchAndRenderPriorCloudPreview();
        }

        // Re-trigger auto-highlight for inpainting modes
        const autoModes = ['scaffold_hopping', 'scaffold_elaboration', 'linker_inpainting', 'core_growing', 'fragment_growing'];
        if (autoModes.includes(state.genMode)) {
            fetchAndHighlightInpaintingMask(state.genMode);
        }

        // Update 2D overlay
        const refSmiles = document.getElementById('ligand-smiles')?.textContent || '';
        if (refSmiles) {
            showMol2D(refSmiles, '', state.refProperties);
        }

        // Unlock sidebar
        _syncSidebarLock();

        showToast(`Ligand #${idx + 1} set as reference`, 'success');
    } catch (e) {
        console.error('Failed to set reference:', e);
        let errMsg = 'Failed to set reference';
        try {
            const p = JSON.parse(e.message);
            if (p.detail) errMsg = p.detail;
        } catch (parseErr) {
            console.debug('Error parse failed:', parseErr.message);
            if (e.message && e.message.length < 300) errMsg = e.message;
        }
        showToast(errMsg, 'error');
    } finally {
        allBtns.forEach(b => { b.disabled = false; });
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

/** Return the appropriate status badge text based on current input state. */
function _getInputBadgeText() {
    if (state.proteinData && state.ligandData) return 'Complex loaded';
    if (state.proteinData) return 'Protein loaded';
    if (state.ligandData) return 'Ligand loaded';
    return null;
}

/**
 * Show a toast notification at the bottom of the screen.
 * Multiple simultaneous toasts stack upward automatically.
 * Returns the toast element (has a .dismiss() helper to remove early).
 */
function showToast(message, type = 'success') {
    // Cap active toasts at 5 — remove the oldest if at limit
    const existingToasts = document.querySelectorAll('.toast');
    if (existingToasts.length >= 5) {
        const oldest = existingToasts[0];
        if (oldest._timer) clearTimeout(oldest._timer);
        oldest.remove();
    }

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
            popup.textContent = el.dataset.tooltip;
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
            defaultMin = Number.parseFloat(defaultMin.toFixed(3));
            defaultMax = Number.parseFloat(defaultMax.toFixed(3));
        }

        const row = document.createElement('div');
        row.className = 'prop-filter-row';
        row.dataset.propName = def.name;
        row.innerHTML = `
            <input type="checkbox" class="prop-cb" title="Enable ${def.label} filter">
            <span class="prop-name" title="${def.name} (ref: ${refVal == null ? 'N/A' : refVal})">${def.label}</span>
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
    if (!cb?.checked) return null;

    const rows = document.querySelectorAll('#property-filter-list .prop-filter-row');
    const filters = [];
    for (const row of rows) {
        const propCb = row.querySelector('.prop-cb');
        if (!propCb?.checked) continue;
        const name = row.dataset.propName;
        const minVal = row.querySelector('.prop-min')?.value;
        const maxVal = row.querySelector('.prop-max')?.value;
        filters.push({
            name,
            min: minVal === '' ? null : Number.parseFloat(minVal),
            max: maxVal === '' ? null : Number.parseFloat(maxVal),
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
                <span>Upload Model</span>
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
            const resp = await authFetch(`${API_BASE}/upload/adme-model/${state.jobId}`, {
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
    if (!cb?.checked) return null;

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
            min: minVal === '' ? null : Number.parseFloat(minVal),
            max: maxVal === '' ? null : Number.parseFloat(maxVal),
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

    // If file was cleared — remove prior cloud & re-fetch without prior center
    if (input.files.length === 0) {
        state._uploadedPriorCenterFilename = null;
        _fetchAndRenderPriorCloudPreview();
        return;
    }

    // Upload the file immediately so the server can compute a prior-cloud preview
    if (!state.jobId) return;
    const formData = new FormData();
    formData.append('file', input.files[0]);
    try {
        const resp = await authFetch(`${API_BASE}/upload_prior_center/${state.jobId}`, {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) return;
        const data = await resp.json();
        // Store the uploaded filename for later use in generation payload
        state._uploadedPriorCenterFilename = data.filename;
        // Clear any previously placed center (file takes precedence)
        state._priorPlacedCenter = null;
        _updatePriorCoordsDisplay(null);
        document.getElementById('reset-prior-pos-btn')?.classList.add('hidden');
        if (data.prior_cloud) {
            renderPriorCloud(data.prior_cloud);
        }
    } catch (e) {
        console.warn('Prior center upload/preview failed:', e);
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
    const growSize = Number.parseInt(document.getElementById('grow-size')?.value) || 5;
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
        const resp = await authFetch(`${API_BASE}/ring-systems/${state.jobId}`);
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
    state._ringSystemIndex = Number.parseInt(value) || 0;
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
let _substructCloudTimeout = null;
function _debouncedSubstructurePriorCloud(delay = 200) {
    if (_substructCloudTimeout) clearTimeout(_substructCloudTimeout);
    _substructCloudTimeout = setTimeout(() => {
        _substructCloudTimeout = null;
        _updateSubstructurePriorCloud();
    }, delay);
}

let _cloudPreviewTimeout = null;
function _debouncedCloudPreview(delay = 300) {
    if (_cloudPreviewTimeout) clearTimeout(_cloudPreviewTimeout);
    _cloudPreviewTimeout = setTimeout(() => {
        _cloudPreviewTimeout = null;
        _fetchAndRenderPriorCloudPreview();
    }, delay);
}

async function _fetchAndRenderPriorCloudPreview() { // NOSONAR
    // Allow preview when we have a ligand OR are in pocket-only / scratch mode
    if (!state.jobId) return;
    if (!state.ligandData && !state.pocketOnlyMode && !state.lbddScratchMode) return;
    let cloudSize;
    if (state.genMode === 'fragment_growing') {
        cloudSize = Number.parseInt(document.getElementById('grow-size')?.value) || 5;
    } else if (_ATOM_SELECT_MODES.has(state.genMode) && state.fixedAtoms.size > 0) {
        // For ALL atom-select modes (substructure, scaffold, linker, core):
        // cloud size = number of user-selected atoms to be replaced
        cloudSize = state.fixedAtoms.size;
    } else {
        // For de novo / other modes use state.numHeavyAtoms (may come from
        // reference ligand or manual entry) or fall back to ligandData.
        cloudSize = state.numHeavyAtoms
            || state.ligandData?.num_heavy_atoms
            || Number.parseInt(document.getElementById('grow-size')?.value) || 20;
    }
    const pocketCutoff = Number.parseFloat(document.getElementById('pocket-cutoff')?.value) || 6;
    const genMode = state.genMode || 'fragment_growing';
    const ringIdx = state._ringSystemIndex || 0;
    const placedCenter = state._priorPlacedCenter;
    try {
        // Build query params
        const params = new URLSearchParams({
            grow_size: cloudSize,
            pocket_cutoff: pocketCutoff,
            anisotropic: state.anisotropicPrior,
            gen_mode: genMode,
            ring_system_index: ringIdx,
            ref_ligand_com_prior: state.refLigandComPrior,
        });

        // Include fixed atoms for modes that use atom selection
        if (_ATOM_SELECT_MODES.has(genMode) && state.fixedAtoms.size > 0) {
            const fixedArr = [...state.fixedAtoms];
            // Convert to heavy-atom indices for the server
            const heavyIndices = fixedArr
                .map(i => state.heavyAtomMap?.get(i))
                .filter(i => i !== undefined);
            if (heavyIndices.length > 0) {
                params.set('fixed_atoms', heavyIndices.join(','));
            }
        }

        const resp = await authFetch(`${API_BASE}/prior-cloud-preview/${state.jobId}?${params.toString()}`);
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
                const newW = Math.max(200, Math.min(800, startWidth + delta));
                panel.style.width = newW + 'px';
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
                const newW = Math.max(200, Math.min(700, startWidth + delta));
                panelLeft.style.width = newW + 'px';
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

/**
 * Return canonical SMILES with all explicit hydrogens removed.
 * Uses RDKit.js when available; falls back to the input string.
 */
function smilesWithoutH(smiles) {
    if (!smiles || !RDKitModule) return smiles || '';
    let mol = null;
    try {
        mol = RDKitModule.get_mol(smiles);
        if (!mol) return smiles;
        mol.remove_all_hs();
        return mol.get_smiles() || smiles;
    } catch (e) {
        console.debug('Canonicalization failed:', e.message);
        return smiles;
    } finally {
        if (mol) try { mol.delete(); } catch (error_) { console.debug('Mol cleanup:', error_.message); }
    }
}

// =========================================================================
// 2D molecule structure overlay (using RDKit.js client-side rendering)
// =========================================================================

function showMol2D(smiles, name, properties) { // NOSONAR
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
    if (nameEl) nameEl.textContent = name ?? '2D Structure';

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
            console.debug('RDKit render failed, falling back:', e.message);
            _fallbackServerRender(smiles, canvas);
        } finally {
            if (mol) try { mol.delete(); } catch (error_) { console.debug('Mol cleanup:', error_.message); }
        }
    } else {
        _fallbackServerRender(smiles, canvas);
    }

    // SMILES — always without explicit Hs
    if (smilesEl) smilesEl.textContent = smilesWithoutH(smiles);

    // Properties (support both worker aliases and canonical RDKit names)
    if (propsEl && properties) {
        const mw = properties.mol_weight ?? properties.MolWt ?? '–';
        const tpsa = properties.tpsa ?? properties.TPSA ?? '–';
        const logp = properties.logp ?? properties.LogP ?? '–';
        propsEl.innerHTML = `<span>MW ${escapeHtml(String(mw))}</span><span>TPSA ${escapeHtml(String(tpsa))}</span><span>logP ${escapeHtml(String(logp))}</span>`;
    } else if (propsEl) {
        propsEl.innerHTML = '';
    }
}

function _fallbackServerRender(smiles, container) {
    container.innerHTML = '<span class="muted-text">Loading…</span>';
    authFetch(`${API_BASE}/mol-image?smiles=${encodeURIComponent(smiles)}&width=380&height=260`)
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

    authFetch(`${API_BASE}/interaction-diagram/${state.jobId}?ligand_idx=${ligIdx}`)
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
// Save / Download Ligands
// =========================================================================

/**
 * Trigger a browser file download from in-memory string content.
 * Creates a temporary Blob URL, clicks an invisible anchor, and cleans up.
 */
function _downloadTextFile(content, filename, mimeType = 'chemical/x-mdl-sdfile') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

// =========================================================================
// Session Save / Restore
// =========================================================================

function _deepCloneResults(allResults) {
    return allResults.map(round => ({
        iteration: round.iteration,
        results: round.results.map(r => ({ ...r })),
    }));
}

function saveSession() {
    const session = {
        version: 1,
        savedAt: new Date().toISOString(),
        savedBy: state._currentUser || 'anonymous',
        workflow: {
            type: state.workflowType,
            baseCkptPath: _initialCkptPath,
            ckptPath: (state._alFinetuned && !state._alCheckpointSaved)
                ? _initialCkptPath
                : _selectedCkptPath,
        },
        protein: state.proteinData ? {
            filename: state.proteinData.filename,
            format: state.proteinData.format,
            pdbData: state.proteinData.pdb_data,
        } : null,
        ligand: state.ligandData ? {
            filename: state.ligandData.filename,
            sdfData: state.ligandData.sdf_data,
            smiles: state.ligandData.smiles,
            smilesNoH: state.ligandData.smiles_noH,
            hasExplicitHs: state.ligandData.has_explicit_hs,
            numAtoms: state.ligandData.num_atoms,
            numHeavyAtoms: state.ligandData.num_heavy_atoms,
            atoms: state.ligandData.atoms,
            bonds: state.ligandData.bonds,
            refProperties: state.ligandData.ref_properties,
        } : null,
        generation: {
            genMode: state.genMode,
            genUsedOptimization: state.genUsedOptimization,
            refHasExplicitHs: state.refHasExplicitHs,
            iterationIdx: state.iterationIdx,
            activeResultIdx: state.activeResultIdx,
            allGeneratedResults: _deepCloneResults(state.allGeneratedResults),
            generatedResults: state.generatedResults,
        },
        settings: {
            fixedAtoms: Array.from(state.fixedAtoms),
            numHeavyAtoms: state.numHeavyAtoms,
            pocketOnlyMode: state.pocketOnlyMode,
            lbddScratchMode: state.lbddScratchMode,
            anisotropicPrior: state.anisotropicPrior,
            refLigandComPrior: state.refLigandComPrior,
            seed: (() => { const v = Number.parseInt(document.getElementById('gen-seed')?.value, 10); return Number.isNaN(v) ? 42 : v; })(),
        },
        viewState: {
            viewMode: state.viewMode,
            activeView: state.activeView,
            surfaceOn: state.surfaceOn,
            refLigandVisible: state.refLigandVisible,
            genHsVisible: state.genHsVisible,
            proteinFullAtom: state.proteinFullAtom,
            bindingSiteVisible: state.bindingSiteVisible,
            priorCloudVisible: state.priorCloudVisible,
        },
        generationParams: {
            nSamples: Number(document.getElementById('n-samples')?.value) || 100,
            batchSize: Number(document.getElementById('batch-size')?.value) || 25,
            integrationSteps: Number(document.getElementById('integration-steps')?.value) || 100,
            pocketCutoff: Number(document.getElementById('pocket-cutoff')?.value) || 6,
            addNoise: document.getElementById('add-noise')?.checked ?? true,
            noiseScale: Number(document.getElementById('noise-scale')?.value) || 0.1,
            sampleMolSizes: document.getElementById('sample-mol-sizes')?.checked ?? false,
            filterValidUnique: document.getElementById('filter-valid-unique')?.checked ?? true,
            filterCondSubstructure: document.getElementById('filter-cond-substructure')?.checked ?? false,
            filterDiversity: document.getElementById('filter-diversity')?.checked ?? true,
            diversityThreshold: Number(document.getElementById('diversity-threshold')?.value) || 0.8,
            optimizeGenLigs: document.getElementById('optimize-gen-ligs')?.checked ?? false,
            optimizeGenLigsHs: document.getElementById('optimize-gen-ligs-hs')?.checked ?? false,
            calculatePbValid: document.getElementById('calculate-pb-valid')?.checked ?? false,
            filterPbValid: document.getElementById('filter-pb-valid')?.checked ?? false,
            calculateStrain: document.getElementById('calculate-strain')?.checked ?? false,
            growSize: Number(document.getElementById('grow-size')?.value) || 5,
            lbddOptimizeMethod: document.getElementById('lbdd-optimize-method')?.value || 'uff',
        },
        originalLigand: state.originalLigand,
        priorCloud: state.priorCloud,
        priorPlacedCenter: state._priorPlacedCenter,
        activeLearning: {
            finetuned: state._alFinetuned && state._alCheckpointSaved,
            ckptPath: (state._alFinetuned && state._alCheckpointSaved) ? state._alCkptPath : null,
            round: state._alRound || 0,
        },
    };

    const json = JSON.stringify(session);
    const ts = new Date().toISOString().replaceAll(/[:.]/g, '-').slice(0, 19);
    _downloadTextFile(json, `flowr-session-${ts}.json`, 'application/json');
    showToast('Session saved', 'success');
}

function _restoreProteinUI(protein, jobId) {
    state.proteinData = {
        job_id: jobId,
        filename: protein.filename,
        format: protein.format,
        pdb_data: protein.pdbData,
    };
    document.getElementById('protein-placeholder')?.classList.add('hidden');
    document.getElementById('protein-success')?.classList.remove('hidden');
    const nameEl = document.getElementById('protein-filename');
    if (nameEl) nameEl.textContent = protein.filename;
    document.getElementById('viewer-overlay')?.classList.add('hidden');
    renderProtein(protein.pdbData, protein.format);
}

function _restoreLigandUI(ligand, jobId) {
    state.ligandData = {
        job_id: jobId,
        filename: ligand.filename,
        smiles: ligand.smiles,
        smiles_noH: ligand.smilesNoH,
        has_explicit_hs: ligand.hasExplicitHs,
        sdf_data: ligand.sdfData,
        atoms: ligand.atoms,
        bonds: ligand.bonds,
        num_atoms: ligand.numAtoms,
        num_heavy_atoms: ligand.numHeavyAtoms,
        ref_properties: ligand.refProperties,
    };

    // Rebuild _atomByIdx map
    state.ligandData._atomByIdx = new Map();
    state.ligandData.atoms.forEach(a => state.ligandData._atomByIdx.set(a.idx, a));

    // Rebuild heavyAtomMap
    state.heavyAtomMap = new Map();
    let heavyIdx = 0;
    state.ligandData.atoms.forEach(a => {
        if (a.atomicNum !== 1) {
            state.heavyAtomMap.set(a.idx, heavyIdx);
            heavyIdx++;
        }
    });

    state.refProperties = ligand.refProperties || null;

    document.getElementById('ligand-placeholder')?.classList.add('hidden');
    document.getElementById('ligand-success')?.classList.remove('hidden');
    const lnameEl = document.getElementById('ligand-filename');
    if (lnameEl) lnameEl.textContent = ligand.filename;
    const infoCard = document.getElementById('ligand-info');
    if (infoCard) infoCard.classList.remove('hidden');
    const smilesEl = document.getElementById('ligand-smiles');
    if (smilesEl) smilesEl.textContent = ligand.smilesNoH || ligand.smiles || 'N/A';
    const haEl = document.getElementById('ligand-heavy-atoms');
    if (haEl) haEl.textContent = ligand.numHeavyAtoms;
    const taEl = document.getElementById('ligand-total-atoms');
    if (taEl) taEl.textContent = ligand.numAtoms;

    document.getElementById('viewer-overlay')?.classList.add('hidden');
    renderLigand(ligand.sdfData);
}

function _restoreViewState(vs) {
    if (vs.surfaceOn && state.proteinModel) toggleSurface();
    if (vs.proteinFullAtom && state.proteinModel) toggleFullAtom();
    if (vs.bindingSiteVisible && state.proteinModel && _getActiveLigandModel()) toggleBindingSite();
    if (vs.refLigandVisible === false && state.ligandModel) {
        state.refLigandVisible = false;
        state.viewer.setStyle({ model: state.ligandModel }, { stick: { hidden: true }, sphere: { hidden: true } });
        state.viewer.render();
    }
    if (vs.genHsVisible !== state.genHsVisible) {
        state.genHsVisible = vs.genHsVisible;
    }
    state.viewMode = vs.viewMode || 'complex';
    state.activeView = vs.activeView || 'ref';

    // ── Prior cloud visibility ──
    if (vs.priorCloudVisible === false && state.priorCloudSpheres?.length > 0) {
        state.priorCloudVisible = false;
        state.priorCloudSpheres.forEach(s => s.opacity = 0);
        const btn = document.getElementById('toggle-prior-cloud');
        if (btn) btn.classList.remove('active');
        state.viewer.render();
    }
}

async function restoreSession(session, fromLanding) {
    // ── Register checkpoint ──
    const baseCkpt = session.workflow.baseCkptPath || session.workflow.ckptPath;
    _initialCkptPath = baseCkpt;
    _selectedCkptPath = session.workflow.ckptPath || baseCkpt;
    _selectedWorkflow = session.workflow.type;
    state.workflowType = session.workflow.type;

    try {
        await authFetch(`${API_BASE}/register-checkpoint`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ckpt_path: baseCkpt,
                workflow_type: session.workflow.type,
            }),
        });
    } catch (e) {
        console.warn('Checkpoint registration failed during restore:', e.message);
    }

    // ── Recreate server-side job FIRST (before UI transition) ──
    const resp = await authFetch(`${API_BASE}/api/session/restore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(session),
    });
    if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody.detail || 'Server failed to restore session');
    }
    const data = await resp.json();
    state.jobId = data.job_id;

    // ── Transition to main app if from landing ──
    if (fromLanding) {
        const landing = document.getElementById('landing-page');
        const mainApp = document.getElementById('main-app');
        landing.classList.add('fade-out');
        await new Promise(resolve => {
            setTimeout(() => {
                landing.classList.add('hidden');
                mainApp.classList.remove('hidden');
                initMainApp();
                resolve();
            }, 450);
        });
    } else {
        resetCurrentSession(true);
    }

    // ── Re-set jobId (resetCurrentSession clears it) ──
    state.jobId = data.job_id;

    // ── Restore state properties ──
    state.genMode = session.generation?.genMode || 'denovo';
    state.genUsedOptimization = session.generation?.genUsedOptimization || false;
    state.refHasExplicitHs = session.generation?.refHasExplicitHs || false;
    state.iterationIdx = session.generation?.iterationIdx || 0;
    state.activeResultIdx = session.generation?.activeResultIdx ?? -1;

    const rawAll = session.generation?.allGeneratedResults || [];
    state.allGeneratedResults = rawAll.map(round => ({
        iteration: round.iteration,
        results: round.results,
    }));
    state.generatedResults = session.generation?.generatedResults || [];
    // Sync generatedResults Set conversion if it belongs to last round
    if (state.allGeneratedResults.length > 0) {
        const lastRound = state.allGeneratedResults.at(-1);
        state.generatedResults = lastRound.results;
        state.iterationIdx = lastRound.iteration + 1;
    }

    // Settings
    state.fixedAtoms = new Set(session.settings?.fixedAtoms || []);
    state.numHeavyAtoms = session.settings?.numHeavyAtoms ?? null;
    state.pocketOnlyMode = session.settings?.pocketOnlyMode || false;
    state.lbddScratchMode = session.settings?.lbddScratchMode || false;
    state.anisotropicPrior = session.settings?.anisotropicPrior || false;
    state.refLigandComPrior = session.settings?.refLigandComPrior || false;

    // Restore seed input
    const seedInput = document.getElementById('gen-seed');
    if (seedInput) seedInput.value = session.settings?.seed ?? 42;

    // ── Restore generation parameters ──
    const gp = session.generationParams;
    if (gp) {
        const _setVal = (id, val) => { const el = document.getElementById(id); if (el && val != null) el.value = val; };
        const _setChk = (id, val) => { const el = document.getElementById(id); if (el && val != null) el.checked = val; };
        _setVal('n-samples', gp.nSamples);
        _setVal('batch-size', gp.batchSize);
        _setVal('integration-steps', gp.integrationSteps);
        _setVal('pocket-cutoff', gp.pocketCutoff);
        _setChk('add-noise', gp.addNoise);
        _setVal('noise-scale', gp.noiseScale);
        _setChk('sample-mol-sizes', gp.sampleMolSizes);
        _setChk('filter-valid-unique', gp.filterValidUnique);
        _setChk('filter-cond-substructure', gp.filterCondSubstructure);
        _setChk('filter-diversity', gp.filterDiversity);
        _setVal('diversity-threshold', gp.diversityThreshold);
        _setChk('optimize-gen-ligs', gp.optimizeGenLigs);
        _setChk('optimize-gen-ligs-hs', gp.optimizeGenLigsHs);
        _setChk('calculate-pb-valid', gp.calculatePbValid);
        _setChk('filter-pb-valid', gp.filterPbValid);
        _setChk('calculate-strain', gp.calculateStrain);
        _setVal('grow-size', gp.growSize);
        _setVal('lbdd-optimize-method', gp.lbddOptimizeMethod);
        // Fire input events for sliders so display labels update
        ['n-samples', 'batch-size', 'integration-steps', 'pocket-cutoff', 'noise-scale', 'diversity-threshold'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.dispatchEvent(new Event('input'));
        });
    }

    // Original ligand, prior cloud
    state.originalLigand = session.originalLigand || null;
    state.priorCloud = session.priorCloud || null;
    state._priorPlacedCenter = session.priorPlacedCenter || null;

    // Active learning
    state._alFinetuned = session.activeLearning?.finetuned || false;
    state._alCkptPath = session.activeLearning?.ckptPath || null;
    state._alCheckpointSaved = !!state._alFinetuned;
    state._alRound = session.activeLearning?.round || 0;

    // Server-side validation: if AL ckpt didn't exist on server, clear frontend state
    if (state._alFinetuned && data.al_valid === false) {
        state._alFinetuned = false;
        state._alCkptPath = null;
        state._alCheckpointSaved = false;
    }
    _updateModelCard();

    // Clear transient state
    state._conformerJobId = null;
    state.showingInteractions = null;

    // ── Suppress intermediate zoomTo() during restore ──
    state._restoringSession = true;

    // ── Restore protein & ligand ──
    if (session.protein) _restoreProteinUI(session.protein, data.job_id);
    if (session.ligand) _restoreLigandUI(session.ligand, data.job_id);

    // ── Apply workflow mode ──
    _applyWorkflowMode(state.workflowType);
    _updateSidebarVisibility();

    // ── Render generated results ──
    if (state.generatedResults.length > 0) {
        await renderResults({
            results: state.generatedResults,
            used_optimization: state.genUsedOptimization,
        });
        if (state.activeResultIdx >= 0 && state.activeResultIdx < state.generatedResults.length) {
            showGeneratedLigand(state.activeResultIdx);
        }
        // Sync Select All button with checkbox state
        updateSaveSelectedCount();
    }

    // ── Re-render prior cloud ──
    if (state.priorCloud) {
        renderPriorCloud(state.priorCloud);
    }

    // ── Restore fixed-atom selection UI ──
    if (session.ligand && state.fixedAtoms.size > 0) {
        updateSelectionUI();
    }

    // ── Show num heavy atoms field in pocket-only mode ──
    if (state.numHeavyAtoms != null && (state.pocketOnlyMode || state.lbddScratchMode)) {
        _showNumHeavyAtomsField(state.numHeavyAtoms, '(restored)');
    }

    // ── Clear restore flag, then apply final view + zoom ──
    state._restoringSession = false;
    _restoreViewState(session.viewState || {});
    state.viewer.zoomTo();
    state.viewer.render();

    showToast('Session restored', 'success');
}

async function _handleSessionFileSelect(file, fromLanding) {
    if (!file) return;
    if (!file.name.endsWith('.json')) {
        showToast('Please select a .json session file', 'error');
        return;
    }

    const status = fromLanding ? document.getElementById('landing-status') : null;
    if (status) {
        status.className = 'landing-status status-loading';
        status.classList.remove('hidden');
        status.textContent = 'Restoring session\u2026';
    }

    try {
        const text = await file.text();
        const session = JSON.parse(text);
        _validateSessionSchema(session);
        await restoreSession(session, fromLanding);
    } catch (e) {
        console.error('Session restore failed:', e);
        showToast('Failed to restore session: ' + e.message, 'error');
    } finally {
        if (status) status.classList.add('hidden');
    }
}

function _validateSessionSchema(session) {
    if (session.version !== 1) throw new Error('Unsupported session version');
    if (!session.workflow?.type || !session.workflow?.ckptPath) {
        throw new Error('Invalid session file: missing workflow data');
    }
}

/**
 * Build a combined multi-molecule SDF string from a results array.
 * Each molecule's SDF block ends with "$$$$\n" as per the standard format.
 * @param {Array} results - Array of result objects with .sdf property
 * @param {Array} [indices] - Optional subset of indices; if omitted, uses all results
 */
function _buildCombinedSdf(results, indices) {
    const parts = [];
    const idxList = indices || results.map((_, i) => i);
    for (const idx of idxList) {
        const r = results[idx];
        if (!r?.sdf) continue;
        let block = r.sdf.trimEnd();
        // Ensure each molecule block ends with the $$$$ record separator
        if (!block.endsWith('$$$$')) block += '\n$$$$';
        parts.push(block);
    }
    return parts.join('\n');
}

function updateSaveSection() {
    const section = document.getElementById('save-section');
    if (!section) return;

    if (state.generatedResults.length === 0 || state.workflowType === 'lbdd') {
        section.classList.add('hidden');
        return;
    }
    section.classList.remove('hidden');
}

function toggleAllSaveCheckboxes() {
    const cbs = document.querySelectorAll('.save-ligand-cb');
    const allChecked = cbs.length > 0 && Array.from(cbs).every(cb => cb.checked);
    const newState = !allChecked;
    cbs.forEach(cb => cb.checked = newState);
    // Update toggle button visual state
    const btn = document.getElementById('select-all-toggle-btn');
    if (btn) btn.classList.toggle('btn-toggled', newState);
    updateSaveSelectedCount();
    _updateOverlayModels();
}

function updateSaveSelectedCount() {
    const ligandCount = document.querySelectorAll('.save-ligand-cb:checked').length;
    const cbs = document.querySelectorAll('.save-ligand-cb');
    const el = document.getElementById('bulk-actions-count');
    if (el) el.textContent = `${ligandCount} selected`;
    // Sync toggle button visual with checkbox state
    const btn = document.getElementById('select-all-toggle-btn');
    if (btn) btn.classList.toggle('btn-toggled', cbs.length > 0 && ligandCount === cbs.length);
    _updateOverlayModels();
}

function getSelectedSaveIndices() {
    const indices = [];
    document.querySelectorAll('.save-ligand-cb:checked').forEach(cb => {
        indices.push(Number.parseInt(cb.dataset.idx));
    });
    return indices;
}

/**
 * Map checked checkbox positions to original (pre-rank-select) indices.
 * After rank & select, state.generatedResults[i]._original_idx gives the
 * original position in the generation results. Without rank & select,
 * the checkbox index IS the original index.
 */
function _getSelectedOriginalIndices() {
    const checkedPositions = getSelectedSaveIndices();
    return checkedPositions.map(i => {
        const r = state.generatedResults[i];
        return (r?._original_idx != null) ? r._original_idx : i;
    });
}

/**
 * Filter server-returned ligand array by current checkbox selection.
 * Previous rounds are always included (their ligands were "kept").
 * Current round is filtered to only checked ligands.
 * If nothing is selected in the current round, current-round ligands are excluded.
 */
function _filterLigandsBySelection(ligands) {
    const currentIter = state.iterationIdx > 0 ? state.iterationIdx - 1 : 0;
    const selectedOriginal = _getSelectedOriginalIndices();
    const selectedSet = new Set(selectedOriginal);
    return ligands.filter(l => {
        if ((l.iteration ?? 0) !== currentIter) return true;
        return selectedSet.has(l.local_idx ?? l.idx);
    });
}

/**
 * Filter affinity distribution data by current checkbox selection.
 * Affinity data uses round-based parallel arrays {values[], labels[], indices[]}.
 * Previous rounds are kept in full; current round is filtered to checked indices.
 */
function _filterAffinityBySelection(data) {
    const currentIter = state.iterationIdx > 0 ? state.iterationIdx - 1 : 0;
    const selectedOriginal = _getSelectedOriginalIndices();
    const selectedSet = new Set(selectedOriginal);
    const filteredDistributions = {};
    const filteredTypes = [];
    for (const affType of (data.affinity_types || [])) {
        const rounds = data.distributions[affType];
        if (!rounds) continue;
        const filteredRounds = [];
        for (const round of rounds) {
            if ((round.iteration ?? 0) !== currentIter) {
                filteredRounds.push(round);
                continue;
            }
            const vals = [], labs = [], idxs = [];
            for (let i = 0; i < round.indices.length; i++) {
                if (selectedSet.has(round.indices[i])) {
                    vals.push(round.values[i]);
                    labs.push(round.labels[i]);
                    idxs.push(round.indices[i]);
                }
            }
            if (vals.length > 0) filteredRounds.push({ iteration: round.iteration, values: vals, labels: labs, indices: idxs });
        }
        if (filteredRounds.length > 0) {
            filteredDistributions[affType] = filteredRounds;
            filteredTypes.push(affType);
        }
    }
    return { ...data, affinity_types: filteredTypes, distributions: filteredDistributions };
}

async function saveSelectedLigands() {
    const indices = getSelectedSaveIndices();
    if (indices.length === 0) {
        showToast('No ligands selected for saving', 'error');
        return;
    }

    try {
        const resp = await authFetch(`${API_BASE}/save-selected-ligands/${state.jobId}`, {
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

    const jobTag = state.jobId ? `_${state.jobId.slice(0, 8)}` : '';

    // Download current round's selected ligands
    const currentRound = state.allGeneratedResults.length > 0
        ? state.allGeneratedResults.length
        : 1;
    const sdf = _buildCombinedSdf(state.generatedResults, indices);
    if (sdf) {
        const filename = `flowr_round${currentRound}_selected${jobTag}_${indices.length}ligands.sdf`;
        _downloadTextFile(sdf, filename);
    }

    // Also download all previous rounds as separate files
    if (state.allGeneratedResults.length > 1) {
        for (let ri = 0; ri < state.allGeneratedResults.length - 1; ri++) {
            const round = state.allGeneratedResults[ri];
            const roundSdf = _buildCombinedSdf(round.results);
            if (roundSdf) {
                const fn = `flowr_round${round.iteration + 1}${jobTag}_${round.results.length}ligands.sdf`;
                _downloadTextFile(roundSdf, fn);
            }
        }
    }

    showToast(`Downloaded ligands from ${state.allGeneratedResults.length} round(s)`, 'success');
    _showSaveStatus(`Downloaded ligands from ${state.allGeneratedResults.length} round(s)`);
}

async function saveAllLigands() {
    if (!state.jobId) return;
    try {
        const resp = await authFetch(`${API_BASE}/save-all-ligands/${state.jobId}`, { method: 'POST' });
        const data = await resp.json();
        showToast(`Saved ${data.saved_count} ligands to output/`, 'success');
        _showSaveStatus(`Saved ${data.saved_count} files to output/`);
    } catch (e) {
        showToast('Error saving ligands: ' + e.message, 'error');
    }

    const jobTag = state.jobId ? `_${state.jobId.slice(0, 8)}` : '';
    let totalDownloaded = 0;

    // Download one SDF file per round
    const rounds = state.allGeneratedResults.length > 0
        ? state.allGeneratedResults
        : [{ iteration: 0, results: state.generatedResults }];

    for (const round of rounds) {
        const sdf = _buildCombinedSdf(round.results);
        if (!sdf) continue;
        const nLigs = round.results.length;
        const filename = `flowr_round${round.iteration + 1}${jobTag}_${nLigs}ligands.sdf`;
        _downloadTextFile(sdf, filename);
        totalDownloaded += nLigs;
    }

    if (totalDownloaded === 0) {
        showToast('No SDF data available', 'error');
        return;
    }

    showToast(`Downloaded ${totalDownloaded} ligands across ${rounds.length} round(s)`, 'success');
    _showSaveStatus(`Downloaded ${totalDownloaded} ligands across ${rounds.length} round(s)`);
}

function _showSaveStatus(msg) {
    const el = document.getElementById('save-status');
    if (!el) return;
    el.classList.remove('hidden');
    el.textContent = msg;
    setTimeout(() => el.classList.add('hidden'), 4000);
}

// =========================================================================
// Active Learning
// =========================================================================

function _getALSelectedIndices() {
    const cbs = document.querySelectorAll('.save-ligand-cb:checked');
    return Array.from(cbs).map(cb => Number.parseInt(cb.dataset.idx, 10));
}

function _estimateEpochs(n) {
    if (n <= 100) return 2;
    if (n <= 300) return 3;
    if (n <= 500) return 5;
    if (n <= 1000) return 8;
    return 10;
}

function _estimateAccBatches(n) {
    if (n <= 100) return 2;
    if (n <= 300) return 3;
    if (n <= 500) return 5;
    if (n <= 1000) return 8;
    return 12;
}

async function showActiveLearningModal() {
    const indices = _getALSelectedIndices();
    if (indices.length === 0) {
        showToast('Select ligands first (use checkboxes on ligand cards)', 'warning');
        return;
    }

    // Populate round selection checkboxes (optional: select ligands from previous rounds)
    const roundSelContainer = document.getElementById('al-round-selection');
    if (roundSelContainer) {
        if (state.allGeneratedResults.length > 1) {
            roundSelContainer.classList.remove('hidden');
            roundSelContainer.innerHTML = '<p class="al-round-sel-label">Include ligands from previous rounds:</p>';
            state.allGeneratedResults.forEach((round, ri) => {
                // Skip the current (latest) round — those are selected via checkboxes
                if (ri === state.allGeneratedResults.length - 1) return;
                const iter = round.iteration;
                const n = round.results.length;
                const rc = _roundColor(iter);
                const id = `al-round-cb-${iter}`;
                const row = document.createElement('label');
                row.className = 'al-round-cb-row';
                row.innerHTML = `<input type="checkbox" id="${id}" class="al-round-cb" data-iteration="${iter}" checked>` +
                    `<span class="round-dot" style="background:${rc.fill};border-color:${rc.line}"></span> ` +
                    `Round ${iter + 1} <span class="muted-text">(${n} ligands)</span>`;
                roundSelContainer.appendChild(row);
            });
        } else {
            roundSelContainer.classList.add('hidden');
            roundSelContainer.innerHTML = '';
        }
    }

    document.getElementById('al-n-selected').textContent = indices.length;
    document.getElementById('al-epochs-input').value = _estimateEpochs(indices.length);
    document.getElementById('al-lr-input').value = '5e-4';
    document.getElementById('al-batch-cost-input').value = 4;
    document.getElementById('al-acc-batches-input').value = _estimateAccBatches(indices.length);

    // Populate base model selector
    const alCkptSelect = document.getElementById('al-ckpt-select');
    if (alCkptSelect) {
        alCkptSelect.innerHTML = '<option value="">Loading…</option>';
        try {
            const ckptResp = await authFetch(`${API_BASE}/checkpoints?workflow=sbdd`);
            const ckptData = await ckptResp.json();
            let options = '';
            const currentPath = _selectedCkptPath || '';

            if (ckptData.base) {
                for (const ckpt of ckptData.base) {
                    const sel = ckpt.path === currentPath ? ' selected' : '';
                    options += `<option value="${escapeHtml(ckpt.path)}"${sel}>${escapeHtml(ckpt.name)}</option>`;
                }
            }
            if (ckptData.project) {
                for (const proj of ckptData.project) {
                    for (const ckpt of proj.checkpoints) {
                        const sel = ckpt.path === currentPath ? ' selected' : '';
                        options += `<option value="${escapeHtml(ckpt.path)}"${sel}>${escapeHtml(proj.project_id)} / ${escapeHtml(ckpt.name)}</option>`;
                    }
                }
            }

            alCkptSelect.innerHTML = options || '<option value="">No checkpoints found</option>';
        } catch (e) {
            console.debug('Failed to load checkpoints for AL modal:', e.message);
            alCkptSelect.innerHTML = '<option value="">Failed to load</option>';
        }
    }

    openModal('al-explain-modal');
}

async function startActiveLearning() {
    const indices = _getALSelectedIndices();
    if (indices.length === 0 || !state.jobId) return;

    // Gather selected previous round iterations
    const prevRoundIters = [];
    document.querySelectorAll('.al-round-cb:checked').forEach(cb => {
        prevRoundIters.push(Number.parseInt(cb.dataset.iteration, 10));
    });

    closeModal('al-explain-modal');
    state._alCancelling = false;

    // Show progress modal (locks screen)
    const modal = document.getElementById('al-progress-modal');
    modal.classList.remove('hidden');
    document.getElementById('al-progress-training').style.display = 'block';
    document.getElementById('al-progress-done').style.display = 'none';
    document.getElementById('al-progress-error').style.display = 'none';
    document.getElementById('al-progress-cancelled')?.style.setProperty('display', 'none');
    document.getElementById('al-cancel-btn')?.removeAttribute('disabled');
    document.getElementById('al-phase1-fill').style.width = '0%';
    document.getElementById('al-phase1-pct').textContent = '0%';
    document.getElementById('al-phase1-section').classList.remove('completed');
    document.getElementById('al-phase2-section').style.display = 'none';
    document.getElementById('al-phase2-fill').style.width = '0%';
    document.getElementById('al-phase2-pct').textContent = '0%';
    document.getElementById('al-progress-message').textContent = 'Starting LoRA finetuning...';

    try {
        // Start AL finetuning
        const resp = await authFetch(`${API_BASE}/active-learning/${state.jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: state.jobId,
                indices,
                prev_round_iterations: prevRoundIters,
                epochs: Number.parseInt(document.getElementById('al-epochs-input').value, 10) || null,
                lr: Number.parseFloat(document.getElementById('al-lr-input').value) || 5e-4,
                batch_cost: Number.parseInt(document.getElementById('al-batch-cost-input').value, 10) || 4,
                acc_batches: Number.parseInt(document.getElementById('al-acc-batches-input').value, 10) || null,
                ckpt_path: document.getElementById('al-ckpt-select')?.value || null,
            }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Failed to start active learning');
        }

        // Poll for progress
        await _pollActiveLearning();

    } catch (e) {
        document.getElementById('al-progress-training').style.display = 'none';
        document.getElementById('al-progress-error').style.display = 'block';
        document.getElementById('al-error-message').textContent = e.message;
    }
}

async function _pollActiveLearning() { // NOSONAR
    const timeout = 3600000; // 1 hour
    const start = Date.now();

    while (Date.now() - start < timeout) {
        await new Promise(r => setTimeout(r, 2000));

        try {
            const resp = await authFetch(`${API_BASE}/al-status/${state.jobId}`);
            if (!resp.ok) continue;
            const data = await resp.json();

            const pct = data.progress || 0;
            const phase = data.al_phase || 'preparing';

            if (phase === 'preparing') {
                document.getElementById('al-phase1-fill').style.width = pct + '%';
                document.getElementById('al-phase1-pct').textContent = pct + '%';
            } else if (phase === 'training') {
                // Mark phase 1 as completed
                const p1 = document.getElementById('al-phase1-section');
                if (!p1.classList.contains('completed')) {
                    p1.classList.add('completed');
                    document.getElementById('al-phase1-fill').style.width = '100%';
                    document.getElementById('al-phase1-pct').textContent = '✓';
                }
                // Show and update phase 2
                document.getElementById('al-phase2-section').style.display = 'block';
                document.getElementById('al-phase2-fill').style.width = pct + '%';
                document.getElementById('al-phase2-pct').textContent = pct + '%';
            }

            if (data.status_message) {
                document.getElementById('al-progress-message').textContent = data.status_message;
            }

            if (data.status === 'completed') {
                // Store finetuned checkpoint info
                state._alFinetuned = true;
                state._alCkptPath = data.finetuned_ckpt_path;
                state._alCheckpointSaved = false;
                state._alRound = (state._alRound || 0) + 1;
                _selectedCkptPath = data.finetuned_ckpt_path;
                _updateModelCard();

                document.getElementById('al-progress-training').style.display = 'none';
                document.getElementById('al-progress-done').style.display = 'block';
                return;
            } else if (data.status === 'cancelled') {
                document.getElementById('al-progress-training').style.display = 'none';
                document.getElementById('al-progress-cancelled').style.display = 'block';
                return;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Active learning finetuning failed');
            }
        } catch (e) {
            if (e.message && !e.message.includes('fetch')) {
                document.getElementById('al-progress-training').style.display = 'none';
                document.getElementById('al-progress-error').style.display = 'block';
                document.getElementById('al-error-message').textContent = e.message;
                return;
            }
        }
    }

    document.getElementById('al-progress-training').style.display = 'none';
    document.getElementById('al-progress-error').style.display = 'block';
    document.getElementById('al-error-message').textContent = 'Finetuning timed out.';
}

function closeActiveLearningDone() {
    document.getElementById('al-progress-modal').classList.add('hidden');
    if (state._alFinetuned) {
        _updateModelCard();
        showToast('Finetuned model active — next generation will use it', 'success');
    }
}

// =========================================================================
// Model Card – Save / Load
// =========================================================================

function _updateModelCard() {
    // Show/hide save button based on whether AL has finetuned a model
    const saveBtn = document.getElementById('model-save-btn');
    if (saveBtn) saveBtn.classList.toggle('hidden', !state._alFinetuned);

    // Update current model name display
    const nameEl = document.getElementById('model-card-name');
    if (nameEl) {
        if (state._alFinetuned) {
            const round = state._alRound || 1;
            nameEl.textContent = `LoRA-finetuned (Round ${round})`;
        } else {
            const path = _selectedCkptPath || '';
            const parts = path.replaceAll('\\', '/').split('/');
            const fname = parts[parts.length - 1] || '\u2014';
            nameEl.textContent = fname.replace(/\.ckpt$/, '');
        }
    }

    // Collapse save/load expansions
    document.getElementById('model-save-expand')?.classList.add('hidden');
    document.getElementById('model-load-expand')?.classList.add('hidden');
}

function expandSaveCheckpoint() {
    if (!state.jobId || !state._alFinetuned) return;

    const expand = document.getElementById('model-save-expand');
    const loadExpand = document.getElementById('model-load-expand');
    if (loadExpand) loadExpand.classList.add('hidden');

    expand.classList.toggle('hidden');
    if (!expand.classList.contains('hidden')) {
        const input = document.getElementById('model-save-name');
        input.value = '';
        input.focus();
        document.getElementById('model-save-status').classList.add('hidden');
    }
}

async function confirmSaveCheckpoint() {
    if (!state.jobId || !state._alFinetuned) return;

    const input = document.getElementById('model-save-name');
    const statusEl = document.getElementById('model-save-status');
    const name = (input.value || '').trim();

    if (!name) {
        statusEl.textContent = 'Please enter a name.';
        statusEl.className = 'model-save-status error';
        statusEl.classList.remove('hidden');
        return;
    }

    if (!/^[a-zA-Z0-9_-]{1,64}$/.test(name)) {
        statusEl.textContent = 'Use 1\u201364 alphanumeric characters, hyphens, or underscores.';
        statusEl.className = 'model-save-status error';
        statusEl.classList.remove('hidden');
        return;
    }

    const confirmBtn = document.getElementById('model-save-confirm');
    confirmBtn.disabled = true;

    try {
        const resp = await authFetch(`${API_BASE}/save-checkpoint/${state.jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            statusEl.textContent = err.detail || 'Failed to save checkpoint';
            statusEl.className = 'model-save-status error';
            statusEl.classList.remove('hidden');
            return;
        }
        const data = await resp.json();
        state._alCkptPath = data.path;
        state._alCheckpointSaved = true;
        _selectedCkptPath = data.path;

        statusEl.textContent = '\u2713 Checkpoint saved to ckpts folder';
        statusEl.className = 'model-save-status';
        statusEl.classList.remove('hidden');
        input.disabled = true;

        // Update model name without collapsing the save section
        const nameEl = document.getElementById('model-card-name');
        if (nameEl) {
            nameEl.textContent = data.name;
        }
        // Keep the status visible, then collapse
        setTimeout(() => {
            const expandEl = document.getElementById('model-save-expand');
            if (expandEl) expandEl.classList.add('hidden');
            input.disabled = false;
            _updateModelCard();
        }, 2500);
    } catch (e) {
        statusEl.textContent = 'Failed to save: ' + e.message;
        statusEl.className = 'model-save-status error';
        statusEl.classList.remove('hidden');
    } finally {
        confirmBtn.disabled = false;
    }
}

async function showLoadCheckpointList() {
    const expand = document.getElementById('model-load-expand');
    const saveExpand = document.getElementById('model-save-expand');
    if (saveExpand) saveExpand.classList.add('hidden');

    // Toggle
    if (!expand.classList.contains('hidden')) {
        expand.classList.add('hidden');
        return;
    }

    const list = document.getElementById('model-load-list');
    list.innerHTML = '<div class="model-load-loading">Loading checkpoints\u2026</div>';
    expand.classList.remove('hidden');

    try {
        const workflow = state.workflowType || 'sbdd';
        const resp = await authFetch(`${API_BASE}/checkpoints?workflow=${workflow}`);
        const data = await resp.json();

        let html = '';
        const currentPath = _selectedCkptPath || '';

        if (data.base && data.base.length > 0) {
            html += '<div class="model-load-group-title">Base Models</div>';
            for (const ckpt of data.base) {
                const isActive = ckpt.path === currentPath;
                html += `<button class="model-load-item${isActive ? ' active-ckpt' : ''}" data-path="${escapeHtml(ckpt.path)}"${isActive ? ' disabled' : ''}>${escapeHtml(ckpt.name)}${isActive ? ' \u2713' : ''}</button>`;
            }
        }
        if (data.project && data.project.length > 0) {
            html += '<div class="model-load-group-title">Project Models</div>';
            for (const proj of data.project) {
                for (const ckpt of proj.checkpoints) {
                    const isActive = ckpt.path === currentPath;
                    html += `<button class="model-load-item${isActive ? ' active-ckpt' : ''}" data-path="${escapeHtml(ckpt.path)}"${isActive ? ' disabled' : ''}>${escapeHtml(proj.project_id)} / ${escapeHtml(ckpt.name)}${isActive ? ' \u2713' : ''}</button>`;
                }
            }
        }

        if (!html) html = '<div class="model-load-loading">No checkpoints found</div>';
        list.innerHTML = html;

        list.querySelectorAll('.model-load-item:not(.active-ckpt)').forEach(btn => {
            btn.addEventListener('click', () => confirmLoadCheckpoint(btn.dataset.path));
        });
    } catch (e) {
        console.debug('Failed to load checkpoints:', e.message);
        list.innerHTML = '<div class="model-load-loading">Failed to load checkpoints</div>';
    }
}

async function confirmLoadCheckpoint(ckptPath) {
    if (!state.jobId) {
        showToast('No active session', 'error');
        return;
    }

    try {
        const resp = await authFetch(`${API_BASE}/swap-checkpoint/${state.jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ckpt_path: ckptPath }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            showToast(err.detail || 'Failed to load checkpoint', 'error');
            return;
        }
        const data = await resp.json();
        _selectedCkptPath = ckptPath;
        state._alFinetuned = false;
        state._alCkptPath = null;
        state._alCheckpointSaved = false;
        state._alRound = 0;
        _updateModelCard();
        document.getElementById('model-load-expand')?.classList.add('hidden');
        showToast(`Model switched to "${escapeHtml(data.name)}"`, 'success');
    } catch (e) {
        showToast('Failed to load checkpoint: ' + e.message, 'error');
    }
}

async function cancelActiveLearning() {
    if (!state.jobId || state._alCancelling) return;
    state._alCancelling = true;

    const btn = document.getElementById('al-cancel-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Cancelling…';
    }

    try {
        await authFetch(`${API_BASE}/cancel/${state.jobId}`, { method: 'POST' });
    } catch (e) { console.debug('Server stop notification failed:', e.message); }
}

// =========================================================================
// 3D Interaction Visualization
// =========================================================================

async function showInteractions3D(target) { // NOSONAR
    if (!state.jobId) return;

    const btnRef = document.getElementById('btn-int-ref');

    // Toggle off if already showing
    if (state.showingInteractions === 'both') {
        clearInteractions3D();
        return;
    }

    clearInteractions3D();
    state.showingInteractions = 'both';

    const hasRefLigand = !!state.ligandData && !state.pocketOnlyMode && !state.lbddScratchMode;
    let totalInteractions = 0;

    if (state.activeView === 'ref') {
        // ── Reference is the full-atom view ──
        // Show full interactions for reference
        if (hasRefLigand) {
            try {
                const refResp = await authFetch(`${API_BASE}/compute-interactions/${state.jobId}?ligand_idx=-1`);
                if (refResp.ok) {
                    const refData = await refResp.json();
                    totalInteractions += (refData.interactions || []).length;
                    renderInteractions3D(refData.interactions, 'ref', false);
                }
            } catch (e) {
                console.warn('Failed to compute ref interactions:', e);
            }
        }
        // Also show dimmed interactions for the first selected generated ligand (if any)
        const selectedIndices = getSelectedSaveIndices();
        if (selectedIndices.length > 0) {
            const firstIdx = selectedIndices[0];
            try {
                const genResp = await authFetch(`${API_BASE}/compute-interactions/${state.jobId}?ligand_idx=${firstIdx}`);
                if (genResp.ok) {
                    const genData = await genResp.json();
                    totalInteractions += (genData.interactions || []).length;
                    renderInteractions3D(genData.interactions, 'gen', true);
                }
            } catch (e) {
                console.warn('Failed to compute gen interactions:', e);
            }
        }
    } else {
        // ── Generated ligand is the full-atom view (existing behaviour) ──
        const hasGenSelected = state.activeResultIdx >= 0;

        // Show interactions for reference ligand (dimmed if gen is also showing)
        if (hasRefLigand && state.refLigandVisible) {
            try {
                const refResp = await authFetch(`${API_BASE}/compute-interactions/${state.jobId}?ligand_idx=-1`);
                if (refResp.ok) {
                    const refData = await refResp.json();
                    totalInteractions += (refData.interactions || []).length;
                    renderInteractions3D(refData.interactions, 'ref', hasGenSelected);
                }
            } catch (e) {
                console.warn('Failed to compute ref interactions:', e);
            }
        }

        // Show full interactions for generated ligand
        if (hasGenSelected) {
            try {
                const genResp = await authFetch(`${API_BASE}/compute-interactions/${state.jobId}?ligand_idx=${state.activeResultIdx}`);
                if (genResp.ok) {
                    const genData = await genResp.json();
                    totalInteractions += (genData.interactions || []).length;
                    renderInteractions3D(genData.interactions, 'gen', false);
                }
            } catch (e) {
                console.warn('Failed to compute gen interactions:', e);
            }
        }
    }

    // Provide user feedback
    if (!hasRefLigand && state.activeResultIdx < 0) {
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
    const opacity = dimmed ? 0.3 : 1;
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
            const sel = { model: state.proteinModel, chain: chain, resi: Number.parseInt(resSeq) };
            state.viewer.addStyle(sel, {
                stick: { radius: 0.12, colorscheme: 'Jmol' },
            });
        });
    }

    state.viewer.render();
}

function clearInteractions3D() {
    state.interactionShapes.forEach(s => {
        try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
    });
    state.interactionShapes = [];
    state.showingInteractions = null;

    // Reset protein style to remove residue sticks
    if (state.proteinModel) {
        applyProteinStyle();
        reapplyHoverable();
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
    if (!cloud?.points || cloud.points.length === 0) return;

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
        btn.classList.add('active');
    }
    document.getElementById('prior-cloud-sep')?.classList.remove('hidden');
    document.getElementById('prior-cloud-group')?.classList.remove('hidden');

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
        btn.classList.remove('active');
    }
    document.getElementById('prior-cloud-sep')?.classList.add('hidden');
    document.getElementById('prior-cloud-group')?.classList.add('hidden');
    const resetBtn = document.getElementById('reset-prior-pos-btn');
    if (resetBtn) resetBtn.classList.add('hidden');
}

/** Remove only the visual spheres (preserves placement state). */
function _clearPriorCloudSpheres() {
    state.priorCloudSpheres.forEach(s => {
        try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
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
            try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
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
        btn.innerHTML = '<span id="place-prior-icon"></span> Placing…';
        canvas.classList.add('prior-placing');
        _attachPriorPlacementListeners();
        showToast('Placement mode ON — Shift+click to place, Shift+drag to move the cloud. Release Shift to rotate/zoom normally.', 'success');
    } else {
        btn.classList.remove('placing');
        btn.innerHTML = '<span id="place-prior-icon"></span> Place';
        canvas.classList.remove('prior-placing', 'prior-dragging');
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
        btn.innerHTML = '<span id="place-prior-icon"></span> Place';
    }
    const canvas = document.getElementById('viewer3d');
    if (canvas) {
        canvas.classList.remove('prior-placing', 'prior-dragging');
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
        const speed = Math.max(0.05, 50 / Math.max(pixPerAng, 1));
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
    const eps = 1;  // 1 Angstrom probe
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
    const eps = 1;

    // Probe along each axis to find which world direction corresponds to screen depth
    const probeX = viewer.modelToScreen({ x: c.x + eps, y: c.y, z: c.z });
    const probeY = viewer.modelToScreen({ x: c.x, y: c.y + eps, z: c.z });
    const probeZ = viewer.modelToScreen({ x: c.x, y: c.y, z: c.z + eps });

    // Screen Z component per world axis
    const dzWx = probeX.z - screenC.z;
    const dzWy = probeY.z - screenC.z;
    const dzWz = probeZ.z - screenC.z;

    // Normalize to get the world-space direction of screen depth
    const norm = Math.hypot(dzWx, dzWy, dzWz);
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
        try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
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
        try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
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
        const resp = await authFetch(maskUrl);
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

        // Pre-populate fixedAtoms with the auto-detected replaced atoms
        // so the user can shift+click to modify them
        if (_ATOM_SELECT_MODES.has(mode) && mode !== 'substructure_inpainting') {
            // Only pre-populate if user hasn't already made a manual selection
            if (state.fixedAtoms.size === 0) {
                // Filter to heavy atoms only for fixedAtoms
                replacedIndices.forEach(idx => {
                    const a = state.ligandData._atomByIdx.get(idx);
                    if (a && a.atomicNum !== 1) {
                        state.fixedAtoms.add(idx);
                    }
                });
                // Store the auto-detected mask for reference
                state._autoDetectedMask = new Set(state.fixedAtoms);
                updateSelectionUI();
            }
        }

        // Draw fixed atoms (blue) – only for modes that keep atoms fixed
        if (isFixedMode) {
            fixedIndices.forEach(idx => {
                const atom = state.ligandData._atomByIdx.get(idx);
                if (!atom || atom.atomicNum === 1) return;
                const pos = { x: atom.x, y: atom.y, z: atom.z };
                const pair = _addDualSphere(state.viewer, pos, COLORS.kept);
                state.autoHighlightSpheres.push(pair.inner, pair.outer);
            });
        }

        // Draw replaced atoms (red) – all modes show these
        replacedIndices.forEach(idx => {
            const atom = state.ligandData._atomByIdx.get(idx);
            if (!atom || atom.atomicNum === 1) return;
            const pos = { x: atom.x, y: atom.y, z: atom.z };
            const pair = _addDualSphere(state.viewer, pos, COLORS.replaced);
            state.autoHighlightSpheres.push(pair.inner, pair.outer);
        });

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

/**
 * Sync the auto-highlight spheres to reflect manual atom selection changes.
 * Rebuild the visual highlighting based on the current fixedAtoms set.
 */
function _syncAutoHighlightWithSelection() {
    if (!state.ligandData) return;
    // Clear old auto-highlight spheres
    state.autoHighlightSpheres.forEach(s => {
        try { state.viewer.removeShape(s); } catch (e) { console.debug('Shape already removed:', e.message); }
    });
    state.autoHighlightSpheres = [];

    const fixedModes = ['core_growing', 'fragment_growing'];
    const isFixedMode = fixedModes.includes(state.genMode);
    const totalHeavy = state.ligandData?.num_heavy_atoms || 0;

    // Draw all non-selected atoms (kept/fixed) in blue for fixed modes
    // Draw all selected atoms (to-be-replaced) in red
    state.ligandData._atomByIdx.forEach((atom, idx) => {
        if (atom.atomicNum === 1) return;
        const isSelected = state.fixedAtoms.has(idx);
        if (isFixedMode) {
            const color = isSelected ? COLORS.replaced : COLORS.kept;
            const pos = { x: atom.x, y: atom.y, z: atom.z };
            const pair = _addDualSphere(state.viewer, pos, color);
            state.autoHighlightSpheres.push(pair.inner, pair.outer);
        } else if (isSelected) {
            // For replaced-atom modes, only show selected atoms in red
            const pos = { x: atom.x, y: atom.y, z: atom.z };
            const pair = _addDualSphere(state.viewer, pos, COLORS.replaced);
            state.autoHighlightSpheres.push(pair.inner, pair.outer);
        }
    });

    state.viewer.render();

    // Update legend with current counts
    const nSelected = state.fixedAtoms.size;
    const nFixed = totalHeavy - nSelected;
    _showInpaintingLegend(state.genMode, nFixed, nSelected, isFixedMode);
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
            modeSelector.after(legend);
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
            ? (Number.parseInt(document.getElementById('grow-size')?.value) || 5)
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
            try { Plotly.purge(el); } catch (e) { console.debug('Plot already purged:', e.message); }
        });
        modal.classList.add('hidden');
    }
}

function onModalOverlayClick(event, modalId) {
    if (event.target.classList.contains('modal-overlay') || event.target.classList.contains('al-overlay')) {
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
    if (!state.jobId || (state.generatedResults.length === 0 && state.allGeneratedResults.length === 0)) {
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
        const resp = await authFetch(`${API_BASE}/chemical-space/${state.jobId}?method=${method}`);
        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(errText);
        }
        const data = await resp.json();
        loading.classList.add('hidden');
        data.ligands = _filterLigandsBySelection(data.ligands);
        if (data.ligands.length === 0) {
            plotDiv.innerHTML = '<div class="modal-error">No ligands selected. Use the checkboxes or Select All to choose ligands for visualization.</div>';
            return;
        }
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

    // Group ligands by iteration round
    const roundMap = new Map(); // iteration → [{x,y,text,idx,localIdx,iteration}]
    data.ligands.forEach(l => {
        const iter = l.iteration ?? 0;
        if (!roundMap.has(iter)) roundMap.set(iter, []);
        const p = l.properties || {};
        roundMap.get(iter).push({
            x: l.x, y: l.y, idx: l.idx, localIdx: l.local_idx ?? l.idx, iteration: iter,
            text: `<b>Round ${iter + 1} · Ligand #${(l.local_idx ?? l.idx) + 1}</b><br>` +
                `MW: ${p.MolWt ?? '–'}<br>` +
                `LogP: ${p.LogP ?? '–'}<br>` +
                `TPSA: ${p.TPSA ?? '–'}<br>` +
                `Fsp3: ${p.FractionCSP3 ?? '–'}<br>` +
                `HBA: ${p.NumHAcceptors ?? '–'}<br>` +
                `RotBonds: ${p.NumRotatableBonds ?? '–'}`,
        });
    });

    // Build one trace per round
    const useGl = data.ligands.length > 500;
    const traces = [];
    const sortedRounds = Array.from(roundMap.keys()).sort((a, b) => a - b);
    const latestIter = sortedRounds.length > 0 ? sortedRounds.at(-1) : 0;

    // ── KDE density contours (one per round, rendered behind scatter points) ──
    const MIN_POINTS_FOR_KDE = 5;
    sortedRounds.forEach(iter => {
        const pts = roundMap.get(iter);
        if (pts.length < MIN_POINTS_FOR_KDE) return;
        const rc = _roundColor(iter);
        const [lr, lg, lb] = _hexToRgb(rc.line);
        traces.push({
            x: pts.map(p => p.x),
            y: pts.map(p => p.y),
            type: 'histogram2dcontour',
            colorscale: _buildKdeColorscale(rc.fill),
            ncontours: 12,
            showscale: false,
            showlegend: false,
            hoverinfo: 'skip',
            contours: {
                coloring: 'fill',
                showlines: true,
            },
            line: { width: 0.5, color: `rgba(${lr},${lg},${lb},0.25)` },
        });
    });

    sortedRounds.forEach(iter => {
        const pts = roundMap.get(iter);
        const rc = _roundColor(iter);
        traces.push({
            x: pts.map(p => p.x),
            y: pts.map(p => p.y),
            mode: 'markers',
            type: useGl ? 'scattergl' : 'scatter',
            name: `Round ${iter + 1}`,
            text: pts.map(p => p.text),
            hovertemplate: '%{text}<extra></extra>',
            customdata: pts.map(p => ({ localIdx: p.localIdx, iteration: p.iteration })),
            marker: {
                size: 9,
                color: rc.fill,
                line: { width: 1.5, color: rc.line },
                opacity: 0.85,
            },
        });
    });

    // Reference ligand trace
    const refP = data.reference.properties || {};
    const refText = `<b>Reference Ligand</b><br>` +
        `MW: ${refP.MolWt ?? '–'}<br>` +
        `LogP: ${refP.LogP ?? '–'}<br>` +
        `TPSA: ${refP.TPSA ?? '–'}<br>` +
        `Fsp3: ${refP.FractionCSP3 ?? '–'}<br>` +
        `HBA: ${refP.NumHAcceptors ?? '–'}<br>` +
        `RotBonds: ${refP.NumRotatableBonds ?? '–'}`;

    traces.push({
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
            opacity: 1,
        },
    });

    // Original ligand trace (shown when reference has been swapped)
    if (data.original) {
        const origP = data.original.properties || {};
        const origText = `<b>Original Ligand</b><br>` +
            `MW: ${origP.MolWt ?? '–'}<br>` +
            `LogP: ${origP.LogP ?? '–'}<br>` +
            `TPSA: ${origP.TPSA ?? '–'}<br>` +
            `Fsp3: ${origP.FractionCSP3 ?? '–'}<br>` +
            `HBA: ${origP.NumHAcceptors ?? '–'}<br>` +
            `RotBonds: ${origP.NumRotatableBonds ?? '–'}`;

        traces.push({
            x: [data.original.x],
            y: [data.original.y],
            mode: 'markers',
            type: 'scatter',
            name: 'Original',
            text: [origText],
            hovertemplate: '%{text}<extra></extra>',
            marker: {
                size: 14,
                color: '#c9943c',
                symbol: 'diamond',
                line: { width: 2, color: '#8b6914' },
                opacity: 1,
            },
        });
    }

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
    Plotly.newPlot(plotDiv, traces, layout, config);

    // Click on generated ligand dot → show that ligand in the sidebar
    // Only responds to clicks on the latest round (those are in state.generatedResults)
    plotDiv.on('plotly_click', (eventData) => {
        const point = eventData.points[0];
        if (point?.data.name === 'Reference') return;
        const cd = point?.customdata;
        if (cd && cd.iteration === latestIter) {
            showGeneratedLigand(cd.localIdx);
        }
    });
}

// =========================================================================
// Property Space Visualization (Violin Plots)
// =========================================================================

async function showPropertySpace() {
    if (!state.jobId || (state.generatedResults.length === 0 && state.allGeneratedResults.length === 0)) {
        showToast('Generate ligands first', 'error');
        return;
    }
    openModal('propspace-modal');

    const loading = document.getElementById('propspace-loading');
    const grid = document.getElementById('propspace-grid');
    loading.classList.remove('hidden');
    grid.innerHTML = '';

    try {
        const resp = await authFetch(`${API_BASE}/property-space/${state.jobId}`);
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        loading.classList.add('hidden');
        data.ligands = _filterLigandsBySelection(data.ligands);
        if (data.ligands.length === 0) {
            grid.innerHTML = '<div class="modal-error">No ligands selected. Use the checkboxes or Select All to choose ligands for visualization.</div>';
            return;
        }
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
    const origProps = data.original || null;
    const allProps = data.property_names || [];
    const contProps = new Set(data.continuous_properties || []);

    // Determine distinct iterations present in the data
    const iterations = new Set(data.ligands.map(l => l.iteration ?? 0));
    const sortedIters = Array.from(iterations).sort((a, b) => a - b);
    const multiRound = sortedIters.length > 1;

    allProps.forEach(propName => { // NOSONAR
        // Collect values per round for this property
        const roundValues = new Map(); // iteration → values[]
        data.ligands.forEach(lig => {
            const v = lig.properties?.[propName];
            if (v === null || v === undefined) return;
            const iter = lig.iteration ?? 0;
            if (!roundValues.has(iter)) roundValues.set(iter, []);
            roundValues.get(iter).push(v);
        });
        // Skip if no values at all
        const totalValues = Array.from(roundValues.values()).reduce((s, a) => s + a.length, 0);
        if (totalValues === 0) return;

        const refVal = refProps[propName];
        const origVal = origProps?.[propName] ?? null;
        const hasOrig = origVal !== null && origVal !== undefined;
        const isContinuous = contProps.has(propName);
        const hasRef = refVal !== null && refVal !== undefined;

        // Create container div
        const div = document.createElement('div');
        div.className = 'propspace-cell';
        const plotId = `propspace-plot-${propName}`;
        div.innerHTML = `<div id="${plotId}" class="propspace-violin"></div>`;
        grid.appendChild(div);

        // Format display name
        const displayName = _propDisplayName(propName);

        if (isContinuous) {
            const traces = [];
            sortedIters.forEach(iter => {
                const vals = roundValues.get(iter);
                if (!vals || vals.length === 0) return;
                const rc = _roundColor(iter);
                traces.push({
                    type: 'violin',
                    y: vals,
                    name: `Round ${iter + 1}`,
                    box: { visible: true },
                    meanline: { visible: true },
                    fillcolor: rc.fillAlpha,
                    line: { color: rc.line, width: 1.5 },
                    points: vals.length <= 50 ? 'all' : false,
                    jitter: 0.3,
                    pointpos: -1.5,
                    marker: { color: rc.line, size: 4, opacity: 0.6 },
                    hoverinfo: 'y',
                    spanmode: 'hard',
                });
            });

            const annotations = [];
            if (hasRef) {
                annotations.push({
                    y: refVal, x: 0, xref: 'x', yref: 'y',
                    text: `Ref: ${typeof refVal === 'number' ? refVal.toFixed(2) : refVal}`,
                    showarrow: true, arrowhead: 0, arrowwidth: 1.5, arrowcolor: '#cc3333',
                    ax: 65, ay: 0,
                    font: { color: '#cc3333', size: 11, family: 'JetBrains Mono, monospace' },
                    bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#cc3333', borderwidth: 1, borderpad: 3,
                });
            }
            if (hasOrig) {
                annotations.push({
                    y: origVal, x: 0, xref: 'x', yref: 'y',
                    text: `Orig: ${typeof origVal === 'number' ? origVal.toFixed(2) : origVal}`,
                    showarrow: true, arrowhead: 0, arrowwidth: 1.5, arrowcolor: '#c9943c',
                    ax: -65, ay: 0,
                    font: { color: '#c9943c', size: 11, family: 'JetBrains Mono, monospace' },
                    bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#c9943c', borderwidth: 1, borderpad: 3,
                });
            }

            const shapes = [];
            if (hasRef) {
                shapes.push({
                    type: 'line', xref: 'paper', x0: 0.05, x1: 0.95,
                    y0: refVal, y1: refVal, yref: 'y',
                    line: { color: '#cc3333', width: 2, dash: 'dash' },
                });
            }
            if (hasOrig) {
                shapes.push({
                    type: 'line', xref: 'paper', x0: 0.05, x1: 0.95,
                    y0: origVal, y1: origVal, yref: 'y',
                    line: { color: '#c9943c', width: 2, dash: 'dot' },
                });
            }

            Plotly.newPlot(plotId, traces, {
                yaxis: { gridcolor: 'rgba(120,94,1,0.1)', zeroline: false },
                xaxis: { title: { text: displayName, font: { size: 12 } }, showticklabels: multiRound, zeroline: false },
                plot_bgcolor: 'rgba(255,255,255,0.3)',
                paper_bgcolor: 'transparent',
                margin: { l: 55, r: 20, t: 20, b: 45 },
                font: { family: 'Inter, sans-serif', size: 11 },
                showlegend: multiRound,
                legend: multiRound ? { x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(255,255,255,0.7)' } : undefined,
                annotations: annotations,
                shapes: shapes,
            }, { responsive: true, displayModeBar: false });
        } else {
            // Histogram — overlay one per round
            const traces = [];
            sortedIters.forEach(iter => {
                const vals = roundValues.get(iter);
                if (!vals || vals.length === 0) return;
                const rc = _roundColor(iter);
                traces.push({
                    type: 'histogram',
                    x: vals,
                    name: `Round ${iter + 1}`,
                    marker: {
                        color: rc.fillAlpha,
                        line: { color: rc.line, width: 1 },
                    },
                    hoverinfo: 'x+y',
                    xbins: { size: 1 },
                    opacity: multiRound ? 0.7 : 1,
                });
            });

            const annotations = [];
            const shapes = [];
            if (hasRef) {
                shapes.push({
                    type: 'line', yref: 'paper', y0: 0, y1: 1,
                    x0: refVal, x1: refVal, xref: 'x',
                    line: { color: '#cc3333', width: 2.5, dash: 'dash' },
                });
                annotations.push({
                    x: refVal, y: 1, xref: 'x', yref: 'paper',
                    text: `Ref: ${refVal}`, showarrow: false,
                    font: { color: '#cc3333', size: 11, family: 'JetBrains Mono, monospace' },
                    bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#cc3333', borderwidth: 1, borderpad: 3, yanchor: 'bottom',
                });
            }
            if (hasOrig) {
                shapes.push({
                    type: 'line', yref: 'paper', y0: 0, y1: 1,
                    x0: origVal, x1: origVal, xref: 'x',
                    line: { color: '#c9943c', width: 2, dash: 'dot' },
                });
                annotations.push({
                    x: origVal, y: 0.9, xref: 'x', yref: 'paper',
                    text: `Orig: ${origVal}`, showarrow: false,
                    font: { color: '#c9943c', size: 11, family: 'JetBrains Mono, monospace' },
                    bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#c9943c', borderwidth: 1, borderpad: 3, yanchor: 'bottom',
                });
            }

            Plotly.newPlot(plotId, traces, {
                xaxis: { title: displayName, dtick: 1, gridcolor: 'rgba(120,94,1,0.1)' },
                yaxis: { title: 'Count', gridcolor: 'rgba(120,94,1,0.1)' },
                plot_bgcolor: 'rgba(255,255,255,0.3)',
                paper_bgcolor: 'transparent',
                margin: { l: 50, r: 20, t: 20, b: 45 },
                font: { family: 'Inter, sans-serif', size: 11 },
                showlegend: multiRound,
                legend: multiRound ? { x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(255,255,255,0.7)' } : undefined,
                barmode: multiRound ? 'overlay' : undefined,
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
        const resp = await authFetch(`${API_BASE}/affinity-distribution/${state.jobId}`);
        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(errText);
        }
        let data = await resp.json();
        loading.classList.add('hidden');
        data = _filterAffinityBySelection(data);
        if (data.affinity_types.length === 0) {
            container.innerHTML = '<div class="modal-error">No ligands selected. Use the checkboxes or Select All to choose ligands for visualization.</div>';
            return;
        }
        // Show only the selected affinity type
        const selectedType = document.getElementById('rank-affinity-type')?.value || 'pic50';
        if (data.affinity_types?.includes(selectedType)) {
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
                <span class="affinity-ref-icon"></span>
                <div class="affinity-ref-text">
                    <strong>Reference ligand:</strong> ${escapeHtml(refAff.p_label)} = ${refAff.p_value.toFixed(2)}
                    <span class="affinity-ref-raw">(from SDF tag <code>${escapeHtml(refAff.raw_tag)}</code>: ${escapeHtml(rawDesc)})</span>
                </div>
            </div>
        `;
    }

    // Create one plot per affinity type
    data.affinity_types.forEach((affType) => {
        const rounds = data.distributions[affType];
        if (!rounds || rounds.length === 0) return;

        const label = _AFFINITY_LABELS[affType] || affType;
        const plotId = `affinity-plot-${affType}`;

        const plotWrapper = document.createElement('div');
        plotWrapper.className = 'affinity-plot-wrapper';
        plotWrapper.innerHTML = `
            <h3 class="affinity-plot-title">${escapeHtml(label)} Distribution</h3>
            <div id="${plotId}" class="affinity-plot"></div>
        `;
        container.appendChild(plotWrapper);

        const traces = [];
        const annotations = [];
        const shapes = [];
        let allValues = [];

        // One violin trace per round
        rounds.forEach((round) => {
            const iter = round.iteration;
            const values = round.values;
            const rc = _roundColor(iter);
            const n = values.length;
            allValues = allValues.concat(values);

            traces.push({
                type: 'violin',
                y: values,
                x: values.map(() => `Round ${iter + 1}`),
                name: `Round ${iter + 1}`,
                legendgroup: `round${iter}`,
                box: { visible: true, fillcolor: 'rgba(255,255,255,0.5)', width: 0.12 },
                meanline: { visible: true, color: rc.line, width: 2 },
                fillcolor: rc.fillAlpha,
                line: { color: rc.line, width: 1.5 },
                points: n <= 80 ? 'all' : 'outliers',
                jitter: 0.45,
                pointpos: -1.8,
                marker: { color: rc.line, size: 5, opacity: 0.7 },
                hovertemplate: `${label}: %{y:.2f}<br>Round ${iter + 1}<extra>%{text}</extra>`,
                text: round.labels,
                customdata: round.indices.map(i => ({ iteration: iter, index: i })),
                spanmode: 'hard',
                bandwidth: n > 5 ? undefined : 0.3,
                scalemode: 'width',
                width: 0.7,
            });
        });

        // Reference line (if same affinity type matches)
        if (refAff) {
            const refKeyMap = { 'pIC50': 'pic50', 'pKi': 'pki', 'pKd': 'pkd', 'pEC50': 'pec50' };
            const mappedKey = refKeyMap[refAff.p_label];

            if (mappedKey === affType) {
                const refVal = refAff.p_value;
                shapes.push({
                    type: 'line',
                    xref: 'paper',
                    x0: 0,
                    x1: 0.95,
                    y0: refVal,
                    y1: refVal,
                    yref: 'y',
                    line: { color: '#cc3333', width: 2, dash: 'dash' },
                });
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

        // Overall stats annotation
        const nTotal = allValues.length;
        const mean = allValues.reduce((a, b) => a + b, 0) / nTotal;
        const std = Math.sqrt(allValues.reduce((a, b) => a + (b - mean) ** 2, 0) / nTotal);
        const sorted = [...allValues].sort((a, b) => a - b);
        const median = nTotal % 2 === 0
            ? (sorted[nTotal / 2 - 1] + sorted[nTotal / 2]) / 2
            : sorted[Math.floor(nTotal / 2)];

        annotations.push({
            x: 1,
            y: 0,
            xref: 'paper',
            yref: 'paper',
            text: [
                `n = ${nTotal}`,
                `mean = ${mean.toFixed(2)}`,
                `median = ${median.toFixed(2)}`,
                `std = ${std.toFixed(2)}`,
            ].join('<br>'),
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
                zeroline: false,
            },
            plot_bgcolor: 'rgba(255,255,255,0.3)',
            paper_bgcolor: 'transparent',
            margin: { l: 65, r: 85, t: 15, b: 40 },
            font: { family: 'Inter, sans-serif', size: 12 },
            showlegend: rounds.length > 1,
            legend: { orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' },
            annotations: annotations,
            shapes: shapes,
            height: 380,
            violinmode: 'group',
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d', 'zoom2d', 'pan2d'],
            displaylogo: false,
        };

        Plotly.purge(plotId);
        Plotly.newPlot(plotId, traces, layout, config);

        // Click → show ligand in sidebar (only works for current round)
        document.getElementById(plotId).on('plotly_click', (eventData) => {
            const cd = eventData.points[0]?.customdata;
            if (typeof cd?.index === 'number' && cd.iteration === state.iterationIdx - 1) {
                showGeneratedLigand(cd.index);
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

// Human-friendly display names for RDKit property keys
const _PROP_DISPLAY_NAMES = {
    MolWt: 'Mol. Weight',
    LogP: 'LogP',
    FractionCSP3: 'Fsp3',
    TPSA: 'TPSA',
    NumHAcceptors: 'H-Bond Acceptors',
    NumHDonors: 'H-Bond Donors',
    NumRotatableBonds: 'Rotatable Bonds',
    NumAromaticRings: 'Aromatic Rings',
    NumRings: 'Rings',
    NumHeavyAtoms: 'Heavy Atoms',
    NumAtoms: 'Atoms',
    NumChiralCenters: 'Chiral Centers',
    NumAlerts: 'PAINS Alerts',
    QED: 'QED',
    SA: 'SA Score',
    BertzCT: 'Bertz CT',
};

function _propDisplayName(name) {
    if (_PROP_DISPLAY_NAMES[name]) return _PROP_DISPLAY_NAMES[name];
    // Fallback: smart camelCase splitting that keeps consecutive caps together
    return name
        .replaceAll(/([a-z0-9])([A-Z])/g, '$1 $2')   // camelCase boundary
        .replaceAll(/([A-Z]+)([A-Z][a-z])/g, '$1 $2') // e.g. "TPSAValue" → "TPSA Value"
        .trim();
}

async function _loadLigandProperties() {
    const container = document.getElementById('mol-2d-properties');
    if (!container || !state.jobId) return;
    if (container.dataset.loaded === 'true') return;

    const ligIdx = state.activeResultIdx >= 0 ? state.activeResultIdx : -1;
    container.innerHTML = '<span class="muted-text">Loading properties…</span>';

    try {
        const resp = await authFetch(`${API_BASE}/ligand-properties/${state.jobId}/${ligIdx}`);
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
                const displayName = _propDisplayName(name);
                const val = typeof props[name] === 'number' ? props[name].toFixed(3) : props[name];
                html += `<tr><td class="prop-name">${escapeHtml(displayName)}</td><td class="prop-value">${val ?? '–'}</td></tr>`;
            }
        });

        // Structural Properties group
        html += '<tr class="prop-group-header"><td colspan="2">Structural Properties</td></tr>';
        structuralProps.forEach(name => {
            if (props[name] !== undefined) {
                const displayName = _propDisplayName(name);
                const val = props[name];
                html += `<tr><td class="prop-name">${escapeHtml(displayName)}</td><td class="prop-value">${val ?? '–'}</td></tr>`;
            }
        });

        // Alerts group
        if (alertProps.length > 0) {
            html += '<tr class="prop-group-header"><td colspan="2">Alerts</td></tr>';
            alertProps.forEach(name => {
                if (props[name] !== undefined) {
                    const displayName = _propDisplayName(name);
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
        console.debug('Property load failed:', e.message);
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
    // Click anywhere on the upload area opens the file dialog
    area.addEventListener('click', (e) => {
        if (e.target.tagName === 'A') return; // let browse link handle itself
        const fileInput = area.querySelector('input[type="file"]');
        if (fileInput && !fileInput.disabled) fileInput.click();
    });
});

// =========================================================================
// Reference Ligand Hydrogens (Add / Remove)
// =========================================================================

async function removeRefHydrogens() {
    if (!state.jobId) return;
    try {
        const resp = await authFetch(`${API_BASE}/ligand-remove-hs/${state.jobId}`, { method: 'POST' });
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
        const resp = await authFetch(`${API_BASE}/ligand-add-hs/${state.jobId}`, { method: 'POST' });
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

    // ── Collapsible round header ──
    const currentIter = Math.max(0, state.iterationIdx - 1);
    const rc = _roundColor(currentIter);
    const nLigs = state.generatedResults.length;

    const section = document.createElement('div');
    section.className = 'current-round-section';
    const hdr = document.createElement('div');
    hdr.className = 'round-header round-header-collapsible';
    hdr.innerHTML = `<span class="round-dot" style="background:${rc.fill};border-color:${rc.line}"></span>` +
        `<span class="round-label">Round ${currentIter + 1}</span>` +
        `<span class="round-count">${nLigs} ligand${nLigs === 1 ? '' : 's'}</span>` +
        `<span class="round-chevron">&#x25B8;</span>`;
    const body = document.createElement('div');
    body.className = 'current-round-body';
    hdr.onclick = () => {
        const isCollapsed = hdr.classList.toggle('collapsed');
        body.classList.toggle('hidden', isCollapsed);
    };
    section.appendChild(hdr);

    state.generatedResults.forEach((result, i) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.onclick = (e) => {
            if (e.target.classList.contains('result-card-checkbox')) return;
            if (e.target.closest('.set-ref-btn')) return;
            showGeneratedLigand(i);
        };

        const mw = result.properties?.mol_weight || '–';
        const tpsa = result.properties?.tpsa ?? result.properties?.TPSA ?? '–';
        const logp = result.properties?.logp ?? result.properties?.LogP ?? '–';
        const ha = result.properties?.num_heavy_atoms || '–';

        card.innerHTML = `
            <input type="checkbox" class="result-card-checkbox save-ligand-cb" data-idx="${i}"
                   onchange="event.stopPropagation(); updateSaveSelectedCount()">
            <div class="result-card-content">
                <div class="result-header">
                    <span class="result-title">Ligand #${i + 1}</span>
                    <span class="result-badge badge badge-idle">${escapeHtml(String(ha))} HA</span>
                </div>
                <div class="result-smiles">${escapeHtml(smilesWithoutH(result.smiles) || 'N/A')}</div>
                <div class="result-props">
                    <span>MW: ${escapeHtml(String(mw))}</span>
                    <span>TPSA: ${escapeHtml(String(tpsa))}</span>
                    <span>logP: ${escapeHtml(String(logp))}</span>
                </div>
                <button class="set-ref-btn" onclick="event.stopPropagation(); setAsReference(${i})" title="Use this ligand as the new reference">Set as Reference</button>
            </div>
        `;
        body.appendChild(card);
    });

    section.appendChild(body);
    list.appendChild(section);
    updateSaveSelectedCount();

    if (state.generatedResults.length > 0) showGeneratedLigand(0);
}

async function applyRankSelect() {
    if (!state.jobId || state.generatedResults.length === 0) return;

    const affinityType = document.getElementById('rank-affinity-type').value;
    const topNInput = document.getElementById('rank-top-n').value;
    const topN = topNInput ? Number.parseInt(topNInput) : null;

    try {
        const resp = await authFetch(`${API_BASE}/rank-select/${state.jobId}`, {
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
        const resp = await authFetch(`${API_BASE}/reset-rank/${state.jobId}`, {
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
