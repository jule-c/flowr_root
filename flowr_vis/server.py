"""
FLOWR Visualization Server (CPU-only Frontend Server)
=====================================================
Lightweight FastAPI backend for the FLOWR ligand generation web interface.
Handles file uploads, preprocessing, atom selection, molecule visualization,
and interaction computation — all on CPU with NO PyTorch dependency.

GPU-intensive generation is delegated to a separate **worker** service
(``worker.py``) which can run on an on-demand GPU instance (e.g. AWS).
When the user clicks "Generate", this server forwards the request to the
worker and proxies progress / results back to the browser.

The worker URL is configured via the ``FLOWR_WORKER_URL`` environment
variable (default: ``http://localhost:8788``).
"""

import asyncio
import atexit
import hmac
import json as json_module
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections import deque, namedtuple
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests as http_requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# RDKit imports
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available – molecule processing will be limited.")

# ---------------------------------------------------------------------------
# Shared chemistry utilities (also used by worker.py)
# ---------------------------------------------------------------------------
from chem_utils import (
    ALL_PROPERTY_NAMES,
    CONT_PROPERTIES_RDKIT,
    DISC_PROPERTIES_RDKIT,
)
from chem_utils import (
    compute_all_properties as _compute_all_properties,
)
from chem_utils import (
    compute_pocket_com as _compute_pocket_com_shared,
)
from chem_utils import (
    crawl_sdf_affinity as _crawl_sdf_affinity,
)
from chem_utils import (
    mol_to_atom_info as _mol_to_atom_info,
)
from chem_utils import (
    mol_to_bond_info as _mol_to_bond_info,
)
from chem_utils import (
    mol_to_morgan_fp as _mol_to_morgan_fp,
)
from chem_utils import (
    mol_to_sdf_string as _mol_to_sdf_string,
)
from chem_utils import (
    mol_to_smiles as _mol_to_smiles,
)
from chem_utils import (
    read_ligand_mol as _read_ligand_mol,
)
from chem_utils import (
    validate_molecule as _validate_molecule,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenEye license setup (must happen BEFORE importing openeye)
# ---------------------------------------------------------------------------
_OE_LICENSE_FILENAME = "oe_license.txt"
if "OE_LICENSE" not in os.environ:
    _script_dir = Path(__file__).resolve().parent
    for _candidate in [
        _script_dir / "tools" / _OE_LICENSE_FILENAME,
        _script_dir / _OE_LICENSE_FILENAME,
        _script_dir.parent / _OE_LICENSE_FILENAME,
    ]:
        if _candidate.is_file():
            os.environ["OE_LICENSE"] = str(_candidate)
            break

# ---------------------------------------------------------------------------
# OpenEye imports (optional – for 2D interaction diagrams)
# ---------------------------------------------------------------------------
OPENEYE_AVAILABLE = False
try:
    from openeye import oechem, oedepict, oegrapheme, oeomega  # noqa: E402

    # Verify that a valid license is present (import can succeed without one)
    if oechem.OEChemIsLicensed():
        OPENEYE_AVAILABLE = True
    else:
        logging.warning(
            "OpenEye is installed but no valid license was found. "
            "Set the OE_LICENSE environment variable to enable "
            "2D interaction diagrams."
        )
except ImportError:
    logging.warning("OpenEye not available – 2D interaction diagrams disabled.")

# ---------------------------------------------------------------------------
# SciPy (optional – for fast pairwise distances)
# ---------------------------------------------------------------------------
try:
    from scipy.spatial.distance import cdist  # noqa: E402

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# scikit-learn / UMAP (optional – for chemical space projection)
# ---------------------------------------------------------------------------
try:
    from sklearn.decomposition import PCA  # noqa: E402
    from sklearn.manifold import TSNE  # noqa: E402

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap  # noqa: E402

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Worker URL & mode
# ---------------------------------------------------------------------------
WORKER_URL = os.environ.get("FLOWR_WORKER_URL", "http://localhost:8788")

# Worker mode: "static" (always-on worker) or "slurm" (submit sbatch on demand)
WORKER_MODE = os.environ.get("FLOWR_WORKER_MODE", "static")

# SLURM configuration (only used when WORKER_MODE == "slurm")
SLURM_SCRIPT = os.environ.get(
    "FLOWR_SLURM_WORKER_SCRIPT",
    str(Path(__file__).parent / "hpc" / "worker_hpc.sh"),
)
SLURM_STARTUP_TIMEOUT = int(os.environ.get("FLOWR_SLURM_STARTUP_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Checkpoint path resolution (for listing available checkpoints)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CKPTS_DIR = Path(os.environ.get("FLOWR_CKPTS_DIR", str(ROOT_DIR / "ckpts")))

# ── Shared error/string constants (SonarQube: python:S1192) ──
_ERR_FILE_TOO_LARGE = "File too large"
_ERR_INVALID_FILE_PATH = "Invalid file path"
_ERR_CANCELLED = "Cancelled by user"
_ERR_NO_PROTEIN = "No protein uploaded."
_ERR_NO_LIGAND = "No ligand uploaded."
_ERR_CANNOT_PARSE_LIGAND = "Cannot parse ligand."
_ERR_LIGAND_IDX_RANGE = "Ligand index out of range."
_ERR_RDKIT_UNAVAILABLE = "RDKit not available."
_ERR_NO_GENERATED = "No generated ligands available."
_REF_LIGAND_FILENAME = "reference_gen.sdf"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(600)
            _cleanup_expired_jobs()

    task = asyncio.create_task(_periodic_cleanup())
    yield
    task.cancel()


app = FastAPI(title="FLOWR Visualization", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FLOWR_CORS_ORIGINS", "http://localhost:8787").split(
        ","
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Mol* requires 'unsafe-eval' (uses new Function() for WASM/Emscripten
        # bindings and template interpolation). Only relax CSP for the embed page
        # and its assets; keep the main app locked down.
        if request.url.path.startswith(
            "/static/lib/molstar/"
        ) or request.url.path.endswith("molstar-embed.html"):
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "connect-src 'self'; "
                "frame-ancestors 'self'"
            )
        else:
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "frame-src 'self'; "
                "frame-ancestors 'self'"
            )
        response.headers["Content-Security-Policy"] = csp
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="flowr_vis_"))


def _cleanup_upload_dir():
    """Remove the upload directory on graceful shutdown."""
    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            logger.info("Cleaned up upload directory: %s", UPLOAD_DIR)
    except Exception:
        logger.debug("Failed to clean up upload directory", exc_info=True)


atexit.register(_cleanup_upload_dir)

JOBS: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# ── Per-user settings (checkpoint, workflow type) ──
_user_settings: Dict[str, Dict[str, Any]] = {}
_user_settings_lock = threading.Lock()


def _get_user_setting(user: str, key: str, default=None):
    with _user_settings_lock:
        return _user_settings.get(user, {}).get(key, default)


def _set_user_setting(user: str, key: str, value):
    with _user_settings_lock:
        _user_settings.setdefault(user, {})[key] = value


def _get_user_job(job_id: str, request: Request) -> dict:
    """Look up a job and verify the requesting user owns it."""
    with _jobs_lock:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("user") != request.state.user:
        raise HTTPException(403, "Access denied")
    return job


def _job_upload_dir(user: str, job_id: str) -> Path:
    """Per-user, per-job upload directory."""
    return UPLOAD_DIR / user / job_id


# ── Input validation helpers ──
_JOB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_job_id(job_id: str) -> str:
    if not job_id or len(job_id) > 128 or not _JOB_ID_PATTERN.match(job_id):
        raise HTTPException(400, "Invalid job ID format")
    return job_id


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = name.replace("\0", "").replace("..", "").strip()
    if not name:
        raise HTTPException(400, "Invalid filename")
    return name


_FLOAT_RE = re.compile(r"-?\d+\.?\d*")


# ── Rate limiting ──
_rate_limits: Dict[str, deque] = {}
_rate_limits_lock = threading.Lock()


def _check_rate_limit(user: str, action: str, max_per_minute: int = 10):
    key = f"{user}:{action}"
    now = time.time()
    with _rate_limits_lock:
        if key not in _rate_limits:
            _rate_limits[key] = deque(maxlen=max_per_minute)
        dq = _rate_limits[key]
        if len(dq) >= max_per_minute and now - dq[0] < 60:
            raise HTTPException(429, "Rate limit exceeded. Please wait.")
        dq.append(now)


def _worker_file_url(job_id: str, filename: str) -> str:
    """Build a authenticated file URL that the worker can download."""
    server_base = os.environ.get("FLOWR_SERVER_URL", "http://localhost:8787")
    base = f"{server_base}/files"
    return f"{base}/{job_id}/{filename}?token={_WORKER_FILE_TOKEN}"


# ── Visualization cache invalidation helper ──
_VIZ_CACHE_KEYS = frozenset({"propspace_cache", "affinity_dist_cache"})


def _invalidate_viz_caches(job: dict):
    """Remove cached visualization data from a job dict."""
    to_del = [k for k in job if k.startswith("chemspace_") or k in _VIZ_CACHE_KEYS]
    for k in to_del:
        del job[k]


# ── Job TTL cleanup ──
_JOB_TTL_SECONDS = 3600  # 1 hour


def _cleanup_expired_jobs():
    """Remove jobs older than _JOB_TTL_SECONDS to prevent unbounded memory growth."""
    now = time.time()
    dirs_to_delete = []
    with _jobs_lock:
        expired = [
            jid
            for jid, jdata in JOBS.items()
            if now - jdata.get("created_at", now) > _JOB_TTL_SECONDS
            and jdata.get("status")
            not in (
                "generating",
                "allocating_gpu",
                "starting",
                "loading_model",
                "queued",
                "finetuning",
            )
        ]
        for jid in expired:
            job_data = JOBS.pop(jid, {})
            user = job_data.get("user", "_unknown")
            dirs_to_delete.append(_job_upload_dir(user, jid))
    # I/O outside the lock
    for d in dirs_to_delete:
        try:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            logger.debug("Failed to remove stale job directory %s", d, exc_info=True)
    # Cleanup stale rate limit keys
    now_rl = time.time()
    with _rate_limits_lock:
        stale = [
            k for k, dq in _rate_limits.items() if not dq or now_rl - dq[-1] > 3600
        ]
        for k in stale:
            del _rate_limits[k]


# ── Dynamic worker state (for SLURM mode) ──
_worker_state = {
    "status": "idle",  # idle | starting | running | stopping
    "slurm_job_id": None,
    "node": None,
    "url": None,  # resolved URL once worker is up
    "error": None,
    "active_jobs": 0,  # Reference count: number of in-flight generation jobs
}
_worker_lock = threading.Lock()


# Shared worker-to-server file token (for /files/ endpoint)
_WORKER_FILE_TOKEN = os.environ.get("FLOWR_WORKER_FILE_TOKEN", secrets.token_hex(32))


@app.get("/api/jobs")
async def list_jobs(request: Request):
    """Return only the current user's jobs."""
    user = request.state.user
    with _jobs_lock:
        user_jobs = {
            jid: {k: v for k, v in j.items() if k != "user"}
            for jid, j in JOBS.items()
            if j.get("user") == user
        }
    return user_jobs


# ── Session restore ──

_MAX_SESSION_BODY = 50 * 1024 * 1024  # 50 MB


@app.post("/api/session/restore")
async def restore_session(request: Request):
    """Recreate server-side job state from a saved session JSON."""
    user = request.state.user

    # ── Size guard ──
    cl = request.headers.get("content-length")
    if cl and int(cl) > _MAX_SESSION_BODY:
        raise HTTPException(413, _ERR_FILE_TOO_LARGE)

    body = await request.body()
    if len(body) > _MAX_SESSION_BODY:
        raise HTTPException(413, _ERR_FILE_TOO_LARGE)

    try:
        session = json_module.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    if not isinstance(session, dict) or session.get("version") != 1:
        raise HTTPException(400, "Unsupported session version")

    # ── Per-field size validation ──
    protein = session.get("protein")
    ligand = session.get("ligand")
    if protein and isinstance(protein.get("pdbData"), str):
        if len(protein["pdbData"]) > _MAX_SESSION_BODY:
            raise HTTPException(413, "pdbData too large")
    if ligand and isinstance(ligand.get("sdfData"), str):
        if len(ligand["sdfData"]) > _MAX_SESSION_BODY:
            raise HTTPException(413, "sdfData too large")

    # ── New job ──
    job_id = str(uuid.uuid4())[:8]
    job_dir = _job_upload_dir(user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    protein_path = None
    protein_filename = None
    ligand_path = None
    ligand_filename = None

    # ── Write protein PDB ──
    if protein and protein.get("pdbData"):
        raw_name = protein.get("filename", "protein.pdb")
        safe_name = _sanitize_filename(raw_name)
        ext = Path(safe_name).suffix.lower()
        if ext not in (".pdb", ".cif", ".mmcif"):
            safe_name = safe_name + ".pdb"
        p = job_dir / safe_name
        p.write_text(protein["pdbData"], encoding="utf-8")
        resolved = p.resolve()
        if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
            p.unlink(missing_ok=True)
            raise HTTPException(400, _ERR_INVALID_FILE_PATH)
        protein_path = str(p)
        protein_filename = safe_name

    # ── Write ligand SDF ──
    if ligand and ligand.get("sdfData"):
        raw_name = ligand.get("filename", "ligand.sdf")
        safe_name = _sanitize_filename(raw_name)
        ext = Path(safe_name).suffix.lower()
        if ext not in (".sdf", ".mol", ".mol2", ".pdb"):
            safe_name = safe_name + ".sdf"
        p = job_dir / safe_name
        p.write_text(ligand["sdfData"], encoding="utf-8")
        resolved = p.resolve()
        if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
            p.unlink(missing_ok=True)
            raise HTTPException(400, _ERR_INVALID_FILE_PATH)
        ligand_path = str(p)
        ligand_filename = safe_name

    # ── Checkpoint / workflow from session ──
    workflow = session.get("workflow", {})
    ckpt_path = workflow.get("ckptPath")
    base_ckpt_path = workflow.get("baseCkptPath")
    workflow_type = workflow.get("type", "sbdd")

    # Validate ckpt_path — fall back to baseCkptPath if file doesn't exist
    if ckpt_path and not Path(ckpt_path).is_file():
        logger.warning(
            "[restore] ckpt_path does not exist: %s, falling back to baseCkptPath",
            ckpt_path,
        )
        ckpt_path = base_ckpt_path
    if ckpt_path and not Path(ckpt_path).is_file():
        logger.warning(
            "[restore] baseCkptPath also does not exist: %s, clearing ckpt_path",
            ckpt_path,
        )
        ckpt_path = None

    if ckpt_path:
        _set_user_setting(user, "ckpt_path", ckpt_path)
    if workflow_type:
        _set_user_setting(user, "workflow_type", workflow_type)

    # ── Restore generation state ──
    gen = session.get("generation", {})
    all_results = gen.get("allGeneratedResults", [])

    # Populate "results" from latest round so endpoints like
    # interaction-diagram, save-ligand, etc. work immediately.
    latest_results = []
    if all_results:
        last_round = all_results[-1]
        if isinstance(last_round, dict):
            latest_results = last_round.get("results", [])

    # ── Reconstruct generated_history for diversity dedup ──
    generated_history = []
    for round_data in all_results:
        if isinstance(round_data, dict):
            for r in round_data.get("results", []):
                smi = r.get("smiles") if isinstance(r, dict) else None
                if smi:
                    generated_history.append(smi)

    # ── Extract ref_affinity from restored ligand SDF ──
    ref_affinity = None
    if ligand_path:
        try:
            ref_affinity = _crawl_sdf_affinity(ligand_path)
        except Exception:
            pass

    # ── Restore original ligand state (for undo-reference, chemical-space) ──
    orig_lig = session.get("originalLigand")
    original_ligand_keys = {}
    if orig_lig and isinstance(orig_lig, dict) and orig_lig.get("sdf_data"):
        orig_sdf_path = job_dir / "original_ligand.sdf"
        orig_sdf_path.write_text(orig_lig["sdf_data"], encoding="utf-8")
        original_ligand_keys = {
            "original_ligand_path": str(orig_sdf_path),
            "original_ligand_smiles": orig_lig.get("smiles", ""),
            "original_ligand_smiles_noH": orig_lig.get("smiles_noH", ""),
            "original_ligand_sdf": orig_lig.get("sdf_data", ""),
            "original_ligand_properties": orig_lig.get("properties"),
            "original_ligand_num_atoms": orig_lig.get("num_atoms", 0),
            "original_ligand_num_heavy_atoms": orig_lig.get("num_heavy_atoms", 0),
            "original_ligand_ref_affinity": orig_lig.get("ref_affinity"),
            "original_ligand_filename": orig_lig.get("filename", ""),
        }

    # ── Settings for pocket-only mode ──
    settings = session.get("settings", {})

    job = {
        "job_id": job_id,
        "user": user,
        "protein_path": protein_path,
        "protein_filename": protein_filename,
        "ligand_path": ligand_path,
        "ligand_filename": ligand_filename,
        "status": "restored",
        "created_at": time.time(),
        "ckpt_path": ckpt_path,
        "workflow_type": workflow_type,
        "iteration_idx": gen.get("iterationIdx", 0),
        "all_results": all_results,
        "results": latest_results,
        "generated_history": generated_history,
        "ref_affinity": ref_affinity,
        "denovo_no_ligand": not ligand_path
        and (
            settings.get("pocketOnlyMode", False)
            or settings.get("lbddScratchMode", False)
        ),
        **original_ligand_keys,
    }

    # ── Active learning finetuned checkpoint ──
    al = session.get("activeLearning", {})
    al_ckpt = al.get("ckptPath")
    al_valid = False
    if al_ckpt and Path(al_ckpt).is_file():
        job["finetuned_ckpt_path"] = al_ckpt
        al_valid = True
    elif al_ckpt:
        logger.warning("[restore] AL ckpt does not exist, skipping: %s", al_ckpt)

    with _jobs_lock:
        JOBS[job_id] = job

    return {"job_id": job_id, "status": "restored", "al_valid": al_valid}


class AnonymousUserMiddleware(BaseHTTPMiddleware):
    """Set request.state.user for all requests (no authentication)."""

    async def dispatch(self, request: Request, call_next):
        request.state.user = "anonymous"
        return await call_next(request)


app.add_middleware(AnonymousUserMiddleware)


logger.info("Upload directory:  %s", UPLOAD_DIR)
logger.info("Checkpoints dir:   %s  (exists=%s)", CKPTS_DIR, CKPTS_DIR.is_dir())
logger.info("RDKit available:   %s", RDKIT_AVAILABLE)
logger.info("OpenEye avail.:    %s", OPENEYE_AVAILABLE)
logger.info("Worker mode:       %s", WORKER_MODE)
if WORKER_MODE == "slurm":
    logger.info("SLURM script:      %s", SLURM_SCRIPT)
    logger.info("Startup timeout:   %ss", SLURM_STARTUP_TIMEOUT)
else:
    logger.info("Worker URL:        %s", WORKER_URL)


# ═══════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════
_VALID_GEN_MODES = {
    "denovo",
    "substructure_inpainting",
    "scaffold_hopping",
    "scaffold_elaboration",
    "linker_inpainting",
    "core_growing",
    "fragment_growing",
}


class GenerationRequest(BaseModel):
    job_id: str
    workflow_type: Optional[str] = None  # request override; falls back to job/global
    ckpt_path: Optional[str] = None  # request override; falls back to job/global
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    gen_mode: str = "denovo"
    fixed_atoms: List[int] = []
    n_samples: int = Field(default=100, ge=1, le=500)
    batch_size: int = Field(default=25, ge=1, le=200)
    integration_steps: int = Field(default=100, ge=1, le=1000)
    pocket_cutoff: float = 6.0
    grow_size: Optional[int] = None
    prior_center_filename: Optional[str] = None
    prior_center_coords: Optional[Dict[str, float]] = (
        None  # {x, y, z} from visual placement
    )
    coord_noise_scale: float = 0.0
    filter_valid_unique: bool = True
    filter_cond_substructure: bool = False
    filter_diversity: bool = True
    diversity_threshold: float = 0.8
    sample_mol_sizes: bool = False
    filter_pb_valid: bool = False
    calculate_pb_valid: bool = False
    calculate_strain_energies: bool = False
    optimize_gen_ligs: bool = False
    optimize_gen_ligs_hs: bool = False
    anisotropic_prior: bool = False
    ring_system_index: int = 0
    ref_ligand_com_prior: bool = False
    # Random seed for reproducibility
    seed: int = Field(default=42, ge=0, le=999999)
    # De novo: number of heavy atoms when no reference ligand
    num_heavy_atoms: Optional[int] = None
    # Property / ADMET filtering
    property_filter: Optional[List[dict]] = None  # [{"name":..., "min":..., "max":...}]
    adme_filter: Optional[List[dict]] = (
        None  # [{"name":..., "min":..., "max":..., "model_file":...}]
    )
    # LBDD-specific
    optimize_method: str = "none"  # "none" | "rdkit" | "xtb"
    sample_n_molecules_per_mol: int = Field(default=1, ge=1, le=50)


class ActiveLearningRequest(BaseModel):
    """Request model for starting active learning LoRA finetuning."""

    job_id: str
    indices: List[int]  # Indices of selected ligands from generation results
    prev_round_iterations: Optional[List[int]] = (
        None  # Include ligands from previous rounds
    )
    lora_rank: int = Field(default=16, ge=4, le=64)
    lora_alpha: int = Field(default=32, ge=8, le=128)
    lr: float = Field(default=5e-4, gt=0)
    batch_cost: int = Field(default=4, ge=1, le=32)
    acc_batches: Optional[int] = Field(
        default=None, ge=1, le=32
    )  # None = dynamic based on n_ligands
    epochs: Optional[int] = (
        None  # If None, epochs are computed dynamically based on n_ligands
    )
    ckpt_path: Optional[str] = None  # User-selected base checkpoint for finetuning


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#  INPAINTING MASK COMPUTATION (pure RDKit + numpy, NO PyTorch)
# ═══════════════════════════════════════════════════════════════════════════
#
#  These are lightweight reimplementations of the extract_* functions from
#  flowr.data.interpolate, using numpy boolean arrays instead of
#  torch.Tensor. They only need RDKit's MurckoScaffold analysis and ring
#  info — no GPU, no model, no torch import.
# ═══════════════════════════════════════════════════════════════════════════


def _get_ring_systems(mol) -> List[set]:
    """Get separate ring systems (groups of fused/shared rings)."""
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if not atom_rings:
        return []
    ring_systems = [set(ring) for ring in atom_rings]
    merged = True
    while merged:
        merged = False
        for i in range(len(ring_systems)):
            for j in range(i + 1, len(ring_systems)):
                if ring_systems[i] & ring_systems[j]:
                    ring_systems[i] = ring_systems[i] | ring_systems[j]
                    ring_systems.pop(j)
                    merged = True
                    break
            if merged:
                break
    return ring_systems


if RDKIT_AVAILABLE:
    _PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts("A=,#[!#6]")
    _PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts("C=,#C")
    _PATT_ACETAL = Chem.MolFromSmarts("[CX4](-[O,N,S])-[O,N,S]")
    _PATT_OXIRANE_ETC = Chem.MolFromSmarts("[O,N,S]1CC1")
else:
    _PATT_DOUBLE_TRIPLE = _PATT_CC_DOUBLE_TRIPLE = _PATT_ACETAL = _PATT_OXIRANE_ETC = (
        None
    )


def _identify_functional_groups(mol):  # NOSONAR
    """Identify functional groups (Ertl IFG algorithm). Pure RDKit."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    marked = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 1):
            marked.add(atom.GetIdx())

    for patt in (
        _PATT_DOUBLE_TRIPLE,
        _PATT_CC_DOUBLE_TRIPLE,
        _PATT_ACETAL,
        _PATT_OXIRANE_ETC,
    ):
        for path in mol.GetSubstructMatches(patt):
            for idx in path:
                marked.add(idx)

    def _merge(mol_, marked_, aset):
        bset = set()
        for idx in aset:
            atom = mol_.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                jdx = nbr.GetIdx()
                if jdx in marked_:
                    marked_.remove(jdx)
                    bset.add(jdx)
        if not bset:
            return
        _merge(mol_, marked_, bset)
        aset.update(bset)

    groups = []
    while marked:
        grp = {marked.pop()}
        _merge(mol, marked, grp)
        groups.append(grp)

    IFG = namedtuple("IFG", ["atomIds", "atoms", "type"])
    ifgs = []
    for g in groups:
        uca = set()
        for aidx in g:
            for n in mol.GetAtomWithIdx(aidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        ifgs.append(
            IFG(
                atomIds=tuple(g),
                atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
                type=Chem.MolFragmentToSmiles(mol, g.union(uca), canonical=True),
            )
        )
    return ifgs


def _extract_scaffold_mask(mol) -> np.ndarray:
    """Boolean mask: True = scaffold atom. Uses Murcko scaffolds."""
    n = mol.GetNumAtoms()
    mask = np.zeros(n, dtype=bool)
    _mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(_mol)
    except Exception:
        return mask
    for a in _mol.GetAtoms():
        a.SetIntProp("org_idx", a.GetIdx())
    scaffold = GetScaffoldForMol(_mol)
    if scaffold is None:
        return mask
    for a in scaffold.GetAtoms():
        mask[a.GetIntProp("org_idx")] = True
    return mask


def _extract_core_mask(mol, ring_system_index: int = 0) -> np.ndarray:
    """Boolean mask: True = core (ring-system within scaffold) atom."""
    n = mol.GetNumAtoms()
    mask = np.zeros(n, dtype=bool)
    _mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(_mol)
    except Exception:
        return mask
    for a in _mol.GetAtoms():
        a.SetIntProp("org_idx", a.GetIdx())
    scaffold = GetScaffoldForMol(_mol)
    if scaffold is None:
        return mask
    scaffold_atoms = {a.GetIntProp("org_idx") for a in scaffold.GetAtoms()}
    ring_systems = _get_ring_systems(_mol)
    if not ring_systems:
        return mask
    if ring_system_index < 0 or ring_system_index >= len(ring_systems):
        ring_system_index = 0
    ring_atoms = ring_systems[ring_system_index]
    core_atoms = [a for a in scaffold_atoms if a in ring_atoms]
    for idx in core_atoms:
        mask[idx] = True
    return mask


def _get_num_ring_systems(mol) -> int:
    """Return the number of separate ring systems in the molecule."""
    _mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(_mol)
    except Exception:
        return 0
    return len(_get_ring_systems(_mol))


def _extract_linker_mask(mol) -> np.ndarray:
    """Boolean mask: True = linker atom (scaffold atoms NOT in any ring)."""
    n = mol.GetNumAtoms()
    mask = np.zeros(n, dtype=bool)
    _mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(_mol)
    except Exception:
        return mask
    for a in _mol.GetAtoms():
        a.SetIntProp("org_idx", a.GetIdx())
    scaffold = GetScaffoldForMol(_mol)
    if scaffold is None:
        return mask
    ring_atoms = set()
    for ring in _mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    linker_atoms = [
        a.GetIntProp("org_idx")
        for a in scaffold.GetAtoms()
        if a.GetIntProp("org_idx") not in ring_atoms
    ]
    for idx in linker_atoms:
        mask[idx] = True
    return mask


def _extract_scaffold_elaboration_mask(mol) -> np.ndarray:  # NOSONAR
    """Boolean mask: True = atom to be REPLACED during scaffold elaboration.

    Replaced = (non-scaffold OR functional-group) AND NOT in-ring.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    n = mol.GetNumAtoms()
    _mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(_mol)
    except Exception:
        return np.zeros(n, dtype=bool)

    for a in _mol.GetAtoms():
        a.SetIntProp("org_idx", a.GetIdx())
    scaffold = GetScaffoldForMol(_mol)
    scaffold_set = set()
    if scaffold is not None:
        scaffold_set = {a.GetIntProp("org_idx") for a in scaffold.GetAtoms()}

    elaboration_set = set()
    fgroups = _identify_functional_groups(_mol)
    for fg in fgroups:
        for idx in fg.atomIds:
            elaboration_set.add(idx)
            for nb in _mol.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetSymbol() == "H":
                    elaboration_set.add(nb.GetIdx())

    ring_set: set = set()
    for ring in _mol.GetRingInfo().AtomRings():
        ring_set.update(ring)

    mask = np.zeros(n, dtype=bool)
    for idx in range(n):
        in_scaffold = idx in scaffold_set
        in_elaboration = idx in elaboration_set
        in_ring = idx in ring_set
        if (not in_scaffold or in_elaboration) and not in_ring:
            mask[idx] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════
#  INTERACTION COMPUTATION HELPERS (CPU, pure Python/numpy)
# ═══════════════════════════════════════════════════════════════════════════


def _parse_pdb_atoms(pdb_path: str) -> List[Dict[str, Any]]:
    """Parse PDB ATOM records to extract protein atoms with coordinates."""
    atoms = []
    with open(pdb_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                atoms.append(
                    {
                        "atom_name": line[12:16].strip(),
                        "res_name": line[17:20].strip(),
                        "chain": line[21:22].strip(),
                        "res_seq": line[22:26].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "element": (
                            line[76:78].strip()
                            if len(line) > 77 and line[76:78].strip()
                            else line[12:14].strip()[0]
                        ),
                    }
                )
            except (ValueError, IndexError):
                continue
    return atoms


def _compute_interactions(  # NOSONAR
    pdb_path: str, ligand_mol, cutoff: float = 4.0
) -> List[Dict[str, Any]]:
    """Protein-ligand interaction detection using vectorized numpy distance computation."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    prot_atoms = _parse_pdb_atoms(pdb_path)
    if not prot_atoms or ligand_mol is None:
        return []

    if ligand_mol.GetNumConformers() == 0:
        return []
    conf = ligand_mol.GetConformer()
    lig_atoms = []
    for atom in ligand_mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lig_atoms.append(
            {
                "idx": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
            }
        )

    if not lig_atoms:
        return []

    # Build coordinate arrays for vectorized distance computation
    prot_coords = np.array([[a["x"], a["y"], a["z"]] for a in prot_atoms])
    lig_coords = np.array([[a["x"], a["y"], a["z"]] for a in lig_atoms])
    lig_center = lig_coords.mean(axis=0)

    # Coarse filter: only keep protein atoms within (cutoff + 10) of ligand center
    max_r = cutoff + 10.0
    coarse_dists = np.linalg.norm(prot_coords - lig_center, axis=1)
    near_mask = coarse_dists <= max_r
    near_indices = np.nonzero(near_mask)[0]

    if len(near_indices) == 0:
        return []

    # Vectorized pairwise distance: (N_near, M_lig)
    near_coords = prot_coords[near_indices]
    if SCIPY_AVAILABLE:
        dist_matrix = cdist(near_coords, lig_coords)
    else:
        diff = near_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    # Find all pairs within cutoff
    hbond_elements = {"N", "O", "S"}
    cationic_residues = {"ARG", "LYS", "HIS"}
    anionic_residues = {"ASP", "GLU"}

    interactions: List[Dict[str, Any]] = []
    seen: set = set()

    pi_indices, li_indices = np.nonzero(dist_matrix <= cutoff)

    for k in range(len(pi_indices)):
        pa_idx = int(near_indices[pi_indices[k]])
        la_local = int(li_indices[k])
        dist = float(dist_matrix[pi_indices[k], la_local])

        pa = prot_atoms[pa_idx]
        la = lig_atoms[la_local]
        p_elem = pa["element"]
        l_elem = la["symbol"]
        int_type = None

        if dist < 3.5 and p_elem in hbond_elements and l_elem in hbond_elements:
            int_type = "HBond"
        elif (
            dist < 4.0
            and pa["res_name"] in cationic_residues
            and p_elem == "N"
            and l_elem in {"O", "S"}
        ):
            int_type = "SaltBridge"
        elif (
            dist < 4.0
            and pa["res_name"] in anionic_residues
            and p_elem == "O"
            and l_elem == "N"
        ):
            int_type = "SaltBridge"

        if int_type:
            key = (pa["res_name"], pa["res_seq"], pa["chain"], int_type, la["idx"])
            if key not in seen:
                seen.add(key)
                interactions.append(
                    {
                        "type": int_type,
                        "distance": round(dist, 2),
                        "protein": {
                            "atom_name": pa["atom_name"],
                            "res_name": pa["res_name"],
                            "chain": pa["chain"],
                            "res_seq": pa["res_seq"],
                            "x": round(pa["x"], 3),
                            "y": round(pa["y"], 3),
                            "z": round(pa["z"], 3),
                        },
                        "ligand": {
                            "idx": la["idx"],
                            "symbol": la["symbol"],
                            "x": round(la["x"], 3),
                            "y": round(la["y"], 3),
                            "z": round(la["z"], 3),
                        },
                    }
                )

    # Keep best interaction per (residue, ligand_atom) pair
    priority = {"HBond": 0, "SaltBridge": 1}
    best: Dict[tuple, dict] = {}
    for ix in interactions:
        k = (
            ix["protein"]["res_name"],
            ix["protein"]["res_seq"],
            ix["protein"]["chain"],
            ix["ligand"]["idx"],
        )
        if k not in best or priority.get(ix["type"], 99) < priority.get(
            best[k]["type"], 99
        ):
            best[k] = ix
    return list(best.values())


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER LIFECYCLE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════


def _get_worker_url() -> str:
    """Return the current worker URL (static or dynamically resolved)."""
    if WORKER_MODE == "slurm":
        return _worker_state.get("url") or WORKER_URL
    return WORKER_URL


def _is_worker_reachable(url: str = None) -> bool:
    """Quick health probe."""
    try:
        r = http_requests.get(f"{url or _get_worker_url()}/health", timeout=3)
        return r.ok
    except Exception:
        return False


def _submit_slurm_worker(ckpt_path: Optional[str] = None) -> str:
    """Submit the worker SLURM job. Returns the SLURM job ID."""
    env = os.environ.copy()
    if ckpt_path:
        env["FLOWR_CKPT_PATH"] = ckpt_path
    # Tell the worker which port to use and the frontend URL for file downloads
    env["FLOWR_WORKER_PORT"] = str(int(os.environ.get("FLOWR_WORKER_PORT", "8788")))
    server_port = int(os.environ.get("FLOWR_PORT", "8787"))
    env["FLOWR_SERVER_URL"] = os.environ.get(
        "FLOWR_SERVER_URL", f"http://localhost:{server_port}"
    )

    # Build sbatch command with CLI overrides from hpc.env environment vars.
    # CLI args take precedence over #SBATCH directives in the script.
    # --export=ALL ensures the full environment propagates to the compute node
    # (some clusters default to --export=NONE).
    sbatch_cmd = ["sbatch", "--parsable", "--export=ALL"]

    # Map FLOWR_SLURM_* env vars to sbatch CLI flags.
    # We use the FLOWR_SLURM_* prefix to avoid conflicts with SLURM-reserved
    # environment variables (SLURM_CPUS_PER_TASK, SLURM_MEM_PER_CPU, etc.).
    _slurm_overrides = {
        "FLOWR_SLURM_PARTITION": "--partition",
        "FLOWR_SLURM_TIME": "--time",
        "FLOWR_SLURM_MEM_PER_CPU": "--mem-per-cpu",
        "FLOWR_SLURM_CPUS_PER_TASK": "--cpus-per-task",
        "FLOWR_SLURM_GRES": "--gres",
    }
    for env_var, flag in _slurm_overrides.items():
        val = os.environ.get(env_var)
        if val:
            sbatch_cmd.append(f"{flag}={val}")

    # SLURM output/error files — use absolute paths so they don't depend on CWD
    slurm_output_dir = os.environ.get(
        "FLOWR_SLURM_OUTPUT_DIR", os.path.join(os.path.expanduser("~"), "slurm_outs")
    )
    os.makedirs(slurm_output_dir, exist_ok=True)
    sbatch_cmd.append(f"--output={slurm_output_dir}/flowr_gpu_%j.out")
    sbatch_cmd.append(f"--error={slurm_output_dir}/flowr_gpu_%j.err")

    sbatch_cmd.append(SLURM_SCRIPT)

    logger.info("Submitting SLURM worker: %s", " ".join(sbatch_cmd))
    result = subprocess.run(
        sbatch_cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    slurm_id = result.stdout.strip().split(";")[0]
    logger.info("Submitted SLURM worker job: %s", slurm_id)
    return slurm_id


def _validate_slurm_id(slurm_job_id: str) -> str:
    """Validate SLURM job ID is numeric (optionally with array suffix)."""
    if not re.match(r"^\d+(_\d+)?$", str(slurm_job_id)):
        raise ValueError(f"Invalid SLURM job ID: {slurm_job_id!r}")
    return slurm_job_id


def _get_slurm_node(slurm_job_id: str) -> Optional[str]:
    """Query SLURM for the node the job is running on."""
    _validate_slurm_id(slurm_job_id)
    try:
        result = subprocess.run(
            ["squeue", "-j", slurm_job_id, "-h", "-o", "%N"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        node = result.stdout.strip()
        if node and node != "(null)" and node != "":
            return node
    except Exception:
        logger.debug("squeue lookup failed for job %s", slurm_job_id, exc_info=True)
    return None


def _cancel_slurm_job(slurm_job_id: str):
    """Cancel a SLURM job."""
    _validate_slurm_id(slurm_job_id)
    try:
        subprocess.run(
            ["scancel", slurm_job_id],
            capture_output=True,
            timeout=10,
        )
        logger.info("Cancelled SLURM job %s", slurm_job_id)
    except Exception as e:
        logger.warning("Failed to cancel SLURM job %s: %s", slurm_job_id, e)


def _ensure_worker_running(job: dict) -> str:  # NOSONAR
    """Ensure a GPU worker is running. Returns the worker URL.

    In static mode, just returns the configured URL.
    In SLURM mode, submits a job if needed and waits for it to come online.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    if WORKER_MODE == "static":
        return WORKER_URL

    with _worker_lock:
        # Already running?
        if _worker_state["status"] == "running" and _worker_state.get("url"):
            cached_url = _worker_state["url"]
        else:
            cached_url = None

        need_submit = False
        if cached_url is None:
            if _worker_state["status"] == "starting":
                # Another thread is already waiting — we'll wait outside the lock
                pass
            else:
                need_submit = True
                _worker_state.update(status="starting", error=None)

    # Probe the cached URL outside the lock (network I/O)
    if cached_url is not None:
        if _is_worker_reachable(cached_url):
            return cached_url
        # Worker died — clean up (guard against concurrent recovery)
        with _worker_lock:
            if _worker_state["status"] == "starting":
                need_submit = False  # Another thread already handling recovery
            else:
                _worker_state.update(
                    status="idle", slurm_job_id=None, node=None, url=None
                )
                need_submit = True
                _worker_state.update(status="starting", error=None)

    if need_submit:
        with _worker_lock:
            try:
                slurm_id = _submit_slurm_worker(ckpt_path=job.get("ckpt_path"))
                _worker_state["slurm_job_id"] = slurm_id
            except Exception as e:
                _worker_state.update(status="idle", error=str(e))
                raise RuntimeError(f"Failed to submit GPU job: {e}")

    # Wait for the worker to come online (outside the lock)
    job["status"] = "allocating_gpu"
    job["progress"] = 1

    worker_port = int(os.environ.get("FLOWR_WORKER_PORT", "8788"))
    deadline = time.time() + SLURM_STARTUP_TIMEOUT
    poll_interval = 3

    while time.time() < deadline:
        slurm_id = _worker_state.get("slurm_job_id")

        # Try to discover the node
        if _worker_state.get("node") is None and slurm_id:
            node = _get_slurm_node(slurm_id)
            if node:
                _worker_state["node"] = node
                _worker_state["url"] = f"http://{node}:{worker_port}"
                logger.info("Worker node discovered: %s", node)
                job["progress"] = 3

        # Probe the worker
        url = _worker_state.get("url")
        if url and _is_worker_reachable(url):
            with _worker_lock:
                _worker_state["status"] = "running"
            logger.info("Worker is online at %s", url)
            return url

        time.sleep(poll_interval)

    # Timeout
    with _worker_lock:
        _worker_state.update(status="idle", error="Worker startup timed out")
        slurm_id_to_cancel = _worker_state.get("slurm_job_id")
    if slurm_id_to_cancel:
        _cancel_slurm_job(slurm_id_to_cancel)
    raise RuntimeError(
        f"GPU worker did not start within {SLURM_STARTUP_TIMEOUT}s. "
        "The SLURM queue may be full."
    )


def _release_worker():
    """Signal the worker to shut down and release the GPU."""
    if WORKER_MODE == "static":
        return  # Don't shut down a static worker

    with _worker_lock:
        url = _worker_state.get("url")
        slurm_id = _worker_state.get("slurm_job_id")

    # Try graceful shutdown (outside lock — slow I/O)
    if url:
        try:
            http_requests.post(f"{url}/shutdown", timeout=5)
            logger.info("Sent shutdown signal to worker")
        except Exception:
            logger.debug("Failed to send shutdown signal to worker", exc_info=True)

    # Cancel SLURM job as fallback
    if slurm_id:
        _cancel_slurm_job(slurm_id)

    with _worker_lock:
        _worker_state.update(status="idle", slurm_job_id=None, node=None, url=None)
    logger.info("GPU worker released")


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATION PROXY (delegates to GPU worker)
# ═══════════════════════════════════════════════════════════════════════════


def _accumulate_generated_history(job: dict, results: list):
    """Track generated SMILES and full results per iteration for diversity
    checking and multi-round visualization."""
    new_smiles = [r.get("smiles") for r in results if r.get("smiles")]
    history = job.get("generated_history", [])
    history.extend(new_smiles)
    job["generated_history"] = history

    # Increment iteration counter and store results per round
    iteration = job.get("iteration_idx", 0)
    all_results = job.get("all_results", [])
    all_results.append({"iteration": iteration, "results": list(results)})
    job["all_results"] = all_results
    job["iteration_idx"] = iteration + 1


def _resolve_prior_center(job: dict, req: dict, job_id: str):
    """Resolve prior center URL for the worker. Returns (url, filename) or (None, None)."""
    prior_center_coords = req.get("prior_center_coords")
    if prior_center_coords:
        job_dir = _job_upload_dir(job.get("user", "_unknown"), job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        xyz_name = "_prior_center_placed.xyz"
        xyz_path = job_dir / xyz_name
        xyz_path.write_text(
            f"1\nPlaced prior center\n"
            f"Ar  {prior_center_coords['x']:.6f}  "
            f"{prior_center_coords['y']:.6f}  "
            f"{prior_center_coords['z']:.6f}\n"
        )
        return _worker_file_url(job_id, xyz_name), xyz_name
    elif req.get("prior_center_filename"):
        safe_name = Path(req["prior_center_filename"]).name
        candidate = _job_upload_dir(job.get("user", "_unknown"), job_id) / safe_name
        if candidate.exists():
            return _worker_file_url(job_id, safe_name), safe_name
    return None, None


def _build_base_worker_payload(
    job_id,
    job,
    req,
    workflow,
    *,
    ligand_url,
    ligand_filename,
    prior_center_url,
    prior_center_filename,
    num_heavy_atoms,
):
    """Build the shared worker payload fields common to SBDD and LBDD."""
    return {
        "job_id": job_id,
        "workflow_type": workflow,
        "ligand_url": ligand_url,
        "ligand_filename": ligand_filename,
        "ckpt_path": (
            job.get("finetuned_ckpt_path")
            or job.get("ckpt_path")
            or _get_user_setting(job.get("user", ""), "ckpt_path")
        ),
        "finetuned_ckpt_url": job.get("finetuned_ckpt_url"),
        "gen_mode": req["gen_mode"],
        "fixed_atoms": req.get("fixed_atoms", []),
        "n_samples": req["n_samples"],
        "batch_size": req["batch_size"],
        "integration_steps": req["integration_steps"],
        "coord_noise_scale": req.get("coord_noise_scale", 0.0),
        "grow_size": req.get("grow_size"),
        "prior_center_url": prior_center_url,
        "prior_center_filename": prior_center_filename,
        "filter_valid_unique": req.get("filter_valid_unique", True),
        "filter_cond_substructure": req.get("filter_cond_substructure", False),
        "filter_diversity": req.get("filter_diversity", False),
        "diversity_threshold": req.get("diversity_threshold", 0.9),
        "sample_mol_sizes": req.get("sample_mol_sizes", False),
        "calculate_strain_energies": req.get("calculate_strain_energies", False),
        "anisotropic_prior": req.get("anisotropic_prior", False),
        "ring_system_index": req.get("ring_system_index", 0),
        "ref_ligand_com_prior": req.get("ref_ligand_com_prior", False),
        "num_heavy_atoms": num_heavy_atoms,
        "property_filter": req.get("property_filter"),
        "adme_filter": req.get("adme_filter"),
        "previous_smiles": job.get("generated_history", []),
    }


def _poll_worker_loop(job: dict, job_id: str, worker_url: str):
    """Poll worker until job completes, fails, is cancelled, or times out."""

    def _is_cancelled():
        return bool(job.get("cancel_requested_at") or job.get("cancelled"))

    generation_timeout = int(os.environ.get("FLOWR_GENERATION_TIMEOUT", "1800"))
    poll_deadline = time.time() + generation_timeout
    while time.time() < poll_deadline:
        if job.get("status") == "cancelled":
            logger.info("[Job %s] Cancelled by user", job_id)
            break
        time.sleep(2)
        worker_data = _poll_worker_job(job_id, worker_url)
        if worker_data is None:
            continue
        w_status = worker_data.get("status")
        w_progress = worker_data.get("progress", 0)
        if not _is_cancelled():
            job["progress"] = max(job.get("progress", 10), w_progress)
        if w_status == "completed":
            if _is_cancelled():
                job["status"] = "cancelled"
            else:
                job.update(
                    status="completed",
                    progress=100,
                    results=worker_data.get("results", []),
                    metrics=worker_data.get("metrics", []),
                    elapsed_time=worker_data.get("elapsed_time"),
                    mode=worker_data.get("mode", "flowr"),
                    n_generated=worker_data.get("n_generated", 0),
                    used_optimization=worker_data.get("used_optimization", False),
                    prior_cloud=worker_data.get("prior_cloud"),
                    warnings=worker_data.get("warnings", []),
                )
                _accumulate_generated_history(job, worker_data.get("results", []))
                _invalidate_viz_caches(job)
            break
        elif w_status == "cancelled":
            job.update(status="cancelled", error=_ERR_CANCELLED)
            break
        elif w_status == "failed":
            if _is_cancelled():
                job["status"] = "cancelled"
            else:
                job.update(
                    status="failed",
                    progress=0,
                    error=worker_data.get("error", "Worker generation failed"),
                    health_check_type=worker_data.get("health_check_type"),
                    health_check_advice=worker_data.get("health_check_advice"),
                )
            break
        elif w_status == "loading_model":
            if not _is_cancelled():
                job["status"] = "loading_model"
        elif w_status in ("generating", "queued"):
            if not _is_cancelled():
                job["status"] = "generating"

    if job.get("status") not in ("completed", "failed", "cancelled"):
        job.update(
            status="failed",
            progress=0,
            error=f"Generation timed out after {generation_timeout}s",
        )


def _proxy_generation(job_id: str, req: dict):  # NOSONAR
    """Send generation request to the GPU worker and track its progress.

    In SLURM mode this will first allocate a GPU node, wait for the worker
    to come online, run generation, and then release the GPU.

    This function is called in a background thread from POST /generate.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        return

    def _is_cancelled() -> bool:
        return bool(job.get("cancel_requested_at") or job.get("cancelled"))

    job["status"] = "allocating_gpu" if WORKER_MODE == "slurm" else "starting"
    job["progress"] = 1
    did_increment = False

    try:
        # 1. Ensure the worker is running (submits SLURM in slurm mode)
        worker_url = _ensure_worker_running(job)

        with _worker_lock:
            _worker_state["active_jobs"] = _worker_state.get("active_jobs", 0) + 1
        did_increment = True

        # Still allocating — model loads when worker receives /generate
        job["progress"] = 5

        # 2. Build file-download URLs that the worker can fetch from us
        workflow = req.get("workflow_type", job.get("workflow_type", "sbdd"))

        # Ligand URL may be None for de novo pocket-only / scratch mode
        ligand_url = None
        ligand_filename = None
        if job.get("ligand_path"):
            ligand_url = _worker_file_url(job_id, Path(job["ligand_path"]).name)
            ligand_filename = Path(job["ligand_path"]).name

        num_heavy_atoms = req.get("num_heavy_atoms") or job.get("num_heavy_atoms")

        # ── LBDD branch: no protein, but supports all gen modes ──
        if workflow == "lbdd":
            prior_center_url, prior_center_filename = _resolve_prior_center(
                job, req, job_id
            )

            worker_payload = _build_base_worker_payload(
                job_id,
                job,
                req,
                "lbdd",
                ligand_url=ligand_url,
                ligand_filename=ligand_filename,
                prior_center_url=prior_center_url,
                prior_center_filename=prior_center_filename,
                num_heavy_atoms=num_heavy_atoms,
            )
            worker_payload.update(
                {
                    "optimize_method": req.get("optimize_method", "none"),
                    "sample_n_molecules_per_mol": req.get(
                        "sample_n_molecules_per_mol", 1
                    ),
                }
            )

            resp = http_requests.post(
                f"{worker_url}/generate",
                json=worker_payload,
                timeout=30,
            )
            resp.raise_for_status()

            if not _is_cancelled():
                job["status"] = "starting"
                job["progress"] = 10

            _poll_worker_loop(job, job_id, worker_url)

        else:
            # ── SBDD branch ──
            protein_url = _worker_file_url(job_id, Path(job["protein_path"]).name)
            prior_center_url, prior_center_filename = _resolve_prior_center(
                job, req, job_id
            )

            worker_payload = _build_base_worker_payload(
                job_id,
                job,
                req,
                "sbdd",
                ligand_url=ligand_url,
                ligand_filename=ligand_filename,
                prior_center_url=prior_center_url,
                prior_center_filename=prior_center_filename,
                num_heavy_atoms=num_heavy_atoms,
            )
            worker_payload.update(
                {
                    "protein_url": protein_url,
                    "protein_filename": Path(job["protein_path"]).name,
                    "pocket_cutoff": req["pocket_cutoff"],
                    "filter_pb_valid": req.get("filter_pb_valid", False),
                    "calculate_pb_valid": req.get("calculate_pb_valid", False),
                    "optimize_gen_ligs": req.get("optimize_gen_ligs", False),
                    "optimize_gen_ligs_hs": req.get("optimize_gen_ligs_hs", False),
                }
            )

            resp = http_requests.post(
                f"{worker_url}/generate",
                json=worker_payload,
                timeout=30,
            )
            resp.raise_for_status()

            if not _is_cancelled():
                job["status"] = "starting"
                job["progress"] = 10

            _poll_worker_loop(job, job_id, worker_url)

    except Exception as exc:
        logger.exception("Generation proxy failed for job %s", job_id)
        err_msg = str(exc)
        if "CUDA out of memory" in err_msg or "OutOfMemoryError" in err_msg:
            err_msg = "CUDA out of memory \u2013 reduce batch size!"
        job.update(status="failed", progress=0, error=err_msg)

    finally:
        # 5. Decrement active job count; release only when no jobs remain
        if did_increment:
            with _worker_lock:
                _worker_state["active_jobs"] = max(
                    0, _worker_state.get("active_jobs", 0) - 1
                )
                should_release = _worker_state["active_jobs"] == 0
            if should_release:
                _release_worker()


def _poll_worker_job(job_id: str, worker_url: str = None) -> Optional[Dict[str, Any]]:
    """Poll the worker for job status. Returns the worker's response dict."""
    url = worker_url or _get_worker_url()
    try:
        resp = http_requests.get(f"{url}/job/{job_id}", timeout=10)
        if resp.status_code == 404:
            return {"status": "unknown", "progress": 0}
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  ROUTES
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/checkpoints")
async def list_checkpoints(workflow: str = "sbdd"):  # NOSONAR
    """Scan the ckpts/{workflow} directory and return available checkpoints."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    base: List[Dict[str, str]] = []
    project: List[Dict[str, Any]] = []

    # Determine root based on workflow type
    if workflow == "lbdd":
        scan_dir = CKPTS_DIR / "lbdd"
    else:
        scan_dir = CKPTS_DIR / "sbdd"

    logger.debug(
        "[checkpoints] Scanning %s (workflow=%s, exists=%s)",
        scan_dir,
        workflow,
        scan_dir.is_dir(),
    )

    if scan_dir.is_dir():
        for f in sorted(scan_dir.iterdir()):
            if f.is_file() and f.suffix == ".ckpt":
                base.append({"name": f.stem, "path": str(f)})
                logger.debug("[checkpoints]   base: %s", f.name)

        project_root = scan_dir / "project_model"
        if project_root.is_dir():
            for proj_dir in sorted(project_root.iterdir()):
                if proj_dir.is_dir():
                    ckpts_in_proj = sorted(proj_dir.glob("**/*.ckpt"))
                    if ckpts_in_proj:
                        project.append(
                            {
                                "project_id": proj_dir.name,
                                "checkpoints": [
                                    {"name": c.stem, "path": str(c)}
                                    for c in ckpts_in_proj
                                ],
                            }
                        )

    logger.debug(
        "[checkpoints] Found %d base, %d project checkpoints", len(base), len(project)
    )
    return {"base": base, "project": project}


@app.get("/health")
async def health():
    """Health check for the frontend server only.

    Does NOT probe the GPU worker — the worker is an ephemeral resource
    that is only allocated when the user clicks Generate.
    """
    return {
        "status": "ok",
        "rdkit": RDKIT_AVAILABLE,
        "openeye": OPENEYE_AVAILABLE,
    }


class LoadModelRequest(BaseModel):
    ckpt_path: Optional[str] = None


class RegisterCheckpointRequest(BaseModel):
    ckpt_path: str
    workflow_type: str = "sbdd"  # "sbdd" or "lbdd"


class SaveCheckpointRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class SwapCheckpointRequest(BaseModel):
    ckpt_path: str


@app.post("/register-checkpoint")
async def register_checkpoint(body: RegisterCheckpointRequest, request: Request):
    """Store the user's checkpoint selection. Does NOT load the model.

    Model loading is deferred until the user clicks Generate, at which
    point the worker will load it on the GPU if not already loaded.
    """
    user = request.state.user
    _set_user_setting(user, "ckpt_path", body.ckpt_path)
    _set_user_setting(user, "workflow_type", body.workflow_type)
    return {
        "status": "registered",
        "ckpt_path": body.ckpt_path,
        "workflow_type": body.workflow_type,
    }


_CKPT_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")


@app.post("/save-checkpoint/{job_id}")
async def save_checkpoint(job_id: str, body: SaveCheckpointRequest, request: Request):
    """Save a finetuned checkpoint to the permanent ckpts/ directory."""
    job_id = _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    user = request.state.user
    _check_rate_limit(user, "save-checkpoint")

    name = body.name
    if not _CKPT_NAME_PATTERN.fullmatch(name):
        raise HTTPException(
            400,
            "Invalid name. Use 1-64 alphanumeric characters, hyphens, or underscores.",
        )

    src_path = job.get("finetuned_ckpt_path")
    if not src_path or not Path(src_path).is_file():
        raise HTTPException(400, "No finetuned checkpoint available")

    workflow = (
        job.get("workflow_type") or _get_user_setting(user, "workflow_type") or "sbdd"
    )
    dest_dir = CKPTS_DIR / workflow / "project_model" / name
    dest = dest_dir / f"{name}.ckpt"

    # Path traversal check
    if not str(dest.resolve()).startswith(str(CKPTS_DIR.resolve())):
        raise HTTPException(403, "Invalid checkpoint path")

    try:
        dest_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise HTTPException(409, "A project checkpoint with this name already exists")

    shutil.copy2(src_path, dest)

    # Update job to use permanent path
    job["finetuned_ckpt_path"] = str(dest)
    _set_user_setting(user, "ckpt_path", str(dest))

    logger.info("[save-checkpoint] user=%s saved %s -> %s", user, src_path, dest)
    return {"saved": True, "name": name, "path": str(dest)}


@app.post("/swap-checkpoint/{job_id}")
async def swap_checkpoint(job_id: str, body: SwapCheckpointRequest, request: Request):
    """Swap the active checkpoint mid-session."""
    job_id = _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    user = request.state.user
    _check_rate_limit(user, "swap-checkpoint")

    p = Path(body.ckpt_path)

    # Path validation: must be under CKPTS_DIR
    if not str(p.resolve()).startswith(str(CKPTS_DIR.resolve())):
        raise HTTPException(403, "Invalid checkpoint path")

    if not p.is_file():
        raise HTTPException(400, "Checkpoint file not found")

    if p.suffix != ".ckpt":
        raise HTTPException(400, "File must have .ckpt extension")

    job["ckpt_path"] = str(p)
    job.pop("finetuned_ckpt_path", None)
    job.pop("finetuned_ckpt_url", None)
    _set_user_setting(user, "ckpt_path", str(p))

    logger.info("[swap-checkpoint] user=%s swapped to %s", user, p)
    return {"swapped": True, "ckpt_path": str(p), "name": p.stem}


@app.post("/load-model")
async def load_model_endpoint(request: Request, body: LoadModelRequest = None):
    """Proxy model loading to the GPU worker (called during generation, not at launch)."""
    user = request.state.user
    ckpt = body.ckpt_path if body else None
    if ckpt:
        _set_user_setting(user, "ckpt_path", ckpt)

    try:
        resp = await asyncio.to_thread(
            http_requests.post,
            f"{_get_worker_url()}/load-model",
            json={"ckpt_path": ckpt},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.ConnectionError:
        # Worker not running yet — that's OK for cloud deployment.
        # The worker will load the model when it starts and receives
        # a generation request.
        return {
            "status": "deferred",
            "message": "Worker not yet available. Model will load on first generation.",
            "ckpt_path": ckpt,
        }
    except Exception as exc:
        raise HTTPException(502, f"Worker communication failed: {exc}")


@app.get("/model-status")
async def model_status():
    """Proxy model status from the worker."""
    try:
        resp = await asyncio.to_thread(
            http_requests.get, f"{_get_worker_url()}/model-status", timeout=5
        )
        resp.raise_for_status()
        return resp.json()
    except http_requests.ConnectionError:
        return {
            "loaded": False,
            "loading": False,
            "error": None,
            "device": "pending",
            "worker_available": False,
        }
    except Exception as exc:
        return {
            "loaded": False,
            "loading": False,
            "error": str(exc),
            "device": "unknown",
            "worker_available": False,
        }


# ── File serving (so the worker can download uploaded files) ──


@app.get("/files/{job_id}/{filename}")
async def serve_file(job_id: str, filename: str, request: Request):
    """Serve uploaded files so the GPU worker can download them.

    Requires a valid worker file token (query param or header).
    """
    # Verify worker file token
    token = request.query_params.get("token") or request.headers.get("X-Worker-Token")
    if not token or not hmac.compare_digest(token, _WORKER_FILE_TOKEN):
        raise HTTPException(403, "Access denied.")
    _validate_job_id(job_id)
    safe_name = _sanitize_filename(filename)
    # Look up the owning user from the job to find the right directory
    with _jobs_lock:
        job = JOBS.get(job_id)
    user = job.get("user", "_unknown") if job else "_unknown"
    file_path = (_job_upload_dir(user, job_id) / safe_name).resolve()
    if not str(file_path).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(403, "Access denied.")
    if not file_path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(str(file_path))


# ── File uploads ──


class CreateDenovoJobRequest(BaseModel):
    """Create a job for de novo generation without a reference ligand."""

    job_id: Optional[str] = None  # re-use existing job (e.g. after protein upload)
    num_heavy_atoms: int = Field(default=25, ge=1, le=200)
    workflow_type: str = "sbdd"  # "sbdd" or "lbdd"


class FetchRCSBRequest(BaseModel):
    """Fetch a PDB structure from RCSB by PDB ID."""

    pdb_id: str = Field(..., min_length=4, max_length=4, pattern=r"^[A-Za-z0-9]{4}$")
    ligand_id: Optional[str] = Field(
        None, min_length=1, max_length=5, pattern=r"^[A-Za-z0-9]{1,5}$"
    )


@app.post("/create-denovo-job")
async def create_denovo_job(body: CreateDenovoJobRequest, request: Request):
    """Create (or update) a job for de novo generation without a reference ligand.

    For SBDD pocket-only: the protein must already be uploaded (job_id required).
    For LBDD scratch: creates a brand-new job with no files at all.
    """
    user = request.state.user
    job_id = body.job_id
    if job_id:
        _validate_job_id(job_id)
        with _jobs_lock:
            job = JOBS.get(job_id)
        if job is not None:
            if job.get("user") != user:
                raise HTTPException(403, "Access denied")
        elif body.workflow_type == "lbdd":
            job = None  # will create below
        else:
            raise HTTPException(
                400,
                "SBDD pocket-only mode requires an existing job_id with a protein upload.",
            )
    else:
        job = None

    if job is None and body.workflow_type == "lbdd":
        # Create a fresh job with no files
        job_id = str(uuid.uuid4())[:8]
        job_dir = _job_upload_dir(user, job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": job_id,
            "user": user,
            "status": "denovo_ready",
            "created_at": time.time(),
            "ckpt_path": _get_user_setting(user, "ckpt_path"),
            "workflow_type": body.workflow_type,
        }
        with _jobs_lock:
            JOBS[job_id] = job
    elif job is None:
        raise HTTPException(
            400,
            "SBDD pocket-only mode requires an existing job_id with a protein upload.",
        )

    job["num_heavy_atoms"] = body.num_heavy_atoms
    job["workflow_type"] = body.workflow_type
    job["denovo_no_ligand"] = True

    return {
        "job_id": job_id,
        "num_heavy_atoms": body.num_heavy_atoms,
        "workflow_type": body.workflow_type,
    }


@app.post("/upload/protein")
async def upload_protein(request: Request, file: UploadFile = File(...)):
    _cleanup_expired_jobs()  # Evict stale jobs on new upload
    user = request.state.user
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(413, _ERR_FILE_TOO_LARGE)
    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in (".pdb", ".cif", ".mmcif"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = str(uuid.uuid4())[:8]
    job_dir = _job_upload_dir(user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    resolved = save_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, _ERR_INVALID_FILE_PATH)

    job = {
        "job_id": job_id,
        "user": user,
        "protein_path": str(save_path),
        "protein_filename": safe_name,
        "status": "protein_uploaded",
        "created_at": time.time(),
        "ckpt_path": _get_user_setting(user, "ckpt_path"),
        "workflow_type": _get_user_setting(user, "workflow_type", "sbdd"),
    }
    with _jobs_lock:
        JOBS[job_id] = job
    return {
        "job_id": job_id,
        "filename": file.filename,
        "format": ext,
        "pdb_data": content.decode("utf-8", errors="replace"),
    }


@app.post("/upload/ligand/{job_id}")
async def upload_ligand(job_id: str, request: Request, file: UploadFile = File(...)):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(413, _ERR_FILE_TOO_LARGE)
    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in (".sdf", ".mol", ".mol2", ".pdb"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_dir = _job_upload_dir(request.state.user, job_id)
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    resolved = save_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, _ERR_INVALID_FILE_PATH)

    mol = _read_ligand_mol(str(save_path))
    if mol is None:
        raise HTTPException(400, "Could not parse ligand file.")

    # Validate molecule chemistry (kekulization, valence, etc.)
    is_valid, validation_error = _validate_molecule(mol)
    if not is_valid:
        raise HTTPException(400, validation_error)

    # Crawl SDF for reference affinity properties
    ref_affinity = _crawl_sdf_affinity(str(save_path))

    job.update(
        ligand_path=str(save_path),
        ligand_filename=safe_name,
        ligand_smiles=_mol_to_smiles(mol),
        status="ligand_uploaded",
        ref_affinity=ref_affinity,
    )
    job.setdefault("ckpt_path", _get_user_setting(request.state.user, "ckpt_path"))
    job.setdefault("workflow_type", "sbdd")

    mol_noh = Chem.RemoveHs(mol)
    smiles_noh = _mol_to_smiles(mol_noh)
    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    return {
        "job_id": job_id,
        "filename": file.filename,
        "smiles": _mol_to_smiles(mol),
        "smiles_noH": smiles_noh,
        "has_explicit_hs": has_explicit_hs,
        "sdf_data": _mol_to_sdf_string(mol),
        "atoms": _mol_to_atom_info(mol),
        "bonds": _mol_to_bond_info(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "ref_affinity": ref_affinity,
        "ref_properties": _compute_all_properties(mol_noh),
    }


# ── Fetch PDB structure from RCSB ──


@app.post("/fetch-rcsb")
async def fetch_rcsb(body: FetchRCSBRequest, request: Request):
    """Fetch a PDB structure from RCSB, split into protein + ligand."""
    _cleanup_expired_jobs()
    user = request.state.user
    pdb_id = body.pdb_id.upper()

    job_id = str(uuid.uuid4())[:8]
    job_dir = _job_upload_dir(user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — biotite may not be in every environment
    try:
        import sys as _sys

        if str(ROOT_DIR) not in _sys.path:
            _sys.path.insert(0, str(ROOT_DIR))
        from flowr.data.preprocess_pdb import transform_pdb_biotite
    except ImportError:
        raise HTTPException(501, "biotite is not installed — RCSB fetch is unavailable")

    ligand_found = True
    ligand_path = None
    protein_path = None
    try:
        protein_path, ligand_path = transform_pdb_biotite(
            str(job_dir), pdb_id, ligand_id=body.ligand_id
        )
    except ValueError:
        # No ligand found — transform_pdb_biotite writes protein AFTER
        # ligand processing, so the file may not exist yet.  Do a
        # protein-only fetch using biotite directly.
        ligand_found = False
        protein_path = str(job_dir / f"{pdb_id}.pdb")
        if not Path(protein_path).exists():
            try:
                import biotite.database.rcsb as rcsb
                import biotite.structure.io.pdb as pdb_io
                import biotite.structure.io.pdbx as pdbx

                with tempfile.TemporaryDirectory() as tmpdir:
                    cif_path = rcsb.fetch(pdb_id, "cif", target_path=tmpdir)
                    cif_file = pdbx.CIFFile.read(cif_path)
                    atoms = pdbx.get_structure(cif_file, model=1, include_bonds=True)
                pdb_out = pdb_io.PDBFile()
                pdb_io.set_structure(pdb_out, atoms)
                pdb_out.write(protein_path)
            except Exception as fetch_exc:
                raise HTTPException(
                    400, f"Failed to fetch protein for {pdb_id}: {fetch_exc}"
                )
    except Exception as exc:
        raise HTTPException(400, f"Failed to fetch PDB {pdb_id}: {exc}")

    # Path traversal check (defence-in-depth)
    for fpath in filter(None, [protein_path, ligand_path]):
        if not str(Path(fpath).resolve()).startswith(str(UPLOAD_DIR.resolve())):
            raise HTTPException(400, _ERR_INVALID_FILE_PATH)

    # Read protein PDB text for immediate frontend rendering
    pdb_data = Path(protein_path).read_text()
    protein_filename = Path(protein_path).name

    # Build job entry (mirrors upload_protein)
    job: Dict[str, Any] = {
        "job_id": job_id,
        "user": user,
        "protein_path": protein_path,
        "protein_filename": protein_filename,
        "status": "protein_uploaded",
        "created_at": time.time(),
        "ckpt_path": _get_user_setting(user, "ckpt_path"),
        "workflow_type": _get_user_setting(user, "workflow_type", "sbdd"),
        "rcsb_pdb_id": pdb_id,
    }

    response: Dict[str, Any] = {
        "job_id": job_id,
        "protein_filename": protein_filename,
        "pdb_data": pdb_data,
        "ligand_found": ligand_found,
    }

    if ligand_found and ligand_path:
        mol = _read_ligand_mol(ligand_path)
        if mol is None:
            # Fallback: protein loaded but ligand parse failed
            ligand_found = False
            response["ligand_found"] = False
        else:
            is_valid, _val_err = _validate_molecule(mol)
            if not is_valid:
                ligand_found = False
                response["ligand_found"] = False
            else:
                mol_noh = Chem.RemoveHs(mol)
                smiles_noh = _mol_to_smiles(mol_noh)
                ligand_filename = Path(ligand_path).name
                # Extract ligand_id from filename: {pdb_id}_{ligand_id}_ligand.sdf
                detected_ligand_id = ligand_filename.replace(f"{pdb_id}_", "").replace(
                    "_ligand.sdf", ""
                )

                job.update(
                    ligand_path=ligand_path,
                    ligand_filename=ligand_filename,
                    ligand_smiles=_mol_to_smiles(mol),
                    status="ligand_uploaded",
                )

                response["ligand"] = {
                    "filename": ligand_filename,
                    "ligand_id": detected_ligand_id,
                    "smiles": _mol_to_smiles(mol),
                    "smiles_noH": smiles_noh,
                    "has_explicit_hs": any(
                        a.GetAtomicNum() == 1 for a in mol.GetAtoms()
                    ),
                    "sdf_data": _mol_to_sdf_string(mol),
                    "atoms": _mol_to_atom_info(mol),
                    "bonds": _mol_to_bond_info(mol),
                    "num_atoms": mol.GetNumAtoms(),
                    "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                    "ref_properties": _compute_all_properties(mol_noh),
                }

    with _jobs_lock:
        JOBS[job_id] = job
    return response


# ── Set a generated ligand as the new reference (accepts raw SDF) ──


class SetReferenceRequest(BaseModel):
    sdf_data: str = Field(..., max_length=500_000)


@app.post("/set-reference/{job_id}")
async def set_reference(job_id: str, req: SetReferenceRequest, request: Request):
    """Replace the reference ligand from a raw SDF string (e.g. a generated ligand)."""
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    # ── Persist original ligand on first reference swap ──
    if "original_ligand_path" not in job:
        orig_path = job.get("ligand_path", "")
        orig_mol = _read_ligand_mol(orig_path) if orig_path else None
        if orig_mol is not None:
            orig_mol_noh = Chem.RemoveHs(orig_mol)
            job["original_ligand_path"] = orig_path
            job["original_ligand_smiles"] = _mol_to_smiles(orig_mol)
            job["original_ligand_smiles_noH"] = _mol_to_smiles(orig_mol_noh)
            job["original_ligand_sdf"] = _mol_to_sdf_string(orig_mol)
            job["original_ligand_properties"] = _compute_all_properties(orig_mol_noh)
            job["original_ligand_num_atoms"] = orig_mol.GetNumAtoms()
            job["original_ligand_num_heavy_atoms"] = orig_mol.GetNumHeavyAtoms()
            job["original_ligand_ref_affinity"] = job.get("ref_affinity")
            job["original_ligand_filename"] = job.get("ligand_filename", "")

    job_dir = _job_upload_dir(request.state.user, job_id)
    save_path = job_dir / _REF_LIGAND_FILENAME
    save_path.write_text(req.sdf_data)

    mol = _read_ligand_mol(str(save_path))
    if mol is None:
        raise HTTPException(400, "Could not parse SDF data.")

    is_valid, validation_error = _validate_molecule(mol)
    if not is_valid:
        raise HTTPException(400, validation_error)

    ref_affinity = _crawl_sdf_affinity(str(save_path))

    # Invalidate cached plot data (reference changed)
    for key in list(job):  # NOSONAR
        if key.startswith("chemspace_"):
            del job[key]
    job.pop("propspace_cache", None)
    job.pop("affinity_dist_cache", None)

    job.update(
        ligand_path=str(save_path),
        ligand_filename=_REF_LIGAND_FILENAME,
        ligand_smiles=_mol_to_smiles(mol),
        status="ligand_uploaded",
        ref_affinity=ref_affinity,
    )

    mol_noh = Chem.RemoveHs(mol)
    smiles_noh = _mol_to_smiles(mol_noh)
    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    # Build original ligand payload (None if this is the first upload)
    original_ligand = None
    if "original_ligand_path" in job:
        original_ligand = {
            "smiles": job["original_ligand_smiles"],
            "smiles_noH": job["original_ligand_smiles_noH"],
            "sdf_data": job["original_ligand_sdf"],
            "properties": job["original_ligand_properties"],
            "num_atoms": job["original_ligand_num_atoms"],
            "num_heavy_atoms": job["original_ligand_num_heavy_atoms"],
            "ref_affinity": job.get("original_ligand_ref_affinity"),
            "filename": job.get("original_ligand_filename", ""),
        }

    return {
        "job_id": job_id,
        "filename": _REF_LIGAND_FILENAME,
        "smiles": _mol_to_smiles(mol),
        "smiles_noH": smiles_noh,
        "has_explicit_hs": has_explicit_hs,
        "sdf_data": _mol_to_sdf_string(mol),
        "atoms": _mol_to_atom_info(mol),
        "bonds": _mol_to_bond_info(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "ref_affinity": ref_affinity,
        "ref_properties": _compute_all_properties(mol_noh),
        "original_ligand": original_ligand,
    }


# ── LBDD: Upload molecule without protein ──


@app.post("/upload/molecule")
async def upload_molecule(request: Request, file: UploadFile = File(...)):
    """Upload a molecule file for LBDD workflow (no protein required).

    Creates a new job and stores the ligand. Accepts SDF, MOL, MOL2.
    """
    _cleanup_expired_jobs()
    user = request.state.user
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(413, _ERR_FILE_TOO_LARGE)
    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in (".sdf", ".mol", ".mol2"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = str(uuid.uuid4())[:8]
    job_dir = _job_upload_dir(user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    resolved = save_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, _ERR_INVALID_FILE_PATH)

    mol = _read_ligand_mol(str(save_path))
    if mol is None:
        raise HTTPException(400, "Could not parse molecule file.")

    # Validate molecule chemistry (kekulization, valence, etc.)
    is_valid, validation_error = _validate_molecule(mol)
    if not is_valid:
        raise HTTPException(400, validation_error)

    mol_noh = Chem.RemoveHs(mol)
    smiles_noh = _mol_to_smiles(mol_noh)
    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    with _jobs_lock:
        JOBS[job_id] = {
            "job_id": job_id,
            "user": user,
            "ligand_path": str(save_path),
            "ligand_filename": safe_name,
            "ligand_smiles": _mol_to_smiles(mol),
            "workflow_type": "lbdd",
            "status": "ligand_uploaded",
            "created_at": time.time(),
            "ckpt_path": _get_user_setting(user, "ckpt_path"),
        }

    return {
        "job_id": job_id,
        "filename": file.filename,
        "smiles": _mol_to_smiles(mol),
        "smiles_noH": smiles_noh,
        "has_explicit_hs": has_explicit_hs,
        "sdf_data": _mol_to_sdf_string(mol),
        "atoms": _mol_to_atom_info(mol),
        "bonds": _mol_to_bond_info(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "ref_properties": _compute_all_properties(mol_noh),
    }


# ── ADMET model upload ──


@app.post("/upload/adme-model/{job_id}")
async def upload_adme_model(
    job_id: str, request: Request, file: UploadFile = File(...)
):
    """Upload an ADMET model file for property-based filtering during generation."""
    _validate_job_id(job_id)
    _get_user_job(job_id, request)  # validates ownership

    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(413, _ERR_FILE_TOO_LARGE)
    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    allowed_exts = (".pt", ".pth", ".pkl", ".joblib", ".ckpt", ".onnx", ".bin")
    if ext not in allowed_exts:
        raise HTTPException(
            400,
            f"Unsupported model format: {ext}. Accepted: {', '.join(allowed_exts)}",
        )

    job_dir = _job_upload_dir(request.state.user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"adme_{safe_name}"
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    resolved = save_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, _ERR_INVALID_FILE_PATH)

    # Build a URL the worker can use to download the model
    model_url = _worker_file_url(job_id, safe_name)

    return {
        "job_id": job_id,
        "filename": safe_name,
        "model_url": model_url,
    }


# ── LBDD: SMILES to molecule + conformer generation ──


class SmilesConformerRequest(BaseModel):
    smiles: str
    max_confs: int = Field(default=10, ge=1, le=50)


@app.post("/generate-conformers")
async def generate_conformers(
    body: SmilesConformerRequest, request: Request
):  # NOSONAR
    """Generate 3D conformers from a SMILES string using OpenEye Omega.

    Returns a list of conformers with SDF data and relative energies.
    Falls back to RDKit ETKDG if OpenEye is not available.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    user = request.state.user
    smiles = body.smiles.strip()
    max_confs = body.max_confs

    if not smiles:
        raise HTTPException(400, "Empty SMILES string.")

    # Create job for this molecule
    job_id = str(uuid.uuid4())[:8]
    job_dir = _job_upload_dir(user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    conformers = []

    if OPENEYE_AVAILABLE:
        try:
            # Parse SMILES into OEMol
            oemol = oechem.OEMol()
            if not oechem.OESmilesToMol(oemol, smiles):
                raise HTTPException(400, f"Invalid SMILES: {smiles}")
            oechem.OEAddExplicitHydrogens(oemol)

            # Run Omega conformer generation
            # Use OEOmegaSampling_Pose (matches working oe_conformer.py).
            omega_opts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
            omega_opts.SetMaxConfs(max_confs)
            omega_opts.SetStrictStereo(False)
            omega = oeomega.OEOmega(omega_opts)
            ret = omega.Build(oemol)
            if ret != oeomega.OEOmegaReturnCode_Success:
                raise HTTPException(
                    500,
                    f"Omega conformer generation failed: "
                    f"{oeomega.OEGetOmegaError(ret)}",
                )

            # Extract each conformer as separate SDF + energy
            for i, conf in enumerate(oemol.GetConfs()):
                # Get energy
                energy = conf.GetEnergy()

                # Convert to RDKit mol for SDF output
                single_mol = oechem.OEMol(conf)
                ofs = oechem.oemolostream()
                ofs.SetFormat(oechem.OEFormat_SDF)
                ofs.openstring()
                oechem.OEWriteMolecule(ofs, single_mol)
                sdf_block = ofs.GetString().decode("utf-8")

                # Also try to get an RDKit mol for property computation
                rdmol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
                props = _compute_all_properties(rdmol) if rdmol else {}

                conformers.append(
                    {
                        "idx": i,
                        "sdf": sdf_block,
                        "energy": round(energy, 4),
                        "properties": props,
                    }
                )

        except HTTPException:
            raise
        except Exception as exc:
            # Fall back to RDKit if OE fails
            logger.warning(
                "OpenEye conformer generation failed: %s, falling back to RDKit", exc
            )
            conformers = _generate_conformers_rdkit(smiles, max_confs)
    else:
        conformers = _generate_conformers_rdkit(smiles, max_confs)

    if not conformers:
        raise HTTPException(500, "Failed to generate any conformers.")

    # Sort by energy (lowest first); None energies (MMFF failures) sort last
    conformers.sort(
        key=lambda c: c["energy"] if c.get("energy") is not None else float("inf")
    )
    # Re-index after sorting
    for i, c in enumerate(conformers):
        c["idx"] = i

    # Save conformers to job directory for later selection
    with _jobs_lock:
        JOBS[job_id] = {
            "job_id": job_id,
            "user": user,
            "workflow_type": "lbdd",
            "conformers": conformers,
            "smiles": smiles,
            "status": "conformers_generated",
            "created_at": time.time(),
            "ckpt_path": _get_user_setting(user, "ckpt_path"),
        }

    return {
        "job_id": job_id,
        "smiles": smiles,
        "n_conformers": len(conformers),
        "conformers": conformers,
        "used_openeye": OPENEYE_AVAILABLE,
    }


def _generate_conformers_rdkit(smiles: str, max_confs: int = 10) -> list:
    """Fallback conformer generation using RDKit ETKDG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)

    # Generate conformers
    params = AllChem.ETKDGv3()
    params.maxAttempts = 200
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=max_confs, params=params)
    if len(cids) == 0:
        return []

    # Minimize with MMFF and get energies
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    conformers = []
    for i, cid in enumerate(cids):
        converged, energy = results[i] if i < len(results) else (1, 0.0)
        # MMFF failure (converged == -1): deprioritize by setting energy to None
        if converged == -1:
            energy = None
        # Create single-conformer mol for SDF output
        single = Chem.Mol(mol)
        single.RemoveAllConformers()
        single.AddConformer(mol.GetConformer(cid), assignId=True)
        sdf_block = _mol_to_sdf_string(single)
        props = _compute_all_properties(single)
        conformers.append(
            {
                "idx": i,
                "sdf": sdf_block,
                "energy": round(energy, 4) if energy is not None else None,
                "properties": props,
            }
        )
    return conformers


class SelectConformerRequest(BaseModel):
    conformer_idx: int


@app.post("/select-conformer/{job_id}")
async def select_conformer(job_id: str, body: SelectConformerRequest, request: Request):
    """Select a specific conformer from the generated set for LBDD generation."""
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    conformers = job.get("conformers", [])
    if not conformers:
        raise HTTPException(400, "No conformers generated for this job.")
    if body.conformer_idx < 0 or body.conformer_idx >= len(conformers):
        raise HTTPException(400, f"Invalid conformer index: {body.conformer_idx}")

    selected = conformers[body.conformer_idx]
    sdf_block = selected["sdf"]

    # Save selected conformer as SDF file
    job_dir = _job_upload_dir(request.state.user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = job_dir / "selected_conformer.sdf"
    sdf_path.write_text(sdf_block)

    # Parse with RDKit for atom/bond info
    mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
    if mol is None:
        raise HTTPException(500, "Failed to parse the selected conformer.")

    mol_noh = Chem.RemoveHs(mol)
    smiles_noh = _mol_to_smiles(mol_noh)
    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    job.update(
        ligand_path=str(sdf_path),
        ligand_filename="selected_conformer.sdf",
        ligand_smiles=_mol_to_smiles(mol),
        status="ligand_uploaded",
        selected_conformer_idx=body.conformer_idx,
    )

    return {
        "job_id": job_id,
        "conformer_idx": body.conformer_idx,
        "smiles": _mol_to_smiles(mol),
        "smiles_noH": smiles_noh,
        "has_explicit_hs": has_explicit_hs,
        "sdf_data": sdf_block,
        "atoms": _mol_to_atom_info(mol),
        "bonds": _mol_to_bond_info(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "energy": selected.get("energy"),
    }


# ── Prior-cloud helpers (CPU-only, numpy) ──


def _parse_xyz_center(filepath: str) -> np.ndarray:  # NOSONAR
    """Parse an XYZ or simple-coordinate file and return the center of mass.

    Supports:
      - Standard XYZ (Element X Y Z)
      - Plain coordinates (X Y Z per line)
      - Numpy array-like format ([[ ... ]])

    Returns np.ndarray of shape (3,).
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    with open(filepath, "r", encoding="utf-8") as fh:
        content = fh.read()

    coords: list[list[float]] = []

    if "[" in content and "]" in content:
        numbers = [float(n) for n in _FLOAT_RE.findall(content)]
        if len(numbers) < 3 or len(numbers) % 3 != 0:
            raise ValueError(f"Cannot parse numpy-style XYZ: {filepath}")
        for i in range(0, len(numbers), 3):
            coords.append([numbers[i], numbers[i + 1], numbers[i + 2]])
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # "Element X Y Z"
            if len(parts) >= 4:
                try:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    continue
                except ValueError:
                    pass
            # "X Y Z"
            if len(parts) >= 3:
                try:
                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    continue
                except ValueError:
                    pass

    if len(coords) == 0:
        raise ValueError(f"No coordinates found in {filepath}")

    arr = np.array(coords)
    return arr.mean(axis=0)


def _compute_pocket_com(
    job: Dict[str, Any],
    cutoff: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Thin wrapper: extracts paths from job dict and delegates to shared impl."""
    protein_path = job.get("protein_path")
    ligand_path = job.get("ligand_path")
    if not protein_path or not ligand_path:
        return None
    return _compute_pocket_com_shared(
        protein_path, ligand_path, pocket_cutoff=cutoff if cutoff is not None else 6.0
    )


def _compute_anisotropic_preview_covariance(  # NOSONAR
    job: Dict[str, Any],
    center: np.ndarray,
    has_prior_center: bool,
    pocket_cutoff: float,
    gen_mode: str = "fragment_growing",
    ring_system_index: int = 0,
) -> Optional[np.ndarray]:
    """Compute an anisotropic covariance matrix for the prior cloud preview.

    Mode-specific logic (mirrors interpolate.py _compute_anisotropic_covariance):
    - fragment_growing + prior_center: directional covariance (fragment→growth)
    - core_growing, scaffold_hopping: shape covariance from VARIABLE atoms
    - linker_inpainting, scaffold_elaboration: shape covariance from ALL atoms
    - Other: shape covariance from ALL atoms

    Returns a (3, 3) covariance matrix, or None if computation fails.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    _ = pocket_cutoff  # reserved for future use
    _ANISO_MODES = {
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
        "fragment_growing",
    }
    if gen_mode not in _ANISO_MODES:
        return None

    if not RDKIT_AVAILABLE:
        return None

    lig_path = job.get("ligand_path")
    if not lig_path:
        return None
    mol = _read_ligand_mol(lig_path)
    if mol is None:
        return None
    mol_noh = Chem.RemoveHs(mol)
    if mol_noh.GetNumConformers() == 0 or mol_noh.GetNumAtoms() < 3:
        return None

    conf = mol_noh.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol_noh.GetNumAtoms())]
    )

    if gen_mode == "fragment_growing" and has_prior_center:
        # Directional covariance: elongate along fragment COM → growth center
        source_com = coords.mean(axis=0)
        direction = center - source_com
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return _shape_covariance_np(coords)
        direction = direction / norm

        # Build orthonormal basis
        ref = (
            np.array([1.0, 0.0, 0.0])
            if abs(direction[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        v2 = ref - np.dot(ref, direction) * direction
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        v3 = np.cross(direction, v2)

        rotation = np.column_stack([v2, v3, direction])
        elongation = 2.0
        eigenvalues = np.array([1.0, 1.0, elongation])
        eigenvalues = np.clip(eigenvalues, 0.3, 3.0)
        eigenvalues = eigenvalues * (3.0 / eigenvalues.sum())

        return rotation @ np.diag(eigenvalues) @ rotation.T

    if gen_mode in (
        "core_growing",
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
    ):
        # Shape covariance from VARIABLE atoms only (matching interpolate.py)
        try:
            _mode_mask_fns_aniso = {
                "scaffold_hopping": lambda m: _extract_scaffold_mask(m),
                "scaffold_elaboration": lambda m: _extract_scaffold_elaboration_mask(m),
                "linker_inpainting": lambda m: _extract_linker_mask(m),
                "core_growing": lambda m: ~_extract_core_mask(
                    m, ring_system_index=ring_system_index
                ),
            }
            mask_fn = _mode_mask_fns_aniso.get(gen_mode)
            if mask_fn:
                mask = mask_fn(mol_noh)
                variable_indices = np.nonzero(mask)[0]
                if len(variable_indices) >= 3:
                    return _shape_covariance_np(coords[variable_indices])
        except Exception:
            logger.debug("Anisotropic covariance computation failed", exc_info=True)
        return _shape_covariance_np(coords)

    # fragment_growing (without prior center)
    return _shape_covariance_np(coords)


def _shape_covariance_np(coords: np.ndarray) -> Optional[np.ndarray]:
    """Compute PCA-based shape covariance from atom coordinates (numpy)."""
    if coords.shape[0] < 3:
        return None
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(coords.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Reverse to descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Normalise so tr(Σ) = 3, clamp, re-normalise
    eigenvalues = eigenvalues / (eigenvalues.sum() / 3.0 + 1e-8)
    eigenvalues = np.clip(eigenvalues, 0.3, 3.0)
    eigenvalues = eigenvalues * (3.0 / eigenvalues.sum())

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _compute_ref_ligand_com_shift_server(
    job: Dict[str, Any],
    gen_mode: str,
    ring_system_index: int = 0,
) -> Optional[np.ndarray]:
    """Compute the reference ligand variable fragment CoM shift for preview.

    Returns the CoM of the variable (to-be-generated) atoms, or None if not applicable.
    Mirrors interpolate.py ``_get_ref_com_shift`` logic.
    """
    _APPLICABLE_MODES = {
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
    }
    if gen_mode not in _APPLICABLE_MODES:
        return None
    if not RDKIT_AVAILABLE:
        return None
    lig_path = job.get("ligand_path")
    if not lig_path:
        return None
    mol = _read_ligand_mol(lig_path)
    if mol is None:
        return None
    mol_noh = Chem.RemoveHs(mol)
    if mol_noh.GetNumConformers() == 0 or mol_noh.GetNumAtoms() < 3:
        return None
    conf = mol_noh.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol_noh.GetNumAtoms())]
    )

    try:
        _mode_mask_fns_ref = {
            "scaffold_hopping": lambda m: _extract_scaffold_mask(m),
            "scaffold_elaboration": lambda m: _extract_scaffold_elaboration_mask(m),
            "linker_inpainting": lambda m: _extract_linker_mask(m),
            "core_growing": lambda m: ~_extract_core_mask(
                m, ring_system_index=ring_system_index
            ),
        }
        mask_fn = _mode_mask_fns_ref.get(gen_mode)
        if mask_fn:
            mask = mask_fn(mol_noh)
            variable_indices = np.nonzero(mask)[0]
            if len(variable_indices) > 0:
                return coords[variable_indices].mean(axis=0)
    except Exception:
        logger.debug("Ref ligand CoM shift computation failed", exc_info=True)
    return None


def _compute_prior_cloud_preview(  # NOSONAR
    job: Dict[str, Any],
    grow_size: int,
    pocket_cutoff: float = 6.0,
    anisotropic: bool = False,
    gen_mode: str = "fragment_growing",
    ring_system_index: int = 0,
    ref_ligand_com_prior: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute a preview prior cloud on the CPU (no torch).

    Uses the uploaded prior-center file if available, otherwise falls back
    to the **protein pocket centre of mass** (matching what the FLOWR
    generation pipeline does: the prior is zero-COM in the pocket-COM frame,
    so in the original frame its centre is at the pocket COM).

    Supports anisotropic (mode-specific) sampling and ref ligand CoM shift.

    Returns dict {center, points, n_atoms, has_prior_center, anisotropic, ref_ligand_com_shifted}.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    has_prior_center = "prior_center_path" in job and job["prior_center_path"]

    if has_prior_center:
        center = _parse_xyz_center(job["prior_center_path"])
    else:
        # Pocket COM — matches the FLOWR generation pipeline where the prior
        # is zero-COM in the pocket-COM-shifted frame.
        pocket_com = _compute_pocket_com(job, cutoff=pocket_cutoff)
        if pocket_com is not None:
            center = pocket_com
        else:
            # Fallback: ligand COM, protein atom COM, or zeros
            lig_path = job.get("ligand_path")
            if lig_path and RDKIT_AVAILABLE:
                mol = _read_ligand_mol(lig_path)
                if mol is not None:
                    mol_noh = Chem.RemoveHs(mol)
                    if mol_noh.GetNumConformers() == 0:
                        center = np.zeros(3)
                    else:
                        conf = mol_noh.GetConformer()
                        pts = np.array(
                            [
                                list(conf.GetAtomPosition(i))
                                for i in range(mol_noh.GetNumAtoms())
                            ]
                        )
                        center = pts.mean(axis=0)
                else:
                    center = np.zeros(3)
            elif job.get("protein_path"):
                # Pocket-only mode: compute COM of all protein atoms
                try:
                    coords = []
                    with open(job["protein_path"], encoding="utf-8") as f:
                        for line in f:
                            if line.startswith(("ATOM", "HETATM")):
                                x = float(line[30:38])
                                y = float(line[38:46])
                                z = float(line[46:54])
                                coords.append([x, y, z])
                    if coords:
                        center = np.mean(coords, axis=0)
                    else:
                        center = np.zeros(3)
                except Exception:
                    center = np.zeros(3)
            else:
                center = np.zeros(3)

    # Apply reference ligand CoM shift if applicable
    _ref_com_shifted = False
    if ref_ligand_com_prior:
        ref_com = _compute_ref_ligand_com_shift_server(job, gen_mode, ring_system_index)
        if ref_com is not None:
            center = ref_com
            _ref_com_shifted = True

    rng = np.random.default_rng(seed=seed)

    _aniso_applied = False
    if anisotropic and RDKIT_AVAILABLE:
        # Compute anisotropic covariance for visualization
        covariance = _compute_anisotropic_preview_covariance(
            job,
            center,
            has_prior_center,
            pocket_cutoff,
            gen_mode=gen_mode,
            ring_system_index=ring_system_index,
        )
        if covariance is not None:
            # Symmetrise + jitter for numerical stability
            covariance = (covariance + covariance.T) / 2
            covariance += 1e-6 * np.eye(3)
            L = np.linalg.cholesky(covariance)
            z = rng.standard_normal((grow_size, 3))
            z -= z.mean(axis=0)
            points = (z @ L.T) + center
            _aniso_applied = True
        else:
            noise = rng.standard_normal((grow_size, 3))
            noise -= noise.mean(axis=0)
            points = noise + center
    else:
        noise = rng.standard_normal((grow_size, 3))
        noise -= noise.mean(axis=0)
        points = noise + center

    return {
        "center": {
            "x": round(float(center[0]), 4),
            "y": round(float(center[1]), 4),
            "z": round(float(center[2]), 4),
        },
        "points": [
            {
                "x": round(float(p[0]), 4),
                "y": round(float(p[1]), 4),
                "z": round(float(p[2]), 4),
            }
            for p in points
        ],
        "n_atoms": grow_size,
        "has_prior_center": has_prior_center,
        "anisotropic": _aniso_applied,
        "ref_ligand_com_shifted": _ref_com_shifted,
    }


@app.post("/upload_prior_center/{job_id}")
async def upload_prior_center(job_id: str, request: Request, file: UploadFile):
    """Upload an XYZ file for fragment growing prior center.

    Returns the filename **and** a preview prior cloud so the frontend can
    immediately visualise where new atoms will be grown.
    """
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(413, _ERR_FILE_TOO_LARGE)
    safe_name = _sanitize_filename(file.filename)
    job_dir = _job_upload_dir(request.state.user, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    resolved = save_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve())):
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, _ERR_INVALID_FILE_PATH)

    job["prior_center_path"] = str(save_path)

    # Compute preview cloud
    grow_size = job.get("grow_size", 5)
    _gen_mode = job.get("gen_mode", "fragment_growing")
    _rsi = job.get("ring_system_index", 0)
    _aniso = job.get("anisotropic_prior", False)
    _ref_com = job.get("ref_ligand_com_prior", False)
    try:
        cloud = _compute_prior_cloud_preview(
            job,
            grow_size,
            anisotropic=_aniso,
            gen_mode=_gen_mode,
            ring_system_index=_rsi,
            ref_ligand_com_prior=_ref_com,
        )
    except Exception as exc:
        cloud = None
        logger.warning("Prior cloud preview failed: %s", exc)

    return {"job_id": job_id, "filename": safe_name, "prior_cloud": cloud}


@app.get("/prior-cloud-preview/{job_id}")
async def prior_cloud_preview(  # NOSONAR
    job_id: str,
    request: Request,
    grow_size: int = 5,
    pocket_cutoff: float = 6.0,
    anisotropic: bool = False,
    gen_mode: str = "fragment_growing",
    ring_system_index: int = 0,
    ref_ligand_com_prior: bool = False,
    fixed_atoms: Optional[str] = None,
):
    """Return a prior-cloud preview for visualisation.

    For fragment_growing, uses the explicit ``grow_size`` parameter.
    For other conditional modes, computes the number of atoms to be
    *replaced* via the inpainting mask so the cloud size matches the
    expected generation output.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    # De novo without ligand: use num_heavy_atoms for cloud size, pocket COM for center
    if "ligand_path" not in job:
        if not job.get("denovo_no_ligand"):
            raise HTTPException(400, "No ligand uploaded yet.")
        # Prefer the query-param grow_size (sent by frontend with the
        # user's current num_heavy_atoms value) so the cloud updates
        # dynamically when the user edits the input field.
        eff_size = grow_size or job.get("num_heavy_atoms") or 20
        try:
            cloud = _compute_prior_cloud_preview(
                job,
                eff_size,
                pocket_cutoff=pocket_cutoff,
                anisotropic=False,
                gen_mode="denovo",
                ref_ligand_com_prior=False,
            )
        except Exception as exc:
            raise HTTPException(500, f"Prior cloud computation failed: {exc}")
        return cloud

    # Parse fixed_atoms if provided
    fixed_atom_indices = None
    if fixed_atoms:
        try:
            fixed_atom_indices = [int(x) for x in fixed_atoms.split(",") if x.strip()]
        except ValueError:
            fixed_atom_indices = None

    # For ANY atom-select mode with explicit fixed_atoms, user selection takes priority
    _atom_select_modes = {
        "substructure_inpainting",
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
    }
    if fixed_atom_indices and gen_mode in _atom_select_modes:
        effective_grow_size = len(fixed_atom_indices)
    else:
        effective_grow_size = grow_size

        # Determine the correct cloud size based on generation mode (auto-detect from mask)
        # Only used as fallback when user hasn't manually selected atoms
        _mode_mask_fns = {
            "scaffold_hopping": lambda mol: _extract_scaffold_mask(mol),
            "scaffold_elaboration": lambda mol: _extract_scaffold_elaboration_mask(mol),
            "linker_inpainting": lambda mol: _extract_linker_mask(mol),
            "core_growing": lambda mol: ~_extract_core_mask(
                mol, ring_system_index=ring_system_index
            ),
        }
        if gen_mode in _mode_mask_fns and RDKIT_AVAILABLE:
            try:
                mol = _read_ligand_mol(job.get("ligand_path", ""))
                if mol is not None:
                    mol_noh = Chem.RemoveHs(mol)
                    mask = _mode_mask_fns[gen_mode](mol_noh)
                    n_replaced = int(np.sum(mask))
                    if n_replaced > 0:
                        effective_grow_size = n_replaced
            except Exception:
                logger.debug("Auto-detect mask for cloud size failed", exc_info=True)

    # Store grow_size for later upload_prior_center calls
    job["grow_size"] = effective_grow_size
    try:
        cloud = _compute_prior_cloud_preview(
            job,
            effective_grow_size,
            pocket_cutoff=pocket_cutoff,
            anisotropic=anisotropic,
            gen_mode=gen_mode,
            ring_system_index=ring_system_index,
            ref_ligand_com_prior=ref_ligand_com_prior,
        )
    except Exception as exc:
        raise HTTPException(500, f"Prior cloud computation failed: {exc}")
    return cloud


# ── Generation (proxied to worker) ──


@app.post("/generate")
async def generate(request: Request, gen_req: GenerationRequest):  # NOSONAR
    """Start generation by delegating to the GPU worker."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    user = request.state.user
    _check_rate_limit(user, "generate")
    job_id = gen_req.job_id
    _validate_job_id(job_id)
    # Validate string params don't contain path separators
    if gen_req.gen_mode and ("/" in gen_req.gen_mode or "\\" in gen_req.gen_mode):
        raise HTTPException(400, "Invalid gen_mode")
    if gen_req.prior_center_filename and (
        "/" in gen_req.prior_center_filename
        or "\\" in gen_req.prior_center_filename
        or "\0" in gen_req.prior_center_filename
    ):
        raise HTTPException(400, "Invalid prior_center_filename")
    job = _get_user_job(job_id, request)
    # Allow missing ligand when num_heavy_atoms is provided (de novo pocket-only / scratch)
    has_ligand = "ligand_path" in job
    has_denovo = gen_req.num_heavy_atoms is not None and gen_req.num_heavy_atoms > 0
    if not has_ligand and not has_denovo:
        raise HTTPException(400, "No ligand uploaded and no num_heavy_atoms specified.")
    # LBDD does not require a protein upload
    workflow = (
        gen_req.workflow_type
        or job.get("workflow_type")
        or _get_user_setting(user, "workflow_type", "sbdd")
        or "sbdd"
    )
    if workflow != "lbdd" and "protein_path" not in job:
        raise HTTPException(400, _ERR_NO_PROTEIN)
    if gen_req.gen_mode not in _VALID_GEN_MODES:
        raise HTTPException(
            400,
            f"Invalid gen_mode '{gen_req.gen_mode}'. "
            f"Must be one of: {', '.join(sorted(_VALID_GEN_MODES))}",
        )
    if job.get("status") in ("generating", "allocating_gpu", "starting"):
        raise HTTPException(409, "Generation already in progress for this job.")

    # Clear stale error/warnings from any previous generation attempt
    job.pop("error", None)
    job.pop("warnings", None)

    # Clear stale results/progress from previous generation or AL round
    job.pop("results", None)
    job.pop("prior_cloud", None)
    job.pop("metrics", None)
    job.pop("n_generated", None)
    job.pop("elapsed_time", None)
    job.pop("mode", None)
    job.pop("used_optimization", None)
    job["progress"] = 0

    # Store gen settings in job for summary
    job.update(
        gen_mode=gen_req.gen_mode,
        fixed_atoms=gen_req.fixed_atoms,
        n_samples=gen_req.n_samples,
        batch_size=gen_req.batch_size,
        integration_steps=gen_req.integration_steps,
    )

    req = {
        "gen_mode": gen_req.gen_mode,
        "fixed_atoms": gen_req.fixed_atoms,
        "n_samples": gen_req.n_samples,
        "batch_size": gen_req.batch_size,
        "integration_steps": gen_req.integration_steps,
        "pocket_cutoff": gen_req.pocket_cutoff,
        "coord_noise_scale": gen_req.coord_noise_scale,
        "grow_size": gen_req.grow_size,
        "prior_center_filename": gen_req.prior_center_filename,
        "prior_center_coords": gen_req.prior_center_coords,
        "filter_valid_unique": gen_req.filter_valid_unique,
        "filter_cond_substructure": gen_req.filter_cond_substructure,
        "filter_diversity": gen_req.filter_diversity,
        "diversity_threshold": gen_req.diversity_threshold,
        "sample_mol_sizes": gen_req.sample_mol_sizes,
        "filter_pb_valid": gen_req.filter_pb_valid,
        "calculate_pb_valid": gen_req.calculate_pb_valid,
        "calculate_strain_energies": gen_req.calculate_strain_energies,
        "optimize_gen_ligs": gen_req.optimize_gen_ligs,
        "optimize_gen_ligs_hs": gen_req.optimize_gen_ligs_hs,
        "anisotropic_prior": gen_req.anisotropic_prior,
        "ring_system_index": gen_req.ring_system_index,
        "ref_ligand_com_prior": gen_req.ref_ligand_com_prior,
        "workflow_type": workflow,
        "optimize_method": gen_req.optimize_method,
        "sample_n_molecules_per_mol": gen_req.sample_n_molecules_per_mol,
        "seed": gen_req.seed,
        "num_heavy_atoms": gen_req.num_heavy_atoms,
        "property_filter": gen_req.property_filter,
        "adme_filter": gen_req.adme_filter,
    }

    # Store workflow_type and checkpoint in job for proxy branch logic
    # Prefer finetuned checkpoint from active learning if available
    resolved_ckpt = (
        gen_req.ckpt_path
        or job.get("finetuned_ckpt_path")
        or job.get("ckpt_path")
        or _get_user_setting(user, "ckpt_path")
    )
    job["workflow_type"] = workflow
    if resolved_ckpt:
        job["ckpt_path"] = resolved_ckpt

    # Run the full lifecycle in a background thread:
    # allocate GPU → load model → generate → release GPU
    job["status"] = "starting"
    thread = threading.Thread(target=_proxy_generation, args=(job_id, req), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "allocating_gpu"}


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str, request: Request):
    """Request cancellation of a running generation or finetuning job."""
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    if job.get("status") not in (
        "starting",
        "allocating_gpu",
        "generating",
        "loading_model",
        "finetuning",
    ):
        raise HTTPException(409, "Job is not currently running.")
    job["cancel_requested_at"] = time.time()
    job["cancelled"] = True
    job["status"] = "cancelled"
    job["error"] = _ERR_CANCELLED

    # Forward cancellation to the worker (best-effort)
    # For AL jobs the worker uses al_{job_id} as the job key
    is_al = job.get("al_indices") is not None
    worker_job_id = f"al_{job_id}" if is_al else job_id
    try:
        worker_url = _get_worker_url()
        await asyncio.to_thread(
            http_requests.post, f"{worker_url}/cancel/{worker_job_id}", timeout=5
        )
    except Exception:
        logger.debug("Failed to forward cancellation to worker", exc_info=True)

    return {"job_id": job_id, "status": "cancelled"}


@app.post("/clear-history/{job_id}")
async def clear_history(job_id: str, request: Request):
    """Clear all accumulated generation history and reset iteration counter.

    Called when the user confirms deletion of all generated ligands across
    all rounds.  This removes the SMILES diversity history, the per-round
    result archive, and any cached visualization data so the next generation
    starts fresh.
    """
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    job.pop("generated_history", None)
    job.pop("all_results", None)
    job.pop("iteration_idx", None)
    job.pop("results", None)
    job.pop("results_full", None)
    _invalidate_viz_caches(job)
    # Clear original ligand tracking (reset on full clear)
    for _k in list(job):  # NOSONAR
        if _k.startswith("original_ligand_"):
            del job[_k]
    return {"job_id": job_id, "status": "cleared"}


@app.get("/job/{job_id}")
async def get_job(job_id: str, request: Request):
    """Poll job progress. The background proxy thread updates JOBS directly."""
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    resp: Dict[str, Any] = {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "health_check_type": job.get("health_check_type"),
        "health_check_advice": job.get("health_check_advice"),
    }
    if job.get("status") == "completed":
        resp.update(
            n_generated=job.get("n_generated", 0),
            elapsed_time=job.get("elapsed_time"),
            mode=job.get("mode", "unknown"),
            results=job.get("results", []),
            metrics=job.get("metrics", []),
            used_optimization=job.get("used_optimization", False),
            prior_cloud=job.get("prior_cloud"),
            warnings=job.get("warnings", []),
        )
    return resp


@app.get("/job/{job_id}/results")
async def get_job_results(job_id: str, request: Request):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    return {"results": job.get("results", []), "mode": job.get("mode", "unknown")}


# ═══════════════════════════════════════════════════════════════════════════
#  ACTIVE LEARNING
# ═══════════════════════════════════════════════════════════════════════════


def _proxy_active_learning(job_id: str, req: dict):  # NOSONAR
    """Proxy AL finetuning to the GPU worker. Background thread."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "allocating_gpu" if WORKER_MODE == "slurm" else "finetuning"
    job["progress"] = 1
    did_increment = False

    try:
        worker_url = _ensure_worker_running(job)

        with _worker_lock:
            _worker_state["active_jobs"] = _worker_state.get("active_jobs", 0) + 1
        did_increment = True

        job["status"] = "finetuning"
        job["progress"] = 2

        # Collect SDF strings from selected result indices (current round)
        results = job.get("results", [])
        indices = req["indices"]
        ligand_sdf_strings = []
        for idx in indices:
            if 0 <= idx < len(results):
                sdf = results[idx].get("sdf", "")
                if sdf:
                    ligand_sdf_strings.append(sdf)

        # Optionally include all ligands from selected previous rounds
        prev_iters = req.get("prev_round_iterations") or []
        if prev_iters:
            prev_iter_set = set(prev_iters)
            for round_entry in job.get("all_results", []):
                if round_entry.get("iteration") in prev_iter_set:
                    for r in round_entry.get("results", []):
                        sdf = r.get("sdf", "")
                        if sdf:
                            ligand_sdf_strings.append(sdf)

        if not ligand_sdf_strings:
            raise RuntimeError("No valid ligand SDF data found for selected indices.")

        # Determine checkpoint (use explicit user selection, finetuned, or base)
        explicit_ckpt = req.get("ckpt_path")
        if explicit_ckpt:
            # Validate: must be under CKPTS_DIR to prevent path traversal
            ep = Path(explicit_ckpt).resolve()
            if not ep.is_relative_to(CKPTS_DIR.resolve()):
                raise RuntimeError("Invalid checkpoint path")
            if not ep.is_file():
                raise RuntimeError("Checkpoint file not found")
            if ep.suffix != ".ckpt":
                raise RuntimeError("File must have .ckpt extension")

        if explicit_ckpt and explicit_ckpt != job.get("ckpt_path"):
            # User selected a different base model — don't continue from previous finetune
            ckpt = explicit_ckpt
        else:
            # Standard chain: continue from finetuned (round 2+), else base, else user setting
            ckpt = (
                job.get("finetuned_ckpt_path")
                or explicit_ckpt
                or job.get("ckpt_path")
                or _get_user_setting(job.get("user", ""), "ckpt_path")
            )

        worker_payload = {
            "job_id": f"al_{job_id}",
            "protein_url": _worker_file_url(job_id, Path(job["protein_path"]).name),
            "protein_filename": Path(job["protein_path"]).name,
            "ligand_sdf_strings": ligand_sdf_strings,
            "ckpt_path": ckpt,
            "finetuned_ckpt_url": (
                _worker_file_url(job_id, "finetuned_last.ckpt")
                if job.get("finetuned_ckpt_path")
                and not (explicit_ckpt and explicit_ckpt != job.get("ckpt_path"))
                else None
            ),
            "lora_rank": req.get("lora_rank", 16),
            "lora_alpha": req.get("lora_alpha", 32),
            "lr": req.get("lr", 5e-4),
            "batch_cost": req.get("batch_cost", 4),
            "acc_batches": req.get("acc_batches"),
            "epochs": req.get("epochs"),
            "pocket_cutoff": req.get("pocket_cutoff", 7.0),
        }

        resp = http_requests.post(
            f"{worker_url}/active-learning",
            json=worker_payload,
            timeout=30,
        )
        resp.raise_for_status()

        job["progress"] = 5

        # Poll worker for AL job progress
        al_job_id = f"al_{job_id}"
        al_timeout = int(os.environ.get("FLOWR_AL_TIMEOUT", "3600"))
        poll_deadline = time.time() + al_timeout
        while time.time() < poll_deadline:
            if job.get("cancelled"):
                job["al_status_message"] = "Finetuning cancelled by user."
                break
            time.sleep(3)
            try:
                poll_resp = http_requests.get(
                    f"{worker_url}/al-job/{al_job_id}", timeout=10
                )
                if poll_resp.status_code != 200:
                    continue
                w_data = poll_resp.json()
            except Exception:
                continue

            w_status = w_data.get("status")
            w_progress = w_data.get("progress", 0)
            w_phase = w_data.get("al_phase", "preparing")
            # When phase changes (preparing→training), progress resets to 0
            cur_phase = job.get("al_phase", "preparing")
            if w_phase == cur_phase:
                job["progress"] = max(job.get("progress", 0), w_progress)
            else:
                job["progress"] = w_progress
            job["al_status_message"] = w_data.get("status_message", "")
            job["al_phase"] = w_phase

            if w_status == "completed":
                # Download the checkpoint from the worker to server-side storage
                ckpt_filename = "finetuned_last.ckpt"
                user = job.get("user", "_unknown")
                job_dir = _job_upload_dir(user, job_id)
                job_dir.mkdir(parents=True, exist_ok=True)
                local_ckpt_path = job_dir / ckpt_filename

                ckpt_resp = http_requests.get(
                    f"{worker_url}/al-ckpt/{al_job_id}",
                    timeout=600,
                    stream=True,
                )
                ckpt_resp.raise_for_status()
                with open(local_ckpt_path, "wb") as f:
                    for chunk in ckpt_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                ckpt_url = _worker_file_url(job_id, ckpt_filename)
                logger.info(
                    "[AL] Checkpoint saved to server: %s (%.1f MB)",
                    local_ckpt_path,
                    local_ckpt_path.stat().st_size / 1024 / 1024,
                )

                job.update(
                    status="completed",
                    progress=100,
                    finetuned_ckpt_path=str(local_ckpt_path),
                    finetuned_ckpt_url=ckpt_url,
                    al_n_ligands=w_data.get("n_ligands", 0),
                    al_status_message="LoRA finetuning complete!",
                )
                break
            elif w_status == "failed":
                raise RuntimeError(
                    w_data.get("error", "Active learning finetuning failed on worker.")
                )
        else:
            raise RuntimeError("Active learning finetuning timed out.")

    except Exception as exc:
        logger.exception("Active learning proxy failed for job %s", job_id)
        job.update(status="failed", progress=0, error=str(exc))
    finally:
        if did_increment:
            with _worker_lock:
                _worker_state["active_jobs"] = max(
                    0, _worker_state.get("active_jobs", 0) - 1
                )
                should_release = _worker_state["active_jobs"] == 0
            if should_release:
                _release_worker()


@app.post("/active-learning/{job_id}")
async def active_learning(
    job_id: str, al_req: ActiveLearningRequest, request: Request
):  # NOSONAR
    """Start active learning LoRA finetuning on user-selected ligands."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    _validate_job_id(job_id)
    _check_rate_limit(request.state.user, "active-learning")
    job = _get_user_job(job_id, request)

    if not job.get("results"):
        raise HTTPException(400, "No generation results to finetune on.")
    if not al_req.indices:
        raise HTTPException(400, "No ligand indices selected.")
    if job.get("workflow_type") == "lbdd":
        raise HTTPException(400, "Active learning is only supported for SBDD workflow.")
    if "protein_path" not in job:
        raise HTTPException(400, _ERR_NO_PROTEIN)
    if job.get("status") in ("generating", "allocating_gpu", "finetuning"):
        raise HTTPException(409, "A task is already in progress.")

    # Clear stale errors and progress from any previous finetuning run
    job.pop("error", None)
    job["al_phase"] = "preparing"
    job["al_status_message"] = "Starting LoRA finetuning..."
    job["progress"] = 0
    job["al_indices"] = al_req.indices

    req = {
        "indices": al_req.indices,
        "prev_round_iterations": al_req.prev_round_iterations or [],
        "lora_rank": al_req.lora_rank,
        "lora_alpha": al_req.lora_alpha,
        "lr": al_req.lr,
        "batch_cost": al_req.batch_cost,
        "acc_batches": al_req.acc_batches,
        "epochs": al_req.epochs,
        "ckpt_path": al_req.ckpt_path,
    }

    job["status"] = "starting"
    thread = threading.Thread(
        target=_proxy_active_learning, args=(job_id, req), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "finetuning"}


@app.get("/al-status/{job_id}")
async def al_status(job_id: str, request: Request):
    """Poll active learning progress."""
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "status_message": job.get("al_status_message", ""),
        "finetuned_ckpt_path": job.get("finetuned_ckpt_path"),
        "al_phase": job.get("al_phase", "preparing"),
    }


@app.get("/mol-image")
async def mol_image(smiles: str, width: int = 300, height: int = 200):
    """Return an SVG image of a molecule from its SMILES string."""
    if not RDKIT_AVAILABLE:
        raise HTTPException(500, "RDKit not available")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return Response(content=svg, media_type="image/svg+xml")


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE LIGANDS
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/save-ligand/{job_id}/{ligand_idx}")
async def save_ligand(job_id: str, ligand_idx: int, request: Request):
    """Save a single generated ligand as SDF."""
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    results = job.get("results", [])
    if ligand_idx < 0 or ligand_idx >= len(results):
        raise HTTPException(
            400, f"Ligand index {ligand_idx} out of range (0-{len(results) - 1})."
        )

    sdf_data = results[ligand_idx].get("sdf", "")
    if not sdf_data:
        raise HTTPException(400, "No SDF data for this ligand.")

    if job.get("workflow_type") == "lbdd":
        base_name = Path(job.get("ligand_filename", "molecule")).stem
    else:
        base_name = Path(job.get("protein_filename", "unknown")).stem
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Include round number in filename to avoid overwrites across rounds
    current_round = max(job.get("iteration_idx", 1), 1)
    round_tag = f"_round{current_round}" if current_round > 1 else ""
    out_path = output_dir / f"{base_name}_{job_id}{round_tag}_ligand_{ligand_idx}.sdf"
    out_path.write_text(sdf_data)

    return {"saved": True, "path": str(out_path), "filename": out_path.name}


@app.post("/save-all-ligands/{job_id}")
async def save_all_ligands(job_id: str, request: Request):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No results to save.")

    if job.get("workflow_type") == "lbdd":
        base_name = Path(job.get("ligand_filename", "molecule")).stem
    else:
        base_name = Path(job.get("protein_filename", "unknown")).stem
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Include round number in filename to avoid overwrites across rounds
    current_round = max(job.get("iteration_idx", 1), 1)
    round_tag = f"_round{current_round}" if current_round > 1 else ""

    saved = []
    for idx, r in enumerate(results):
        sdf_data = r.get("sdf", "")
        if sdf_data:
            out_path = output_dir / f"{base_name}_{job_id}{round_tag}_ligand_{idx}.sdf"
            out_path.write_text(sdf_data)
            saved.append({"path": str(out_path), "filename": out_path.name})

    return {"saved_count": len(saved), "files": saved, "output_dir": str(output_dir)}


class SaveSelectedRequest(BaseModel):
    indices: List[int]


@app.post("/save-selected-ligands/{job_id}")
async def save_selected_ligands(
    job_id: str, body: SaveSelectedRequest, request: Request
):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No results to save.")

    if job.get("workflow_type") == "lbdd":
        base_name = Path(job.get("ligand_filename", "molecule")).stem
    else:
        base_name = Path(job.get("protein_filename", "unknown")).stem
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Include round number in filename to avoid overwrites across rounds
    current_round = max(job.get("iteration_idx", 1), 1)
    round_tag = f"_round{current_round}" if current_round > 1 else ""

    saved = []
    for idx in body.indices:
        if 0 <= idx < len(results):
            sdf_data = results[idx].get("sdf", "")
            if sdf_data:
                out_path = (
                    output_dir / f"{base_name}_{job_id}{round_tag}_ligand_{idx}.sdf"
                )
                out_path.write_text(sdf_data)
                saved.append({"path": str(out_path), "filename": out_path.name})

    return {"saved_count": len(saved), "files": saved, "output_dir": str(output_dir)}


# ═══════════════════════════════════════════════════════════════════════════
#  REFERENCE LIGAND HYDROGENS
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/ligand-remove-hs/{job_id}")
async def ligand_remove_hs(job_id: str, request: Request):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    ligand_path = job.get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, _ERR_NO_LIGAND)

    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, _ERR_CANNOT_PARSE_LIGAND)

    mol_noh = Chem.RemoveHs(mol)
    smiles_noh = _mol_to_smiles(mol_noh)

    return {
        "sdf_data": _mol_to_sdf_string(mol_noh),
        "atoms": _mol_to_atom_info(mol_noh),
        "bonds": _mol_to_bond_info(mol_noh),
        "smiles": smiles_noh,
        "smiles_noH": smiles_noh,
        "num_atoms": mol_noh.GetNumAtoms(),
        "num_heavy_atoms": mol_noh.GetNumHeavyAtoms(),
    }


@app.post("/ligand-add-hs/{job_id}")
async def ligand_add_hs(job_id: str, request: Request):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    ligand_path = job.get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, _ERR_NO_LIGAND)

    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, _ERR_CANNOT_PARSE_LIGAND)

    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    if has_explicit_hs:
        mol_with_h = mol
    else:
        mol_with_h = Chem.AddHs(mol, addCoords=True)

    smiles_with_h = _mol_to_smiles(mol_with_h)
    mol_noh_for_smiles = Chem.RemoveHs(mol_with_h)
    smiles_noh = _mol_to_smiles(mol_noh_for_smiles)

    return {
        "sdf_data": _mol_to_sdf_string(mol_with_h),
        "atoms": _mol_to_atom_info(mol_with_h),
        "bonds": _mol_to_bond_info(mol_with_h),
        "smiles": smiles_with_h,
        "smiles_noH": smiles_noh,
        "num_atoms": mol_with_h.GetNumAtoms(),
        "num_heavy_atoms": mol_with_h.GetNumHeavyAtoms(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  INTERACTION DIAGRAMS (2D – OpenEye)
# ═══════════════════════════════════════════════════════════════════════════


def _generate_interaction_svg_oe(
    protein_path: str, ligand_path: str, width: float = 2400, height: float = 1800
) -> Optional[str]:
    """Generate 2D interaction diagram SVG using OpenEye."""
    if not OPENEYE_AVAILABLE:
        return None

    protein = oechem.OEGraphMol()
    ligand = oechem.OEGraphMol()

    ifs = oechem.oemolistream()
    if not ifs.open(protein_path):
        return None
    if not oechem.OEReadMolecule(ifs, protein):
        return None

    ifs2 = oechem.oemolistream()
    if not ifs2.open(ligand_path):
        return None
    if not oechem.OEReadMolecule(ifs2, ligand):
        return None

    if not oechem.OEHasResidues(protein):
        oechem.OEPerceiveResidues(protein, oechem.OEPreserveResInfo_All)

    image = oedepict.OEImage(width, height)

    legend_width = width * 0.20
    content_width = width * 0.80
    main_frame = oedepict.OEImageFrame(
        image, content_width, height, oedepict.OE2DPoint(legend_width, 0.0)
    )
    legend_frame = oedepict.OEImageFrame(
        image, legend_width, height, oedepict.OE2DPoint(0.0, 0.0)
    )

    opts = oegrapheme.OE2DActiveSiteDisplayOptions(content_width, height)
    opts.SetRenderInteractiveLegend(False)

    asite = oechem.OEInteractionHintContainer(protein, ligand)
    if not asite.IsValid():
        return None
    asite.SetTitle("")
    oechem.OEPerceiveInteractionHints(asite)

    try:
        oegrapheme.OEPrepareActiveSiteDepiction(asite)
        adisp = oegrapheme.OE2DActiveSiteDisplay(asite, opts)
        oegrapheme.OERenderActiveSite(main_frame, adisp)
    except Exception as exc:
        logger.warning("OpenEye active-site depiction failed: %s", exc)
        return None

    lopts = oegrapheme.OE2DActiveSiteLegendDisplayOptions(18, 1)
    oegrapheme.OEDrawActiveSiteLegend(legend_frame, adisp, lopts)

    oedepict.OEDrawCurvedBorder(image, oedepict.OELightGreyPen, 10.0)

    fd, tmp_path = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        oedepict.OEWriteImage(tmp_path, image)
        with open(tmp_path, "r", encoding="utf-8") as f:
            svg = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return svg


@app.get("/interaction-diagram/{job_id}")
async def interaction_diagram(job_id: str, request: Request, ligand_idx: int = -1):
    if not OPENEYE_AVAILABLE:
        raise HTTPException(
            501,
            "OpenEye not available. Install openeye-toolkits and set the "
            "OE_LICENSE environment variable to a valid license file for "
            "2D interaction diagrams.",
        )

    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    protein_path = job.get("protein_path")
    if not protein_path:
        raise HTTPException(400, _ERR_NO_PROTEIN)

    if ligand_idx < 0:
        ligand_path = job.get("ligand_path")
        if not ligand_path:
            raise HTTPException(400, "No reference ligand uploaded.")
    else:
        results = job.get("results", [])
        if ligand_idx >= len(results):
            raise HTTPException(400, _ERR_LIGAND_IDX_RANGE)
        sdf_data = results[ligand_idx].get("sdf", "")
        if not sdf_data:
            raise HTTPException(400, "No SDF data.")
        ligand_path = str(
            _job_upload_dir(request.state.user, job_id) / f"gen_lig_{ligand_idx}.sdf"
        )
        Path(ligand_path).write_text(sdf_data)

    tmp_ligand = ligand_idx >= 0
    try:
        svg = _generate_interaction_svg_oe(protein_path, ligand_path)
        if svg is None:
            raise HTTPException(500, "Failed to generate interaction diagram.")
        return Response(content=svg, media_type="image/svg+xml")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Interaction diagram failed: {exc}")
    finally:
        if tmp_ligand and Path(ligand_path).exists():
            os.unlink(ligand_path)


# ═══════════════════════════════════════════════════════════════════════════
#  3D INTERACTION COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/compute-interactions/{job_id}")
async def compute_interactions_endpoint(
    job_id: str, request: Request, ligand_idx: int = -1
):
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, _ERR_RDKIT_UNAVAILABLE)
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    protein_path = job.get("protein_path")
    if not protein_path:
        raise HTTPException(400, _ERR_NO_PROTEIN)

    if ligand_idx < 0:
        ligand_path = job.get("ligand_path")
        if not ligand_path:
            raise HTTPException(400, "No reference ligand.")
        mol = _read_ligand_mol(ligand_path)
    else:
        results = job.get("results", [])
        if ligand_idx >= len(results):
            raise HTTPException(400, _ERR_LIGAND_IDX_RANGE)
        sdf_data = results[ligand_idx].get("sdf", "")
        if not sdf_data:
            raise HTTPException(400, "No SDF data.")
        mol = Chem.MolFromMolBlock(sdf_data, removeHs=True)

    if mol is None:
        raise HTTPException(400, _ERR_CANNOT_PARSE_LIGAND)

    interactions = _compute_interactions(protein_path, mol)
    return {"interactions": interactions, "count": len(interactions)}


# ═══════════════════════════════════════════════════════════════════════════
#  INPAINTING MASK COMPUTATION (CPU-only, no torch)
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/ring-systems/{job_id}")
async def get_ring_systems(job_id: str, request: Request):
    """Return the number of ring systems in the uploaded ligand."""
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, _ERR_RDKIT_UNAVAILABLE)
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    ligand_path = job.get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, _ERR_NO_LIGAND)
    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, _ERR_CANNOT_PARSE_LIGAND)
    mol_noh = Chem.RemoveHs(mol)
    n = _get_num_ring_systems(mol_noh)
    return {"num_ring_systems": n}


@app.get("/inpainting-mask/{job_id}")
async def compute_inpainting_mask_endpoint(  # NOSONAR
    job_id: str,
    request: Request,
    mode: str = "scaffold_hopping",
    ring_system_index: int = 0,
):
    """Return atom indices that will be REPLACED for a given inpainting mode.

    This is a pure RDKit + numpy implementation — no PyTorch required.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, "RDKit not available for mask computation.")

    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    ligand_path = job.get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, _ERR_NO_LIGAND)

    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, _ERR_CANNOT_PARSE_LIGAND)

    mol_noh = Chem.RemoveHs(mol)

    try:
        if mode == "scaffold_hopping":
            replaced_mask = _extract_scaffold_mask(
                mol_noh
            )  # scaffold atoms are replaced
        elif mode == "scaffold_elaboration":
            replaced_mask = _extract_scaffold_elaboration_mask(mol_noh)
        elif mode == "linker_inpainting":
            replaced_mask = _extract_linker_mask(mol_noh)  # linker atoms are replaced
        elif mode == "core_growing":
            core_mask = _extract_core_mask(mol_noh, ring_system_index=ring_system_index)
            replaced_mask = ~core_mask  # non-core atoms are replaced
        elif mode == "fragment_growing":
            replaced_mask = np.zeros(mol_noh.GetNumAtoms(), dtype=bool)
        else:
            return {"replaced": [], "fixed": [], "mode": mode}

        replaced_heavy = np.nonzero(replaced_mask)[0].tolist()
        fixed_heavy = np.nonzero(~replaced_mask)[0].tolist()

        # Map heavy-atom indices → H-inclusive indices
        h_replaced: List[int] = []
        h_fixed: List[int] = []
        heavy_idx = 0
        h_assignment: Dict[int, str] = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                if heavy_idx in replaced_heavy:
                    h_replaced.append(atom.GetIdx())
                    for nbr in atom.GetNeighbors():
                        if nbr.GetAtomicNum() == 1:
                            h_assignment[nbr.GetIdx()] = "replaced"
                elif heavy_idx in fixed_heavy:
                    h_fixed.append(atom.GetIdx())
                    for nbr in atom.GetNeighbors():
                        if nbr.GetAtomicNum() == 1:
                            h_assignment[nbr.GetIdx()] = "fixed"
                heavy_idx += 1
        for h_idx, assignment in h_assignment.items():
            if assignment == "replaced":
                h_replaced.append(h_idx)
            else:
                h_fixed.append(h_idx)

        return {"replaced": h_replaced, "fixed": h_fixed, "mode": mode}

    except Exception as exc:
        raise HTTPException(500, f"Mask computation failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  CHEMICAL SPACE & PROPERTY SPACE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/chemical-space/{job_id}")
async def chemical_space(  # NOSONAR
    job_id: str, request: Request, method: str = "pca", perplexity: int = 30
):
    """Compute 2D projection of Morgan fingerprints for generated ligands + reference.

    Returns coordinates for scatter plot (PCA, t-SNE, or UMAP) plus properties.
    Uses all accumulated results across iterations for multi-round visualisation.
    Results are cached per job_id + method to avoid recomputation.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, _ERR_RDKIT_UNAVAILABLE)
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    # Use all accumulated results if available, otherwise fall back to latest
    all_results = job.get("all_results", [])
    latest_results = job.get("results", [])
    if not all_results and not latest_results:
        raise HTTPException(400, _ERR_NO_GENERATED)

    method = method.lower()
    if method not in ("pca", "tsne", "umap"):
        raise HTTPException(400, "Method must be 'pca', 'tsne', or 'umap'.")

    # Check cache
    cache_key = f"chemspace_{method}_{perplexity}"
    cached = job.get(cache_key)
    if cached is not None:
        return cached

    # Reference ligand fingerprint + properties
    ref_mol = _read_ligand_mol(job.get("ligand_path", ""))
    if ref_mol is None:
        raise HTTPException(400, "Cannot parse reference ligand.")
    ref_mol_noh = Chem.RemoveHs(ref_mol)
    ref_fp = _mol_to_morgan_fp(ref_mol_noh)
    if ref_fp is None:
        raise HTTPException(500, "Failed to compute reference fingerprint.")
    ref_props = _compute_all_properties(ref_mol_noh)

    # Original ligand (if reference was swapped)
    orig_fp = None
    orig_props = None
    has_original = "original_ligand_path" in job
    if has_original:
        orig_mol = _read_ligand_mol(job["original_ligand_path"])
        if orig_mol is not None:
            orig_mol_noh = Chem.RemoveHs(orig_mol)
            orig_fp = _mol_to_morgan_fp(orig_mol_noh)
            orig_props = job.get(
                "original_ligand_properties"
            ) or _compute_all_properties(orig_mol_noh)

    # Flatten all iterations into a single list with iteration tags
    # Order: [reference, (original if present), generated...]
    fps = [ref_fp]
    orig_offset = 0
    if orig_fp is not None:
        fps.append(orig_fp)
        orig_offset = 1
    gen_data = []
    if all_results:
        global_idx = 0
        for round_entry in all_results:
            iteration = round_entry.get("iteration", 0)
            for local_idx, r in enumerate(round_entry.get("results", [])):
                sdf = r.get("sdf_no_hs") or r.get("sdf", "")
                mol = Chem.MolFromMolBlock(sdf, removeHs=True) if sdf else None
                if mol is None:
                    global_idx += 1
                    continue
                fp = _mol_to_morgan_fp(mol)
                if fp is None:
                    global_idx += 1
                    continue
                fps.append(fp)
                props = r.get("properties") or _compute_all_properties(mol)
                gen_data.append(
                    {
                        "idx": global_idx,
                        "local_idx": local_idx,
                        "iteration": iteration,
                        "smiles": r.get("smiles", ""),
                        "properties": props,
                    }
                )
                global_idx += 1
    else:
        # Fallback: single-round results (no iteration info yet)
        for i, r in enumerate(latest_results):
            sdf = r.get("sdf_no_hs") or r.get("sdf", "")
            mol = Chem.MolFromMolBlock(sdf, removeHs=True) if sdf else None
            if mol is None:
                continue
            fp = _mol_to_morgan_fp(mol)
            if fp is None:
                continue
            fps.append(fp)
            props = r.get("properties") or _compute_all_properties(mol)
            gen_data.append(
                {
                    "idx": i,
                    "local_idx": i,
                    "iteration": 0,
                    "smiles": r.get("smiles", ""),
                    "properties": props,
                }
            )

    if len(fps) < 2:
        raise HTTPException(400, "Not enough valid molecules for projection.")

    # Dimensionality reduction
    X = np.array(fps, dtype=np.float32)
    n_samples = X.shape[0]

    try:
        if method == "pca":
            if not SKLEARN_AVAILABLE:
                raise HTTPException(501, "scikit-learn not installed on server.")
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(X)
        elif method == "tsne":
            if not SKLEARN_AVAILABLE:
                raise HTTPException(501, "scikit-learn not installed on server.")
            eff_perp = min(perplexity, max(1, n_samples - 1))
            reducer = TSNE(
                n_components=2, perplexity=eff_perp, random_state=42, max_iter=1000
            )
            coords = reducer.fit_transform(X)
        elif method == "umap":
            if not UMAP_AVAILABLE:
                raise HTTPException(501, "umap-learn not installed on server.")
            n_neighbors = min(15, n_samples - 1)
            if n_neighbors < 2:
                raise HTTPException(
                    400,
                    "Need at least 3 molecules (1 reference + 2 generated) for UMAP.",
                )
            reducer = umap.UMAP(
                n_components=2, n_neighbors=n_neighbors, random_state=42
            )
            coords = reducer.fit_transform(X)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Dimensionality reduction failed: {exc}")

    # Build response
    ref_coord = coords[0]
    response = {
        "method": method,
        "reference": {
            "x": round(float(ref_coord[0]), 4),
            "y": round(float(ref_coord[1]), 4),
            "properties": ref_props,
        },
        "ligands": [],
    }

    # Original ligand coordinates (if present)
    if orig_fp is not None and orig_props is not None:
        orig_coord = coords[1]
        response["original"] = {
            "x": round(float(orig_coord[0]), 4),
            "y": round(float(orig_coord[1]), 4),
            "properties": orig_props,
        }

    for j, gd in enumerate(gen_data):
        c = coords[j + 1 + orig_offset]  # offset by 1 for reference + orig_offset
        gd["x"] = round(float(c[0]), 4)
        gd["y"] = round(float(c[1]), 4)
        response["ligands"].append(gd)

    # Cache result
    job[cache_key] = response
    return response


@app.get("/property-space/{job_id}")
async def property_space(job_id: str, request: Request):  # NOSONAR
    """Compute all RDKit properties for reference + generated ligands.

    Uses all accumulated results across iterations for multi-round visualisation.
    Returns data suitable for violin / distribution plots.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, _ERR_RDKIT_UNAVAILABLE)
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    all_results = job.get("all_results", [])
    latest_results = job.get("results", [])
    if not all_results and not latest_results:
        raise HTTPException(400, _ERR_NO_GENERATED)

    # Check cache
    cached = job.get("propspace_cache")
    if cached is not None:
        return cached

    # Reference
    ref_mol = _read_ligand_mol(job.get("ligand_path", ""))
    ref_mol_noh = Chem.RemoveHs(ref_mol) if ref_mol else None
    ref_props = _compute_all_properties(ref_mol_noh) if ref_mol_noh else {}

    # Original ligand properties (if reference was swapped)
    orig_props = None
    if "original_ligand_path" in job:
        orig_props = job.get("original_ligand_properties")
        if orig_props is None:
            orig_mol = _read_ligand_mol(job["original_ligand_path"])
            if orig_mol is not None:
                orig_props = _compute_all_properties(Chem.RemoveHs(orig_mol))

    # Generated — flatten all iterations with iteration tags
    ligands = []
    if all_results:
        global_idx = 0
        for round_entry in all_results:
            iteration = round_entry.get("iteration", 0)
            for local_idx, r in enumerate(round_entry.get("results", [])):
                existing_props = r.get("properties")
                if existing_props:
                    ligands.append(
                        {
                            "idx": global_idx,
                            "local_idx": local_idx,
                            "iteration": iteration,
                            "smiles": r.get("smiles", ""),
                            "properties": existing_props,
                        }
                    )
                else:
                    sdf = r.get("sdf_no_hs") or r.get("sdf", "")
                    mol = Chem.MolFromMolBlock(sdf, removeHs=True) if sdf else None
                    if mol is None:
                        global_idx += 1
                        continue
                    ligands.append(
                        {
                            "idx": global_idx,
                            "local_idx": local_idx,
                            "iteration": iteration,
                            "smiles": r.get("smiles", ""),
                            "properties": _compute_all_properties(mol),
                        }
                    )
                global_idx += 1
    else:
        for i, r in enumerate(latest_results):
            existing_props = r.get("properties")
            if existing_props:
                ligands.append(
                    {
                        "idx": i,
                        "local_idx": i,
                        "iteration": 0,
                        "smiles": r.get("smiles", ""),
                        "properties": existing_props,
                    }
                )
            else:
                sdf = r.get("sdf_no_hs") or r.get("sdf", "")
                mol = Chem.MolFromMolBlock(sdf, removeHs=True) if sdf else None
                if mol is None:
                    continue
                ligands.append(
                    {
                        "idx": i,
                        "local_idx": i,
                        "iteration": 0,
                        "smiles": r.get("smiles", ""),
                        "properties": _compute_all_properties(mol),
                    }
                )

    response = {
        "reference": ref_props,
        "original": orig_props,
        "ligands": ligands,
        "property_names": ALL_PROPERTY_NAMES,
        "continuous_properties": list(CONT_PROPERTIES_RDKIT.keys()),
        "discrete_properties": list(DISC_PROPERTIES_RDKIT.keys())
        + ["NumChiralCenters"],
    }

    job["propspace_cache"] = response
    return response


@app.get("/ligand-properties/{job_id}/{ligand_idx}")
async def ligand_properties(job_id: str, ligand_idx: int, request: Request):
    """Return full RDKit property set for a single ligand.

    ligand_idx = -1 returns reference ligand properties.
    """
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, _ERR_RDKIT_UNAVAILABLE)
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)

    if ligand_idx < 0:
        mol = _read_ligand_mol(job.get("ligand_path", ""))
        if mol is None:
            raise HTTPException(400, "Cannot parse reference ligand.")
        mol = Chem.RemoveHs(mol)
    else:
        results = job.get("results", [])
        if ligand_idx >= len(results):
            raise HTTPException(400, _ERR_LIGAND_IDX_RANGE)
        sdf = results[ligand_idx].get("sdf_no_hs") or results[ligand_idx].get("sdf", "")
        mol = Chem.MolFromMolBlock(sdf, removeHs=True) if sdf else None
        if mol is None:
            raise HTTPException(400, _ERR_CANNOT_PARSE_LIGAND)

    return {
        "ligand_idx": ligand_idx,
        "properties": _compute_all_properties(mol),
        "property_names": ALL_PROPERTY_NAMES,
    }


# ---------------------------------------------------------------------------
# Rank & Select by Affinity
# ---------------------------------------------------------------------------

_AFFINITY_TYPES = {"pic50", "pki", "pkd", "pec50"}


class RankSelectRequest(BaseModel):
    affinity_type: str  # "pic50", "pki", "pkd", or "pec50"
    top_n: Optional[int] = None  # None = sort only; int = keep top N


@app.post("/rank-select/{job_id}")
async def rank_select(job_id: str, body: RankSelectRequest, request: Request):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No results to rank.")
    if body.affinity_type not in _AFFINITY_TYPES:
        raise HTTPException(
            400,
            f"Invalid affinity_type '{body.affinity_type}'. "
            f"Must be one of {sorted(_AFFINITY_TYPES)}.",
        )

    # Back up original results (only once)
    if "results_full" not in job:
        job["results_full"] = list(results)

    aff = body.affinity_type

    def _sort_key(r):
        v = r.get("properties", {}).get(aff)
        return (v is not None, v if v is not None else float("-inf"))

    sorted_results = sorted(job["results_full"], key=_sort_key, reverse=True)
    if body.top_n is not None and body.top_n > 0:
        sorted_results = sorted_results[: body.top_n]

    # Include original index so the client can map back after reordering
    results_full = job["results_full"]
    id_to_orig_idx = {id(r): i for i, r in enumerate(results_full)}
    for r in sorted_results:
        r["_original_idx"] = id_to_orig_idx.get(id(r), 0)

    job["results"] = sorted_results

    # Invalidate visualisation caches (preserve results_full for re-ranking)
    for k in [
        k
        for k in job
        if k.startswith("chemspace_") or k in ("propspace_cache", "affinity_dist_cache")
    ]:
        del job[k]

    return {
        "results": sorted_results,
        "affinity_type": aff,
        "total_before": len(job["results_full"]),
        "total_after": len(sorted_results),
    }


@app.post("/reset-rank/{job_id}")
async def reset_rank(job_id: str, request: Request):
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    if "results_full" not in job:
        raise HTTPException(400, "No rank selection to reset.")

    job["results"] = job.pop("results_full")

    _invalidate_viz_caches(job)

    return {"results": job["results"], "total": len(job["results"])}


# ---------------------------------------------------------------------------
# Affinity Distribution
# ---------------------------------------------------------------------------


def _extract_affinity_distributions(
    all_results: list,
) -> tuple:
    """Build per-round affinity distributions for each affinity key."""
    aff_keys = ["pic50", "pki", "pkd", "pec50"]
    distributions: Dict[str, list] = {}
    available_types: List[str] = []

    for key in aff_keys:
        rounds: list = []
        for round_entry in all_results:
            iteration = round_entry.get("iteration", 0)
            values, labels, indices = [], [], []
            for i, r in enumerate(round_entry.get("results", [])):
                v = r.get("properties", {}).get(key)
                if v is not None:
                    values.append(round(float(v), 4))
                    labels.append(f"Ligand #{i + 1}")
                    indices.append(i)
            if values:
                rounds.append(
                    {
                        "iteration": iteration,
                        "values": values,
                        "labels": labels,
                        "indices": indices,
                    }
                )
        if rounds:
            distributions[key] = rounds
            available_types.append(key)

    return distributions, available_types


@app.get("/affinity-distribution/{job_id}")
async def affinity_distribution(job_id: str, request: Request):
    """Return predicted affinity values for all generated ligands plus
    the reference ligand's experimental affinity (if detected from SDF).

    Response shape:
      {
        affinity_types: ["pic50", ...],   // which affinity types have data
        distributions: {
          "pic50": { values: [...], labels: [...] },
          ...
        },
        ref_affinity: { p_label, p_value, raw_tag, assay_type, unit } | null,
        n_ligands: int
      }
    """
    _validate_job_id(job_id)
    job = _get_user_job(job_id, request)
    all_results = job.get("all_results", [])
    # Fallback: if all_results not populated yet, use current results as round 0
    if not all_results:
        results = job.get("results", [])
        if not results:
            raise HTTPException(400, _ERR_NO_GENERATED)
        all_results = [{"iteration": 0, "results": results}]

    # Check cache
    cached = job.get("affinity_dist_cache")
    if cached is not None:
        return cached

    distributions, available_types = _extract_affinity_distributions(all_results)

    if not available_types:
        raise HTTPException(
            400,
            "No affinity predictions available. The model checkpoint may not "
            "include an affinity prediction head.",
        )

    ref_affinity = job.get("ref_affinity")

    n_total = sum(len(entry.get("results", [])) for entry in all_results)
    response = {
        "affinity_types": available_types,
        "distributions": distributions,
        "ref_affinity": ref_affinity,
        "n_ligands": n_total,
    }
    job["affinity_dist_cache"] = response
    return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("FLOWR_PORT", 8787))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
