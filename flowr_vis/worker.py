"""
FLOWR GPU Worker Service
========================
Standalone FastAPI service that handles GPU-intensive ligand generation.
This worker is designed to run on a GPU-equipped machine (AWS GPU instance,
etc.) and is called by the lightweight CPU-only frontend server when the
user clicks "Generate".

Endpoints
---------
- POST /generate          Start a generation job
- GET  /job/{job_id}      Poll job progress / retrieve results
- POST /load-model        Load a checkpoint into GPU memory
- GET  /model-status      Check whether the model is loaded
- GET  /health            Liveness probe

The worker downloads input files (protein PDB, ligand SDF) from URLs
provided by the frontend server, runs FLOWR generation, and returns
the results as JSON.
"""

import gc
import io
import logging
import os
import re as _re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

# ---------------------------------------------------------------------------
# RDKit imports
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, Geometry
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available – molecule processing will be limited.")

# ---------------------------------------------------------------------------
# Shared chemistry utilities (also used by server.py)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PyTorch – detect device
# ---------------------------------------------------------------------------
import torch
from chem_utils import (
    compute_all_properties as _compute_all_properties,
)
from chem_utils import (
    compute_pocket_com as _compute_pocket_com_shared,
)
from chem_utils import (
    mol_to_atom_info as _mol_to_atom_info,
)
from chem_utils import (
    mol_to_bond_info as _mol_to_bond_info,
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

from flowr.util.device import get_device, get_map_location

DEVICE = str(get_device())
logger.info("PyTorch device: %s", DEVICE)


def _sanitize_job_id(job_id: str) -> str:
    """Validate and return job_id, or raise ValueError if malicious."""
    if not job_id or not _re.match(r"^[a-zA-Z0-9_-]+$", job_id):
        raise ValueError(f"Invalid job_id format: {job_id!r}")
    return job_id


# ---------------------------------------------------------------------------
# FLOWR imports
# ---------------------------------------------------------------------------
FLOWR_AVAILABLE = False
FLOWR_MOL_AVAILABLE = False
try:
    import flowr.gen.utils as gen_util
    import flowr.util.rdkit as smolRD
    from flowr.data.dataset import GeometricDataset
    from flowr.gen.generate import generate_ligands_per_target
    from flowr.gen.mol_filter import (
        ADMECriterion,
        ADMEFilter,
        MolFilterPipeline,
        PropertyCriterion,
        PropertyFilter,
    )
    from flowr.gen.utils import load_data_from_pdb, load_util
    from flowr.scriptutil import load_model
    from flowr.util.functional import LigandPocketOptimization
    from flowr.util.metrics import calc_strain, evaluate_pb_validity
    from flowr.util.pocket import PocketComplexBatch

    FLOWR_AVAILABLE = True
except ImportError as exc:
    logger.warning(
        "FLOWR modules not importable (%s) – generation will "
        "not work. Make sure the flowr package is on PYTHONPATH.",
        exc,
    )

# LBDD-specific imports (molecule-only, no pocket)
try:
    from flowr.gen.generate import generate_molecules
    from flowr.gen.utils import load_data_from_sdf_mol, load_util_mol
    from flowr.scriptutil import load_mol_model
    from flowr.util.molrepr import GeometricMolBatch

    FLOWR_MOL_AVAILABLE = True
except ImportError as exc:
    logger.warning(
        "FLOWR molecule modules not importable (%s) – LBDD "
        "generation will not work.",
        exc,
    )

# ---------------------------------------------------------------------------
# Checkpoint path resolution
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CKPTS_DIR = ROOT_DIR / "ckpts"
DEFAULT_CKPT_PATH = CKPTS_DIR / "flowr_root.ckpt"
CKPT_PATH = os.environ.get("FLOWR_CKPT_PATH", str(DEFAULT_CKPT_PATH))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="FLOWR GPU Worker", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FLOWR_CORS_ORIGINS", "http://localhost:8787").split(
        ","
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path(tempfile.mkdtemp(prefix="flowr_worker_"))
JOBS: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

_JOB_TTL_SECONDS = 3600  # 1 hour


def _cleanup_expired_jobs():
    """Remove completed/failed jobs older than TTL."""
    now = time.time()
    dirs_to_clean: list[Path] = []
    with _jobs_lock:
        expired = [
            jid
            for jid, j in JOBS.items()
            if j.get("status") in ("completed", "failed")
            and now - j.get("created_at", now) > _JOB_TTL_SECONDS
        ]
        for jid in expired:
            JOBS.pop(jid, None)
            dirs_to_clean.append(WORK_DIR / jid)
    # I/O outside lock to avoid blocking other threads
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


# Serializes GPU generation — only one job at a time to prevent
# model state races and GPU OOM from concurrent inference.
_generation_semaphore = threading.Semaphore(1)

# Auto-shutdown: worker exits after this many seconds of inactivity.
# Set via FLOWR_WORKER_IDLE_TIMEOUT (0 = disabled).
IDLE_TIMEOUT = int(os.environ.get("FLOWR_WORKER_IDLE_TIMEOUT", "0"))
_last_activity = time.time()
_shutdown_requested = False

logger.info("Work directory: %s", WORK_DIR)
logger.info("FLOWR available: %s", FLOWR_AVAILABLE)
logger.info("Checkpoint path: %s", CKPT_PATH)
if IDLE_TIMEOUT > 0:
    logger.info("Idle timeout: %ss", IDLE_TIMEOUT)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════
_model_state: Dict[str, Any] = {
    "loaded": False,
    "loading": False,
    "error": None,
    "model": None,
    "hparams": None,
    "vocab": None,
    "vocab_charges": None,
    "vocab_hybridization": None,
    "vocab_aromatic": None,
}
_model_lock = threading.Lock()

# LBDD (molecule-only) model singleton
_mol_model_state: Dict[str, Any] = {
    "loaded": False,
    "loading": False,
    "error": None,
    "model": None,
    "hparams": None,
    "vocab": None,
    "vocab_charges": None,
    "vocab_hybridization": None,
    "vocab_aromatic": None,
}
_mol_model_lock = threading.Lock()


def _default_args(**overrides) -> Namespace:
    """Build a Namespace with all the defaults expected by the FLOWR pipeline."""
    defaults = {
        # ── File paths ──
        "pdb_file": "",
        "ligand_file": "",
        "pdb_id": None,
        "ligand_id": None,
        "res_txt_file": None,
        "chain_id": None,
        # ── Preprocessing ──
        "canonicalize_conformer": False,
        "pocket_noise": "fix",
        "cut_pocket": True,
        "pocket_cutoff": 6.0,
        "protonate_pocket": False,
        "compute_interactions": False,
        "compute_interaction_recovery": False,
        "add_hs": False,
        "add_hs_and_optimize": False,
        "optimize_gen_ligs": False,
        "optimize_gen_ligs_hs": False,
        "kekulize": False,
        "use_pdbfixer": False,
        "add_bonds_to_protein": True,
        "add_hs_to_protein": False,
        "max_pocket_size": 1000,
        "min_pocket_size": 10,
        # ── Runtime ──
        "seed": 42,
        "gpus": 1,
        "mp_index": 0,
        "num_workers": 0,
        # ── Architecture ──
        "arch": "pocket",
        "pocket_type": "holo",
        "pocket_coord_noise_std": 0.0,
        "ckpt_path": str(CKPT_PATH),
        "lora_finetuned": False,
        "data_path": "",
        "splits_path": None,
        "dataset": "spindr",
        "save_dir": "",
        "save_file": None,
        # ── Sampling ──
        "coord_noise_scale": 0.1,
        "max_sample_iter": 100,
        "sample_n_molecules_per_target": 10,
        "sample_mol_sizes": False,
        "corrector_iters": 0,
        "rotation_alignment": False,
        "permutation_alignment": False,
        "save_traj": False,
        # ── Filtering ──
        "filter_valid_unique": True,
        "filter_diversity": False,
        "diversity_threshold": 0.9,
        "filter_pb_valid": False,
        "calculate_pb_valid": False,
        "filter_cond_substructure": False,
        "calculate_strain_energies": False,
        # ── Batching ──
        "batch_cost": 25,
        "dataset_split": None,
        # ── Inpainting modes (all off by default) ──
        "ligand_time": None,
        "pocket_time": None,
        "interaction_time": None,
        "fixed_interactions": False,
        "interaction_conditional": False,
        "scaffold_hopping": False,
        "scaffold_elaboration": False,
        "linker_inpainting": False,
        "fragment_inpainting": False,
        "fragment_growing": False,
        "grow_size": None,
        "prior_center_file": None,
        "max_fragment_cuts": 3,
        "core_growing": False,
        "ring_system_index": 0,
        "substructure_inpainting": False,
        "substructure": None,
        "graph_inpainting": None,
        "final_inpaint": False,
        "separate_pocket_interpolation": False,
        "separate_interaction_interpolation": False,
        "anisotropic_prior": False,
        "ref_ligand_com_prior": False,
        # ── De novo placeholder ligand ──
        "add_placeholder_ligand": False,
        "num_heavy_atoms": None,
        # ── Integration ──
        "integration_steps": 100,
        "cat_sampling_noise_level": 1,
        "ode_sampling_strategy": "linear",
        "solver": "euler",
        "categorical_strategy": "uniform-sample",
        "use_sde_simulation": False,
        "use_cosine_scheduler": False,
        "bucket_cost_scale": "quadratic",
        # ── Guidance ──
        "guidance_config": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _load_model_if_needed(ckpt_path: str = None) -> bool:
    """Load the FLOWR model singleton (thread-safe, idempotent)."""
    resolved = ckpt_path or CKPT_PATH

    if _model_state["loaded"] and _model_state.get("ckpt_path") == resolved:
        return True

    with _model_lock:
        if _model_state["loaded"] and _model_state.get("ckpt_path") == resolved:
            return True
        if not FLOWR_AVAILABLE:
            _model_state["error"] = "FLOWR modules not available"
            return False
        if not Path(resolved).exists():
            _model_state["error"] = f"Checkpoint not found: {resolved}"
            return False

        _model_state.update(loading=True, loaded=False)
        try:
            logger.info("Loading FLOWR model from %s …", resolved)
            args = _default_args(ckpt_path=resolved)

            (
                model,
                hparams,
                vocab,
                vocab_charges,
                vocab_hybridization,
                vocab_aromatic,
                _vocab_pocket_atoms,
                _vocab_pocket_res,
            ) = load_model(args)

            model = model.to(DEVICE)
            model.eval()
            logger.info("Model loaded on %s.", DEVICE)

            _model_state.update(
                loaded=True,
                loading=False,
                error=None,
                model=model,
                hparams=hparams,
                vocab=vocab,
                vocab_charges=vocab_charges,
                vocab_hybridization=vocab_hybridization,
                vocab_aromatic=vocab_aromatic,
                ckpt_path=resolved,
            )
            return True

        except Exception as exc:
            logger.exception("Failed to load FLOWR model")
            _model_state.update(loading=False, error=str(exc))
            return False
        finally:
            # Ensure GPU cleanup runs even on load failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _default_args_mol(**overrides) -> Namespace:
    """Build a Namespace with defaults for the LBDD (molecule-only) pipeline."""
    defaults = {
        # ── File paths ──
        "sdf_path": "",
        "ligand_idx": 0,
        # ── Runtime ──
        "seed": 42,
        "gpus": 1,
        "mp_index": 0,
        "num_workers": 0,
        # ── Architecture ──
        "arch": "flowr",
        "ckpt_path": str(CKPT_PATH),
        "lora_finetuned": False,
        "data_path": "",
        "splits_path": None,
        "dataset": "spindr",
        "save_dir": "",
        "save_file": None,
        # ── Sampling ──
        "coord_noise_scale": 0.1,
        "max_sample_iter": 100,
        "sample_n_molecules": 10,
        "sample_n_molecules_per_mol": 1,
        "sample_mol_sizes": False,
        "corrector_iters": 0,
        "rotation_alignment": False,
        "permutation_alignment": False,
        "save_traj": False,
        # ── Filtering ──
        "filter_valid_unique": True,
        "filter_diversity": False,
        "diversity_threshold": 0.9,
        "filter_pb_valid": False,
        "calculate_pb_valid": False,
        "filter_cond_substructure": False,
        "calculate_strain_energies": False,
        # ── Batching ──
        "batch_cost": 25,
        "dataset_split": None,
        # ── Integration ──
        "integration_steps": 100,
        "cat_sampling_noise_level": 1,
        "ode_sampling_strategy": "linear",
        "solver": "euler",
        "categorical_strategy": "uniform-sample",
        "use_sde_simulation": False,
        "use_cosine_scheduler": False,
        "bucket_cost_scale": "quadratic",
        # ── Guidance ──
        "guidance_config": None,
        # ── LBDD-specific – no pocket params ──
        "anisotropic_prior": False,
        "ref_ligand_com_prior": False,
        "kekulize": False,
        "add_hs": False,
        "add_hs_and_optimize": False,
        "optimize_gen_ligs": False,
        "optimize_gen_ligs_hs": False,
        # ── Inpainting mode flags (required by load_mol_model) ──
        "scaffold_hopping": False,
        "scaffold_elaboration": False,
        "linker_inpainting": False,
        "core_growing": False,
        "fragment_inpainting": False,
        "fragment_growing": False,
        "substructure_inpainting": False,
        "substructure": False,
        "interaction_conditional": False,
        # ── Fragment growing / inpainting params ──
        "grow_size": None,
        "prior_center_file": None,
        "max_fragment_cuts": 3,
        "ring_system_index": 0,
        "canonicalize_conformer": False,
        "graph_inpainting": None,
        "final_inpaint": False,
        # ── De novo placeholder ligand ──
        "add_placeholder_ligand": False,
        "num_heavy_atoms": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _load_mol_model_if_needed(ckpt_path: str = None) -> bool:
    """Load the FLOWR *molecule-only* model singleton (thread-safe)."""
    resolved = ckpt_path or CKPT_PATH

    if _mol_model_state["loaded"] and _mol_model_state.get("ckpt_path") == resolved:
        return True

    with _mol_model_lock:
        if _mol_model_state["loaded"] and _mol_model_state.get("ckpt_path") == resolved:
            return True
        if not FLOWR_MOL_AVAILABLE:
            _mol_model_state["error"] = "FLOWR molecule modules not available"
            return False
        if not Path(resolved).exists():
            _mol_model_state["error"] = f"Checkpoint not found: {resolved}"
            return False

        _mol_model_state.update(loading=True, loaded=False)
        try:
            logger.info("Loading FLOWR *mol* model from %s …", resolved)
            args = _default_args_mol(ckpt_path=resolved)

            (
                model,
                hparams,
                vocab,
                vocab_charges,
                vocab_hybridization,
                vocab_aromatic,
            ) = load_mol_model(args)

            model = model.to(DEVICE)
            model.eval()
            logger.info("Mol model loaded on %s.", DEVICE)

            _mol_model_state.update(
                loaded=True,
                loading=False,
                error=None,
                model=model,
                hparams=hparams,
                vocab=vocab,
                vocab_charges=vocab_charges,
                vocab_hybridization=vocab_hybridization,
                vocab_aromatic=vocab_aromatic,
                ckpt_path=resolved,
            )
            return True

        except Exception as exc:
            logger.exception("Failed to load FLOWR mol model")
            _mol_model_state.update(loading=False, error=str(exc))
            return False
        finally:
            # Ensure GPU cleanup runs even on load failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════


class ActiveLearningRequest(BaseModel):
    """Request model for active learning LoRA finetuning."""

    job_id: str
    protein_url: str
    protein_filename: str = "protein.pdb"
    ligand_sdf_strings: List[str]  # SDF strings of selected ligands
    ckpt_path: Optional[str] = None
    finetuned_ckpt_url: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lr: float = 5e-4
    batch_cost: int = 4
    acc_batches: Optional[int] = None  # None = dynamic based on n_ligands
    epochs: Optional[int] = None  # None = dynamic based on n_ligands
    pocket_cutoff: float = 7.0


class GenerationRequest(BaseModel):
    job_id: str
    # SBDD fields (optional for LBDD)
    protein_url: Optional[str] = None
    ligand_url: Optional[str] = None
    protein_filename: Optional[str] = None
    ligand_filename: Optional[str] = None
    ckpt_path: Optional[str] = None
    gen_mode: str = "denovo"
    fixed_atoms: List[int] = []
    n_samples: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=25, ge=1, le=200)
    integration_steps: int = 100
    pocket_cutoff: float = 6.0
    grow_size: Optional[int] = None
    prior_center_url: Optional[str] = None
    prior_center_filename: Optional[str] = None
    coord_noise_scale: float = 0.0
    filter_valid_unique: bool = True
    filter_cond_substructure: bool = False
    filter_diversity: bool = False
    diversity_threshold: float = 0.9
    sample_mol_sizes: bool = False
    filter_pb_valid: bool = False
    calculate_pb_valid: bool = False
    calculate_strain_energies: bool = False
    optimize_gen_ligs: bool = False
    optimize_gen_ligs_hs: bool = False
    anisotropic_prior: bool = False
    ref_ligand_com_prior: bool = False
    ring_system_index: int = 0
    # ── LBDD-specific fields ──
    workflow_type: str = "sbdd"  # "sbdd" or "lbdd"
    optimize_method: str = "none"  # "none", "rdkit", or "xtb"
    sample_n_molecules_per_mol: int = 1
    # ── De novo without ligand ──
    num_heavy_atoms: Optional[int] = None
    # ── Property / ADMET filtering ──
    property_filter: Optional[List[dict]] = None
    adme_filter: Optional[List[dict]] = None
    # ── Diversity history (from previous generation rounds) ──
    previous_smiles: Optional[List[str]] = None
    # ── Finetuned checkpoint URL (server-hosted, for persistence across workers) ──
    finetuned_ckpt_url: Optional[str] = None


class LoadModelRequest(BaseModel):
    ckpt_path: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
#  FILE HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _download_file(url: str, dest_path: Path) -> bool:
    """Download a file from a URL to a local path."""
    try:
        resp = http_requests.get(url, timeout=120)
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        return True
    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return False


def _download_file_streaming(url: str, dest_path: Path, timeout: int = 600) -> bool:
    """Download a large file using streaming to avoid loading it all into memory."""
    try:
        resp = http_requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.error("Failed to stream-download %s: %s", url, exc)
        if dest_path.exists():
            dest_path.unlink()
        return False


_ERR_CANCELLED = "Cancelled by user"
_ERR_JOB_NOT_FOUND = "Job not found."

_HEALTH_GRACE_BATCHES = 3
_HEALTH_THRESHOLD = 0.15
_CONDITIONAL_MODES = frozenset(
    {
        "substructure_inpainting",
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
    }
)


class _JobCancelled(Exception):
    """Internal signal for cooperative cancellation."""


class _HealthCheckFailed(Exception):
    """Raised when generation health checks detect unrecoverable poor performance."""

    def __init__(self, check_type: str, message: str, advice: str):
        self.check_type = check_type
        self.message = message
        self.advice = advice
        super().__init__(message)


class _TeeStream:
    """Write to two streams simultaneously (for capturing + printing stdout).

    Can be used as a context manager to automatically restore sys.stdout
    (and optionally sys.stderr).
    """

    def __init__(self, *streams, stream_name: str = "stdout"):
        self.streams = streams
        self._stream_name = stream_name
        self._old_stream = None

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass  # individual stream failure is non-critical

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass  # individual stream failure is non-critical

    def __enter__(self):
        self._old_stream = getattr(sys, self._stream_name)
        setattr(sys, self._stream_name, self)
        return self

    def __exit__(self, *exc):
        if self._old_stream is not None:
            setattr(sys, self._stream_name, self._old_stream)
        return False


# Known warning patterns from the FLOWR library that should be surfaced.
# Checked against merged stdout+stderr output.
_GENERATION_WARNING_PATTERNS = [
    (
        "Falling back to de novo",
        (
            "Generation mode fell back to de novo because the reference molecule "
            "could not be processed for the selected mode. This usually indicates "
            "an invalid or problematic molecule (e.g. kekulization failure). "
            "Please check your input molecule."
        ),
    ),
    (
        "could not be sanitized",
        (
            "Reference molecule could not be sanitized — substructure extraction "
            "failed. The molecule may have invalid bond orders or aromatic assignments."
        ),
    ),
    (
        "Can't kekulize mol",
        (
            "RDKit kekulization failed on the reference molecule. "
            "This can happen when aromatic bonds are inconsistent."
        ),
    ),
]


def _extract_generation_warnings(
    captured_stdout: str,
    captured_stderr: str = "",
    gen_mode: str = "denovo",
    n_generated: int = 0,
) -> List[str]:
    """Scan captured output for known warning patterns and detect fallback.

    Checks both stdout (Python ``print()`` messages) and stderr (RDKit C++
    messages) for known patterns.  Also performs direct result-based
    detection: if an inpainting mode was requested but zero molecules were
    generated, this strongly suggests a silent fallback occurred.
    """
    combined = captured_stdout + "\n" + captured_stderr
    warnings = []
    seen = set()
    for pattern, msg in _GENERATION_WARNING_PATTERNS:
        if msg is None:
            continue
        if pattern in combined and msg not in seen:
            warnings.append(msg)
            seen.add(msg)

    # Direct detection: inpainting mode with no output → likely fallback
    _inpainting_modes = {
        "substructure_inpainting",
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
        "fragment_growing",
        "fragment_inpainting",
    }
    if gen_mode in _inpainting_modes and n_generated == 0:
        msg = (
            f"No valid molecules were generated in '{gen_mode}' mode. "
            "The generation may have silently fallen back to de novo. "
            "Please check your input molecule and fixed atoms."
        )
        if msg not in seen:
            warnings.append(msg)
            seen.add(msg)

    return warnings


def _raise_if_cancelled(job: Dict[str, Any], stage: str = ""):
    if job.get("cancelled") or job.get("status") == "cancelled":
        msg = _ERR_CANCELLED
        if stage:
            msg = f"{_ERR_CANCELLED} ({stage})"
        raise _JobCancelled(msg)


def _run_health_checks(
    *,
    batch_count: int,
    gen_mode: str,
    # validity/uniqueness
    raw_generated: int,
    valid_count: int,
    unique_count: int,
    filter_valid_unique: bool,
    # substructure
    filter_cond_substructure: bool,
    sub_total_before: int,
    sub_total_after: int,
    # diversity
    filter_diversity: bool,
    diversity_input: int,
    diversity_removed: int,
    # cross-round novelty
    previous_smiles: Optional[list],
    history_input: int,
    history_removed: int,
) -> None:
    """Check generation health metrics after each batch.

    Raises _HealthCheckFailed if any metric falls below _HEALTH_THRESHOLD
    after _HEALTH_GRACE_BATCHES have completed.
    """
    if batch_count < _HEALTH_GRACE_BATCHES:
        return

    is_conditional = gen_mode in _CONDITIONAL_MODES
    mode_advice = (
        " Consider changing the generation mode or adjusting your atom selection."
        if is_conditional
        else ""
    )

    # 1. Validity check
    if filter_valid_unique and raw_generated > 0:
        validity_rate = valid_count / raw_generated
        if validity_rate < _HEALTH_THRESHOLD:
            raise _HealthCheckFailed(
                check_type="low_validity",
                message=(
                    f"Very low validity rate ({validity_rate:.1%}) after "
                    f"{batch_count} batches ({valid_count}/{raw_generated} valid)."
                ),
                advice=(
                    "The model is performing poorly on this target \u2014 most generated "
                    "molecules are invalid. Check your protein/ligand input files for "
                    "issues (e.g. missing atoms, incorrect protonation)." + mode_advice
                ),
            )

    # 2. Uniqueness check
    if filter_valid_unique and valid_count > 0:
        uniqueness_rate = unique_count / valid_count
        if uniqueness_rate < _HEALTH_THRESHOLD:
            raise _HealthCheckFailed(
                check_type="low_uniqueness",
                message=(
                    f"Very low uniqueness rate ({uniqueness_rate:.1%}) after "
                    f"{batch_count} batches ({unique_count}/{valid_count} unique)."
                ),
                advice=(
                    "The model is generating mostly duplicate molecules. "
                    "Try increasing the coordinate noise scale or integration steps."
                    + mode_advice
                ),
            )

    # 3. Substructure match rate check (only when filter enabled)
    if filter_cond_substructure and sub_total_before > 0:
        sub_rate = sub_total_after / sub_total_before
        if sub_rate < _HEALTH_THRESHOLD:
            raise _HealthCheckFailed(
                check_type="low_substructure_match",
                message=(
                    f"Very low substructure match rate ({sub_rate:.1%}) after "
                    f"{batch_count} batches ({sub_total_after}/{sub_total_before} matched)."
                ),
                advice=(
                    "The model cannot maintain the requested substructure constraints. "
                    "RDKit may be failing to match the substructure pattern. "
                    "Try a different generation mode, adjust your atom selection, "
                    "or run without substructure filtering."
                ),
            )

    # 4. Diversity check (only when filter enabled)
    if filter_diversity and diversity_input > 0:
        diversity_passed = diversity_input - diversity_removed
        diversity_rate = diversity_passed / diversity_input
        if diversity_rate < _HEALTH_THRESHOLD:
            raise _HealthCheckFailed(
                check_type="low_diversity",
                message=(
                    f"Very low diversity rate ({diversity_rate:.1%}) after "
                    f"{batch_count} batches ({diversity_passed}/{diversity_input} diverse)."
                ),
                advice=(
                    "Mode collapse detected \u2014 the model keeps generating very similar "
                    "molecules. Try increasing the coordinate noise scale."
                    + mode_advice
                ),
            )

    # 5. Cross-round novelty check (only when previous_smiles provided)
    if previous_smiles and history_input > 0:
        history_passed = history_input - history_removed
        novelty_rate = history_passed / history_input
        if novelty_rate < _HEALTH_THRESHOLD:
            raise _HealthCheckFailed(
                check_type="low_cross_round_novelty",
                message=(
                    f"Very low cross-round novelty ({novelty_rate:.1%}) after "
                    f"{batch_count} batches ({history_passed}/{history_input} novel "
                    f"vs. {len(previous_smiles)} previous)."
                ),
                advice=(
                    "Cross-round mode collapse \u2014 the model is regenerating molecules "
                    "from previous rounds. Try increasing the coordinate noise scale "
                    "or resetting the generation history."
                ),
            )


# ═══════════════════════════════════════════════════════════════════════════
#  PRIOR CLOUD (for fragment-growing visualisation)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_pocket_com_worker(
    protein_path: str,
    ligand_path: str,
    pocket_cutoff: float = 6.0,
) -> Optional[np.ndarray]:
    """Thin wrapper delegating to the shared implementation in chem_utils."""
    return _compute_pocket_com_shared(protein_path, ligand_path, pocket_cutoff)


def _shape_covariance_np_worker(coords: np.ndarray) -> Optional[np.ndarray]:
    """Compute PCA-based shape covariance from atom coordinates (numpy)."""
    if coords.shape[0] < 3:
        return None
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(coords.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    eigenvalues = eigenvalues / (eigenvalues.sum() / 3.0 + 1e-8)
    eigenvalues = np.clip(eigenvalues, 0.3, 3.0)
    eigenvalues = eigenvalues * (3.0 / eigenvalues.sum())
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _directional_covariance_np(
    source_com: np.ndarray, target_com: np.ndarray
) -> Optional[np.ndarray]:
    """Build an anisotropic covariance elongated along source→target direction."""
    direction = target_com - source_com
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    direction = direction / norm
    ref = (
        np.array([1.0, 0.0, 0.0])
        if abs(direction[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    v2 = ref - np.dot(ref, direction) * direction
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    v3 = np.cross(direction, v2)
    rotation = np.column_stack([v2, v3, direction])
    eigenvalues = np.array([1.0, 1.0, 2.0])
    eigenvalues = np.clip(eigenvalues, 0.3, 3.0)
    eigenvalues = eigenvalues * (3.0 / eigenvalues.sum())
    return rotation @ np.diag(eigenvalues) @ rotation.T


def _compute_anisotropic_cloud_covariance_worker(  # NOSONAR
    ligand_path: str,
    gen_mode: str,
    center: np.ndarray,
    has_prior_center: bool,
    fixed_atoms: Optional[List[int]] = None,
    ring_system_index: int = 0,
) -> Optional[np.ndarray]:
    """Compute mode-specific anisotropic covariance for the worker prior cloud.

    Mirrors the logic in interpolate.py ``_compute_anisotropic_covariance``.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    _ = fixed_atoms
    if not RDKIT_AVAILABLE:
        return None
    if not ligand_path:
        return None

    _ANISO_MODES = {
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
        "fragment_growing",
    }
    if gen_mode not in _ANISO_MODES:
        return None

    mol = _read_ligand_mol(ligand_path)
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
        # Directional covariance: fragment COM → growth center
        source_com = coords.mean(axis=0)
        return _directional_covariance_np(source_com, center)

    if gen_mode in ("core_growing", "scaffold_hopping"):
        # Shape covariance from VARIABLE atoms only
        try:
            from rdkit.Chem.Scaffolds.MurckoScaffold import (
                GetScaffoldForMol as _GetScaf,
            )

            if gen_mode == "scaffold_hopping":
                scaffold = _GetScaf(Chem.Mol(mol_noh))
                if scaffold:
                    scaffold_indices = set()
                    s2 = Chem.Mol(mol_noh)
                    for a in s2.GetAtoms():
                        a.SetIntProp("org_idx", a.GetIdx())
                    scf = _GetScaf(s2)
                    if scf:
                        scaffold_indices = {
                            a.GetIntProp("org_idx") for a in scf.GetAtoms()
                        }
                    variable_indices = [
                        i for i in range(mol_noh.GetNumAtoms()) if i in scaffold_indices
                    ]
                    if len(variable_indices) >= 3:
                        return _shape_covariance_np_worker(coords[variable_indices])
            elif gen_mode == "core_growing":
                s2 = Chem.Mol(mol_noh)
                for a in s2.GetAtoms():
                    a.SetIntProp("org_idx", a.GetIdx())
                scaffold = _GetScaf(s2)
                if scaffold:
                    scaffold_atoms = {
                        a.GetIntProp("org_idx") for a in scaffold.GetAtoms()
                    }
                    ring_systems = []
                    ri = s2.GetRingInfo()
                    for ring in ri.AtomRings():
                        rs = set(ring)
                        merged = False
                        for existing in ring_systems:
                            if rs & existing:
                                existing |= rs
                                merged = True
                                break
                        if not merged:
                            ring_systems.append(rs)
                    ridx = (
                        ring_system_index
                        if ring_system_index < len(ring_systems)
                        else 0
                    )
                    core = (
                        {a for a in scaffold_atoms if a in ring_systems[ridx]}
                        if ring_systems
                        else set()
                    )
                    variable_indices = [
                        i for i in range(mol_noh.GetNumAtoms()) if i not in core
                    ]
                    if len(variable_indices) >= 3:
                        return _shape_covariance_np_worker(coords[variable_indices])
        except Exception:
            pass  # fallback to full-atom covariance below
        return _shape_covariance_np_worker(coords)

    if gen_mode in ("linker_inpainting", "scaffold_elaboration"):
        # Shape covariance from VARIABLE (to-be-replaced) atoms only
        try:
            from rdkit.Chem.Scaffolds.MurckoScaffold import (
                GetScaffoldForMol as _GetScaf,
            )

            s2 = Chem.Mol(mol_noh)
            for a in s2.GetAtoms():
                a.SetIntProp("org_idx", a.GetIdx())
            scaffold = _GetScaf(s2)

            if gen_mode == "scaffold_elaboration" and scaffold:
                scaffold_set = {a.GetIntProp("org_idx") for a in scaffold.GetAtoms()}
                ring_set = set()
                for ring in s2.GetRingInfo().AtomRings():
                    ring_set.update(ring)
                # Approximation of scaffold elaboration mask:
                # replaced = not in scaffold AND not in ring
                variable_indices = [
                    i
                    for i in range(mol_noh.GetNumAtoms())
                    if i not in scaffold_set and i not in ring_set
                ]
                if len(variable_indices) >= 3:
                    return _shape_covariance_np_worker(coords[variable_indices])
            elif gen_mode == "linker_inpainting" and scaffold:
                scaffold_set = {a.GetIntProp("org_idx") for a in scaffold.GetAtoms()}
                ring_atoms = set()
                for ring in mol_noh.GetRingInfo().AtomRings():
                    ring_atoms.update(ring)
                variable_indices = [i for i in scaffold_set if i not in ring_atoms]
                if len(variable_indices) >= 3:
                    return _shape_covariance_np_worker(coords[variable_indices])
        except Exception:
            pass  # fallback to full-atom covariance below
        return _shape_covariance_np_worker(coords)

    return _shape_covariance_np_worker(coords)


def _compute_ref_ligand_com_shift_worker(  # NOSONAR
    ligand_path: str,
    gen_mode: str,
    fixed_atoms: Optional[List[int]] = None,
    ring_system_index: int = 0,
) -> Optional[np.ndarray]:
    """Compute the reference ligand variable fragment CoM shift.

    Mirrors interpolate.py ``_get_ref_com_shift``: returns the CoM of the
    to-be-generated (variable) atoms in the reference ligand, or None
    if not applicable.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    _ = fixed_atoms
    _APPLICABLE_MODES = {
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
    }
    if gen_mode not in _APPLICABLE_MODES:
        return None
    if not RDKIT_AVAILABLE or not ligand_path:
        return None

    mol = _read_ligand_mol(ligand_path)
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
        from rdkit.Chem.Scaffolds.MurckoScaffold import (
            GetScaffoldForMol as _GetScaf,
        )

        s2 = Chem.Mol(mol_noh)
        for a in s2.GetAtoms():
            a.SetIntProp("org_idx", a.GetIdx())
        scaffold = _GetScaf(s2)

        if gen_mode == "scaffold_hopping":
            if scaffold:
                scaffold_indices = {
                    a.GetIntProp("org_idx") for a in scaffold.GetAtoms()
                }
                variable_indices = [
                    i for i in range(mol_noh.GetNumAtoms()) if i in scaffold_indices
                ]
            else:
                variable_indices = list(range(mol_noh.GetNumAtoms()))
        elif gen_mode == "scaffold_elaboration":
            if scaffold:
                scaffold_set = {a.GetIntProp("org_idx") for a in scaffold.GetAtoms()}
                ring_set = set()
                for ring in s2.GetRingInfo().AtomRings():
                    ring_set.update(ring)
                variable_indices = [
                    i
                    for i in range(mol_noh.GetNumAtoms())
                    if i not in scaffold_set and i not in ring_set
                ]
            else:
                variable_indices = list(range(mol_noh.GetNumAtoms()))
        elif gen_mode == "linker_inpainting":
            if scaffold:
                ring_atoms = set()
                for ring in mol_noh.GetRingInfo().AtomRings():
                    ring_atoms.update(ring)
                scaffold_set = {a.GetIntProp("org_idx") for a in scaffold.GetAtoms()}
                variable_indices = [i for i in scaffold_set if i not in ring_atoms]
            else:
                variable_indices = list(range(mol_noh.GetNumAtoms()))
        elif gen_mode == "core_growing":
            if scaffold:
                scaffold_atoms = {a.GetIntProp("org_idx") for a in scaffold.GetAtoms()}
                ring_systems = []
                ri = s2.GetRingInfo()
                for ring in ri.AtomRings():
                    rs = set(ring)
                    merged = False
                    for existing in ring_systems:
                        if rs & existing:
                            existing |= rs
                            merged = True
                            break
                    if not merged:
                        ring_systems.append(rs)
                ridx = ring_system_index if ring_system_index < len(ring_systems) else 0
                core = (
                    {a for a in scaffold_atoms if a in ring_systems[ridx]}
                    if ring_systems
                    else set()
                )
                variable_indices = [
                    i for i in range(mol_noh.GetNumAtoms()) if i not in core
                ]
            else:
                variable_indices = list(range(mol_noh.GetNumAtoms()))
        else:
            return None

        if not variable_indices:
            return None
        return coords[variable_indices].mean(axis=0)
    except Exception:
        return None


def _compute_prior_cloud(  # NOSONAR
    ligand_path: str,
    prior_center_file: Optional[str],
    grow_size: int,
    protein_path: Optional[str] = None,
    pocket_cutoff: float = 6.0,
    anisotropic_prior: bool = False,
    gen_mode: str = "fragment_growing",
    ring_system_index: int = 0,
    ref_ligand_com_prior: bool = False,
    fixed_atoms: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute a representative prior point cloud for visualisation.

    The cloud approximates the starting positions of the variable atoms that
    the flow-matching process will transform into the generated fragment.
    Coordinates are in the *original* PDB / SDF frame so they can be rendered
    directly in the 3Dmol viewer alongside the uploaded structures.

    Supports:
    - Isotropic (default) and anisotropic (mode-specific) sampling
    - Reference ligand CoM shift for applicable modes
    - All generation modes from de novo to fragment_growing

    Returns
    -------
    dict  {center, points, n_atoms, has_prior_center, anisotropic, ref_ligand_com_shifted}
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    has_prior_center = prior_center_file is not None

    if has_prior_center:
        from flowr.gen.utils import load_prior_center

        center = load_prior_center(prior_center_file)  # np (3,)
    else:
        # Use pocket COM (matches generation pipeline)
        pocket_com = (
            _compute_pocket_com_worker(protein_path, ligand_path, pocket_cutoff)
            if protein_path
            else None
        )

        if pocket_com is not None:
            center = pocket_com
        else:
            # Fallback: use the ligand (fragment) centre of mass
            mol = _read_ligand_mol(ligand_path)
            if mol is not None:
                mol_noh = Chem.RemoveHs(mol)
                conf = mol_noh.GetConformer()
                coords = np.array(
                    [
                        list(conf.GetAtomPosition(i))
                        for i in range(mol_noh.GetNumAtoms())
                    ]
                )
                center = coords.mean(axis=0)
            else:
                center = np.zeros(3)

    # Apply reference ligand CoM shift if applicable
    _ref_com_shifted = False
    if ref_ligand_com_prior:
        ref_com = _compute_ref_ligand_com_shift_worker(
            ligand_path, gen_mode, fixed_atoms, ring_system_index
        )
        if ref_com is not None:
            center = ref_com
            _ref_com_shifted = True

    # Sample representative Gaussian cloud
    rng = np.random.default_rng(seed=42)
    _aniso_applied = False

    if anisotropic_prior:
        covariance = _compute_anisotropic_cloud_covariance_worker(
            ligand_path,
            gen_mode,
            center,
            has_prior_center,
            fixed_atoms,
            ring_system_index,
        )
        if covariance is not None:
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


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED POST-PROCESSING HELPERS (used by both SBDD and LBDD pipelines)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_strain_values(mols: list) -> List[float]:
    """Compute strain energy for each non-None molecule.

    Returns a list of rounded strain values (kcal/mol) for molecules
    where computation succeeded.
    """
    strain_vals: List[float] = []
    for mol in mols:
        if mol is not None:
            try:
                s = calc_strain(mol, add_hs=False)
                strain_vals.append(round(s, 2))
            except Exception:
                pass  # skip molecules where strain calc fails
    return strain_vals


def _build_result_dicts(  # NOSONAR
    gen_mols: list,
    optimized_mols: list,
    used_optimization: bool,
    extract_affinity: bool = False,
) -> List[Dict[str, Any]]:
    """Build per-molecule result dictionaries for the frontend.

    Parameters
    ----------
    gen_mols : list
        Original generated molecules (before optimization).
    optimized_mols : list
        Post-processed molecules (with Hs and/or optimized).
    used_optimization : bool
        Whether optimization was applied (controls display SDF).
    extract_affinity : bool
        If True, extract affinity predictions (pic50, pki, pkd, pec50)
        from molecule properties.  Used by the SBDD pipeline only.

    Returns
    -------
    list of dict
        Result dicts with id, smiles, sdf, atoms, bonds, properties.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    results: List[Dict[str, Any]] = []
    for idx, mol in enumerate(gen_mols):
        if mol is None:
            continue
        try:
            opt_mol = (
                optimized_mols[idx]
                if idx < len(optimized_mols) and optimized_mols[idx] is not None
                else mol
            )

            sdf_with_hs = _mol_to_sdf_string(opt_mol)
            mol_noh = Chem.RemoveHs(opt_mol)
            sdf_no_hs = _mol_to_sdf_string(mol_noh)

            if used_optimization:
                display_sdf = sdf_with_hs
                display_mol = opt_mol
            else:
                display_sdf = sdf_no_hs
                display_mol = mol_noh

            # Compute properties using shared utility
            props = _compute_all_properties(mol)
            # Add aliases expected by the frontend (avoid recomputing)
            props.update(
                {
                    "num_atoms": mol.GetNumAtoms(),
                    "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                    "mol_weight": props.get("MolWt"),
                    "tpsa": props.get("TPSA"),
                    "logp": props.get("LogP"),
                }
            )

            # Extract affinity predictions if present
            if extract_affinity:
                for aff_key in ("pic50", "pki", "pkd", "pec50"):
                    if mol.HasProp(aff_key):
                        try:
                            props[aff_key] = round(float(mol.GetProp(aff_key)), 4)
                        except (ValueError, TypeError):
                            pass

            results.append(
                {
                    "id": idx,
                    "smiles": _mol_to_smiles(mol),
                    "sdf": display_sdf,
                    "sdf_with_hs": sdf_with_hs,
                    "sdf_no_hs": sdf_no_hs,
                    "atoms": _mol_to_atom_info(display_mol),
                    "bonds": _mol_to_bond_info(display_mol),
                    "properties": props,
                }
            )
        except Exception:
            continue
    return results


def _compute_cloud_size(  # NOSONAR
    gen_mode: str,
    ligand_path: str,
    fixed_atoms: List[int],
    ring_system_index: int,
    grow_size: Optional[int],
    num_heavy_atoms: Optional[int],
) -> int:
    """Compute the prior-cloud size (number of variable atoms) for visualisation.

    For *de novo* mode, returns the user-specified heavy-atom count or the
    reference-ligand heavy-atom count, defaulting to 20.

    For inpainting modes, estimates the number of atoms that will be replaced
    based on scaffold / ring-system analysis of the reference ligand.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    if gen_mode == "denovo":
        if num_heavy_atoms:
            return num_heavy_atoms
        if ligand_path:
            try:
                _ref_dn = _read_ligand_mol(ligand_path)
                if _ref_dn is not None:
                    _ref_dn_noh = Chem.RemoveHs(_ref_dn)
                    return max(_ref_dn_noh.GetNumAtoms(), 1)
            except Exception:
                pass  # fallback to default cloud size below
        return 20

    # Non-de-novo modes

    # User atom selection takes priority over auto-detected masks
    _user_select_modes = {
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "core_growing",
    }
    if gen_mode in _user_select_modes and fixed_atoms:
        return len(fixed_atoms)

    cloud_size = grow_size if (gen_mode == "fragment_growing" and grow_size) else 20

    if gen_mode != "fragment_growing":
        try:
            _ref = _read_ligand_mol(ligand_path)
            if _ref is not None:
                _ref_noh = Chem.RemoveHs(_ref)
                from rdkit.Chem.Scaffolds.MurckoScaffold import (
                    GetScaffoldForMol as _GetScaffold,
                )

                if gen_mode == "scaffold_hopping":
                    _scaffold = _GetScaffold(Chem.Mol(_ref_noh))
                    cloud_size = (
                        _scaffold.GetNumAtoms() if _scaffold else _ref_noh.GetNumAtoms()
                    )
                elif gen_mode == "linker_inpainting":
                    _scaffold = _GetScaffold(Chem.Mol(_ref_noh))
                    if _scaffold:
                        _ring_atoms = set()
                        for ring in _ref_noh.GetRingInfo().AtomRings():
                            _ring_atoms.update(ring)
                        _s2 = Chem.Mol(_ref_noh)
                        for a in _s2.GetAtoms():
                            a.SetIntProp("org_idx", a.GetIdx())
                        _scf = _GetScaffold(_s2)
                        _scaffold_orig = (
                            {a.GetIntProp("org_idx") for a in _scf.GetAtoms()}
                            if _scf
                            else set()
                        )
                        cloud_size = len(
                            [a for a in _scaffold_orig if a not in _ring_atoms]
                        )
                        if cloud_size == 0:
                            cloud_size = _ref_noh.GetNumAtoms()
                    else:
                        cloud_size = _ref_noh.GetNumAtoms()
                elif gen_mode == "scaffold_elaboration":
                    _s2 = Chem.Mol(_ref_noh)
                    try:
                        Chem.SanitizeMol(_s2)
                    except Exception:
                        pass  # proceed with unsanitized mol copy
                    for a in _s2.GetAtoms():
                        a.SetIntProp("org_idx", a.GetIdx())
                    _scf = _GetScaffold(_s2)
                    _scaffold_set = (
                        {a.GetIntProp("org_idx") for a in _scf.GetAtoms()}
                        if _scf
                        else set()
                    )
                    _ring_set = set()
                    for ring in _s2.GetRingInfo().AtomRings():
                        _ring_set.update(ring)
                    n_replaced = sum(
                        1
                        for idx in range(_ref_noh.GetNumAtoms())
                        if idx not in _scaffold_set and idx not in _ring_set
                    )
                    cloud_size = max(n_replaced, 1)
                elif gen_mode == "core_growing":
                    _s2 = Chem.Mol(_ref_noh)
                    for a in _s2.GetAtoms():
                        a.SetIntProp("org_idx", a.GetIdx())
                    _scaffold = _GetScaffold(_s2)
                    if _scaffold:
                        _scaffold_atoms = {
                            a.GetIntProp("org_idx") for a in _scaffold.GetAtoms()
                        }
                        _ring_systems = []
                        ri = _s2.GetRingInfo()
                        for ring in ri.AtomRings():
                            rs = set(ring)
                            merged = False
                            for existing in _ring_systems:
                                if rs & existing:
                                    existing |= rs
                                    merged = True
                                    break
                            if not merged:
                                _ring_systems.append(rs)
                        _ridx = (
                            ring_system_index
                            if ring_system_index < len(_ring_systems)
                            else 0
                        )
                        _core = (
                            {a for a in _scaffold_atoms if a in _ring_systems[_ridx]}
                            if _ring_systems
                            else set()
                        )
                        cloud_size = (
                            _ref_noh.GetNumAtoms() - len(_core)
                            if _core
                            else _ref_noh.GetNumAtoms()
                        )
                    else:
                        cloud_size = _ref_noh.GetNumAtoms()
                elif gen_mode == "substructure_inpainting":
                    cloud_size = _ref_noh.GetNumAtoms() - len(fixed_atoms)
                    if cloud_size <= 0:
                        cloud_size = _ref_noh.GetNumAtoms()
                else:
                    cloud_size = _ref_noh.GetNumAtoms()
        except Exception:
            pass  # keep previously computed cloud_size

    return cloud_size


# ═══════════════════════════════════════════════════════════════════════════
#  FLOWR GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def _filter_against_history(gen_ligs, previous_smiles, threshold, vocab):  # NOSONAR
    """Filter generated ligands against previously generated SMILES for novelty."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    if not previous_smiles or not gen_ligs:
        return gen_ligs

    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    # Compute fingerprints for history
    history_fps = []
    for smi in previous_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            history_fps.append(fp)

    if not history_fps:
        return gen_ligs

    filtered = []
    for lig in gen_ligs:
        try:
            if hasattr(lig, "to_rdkit"):
                rdkit_mol = lig.to_rdkit(vocab=vocab)
            elif hasattr(lig, "GetNumAtoms"):
                # Already an RDKit Mol object (LBDD pipeline)
                rdkit_mol = lig
            else:
                rdkit_mol = None
            if rdkit_mol is None:
                filtered.append(lig)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=2048)
            max_sim = max(
                DataStructs.TanimotoSimilarity(fp, hfp) for hfp in history_fps
            )
            if max_sim <= threshold:
                filtered.append(lig)
        except Exception:
            filtered.append(lig)  # Keep on error

    return filtered


@dataclass
class _GenerationConfig:
    """Bundled configuration for _run_flowr_generation."""

    gen_mode: str
    fixed_atoms: List[int]
    n_samples: int
    batch_size: int
    integration_steps: int
    pocket_cutoff: float
    grow_size: Optional[int] = None
    prior_center_file: Optional[str] = None
    coord_noise_scale: float = 0.0
    filter_valid_unique: bool = True
    filter_cond_substructure: bool = False
    filter_diversity: bool = False
    diversity_threshold: float = 0.9
    sample_mol_sizes: bool = False
    filter_pb_valid: bool = False
    calculate_pb_valid: bool = False
    calculate_strain_energies: bool = False
    optimize_gen_ligs: bool = False
    optimize_gen_ligs_hs: bool = False
    anisotropic_prior: bool = False
    ref_ligand_com_prior: bool = False
    ring_system_index: int = 0
    num_heavy_atoms: Optional[int] = None
    property_filter: Optional[List[dict]] = None
    adme_filter: Optional[List[dict]] = None
    previous_smiles: Optional[List[str]] = None


def _run_flowr_generation(  # NOSONAR
    job: Dict[str, Any],
    config: _GenerationConfig,
) -> Dict[str, Any]:
    """Run actual FLOWR generation using the loaded model."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    gen_mode = config.gen_mode
    fixed_atoms = config.fixed_atoms
    n_samples = config.n_samples
    batch_size = config.batch_size
    integration_steps = config.integration_steps
    pocket_cutoff = config.pocket_cutoff
    grow_size = config.grow_size
    prior_center_file = config.prior_center_file
    coord_noise_scale = config.coord_noise_scale
    filter_valid_unique = config.filter_valid_unique
    filter_cond_substructure = config.filter_cond_substructure
    filter_diversity = config.filter_diversity
    diversity_threshold = config.diversity_threshold
    sample_mol_sizes = config.sample_mol_sizes
    filter_pb_valid = config.filter_pb_valid
    calculate_pb_valid = config.calculate_pb_valid
    calculate_strain_energies = config.calculate_strain_energies
    optimize_gen_ligs = config.optimize_gen_ligs
    optimize_gen_ligs_hs = config.optimize_gen_ligs_hs
    anisotropic_prior = config.anisotropic_prior
    ref_ligand_com_prior = config.ref_ligand_com_prior
    ring_system_index = config.ring_system_index
    num_heavy_atoms = config.num_heavy_atoms
    property_filter = config.property_filter
    adme_filter = config.adme_filter
    previous_smiles = config.previous_smiles

    _raise_if_cancelled(job, "before generation")
    # Snapshot all model state under lock to prevent races with concurrent /load-model
    with _model_lock:
        model = _model_state["model"]
        hparams = _model_state["hparams"]
        vocab = _model_state["vocab"]
        vocab_charges = _model_state["vocab_charges"]
        vocab_hybridization = _model_state.get("vocab_hybridization")
        vocab_aromatic = _model_state.get("vocab_aromatic")

    use_sub = gen_mode == "substructure_inpainting" and len(fixed_atoms) > 0
    # When the user manually selects atoms in a conditional mode other than
    # substructure_inpainting, we do NOT switch to substructure_inpainting
    # (that would change the prior).  Instead we pass the user’s atom selection
    # as custom_replace_indices on the interpolant so only the mask is overridden
    # while preserving the original mode’s prior (anisotropic, ref COM, etc.).
    _user_overrides_mask = (
        gen_mode not in ("substructure_inpainting", "denovo", "fragment_growing")
        and len(fixed_atoms) > 0
    )
    is_inpainting = gen_mode != "denovo"
    output_dir = WORK_DIR / job["job_id"] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build property / ADME filter pipeline ──
    _mol_filters: List = []
    if property_filter:
        criteria = [
            PropertyCriterion(
                name=pf["name"],
                min_val=pf.get("min"),
                max_val=pf.get("max"),
            )
            for pf in property_filter
            if pf.get("name")
        ]
        if criteria:
            _pf = PropertyFilter(criteria)
            _mol_filters.append(_pf)
            logger.info("[FLOWR] Property filter active: %s", _pf)
    if adme_filter:
        criteria = [
            ADMECriterion(
                name=af["name"],
                min_val=af.get("min"),
                max_val=af.get("max"),
                model_path=af.get("model_file"),
            )
            for af in adme_filter
            if af.get("name")
        ]
        if criteria:
            _af = ADMEFilter(criteria)
            _mol_filters.append(_af)
            logger.info("[FLOWR] ADME filter active: %s", _af)
    mol_filter_pipeline = MolFilterPipeline(_mol_filters)

    mode_flags = {
        "substructure_inpainting": use_sub,
        "substructure": fixed_atoms if use_sub else None,
        "scaffold_hopping": (gen_mode == "scaffold_hopping"),
        "scaffold_elaboration": (gen_mode == "scaffold_elaboration"),
        "linker_inpainting": (gen_mode == "linker_inpainting"),
        "core_growing": (gen_mode == "core_growing"),
        "fragment_growing": (gen_mode == "fragment_growing"),
    }

    if gen_mode == "fragment_growing":
        mode_flags["grow_size"] = grow_size
        mode_flags["prior_center_file"] = prior_center_file

    # De novo placeholder: empty-string ligand_path must become None so
    # process_complex enters the add_placeholder_ligand branch.
    _is_placeholder = not job["ligand_path"] and num_heavy_atoms is not None
    _effective_ligand = job["ligand_path"] if job["ligand_path"] else None

    args = _default_args(
        pdb_file=job["protein_path"],
        ligand_file=_effective_ligand,
        pocket_cutoff=pocket_cutoff,
        # Pocket-only mode: the uploaded PDB IS a pocket, don't try to cut
        cut_pocket=not _is_placeholder,
        sample_n_molecules_per_target=n_samples,
        batch_cost=batch_size,
        integration_steps=integration_steps,
        coord_noise_scale=coord_noise_scale,
        filter_cond_substructure=filter_cond_substructure,
        filter_valid_unique=filter_valid_unique,
        filter_diversity=filter_diversity,
        diversity_threshold=diversity_threshold,
        sample_mol_sizes=sample_mol_sizes,
        filter_pb_valid=filter_pb_valid,
        calculate_pb_valid=calculate_pb_valid,
        calculate_strain_energies=calculate_strain_energies,
        optimize_gen_ligs=optimize_gen_ligs,
        optimize_gen_ligs_hs=optimize_gen_ligs_hs,
        anisotropic_prior=anisotropic_prior,
        ref_ligand_com_prior=ref_ligand_com_prior,
        ring_system_index=ring_system_index,
        save_dir=str(output_dir),
        # De novo placeholder ligand support
        add_placeholder_ligand=_is_placeholder,
        num_heavy_atoms=num_heavy_atoms,
        **mode_flags,
    )

    _mode_inf_map = {
        "scaffold_hopping": "fragment",
        "scaffold_elaboration": "fragment",
        "linker_inpainting": "fragment",
        "core_growing": "fragment",
        "fragment_growing": "fragment",
        "substructure_inpainting": "fragment",
    }
    model.inpainting_mode = is_inpainting
    model.inpainting_mode_inf = _mode_inf_map.get(gen_mode) if is_inpainting else None
    logger.info(
        "[FLOWR] gen_mode=%s, inpainting_mode=%s, inpainting_mode_inf=%s",
        gen_mode,
        model.inpainting_mode,
        model.inpainting_mode_inf,
    )

    transform, interpolant = load_util(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    # If the user overrode the atom selection, pass their replace-indices
    # to the interpolant so it overrides the auto-detected mask while keeping
    # the original mode (and its prior behaviour).
    if _user_overrides_mask:
        interpolant.custom_replace_indices = fixed_atoms
        logger.info(
            "[FLOWR] User overrides atom mask for %s: %d atoms marked for replacement",
            gen_mode,
            len(fixed_atoms),
        )

    if gen_mode in (
        "core_growing",
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "fragment_growing",
    ):
        logger.debug(
            "[FLOWR] Interpolant flags — core_growing=%s, scaffold_hopping=%s, "
            "scaffold_elaboration=%s, linker_inpainting=%s, fragment_growing=%s, "
            "inpainting_mode=%s, inference=%s",
            interpolant.core_growing,
            interpolant.scaffold_hopping,
            interpolant.scaffold_elaboration,
            interpolant.linker_inpainting,
            interpolant.fragment_growing,
            interpolant.inpainting_mode,
            interpolant.inference,
        )

    system = load_data_from_pdb(
        args,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
        canonicalize_conformer=args.canonicalize_conformer,
    )

    systems = PocketComplexBatch([system])
    dataset = GeometricDataset(
        systems, data_cls=PocketComplexBatch, transform=transform
    )
    # Replicate the pocket so the DataLoader can form full-sized batches.
    # Use at least batch_size so each batch truly contains batch_size items.
    _dataset_size = max(n_samples, batch_size)
    dataset = dataset.sample_n_molecules_per_target(_dataset_size)

    inpainting_mode = gen_util.get_conditional_mode(args)
    all_gen_ligs: list = []
    k = 0
    max_sample_iter = args.max_sample_iter
    last_data = None

    metrics_log: List[str] = []
    _sub_total_before = 0
    _sub_total_after = 0

    original_ref_mol = None
    if filter_cond_substructure and inpainting_mode is not None:
        original_ref_mol = _read_ligand_mol(job["ligand_path"])
        if original_ref_mol is not None:
            if hparams.get("remove_hs", True):
                original_ref_mol = Chem.RemoveHs(original_ref_mol)
            try:
                Chem.SanitizeMol(original_ref_mol)
            except Exception as exc:
                logger.warning(
                    "[FLOWR] Original ref ligand sanitization failed: %s. "
                    "Substructure filter may not work correctly.",
                    exc,
                )
                original_ref_mol = None

    # Capture stdout AND stderr to detect library warnings.
    # stdout captures Python print() messages from the FLOWR library;
    # stderr captures RDKit C++ messages (e.g. "Can't kekulize mol").
    _captured_stdout = io.StringIO()
    _captured_stderr = io.StringIO()
    _gen_warnings: List[str] = []

    with (
        _TeeStream(sys.stdout, _captured_stdout, stream_name="stdout"),
        _TeeStream(sys.stderr, _captured_stderr, stream_name="stderr"),
        torch.no_grad(),
    ):
        _raw_generated = 0  # total molecules generated (before any filtering)
        _valid_count = 0  # molecules that passed sanitization
        _unique_count = 0  # molecules that passed uniqueness filter
        _batch_count = 0  # completed batches (for health checks)
        _diversity_input = 0
        _diversity_removed = 0
        _history_input = 0
        _history_removed = 0
        _pf_total_before = 0
        _pf_total_after = 0

        while len(all_gen_ligs) < n_samples and k <= max_sample_iter:
            _raise_if_cancelled(job, "sampling")
            dataloader = gen_util.get_dataloader(args, dataset, interpolant, iter=k)
            n_batches = len(dataloader) if hasattr(dataloader, "__len__") else 1

            for i, batch in enumerate(dataloader):
                # Stop generating once we have enough filtered molecules
                if len(all_gen_ligs) >= n_samples:
                    break

                _raise_if_cancelled(job, "sampling")
                total_iters_est = max(1, n_batches * min(max_sample_iter, 3))
                pct = 10 + int(75 * ((k * n_batches + i) / total_iters_est))
                job["progress"] = min(pct, 85)

                # Keep the idle watchdog alive while generating
                global _last_activity
                _last_activity = time.time()

                prior, data, _, _ = batch
                last_data = data

                if filter_cond_substructure and inpainting_mode is not None:
                    ref_ligs_batch = model._generate_ligs(
                        data, lig_mask=data["lig_mask"].bool(), scale=model.coord_scale
                    )
                else:
                    ref_ligs_batch = None

                gen_ligs_batch = generate_ligands_per_target(
                    args,
                    model,
                    prior=prior,
                    posterior=data,
                    pocket_noise=args.pocket_noise,
                    device=DEVICE,
                    save_traj=False,
                    iter=f"{k}_{i}",
                )

                _raw_generated += len(gen_ligs_batch)

                if filter_valid_unique:
                    gen_ligs_batch = smolRD.sanitize_list(
                        gen_ligs_batch,
                        filter_uniqueness=False,
                        sanitize=True,
                    )
                    _valid_count += len(gen_ligs_batch)
                    gen_ligs_batch = smolRD.sanitize_list(
                        gen_ligs_batch,
                        filter_uniqueness=True,
                        sanitize=False,
                    )
                    _unique_count += len(gen_ligs_batch)
                else:
                    _valid_count += len(gen_ligs_batch)
                    _unique_count += len(gen_ligs_batch)

                if args.filter_cond_substructure and inpainting_mode is not None:
                    num_before_sub = len(gen_ligs_batch)
                    if num_before_sub > 0:
                        filtered = False
                        if ref_ligs_batch:
                            try:
                                gen_ligs_batch = gen_util.filter_substructure(
                                    gen_ligs_batch,
                                    ref_ligs_batch[: len(gen_ligs_batch)],
                                    inpainting_mode=inpainting_mode,
                                    substructure_query=args.substructure,
                                    max_fragment_cuts=3,
                                    canonicalize_conformer=args.canonicalize_conformer,
                                )
                                filtered = True
                            except Exception as exc:
                                logger.warning(
                                    "[FLOWR] Substructure filter with tensor ref "
                                    "failed (%s), trying original ref mol...",
                                    exc,
                                )
                        if not filtered and original_ref_mol is not None:
                            try:
                                refs_fallback = [original_ref_mol] * num_before_sub
                                gen_ligs_batch = gen_util.filter_substructure(
                                    gen_ligs_batch,
                                    refs_fallback,
                                    inpainting_mode=inpainting_mode,
                                    substructure_query=args.substructure,
                                    max_fragment_cuts=3,
                                    canonicalize_conformer=args.canonicalize_conformer,
                                )
                                filtered = True
                            except Exception as exc:
                                logger.warning(
                                    "[FLOWR] Substructure filter with original "
                                    "ref also failed (%s), skipping filter.",
                                    exc,
                                )
                        num_after_sub = len(gen_ligs_batch)
                        _sub_total_before += num_before_sub
                        _sub_total_after += num_after_sub
                        if filtered:
                            match_rate = (
                                round(num_after_sub / num_before_sub, 2)
                                if num_before_sub > 0
                                else 0.0
                            )
                            logger.info(
                                "[FLOWR] Substructure match rate: %s (%d/%d)",
                                match_rate,
                                num_after_sub,
                                num_before_sub,
                            )

                # ── Property / ADME filter (before extending accumulated list) ──
                if mol_filter_pipeline.active:
                    _pf_before = len(gen_ligs_batch)
                    gen_ligs_batch = mol_filter_pipeline(gen_ligs_batch)
                    _pf_after = len(gen_ligs_batch)
                    _pf_total_before += _pf_before
                    _pf_total_after += _pf_after
                    if _pf_before > 0:
                        logger.info(
                            "[FLOWR] Property/ADME filter: %d/%d passed (rate %s)",
                            _pf_after,
                            _pf_before,
                            round(_pf_after / _pf_before, 2),
                        )

                _new_batch_count = len(gen_ligs_batch)
                all_gen_ligs.extend(gen_ligs_batch)

                if filter_diversity:
                    _diversity_input += _new_batch_count
                    _list_before = len(all_gen_ligs)
                    all_gen_ligs = gen_util.filter_diverse_ligands_bulk(
                        all_gen_ligs,
                        threshold=diversity_threshold,
                    )
                    _list_after = len(all_gen_ligs)
                    _diversity_removed += _list_before - _list_after

                # Filter against previously generated ligands (active learning)
                if previous_smiles and (filter_valid_unique or filter_diversity):
                    _hist_before = len(all_gen_ligs)
                    all_gen_ligs = _filter_against_history(
                        all_gen_ligs, previous_smiles, diversity_threshold, vocab
                    )
                    _hist_after = len(all_gen_ligs)
                    _history_input += _hist_before
                    _history_removed += _hist_before - _hist_after

                # ── Health checks (after all per-batch filtering) ──
                _batch_count += 1
                _run_health_checks(
                    batch_count=_batch_count,
                    gen_mode=gen_mode,
                    raw_generated=_raw_generated,
                    valid_count=_valid_count,
                    unique_count=_unique_count,
                    filter_valid_unique=filter_valid_unique,
                    filter_cond_substructure=filter_cond_substructure
                    and inpainting_mode is not None,
                    sub_total_before=_sub_total_before,
                    sub_total_after=_sub_total_after,
                    filter_diversity=filter_diversity,
                    diversity_input=_diversity_input,
                    diversity_removed=_diversity_removed,
                    previous_smiles=previous_smiles,
                    history_input=_history_input,
                    history_removed=_history_removed,
                )

            k += 1

    # Extract any library warnings from captured stdout + stderr
    _gen_warnings = _extract_generation_warnings(
        _captured_stdout.getvalue(),
        _captured_stderr.getvalue(),
        gen_mode=gen_mode,
        n_generated=len(all_gen_ligs),
    )

    data = last_data

    if data is None:
        return {
            "results": [],
            "metrics": ["No batches were processed \u2014 check input files."],
            "used_optimization": False,
            "prior_cloud": None,
        }

    if len(all_gen_ligs) == 0:
        metrics_log.append("No valid molecules generated after all iterations.")
    elif len(all_gen_ligs) < n_samples:
        metrics_log.append(
            f"Reached max iterations ({max_sample_iter}), "
            f"generated {len(all_gen_ligs)}/{n_samples} valid molecules."
        )

    if len(all_gen_ligs) > n_samples:
        all_gen_ligs = all_gen_ligs[:n_samples]

    gen_ligs = all_gen_ligs
    n_output = len(gen_ligs)
    metrics_log.append(f"Generated: {n_output} molecules")

    # ── Generation quality metrics ──
    if _raw_generated > 0:
        validity_rate = round(_valid_count / _raw_generated, 3)
        uniqueness_rate = round(_unique_count / max(_valid_count, 1), 3)
        metrics_log.append(
            f"Validity rate: {validity_rate} ({_valid_count}/{_raw_generated})"
        )
        metrics_log.append(
            f"Uniqueness rate: {uniqueness_rate} ({_unique_count}/{max(_valid_count, 1)})"
        )
        if k > 1:
            metrics_log.append(
                f"Sampling iterations: {k} (extra iterations to reach {n_output} valid molecules)"
            )
        logger.info(
            "[FLOWR] Generation metrics — raw=%d, valid=%d, unique=%d, "
            "diversity_passed=%d/%d, output=%d, iterations=%d",
            _raw_generated,
            _valid_count,
            _unique_count,
            _diversity_input - _diversity_removed,
            _diversity_input,
            n_output,
            k,
        )

    if filter_diversity and _diversity_input > 0:
        _diversity_passed = _diversity_input - _diversity_removed
        div_rate = round(_diversity_passed / _diversity_input, 3)
        metrics_log.append(
            f"Diversity filter rate: {div_rate} ({_diversity_passed}/{_diversity_input} passed, threshold={diversity_threshold})"
        )

    if previous_smiles and _history_input > 0:
        _history_passed = _history_input - _history_removed
        hist_rate = round(_history_passed / _history_input, 3)
        metrics_log.append(
            f"Cross-round novelty: {hist_rate} ({_history_passed}/{_history_input} passed vs. {len(previous_smiles)} previous)"
        )

    if mol_filter_pipeline.active and _pf_total_before > 0:
        pf_rate = round(_pf_total_after / _pf_total_before, 3)
        metrics_log.append(
            f"Property/ADME filter rate: {pf_rate} ({_pf_total_after}/{_pf_total_before} passed)"
        )

    if filter_cond_substructure and _sub_total_before > 0:
        sub_rate = round(_sub_total_after / _sub_total_before, 3)
        metrics_log.append(
            f"Substructure match rate: {sub_rate} "
            f"({_sub_total_after}/{_sub_total_before})"
        )

    job["progress"] = 92
    _raise_if_cancelled(job, "post-processing")

    # ── Post-processing: protonation & optimization ──
    all_gen_ligs_final = gen_ligs
    if optimize_gen_ligs or optimize_gen_ligs_hs:
        try:
            optimizer = LigandPocketOptimization(
                pocket_cutoff=pocket_cutoff, strip_invalid=True
            )
            all_hs = [Chem.AddHs(lig, addCoords=True) for lig in gen_ligs]
            all_hs_before = [Chem.Mol(lig) for lig in all_hs]
            ref_pdb_with_hs = model.retrieve_pdbs_with_hs(
                data, save_dir=output_dir / "ref_pdbs", save_idx=0
            )
            from functools import partial as _partial

            process_fn = _partial(
                gen_util.process_lig,
                pdb_file=ref_pdb_with_hs,
                optimizer=optimizer,
                add_ligand_hs=False,
                only_ligand_hs=optimize_gen_ligs_hs and not optimize_gen_ligs,
                process_pocket=True,
            )
            all_gen_ligs_final = [process_fn(lig) for lig in all_hs_before]
            rmsds = []
            for before, after in zip(all_hs_before, all_gen_ligs_final):
                if before is not None and after is not None:
                    try:
                        from rdkit.Chem import AllChem

                        rmsd = AllChem.GetBestRMS(before, after)
                        rmsds.append(round(rmsd, 3))
                    except Exception:
                        pass  # skip molecules where RMSD calc fails
            opt_label = "Full" if optimize_gen_ligs else "H-only"
            metrics_log.append(
                f"{opt_label} optimization: {len([x for x in all_gen_ligs_final if x is not None])}/{len(all_hs_before)} valid"
            )
            if rmsds:
                mean_rmsd = round(np.mean(rmsds), 3)
                metrics_log.append(f"Mean RMSD (before→after): {mean_rmsd} Å")
            strain_vals = _compute_strain_values(all_gen_ligs_final)
            if strain_vals:
                metrics_log.append(
                    f"Strain after opt (mean): {round(np.mean(strain_vals), 2)} kcal/mol"
                )
        except Exception as exc:
            metrics_log.append(f"Optimization failed: {exc}")
            if hparams.get("remove_hs", True):
                all_gen_ligs_final = [
                    Chem.AddHs(lig, addCoords=True) if lig is not None else None
                    for lig in gen_ligs
                ]
            else:
                all_gen_ligs_final = list(gen_ligs)
    else:
        if hparams.get("remove_hs", True):
            all_gen_ligs_final = [
                Chem.AddHs(lig, addCoords=True) if lig is not None else None
                for lig in gen_ligs
            ]
        else:
            all_gen_ligs_final = list(gen_ligs)

    # ── Calculate strain energies ──
    if calculate_strain_energies and not (optimize_gen_ligs or optimize_gen_ligs_hs):
        _raise_if_cancelled(job, "strain-energy")
        strain_vals = _compute_strain_values(all_gen_ligs_final)
        if strain_vals:
            metrics_log.append(
                f"Strain energy (mean): {round(np.mean(strain_vals), 2)} kcal/mol"
            )
            metrics_log.append(
                f"Strain energy (std):  {round(np.std(strain_vals), 2)} kcal/mol"
            )

    # ── PoseBusters validity ──
    if calculate_pb_valid or filter_pb_valid:
        _raise_if_cancelled(job, "posebusters")
        try:
            valid_pairs = [
                (gen, fin)
                for gen, fin in zip(gen_ligs, all_gen_ligs_final)
                if fin is not None
            ]
            if valid_pairs:
                gen_for_pb = [p[0] for p in valid_pairs]
                fin_for_pb = [p[1] for p in valid_pairs]
            else:
                gen_for_pb, fin_for_pb = [], []

            pb_valid = (
                evaluate_pb_validity(
                    fin_for_pb,
                    pdb_file=job["protein_path"],
                    return_list=True,
                )
                if fin_for_pb
                else []
            )
            if pb_valid:
                pb_mean = round(np.mean(pb_valid), 3)
                metrics_log.append(
                    f"PB-validity: {pb_mean:.1%} (mean), {round(np.std(pb_valid), 3):.1%} (std)"
                )
            if filter_pb_valid and pb_valid:
                n_before_pb = len(all_gen_ligs_final)
                all_gen_ligs_final = [fin for fin, v in zip(fin_for_pb, pb_valid) if v]
                gen_ligs = [gen for gen, v in zip(gen_for_pb, pb_valid) if v]
                metrics_log.append(
                    f"PB-valid filter: {len(all_gen_ligs_final)}/{n_before_pb} passed"
                )
        except Exception as exc:
            metrics_log.append(f"PB-validity failed: {exc}")

    job["progress"] = 96
    _raise_if_cancelled(job, "result-building")

    # ── Build results list ──
    used_optimization = optimize_gen_ligs or optimize_gen_ligs_hs
    results = _build_result_dicts(
        gen_ligs, all_gen_ligs_final, used_optimization, extract_affinity=True
    )
    # ── Compute prior cloud for visualisation (all modes incl. de novo) ──
    prior_cloud = None
    cloud_size = _compute_cloud_size(
        gen_mode,
        job["ligand_path"],
        fixed_atoms,
        ring_system_index,
        grow_size,
        num_heavy_atoms,
    )
    if gen_mode == "denovo":
        try:
            prior_cloud = _compute_prior_cloud(
                job["ligand_path"] or "",
                prior_center_file,
                cloud_size,
                protein_path=job.get("protein_path"),
                pocket_cutoff=pocket_cutoff if pocket_cutoff else 6.0,
                anisotropic_prior=anisotropic_prior,
                gen_mode=gen_mode,
                ring_system_index=ring_system_index,
                ref_ligand_com_prior=False,
                fixed_atoms=fixed_atoms,
            )
        except Exception as exc:
            logger.warning(
                "[FLOWR] De novo prior cloud computation failed (non-critical): %s", exc
            )
    else:
        try:
            prior_cloud = _compute_prior_cloud(
                job["ligand_path"],
                prior_center_file,
                cloud_size,
                protein_path=job.get("protein_path"),
                pocket_cutoff=pocket_cutoff if pocket_cutoff else 6.0,
                anisotropic_prior=anisotropic_prior,
                gen_mode=gen_mode,
                ring_system_index=ring_system_index,
                ref_ligand_com_prior=ref_ligand_com_prior,
                fixed_atoms=fixed_atoms,
            )
        except Exception as exc:
            logger.warning(
                "[FLOWR] Prior cloud computation failed (non-critical): %s", exc
            )

    return {
        "results": results,
        "metrics": metrics_log,
        "used_optimization": used_optimization,
        "prior_cloud": prior_cloud,
        "warnings": _gen_warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FLOWR LBDD (MOLECULE-ONLY) GENERATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _GenerationMolConfig:
    """Bundled configuration for _run_flowr_generation_mol."""

    n_samples: int
    batch_size: int
    integration_steps: int
    gen_mode: str = "denovo"
    fixed_atoms: Optional[List[int]] = None
    grow_size: Optional[int] = None
    prior_center_file: Optional[str] = None
    coord_noise_scale: float = 0.0
    filter_valid_unique: bool = True
    filter_cond_substructure: bool = False
    filter_diversity: bool = False
    diversity_threshold: float = 0.9
    sample_mol_sizes: bool = False
    calculate_strain_energies: bool = False
    optimize_method: str = "none"
    anisotropic_prior: bool = False
    ref_ligand_com_prior: bool = False
    ring_system_index: int = 0
    sample_n_molecules_per_mol: int = 1
    num_heavy_atoms: Optional[int] = None
    property_filter: Optional[List[dict]] = None
    adme_filter: Optional[List[dict]] = None
    previous_smiles: Optional[List[str]] = None


def _run_flowr_generation_mol(  # NOSONAR
    job: Dict[str, Any],
    config: _GenerationMolConfig,
) -> Dict[str, Any]:
    """Run molecule-only FLOWR generation (LBDD pipeline).

    Supports all generation modes identical to SBDD:
    denovo, scaffold_hopping, scaffold_elaboration, linker_inpainting,
    core_growing, fragment_inpainting, fragment_growing, substructure_inpainting.
    The only difference is that no protein/pocket is used.
    """
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    n_samples = config.n_samples
    batch_size = config.batch_size
    integration_steps = config.integration_steps
    gen_mode = config.gen_mode
    fixed_atoms = config.fixed_atoms if config.fixed_atoms is not None else []
    grow_size = config.grow_size
    prior_center_file = config.prior_center_file
    coord_noise_scale = config.coord_noise_scale
    filter_valid_unique = config.filter_valid_unique
    filter_cond_substructure = config.filter_cond_substructure
    filter_diversity = config.filter_diversity
    diversity_threshold = config.diversity_threshold
    sample_mol_sizes = config.sample_mol_sizes
    calculate_strain_energies = config.calculate_strain_energies
    optimize_method = config.optimize_method
    anisotropic_prior = config.anisotropic_prior
    ref_ligand_com_prior = config.ref_ligand_com_prior
    ring_system_index = config.ring_system_index
    sample_n_molecules_per_mol = config.sample_n_molecules_per_mol
    num_heavy_atoms = config.num_heavy_atoms
    property_filter = config.property_filter
    adme_filter = config.adme_filter
    previous_smiles = config.previous_smiles

    _raise_if_cancelled(job, "before generation")

    with _mol_model_lock:
        model = _mol_model_state["model"]
        hparams = _mol_model_state["hparams"]
        vocab = _mol_model_state["vocab"]
        vocab_charges = _mol_model_state["vocab_charges"]
        vocab_hybridization = _mol_model_state.get("vocab_hybridization")
        vocab_aromatic = _mol_model_state.get("vocab_aromatic")

    use_sub = gen_mode == "substructure_inpainting" and len(fixed_atoms) > 0
    # See SBDD counterpart: custom atom selection is passed via the interpolant’s
    # custom_replace_indices, NOT by switching to substructure_inpainting.
    _user_overrides_mask = (
        gen_mode not in ("substructure_inpainting", "denovo", "fragment_growing")
        and len(fixed_atoms) > 0
    )
    is_inpainting = gen_mode != "denovo"
    output_dir = WORK_DIR / job["job_id"] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build property / ADME filter pipeline ──
    _mol_filters: List = []
    if property_filter:
        criteria = [
            PropertyCriterion(
                name=pf["name"],
                min_val=pf.get("min"),
                max_val=pf.get("max"),
            )
            for pf in property_filter
            if pf.get("name")
        ]
        if criteria:
            _pf = PropertyFilter(criteria)
            _mol_filters.append(_pf)
            logger.info("[FLOWR-MOL] Property filter active: %s", _pf)
    if adme_filter:
        criteria = [
            ADMECriterion(
                name=af["name"],
                min_val=af.get("min"),
                max_val=af.get("max"),
                model_path=af.get("model_file"),
            )
            for af in adme_filter
            if af.get("name")
        ]
        if criteria:
            _af = ADMEFilter(criteria)
            _mol_filters.append(_af)
            logger.info("[FLOWR-MOL] ADME filter active: %s", _af)
    mol_filter_pipeline = MolFilterPipeline(_mol_filters)

    # ── Build mode flags (identical to SBDD) ──
    mode_flags = {
        "substructure_inpainting": use_sub,
        "substructure": fixed_atoms if use_sub else None,
        "scaffold_hopping": (gen_mode == "scaffold_hopping"),
        "scaffold_elaboration": (gen_mode == "scaffold_elaboration"),
        "linker_inpainting": (gen_mode == "linker_inpainting"),
        "core_growing": (gen_mode == "core_growing"),
        "fragment_growing": (gen_mode == "fragment_growing"),
    }
    if gen_mode == "fragment_growing":
        mode_flags["grow_size"] = grow_size
        mode_flags["prior_center_file"] = prior_center_file

    # De novo scratch: empty-string sdf_path must become None
    _is_scratch = not job["ligand_path"] and num_heavy_atoms is not None
    _effective_sdf = job["ligand_path"] if job["ligand_path"] else None

    args = _default_args_mol(
        sdf_path=_effective_sdf,
        sample_n_molecules=n_samples,
        sample_n_molecules_per_mol=sample_n_molecules_per_mol,
        batch_cost=batch_size,
        integration_steps=integration_steps,
        coord_noise_scale=coord_noise_scale,
        filter_valid_unique=filter_valid_unique,
        filter_cond_substructure=filter_cond_substructure,
        filter_diversity=filter_diversity,
        diversity_threshold=diversity_threshold,
        sample_mol_sizes=sample_mol_sizes,
        calculate_strain_energies=calculate_strain_energies,
        anisotropic_prior=anisotropic_prior,
        ref_ligand_com_prior=ref_ligand_com_prior,
        ring_system_index=ring_system_index,
        save_dir=str(output_dir),
        # De novo placeholder ligand support
        add_placeholder_ligand=_is_scratch,
        num_heavy_atoms=num_heavy_atoms,
        **mode_flags,
    )

    # ── Set inpainting mode on model (identical to SBDD) ──
    _mode_inf_map = {
        "scaffold_hopping": "fragment",
        "scaffold_elaboration": "fragment",
        "linker_inpainting": "fragment",
        "core_growing": "fragment",
        "fragment_growing": "fragment",
        "substructure_inpainting": "fragment",
    }
    model.inpainting_mode = is_inpainting
    model.inpainting_mode_inf = _mode_inf_map.get(gen_mode) if is_inpainting else None
    logger.info(
        "[FLOWR-MOL] gen_mode=%s, inpainting_mode=%s, inpainting_mode_inf=%s",
        gen_mode,
        model.inpainting_mode,
        model.inpainting_mode_inf,
    )

    transform, interpolant = load_util_mol(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    # If the user overrode the atom selection, pass their replace-indices
    # to the interpolant so it overrides the auto-detected mask while keeping
    # the original mode (and its prior behaviour).
    if _user_overrides_mask:
        interpolant.custom_replace_indices = fixed_atoms
        logger.info(
            "[FLOWR-MOL] User overrides atom mask for %s: %d atoms marked for replacement",
            gen_mode,
            len(fixed_atoms),
        )

    if gen_mode in (
        "core_growing",
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "fragment_growing",
    ):
        logger.debug(
            "[FLOWR-MOL] Interpolant flags — core_growing=%s, scaffold_hopping=%s, "
            "scaffold_elaboration=%s, linker_inpainting=%s, fragment_growing=%s, "
            "inpainting_mode=%s, inference=%s",
            interpolant.core_growing,
            interpolant.scaffold_hopping,
            interpolant.scaffold_elaboration,
            interpolant.linker_inpainting,
            interpolant.fragment_growing,
            interpolant.inpainting_mode,
            interpolant.inference,
        )

    # ── Load or create molecule data ──
    if not job["ligand_path"] and num_heavy_atoms is not None:
        # Scratch mode: create a placeholder molecule and transform it.
        # The placeholder only provides atom count + rough geometry for
        # de-novo generation.  We must apply the transform so that tensor
        # shapes (one-hot atomics/bonds/charges) match what the dataloader
        # and interpolant expect.
        from flowr.util.molrepr import GeometricMol as _GM

        placeholder = _GM.from_placeholder(num_heavy_atoms)
        placeholder = transform(placeholder)
        molecules = [placeholder] * n_samples
    else:
        molecules = load_data_from_sdf_mol(
            args,
            remove_hs=hparams["remove_hs"],
            remove_aromaticity=hparams["remove_aromaticity"],
            transform=transform,
            sample=sample_n_molecules_per_mol > 1,
            sample_n_molecules=None,
            sample_n_molecules_per_mol=(
                sample_n_molecules_per_mol if sample_n_molecules_per_mol > 1 else None
            ),
        )
    # GeometricMolSDFDataset lacks sample_n_molecules(); replicate entries
    # to match n_samples (mirrors SBDD's dataset.sample_n_molecules_per_target)
    if len(molecules) == 0:
        raise RuntimeError(
            "LBDD molecule loading returned an empty list. "
            "Check that the input SDF file contains valid molecules."
        )
    # Ensure at least batch_size entries so each DataLoader batch is full.
    _target_len = max(n_samples, batch_size)
    if sample_n_molecules_per_mol <= 1 and len(molecules) < _target_len:
        import math

        repeats = math.ceil(_target_len / len(molecules))
        molecules = (molecules * repeats)[:_target_len]
    elif sample_n_molecules_per_mol <= 1 and len(molecules) > _target_len:
        molecules = molecules[:_target_len]
    dataset = GeometricMolBatch(molecules)

    inpainting_mode = gen_util.get_conditional_mode(args)
    all_gen_mols: list = []
    k = 0
    max_sample_iter = args.max_sample_iter
    sample_target = n_samples

    metrics_log: List[str] = []
    _raw_generated = 0
    _valid_count = 0
    _unique_count = 0
    _batch_count = 0  # completed batches (for health checks)
    _sub_total_before = 0
    _sub_total_after = 0
    _diversity_input = 0
    _diversity_removed = 0
    _history_input = 0
    _history_removed = 0
    _pf_total_before = 0
    _pf_total_after = 0

    # Pre-load reference molecule for substructure filtering (mirrors SBDD)
    original_ref_mol = None
    if filter_cond_substructure and inpainting_mode is not None and job["ligand_path"]:
        original_ref_mol = _read_ligand_mol(job["ligand_path"])
        if original_ref_mol is not None:
            if hparams.get("remove_hs", True):
                original_ref_mol = Chem.RemoveHs(original_ref_mol)
            try:
                Chem.SanitizeMol(original_ref_mol)
            except Exception as exc:
                logger.warning(
                    "[FLOWR-MOL] Original ref mol sanitization failed: %s. "
                    "Substructure filter may not work correctly.",
                    exc,
                )
                original_ref_mol = None

    # ── Compute reference ligand COM for coordinate shift-back ──
    # The LBDD model operates in zero-COM space (GeometricInterpolant
    # centres molecules at origin).  Generated coordinates therefore
    # come out near (0,0,0).  We translate them back to the reference
    # ligand's original COM so they overlap with the uploaded structure
    # in the 3Dmol viewer.
    # The COM must match how GeometricInterpolant computed it:
    #   - remove_hs=True  → COM over heavy atoms only
    #   - remove_hs=False → COM over all atoms
    _ref_com = np.zeros(3)
    if job["ligand_path"]:
        _remove_hs = hparams.get("remove_hs", True)
        try:
            _ref_mol_com = _read_ligand_mol(job["ligand_path"])
            if _ref_mol_com is not None:
                _ref_for_com = (
                    Chem.RemoveHs(_ref_mol_com) if _remove_hs else _ref_mol_com
                )
                if _ref_for_com.GetNumConformers() > 0:
                    _conf_com = _ref_for_com.GetConformer()
                    _coords_com = np.array(
                        [
                            list(_conf_com.GetAtomPosition(a))
                            for a in range(_ref_for_com.GetNumAtoms())
                        ]
                    )
                    _ref_com = _coords_com.mean(axis=0)
        except Exception as exc:
            logger.warning(
                "[FLOWR-MOL] Failed to compute reference ligand COM: %s", exc
            )

    # Capture stdout AND stderr to detect library warnings
    _captured_stdout_mol = io.StringIO()
    _captured_stderr_mol = io.StringIO()
    _gen_warnings_mol: List[str] = []

    with (
        _TeeStream(sys.stdout, _captured_stdout_mol, stream_name="stdout"),
        _TeeStream(sys.stderr, _captured_stderr_mol, stream_name="stderr"),
        torch.no_grad(),
    ):
        while len(all_gen_mols) < sample_target and k <= max_sample_iter:
            _raise_if_cancelled(job, "sampling")
            dataloader = gen_util.get_dataloader(args, dataset, interpolant, iter=k)
            n_batches = len(dataloader) if hasattr(dataloader, "__len__") else 1

            for i, batch in enumerate(dataloader):
                # Stop generating once we have enough filtered molecules
                if len(all_gen_mols) >= sample_target:
                    break

                _raise_if_cancelled(job, "sampling")
                total_iters_est = max(1, n_batches * min(max_sample_iter, 3))
                pct = 10 + int(75 * ((k * n_batches + i) / total_iters_est))
                job["progress"] = min(pct, 85)

                global _last_activity
                _last_activity = time.time()

                prior, _, _, _ = batch

                gen_mols = generate_molecules(
                    args,
                    model=model,
                    prior=prior,
                    device=DEVICE,
                    save_traj=False,
                    iter=f"{k}_{i}",
                )

                # ── Translate generated mols back to reference ligand COM ──
                for _gm in gen_mols:
                    if _gm is None or _gm.GetNumConformers() == 0:
                        continue
                    _gc = _gm.GetConformer()
                    for _ai in range(_gm.GetNumAtoms()):
                        _pos = _gc.GetAtomPosition(_ai)
                        _gc.SetAtomPosition(
                            _ai,
                            Geometry.Point3D(
                                _pos.x + _ref_com[0],
                                _pos.y + _ref_com[1],
                                _pos.z + _ref_com[2],
                            ),
                        )

                _raw_generated += len(gen_mols)

                if filter_valid_unique:
                    gen_mols = smolRD.sanitize_list(
                        gen_mols,
                        filter_uniqueness=False,
                        sanitize=True,
                    )
                    _valid_count += len(gen_mols)
                    gen_mols = smolRD.sanitize_list(
                        gen_mols,
                        filter_uniqueness=True,
                        sanitize=False,
                    )
                    _unique_count += len(gen_mols)
                else:
                    _valid_count += len(gen_mols)
                    _unique_count += len(gen_mols)

                # ── Substructure filter (parity with SBDD) ──
                if filter_cond_substructure and inpainting_mode is not None:
                    num_before_sub = len(gen_mols)
                    if num_before_sub > 0 and original_ref_mol is not None:
                        try:
                            refs_for_filter = [original_ref_mol] * num_before_sub
                            gen_mols = gen_util.filter_substructure(
                                gen_mols,
                                refs_for_filter,
                                inpainting_mode=inpainting_mode,
                                substructure_query=args.substructure,
                                max_fragment_cuts=3,
                                canonicalize_conformer=args.canonicalize_conformer,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[FLOWR-MOL] Substructure filter failed (%s), "
                                "skipping filter for this batch.",
                                exc,
                            )
                        num_after_sub = len(gen_mols)
                        _sub_total_before += num_before_sub
                        _sub_total_after += num_after_sub
                        if num_before_sub > 0:
                            match_rate = round(num_after_sub / num_before_sub, 2)
                            logger.info(
                                "[FLOWR-MOL] Substructure filter: %d/%d (match rate %s)",
                                num_after_sub,
                                num_before_sub,
                                match_rate,
                            )

                # ── Property / ADME filter (before extending accumulated list) ──
                if mol_filter_pipeline.active:
                    _pf_before = len(gen_mols)
                    gen_mols = mol_filter_pipeline(gen_mols)
                    _pf_after = len(gen_mols)
                    _pf_total_before += _pf_before
                    _pf_total_after += _pf_after
                    if _pf_before > 0:
                        logger.info(
                            "[FLOWR-MOL] Property/ADME filter: %d/%d passed (rate %s)",
                            _pf_after,
                            _pf_before,
                            round(_pf_after / _pf_before, 2),
                        )

                _new_batch_count = len(gen_mols)
                all_gen_mols.extend(gen_mols)

                if filter_diversity:
                    _diversity_input += _new_batch_count
                    _list_before = len(all_gen_mols)
                    all_gen_mols = gen_util.filter_diverse_ligands_bulk(
                        all_gen_mols,
                        threshold=diversity_threshold,
                    )
                    _list_after = len(all_gen_mols)
                    _diversity_removed += _list_before - _list_after

                # Filter against previously generated ligands (active learning)
                if previous_smiles and (filter_valid_unique or filter_diversity):
                    _hist_before = len(all_gen_mols)
                    all_gen_mols = _filter_against_history(
                        all_gen_mols, previous_smiles, diversity_threshold, vocab
                    )
                    _hist_after = len(all_gen_mols)
                    _history_input += _hist_before
                    _history_removed += _hist_before - _hist_after

                # ── Health checks (after all per-batch filtering) ──
                _batch_count += 1
                _run_health_checks(
                    batch_count=_batch_count,
                    gen_mode=gen_mode,
                    raw_generated=_raw_generated,
                    valid_count=_valid_count,
                    unique_count=_unique_count,
                    filter_valid_unique=filter_valid_unique,
                    filter_cond_substructure=filter_cond_substructure
                    and inpainting_mode is not None,
                    sub_total_before=_sub_total_before,
                    sub_total_after=_sub_total_after,
                    filter_diversity=filter_diversity,
                    diversity_input=_diversity_input,
                    diversity_removed=_diversity_removed,
                    previous_smiles=previous_smiles,
                    history_input=_history_input,
                    history_removed=_history_removed,
                )

            k += 1

    # Extract any library warnings from captured stdout + stderr
    _gen_warnings_mol = _extract_generation_warnings(
        _captured_stdout_mol.getvalue(),
        _captured_stderr_mol.getvalue(),
        gen_mode=gen_mode,
        n_generated=len(all_gen_mols),
    )

    if len(all_gen_mols) == 0:
        return {
            "results": [],
            "metrics": ["No valid molecules generated after all iterations."],
            "used_optimization": False,
            "prior_cloud": None,
            "warnings": _gen_warnings_mol,
        }

    if len(all_gen_mols) > sample_target:
        all_gen_mols = all_gen_mols[:sample_target]

    n_output = len(all_gen_mols)
    metrics_log.append(f"Generated: {n_output} molecules")

    if _raw_generated > 0:
        validity_rate = round(_valid_count / _raw_generated, 3)
        uniqueness_rate = round(_unique_count / max(_valid_count, 1), 3)
        metrics_log.append(
            f"Validity rate: {validity_rate} ({_valid_count}/{_raw_generated})"
        )
        metrics_log.append(
            f"Uniqueness rate: {uniqueness_rate} ({_unique_count}/{max(_valid_count, 1)})"
        )
        if k > 1:
            metrics_log.append(
                f"Sampling iterations: {k} (extra iterations to reach {n_output} valid molecules)"
            )
        logger.info(
            "[FLOWR] Generation metrics — raw=%d, valid=%d, unique=%d, "
            "diversity_passed=%d/%d, output=%d, iterations=%d",
            _raw_generated,
            _valid_count,
            _unique_count,
            _diversity_input - _diversity_removed,
            _diversity_input,
            n_output,
            k,
        )

    if filter_diversity and _diversity_input > 0:
        _diversity_passed = _diversity_input - _diversity_removed
        div_rate = round(_diversity_passed / _diversity_input, 3)
        metrics_log.append(
            f"Diversity filter rate: {div_rate} ({_diversity_passed}/{_diversity_input} passed, threshold={diversity_threshold})"
        )

    if previous_smiles and _history_input > 0:
        _history_passed = _history_input - _history_removed
        hist_rate = round(_history_passed / _history_input, 3)
        metrics_log.append(
            f"Cross-round novelty: {hist_rate} ({_history_passed}/{_history_input} passed vs. {len(previous_smiles)} previous)"
        )

    if mol_filter_pipeline.active and _pf_total_before > 0:
        pf_rate = round(_pf_total_after / _pf_total_before, 3)
        metrics_log.append(
            f"Property/ADME filter rate: {pf_rate} ({_pf_total_after}/{_pf_total_before} passed)"
        )

    if filter_cond_substructure and _sub_total_before > 0:
        sub_rate = round(_sub_total_after / _sub_total_before, 3)
        metrics_log.append(
            f"Substructure match rate: {sub_rate} "
            f"({_sub_total_after}/{_sub_total_before})"
        )

    job["progress"] = 90
    _raise_if_cancelled(job, "post-processing")

    # ── Post-processing: protonation and optional optimization ──
    gen_mols_final = all_gen_mols
    used_optimization = False

    if optimize_method in ("rdkit", "xtb"):
        used_optimization = True
        # Add hydrogens first (required for optimization)
        gen_mols_hs = [Chem.AddHs(mol, addCoords=True) for mol in all_gen_mols]

        if optimize_method == "xtb":
            try:
                import tempfile

                optimized, energy_gains, rmsds = [], [], []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for mol in gen_mols_hs:
                        opt_mol, eg, rmsd = gen_util.optimize_molecule_xtb(mol, tmpdir)
                        if opt_mol is not None:
                            optimized.append(opt_mol)
                            energy_gains.append(eg if eg is not None else float("nan"))
                            rmsds.append(rmsd if rmsd is not None else float("nan"))
                        else:
                            optimized.append(mol)
                            energy_gains.append(float("nan"))
                            rmsds.append(float("nan"))
                gen_mols_final = smolRD.sanitize_list(
                    optimized,
                    filter_uniqueness=False,
                    sanitize=True,
                )
                metrics_log.append(
                    f"xTB optimization: {len(gen_mols_final)}/{len(gen_mols_hs)} valid"
                )
                valid_eg = [e for e in energy_gains if not np.isnan(e)]
                if valid_eg:
                    metrics_log.append(
                        f"Mean energy gain: {round(np.mean(valid_eg), 3)} kcal/mol"
                    )
            except Exception as exc:
                metrics_log.append(f"xTB optimization failed: {exc}")
                gen_mols_final = gen_mols_hs
        else:
            # rdkit optimization
            try:
                optimized, energy_gains, rmsds = [], [], []
                for mol in gen_mols_hs:
                    opt_mol, eg, rmsd = gen_util.optimize_molecule_rdkit(mol)
                    if opt_mol is not None:
                        optimized.append(opt_mol)
                        energy_gains.append(eg if eg is not None else float("nan"))
                        rmsds.append(rmsd if rmsd is not None else float("nan"))
                    else:
                        optimized.append(mol)
                        energy_gains.append(float("nan"))
                        rmsds.append(float("nan"))
                gen_mols_final = smolRD.sanitize_list(
                    optimized,
                    filter_uniqueness=False,
                    sanitize=True,
                )
                metrics_log.append(
                    f"RDKit optimization: {len(gen_mols_final)}/{len(gen_mols_hs)} valid"
                )
                valid_eg = [e for e in energy_gains if not np.isnan(e)]
                if valid_eg:
                    metrics_log.append(
                        f"Mean energy gain: {round(np.mean(valid_eg), 3)} kcal/mol"
                    )
            except Exception as exc:
                metrics_log.append(f"RDKit optimization failed: {exc}")
                gen_mols_final = gen_mols_hs
    else:
        # Just add Hs if model removes them
        if hparams.get("remove_hs", True):
            gen_mols_final = [
                Chem.AddHs(mol, addCoords=True) if mol is not None else None
                for mol in all_gen_mols
            ]

    # ── Calculate strain energies ──
    if calculate_strain_energies:
        _raise_if_cancelled(job, "strain-energy")
        strain_vals = _compute_strain_values(gen_mols_final)
        if strain_vals:
            metrics_log.append(
                f"Strain energy (mean): {round(np.mean(strain_vals), 2)} kcal/mol"
            )

    job["progress"] = 96
    _raise_if_cancelled(job, "result-building")

    # ── Build results list ──
    results = _build_result_dicts(all_gen_mols, gen_mols_final, used_optimization)

    # ── Compute prior cloud for visualisation (all modes incl. de novo) ──
    prior_cloud = None
    cloud_size = _compute_cloud_size(
        gen_mode,
        job["ligand_path"],
        fixed_atoms,
        ring_system_index,
        grow_size,
        num_heavy_atoms,
    )
    if gen_mode == "denovo":
        try:
            prior_cloud = _compute_prior_cloud(
                job["ligand_path"] or "",
                prior_center_file,
                cloud_size,
                protein_path=None,  # LBDD: no protein
                pocket_cutoff=6.0,
                anisotropic_prior=anisotropic_prior,
                gen_mode=gen_mode,
                ring_system_index=ring_system_index,
                ref_ligand_com_prior=False,
                fixed_atoms=fixed_atoms,
            )
        except Exception as exc:
            logger.warning(
                "[FLOWR-MOL] De novo prior cloud computation failed (non-critical): %s",
                exc,
            )
    else:
        try:
            prior_cloud = _compute_prior_cloud(
                job["ligand_path"],
                prior_center_file,
                cloud_size,
                protein_path=None,  # LBDD: no protein
                pocket_cutoff=6.0,
                anisotropic_prior=anisotropic_prior,
                gen_mode=gen_mode,
                ring_system_index=ring_system_index,
                ref_ligand_com_prior=ref_ligand_com_prior,
                fixed_atoms=fixed_atoms,
            )
        except Exception as exc:
            logger.warning(
                "[FLOWR-MOL] Prior cloud computation failed (non-critical): %s", exc
            )

    return {
        "results": results,
        "metrics": metrics_log,
        "used_optimization": used_optimization,
        "prior_cloud": prior_cloud,
        "warnings": _gen_warnings_mol,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BACKGROUND GENERATION WORKER
# ═══════════════════════════════════════════════════════════════════════════


def _generation_worker(job_id: str, req: dict):  # NOSONAR
    """Run generation in a background thread, updating job state."""
    # Acknowledged: high cognitive complexity — refactoring deferred (python:S3776).
    _sanitize_job_id(job_id)
    global _last_activity
    job = JOBS[job_id]
    job["status"] = "queued"
    job["progress"] = 0
    _last_activity = time.time()  # keep idle watchdog alive

    # Serialize GPU access — only one generation at a time
    wait_deadline = time.time() + 600
    acquired = False
    while time.time() < wait_deadline:
        _raise_if_cancelled(job, "queue")
        acquired = _generation_semaphore.acquire(timeout=1)
        if acquired:
            break
    if not acquired:
        job.update(
            status="failed",
            progress=0,
            error="Another generation is already running. Please wait.",
        )
        return

    job["status"] = "generating"
    job["progress"] = 2

    # Cap input values to prevent GPU OOM from unchecked requests
    _MAX_SAMPLES = 500
    _MAX_BATCH = 200
    req["n_samples"] = min(req.get("n_samples", 10), _MAX_SAMPLES)
    req["batch_size"] = min(req.get("batch_size", 25), _MAX_BATCH)

    try:
        _raise_if_cancelled(job, "startup")
        # Download input files from frontend server
        job_dir = WORK_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        workflow_type = req.get("workflow_type", "sbdd")
        is_lbdd = workflow_type == "lbdd"

        num_heavy_atoms = req.get("num_heavy_atoms")

        if is_lbdd:
            # LBDD: need a ligand/molecule SDF — unless de novo scratch mode
            ligand_url = req.get("ligand_url")
            if ligand_url:
                _raise_if_cancelled(job, "download")
                ligand_filename = req.get("ligand_filename") or "molecule.sdf"
                ligand_path = job_dir / Path(ligand_filename).name
                if not _download_file(ligand_url, ligand_path):
                    raise RuntimeError(f"Failed to download ligand from {ligand_url}")
                job["ligand_path"] = str(ligand_path)
            elif num_heavy_atoms and num_heavy_atoms > 0:
                # Scratch mode: no ligand file — will use placeholder
                job["ligand_path"] = ""
                job["num_heavy_atoms"] = num_heavy_atoms
            else:
                raise RuntimeError(
                    "LBDD generation requires a ligand/molecule file or num_heavy_atoms."
                )
        else:
            # SBDD: need protein; ligand is optional (pocket-only mode)
            protein_path = job_dir / Path(req["protein_filename"]).name

            job["progress"] = 3
            _raise_if_cancelled(job, "download")
            if not _download_file(req["protein_url"], protein_path):
                raise RuntimeError(
                    f"Failed to download protein from {req['protein_url']}"
                )
            job["protein_path"] = str(protein_path)

            ligand_url = req.get("ligand_url")
            if ligand_url and req.get("ligand_filename"):
                _raise_if_cancelled(job, "download")
                ligand_path = job_dir / Path(req["ligand_filename"]).name
                if not _download_file(ligand_url, ligand_path):
                    raise RuntimeError(f"Failed to download ligand from {ligand_url}")
                job["ligand_path"] = str(ligand_path)
            elif num_heavy_atoms and num_heavy_atoms > 0:
                # Pocket-only mode: no ligand file — will use placeholder
                job["ligand_path"] = ""
                job["num_heavy_atoms"] = num_heavy_atoms
            else:
                raise RuntimeError(
                    "SBDD generation requires a ligand file or num_heavy_atoms."
                )

        # Download prior center file if provided (SBDD and LBDD)
        prior_center_path = None
        if req.get("prior_center_url") and req.get("prior_center_filename"):
            _raise_if_cancelled(job, "download")
            prior_path = job_dir / Path(req["prior_center_filename"]).name
            if _download_file(req["prior_center_url"], prior_path):
                prior_center_path = str(prior_path)

        # Download ADME model files if provided
        if req.get("adme_filter"):
            for af in req["adme_filter"]:
                _raise_if_cancelled(job, "download")
                model_url = af.get("model_url")
                model_fname = af.get("model_file")
                if model_url and model_fname:
                    model_dest = job_dir / model_fname
                    if _download_file(model_url, model_dest):
                        af["model_file"] = str(model_dest)
                        logger.info("[FLOWR] Downloaded ADME model: %s", model_fname)
                    else:
                        logger.warning(
                            "[FLOWR] Failed to download ADME model from %s", model_url
                        )

        job["progress"] = 5

        # If a finetuned checkpoint URL is provided, download it from the server
        finetuned_url = req.get("finetuned_ckpt_url")
        if finetuned_url:
            local_ckpt = job_dir / "finetuned_last.ckpt"
            if not local_ckpt.exists():
                _raise_if_cancelled(job, "download")
                logger.info("[FLOWR] Downloading finetuned checkpoint from server...")
                if not _download_file_streaming(finetuned_url, local_ckpt):
                    raise RuntimeError(
                        "Failed to download finetuned checkpoint from server"
                    )
                logger.info(
                    "[FLOWR] Finetuned checkpoint downloaded: %.1f MB",
                    local_ckpt.stat().st_size / 1024 / 1024,
                )
            req["ckpt_path"] = str(local_ckpt)

        # Ensure correct model is loaded
        ckpt = req.get("ckpt_path") or CKPT_PATH

        if is_lbdd:
            _raise_if_cancelled(job, "model-load")
            if (
                not _mol_model_state["loaded"]
                or _mol_model_state.get("ckpt_path") != ckpt
            ):
                job["status"] = "loading_model"
                if not _load_mol_model_if_needed(ckpt):
                    raise RuntimeError(
                        f"Mol model loading failed: {_mol_model_state.get('error', 'unknown')}"
                    )
                job["status"] = "generating"
        else:
            _raise_if_cancelled(job, "model-load")
            if not _model_state["loaded"] or _model_state.get("ckpt_path") != ckpt:
                job["status"] = "loading_model"
                if not _load_model_if_needed(ckpt):
                    raise RuntimeError(
                        f"Model loading failed: {_model_state.get('error', 'unknown')}"
                    )
                job["status"] = "generating"

        start = time.time()
        _last_activity = time.time()  # refresh after model load
        job["progress"] = 10
        _raise_if_cancelled(job, "generation")

        if is_lbdd:
            gen_out = _run_flowr_generation_mol(
                job,
                config=_GenerationMolConfig(
                    n_samples=req["n_samples"],
                    batch_size=req["batch_size"],
                    integration_steps=req["integration_steps"],
                    gen_mode=req.get("gen_mode", "denovo"),
                    fixed_atoms=req.get("fixed_atoms", []),
                    grow_size=req.get("grow_size"),
                    prior_center_file=prior_center_path,
                    coord_noise_scale=req.get("coord_noise_scale", 0.0),
                    filter_valid_unique=req.get("filter_valid_unique", True),
                    filter_cond_substructure=req.get("filter_cond_substructure", False),
                    filter_diversity=req.get("filter_diversity", False),
                    diversity_threshold=req.get("diversity_threshold", 0.9),
                    sample_mol_sizes=req.get("sample_mol_sizes", False),
                    calculate_strain_energies=req.get(
                        "calculate_strain_energies", False
                    ),
                    optimize_method=req.get("optimize_method", "none"),
                    anisotropic_prior=req.get("anisotropic_prior", False),
                    ref_ligand_com_prior=req.get("ref_ligand_com_prior", False),
                    ring_system_index=req.get("ring_system_index", 0),
                    sample_n_molecules_per_mol=req.get("sample_n_molecules_per_mol", 1),
                    num_heavy_atoms=num_heavy_atoms,
                    property_filter=req.get("property_filter"),
                    adme_filter=req.get("adme_filter"),
                    previous_smiles=req.get("previous_smiles"),
                ),
            )
        else:
            gen_out = _run_flowr_generation(
                job,
                config=_GenerationConfig(
                    gen_mode=req["gen_mode"],
                    fixed_atoms=req["fixed_atoms"],
                    n_samples=req["n_samples"],
                    batch_size=req["batch_size"],
                    integration_steps=req["integration_steps"],
                    pocket_cutoff=req["pocket_cutoff"],
                    grow_size=req.get("grow_size"),
                    prior_center_file=prior_center_path,
                    coord_noise_scale=req.get("coord_noise_scale", 0.0),
                    filter_valid_unique=req.get("filter_valid_unique", True),
                    filter_cond_substructure=req.get("filter_cond_substructure", False),
                    filter_diversity=req.get("filter_diversity", False),
                    diversity_threshold=req.get("diversity_threshold", 0.9),
                    sample_mol_sizes=req.get("sample_mol_sizes", False),
                    filter_pb_valid=req.get("filter_pb_valid", False),
                    calculate_pb_valid=req.get("calculate_pb_valid", False),
                    calculate_strain_energies=req.get(
                        "calculate_strain_energies", False
                    ),
                    optimize_gen_ligs=req.get("optimize_gen_ligs", False),
                    optimize_gen_ligs_hs=req.get("optimize_gen_ligs_hs", False),
                    anisotropic_prior=req.get("anisotropic_prior", False),
                    ref_ligand_com_prior=req.get("ref_ligand_com_prior", False),
                    ring_system_index=req.get("ring_system_index", 0),
                    num_heavy_atoms=num_heavy_atoms,
                    property_filter=req.get("property_filter"),
                    adme_filter=req.get("adme_filter"),
                    previous_smiles=req.get("previous_smiles"),
                ),
            )

        _raise_if_cancelled(job, "finalize")
        elapsed = round(time.time() - start, 2)
        job.update(
            status="completed",
            progress=100,
            results=gen_out["results"],
            metrics=gen_out.get("metrics", []),
            elapsed_time=elapsed,
            mode="flowr",
            n_generated=len(gen_out["results"]),
            used_optimization=gen_out.get("used_optimization", False),
            prior_cloud=gen_out.get("prior_cloud"),
            warnings=gen_out.get("warnings", []),
        )

    except _HealthCheckFailed as exc:
        logger.warning(
            "Health check failed for job %s: [%s] %s",
            job_id,
            exc.check_type,
            exc.message,
        )
        job.update(
            status="failed",
            progress=0,
            error=exc.message,
            health_check_type=exc.check_type,
            health_check_advice=exc.advice,
        )
    except _JobCancelled as exc:
        job.update(status="cancelled", error=str(exc))
    except Exception as exc:
        logger.exception("Generation failed for job %s", job_id)
        err_msg = str(exc)
        if "CUDA out of memory" in err_msg or "OutOfMemoryError" in err_msg:
            err_msg = "CUDA out of memory \u2013 reduce batch size!"
        job.update(status="failed", progress=0, error=err_msg)
    finally:
        # Cleanup GPU memory after generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _generation_semaphore.release()
        # Clean up temporary files after releasing semaphore to avoid
        # blocking future generations if rmtree hangs (e.g. NFS issues)
        job_dir = WORK_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
#  ROUTES
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    global _last_activity
    _last_activity = time.time()
    return {
        "status": "ok",
        "device": DEVICE,
        "flowr": FLOWR_AVAILABLE,
        "flowr_mol": FLOWR_MOL_AVAILABLE,
        "rdkit": RDKIT_AVAILABLE,
        "model_loaded": _model_state["loaded"],
        "model_loading": _model_state["loading"],
        "model_error": _model_state["error"],
        "ckpt_path": _model_state.get("ckpt_path"),
        "mol_model_loaded": _mol_model_state["loaded"],
        "mol_model_loading": _mol_model_state["loading"],
        "mol_model_error": _mol_model_state["error"],
        "mol_ckpt_path": _mol_model_state.get("ckpt_path"),
    }


@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown. The worker will exit after responding."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested — exiting after current work completes.")

    def _do_exit():
        time.sleep(1)
        os._exit(0)

    threading.Thread(target=_do_exit, daemon=True).start()
    return {"status": "shutting_down"}


@app.post("/load-model")
async def load_model_endpoint(request: LoadModelRequest = None):
    """Trigger model loading (non-blocking)."""
    ckpt = (request.ckpt_path if request else None) or CKPT_PATH

    if _model_state["loaded"] and _model_state.get("ckpt_path") == ckpt:
        return {"status": "loaded", "device": DEVICE, "ckpt_path": ckpt}
    if _model_state["loading"]:
        return {"status": "loading"}

    # Prevent model swap while a generation is in progress.
    # We transfer semaphore ownership to the loading thread so no generation
    # can start until loading completes (avoids GPU OOM from two models).
    if not _generation_semaphore.acquire(blocking=False):
        raise HTTPException(
            409, "Cannot load a new model while generation is in progress."
        )

    def _load_and_release(ckpt_path):
        try:
            _load_model_if_needed(ckpt_path)
        finally:
            _generation_semaphore.release()

    thread = threading.Thread(target=_load_and_release, args=(ckpt,), daemon=True)
    thread.start()
    return {"status": "loading", "message": "Model loading started…", "ckpt_path": ckpt}


@app.get("/model-status")
async def model_status():
    return {
        "loaded": _model_state["loaded"],
        "loading": _model_state["loading"],
        "error": _model_state["error"],
        "device": DEVICE,
        "ckpt_path": _model_state.get("ckpt_path"),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ACTIVE LEARNING FINETUNING
# ═══════════════════════════════════════════════════════════════════════════


def _active_learning_worker(job_id: str, req: dict):  # NOSONAR
    """Run LoRA finetuning in a background thread, updating job state."""
    _sanitize_job_id(job_id)
    global _last_activity
    job = JOBS[job_id]
    job["status"] = "finetuning"
    job["progress"] = 0
    _last_activity = time.time()

    # Serialize GPU access
    wait_deadline = time.time() + 600
    acquired = False
    while time.time() < wait_deadline:
        if job.get("cancelled"):
            job.update(status="cancelled", error=_ERR_CANCELLED)
            return
        acquired = _generation_semaphore.acquire(timeout=1)
        if acquired:
            break
    if not acquired:
        job.update(
            status="failed",
            progress=0,
            error="GPU is busy. Please wait for the current task to finish.",
        )
        return

    try:
        job["progress"] = 2
        job_dir = WORK_DIR / job_id
        # Clean stale data from previous finetuning run so data is reprocessed
        data_dir = job_dir / "data"
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir, ignore_errors=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        _raise_if_cancelled(job, "before data download")

        # Download protein PDB from server
        protein_path = job_dir / Path(req.get("protein_filename", "protein.pdb")).name
        if not _download_file(req["protein_url"], protein_path):
            raise RuntimeError(f"Failed to download protein from {req['protein_url']}")

        job["progress"] = 3
        _raise_if_cancelled(job, "before data preparation")

        # Prepare data and run finetuning
        from flowr.active_learning import prepare_al_data, run_al_finetuning

        # If a finetuned checkpoint URL is provided, download it from the server
        finetuned_url = req.get("finetuned_ckpt_url")
        if finetuned_url:
            local_ckpt = job_dir / "finetuned_last.ckpt"
            if not local_ckpt.exists():
                _raise_if_cancelled(job, "download ckpt")
                logger.info("[AL] Downloading finetuned checkpoint from server...")
                if not _download_file_streaming(finetuned_url, local_ckpt):
                    raise RuntimeError(
                        "Failed to download finetuned checkpoint from server"
                    )
                logger.info(
                    "[AL] Finetuned checkpoint downloaded: %.1f MB",
                    local_ckpt.stat().st_size / 1024 / 1024,
                )
            req["ckpt_path"] = str(local_ckpt)

        ckpt = req.get("ckpt_path") or CKPT_PATH
        ligand_sdf_strings = req["ligand_sdf_strings"]

        def progress_cb(pct, msg, phase="preparing"):
            global _last_activity
            _last_activity = time.time()
            job["progress"] = pct
            job["status_message"] = msg
            job["al_phase"] = phase
            # Check cancellation during long-running callbacks
            _raise_if_cancelled(job, f"finetuning ({phase})")

        # Prepare LMDB dataset
        data_dir = job_dir / "data"
        data_info = prepare_al_data(
            protein_pdb_path=str(protein_path),
            ligand_sdf_strings=ligand_sdf_strings,
            output_dir=str(data_dir),
            pocket_cutoff=req.get("pocket_cutoff", 7.0),
            remove_hs=True,  # Match base model
            progress_callback=progress_cb,
        )

        _raise_if_cancelled(job, "before training")

        # Run LoRA finetuning
        save_dir = job_dir / "ckpt"
        finetuned_ckpt_path = run_al_finetuning(
            ckpt_path=ckpt,
            data_path=data_info["data_path"],
            splits_path=data_info["splits_path"],
            statistics_path=data_info["statistics_path"],
            save_dir=str(save_dir),
            n_ligands=len(ligand_sdf_strings),
            epochs=req.get("epochs"),
            lora_rank=req.get("lora_rank", 16),
            lora_alpha=req.get("lora_alpha", 32),
            lr=req.get("lr", 5e-4),
            batch_cost=req.get("batch_cost", 4),
            acc_batches=req.get("acc_batches"),
            is_continuation=bool(finetuned_url),
            progress_callback=progress_cb,
        )

        # Invalidate cached models so the next generation loads the finetuned one.
        # Must clear BOTH SBDD and LBDD caches, and set model=None to free GPU memory.
        with _model_lock:
            old_model = _model_state.get("model")
            _model_state.update(
                loaded=False, loading=False, ckpt_path=None, model=None, error=None
            )
        with _mol_model_lock:
            old_mol_model = _mol_model_state.get("model")
            _mol_model_state.update(
                loaded=False, loading=False, ckpt_path=None, model=None, error=None
            )
        # Release references so GC can free GPU tensors
        del old_model, old_mol_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        job.update(
            status="completed",
            progress=100,
            finetuned_ckpt_path=finetuned_ckpt_path,
            n_ligands=len(ligand_sdf_strings),
        )

    except _JobCancelled:
        logger.info("[AL] Finetuning cancelled for job %s", job_id)
        job.update(status="cancelled", progress=0, error=_ERR_CANCELLED)
    except Exception as exc:
        logger.exception("Active learning failed for job %s", job_id)
        err_msg = str(exc)
        if "CUDA out of memory" in err_msg or "OutOfMemoryError" in err_msg:
            err_msg = (
                "CUDA out of memory during finetuning \u2013 "
                "try reducing the batch size."
            )
        job.update(status="failed", progress=0, error=err_msg)
    finally:
        # Cleanup GPU memory after active learning
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _generation_semaphore.release()
        # Don't clean up job_dir here — the checkpoint needs to persist
        # until the server copies/references it. Cleanup via TTL.


@app.post("/active-learning")
async def active_learning(request: ActiveLearningRequest):
    """Start an active learning LoRA finetuning job."""
    global _last_activity
    _last_activity = time.time()
    job_id = _sanitize_job_id(request.job_id)

    _active_states = {"queued", "loading_model", "generating", "finetuning"}
    if job_id in JOBS and JOBS[job_id].get("status") in _active_states:
        raise HTTPException(409, "A task is already in progress for this job.")

    if not request.ligand_sdf_strings:
        raise HTTPException(400, "No ligands provided for finetuning.")

    with _jobs_lock:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "finetuning",
            "progress": 0,
            "created_at": time.time(),
        }

    req = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    thread = threading.Thread(
        target=_active_learning_worker, args=(job_id, req), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "finetuning"}


@app.get("/al-job/{job_id}")
async def get_al_job(job_id: str):
    """Poll active learning job progress."""
    _sanitize_job_id(job_id)
    global _last_activity
    _last_activity = time.time()
    if job_id not in JOBS:
        raise HTTPException(404, _ERR_JOB_NOT_FOUND)
    job = JOBS[job_id]

    resp: Dict[str, Any] = {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "status_message": job.get("status_message", ""),
        "al_phase": job.get("al_phase", "preparing"),
    }
    if job.get("status") == "completed":
        resp["finetuned_ckpt_path"] = job.get("finetuned_ckpt_path")
        resp["n_ligands"] = job.get("n_ligands", 0)
    return resp


@app.get("/al-ckpt/{job_id}")
async def serve_al_checkpoint(job_id: str):
    """Serve the finetuned checkpoint file for download by the server."""
    _sanitize_job_id(job_id)
    global _last_activity
    _last_activity = time.time()
    if job_id not in JOBS:
        raise HTTPException(404, _ERR_JOB_NOT_FOUND)
    job = JOBS[job_id]
    ckpt_path = job.get("finetuned_ckpt_path")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise HTTPException(404, "Checkpoint file not found.")
    return FileResponse(
        str(ckpt_path), media_type="application/octet-stream", filename="last.ckpt"
    )


@app.post("/generate")
async def generate(request: GenerationRequest):
    """Start a generation job. Returns immediately; poll /job/{id} for progress."""
    global _last_activity
    _last_activity = time.time()
    job_id = _sanitize_job_id(request.job_id)

    _active_states = {"queued", "loading_model", "generating"}
    if job_id in JOBS and JOBS[job_id].get("status") in _active_states:
        raise HTTPException(409, "Generation already in progress for this job.")

    with _jobs_lock:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": time.time(),
        }

    req = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    thread = threading.Thread(
        target=_generation_worker, args=(job_id, req), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "generating"}


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Request cancellation of a running generation job."""
    _sanitize_job_id(job_id)
    if job_id not in JOBS:
        raise HTTPException(404, _ERR_JOB_NOT_FOUND)
    job = JOBS[job_id]
    job["cancelled"] = True
    job["status"] = "cancelled"
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    _sanitize_job_id(job_id)
    global _last_activity
    _last_activity = time.time()
    if job_id not in JOBS:
        raise HTTPException(404, _ERR_JOB_NOT_FOUND)
    job = JOBS[job_id]

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
            mode=job.get("mode", "flowr"),
            results=job.get("results", []),
            metrics=job.get("metrics", []),
            used_optimization=job.get("used_optimization", False),
            prior_cloud=job.get("prior_cloud"),
            warnings=job.get("warnings", []),
        )
    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _idle_watchdog():
    """Background thread that exits the process after idle timeout."""
    global _shutdown_requested
    while not _shutdown_requested:
        time.sleep(10)
        _cleanup_expired_jobs()
        if IDLE_TIMEOUT > 0 and (time.time() - _last_activity) > IDLE_TIMEOUT:
            # Check no jobs are running
            running = any(
                j.get("status") in ("queued", "generating", "loading_model")
                for j in list(JOBS.values())  # NOSONAR copy to avoid RuntimeError
            )
            if not running:
                logger.info("[Worker] Idle shutdown: no activity for %ss", IDLE_TIMEOUT)
                _shutdown_requested = True
                os._exit(0)


@app.on_event("startup")
async def _start_worker_tasks():
    import asyncio

    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(600)
            _cleanup_expired_jobs()

    app.state.cleanup_task = asyncio.create_task(_periodic_cleanup())

    if IDLE_TIMEOUT > 0:
        threading.Thread(target=_idle_watchdog, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("FLOWR_WORKER_PORT", 8788))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
