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

import os
import shutil
import tempfile
import threading
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# RDKit imports
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available – molecule processing will be limited.")

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


def _detect_device() -> str:
    """Detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _detect_device()
print(f"PyTorch device: {DEVICE}")

# ---------------------------------------------------------------------------
# FLOWR imports
# ---------------------------------------------------------------------------
FLOWR_AVAILABLE = False
try:
    import flowr.gen.utils as gen_util
    import flowr.util.rdkit as smolRD
    from flowr.data.dataset import GeometricDataset
    from flowr.gen.generate import generate_ligands_per_target
    from flowr.gen.utils import load_data_from_pdb, load_util
    from flowr.scriptutil import load_model
    from flowr.util.functional import LigandPocketOptimization
    from flowr.util.metrics import calc_strain, evaluate_pb_validity
    from flowr.util.pocket import PocketComplexBatch

    FLOWR_AVAILABLE = True
except ImportError as exc:
    print(
        f"WARNING: FLOWR modules not importable ({exc}) – generation will "
        "not work. Make sure the flowr package is on PYTHONPATH."
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path(tempfile.mkdtemp(prefix="flowr_worker_"))
JOBS: Dict[str, Dict[str, Any]] = {}

_JOB_TTL_SECONDS = 3600  # 1 hour


def _cleanup_expired_jobs():
    """Remove completed/failed jobs older than TTL."""
    now = time.time()
    expired = [
        jid
        for jid, j in JOBS.items()
        if j.get("status") in ("completed", "failed")
        and now - j.get("created_at", now) > _JOB_TTL_SECONDS
    ]
    for jid in expired:
        JOBS.pop(jid, None)


# Serializes GPU generation — only one job at a time to prevent
# model state races and GPU OOM from concurrent inference.
_generation_semaphore = threading.Semaphore(1)

# Auto-shutdown: worker exits after this many seconds of inactivity.
# Set via FLOWR_WORKER_IDLE_TIMEOUT (0 = disabled).
IDLE_TIMEOUT = int(os.environ.get("FLOWR_WORKER_IDLE_TIMEOUT", "0"))
_last_activity = time.time()
_shutdown_requested = False

print(f"Work directory:   {WORK_DIR}")
print(f"FLOWR available:  {FLOWR_AVAILABLE}")
print(f"Checkpoint path:  {CKPT_PATH}")
if IDLE_TIMEOUT > 0:
    print(f"Idle timeout:     {IDLE_TIMEOUT}s")


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


def _default_args(**overrides) -> Namespace:
    """Build a Namespace with all the defaults expected by the FLOWR pipeline."""
    defaults = dict(
        # ── File paths ──
        pdb_file="",
        ligand_file="",
        pdb_id=None,
        ligand_id=None,
        res_txt_file=None,
        chain_id=None,
        # ── Preprocessing ──
        canonicalize_conformer=False,
        pocket_noise="fix",
        cut_pocket=True,
        pocket_cutoff=6.0,
        protonate_pocket=False,
        compute_interactions=False,
        compute_interaction_recovery=False,
        add_hs=False,
        add_hs_and_optimize=False,
        optimize_gen_ligs=False,
        optimize_gen_ligs_hs=False,
        kekulize=False,
        use_pdbfixer=False,
        add_bonds_to_protein=True,
        add_hs_to_protein=False,
        max_pocket_size=1000,
        min_pocket_size=10,
        # ── Runtime ──
        seed=42,
        gpus=1,
        mp_index=0,
        num_workers=0,
        # ── Architecture ──
        arch="pocket",
        pocket_type="holo",
        pocket_coord_noise_std=0.0,
        ckpt_path=str(CKPT_PATH),
        lora_finetuned=False,
        data_path="",
        splits_path=None,
        dataset="spindr",
        save_dir="",
        save_file=None,
        # ── Sampling ──
        coord_noise_scale=0.1,
        max_sample_iter=100,
        sample_n_molecules_per_target=10,
        sample_mol_sizes=False,
        corrector_iters=0,
        rotation_alignment=False,
        permutation_alignment=False,
        save_traj=False,
        # ── Filtering ──
        filter_valid_unique=True,
        filter_diversity=False,
        diversity_threshold=0.9,
        filter_pb_valid=False,
        calculate_pb_valid=False,
        filter_cond_substructure=False,
        calculate_strain_energies=False,
        # ── Batching ──
        batch_cost=25,
        dataset_split=None,
        # ── Inpainting modes (all off by default) ──
        ligand_time=None,
        pocket_time=None,
        interaction_time=None,
        fixed_interactions=False,
        interaction_conditional=False,
        scaffold_hopping=False,
        scaffold_elaboration=False,
        linker_inpainting=False,
        fragment_inpainting=False,
        fragment_growing=False,
        grow_size=None,
        prior_center_file=None,
        max_fragment_cuts=3,
        core_growing=False,
        ring_system_index=0,
        substructure_inpainting=False,
        substructure=None,
        graph_inpainting=None,
        final_inpaint=False,
        separate_pocket_interpolation=False,
        separate_interaction_interpolation=False,
        anisotropic_prior=False,
        # ── Integration ──
        integration_steps=100,
        cat_sampling_noise_level=1,
        ode_sampling_strategy="linear",
        solver="euler",
        categorical_strategy="uniform-sample",
        use_sde_simulation=False,
        use_cosine_scheduler=False,
        bucket_cost_scale="quadratic",
        # ── Guidance ──
        guidance_config=None,
    )
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

        _model_state["loading"] = True
        _model_state["loaded"] = False
        try:
            print(f"Loading FLOWR model from {resolved} …")
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
            print(f"Model loaded on {DEVICE}.")

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
            traceback.print_exc()
            _model_state["loading"] = False
            _model_state["error"] = str(exc)
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════


class GenerationRequest(BaseModel):
    job_id: str
    protein_url: str  # URL to download the protein file from
    ligand_url: str  # URL to download the ligand file from
    protein_filename: str  # original filename (for extension detection)
    ligand_filename: str  # original filename
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
        print(f"Failed to download {url}: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  PRIOR CLOUD (for fragment-growing visualisation)
# ═══════════════════════════════════════════════════════════════════════════

# Standard amino acid 3-letter codes (for pocket extraction, matching FLOWR)
_STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def _compute_pocket_com_worker(
    protein_path: str,
    ligand_path: str,
    pocket_cutoff: float = 6.0,
) -> Optional[np.ndarray]:
    """Compute pocket COM by parsing PDB ATOM records (no torch required).

    Mimics the FLOWR pipeline: selects whole standard-AA residues with at least
    one atom within ``pocket_cutoff`` of any ligand heavy atom, returns their
    unweighted geometric centroid.
    """
    try:
        mol = _read_ligand_mol(ligand_path)
        if mol is None:
            return None
        mol_noH = Chem.RemoveHs(mol)
        if mol_noH.GetNumConformers() == 0:
            return None
        conf = mol_noH.GetConformer()
        lig_coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol_noH.GetNumAtoms())]
        )
    except Exception:
        return None

    try:
        protein_atoms = []
        with open(protein_path, "r") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    res_seq = line[22:26].strip()
                except (ValueError, IndexError):
                    continue
                protein_atoms.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "res_name": res_name,
                        "chain_id": chain_id,
                        "res_seq": res_seq,
                    }
                )
        if not protein_atoms:
            return None
    except Exception:
        return None

    prot_coords = np.array([[a["x"], a["y"], a["z"]] for a in protein_atoms])
    diff = prot_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=2))
    within_cutoff = (dists < pocket_cutoff).any(axis=1)

    residue_in_pocket = set()
    for i, atom in enumerate(protein_atoms):
        if within_cutoff[i] and atom["res_name"] in _STANDARD_AA:
            residue_in_pocket.add((atom["chain_id"], atom["res_seq"]))

    if not residue_in_pocket:
        return None

    pocket_coords = []
    for i, atom in enumerate(protein_atoms):
        if (atom["chain_id"], atom["res_seq"]) in residue_in_pocket and atom[
            "res_name"
        ] in _STANDARD_AA:
            pocket_coords.append([atom["x"], atom["y"], atom["z"]])

    if not pocket_coords:
        return None

    return np.array(pocket_coords).mean(axis=0)


def _compute_prior_cloud(
    ligand_path: str,
    prior_center_file: Optional[str],
    grow_size: int,
    protein_path: Optional[str] = None,
    pocket_cutoff: float = 6.0,
) -> Dict[str, Any]:
    """Compute a representative prior point cloud for fragment-growing.

    The cloud approximates the starting positions of the variable atoms that
    the flow-matching process will transform into the generated fragment.
    Coordinates are in the *original* PDB / SDF frame so they can be rendered
    directly in the 3Dmol viewer alongside the uploaded structures.

    When no prior center file is provided, the centre is set to the **protein
    pocket COM** — matching the FLOWR pipeline where the prior is zero-COM in
    the pocket-COM-shifted coordinate frame, thus effectively starting at the
    pocket COM in the original frame.

    Returns
    -------
    dict  {center, points, n_atoms, has_prior_center}
    """
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
                mol_noH = Chem.RemoveHs(mol)
                conf = mol_noH.GetConformer()
                coords = np.array(
                    [
                        list(conf.GetAtomPosition(i))
                        for i in range(mol_noH.GetNumAtoms())
                    ]
                )
                center = coords.mean(axis=0)
            else:
                center = np.zeros(3)

    # Sample representative Gaussian cloud (zero-COM, like the actual prior)
    rng = np.random.default_rng(seed=42)
    noise = rng.standard_normal((grow_size, 3))
    noise -= noise.mean(axis=0)  # zero COM
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
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FLOWR GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def _run_flowr_generation(
    job: Dict[str, Any],
    gen_mode: str,
    fixed_atoms: List[int],
    n_samples: int,
    batch_size: int,
    integration_steps: int,
    pocket_cutoff: float,
    grow_size: Optional[int] = None,
    prior_center_file: Optional[str] = None,
    coord_noise_scale: float = 0.0,
    filter_valid_unique: bool = True,
    filter_cond_substructure: bool = False,
    filter_diversity: bool = False,
    diversity_threshold: float = 0.9,
    sample_mol_sizes: bool = False,
    filter_pb_valid: bool = False,
    calculate_pb_valid: bool = False,
    calculate_strain_energies: bool = False,
    optimize_gen_ligs: bool = False,
    optimize_gen_ligs_hs: bool = False,
    anisotropic_prior: bool = False,
    ring_system_index: int = 0,
) -> Dict[str, Any]:
    """Run actual FLOWR generation using the loaded model."""
    # Snapshot all model state under lock to prevent races with concurrent /load-model
    with _model_lock:
        model = _model_state["model"]
        hparams = _model_state["hparams"]
        vocab = _model_state["vocab"]
        vocab_charges = _model_state["vocab_charges"]
        vocab_hybridization = _model_state.get("vocab_hybridization")
        vocab_aromatic = _model_state.get("vocab_aromatic")

    use_sub = gen_mode == "substructure_inpainting" and len(fixed_atoms) > 0
    is_inpainting = gen_mode != "denovo"
    output_dir = WORK_DIR / job["job_id"] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_flags = dict(
        substructure_inpainting=use_sub,
        substructure=fixed_atoms if use_sub else None,
        scaffold_hopping=(gen_mode == "scaffold_hopping"),
        scaffold_elaboration=(gen_mode == "scaffold_elaboration"),
        linker_inpainting=(gen_mode == "linker_inpainting"),
        core_growing=(gen_mode == "core_growing"),
        fragment_growing=(gen_mode == "fragment_growing"),
    )

    if gen_mode == "fragment_growing":
        mode_flags["grow_size"] = grow_size
        mode_flags["prior_center_file"] = prior_center_file

    args = _default_args(
        pdb_file=job["protein_path"],
        ligand_file=job["ligand_path"],
        pocket_cutoff=pocket_cutoff,
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
        ring_system_index=ring_system_index,
        save_dir=str(output_dir),
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
    print(
        f"[FLOWR] gen_mode={gen_mode}, inpainting_mode={model.inpainting_mode}, "
        f"inpainting_mode_inf={model.inpainting_mode_inf}"
    )

    transform, interpolant = load_util(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    if gen_mode in (
        "core_growing",
        "scaffold_hopping",
        "scaffold_elaboration",
        "linker_inpainting",
        "fragment_growing",
    ):
        print(
            f"[FLOWR] Interpolant flags — core_growing={interpolant.core_growing}, "
            f"scaffold_hopping={interpolant.scaffold_hopping}, "
            f"scaffold_elaboration={interpolant.scaffold_elaboration}, "
            f"linker_inpainting={interpolant.linker_inpainting}, "
            f"fragment_growing={interpolant.fragment_growing}, "
            f"inpainting_mode={interpolant.inpainting_mode}, "
            f"inference={interpolant.inference}"
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
    dataset = dataset.sample_n_molecules_per_target(n_samples)

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
                print(
                    f"[FLOWR] WARNING: Original ref ligand sanitization failed: {exc}. "
                    "Substructure filter may not work correctly."
                )
                original_ref_mol = None

    with torch.no_grad():
        _raw_generated = 0  # total molecules generated (before any filtering)
        _valid_count = 0  # molecules that passed sanitization
        _unique_count = 0  # molecules that passed uniqueness filter

        while len(all_gen_ligs) < n_samples and k <= max_sample_iter:
            dataloader = gen_util.get_dataloader(args, dataset, interpolant, iter=k)
            n_batches = len(dataloader) if hasattr(dataloader, "__len__") else 1

            for i, batch in enumerate(dataloader):
                total_iters_est = max(1, n_batches * min(max_sample_iter, 3))
                pct = 10 + int(75 * ((k * n_batches + i) / total_iters_est))
                job["progress"] = min(pct, 85)

                # Keep the idle watchdog alive while generating
                global _last_activity
                _last_activity = time.time()

                prior, data, _, _ = batch
                last_data = data

                ref_ligs_batch = model._generate_ligs(
                    data, lig_mask=data["lig_mask"].bool(), scale=model.coord_scale
                )

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
                                print(
                                    f"[FLOWR] Substructure filter with tensor ref "
                                    f"failed ({exc}), trying original ref mol..."
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
                                print(
                                    f"[FLOWR] Substructure filter with original "
                                    f"ref also failed ({exc}), skipping filter."
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
                            print(
                                f"[FLOWR] Substructure match rate: {match_rate} "
                                f"({num_after_sub}/{num_before_sub})"
                            )

                all_gen_ligs.extend(gen_ligs_batch)

                if filter_diversity:
                    all_gen_ligs = gen_util.filter_diverse_ligands_bulk(
                        all_gen_ligs,
                        threshold=diversity_threshold,
                    )

            k += 1

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
    metrics_log.append(f"Generated: {len(gen_ligs)} molecules")

    # ── Generation quality metrics ──
    # Metrics are computed over ALL molecules generated across iterations,
    # not just the final truncated set (n_samples).  When multiple iterations
    # were needed (e.g. some molecules were filtered), the denominators will
    # exceed the final output count — this is expected.
    if _raw_generated > 0:
        validity_rate = round(_valid_count / _raw_generated, 3)
        uniqueness_rate = round(_unique_count / max(_valid_count, 1), 3)
        metrics_log.append(
            f"Validity rate: {validity_rate} ({_valid_count}/{_raw_generated})"
        )
        metrics_log.append(
            f"Uniqueness rate: {uniqueness_rate} ({_unique_count}/{_valid_count})"
        )
        print(
            f"[FLOWR] Generation metrics — "
            f"validity: {validity_rate} ({_valid_count}/{_raw_generated}), "
            f"uniqueness: {uniqueness_rate} ({_unique_count}/{_valid_count})"
        )

    if filter_cond_substructure and _sub_total_before > 0:
        sub_rate = round(_sub_total_after / _sub_total_before, 2)
        metrics_log.append(
            f"Substructure match rate: {sub_rate} "
            f"({_sub_total_after}/{_sub_total_before})"
        )

    job["progress"] = 92

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
                        pass
            opt_label = "Full" if optimize_gen_ligs else "H-only"
            metrics_log.append(
                f"{opt_label} optimization: {len([x for x in all_gen_ligs_final if x is not None])}/{len(all_hs_before)} valid"
            )
            if rmsds:
                mean_rmsd = round(np.mean(rmsds), 3)
                metrics_log.append(f"Mean RMSD (before→after): {mean_rmsd} Å")
            strain_vals = []
            for mol in all_gen_ligs_final:
                if mol is not None:
                    try:
                        s = calc_strain(mol, add_hs=False)
                        strain_vals.append(round(s, 2))
                    except Exception:
                        pass
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
        strain_vals = []
        for mol in all_gen_ligs_final:
            if mol is not None:
                try:
                    s = calc_strain(mol, add_hs=False)
                    strain_vals.append(round(s, 2))
                except Exception:
                    pass
        if strain_vals:
            metrics_log.append(
                f"Strain energy (mean): {round(np.mean(strain_vals), 2)} kcal/mol"
            )
            metrics_log.append(
                f"Strain energy (std):  {round(np.std(strain_vals), 2)} kcal/mol"
            )

    # ── PoseBusters validity ──
    if calculate_pb_valid or filter_pb_valid:
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

    # ── Build results list ──
    used_optimization = optimize_gen_ligs or optimize_gen_ligs_hs
    results = []
    for idx, mol in enumerate(gen_ligs):
        if mol is None:
            continue
        try:
            opt_mol = (
                all_gen_ligs_final[idx]
                if idx < len(all_gen_ligs_final) and all_gen_ligs_final[idx] is not None
                else mol
            )

            sdf_with_hs = _mol_to_sdf_string(opt_mol)
            mol_noH = Chem.RemoveHs(opt_mol)
            sdf_no_hs = _mol_to_sdf_string(mol_noH)

            if used_optimization:
                display_sdf = sdf_with_hs
                display_mol = opt_mol
            else:
                display_sdf = sdf_no_hs
                display_mol = mol_noH

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

            # Extract affinity predictions if present (set by generate_ligands_per_target)
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
    # ── Compute prior cloud for visualisation (all conditional modes) ──
    prior_cloud = None
    if gen_mode != "denovo":
        # Determine mode-specific cloud size (number of replaced atoms)
        cloud_size = grow_size if (gen_mode == "fragment_growing" and grow_size) else 20
        if gen_mode != "fragment_growing":
            try:
                _ref = _read_ligand_mol(job["ligand_path"])
                if _ref is not None:
                    _ref_noH = Chem.RemoveHs(_ref)
                    from rdkit.Chem.Scaffolds.MurckoScaffold import (
                        GetScaffoldForMol as _GetScaffold,
                    )

                    if gen_mode == "scaffold_hopping":
                        # Scaffold atoms = replaced
                        _scaffold = _GetScaffold(Chem.Mol(_ref_noH))
                        cloud_size = (
                            _scaffold.GetNumAtoms()
                            if _scaffold
                            else _ref_noH.GetNumAtoms()
                        )
                    elif gen_mode == "linker_inpainting":
                        # Linker atoms (scaffold non-ring) = replaced
                        _scaffold = _GetScaffold(Chem.Mol(_ref_noH))
                        if _scaffold:
                            _ring_atoms = set()
                            for ring in _ref_noH.GetRingInfo().AtomRings():
                                _ring_atoms.update(ring)
                            for a in _scaffold.GetAtoms():
                                a.SetIntProp("org_idx", a.GetIdx())
                            _scaffold_orig = set()
                            _s2 = Chem.Mol(_ref_noH)
                            for a in _s2.GetAtoms():
                                a.SetIntProp("org_idx", a.GetIdx())
                            _scf = _GetScaffold(_s2)
                            if _scf:
                                _scaffold_orig = {
                                    a.GetIntProp("org_idx") for a in _scf.GetAtoms()
                                }
                            cloud_size = len(
                                [a for a in _scaffold_orig if a not in _ring_atoms]
                            )
                            if cloud_size == 0:
                                cloud_size = _ref_noH.GetNumAtoms()
                        else:
                            cloud_size = _ref_noH.GetNumAtoms()
                    elif gen_mode == "scaffold_elaboration":
                        # Elaboration replaces non-scaffold non-ring atoms + functional groups
                        _s2 = Chem.Mol(_ref_noH)
                        try:
                            Chem.SanitizeMol(_s2)
                        except Exception:
                            pass
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
                        # Count non-ring atoms that are outside the scaffold
                        # (simplified: counts atoms to be replaced)
                        n_replaced = sum(
                            1
                            for idx in range(_ref_noH.GetNumAtoms())
                            if idx not in _scaffold_set and idx not in _ring_set
                        )
                        cloud_size = max(n_replaced, 1)
                    elif gen_mode == "core_growing":
                        # Non-core atoms = replaced
                        _s2 = Chem.Mol(_ref_noH)
                        for a in _s2.GetAtoms():
                            a.SetIntProp("org_idx", a.GetIdx())
                        _scaffold = _GetScaffold(_s2)
                        if _scaffold:
                            _scaffold_atoms = {
                                a.GetIntProp("org_idx") for a in _scaffold.GetAtoms()
                            }
                            _ring_systems = []
                            # Simple ring system detection
                            ri = _s2.GetRingInfo()
                            visited = set()
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
                                {
                                    a
                                    for a in _scaffold_atoms
                                    if a in _ring_systems[_ridx]
                                }
                                if _ring_systems
                                else set()
                            )
                            cloud_size = (
                                _ref_noH.GetNumAtoms() - len(_core)
                                if _core
                                else _ref_noH.GetNumAtoms()
                            )
                        else:
                            cloud_size = _ref_noH.GetNumAtoms()
                    elif gen_mode == "substructure_inpainting":
                        # Replaced = total heavy - fixed
                        cloud_size = _ref_noH.GetNumAtoms() - len(fixed_atoms)
                        if cloud_size <= 0:
                            cloud_size = _ref_noH.GetNumAtoms()
                    else:
                        cloud_size = _ref_noH.GetNumAtoms()
            except Exception:
                pass
        try:
            prior_cloud = _compute_prior_cloud(
                job["ligand_path"],
                prior_center_file,
                cloud_size,
                protein_path=job.get("protein_path"),
                pocket_cutoff=pocket_cutoff if pocket_cutoff else 6.0,
            )
        except Exception as exc:
            print(f"[FLOWR] Prior cloud computation failed (non-critical): {exc}")

    return {
        "results": results,
        "metrics": metrics_log,
        "used_optimization": used_optimization,
        "prior_cloud": prior_cloud,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BACKGROUND GENERATION WORKER
# ═══════════════════════════════════════════════════════════════════════════


def _generation_worker(job_id: str, req: dict):
    """Run generation in a background thread, updating job state."""
    global _last_activity
    job = JOBS[job_id]
    job["status"] = "queued"
    job["progress"] = 0
    _last_activity = time.time()  # keep idle watchdog alive

    # Serialize GPU access — only one generation at a time
    acquired = _generation_semaphore.acquire(timeout=600)
    if not acquired:
        job.update(
            status="failed",
            progress=0,
            error="Another generation is already running. Please wait.",
        )
        return

    job["status"] = "generating"
    job["progress"] = 2

    try:
        # Download input files from frontend server
        job_dir = WORK_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        protein_path = job_dir / Path(req["protein_filename"]).name
        ligand_path = job_dir / Path(req["ligand_filename"]).name

        job["progress"] = 3
        if not _download_file(req["protein_url"], protein_path):
            raise RuntimeError(f"Failed to download protein from {req['protein_url']}")
        if not _download_file(req["ligand_url"], ligand_path):
            raise RuntimeError(f"Failed to download ligand from {req['ligand_url']}")

        job["protein_path"] = str(protein_path)
        job["ligand_path"] = str(ligand_path)

        # Download prior center file if provided
        prior_center_path = None
        if req.get("prior_center_url") and req.get("prior_center_filename"):
            prior_path = job_dir / Path(req["prior_center_filename"]).name
            if _download_file(req["prior_center_url"], prior_path):
                prior_center_path = str(prior_path)

        job["progress"] = 5

        # Ensure model is loaded
        ckpt = req.get("ckpt_path") or CKPT_PATH
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

        gen_out = _run_flowr_generation(
            job,
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
            calculate_strain_energies=req.get("calculate_strain_energies", False),
            optimize_gen_ligs=req.get("optimize_gen_ligs", False),
            optimize_gen_ligs_hs=req.get("optimize_gen_ligs_hs", False),
            anisotropic_prior=req.get("anisotropic_prior", False),
            ring_system_index=req.get("ring_system_index", 0),
        )

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
        )

    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", progress=0, error=str(exc))
    finally:
        # Clean up temporary files
        job_dir = WORK_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        _generation_semaphore.release()


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    global _last_activity
    _last_activity = time.time()
    return {
        "status": "ok",
        "device": DEVICE,
        "flowr": FLOWR_AVAILABLE,
        "rdkit": RDKIT_AVAILABLE,
        "model_loaded": _model_state["loaded"],
        "model_loading": _model_state["loading"],
        "model_error": _model_state["error"],
        "ckpt_path": _model_state.get("ckpt_path"),
    }


@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown. The worker will exit after responding."""
    global _shutdown_requested
    _shutdown_requested = True
    print("Shutdown requested — exiting after current work completes.")

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


@app.post("/generate")
async def generate(request: GenerationRequest):
    """Start a generation job. Returns immediately; poll /job/{id} for progress."""
    global _last_activity
    _last_activity = time.time()
    job_id = request.job_id

    if job_id in JOBS and JOBS[job_id].get("status") == "generating":
        raise HTTPException(409, "Generation already in progress for this job.")

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


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    global _last_activity
    _last_activity = time.time()
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]

    resp: Dict[str, Any] = {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
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
        )
    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _idle_watchdog():
    """Background thread that exits the process after idle timeout."""
    while not _shutdown_requested:
        time.sleep(10)
        _cleanup_expired_jobs()
        if IDLE_TIMEOUT > 0 and (time.time() - _last_activity) > IDLE_TIMEOUT:
            # Check no jobs are running
            running = any(
                j.get("status") in ("queued", "generating", "loading_model")
                for j in JOBS.values()
            )
            if not running:
                print(f"Idle for >{IDLE_TIMEOUT}s with no active jobs — shutting down.")
                os._exit(0)


@app.on_event("startup")
async def _start_worker_tasks():
    import asyncio

    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(600)
            _cleanup_expired_jobs()

    asyncio.create_task(_periodic_cleanup())

    if IDLE_TIMEOUT > 0:
        threading.Thread(target=_idle_watchdog, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("FLOWR_WORKER_PORT", 8788))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
