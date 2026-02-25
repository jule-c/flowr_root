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
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests as http_requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# RDKit imports
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available – molecule processing will be limited.")

# ---------------------------------------------------------------------------
# Shared chemistry utilities (also used by worker.py)
# ---------------------------------------------------------------------------
from chem_utils import (
    _STRUCTURAL_ALERT_SMARTS,
    _STRUCTURAL_ALERTS,
    ALL_PROPERTY_NAMES,
    CONT_PROPERTIES_RDKIT,
    DISC_PROPERTIES_RDKIT,
)
from chem_utils import (
    RDKIT_AVAILABLE as _RDKIT_CHECK,
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

# ---------------------------------------------------------------------------
# OpenEye license setup (must happen BEFORE importing openeye)
# ---------------------------------------------------------------------------
if "OE_LICENSE" not in os.environ:
    _script_dir = Path(__file__).resolve().parent
    for _candidate in [
        _script_dir / "tools" / "oe_license.txt",
        _script_dir / "oe_license.txt",
        _script_dir.parent / "oe_license.txt",
    ]:
        if _candidate.is_file():
            os.environ["OE_LICENSE"] = str(_candidate)
            break

# ---------------------------------------------------------------------------
# OpenEye imports (optional – for 2D interaction diagrams)
# ---------------------------------------------------------------------------
OPENEYE_AVAILABLE = False
try:
    from openeye import oechem, oedepict, oegrapheme, oeomega

    OPENEYE_AVAILABLE = True
except ImportError:
    print("WARNING: OpenEye not available – 2D interaction diagrams disabled.")

# ---------------------------------------------------------------------------
# SciPy (optional – for fast pairwise distances)
# ---------------------------------------------------------------------------
try:
    from scipy.spatial.distance import cdist

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# scikit-learn / UMAP (optional – for chemical space projection)
# ---------------------------------------------------------------------------
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap

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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="FLOWR Visualization", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)

UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="flowr_vis_"))


def _cleanup_upload_dir():
    """Remove the upload directory on graceful shutdown."""
    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            print(f"Cleaned up upload directory: {UPLOAD_DIR}")
    except Exception:
        pass


atexit.register(_cleanup_upload_dir)

JOBS: Dict[str, Dict[str, Any]] = {}
_selected_ckpt_path: Optional[str] = None  # Set from landing page, sent to worker
_selected_workflow_type: str = "sbdd"  # "sbdd" or "lbdd" – set from landing page

# ── Job TTL cleanup ──
_JOB_TTL_SECONDS = 3600  # 1 hour


def _cleanup_expired_jobs():
    """Remove jobs older than _JOB_TTL_SECONDS to prevent unbounded memory growth."""
    now = time.time()
    expired = [
        jid
        for jid, jdata in JOBS.items()
        if now - jdata.get("created_at", now) > _JOB_TTL_SECONDS
        and jdata.get("status")
        not in ("generating", "allocating_gpu", "starting", "loading_model")
    ]
    for jid in expired:
        job_dir = UPLOAD_DIR / jid
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        JOBS.pop(jid, None)


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


@app.on_event("startup")
async def _start_cleanup_loop():
    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(600)
            _cleanup_expired_jobs()

    asyncio.create_task(_periodic_cleanup())


print(f"Upload directory:  {UPLOAD_DIR}")
print(f"Checkpoints dir:   {CKPTS_DIR}  (exists={CKPTS_DIR.is_dir()})")
print(f"RDKit available:   {RDKIT_AVAILABLE}")
print(f"OpenEye avail.:    {OPENEYE_AVAILABLE}")
print(f"Worker mode:       {WORKER_MODE}")
if WORKER_MODE == "slurm":
    print(f"SLURM script:      {SLURM_SCRIPT}")
    print(f"Startup timeout:   {SLURM_STARTUP_TIMEOUT}s")
else:
    print(f"Worker URL:        {WORKER_URL}")


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
    n_samples: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=25, ge=1, le=200)
    integration_steps: int = 100
    pocket_cutoff: float = 6.0
    grow_size: Optional[int] = None
    prior_center_filename: Optional[str] = None
    prior_center_coords: Optional[Dict[str, float]] = (
        None  # {x, y, z} from visual placement
    )
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
    ring_system_index: int = 0
    ref_ligand_com_prior: bool = False
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


def _identify_functional_groups(mol):
    """Identify functional groups (Ertl IFG algorithm). Pure RDKit."""
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
        grp = set([marked.pop()])
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
                atomIds=tuple(list(g)),
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


def _extract_scaffold_elaboration_mask(mol) -> np.ndarray:
    """Boolean mask: True = atom to be REPLACED during scaffold elaboration.

    Replaced = (non-scaffold OR functional-group) AND NOT in-ring.
    """
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
    with open(pdb_path) as f:
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


def _compute_interactions(
    pdb_path: str, ligand_mol, cutoff: float = 4.0
) -> List[Dict[str, Any]]:
    """Protein-ligand interaction detection using vectorized numpy distance computation."""
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
    near_indices = np.where(near_mask)[0]

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

    pi_indices, li_indices = np.where(dist_matrix <= cutoff)

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


def _submit_slurm_worker() -> str:
    """Submit the worker SLURM job. Returns the SLURM job ID."""
    env = os.environ.copy()
    if _selected_ckpt_path:
        env["FLOWR_CKPT_PATH"] = _selected_ckpt_path
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

    print(f"Submitting SLURM worker: {' '.join(sbatch_cmd)}")
    result = subprocess.run(
        sbatch_cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    slurm_id = result.stdout.strip().split(";")[0]
    print(f"Submitted SLURM worker job: {slurm_id}")
    return slurm_id


def _get_slurm_node(slurm_job_id: str) -> Optional[str]:
    """Query SLURM for the node the job is running on."""
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
        pass
    return None


def _cancel_slurm_job(slurm_job_id: str):
    """Cancel a SLURM job."""
    try:
        subprocess.run(
            ["scancel", slurm_job_id],
            capture_output=True,
            timeout=10,
        )
        print(f"Cancelled SLURM job {slurm_job_id}")
    except Exception as e:
        print(f"Warning: failed to cancel SLURM job {slurm_job_id}: {e}")


def _ensure_worker_running(job: dict) -> str:
    """Ensure a GPU worker is running. Returns the worker URL.

    In static mode, just returns the configured URL.
    In SLURM mode, submits a job if needed and waits for it to come online.
    """
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
                slurm_id = _submit_slurm_worker()
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
                print(f"Worker node discovered: {node}")
                job["progress"] = 3

        # Probe the worker
        url = _worker_state.get("url")
        if url and _is_worker_reachable(url):
            with _worker_lock:
                _worker_state["status"] = "running"
            print(f"Worker is online at {url}")
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
            print("Sent shutdown signal to worker")
        except Exception:
            pass

    # Cancel SLURM job as fallback
    if slurm_id:
        _cancel_slurm_job(slurm_id)

    with _worker_lock:
        _worker_state.update(status="idle", slurm_job_id=None, node=None, url=None)
    print("GPU worker released")


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATION PROXY (delegates to GPU worker)
# ═══════════════════════════════════════════════════════════════════════════


def _proxy_generation(job_id: str, req: dict):
    """Send generation request to the GPU worker and track its progress.

    In SLURM mode this will first allocate a GPU node, wait for the worker
    to come online, run generation, and then release the GPU.

    This function is called in a background thread from POST /generate.
    """
    job = JOBS[job_id]

    def _is_cancelled() -> bool:
        return bool(job.get("cancel_requested_at") or job.get("cancelled"))

    job["status"] = "allocating_gpu" if WORKER_MODE == "slurm" else "generating"
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
        server_base = os.environ.get("FLOWR_SERVER_URL", "http://localhost:8787")
        workflow = req.get("workflow_type", job.get("workflow_type", "sbdd"))

        # Ligand URL may be None for de novo pocket-only / scratch mode
        ligand_url = None
        ligand_filename = None
        if job.get("ligand_path"):
            ligand_url = f"{server_base}/files/{job_id}/{Path(job['ligand_path']).name}"
            ligand_filename = Path(job["ligand_path"]).name

        num_heavy_atoms = req.get("num_heavy_atoms") or job.get("num_heavy_atoms")

        # ── LBDD branch: no protein, but supports all gen modes ──
        if workflow == "lbdd":
            # Handle prior center for fragment growing (same as SBDD)
            prior_center_url = None
            prior_center_filename = None
            prior_center_coords = req.get("prior_center_coords")

            if prior_center_coords:
                job_dir = UPLOAD_DIR / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                xyz_name = "_prior_center_placed.xyz"
                xyz_path = job_dir / xyz_name
                xyz_path.write_text(
                    f"1\nPlaced prior center\n"
                    f"Ar  {prior_center_coords['x']:.6f}  "
                    f"{prior_center_coords['y']:.6f}  "
                    f"{prior_center_coords['z']:.6f}\n"
                )
                prior_center_url = f"{server_base}/files/{job_id}/{xyz_name}"
                prior_center_filename = xyz_name
            elif req.get("prior_center_filename"):
                safe_name = Path(req["prior_center_filename"]).name
                candidate = UPLOAD_DIR / job_id / safe_name
                if candidate.exists():
                    prior_center_url = f"{server_base}/files/{job_id}/{safe_name}"
                    prior_center_filename = safe_name

            worker_payload = {
                "job_id": job_id,
                "workflow_type": "lbdd",
                "ligand_url": ligand_url,
                "ligand_filename": ligand_filename,
                "ckpt_path": job.get("ckpt_path") or _selected_ckpt_path,
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
                "calculate_strain_energies": req.get(
                    "calculate_strain_energies", False
                ),
                "optimize_method": req.get("optimize_method", "none"),
                "anisotropic_prior": req.get("anisotropic_prior", False),
                "ring_system_index": req.get("ring_system_index", 0),
                "ref_ligand_com_prior": req.get("ref_ligand_com_prior", False),
                "sample_n_molecules_per_mol": req.get("sample_n_molecules_per_mol", 1),
                "num_heavy_atoms": num_heavy_atoms,
                "property_filter": req.get("property_filter"),
                "adme_filter": req.get("adme_filter"),
            }

            resp = http_requests.post(
                f"{worker_url}/generate",
                json=worker_payload,
                timeout=30,
            )
            resp.raise_for_status()

            if not _is_cancelled():
                job["status"] = "generating"
                job["progress"] = 10

            # Poll worker until done
            generation_timeout = int(os.environ.get("FLOWR_GENERATION_TIMEOUT", "1800"))
            poll_deadline = time.time() + generation_timeout
            while time.time() < poll_deadline:
                if job.get("status") == "cancelled":
                    print(f"[Job {job_id}] Cancelled by user")
                    break
                time.sleep(2)
                worker_data = _poll_worker_job(job_id, worker_url)
                if worker_data is None:
                    continue
                w_status = worker_data.get("status")
                w_progress = worker_data.get("progress", 0)
                if not _is_cancelled():
                    job["progress"] = max(job["progress"], w_progress)
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
                            used_optimization=worker_data.get(
                                "used_optimization", False
                            ),
                            prior_cloud=worker_data.get("prior_cloud"),
                        )
                        # Invalidate visualization caches after status is set
                        for key in list(job.keys()):
                            if key.startswith("chemspace_") or key in (
                                "propspace_cache",
                                "affinity_dist_cache",
                                "results_full",
                            ):
                                del job[key]
                    break
                elif w_status == "cancelled":
                    job.update(status="cancelled", error="Cancelled by user")
                    break
                elif w_status == "failed":
                    if _is_cancelled():
                        job["status"] = "cancelled"
                    else:
                        job.update(
                            status="failed",
                            progress=0,
                            error=worker_data.get("error", "Worker generation failed"),
                        )
                    break
                elif w_status == "loading_model":
                    if not _is_cancelled():
                        job["status"] = "loading_model"

            if job["status"] not in ("completed", "failed", "cancelled"):
                job["status"] = "failed"
                job["error"] = "Generation timed out"

        else:
            # ── SBDD branch (existing logic) ──
            protein_url = (
                f"{server_base}/files/{job_id}/{Path(job['protein_path']).name}"
            )

            prior_center_url = None
            prior_center_filename = None
            prior_center_coords = req.get("prior_center_coords")

            if prior_center_coords:
                # User visually placed the prior cloud — write a temp XYZ file
                # so the worker can load it via the standard prior_center_file path
                job_dir = UPLOAD_DIR / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                xyz_name = "_prior_center_placed.xyz"
                xyz_path = job_dir / xyz_name
                xyz_path.write_text(
                    f"1\nPlaced prior center\n"
                    f"Ar  {prior_center_coords['x']:.6f}  "
                    f"{prior_center_coords['y']:.6f}  "
                    f"{prior_center_coords['z']:.6f}\n"
                )
                prior_center_url = f"{server_base}/files/{job_id}/{xyz_name}"
                prior_center_filename = xyz_name
            elif req.get("prior_center_filename"):
                safe_name = Path(req["prior_center_filename"]).name
                candidate = UPLOAD_DIR / job_id / safe_name
                if candidate.exists():
                    prior_center_url = f"{server_base}/files/{job_id}/{safe_name}"
                    prior_center_filename = safe_name

            # 3. Send generation request to the worker
            worker_payload = {
                "job_id": job_id,
                "workflow_type": "sbdd",
                "protein_url": protein_url,
                "ligand_url": ligand_url,
                "protein_filename": Path(job["protein_path"]).name,
                "ligand_filename": ligand_filename,
                "ckpt_path": job.get("ckpt_path") or _selected_ckpt_path,
                "gen_mode": req["gen_mode"],
                "fixed_atoms": req["fixed_atoms"],
                "n_samples": req["n_samples"],
                "batch_size": req["batch_size"],
                "integration_steps": req["integration_steps"],
                "pocket_cutoff": req["pocket_cutoff"],
                "coord_noise_scale": req.get("coord_noise_scale", 0.0),
                "grow_size": req.get("grow_size"),
                "prior_center_url": prior_center_url,
                "prior_center_filename": prior_center_filename,
                "filter_valid_unique": req.get("filter_valid_unique", True),
                "filter_cond_substructure": req.get("filter_cond_substructure", False),
                "filter_diversity": req.get("filter_diversity", False),
                "diversity_threshold": req.get("diversity_threshold", 0.9),
                "sample_mol_sizes": req.get("sample_mol_sizes", False),
                "filter_pb_valid": req.get("filter_pb_valid", False),
                "calculate_pb_valid": req.get("calculate_pb_valid", False),
                "calculate_strain_energies": req.get(
                    "calculate_strain_energies", False
                ),
                "optimize_gen_ligs": req.get("optimize_gen_ligs", False),
                "optimize_gen_ligs_hs": req.get("optimize_gen_ligs_hs", False),
                "anisotropic_prior": req.get("anisotropic_prior", False),
                "ring_system_index": req.get("ring_system_index", 0),
                "ref_ligand_com_prior": req.get("ref_ligand_com_prior", False),
                "num_heavy_atoms": num_heavy_atoms,
                "property_filter": req.get("property_filter"),
                "adme_filter": req.get("adme_filter"),
            }

            resp = http_requests.post(
                f"{worker_url}/generate",
                json=worker_payload,
                timeout=30,
            )
            resp.raise_for_status()

            if not _is_cancelled():
                job["status"] = "generating"
                job["progress"] = 10

            # 4. Poll worker until done (blocking in this background thread)
            generation_timeout = int(
                os.environ.get("FLOWR_GENERATION_TIMEOUT", "1800")
            )  # 30 min
            poll_deadline = time.time() + generation_timeout
            while time.time() < poll_deadline:
                # Check for user-initiated cancellation
                if job.get("status") == "cancelled":
                    print(f"[Job {job_id}] Cancelled by user")
                    break

                time.sleep(2)
                worker_data = _poll_worker_job(job_id, worker_url)
                if worker_data is None:
                    continue

                w_status = worker_data.get("status")
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
                            used_optimization=worker_data.get(
                                "used_optimization", False
                            ),
                            prior_cloud=worker_data.get("prior_cloud"),
                        )
                        # Invalidate visualization caches after status is set
                        for key in list(job.keys()):
                            if key.startswith("chemspace_") or key in (
                                "propspace_cache",
                                "affinity_dist_cache",
                                "results_full",
                            ):
                                del job[key]
                    break
                elif w_status == "cancelled":
                    job.update(status="cancelled", error="Cancelled by user")
                    break
                elif w_status == "failed":
                    if _is_cancelled():
                        job["status"] = "cancelled"
                    else:
                        job.update(
                            status="failed",
                            progress=0,
                            error=worker_data.get("error", "Worker generation failed"),
                        )
                    break
                elif w_status == "loading_model":
                    if not _is_cancelled():
                        job["status"] = "loading_model"
                else:
                    if not _is_cancelled():
                        job["progress"] = max(
                            job.get("progress", 10),
                            worker_data.get("progress", 0),
                        )

            # If we exited the loop without break, we timed out
            if job.get("status") not in ("completed", "failed", "cancelled"):
                job.update(
                    status="failed",
                    progress=0,
                    error=f"Generation timed out after {generation_timeout}s",
                )

    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", progress=0, error=f"Generation failed: {exc}")

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


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════


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
async def list_checkpoints(workflow: str = "sbdd"):
    """Scan the ckpts/{workflow} directory and return available checkpoints."""
    base: List[Dict[str, str]] = []
    project: List[Dict[str, Any]] = []

    # Determine root based on workflow type
    if workflow == "lbdd":
        scan_dir = CKPTS_DIR / "lbdd"
    else:
        scan_dir = CKPTS_DIR / "sbdd"

    print(
        f"[checkpoints] Scanning {scan_dir} (workflow={workflow}, exists={scan_dir.is_dir()})"
    )

    if scan_dir.is_dir():
        for f in sorted(scan_dir.iterdir()):
            if f.is_file() and f.suffix == ".ckpt":
                base.append({"name": f.stem, "path": str(f)})
                print(f"[checkpoints]   base: {f.name}")

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

    print(f"[checkpoints] Found {len(base)} base, {len(project)} project checkpoints")
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
        "checkpoint_registered": _selected_ckpt_path is not None,
    }


class LoadModelRequest(BaseModel):
    ckpt_path: Optional[str] = None


class RegisterCheckpointRequest(BaseModel):
    ckpt_path: str
    workflow_type: str = "sbdd"  # "sbdd" or "lbdd"


@app.post("/register-checkpoint")
async def register_checkpoint(request: RegisterCheckpointRequest):
    """Store the user's checkpoint selection. Does NOT load the model.

    Model loading is deferred until the user clicks Generate, at which
    point the worker will load it on the GPU if not already loaded.
    """
    global _selected_ckpt_path, _selected_workflow_type
    _selected_ckpt_path = request.ckpt_path
    _selected_workflow_type = request.workflow_type
    return {
        "status": "registered",
        "ckpt_path": request.ckpt_path,
        "workflow_type": request.workflow_type,
    }


@app.post("/load-model")
async def load_model_endpoint(request: LoadModelRequest = None):
    """Proxy model loading to the GPU worker (called during generation, not at launch)."""
    global _selected_ckpt_path
    ckpt = request.ckpt_path if request else None
    if ckpt:
        _selected_ckpt_path = ckpt

    try:
        resp = http_requests.post(
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
        resp = http_requests.get(f"{_get_worker_url()}/model-status", timeout=5)
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
async def serve_file(job_id: str, filename: str):
    """Serve uploaded files so the GPU worker can download them."""
    if not re.fullmatch(r"[a-f0-9\-]+", job_id):
        raise HTTPException(400, "Invalid job ID.")
    safe_name = Path(filename).name
    file_path = (UPLOAD_DIR / job_id / safe_name).resolve()
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


@app.post("/create-denovo-job")
async def create_denovo_job(request: CreateDenovoJobRequest):
    """Create (or update) a job for de novo generation without a reference ligand.

    For SBDD pocket-only: the protein must already be uploaded (job_id required).
    For LBDD scratch: creates a brand-new job with no files at all.
    """
    job_id = request.job_id
    if job_id and job_id in JOBS:
        # Re-use existing job (e.g. after protein upload)
        job = JOBS[job_id]
    elif request.workflow_type == "lbdd":
        # Create a fresh job with no files
        job_id = str(uuid.uuid4())[:8]
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "denovo_ready",
            "created_at": time.time(),
            "ckpt_path": _selected_ckpt_path,
            "workflow_type": request.workflow_type,
        }
        job = JOBS[job_id]
    else:
        raise HTTPException(
            400,
            "SBDD pocket-only mode requires an existing job_id with a protein upload.",
        )

    job["num_heavy_atoms"] = request.num_heavy_atoms
    job["workflow_type"] = request.workflow_type
    job["denovo_no_ligand"] = True

    return {
        "job_id": job_id,
        "num_heavy_atoms": request.num_heavy_atoms,
        "workflow_type": request.workflow_type,
    }


@app.post("/upload/protein")
async def upload_protein(file: UploadFile = File(...)):
    _cleanup_expired_jobs()  # Evict stale jobs on new upload
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdb", ".cif", ".mmcif"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    JOBS[job_id] = {
        "job_id": job_id,
        "protein_path": str(save_path),
        "protein_filename": safe_name,
        "status": "protein_uploaded",
        "created_at": time.time(),
        "ckpt_path": _selected_ckpt_path,
        "workflow_type": _selected_workflow_type,
    }
    return {
        "job_id": job_id,
        "filename": file.filename,
        "format": ext,
        "pdb_data": content.decode("utf-8", errors="replace"),
    }


@app.post("/upload/ligand/{job_id}")
async def upload_ligand(job_id: str, file: UploadFile = File(...)):
    if job_id not in JOBS:
        raise HTTPException(404, "Upload protein first.")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".sdf", ".mol", ".mol2", ".pdb"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_dir = UPLOAD_DIR / job_id
    safe_name = Path(file.filename).name
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    mol = _read_ligand_mol(str(save_path))
    if mol is None:
        raise HTTPException(400, "Could not parse ligand file.")

    # Crawl SDF for reference affinity properties
    ref_affinity = _crawl_sdf_affinity(str(save_path))

    JOBS[job_id].update(
        ligand_path=str(save_path),
        ligand_filename=safe_name,
        ligand_smiles=_mol_to_smiles(mol),
        status="ligand_uploaded",
        ref_affinity=ref_affinity,
    )
    JOBS[job_id].setdefault("ckpt_path", _selected_ckpt_path)
    JOBS[job_id].setdefault("workflow_type", "sbdd")

    mol_noH = Chem.RemoveHs(mol)
    smiles_noH = _mol_to_smiles(mol_noH)
    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    return {
        "job_id": job_id,
        "filename": file.filename,
        "smiles": _mol_to_smiles(mol),
        "smiles_noH": smiles_noH,
        "has_explicit_hs": has_explicit_hs,
        "sdf_data": _mol_to_sdf_string(mol),
        "atoms": _mol_to_atom_info(mol),
        "bonds": _mol_to_bond_info(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "ref_affinity": ref_affinity,
        "ref_properties": _compute_all_properties(mol_noH),
    }


# ── LBDD: Upload molecule without protein ──


@app.post("/upload/molecule")
async def upload_molecule(file: UploadFile = File(...)):
    """Upload a molecule file for LBDD workflow (no protein required).

    Creates a new job and stores the ligand. Accepts SDF, MOL, MOL2.
    """
    _cleanup_expired_jobs()
    ext = Path(file.filename).suffix.lower()
    if ext not in (".sdf", ".mol", ".mol2"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    mol = _read_ligand_mol(str(save_path))
    if mol is None:
        raise HTTPException(400, "Could not parse molecule file.")

    mol_noH = Chem.RemoveHs(mol)
    smiles_noH = _mol_to_smiles(mol_noH)
    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    JOBS[job_id] = {
        "job_id": job_id,
        "ligand_path": str(save_path),
        "ligand_filename": safe_name,
        "ligand_smiles": _mol_to_smiles(mol),
        "workflow_type": "lbdd",
        "status": "ligand_uploaded",
        "created_at": time.time(),
        "ckpt_path": _selected_ckpt_path,
    }

    return {
        "job_id": job_id,
        "filename": file.filename,
        "smiles": _mol_to_smiles(mol),
        "smiles_noH": smiles_noH,
        "has_explicit_hs": has_explicit_hs,
        "sdf_data": _mol_to_sdf_string(mol),
        "atoms": _mol_to_atom_info(mol),
        "bonds": _mol_to_bond_info(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "ref_properties": _compute_all_properties(mol_noH),
    }


# ── ADMET model upload ──


@app.post("/upload/adme-model/{job_id}")
async def upload_adme_model(job_id: str, file: UploadFile = File(...)):
    """Upload an ADMET model file for property-based filtering during generation."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found. Upload protein/ligand first.")

    ext = Path(file.filename).suffix.lower()
    allowed_exts = (".pt", ".pth", ".pkl", ".joblib", ".ckpt", ".onnx", ".bin")
    if ext not in allowed_exts:
        raise HTTPException(
            400,
            f"Unsupported model format: {ext}. Accepted: {', '.join(allowed_exts)}",
        )

    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"adme_{Path(file.filename).name}"
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)

    # Build a URL the worker can use to download the model
    server_base = os.environ.get("FLOWR_SERVER_URL", "http://localhost:8787")
    model_url = f"{server_base}/files/{job_id}/{safe_name}"

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
async def generate_conformers(request: SmilesConformerRequest):
    """Generate 3D conformers from a SMILES string using OpenEye Omega.

    Returns a list of conformers with SDF data and relative energies.
    Falls back to RDKit ETKDG if OpenEye is not available.
    """
    smiles = request.smiles.strip()
    max_confs = request.max_confs

    if not smiles:
        raise HTTPException(400, "Empty SMILES string.")

    # Create job for this molecule
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
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
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
            omegaOpts.SetMaxConfs(max_confs)
            omegaOpts.SetStrictStereo(False)
            omega = oeomega.OEOmega(omegaOpts)
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
            print(f"OpenEye conformer generation failed: {exc}, falling back to RDKit")
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
    JOBS[job_id] = {
        "job_id": job_id,
        "workflow_type": "lbdd",
        "conformers": conformers,
        "smiles": smiles,
        "status": "conformers_generated",
        "created_at": time.time(),
        "ckpt_path": _selected_ckpt_path,
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
async def select_conformer(job_id: str, request: SelectConformerRequest):
    """Select a specific conformer from the generated set for LBDD generation."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    conformers = job.get("conformers", [])
    if not conformers:
        raise HTTPException(400, "No conformers generated for this job.")
    if request.conformer_idx < 0 or request.conformer_idx >= len(conformers):
        raise HTTPException(400, f"Invalid conformer index: {request.conformer_idx}")

    selected = conformers[request.conformer_idx]
    sdf_block = selected["sdf"]

    # Save selected conformer as SDF file
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = job_dir / "selected_conformer.sdf"
    sdf_path.write_text(sdf_block)

    # Parse with RDKit for atom/bond info
    mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
    if mol is None:
        raise HTTPException(500, "Failed to parse the selected conformer.")

    mol_noH = Chem.RemoveHs(mol)
    smiles_noH = _mol_to_smiles(mol_noH)
    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    job.update(
        ligand_path=str(sdf_path),
        ligand_filename="selected_conformer.sdf",
        ligand_smiles=_mol_to_smiles(mol),
        status="ligand_uploaded",
        selected_conformer_idx=request.conformer_idx,
    )

    return {
        "job_id": job_id,
        "conformer_idx": request.conformer_idx,
        "smiles": _mol_to_smiles(mol),
        "smiles_noH": smiles_noH,
        "has_explicit_hs": has_explicit_hs,
        "sdf_data": sdf_block,
        "atoms": _mol_to_atom_info(mol),
        "bonds": _mol_to_bond_info(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "energy": selected.get("energy"),
    }


# ── Prior-cloud helpers (CPU-only, numpy) ──


def _parse_xyz_center(filepath: str) -> np.ndarray:
    """Parse an XYZ or simple-coordinate file and return the center of mass.

    Supports:
      - Standard XYZ (Element X Y Z)
      - Plain coordinates (X Y Z per line)
      - Numpy array-like format ([[ ... ]])

    Returns np.ndarray of shape (3,).
    """
    with open(filepath, "r") as fh:
        content = fh.read()

    coords: list[list[float]] = []

    if "[" in content and "]" in content:
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", content)]
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


def _compute_anisotropic_preview_covariance(
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
    mol_noH = Chem.RemoveHs(mol)
    if mol_noH.GetNumConformers() == 0 or mol_noH.GetNumAtoms() < 3:
        return None

    conf = mol_noH.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol_noH.GetNumAtoms())]
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

    if gen_mode in ("core_growing", "scaffold_hopping"):
        # Shape covariance from VARIABLE atoms only (matching interpolate.py)
        try:
            _mode_mask_fns_aniso = {
                "scaffold_hopping": lambda m: _extract_scaffold_mask(m),
                "core_growing": lambda m: ~_extract_core_mask(
                    m, ring_system_index=ring_system_index
                ),
            }
            mask_fn = _mode_mask_fns_aniso.get(gen_mode)
            if mask_fn:
                mask = mask_fn(mol_noH)
                variable_indices = np.where(mask)[0]
                if len(variable_indices) >= 3:
                    return _shape_covariance_np(coords[variable_indices])
        except Exception:
            pass
        return _shape_covariance_np(coords)

    # linker_inpainting, scaffold_elaboration, fragment_growing (without prior center)
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
    mol_noH = Chem.RemoveHs(mol)
    if mol_noH.GetNumConformers() == 0 or mol_noH.GetNumAtoms() < 3:
        return None
    conf = mol_noH.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol_noH.GetNumAtoms())]
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
            mask = mask_fn(mol_noH)
            variable_indices = np.where(mask)[0]
            if len(variable_indices) > 0:
                return coords[variable_indices].mean(axis=0)
    except Exception:
        pass
    return None


def _compute_prior_cloud_preview(
    job: Dict[str, Any],
    grow_size: int,
    pocket_cutoff: float = 6.0,
    anisotropic: bool = False,
    gen_mode: str = "fragment_growing",
    ring_system_index: int = 0,
    ref_ligand_com_prior: bool = False,
) -> Dict[str, Any]:
    """Compute a preview prior cloud on the CPU (no torch).

    Uses the uploaded prior-center file if available, otherwise falls back
    to the **protein pocket centre of mass** (matching what the FLOWR
    generation pipeline does: the prior is zero-COM in the pocket-COM frame,
    so in the original frame its centre is at the pocket COM).

    Supports anisotropic (mode-specific) sampling and ref ligand CoM shift.

    Returns dict {center, points, n_atoms, has_prior_center, anisotropic, ref_ligand_com_shifted}.
    """
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
                    mol_noH = Chem.RemoveHs(mol)
                    if mol_noH.GetNumConformers() == 0:
                        center = np.zeros(3)
                    else:
                        conf = mol_noH.GetConformer()
                        pts = np.array(
                            [
                                list(conf.GetAtomPosition(i))
                                for i in range(mol_noH.GetNumAtoms())
                            ]
                        )
                        center = pts.mean(axis=0)
                else:
                    center = np.zeros(3)
            elif job.get("protein_path"):
                # Pocket-only mode: compute COM of all protein atoms
                try:
                    coords = []
                    with open(job["protein_path"]) as f:
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

    rng = np.random.default_rng(seed=42)

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
async def upload_prior_center(job_id: str, file: UploadFile):
    """Upload an XYZ file for fragment growing prior center.

    Returns the filename **and** a preview prior cloud so the frontend can
    immediately visualise where new atoms will be grown.
    """
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    save_path = job_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)
    JOBS[job_id]["prior_center_path"] = str(save_path)

    # Compute preview cloud
    grow_size = JOBS[job_id].get("grow_size", 5)
    _gen_mode = JOBS[job_id].get("gen_mode", "fragment_growing")
    _rsi = JOBS[job_id].get("ring_system_index", 0)
    _aniso = JOBS[job_id].get("anisotropic_prior", False)
    _ref_com = JOBS[job_id].get("ref_ligand_com_prior", False)
    try:
        cloud = _compute_prior_cloud_preview(
            JOBS[job_id],
            grow_size,
            anisotropic=_aniso,
            gen_mode=_gen_mode,
            ring_system_index=_rsi,
            ref_ligand_com_prior=_ref_com,
        )
    except Exception as exc:
        cloud = None
        print(f"Prior cloud preview failed: {exc}")

    return {"job_id": job_id, "filename": safe_name, "prior_cloud": cloud}


@app.get("/prior-cloud-preview/{job_id}")
async def prior_cloud_preview(
    job_id: str,
    grow_size: int = 5,
    pocket_cutoff: float = 6.0,
    anisotropic: bool = False,
    gen_mode: str = "fragment_growing",
    ring_system_index: int = 0,
    ref_ligand_com_prior: bool = False,
):
    """Return a prior-cloud preview for visualisation.

    For fragment_growing, uses the explicit ``grow_size`` parameter.
    For other conditional modes, computes the number of atoms to be
    *replaced* via the inpainting mask so the cloud size matches the
    expected generation output.
    """
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]

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

    # Determine the correct cloud size based on generation mode
    effective_grow_size = grow_size
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
                mol_noH = Chem.RemoveHs(mol)
                mask = _mode_mask_fns[gen_mode](mol_noH)
                n_replaced = int(np.sum(mask))
                if n_replaced > 0:
                    effective_grow_size = n_replaced
        except Exception:
            pass  # Fall back to the explicit grow_size

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
async def generate(request: GenerationRequest):
    """Start generation by delegating to the GPU worker."""
    job_id = request.job_id
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    # Allow missing ligand when num_heavy_atoms is provided (de novo pocket-only / scratch)
    has_ligand = "ligand_path" in job
    has_denovo = request.num_heavy_atoms is not None and request.num_heavy_atoms > 0
    if not has_ligand and not has_denovo:
        raise HTTPException(400, "No ligand uploaded and no num_heavy_atoms specified.")
    # LBDD does not require a protein upload
    workflow = (
        request.workflow_type
        or job.get("workflow_type")
        or _selected_workflow_type
        or "sbdd"
    )
    if workflow != "lbdd" and "protein_path" not in job:
        raise HTTPException(400, "No protein uploaded.")
    if request.gen_mode not in _VALID_GEN_MODES:
        raise HTTPException(
            400,
            f"Invalid gen_mode '{request.gen_mode}'. "
            f"Must be one of: {', '.join(sorted(_VALID_GEN_MODES))}",
        )
    if job.get("status") in ("generating", "allocating_gpu", "starting"):
        raise HTTPException(409, "Generation already in progress for this job.")

    # Store gen settings in job for summary
    job.update(
        gen_mode=request.gen_mode,
        fixed_atoms=request.fixed_atoms,
        n_samples=request.n_samples,
        batch_size=request.batch_size,
        integration_steps=request.integration_steps,
    )

    req = {
        "gen_mode": request.gen_mode,
        "fixed_atoms": request.fixed_atoms,
        "n_samples": request.n_samples,
        "batch_size": request.batch_size,
        "integration_steps": request.integration_steps,
        "pocket_cutoff": request.pocket_cutoff,
        "coord_noise_scale": request.coord_noise_scale,
        "grow_size": request.grow_size,
        "prior_center_filename": request.prior_center_filename,
        "prior_center_coords": request.prior_center_coords,
        "filter_valid_unique": request.filter_valid_unique,
        "filter_cond_substructure": request.filter_cond_substructure,
        "filter_diversity": request.filter_diversity,
        "diversity_threshold": request.diversity_threshold,
        "sample_mol_sizes": request.sample_mol_sizes,
        "filter_pb_valid": request.filter_pb_valid,
        "calculate_pb_valid": request.calculate_pb_valid,
        "calculate_strain_energies": request.calculate_strain_energies,
        "optimize_gen_ligs": request.optimize_gen_ligs,
        "optimize_gen_ligs_hs": request.optimize_gen_ligs_hs,
        "anisotropic_prior": request.anisotropic_prior,
        "ring_system_index": request.ring_system_index,
        "ref_ligand_com_prior": request.ref_ligand_com_prior,
        "workflow_type": workflow,
        "optimize_method": request.optimize_method,
        "sample_n_molecules_per_mol": request.sample_n_molecules_per_mol,
        "num_heavy_atoms": request.num_heavy_atoms,
        "property_filter": request.property_filter,
        "adme_filter": request.adme_filter,
    }

    # Store workflow_type and checkpoint in job for proxy branch logic
    resolved_ckpt = request.ckpt_path or job.get("ckpt_path") or _selected_ckpt_path
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
async def cancel_job(job_id: str):
    """Request cancellation of a running generation job."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    if job.get("status") not in (
        "starting",
        "allocating_gpu",
        "generating",
        "loading_model",
    ):
        raise HTTPException(409, "Job is not currently running.")
    job["cancel_requested_at"] = time.time()
    job["cancelled"] = True
    job["status"] = "cancelled"
    job["error"] = "Cancelled by user"

    # Forward cancellation to the worker (best-effort)
    try:
        worker_url = _get_worker_url()
        http_requests.post(f"{worker_url}/cancel/{job_id}", timeout=5)
    except Exception:
        pass  # Worker may not be reachable — server-side cancel still works

    return {"job_id": job_id, "status": "cancelled"}


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """Poll job progress. The background proxy thread updates JOBS directly."""
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
            mode=job.get("mode", "unknown"),
            results=job.get("results", []),
            metrics=job.get("metrics", []),
            used_optimization=job.get("used_optimization", False),
            prior_cloud=job.get("prior_cloud"),
        )
    return resp


@app.get("/job/{job_id}/results")
async def get_job_results(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    return {"results": job.get("results", []), "mode": job.get("mode", "unknown")}


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
async def save_ligand(job_id: str, ligand_idx: int):
    """Save a single generated ligand as SDF."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
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

    out_path = output_dir / f"{base_name}_{job_id}_ligand_{ligand_idx}.sdf"
    out_path.write_text(sdf_data)

    return {"saved": True, "path": str(out_path), "filename": out_path.name}


@app.post("/save-all-ligands/{job_id}")
async def save_all_ligands(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No results to save.")

    if job.get("workflow_type") == "lbdd":
        base_name = Path(job.get("ligand_filename", "molecule")).stem
    else:
        base_name = Path(job.get("protein_filename", "unknown")).stem
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for idx, r in enumerate(results):
        sdf_data = r.get("sdf", "")
        if sdf_data:
            out_path = output_dir / f"{base_name}_{job_id}_ligand_{idx}.sdf"
            out_path.write_text(sdf_data)
            saved.append({"path": str(out_path), "filename": out_path.name})

    return {"saved_count": len(saved), "files": saved, "output_dir": str(output_dir)}


class SaveSelectedRequest(BaseModel):
    indices: List[int]


@app.post("/save-selected-ligands/{job_id}")
async def save_selected_ligands(job_id: str, request: SaveSelectedRequest):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No results to save.")

    if job.get("workflow_type") == "lbdd":
        base_name = Path(job.get("ligand_filename", "molecule")).stem
    else:
        base_name = Path(job.get("protein_filename", "unknown")).stem
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for idx in request.indices:
        if 0 <= idx < len(results):
            sdf_data = results[idx].get("sdf", "")
            if sdf_data:
                out_path = output_dir / f"{base_name}_{job_id}_ligand_{idx}.sdf"
                out_path.write_text(sdf_data)
                saved.append({"path": str(out_path), "filename": out_path.name})

    return {"saved_count": len(saved), "files": saved, "output_dir": str(output_dir)}


# ═══════════════════════════════════════════════════════════════════════════
#  REFERENCE LIGAND HYDROGENS
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/ligand-remove-hs/{job_id}")
async def ligand_remove_hs(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    ligand_path = job.get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, "No ligand uploaded.")

    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, "Cannot parse ligand.")

    mol_noH = Chem.RemoveHs(mol)
    smiles_noH = _mol_to_smiles(mol_noH)

    return {
        "sdf_data": _mol_to_sdf_string(mol_noH),
        "atoms": _mol_to_atom_info(mol_noH),
        "bonds": _mol_to_bond_info(mol_noH),
        "smiles": smiles_noH,
        "smiles_noH": smiles_noH,
        "num_atoms": mol_noH.GetNumAtoms(),
        "num_heavy_atoms": mol_noH.GetNumHeavyAtoms(),
    }


@app.post("/ligand-add-hs/{job_id}")
async def ligand_add_hs(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    ligand_path = job.get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, "No ligand uploaded.")

    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, "Cannot parse ligand.")

    has_explicit_hs = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    if has_explicit_hs:
        mol_withH = mol
    else:
        mol_withH = Chem.AddHs(mol, addCoords=True)

    smiles_withH = _mol_to_smiles(mol_withH)
    mol_noH_for_smiles = Chem.RemoveHs(mol_withH)
    smiles_noH = _mol_to_smiles(mol_noH_for_smiles)

    return {
        "sdf_data": _mol_to_sdf_string(mol_withH),
        "atoms": _mol_to_atom_info(mol_withH),
        "bonds": _mol_to_bond_info(mol_withH),
        "smiles": smiles_withH,
        "smiles_noH": smiles_noH,
        "num_atoms": mol_withH.GetNumAtoms(),
        "num_heavy_atoms": mol_withH.GetNumHeavyAtoms(),
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
        print(f"WARNING: OpenEye active-site depiction failed: {exc}")
        return None

    lopts = oegrapheme.OE2DActiveSiteLegendDisplayOptions(18, 1)
    oegrapheme.OEDrawActiveSiteLegend(legend_frame, adisp, lopts)

    oedepict.OEDrawCurvedBorder(image, oedepict.OELightGreyPen, 10.0)

    fd, tmp_path = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        oedepict.OEWriteImage(tmp_path, image)
        with open(tmp_path, "r") as f:
            svg = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return svg


@app.get("/interaction-diagram/{job_id}")
async def interaction_diagram(job_id: str, ligand_idx: int = -1):
    if not OPENEYE_AVAILABLE:
        raise HTTPException(
            501,
            "OpenEye not available. Install openeye-toolkits for 2D interaction diagrams.",
        )

    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    protein_path = job.get("protein_path")
    if not protein_path:
        raise HTTPException(400, "No protein uploaded.")

    if ligand_idx < 0:
        ligand_path = job.get("ligand_path")
        if not ligand_path:
            raise HTTPException(400, "No reference ligand uploaded.")
    else:
        results = job.get("results", [])
        if ligand_idx >= len(results):
            raise HTTPException(400, "Ligand index out of range.")
        sdf_data = results[ligand_idx].get("sdf", "")
        if not sdf_data:
            raise HTTPException(400, "No SDF data.")
        ligand_path = str(UPLOAD_DIR / job_id / f"gen_lig_{ligand_idx}.sdf")
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
async def compute_interactions_endpoint(job_id: str, ligand_idx: int = -1):
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, "RDKit not available.")
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    protein_path = job.get("protein_path")
    if not protein_path:
        raise HTTPException(400, "No protein uploaded.")

    if ligand_idx < 0:
        ligand_path = job.get("ligand_path")
        if not ligand_path:
            raise HTTPException(400, "No reference ligand.")
        mol = _read_ligand_mol(ligand_path)
    else:
        results = job.get("results", [])
        if ligand_idx >= len(results):
            raise HTTPException(400, "Ligand index out of range.")
        sdf_data = results[ligand_idx].get("sdf", "")
        if not sdf_data:
            raise HTTPException(400, "No SDF data.")
        mol = Chem.MolFromMolBlock(sdf_data, removeHs=True)

    if mol is None:
        raise HTTPException(400, "Cannot parse ligand.")

    interactions = _compute_interactions(protein_path, mol)
    return {"interactions": interactions, "count": len(interactions)}


# ═══════════════════════════════════════════════════════════════════════════
#  INPAINTING MASK COMPUTATION (CPU-only, no torch)
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/ring-systems/{job_id}")
async def get_ring_systems(job_id: str):
    """Return the number of ring systems in the uploaded ligand."""
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, "RDKit not available.")
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    ligand_path = JOBS[job_id].get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, "No ligand uploaded.")
    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, "Cannot parse ligand.")
    mol_noH = Chem.RemoveHs(mol)
    n = _get_num_ring_systems(mol_noH)
    return {"num_ring_systems": n}


@app.get("/inpainting-mask/{job_id}")
async def compute_inpainting_mask_endpoint(
    job_id: str, mode: str = "scaffold_hopping", ring_system_index: int = 0
):
    """Return atom indices that will be REPLACED for a given inpainting mode.

    This is a pure RDKit + numpy implementation — no PyTorch required.
    """
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, "RDKit not available for mask computation.")

    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    ligand_path = job.get("ligand_path")
    if not ligand_path:
        raise HTTPException(400, "No ligand uploaded.")

    mol = _read_ligand_mol(ligand_path)
    if mol is None:
        raise HTTPException(400, "Cannot parse ligand.")

    mol_noH = Chem.RemoveHs(mol)

    try:
        if mode == "scaffold_hopping":
            replaced_mask = _extract_scaffold_mask(mol_noH)  # scaffold = REPLACED
        elif mode == "scaffold_elaboration":
            replaced_mask = _extract_scaffold_elaboration_mask(mol_noH)
        elif mode == "linker_inpainting":
            replaced_mask = _extract_linker_mask(mol_noH)  # linkers = REPLACED
        elif mode == "core_growing":
            core_mask = _extract_core_mask(mol_noH, ring_system_index=ring_system_index)
            replaced_mask = ~core_mask  # non-core = REPLACED
        elif mode == "fragment_growing":
            replaced_mask = np.zeros(mol_noH.GetNumAtoms(), dtype=bool)
        else:
            return {"replaced": [], "fixed": [], "mode": mode}

        replaced_heavy = np.where(replaced_mask)[0].tolist()
        fixed_heavy = np.where(~replaced_mask)[0].tolist()

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
async def chemical_space(job_id: str, method: str = "pca", perplexity: int = 30):
    """Compute 2D projection of Morgan fingerprints for generated ligands + reference.

    Returns coordinates for scatter plot (PCA, t-SNE, or UMAP) plus properties.
    Results are cached per job_id + method to avoid recomputation.
    """
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, "RDKit not available.")
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No generated ligands available.")

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
    ref_mol_noH = Chem.RemoveHs(ref_mol)
    ref_fp = _mol_to_morgan_fp(ref_mol_noH)
    if ref_fp is None:
        raise HTTPException(500, "Failed to compute reference fingerprint.")
    ref_props = _compute_all_properties(ref_mol_noH)

    # Generated ligand fingerprints + properties (use pre-computed when available)
    fps = [ref_fp]
    gen_data = []
    for i, r in enumerate(results):
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
    for j, gd in enumerate(gen_data):
        c = coords[j + 1]  # offset by 1 for reference
        gd["x"] = round(float(c[0]), 4)
        gd["y"] = round(float(c[1]), 4)
        response["ligands"].append(gd)

    # Cache result
    job[cache_key] = response
    return response


@app.get("/property-space/{job_id}")
async def property_space(job_id: str):
    """Compute all RDKit properties for reference + generated ligands.

    Returns data suitable for violin / distribution plots.
    """
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, "RDKit not available.")
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No generated ligands available.")

    # Check cache
    cached = job.get("propspace_cache")
    if cached is not None:
        return cached

    # Reference
    ref_mol = _read_ligand_mol(job.get("ligand_path", ""))
    ref_mol_noH = Chem.RemoveHs(ref_mol) if ref_mol else None
    ref_props = _compute_all_properties(ref_mol_noH) if ref_mol_noH else {}

    # Generated — use pre-computed properties when available
    ligands = []
    for i, r in enumerate(results):
        existing_props = r.get("properties")
        if existing_props:
            ligands.append(
                {
                    "idx": i,
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
                    "smiles": r.get("smiles", ""),
                    "properties": _compute_all_properties(mol),
                }
            )

    response = {
        "reference": ref_props,
        "ligands": ligands,
        "property_names": ALL_PROPERTY_NAMES,
        "continuous_properties": list(CONT_PROPERTIES_RDKIT.keys()),
        "discrete_properties": list(DISC_PROPERTIES_RDKIT.keys())
        + ["NumChiralCenters"],
    }

    job["propspace_cache"] = response
    return response


@app.get("/ligand-properties/{job_id}/{ligand_idx}")
async def ligand_properties(job_id: str, ligand_idx: int):
    """Return full RDKit property set for a single ligand.

    ligand_idx = -1 returns reference ligand properties.
    """
    if not RDKIT_AVAILABLE:
        raise HTTPException(501, "RDKit not available.")
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]

    if ligand_idx < 0:
        mol = _read_ligand_mol(job.get("ligand_path", ""))
        if mol is None:
            raise HTTPException(400, "Cannot parse reference ligand.")
        mol = Chem.RemoveHs(mol)
    else:
        results = job.get("results", [])
        if ligand_idx >= len(results):
            raise HTTPException(400, "Ligand index out of range.")
        sdf = results[ligand_idx].get("sdf_no_hs") or results[ligand_idx].get("sdf", "")
        mol = Chem.MolFromMolBlock(sdf, removeHs=True) if sdf else None
        if mol is None:
            raise HTTPException(400, "Cannot parse ligand.")

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
async def rank_select(job_id: str, request: RankSelectRequest):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No results to rank.")
    if request.affinity_type not in _AFFINITY_TYPES:
        raise HTTPException(
            400,
            f"Invalid affinity_type '{request.affinity_type}'. "
            f"Must be one of {sorted(_AFFINITY_TYPES)}.",
        )

    # Back up original results (only once)
    if "results_full" not in job:
        job["results_full"] = list(results)

    aff = request.affinity_type

    def _sort_key(r):
        v = r.get("properties", {}).get(aff)
        return (v is not None, v if v is not None else float("-inf"))

    sorted_results = sorted(job["results_full"], key=_sort_key, reverse=True)
    if request.top_n is not None and request.top_n > 0:
        sorted_results = sorted_results[: request.top_n]

    job["results"] = sorted_results

    # Invalidate visualisation caches
    for key in list(job.keys()):
        if key.startswith("chemspace_") or key in (
            "propspace_cache",
            "affinity_dist_cache",
        ):
            del job[key]

    return {
        "results": sorted_results,
        "affinity_type": aff,
        "total_before": len(job["results_full"]),
        "total_after": len(sorted_results),
    }


@app.post("/reset-rank/{job_id}")
async def reset_rank(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    if "results_full" not in job:
        raise HTTPException(400, "No rank selection to reset.")

    job["results"] = job.pop("results_full")

    # Invalidate visualisation caches
    for key in list(job.keys()):
        if key.startswith("chemspace_") or key in (
            "propspace_cache",
            "affinity_dist_cache",
        ):
            del job[key]

    return {"results": job["results"], "total": len(job["results"])}


# ---------------------------------------------------------------------------
# Affinity Distribution
# ---------------------------------------------------------------------------


@app.get("/affinity-distribution/{job_id}")
async def affinity_distribution(job_id: str):
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
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    results = job.get("results", [])
    if not results:
        raise HTTPException(400, "No generated ligands available.")

    # Check cache
    cached = job.get("affinity_dist_cache")
    if cached is not None:
        return cached

    aff_keys = ["pic50", "pki", "pkd", "pec50"]
    distributions: Dict[str, Dict[str, list]] = {}
    available_types: List[str] = []

    for key in aff_keys:
        values = []
        labels = []
        indices = []  # 0-based result index for click-to-select
        for i, r in enumerate(results):
            v = r.get("properties", {}).get(key)
            if v is not None:
                values.append(round(float(v), 4))
                labels.append(f"Ligand #{i + 1}")
                indices.append(i)
        if values:
            distributions[key] = {
                "values": values,
                "labels": labels,
                "indices": indices,
            }
            available_types.append(key)

    if not available_types:
        raise HTTPException(
            400,
            "No affinity predictions available. The model checkpoint may not "
            "include an affinity prediction head.",
        )

    ref_affinity = job.get("ref_affinity")

    response = {
        "affinity_types": available_types,
        "distributions": distributions,
        "ref_affinity": ref_affinity,
        "n_ligands": len(results),
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
