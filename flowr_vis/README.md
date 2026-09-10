# FLOWR Visualization Web App

Interactive web application for structure-based ligand generation with
[FLOWR](../README.md). Upload a protein structure, select atoms for
inpainting, and generate novel ligands — all from your browser.

## Architecture

The app uses a **two-tier** design:

| Component | File | Role | Requires GPU? |
|-----------|------|------|:---:|
| **Frontend server** | `server.py` | Web UI, molecule parsing, property computation, SLURM lifecycle | No |
| **GPU worker** | `worker.py` | Model loading & ligand generation | Yes |

The frontend proxies generation requests to the worker. On HPC, the
frontend auto-submits a SLURM GPU job on demand and tears it down after
idle timeout.

```
Browser ──► server.py (CPU) ──► worker.py (GPU)
                │                     │
                ├─ frontend/          └─ flowr model
                │   ├─ index.html
                │   ├─ app.js
                │   └─ style.css
                └─ chem_utils.py (shared chemistry)
```

## Quick Start

### 1. Set up the environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/) from the
single `pyproject.toml` at the project root. The visualization app's own
dependencies (FastAPI, RDKit, scikit-learn, UMAP, biotite, …) live in the
`vis` extra; PyTorch comes from the mutually exclusive `cpu` / `gpu`
extras.

```bash
# From the project root — run once:
uv sync --extra cpu --extra vis     # macOS / Apple Silicon (CPU)
uv sync --extra gpu --extra vis     # Linux with CUDA
```

That creates `.venv/` at the project root with RDKit, FastAPI, PyTorch
and the `flowr` package itself (editable install — no `PYTHONPATH`
needed). Every script below runs Python via `uv run --no-sync`, so
there is nothing to activate.

> **Note:** the old `requirements.txt` / `requirements_worker.txt` files
> have been removed — those dependencies are now the `vis` extra of the
> root `pyproject.toml`. Use `uv sync --extra … --extra vis` instead of
> `pip install -r`.

`cpu` and `gpu` are mutually exclusive — they only decide which PyTorch
wheel index is used. Note that `torch` is a base dependency of the root
project, so it is installed either way, even on a frontend-only machine;
picking `--extra cpu` there just gets you the much smaller CPU wheel.

### 2. Place a model checkpoint

The landing page lists checkpoints from exactly two directories under
`ckpts/` at the project root — **nothing else is scanned**, and neither
directory exists in a fresh clone:

| Workflow | Directory | Checkpoints that belong there |
|----------|-----------|-------------------------------|
| Structure-based (SBDD) | `ckpts/sbdd/` | `flowr_root_v2.2.ckpt`, `flowr_root_v2.ckpt`, `flowr_root_spindr_base.ckpt` |
| Ligand-based (LBDD) | `ckpts/lbdd/` | `flowr_root_v2_mol.ckpt` |

```bash
# From the project root — create the directory for the workflow you need:
mkdir -p ckpts/sbdd ckpts/lbdd

mv flowr_root_v2.2.ckpt   ckpts/sbdd/
mv flowr_root_v2_mol.ckpt ckpts/lbdd/
```

A `.ckpt` left directly in `ckpts/` is **not** picked up: the landing
page shows "No base checkpoints found" and the **Launch** button stays
disabled. Download links for every checkpoint are in the root
[README](../README.md#checkpoints).

Checkpoints that the app fine-tunes itself are written to
`ckpts/<workflow>/project_model/<project>/` and appear under **Project**
in the checkpoint picker.

### Launch locally

```bash
# From the project root:
./flowr_vis/run_local.sh
```

This starts both the frontend (port 8787) and the worker (port 8788).
Open **<http://localhost:8787>** in your browser. Press Ctrl+C to stop
both. (The scripts resolve their own paths, so they work from any
working directory.)

#### Options

```bash
./flowr_vis/run_local.sh --server-port 9000 --worker-port 9001
./flowr_vis/run_local.sh --ckpt /path/to/checkpoint.ckpt
```

`--ckpt` only sets the worker's *fallback* checkpoint — the model that
actually gets loaded is the one you select on the landing page. If the
legacy default path `ckpts/flowr_root.ckpt` is absent, `run_local.sh`
prints a `WARNING: Checkpoint not found` line that is harmless as long
as `ckpts/sbdd/` or `ckpts/lbdd/` is populated.

### Launch on HPC (SLURM)

On HPC clusters the frontend runs on a login/CPU node and dynamically
allocates GPU jobs via SLURM when the user clicks **Generate**.

#### 1. Create the environment and your configuration

On the cluster (login node, on the shared filesystem) run the one-time
sync so that `.venv/` exists at the project root:

```bash
cd <project root>
uv sync --extra gpu --extra vis
```

The scripts prepend `~/.local/bin` to `PATH` so `uv` is also found in the
non-interactive shell that SLURM gives the worker job.

All remaining user-specific paths live in a single config file. Copy the
template and fill in the values for your cluster:

```bash
cd flowr_vis/hpc/
cp hpc.env.template hpc.env
```

Open `hpc.env` in your editor and set at minimum:

| Variable | What to set |
|----------|-------------|
| `PROJECT_ROOT` | Absolute path to the project root (the directory holding `.venv`). Leave blank to auto-detect. |
| `CKPT_PATH` | Path to the model checkpoint, absolute or relative to the project root. |
| `FLOWR_SLURM_PARTITION` | Your cluster's GPU partition name. |
| `FLOWR_SLURM_TIME` | Wall-clock limit for the GPU job (default: `04:00:00`). |
| `FLOWR_SLURM_MEM_PER_CPU` | Memory per CPU core (default: `12G`). |
| `FLOWR_SLURM_CPUS_PER_TASK` | CPU cores per worker job (default: `8`). |
| `FLOWR_SLURM_GRES` | GPU request passed to `--gres` (default: `gpu:1`). |
| `FLOWR_SLURM_OUTPUT_DIR` | Where SLURM stdout/stderr logs go (default: `~/slurm_outs`). |

Set the job's resources **here — not** in the `#SBATCH` headers of
`worker_hpc.sh`. The frontend always builds the `sbatch` command line
from the `FLOWR_SLURM_*` values above (`--partition`, `--time`,
`--mem-per-cpu`, `--cpus-per-task`, `--gres`, `--output`, `--error`),
and command-line flags override in-script `#SBATCH` directives — so
editing those headers has no effect. Only the headers the frontend does
*not* pass (`-J`, `--nodes`, `--ntasks-per-node`) still apply.

> **Note:** `hpc.env` is git-ignored so your personal paths won't be
> committed.

#### 2. Launch the frontend

```bash
# From the project root:
./flowr_vis/hpc/run_hpc_frontend.sh

# Or with a custom config location:
./flowr_vis/hpc/run_hpc_frontend.sh --config /path/to/my.env
```

#### 3. Connect from your laptop

SSH-tunnel the frontend port to your local machine:

```bash
ssh -N -L 8787:<node>:8787 <user>@<hpc-login-node>
```

Then open **<http://localhost:8787>** in your browser.

GPU workers are submitted automatically when a user clicks **Generate**.
They auto-shutdown after the idle timeout set by `WORKER_IDLE_TIMEOUT`
in `hpc.env` (default: `120` seconds).

#### HPC File Overview

| File | Purpose |
|------|---------|
| `hpc.env.template` | Configuration template — copy to `hpc.env` and edit |
| `hpc.env` | Your local config (git-ignored) |
| `run_hpc_frontend.sh` | Starts the CPU-only frontend server (thin wrapper) |
| `run_hpc_frontend_common.sh` | Shared frontend startup logic (sourced by `run_hpc_frontend.sh`) |
| `worker_hpc.sh` | SLURM job script submitted for GPU workers |
| `worker_common.sh` | Shared worker startup logic (sourced by `worker_hpc.sh`) |

### Running Components Separately

Useful when the frontend (CPU) and worker (GPU) run on different machines.

#### Frontend only

```bash
# From the project root — on a CPU-only frontend host:
uv sync --extra cpu --extra vis

# Start the frontend server:
./flowr_vis/run_server.sh \
    --worker-url http://gpu-host:8788 \
    --server-url http://cpu-host:8787
```

`--server-url` is **required** whenever the worker runs on another
machine. The worker downloads the uploaded protein/ligand files back
from the frontend over HTTP, using the address the frontend hands it.
That address defaults to `http://localhost:<port>`, which a remote
worker resolves to its own loopback — so every generation fails at the
download step. Point it at a host name or IP the GPU host can reach
(the `FLOWR_SERVER_URL` environment variable does the same job).

The frontend server code does **not** import PyTorch, and it touches the
`flowr` package only through one lazy import inside the "fetch from
RCSB" route (`flowr.data.preprocess_pdb`), which degrades to a clear
error message when it is unavailable. Otherwise it needs only RDKit,
FastAPI, scikit-learn and umap-learn — exactly the `vis` extra. (Torch
still gets installed because the root project depends on it; picking
`--extra cpu` keeps that download small on a frontend-only box.)

#### Worker only

```bash
# From the project root — the worker needs a torch extra as well:
uv sync --extra gpu --extra vis     # Linux / CUDA
uv sync --extra cpu --extra vis     # macOS / CPU

# Start the GPU worker:
./flowr_vis/run_worker.sh --port 8788 --ckpt /path/to/model.ckpt
```

The worker runs on CUDA where available and falls back to CPU otherwise; it needs
the `flowr` package, which the sync installs in editable mode. Apple's MPS backend is
**not** selected automatically -- see "Device selection" in the root README. CPU
generation works but takes minutes rather than seconds.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLOWR_PORT` | `8787` | Frontend server port |
| `FLOWR_WORKER_URL` | `http://localhost:8788` | Worker address (static mode) |
| `FLOWR_WORKER_MODE` | `static` | `static` or `slurm` |
| `FLOWR_CKPT_PATH` | `ckpts/flowr_root.ckpt` | Worker's fallback checkpoint — the landing-page selection overrides it per request |
| `FLOWR_WORKER_PORT` | `8788` | Worker listen port |
| `FLOWR_WORKER_IDLE_TIMEOUT` | `0` (disabled) | Auto-shutdown after N seconds idle. `0` means the watchdog is never armed, so a locally launched worker holds its GPU until you stop it; the HPC scripts set `120`. |
| `FLOWR_SLURM_WORKER_SCRIPT` | `hpc/worker_hpc.sh` | SLURM submission script |
| `FLOWR_SLURM_STARTUP_TIMEOUT` | `300` | Max seconds to wait for GPU node |
| `FLOWR_CKPTS_DIR` | `<project root>/ckpts` | Root of the checkpoint tree; only its `sbdd/` and `lbdd/` subdirectories are listed |
| `FLOWR_SERVER_URL` | `http://localhost:<port>` | Address the worker uses to download uploaded files from the frontend |

## Optional: OpenEye 2D Interaction Diagrams

If you have an OpenEye license, the frontend can render 2D protein–ligand
interaction diagrams. Place the license file at one of:

- `flowr_vis/tools/oe_license.txt`
- `oe_license.txt` (project root)

The server auto-detects the license on startup.

## Directory Structure

```
flowr_vis/
├── server.py              # Frontend FastAPI server (CPU-only)
├── worker.py              # GPU worker FastAPI server
├── chem_utils.py          # Shared chemistry utilities
├── oe_conformer.py        # OpenEye conformer/alignment helpers (optional)
├── run_local.sh           # Launch both locally
├── run_server.sh          # Launch frontend only
├── run_worker.sh          # Launch worker only
├── LICENSE                # Source-available license for this app
├── frontend/              # Static web assets
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   ├── molstar-embed.html # Mol* viewer, loaded in an iframe
│   └── lib/               # Vendored JS libraries (3Dmol, RDKit.js, Mol*, Plotly)
├── hpc/                   # HPC/SLURM scripts
│   ├── hpc.env.template   # Configuration template (copy → hpc.env)
│   ├── run_hpc_frontend.sh
│   ├── run_hpc_frontend_common.sh
│   ├── worker_common.sh
│   └── worker_hpc.sh
└── tools/                 # Optional utilities
    └── interact_openeye.py
```

## Frontend Libraries

- [3Dmol.js 2.4.2](https://3dmol.csb.pitt.edu/) — 3D molecular viewer
- [RDKit.js 2025.03.4](https://github.com/rdkit/rdkit-js) — 2D structure rendering
- [Plotly.js 2.35.0](https://plotly.com/javascript/) — Chemical/property space charts

## License

The `flowr_vis/` visualization app is released under a separate
**source-available license** — distinct from the MIT license that covers
the rest of the FLOWR project. See the [LICENSE](LICENSE) file in this
directory for full terms.

**In short:** You may freely use this software for any purpose — academic
research, education, and commercial research within your organization.
You may **not** redistribute, clone, modify for distribution, sublicense,
or sell copies of this software.
