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
uv sync --extra cpu --extra vis     # macOS / Apple Silicon (MPS or CPU)
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

Put at least one `.ckpt` file in the `ckpts/` directory at the project
root. The default path is `ckpts/flowr_root.ckpt`.

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
| `FLOWR_SLURM_OUTPUT_DIR` | Where SLURM stdout/stderr logs go (default: `~/slurm_outs`). |

You may also want to adjust the `#SBATCH` headers directly in
`worker_hpc.sh` (time limit, memory, GPU type, partition) to match
your cluster's resource limits.

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
They auto-shutdown after the configured idle timeout (default: 2 min).

#### HPC File Overview

| File | Purpose |
|------|---------|
| `hpc.env.template` | Configuration template — copy to `hpc.env` and edit |
| `hpc.env` | Your local config (git-ignored) |
| `run_hpc_frontend.sh` | Starts the CPU-only frontend server |
| `worker_hpc.sh` | SLURM job script submitted for GPU workers |
| `worker_common.sh` | Shared worker startup logic (sourced by `worker_hpc.sh`) |

### Running Components Separately

Useful when the frontend (CPU) and worker (GPU) run on different machines.

#### Frontend only

```bash
# From the project root — on a CPU-only frontend host:
uv sync --extra cpu --extra vis

# Start the frontend server:
./flowr_vis/run_server.sh --worker-url http://gpu-host:8788
```

The frontend server code does **not** import PyTorch or the `flowr`
package — it only needs RDKit, FastAPI, scikit-learn and umap-learn,
which are exactly the `vis` extra. (Torch still gets installed because
the root project depends on it; `--extra cpu` keeps that download small
on a frontend-only box.)

#### Worker only

```bash
# From the project root — the worker needs a torch extra as well:
uv sync --extra gpu --extra vis     # Linux / CUDA
uv sync --extra cpu --extra vis     # macOS / CPU (MPS)

# Start the GPU worker:
./flowr_vis/run_worker.sh --port 8788 --ckpt /path/to/model.ckpt
```

The worker requires PyTorch with CUDA or MPS support plus the `flowr`
package, which the sync installs in editable mode.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLOWR_PORT` | `8787` | Frontend server port |
| `FLOWR_WORKER_URL` | `http://localhost:8788` | Worker address (static mode) |
| `FLOWR_WORKER_MODE` | `static` | `static` or `slurm` |
| `FLOWR_CKPT_PATH` | `ckpts/flowr_root.ckpt` | Model checkpoint |
| `FLOWR_WORKER_PORT` | `8788` | Worker listen port |
| `FLOWR_WORKER_IDLE_TIMEOUT` | `120` | Auto-shutdown after N seconds idle |
| `FLOWR_SLURM_WORKER_SCRIPT` | `hpc/worker_hpc.sh` | SLURM submission script |
| `FLOWR_SLURM_STARTUP_TIMEOUT` | `300` | Max seconds to wait for GPU node |
| `FLOWR_CKPTS_DIR` | `../ckpts` | Directory to list available checkpoints |
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
├── run_local.sh           # Launch both locally
├── run_server.sh          # Launch frontend only
├── run_worker.sh          # Launch worker only
├── frontend/              # Static web assets
│   ├── index.html
│   ├── app.js
│   └── style.css
├── hpc/                   # HPC/SLURM scripts
│   ├── hpc.env.template   # Configuration template (copy → hpc.env)
│   ├── run_hpc_frontend.sh
│   ├── worker_common.sh
│   └── worker_hpc.sh
└── tools/                 # Optional utilities
    ├── interact_openeye.py
    └── oe_license.txt
```

## Frontend Libraries

- [3Dmol.js 2.4.2](https://3dmol.csb.pitt.edu/) — 3D molecular viewer
- [RDKit.js 2024.3.3](https://github.com/rdkit/rdkit-js) — 2D structure rendering
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
