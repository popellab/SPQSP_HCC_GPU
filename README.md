# SPQSP HCC — GPU Agent-Based Model

GPU-accelerated agent-based model (FLAME GPU 2) with CPU QSP coupling (SUNDIALS CVODE) for simulating hepatocellular carcinoma (HCC) tumor microenvironment dynamics.

## Requirements

- **CUDA Toolkit** 11.0+ (tested up to 12.8)
- **CMake** 3.18+
- **C++17 compiler** (g++ 7+)
- **git**

FLAME GPU 2, SUNDIALS, and Boost are **automatically downloaded and built** if not already present on the system.

## Quick Start (HPC)

```bash
cd HCC/sim
cp cluster.conf.example cluster.conf   # edit ACCOUNT for your allocation
./setup_deps.sh                        # one command — fetches FLAMEGPU2, SUNDIALS, Boost
sbatch submit.sh -s 10 -g 11           # build + test run (first build ~8 min)
```

`submit.sh` auto-detects the cluster (Delta, Anvil), loads modules, builds if needed, runs the simulation on scratch, and copies outputs back to `HCC/sim/outputs/<job_id>/`.

### Supported Clusters

| Cluster | Partition (default) | CUDA arch | Scratch |
|---------|-------------------|-----------|---------|
| Delta (NCSA) | `gpuA100x4` | 80 (A100), 86 (A40), 90 (H200) | `/work/hdd/<project>/<user>` |
| Anvil (Purdue) | `gpu` | 80 (A100) | `/anvil/scratch/<user>` |

Adding a new cluster: add a `setup_<name>()` function in `submit.sh` and a hostname pattern in `detect_cluster()`.

## Quick Start (Local)

If you have CUDA and cmake available locally (workstation, Docker, etc.):

```bash
cd HCC/sim
./build.sh                    # auto-fetches deps via network
./build/bin/hcc -s 10 -g 11  # quick test run
```

First build takes ~8 minutes (downloads and compiles all dependencies). Subsequent builds are incremental (~1-2 min).

## Setup Details

### `cluster.conf`

Per-user SLURM settings. Only `ACCOUNT` is required — everything else is auto-detected:

```bash
ACCOUNT="bgre-delta-gpu"     # required
# PARTITION=""                # override auto-detected partition
# CUDA_ARCH=""                # override auto-detected GPU arch
# SCRATCH_BASE=""             # override auto-detected scratch path
```

### `setup_deps.sh`

Fetches FLAMEGPU2, SUNDIALS, and Boost into `external/`. Idempotent — skips deps already present. Run on a node with internet access (login nodes).

```bash
./setup_deps.sh              # fetch all deps
./setup_deps.sh --status     # check what's fetched
./setup_deps.sh --clean      # remove external/ and re-fetch
```

If your cluster has internet on GPU nodes, you can skip `setup_deps.sh` — CMake will auto-fetch deps during the build.

### `build.sh`

Portable build script. Works anywhere with cmake, nvcc, g++, and git on PATH.

```
./build.sh [options]

  --cuda-arch ARCH        Target GPU architecture (e.g., 80 for A100, 90 for H100)
  --debug                 Debug build with CUDA device debugging
  -j, --jobs N            Parallel build jobs (default: nproc)
  --flamegpu PATH         Use local FLAME GPU 2 source instead of fetching
  --clean                 Remove build directory
```

System-installed libraries can be used via environment variables:

```bash
SUNDIALS_DIR=/path/to/sundials BOOST_ROOT=/path/to/boost ./build.sh
```

## Running

```bash
./build/bin/hcc [options]

  -p, --param-file PATH   XML parameter file (default: resource/param_all_test.xml)
  -g, --grid-size N       Grid dimensions N×N×N (default: from XML, typically 50 = 1 mm³ at 20 µm/voxel)
  -s, --steps N           Main-loop simulation steps (default: 200, each step = 6 hours)
  -G, --grid-output N     Grid snapshot output: 0=none, 1=ABM only, 2=PDE+ECM only, 3=both (default: 0)
  -oi, --out_int N        Grid output interval in steps (default: 1)
      --seed N            RNG seed (default: 12345). Also stamps output CSV filenames.
  -vm, --vascular-mode S  Vasculature init: random | xml | test (default: random)
  -vx, --vascular-xml F   XML file for vasculature when --vascular-mode=xml
  -h, --help              Show this help
```

Several simulation parameters that used to be CLI flags now live in the XML instead:

| What | XML path |
|------|----------|
| Initial tumor radius (voxels) | `ABM.Init.tumor_radius` |
| Initial T cell count | `ABM.Init.init_Tcell_n` |
| Drug regimens (nivo / ipi / cabo) | `ABM.Pharmacokinetics.*` |
| Agent move steps per slice | `ABM.{TCell,Mac,Fib,MDSC,Cancer}.moveSteps` |

The **pre-simulation** (ABM + QSP, no drugs, runs until the QSP tumor compartment reaches the target volume) happens unconditionally before the main loop — there is no `-i` / `--qsp-init` flag anymore. To skip the main loop and keep only the presim outputs (useful for spatial calibration), pass `-s 0`.

### SLURM Submission

```bash
sbatch submit.sh                       # use script defaults
sbatch submit.sh -s 1000 -g 101        # override steps / grid
sbatch submit.sh -s 0 -G 1             # presim-only, write ABM snapshots
```

What `submit.sh` does:
1. Reads `cluster.conf` for account name
2. Auto-detects cluster, loads modules, picks partition and CUDA arch
3. Builds the binary on the GPU node if it doesn't exist
4. Creates a run directory on scratch for fast I/O
5. Runs the simulation
6. Copies outputs back to `HCC/sim/outputs/<job_id>/`

### Rebuilding

```bash
./build.sh --clean          # removes old build
sbatch submit.sh            # next submission rebuilds
```

## Output Files

Outputs are written to `./outputs/` relative to the working directory. On SLURM, they are also copied to `HCC/sim/outputs/<job_id>/`.

**Main-loop outputs**

| File | Contents |
|------|----------|
| `outputs/abm/agents_step_NNNNNN.abm.lz4` | Agent positions, states, properties (LZ4 int32, 8 cols) |
| `outputs/pde/pde_step_NNNNNN.pde.lz4`    | Chemical concentrations (10 species, LZ4 float32) |
| `outputs/ecm/ecm_step_NNNNNN.ecm.lz4`    | ECM (fibroblast) density field (LZ4) |
| `outputs/qsp_<seed>.csv`                 | QSP ODE state per step |
| `outputs/stats_<seed>.csv`               | Per-step agent counts, recruitment, proliferation, death events, PDL1 fraction |
| `outputs/timing_<seed>.csv`              | Per-step wall-time breakdown |
| `outputs/layer_timing.csv`               | Per-layer wall-time breakdown |
| `outputs/abm_lz4_def.txt`, `pde_lz4_def.txt` | Binary format descriptions written at run start |

**Pre-simulation outputs** (always written — capture the warmed-up TME before drugs)

| File | Contents |
|------|----------|
| `outputs/presim/abm/agents_presim_NNNNNN.abm.lz4` | Agent snapshot every presim step (final file = day-0 state) |
| `outputs/presim/pde/pde_presim_NNNNNN.pde.lz4`    | Substrate fields per presim step |
| `outputs/presim/ecm/ecm_presim_NNNNNN.ecm.lz4`    | ECM field per presim step |
| `outputs/presim/qsp_<seed>.csv`                   | QSP state during the presim phase |
| `outputs/presim/stats_<seed>.csv`                 | Agent counts / events during presim |

### CUDA Architecture Reference

| GPU | Architecture |
|-----|-------------|
| V100 | 70 |
| RTX 2080 / T4 | 75 |
| A100 | 80 |
| A40 / RTX 3090 | 86 |
| RTX 4090 | 89 |
| H100 / H200 | 90 |

## Notes

- **First run** after build takes 5-10 minutes for CUDA JIT warmup (not a hang).
- **Memory**: Grid 50^3 uses ~2 GB VRAM; 320^3 uses ~8 GB.
- SLURM logs go to `hcc_<job_id>.out` / `.err` in the directory you submit from.
- Output files are written to `./outputs/` **relative to the working directory** — run from a clean run dir to keep outputs isolated.
- FLAMEGPU2 RTC is disabled (`FLAMEGPU_ENABLE_RTC=OFF`) so there is no JIT at startup.
- For spatial-only analyses (FunCN, etc.) pass `-s 0` so only the presim phase runs; see `HCC/calibration/` for the end-to-end calibration pipeline.
