#!/bin/bash
#SBATCH --job-name=hcc_sweep
#SBATCH --account=bgre-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchris80@jh.edu

# ============================================================================
# HCC moveSteps parameter sweep on Delta (A100 / gpuA100x4).
#
# Runs all 125 (TCell, Fib, Mac) combinations sequentially — one at a time —
# so only one hcc process ever holds the GPU.  Each combination runs 5 seeds,
# also sequentially.  Total: 625 presim simulations.
#
# After the job finishes, merge all per-task CSVs:
#   python -m HCC.calibration.sweep merge \
#       --results-dir HCC/calibration/results/<JOB_ID>/
#
# Usage:
#   sbatch HCC/calibration/submit_sweep.sh
#
#   # Override grid size (default: 50):
#   GRID=40 sbatch HCC/calibration/submit_sweep.sh
#
#   # Dry-run a subset (tasks 0–4 only):
#   TASK_END=4 sbatch HCC/calibration/submit_sweep.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CALIB_DIR="${REPO_ROOT}/HCC/calibration"
SIM_DIR="${REPO_ROOT}/HCC/sim"
HCC_BIN="${SIM_DIR}/build/bin/hcc"
BASE_XML="${SIM_DIR}/resource/param_all_test.xml"

mkdir -p "${CALIB_DIR}/logs"

# ── Scratch layout ───────────────────────────────────────────────────────────
SCRATCH_BASE="${SCRATCH_BASE:-/work/hdd/bgre/${USER}}"
if [[ ! -d "${SCRATCH_BASE}" ]]; then
    SCRATCH_BASE="/tmp/${USER}"
fi
JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${SCRATCH_BASE}/hcc_sweep/${JOB_ID}"
mkdir -p "${WORK_DIR}"

# Point Python's tempfile at scratch so hcc_wrapper tempdirs land here.
export TMPDIR="${WORK_DIR}"

RESULTS_DIR="${CALIB_DIR}/results/${JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
    local rc=$?
    echo ""
    echo "── Cleanup (exit=${rc}) ──────────────────────────────────"
    if [[ -d "${WORK_DIR}" ]]; then
        du -sh "${WORK_DIR}" 2>/dev/null || true
        rm -rf "${WORK_DIR}"
        echo "Removed ${WORK_DIR}"
    fi
}
trap cleanup EXIT

# ── Delta modules ────────────────────────────────────────────────────────────
module purge
module load gcc-native/13.2
module load cuda/12.8
module load cmake/3.31.8
module load python/3.13.5-gcc13.3.1

# ── Build ────────────────────────────────────────────────────────────────────
echo "── Building hcc binary ───────────────────────────────────"
pushd "${SIM_DIR}" > /dev/null
./build.sh --cuda-arch 80 -j 4
popd > /dev/null
if [[ ! -x "${HCC_BIN}" ]]; then
    echo "ERROR: build completed but ${HCC_BIN} is not executable." >&2
    exit 1
fi
echo "  binary: ${HCC_BIN}"
export HCC_BINARY="${HCC_BIN}"

# ── Python virtualenv ────────────────────────────────────────────────────────
VENV_DIR="${CALIB_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "── Creating Python venv at ${VENV_DIR} ──────────────────"
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "${CALIB_DIR}/requirements.txt"
else
    source "${VENV_DIR}/bin/activate"
fi

echo "============================================================"
echo "HCC sweep job ${JOB_ID}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Scratch:  ${WORK_DIR}"
echo "  Results:  ${RESULTS_DIR}"
echo "============================================================"

# Run from repo root so Python module paths resolve.
cd "${REPO_ROOT}"

GRID="${GRID:-50}"
GRID_ARG=(--grid "${GRID}")

# ── Sequential sweep ─────────────────────────────────────────────────────────
# One combination at a time — no parallel processes, no GPU contention.
TASK_END="${TASK_END:-124}"
N_TASKS=$(( TASK_END + 1 ))
echo ""
echo "── Running tasks 0–${TASK_END} (${N_TASKS} combos × 5 seeds each) ──"

for task_id in $(seq 0 "${TASK_END}"); do
    echo ""
    echo "── Task ${task_id}/${TASK_END} ──────────────────────────────────"
    python -m HCC.calibration.sweep run \
        --task "${task_id}" \
        --results-dir "${RESULTS_DIR}" \
        --base-xml "${BASE_XML}" \
        "${GRID_ARG[@]}"
done

# ── Merge all per-task CSVs ──────────────────────────────────────────────────
echo ""
echo "── Merging results ───────────────────────────────────────"
python -m HCC.calibration.sweep merge --results-dir "${RESULTS_DIR}"

echo ""
echo "============================================================"
echo "Sweep complete. Results: ${RESULTS_DIR}/sweep_results.csv"
echo "============================================================"
