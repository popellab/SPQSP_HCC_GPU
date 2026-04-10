#!/bin/bash
#SBATCH --job-name=hcc_calib
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/calib_%j.out
#SBATCH --error=logs/calib_%j.err

# ============================================================================
# HCC calibration pipeline on Delta (A100 / gpuA100x4).
#
# Stages:
#   1. Build check (build if binary missing)
#   2. Ground-truth simulation  -> ground_truth.json
#   3. PyABC sampling           -> abc.db
#   4. Analysis + PPC           -> figures/*.png
#
# All per-simulation outputs are produced in job-local scratch and cleaned up
# continuously by the hcc_wrapper (tempdirs are removed after each run). At
# the end of the job, only the small artifacts (ground_truth.json, abc.db,
# figures/, logs) are copied back to the project tree; scratch is wiped.
#
# Usage:
#   sbatch --account=<alloc> HCC/calibration/submit_calibration.sh
#
#   # With overrides:
#   POP_SIZE=50 MAX_POPS=4 PPC_N=30 sbatch --account=<alloc> \
#       HCC/calibration/submit_calibration.sh
#
# Required: cluster.conf (for ACCOUNT) or --account= on the sbatch line.
# ============================================================================

set -euo pipefail

# ── Resolve project layout ──────────────────────────────────────────────────
# Script is HCC/calibration/submit_calibration.sh -> repo root = parents[2]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_DIR="${REPO_ROOT}/HCC/sim"
HCC_BIN="${SIM_DIR}/build/bin/hcc"
BASE_XML="${SIM_DIR}/resource/param_all_test.xml"

mkdir -p "${CALIB_DIR}/logs"

# ── Load cluster.conf for ACCOUNT (if --account= not on sbatch line) ────────
ACCOUNT="${SLURM_JOB_ACCOUNT:-}"
if [[ -z "${ACCOUNT}" && -f "${SIM_DIR}/cluster.conf" ]]; then
    source "${SIM_DIR}/cluster.conf"
fi
if [[ -z "${ACCOUNT:-}" ]]; then
    echo "ERROR: ACCOUNT not set. Put one in HCC/sim/cluster.conf or pass --account= to sbatch." >&2
    exit 1
fi

# ── Job-local scratch layout (everything intermediate lives here) ───────────
# On Delta, /work/hdd/<group>/<user> is the per-user project scratch.
ACCOUNT_GROUP="${ACCOUNT%%-*}"
SCRATCH_BASE="${SCRATCH_BASE:-/work/hdd/${ACCOUNT_GROUP}/${USER}}"
if [[ ! -d "${SCRATCH_BASE}" ]]; then
    SCRATCH_BASE="/tmp/${USER}"
fi
JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${SCRATCH_BASE}/hcc_calibration/${JOB_ID}"
TMPDIR_LOCAL="${WORK_DIR}/tmp"
mkdir -p "${WORK_DIR}" "${TMPDIR_LOCAL}"

# Redirect all Python tempfile.mkdtemp() calls (used by hcc_wrapper) into
# job-local scratch so an entire wipe cleans up any strays.
export TMPDIR="${TMPDIR_LOCAL}"

# Artifact destinations
GT_JSON="${WORK_DIR}/ground_truth.json"
ABC_DB_PATH="${WORK_DIR}/abc.db"
ABC_DB_URL="sqlite:///${ABC_DB_PATH}"
FIG_DIR="${WORK_DIR}/figures"

# Final copy target (small artifacts only)
RESULTS_DIR="${CALIB_DIR}/results/${JOB_ID}"

# ── Cleanup on exit: wipe job-local scratch unconditionally ─────────────────
cleanup() {
    local rc=$?
    echo ""
    echo "── Cleanup (exit=${rc}) ───────────────────────────────────"
    if [[ -d "${WORK_DIR}" ]]; then
        du -sh "${WORK_DIR}" 2>/dev/null || true
        rm -rf "${WORK_DIR}"
        echo "Removed ${WORK_DIR}"
    fi
}
trap cleanup EXIT

# ── Delta modules ───────────────────────────────────────────────────────────
module purge
module load gcc/11.4.0
module load cuda/12.3
module load cmake/3.27
# Prefer a Python module if available; otherwise rely on system python3.
module load anaconda3_gpu 2>/dev/null || module load anaconda3 2>/dev/null || true

# ── Job banner ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "HCC calibration job ${JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  nvcc:        $(nvcc --version | grep release || echo 'not found')"
echo "  Repo:        ${REPO_ROOT}"
echo "  Scratch:     ${WORK_DIR}"
echo "  Results:     ${RESULTS_DIR}"
echo "============================================================"

# ── Stage 0: build check ────────────────────────────────────────────────────
if [[ ! -x "${HCC_BIN}" ]]; then
    echo ""
    echo "── Stage 0: building hcc binary ──────────────────────────"
    pushd "${SIM_DIR}" > /dev/null
    ./build.sh --cuda-arch 80 -j 4
    popd > /dev/null
fi
if [[ ! -x "${HCC_BIN}" ]]; then
    echo "ERROR: build completed but ${HCC_BIN} is not executable." >&2
    exit 1
fi
echo "  binary: ${HCC_BIN}"
export HCC_BINARY="${HCC_BIN}"

# ── Python virtualenv for numpy/lz4/pyabc/matplotlib ────────────────────────
VENV_DIR="${CALIB_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo ""
    echo "── Creating Python venv at ${VENV_DIR} ───────────────────"
    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "${CALIB_DIR}/requirements.txt"
else
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
echo "  python:  $(python --version)"
echo "  pyabc:   $(python -c 'import pyabc; print(pyabc.__version__)' 2>/dev/null || echo 'MISSING')"

# Runs are invoked from the repo root so module paths resolve.
cd "${REPO_ROOT}"

# ── Pipeline parameters (override via env) ──────────────────────────────────
SEED="${SEED:-12345}"
GRID="${GRID:-}"
POP_SIZE="${POP_SIZE:-100}"
MAX_POPS="${MAX_POPS:-5}"
MIN_EPSILON="${MIN_EPSILON:-0}"
PPC_N="${PPC_N:-30}"

GRID_ARG=()
if [[ -n "${GRID}" ]]; then
    GRID_ARG=(--grid "${GRID}")
fi

echo ""
echo "── Pipeline parameters ───────────────────────────────────"
echo "  SEED=${SEED} GRID=${GRID:-'(XML default)'}"
echo "  POP_SIZE=${POP_SIZE}  MAX_POPS=${MAX_POPS}  MIN_EPSILON=${MIN_EPSILON}"
echo "  PPC_N=${PPC_N}"

# ── Stage 1: ground truth ───────────────────────────────────────────────────
echo ""
echo "── Stage 1: ground-truth simulation ──────────────────────"
t0=$SECONDS
python -m HCC.calibration.run_pyabc ground-truth \
    --base-xml "${BASE_XML}" \
    --out "${GT_JSON}" \
    --seed "${SEED}" \
    "${GRID_ARG[@]}"
echo "  ground-truth done in $((SECONDS - t0))s"
echo "  tempfile residue:"
du -sh "${TMPDIR_LOCAL}" 2>/dev/null || true
# Clear any lingering per-sim tempdirs (wrapper cleans its own, belt-and-braces)
find "${TMPDIR_LOCAL}" -mindepth 1 -maxdepth 1 -type d -name 'hcc_calib_*' -exec rm -rf {} + 2>/dev/null || true

# ── Stage 2: PyABC sampling ─────────────────────────────────────────────────
echo ""
echo "── Stage 2: PyABC sampling ───────────────────────────────"
t0=$SECONDS
python -m HCC.calibration.run_pyabc abc \
    --target "${GT_JSON}" \
    --db "${ABC_DB_URL}" \
    --population-size "${POP_SIZE}" \
    --max-populations "${MAX_POPS}" \
    --min-epsilon "${MIN_EPSILON}"
echo "  ABC done in $((SECONDS - t0))s"
echo "  db size: $(du -h "${ABC_DB_PATH}" | cut -f1)"
find "${TMPDIR_LOCAL}" -mindepth 1 -maxdepth 1 -type d -name 'hcc_calib_*' -exec rm -rf {} + 2>/dev/null || true

# ── Stage 3: analysis + PPC ─────────────────────────────────────────────────
echo ""
echo "── Stage 3: analysis + posterior predictive check ────────"
t0=$SECONDS
python -m HCC.calibration.analyze \
    --db "${ABC_DB_URL}" \
    --target "${GT_JSON}" \
    --out "${FIG_DIR}" \
    --ppc "${PPC_N}"
echo "  analysis done in $((SECONDS - t0))s"
find "${TMPDIR_LOCAL}" -mindepth 1 -maxdepth 1 -type d -name 'hcc_calib_*' -exec rm -rf {} + 2>/dev/null || true

# ── Copy artifacts back to the repo tree ────────────────────────────────────
echo ""
echo "── Copying artifacts to ${RESULTS_DIR} ───────────────────"
mkdir -p "${RESULTS_DIR}"
cp "${GT_JSON}" "${RESULTS_DIR}/"
cp "${ABC_DB_PATH}" "${RESULTS_DIR}/"
cp -r "${FIG_DIR}" "${RESULTS_DIR}/figures"
# Snapshot the base XML so results are self-describing
cp "${BASE_XML}" "${RESULTS_DIR}/param_snapshot.xml"

echo ""
echo "============================================================"
echo "Pipeline complete. Results: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}" "${RESULTS_DIR}/figures"
echo "============================================================"
