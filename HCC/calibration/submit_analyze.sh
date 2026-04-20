#!/bin/bash
#SBATCH --job-name=hcc_analyze
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchris80@jh.edu

# ============================================================================
# Run analyze.py on an existing abc.db + ground_truth.json and move all
# files from a results/tmp directory back to calibration/results/<JOB_DIR>.
#
# Usage:
#   # Analyze the most-recent results dir automatically:
#   sbatch --account=<alloc> HCC/calibration/submit_analyze.sh
#
#   # Point at a specific results dir:
#   RESULTS_SRC=/path/to/results/dir \
#       sbatch --account=<alloc> HCC/calibration/submit_analyze.sh
#
#   # Override number of PPC samples:
#   PPC_N=50 sbatch --account=<alloc> HCC/calibration/submit_analyze.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CALIB_DIR="${REPO_ROOT}/HCC/calibration"
SIM_DIR="${REPO_ROOT}/HCC/sim"
HCC_BIN="${SIM_DIR}/build/bin/hcc"
BASE_XML="${SIM_DIR}/resource/param_all_test.xml"

mkdir -p "${CALIB_DIR}/logs"

JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

# ── Locate source results dir (abc.db + ground_truth.json) ──────────────────
# Default: most-recently modified directory under calibration/results/
if [[ -n "${RESULTS_SRC:-}" ]]; then
    SRC_DIR="${RESULTS_SRC}"
else
    SRC_DIR="$(ls -td "${CALIB_DIR}/results/"*/ 2>/dev/null | head -1)"
fi

if [[ -z "${SRC_DIR}" || ! -d "${SRC_DIR}" ]]; then
    echo "ERROR: No results directory found. Set RESULTS_SRC= or run submit_calibration.sh first." >&2
    exit 1
fi

ABC_DB_PATH="${SRC_DIR}/abc.db"
GT_JSON="${SRC_DIR}/ground_truth.json"

if [[ ! -f "${ABC_DB_PATH}" ]]; then
    echo "ERROR: abc.db not found in ${SRC_DIR}" >&2
    exit 1
fi
if [[ ! -f "${GT_JSON}" ]]; then
    echo "ERROR: ground_truth.json not found in ${SRC_DIR}" >&2
    exit 1
fi

ABC_DB_URL="sqlite:///${ABC_DB_PATH}"
FIG_DIR="${SRC_DIR}/figures"
PPC_N="${PPC_N:-30}"

# ── Delta modules ───────────────────────────────────────────────────────────
module purge
module load gcc-native/13.2
module load cuda/12.8
module load cmake/3.31.8
module load python/3.13.5-gcc13.3.1

# ── Python virtualenv ────────────────────────────────────────────────────────
VENV_DIR="${CALIB_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "── Creating Python venv at ${VENV_DIR} ───────────────────"
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "${CALIB_DIR}/requirements.txt"
else
    source "${VENV_DIR}/bin/activate"
fi

export HCC_BINARY="${HCC_BIN}"

echo "============================================================"
echo "HCC analyze job ${JOB_ID}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Source:   ${SRC_DIR}"
echo "  DB:       ${ABC_DB_PATH}  ($(du -h "${ABC_DB_PATH}" | cut -f1))"
echo "  GT:       ${GT_JSON}"
echo "  Figures:  ${FIG_DIR}"
echo "  PPC_N:    ${PPC_N}"
echo "============================================================"

# ── Build (keep binary in sync with source) ─────────────────────────────────
echo ""
echo "── Building hcc binary ───────────────────────────────────"
cd "${SIM_DIR}"
./build.sh --cuda-arch 80 -j 4
if [[ ! -x "${HCC_BIN}" ]]; then
    echo "ERROR: build completed but ${HCC_BIN} is not executable." >&2
    exit 1
fi
echo "  binary: ${HCC_BIN}"

# Runs are invoked from the repo root so module paths resolve.
cd "${REPO_ROOT}"

# ── Analysis + PPC ──────────────────────────────────────────────────────────
echo ""
echo "── Running analyze.py ────────────────────────────────────"
t0=$SECONDS
python -m HCC.calibration.analyze \
    --db "${ABC_DB_URL}" \
    --target "${GT_JSON}" \
    --out "${FIG_DIR}" \
    --ppc "${PPC_N}"
echo "  done in $((SECONDS - t0))s"

echo ""
echo "============================================================"
echo "Analysis complete."
ls -lh "${FIG_DIR}" 2>/dev/null || true
echo "Results remain in: ${SRC_DIR}"
echo "============================================================"
