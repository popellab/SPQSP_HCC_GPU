#!/bin/bash
#SBATCH --job-name=pdac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=pdac_%j.out
#SBATCH --error=pdac_%j.err

# ============================================================================
# SPQSP PDAC — SLURM submission script for Anvil
#
# Source lives on /anvil/projects/, outputs go to /anvil/scratch/.
#
# Usage:
#   sbatch submit.sh                          # defaults: 500 steps, 50^3 grid
#   sbatch submit.sh -s 1000 -g 101           # custom run
#   SCRATCH_DIR=/custom/path sbatch submit.sh  # override scratch location
# ============================================================================

# --- Configuration ---
# Project directory (where the repo lives — auto-detected from this script's location)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDAC_BIN="${PROJECT_DIR}/build/bin/pdac"

# Scratch directory for outputs (override with SCRATCH_DIR env var)
SCRATCH_BASE="${SCRATCH_DIR:-/anvil/scratch/${USER}}"
RUN_DIR="${SCRATCH_BASE}/pdac_runs/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

# Default simulation arguments (overridden by anything passed after sbatch submit.sh)
DEFAULT_ARGS="-s 500 -g 50 -oa 1 -op 1"

# --- Load modules ---
module purge
module load cuda cmake gcc
# Uncomment if available on your cluster (avoids FetchContent rebuild):
# module load boost sundials

# --- Setup scratch output directory ---
mkdir -p "${RUN_DIR}"
echo "================================================"
echo "PDAC Run: ${SLURM_JOB_ID}"
echo "  Binary:     ${PDAC_BIN}"
echo "  Working dir: ${RUN_DIR}"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Args:       ${@:-${DEFAULT_ARGS}}"
echo "================================================"

# --- Run from scratch directory ---
# The executable writes to relative ./outputs/, so cd to scratch first
cd "${RUN_DIR}"

# Symlink the param file so -p flag isn't needed (already resolved via /proc/self/exe)
# But copy it here for reproducibility logging
cp "${PROJECT_DIR}/resource/param_all_test.xml" "${RUN_DIR}/param_snapshot.xml"

# Run simulation
${PDAC_BIN} ${@:-${DEFAULT_ARGS}}

echo ""
echo "================================================"
echo "Run complete. Outputs in: ${RUN_DIR}/outputs/"
echo "================================================"
