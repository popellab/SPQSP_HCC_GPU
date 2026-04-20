#!/bin/bash
#SBATCH --job-name=hcc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=hcc_%j.out
#SBATCH --error=hcc_%j.err
#SBATCH --account=bgre-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1

# ============================================================================
# SPQSP HCC — Cluster-agnostic SLURM submission script
#
# Auto-detects cluster (Delta, Anvil, etc.) and configures modules, partitions,
# scratch paths, and CUDA architecture accordingly.
#
# First-time setup:
#   cp cluster.conf.example cluster.conf   # edit ACCOUNT
#   ./setup_deps.sh                        # fetch deps (needs internet)
#   sbatch submit.sh                       # build + run
#
# Usage:
#   sbatch submit.sh                       # defaults: 500 steps, 50^3 grid
#   sbatch submit.sh -s 100 -g 30         # custom run
#
# Profiling (Nsight Systems):
#   sbatch --export=ALL,NSYS=1 submit.sh -s 280 -g 50 -G 3
#     - forces build with FLAMEGPU2 NVTX markers if needed
#     - wraps hcc in `nsys profile` and writes .nsys-rep to
#       ${PROJECT_DIR}/profiling/${SLURM_JOB_ID}/
# ============================================================================

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HCC_BIN="${PROJECT_DIR}/build/bin/hcc"
EXT_DIR="${PROJECT_DIR}/external"
DEFAULT_ARGS="-s 500 -g 50 -oa 1 -op 1"

# Profiling: set NSYS=1 to wrap hcc in Nsight Systems capture
NSYS="${NSYS:-0}"

# ============================================================================
# Load user config
# ============================================================================

ACCOUNT=""
CLUSTER=""
PARTITION=""
CUDA_ARCH=""
SCRATCH_BASE=""
EXTRA_MODULES=""

if [[ -f "${PROJECT_DIR}/cluster.conf" ]]; then
    source "${PROJECT_DIR}/cluster.conf"
else
    echo "ERROR: cluster.conf not found."
    echo "  cp cluster.conf.example cluster.conf"
    echo "  Then edit ACCOUNT and re-submit."
    exit 1
fi

if [[ -z "${ACCOUNT}" ]]; then
    echo "ERROR: ACCOUNT not set in cluster.conf"
    exit 1
fi

# ============================================================================
# Auto-detect cluster from hostname
# ============================================================================

detect_cluster() {
    local host
    host=$(hostname -f 2>/dev/null || hostname)
    case "${host}" in
        *delta*|*dt-login*|*dt-gpu*|*ncsa*)  echo "delta" ;;
        *anvil*|*bell*)                       echo "anvil" ;;
        *)                                    echo "unknown" ;;
    esac
}

CLUSTER="${CLUSTER:-$(detect_cluster)}"

# ============================================================================
# Cluster-specific configuration
# ============================================================================

setup_delta() {
    # Delta @ NCSA — Cray PE with gcc-toolset, cuda via cudatoolkit module
    # Default env already has gcc 13.x and CUDA 12.8 via PrgEnv-gnu + cudatoolkit
    module load cmake/3.31.8 2>/dev/null || module load cmake 2>/dev/null || true

    PARTITION="${PARTITION:-gpuA100x4}"
    # Delta work dirs are under /work/hdd/<account>/<user>
    # Fall back to /tmp if the account dir doesn't exist
    local acct_dir="/work/hdd/${ACCOUNT%%-*}/${USER}"
    if [[ -d "${acct_dir}" ]]; then
        SCRATCH_BASE="${SCRATCH_BASE:-${acct_dir}}"
    else
        SCRATCH_BASE="${SCRATCH_BASE:-/tmp/${USER}}"
        echo "WARNING: ${acct_dir} not found, using ${SCRATCH_BASE}"
    fi

    # Pick CUDA arch from partition if not overridden
    if [[ -z "${CUDA_ARCH}" ]]; then
        case "${PARTITION}" in
            *A100*)  CUDA_ARCH="80" ;;
            *A40*)   CUDA_ARCH="86" ;;
            *H200*)  CUDA_ARCH="90" ;;
            *)       CUDA_ARCH="80" ;;
        esac
    fi
}

setup_anvil() {
    # Anvil @ Purdue
    module purge
    module load modtree/gpu
    module load gcc/11.2.0 cuda/12.8.0

    # Anvil system cmake may be too old; use local if available
    LOCAL_CMAKE="${EXT_DIR}/cmake-3.28.3-linux-x86_64/bin"
    if [[ -x "${LOCAL_CMAKE}/cmake" ]]; then
        export PATH="${LOCAL_CMAKE}:${PATH}"
    fi

    PARTITION="${PARTITION:-gpu}"
    SCRATCH_BASE="${SCRATCH_BASE:-/anvil/scratch/${USER}}"
    CUDA_ARCH="${CUDA_ARCH:-80}"
}

setup_unknown() {
    # Generic fallback — assume modules are already loaded by user
    echo "WARNING: Unrecognized cluster '$(hostname)'. Using environment as-is."
    echo "  Set CLUSTER in cluster.conf if auto-detection is wrong."
    PARTITION="${PARTITION:-gpu}"
    SCRATCH_BASE="${SCRATCH_BASE:-/tmp/${USER}}"
    CUDA_ARCH="${CUDA_ARCH:-80}"
}

case "${CLUSTER}" in
    delta)   setup_delta   ;;
    anvil)   setup_anvil   ;;
    *)       setup_unknown ;;
esac

# Load any extra user modules
if [[ -n "${EXTRA_MODULES}" ]]; then
    for mod in ${EXTRA_MODULES}; do
        module load "${mod}" 2>/dev/null || echo "WARNING: could not load module ${mod}"
    done
fi

# Redirect nvcc temp files to scratch (shared /tmp may be too small for builds)
export TMPDIR="${SCRATCH_BASE}/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"

# ============================================================================
# Job info
# ============================================================================

RUN_DIR="${SCRATCH_BASE}/hcc_runs/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "================================================"
echo "HCC Job: ${SLURM_JOB_ID:-interactive}"
echo "  Cluster:    ${CLUSTER}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A (login node)')"
echo "  nvcc:       $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "  cmake:      $(cmake --version 2>/dev/null | head -1 || echo 'not found')"
echo "  gcc:        $(gcc --version 2>/dev/null | head -1 || echo 'not found')"
echo "  Account:    ${ACCOUNT}"
echo "  Partition:  ${PARTITION}"
echo "  CUDA arch:  ${CUDA_ARCH}"
echo "  Scratch:    ${SCRATCH_BASE}"
echo "================================================"

# ============================================================================
# Build if needed
# ============================================================================

# If NSYS=1, also rebuild if the existing binary was not built with NVTX
NEED_BUILD=0
if [[ ! -x "${HCC_BIN}" ]]; then
    NEED_BUILD=1
elif [[ "${NSYS}" == "1" ]] && [[ -f "${PROJECT_DIR}/build/CMakeCache.txt" ]]; then
    if ! grep -q "FLAMEGPU_ENABLE_NVTX:BOOL=ON" "${PROJECT_DIR}/build/CMakeCache.txt"; then
        echo ""
        echo "=== NSYS=1 but existing binary lacks FLAMEGPU2 NVTX — rebuilding with --nvtx ==="
        NEED_BUILD=1
    fi
fi

if [[ "${NEED_BUILD}" == "1" ]]; then
    echo ""
    echo "=== Building hcc ==="
    cd "${PROJECT_DIR}"

    BUILD_ARGS=(--cuda-arch "${CUDA_ARCH}")
    [[ "${NSYS}" == "1" ]] && BUILD_ARGS+=(--nvtx)

    # Use pre-staged deps from external/ if available
    if [[ -f "${EXT_DIR}/flamegpu2/CMakeLists.txt" ]]; then
        BUILD_ARGS+=(--flamegpu "${EXT_DIR}/flamegpu2")
        echo "  FLAMEGPU: using ${EXT_DIR}/flamegpu2"
    else
        echo "  FLAMEGPU: will fetch from GitHub"
    fi

    if [[ -f "${EXT_DIR}/sundials/CMakeLists.txt" ]]; then
        export SUNDIALS_DIR="${EXT_DIR}/sundials"
        echo "  SUNDIALS:  using ${EXT_DIR}/sundials"
    else
        echo "  SUNDIALS:  will fetch from GitHub"
    fi

    if [[ -f "${EXT_DIR}/boost/CMakeLists.txt" ]]; then
        export BOOST_ROOT="${EXT_DIR}/boost"
        echo "  Boost:     using ${EXT_DIR}/boost"
    else
        echo "  Boost:     will fetch from GitHub"
    fi

    echo ""
    ./build.sh "${BUILD_ARGS[@]}"
    echo ""
fi

# ============================================================================
# Run simulation
# ============================================================================

mkdir -p "${RUN_DIR}"

# If profiling, set up persistent profile dir (outside scratch so it survives
# cluster cleanup) and locate nsys.
PROFILE_DIR=""
if [[ "${NSYS}" == "1" ]]; then
    PROFILE_DIR="${PROJECT_DIR}/profiling/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${PROFILE_DIR}"
    if ! command -v nsys >/dev/null 2>&1; then
        echo "ERROR: NSYS=1 but 'nsys' not on PATH. Load a cudatoolkit module providing Nsight Systems."
        exit 1
    fi
fi

echo "================================================"
echo "  Binary:      ${HCC_BIN}"
echo "  Working dir: ${RUN_DIR}"
echo "  Args:        ${@:-${DEFAULT_ARGS}}"
if [[ "${NSYS}" == "1" ]]; then
    echo "  Profile:     enabled (nsys) → ${PROFILE_DIR}"
    echo "  nsys:        $(nsys --version 2>/dev/null | head -1)"
fi
echo "================================================"

cd "${RUN_DIR}"

# Copy param file for reproducibility
cp "${PROJECT_DIR}/resource/param_all_test.xml" "${RUN_DIR}/param_snapshot.xml"

# Run (optionally wrapped in nsys profile)
if [[ "${NSYS}" == "1" ]]; then
    nsys profile \
        --output="${PROFILE_DIR}/hcc_prof" \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --stats=false \
        "${HCC_BIN}" ${@:-${DEFAULT_ARGS}}
    # Also snapshot the args used for the profile, so later analysis has context
    echo "args: ${@:-${DEFAULT_ARGS}}" > "${PROFILE_DIR}/run.meta"
    echo "host: $(hostname)" >> "${PROFILE_DIR}/run.meta"
    echo "gpu:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" >> "${PROFILE_DIR}/run.meta"
    cp "${PROJECT_DIR}/resource/param_all_test.xml" "${PROFILE_DIR}/param_snapshot.xml"
else
    "${HCC_BIN}" ${@:-${DEFAULT_ARGS}}
fi

# ============================================================================
# Copy outputs to home directory for persistent storage
# ============================================================================

HOME_OUT="${PROJECT_DIR}/outputs/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${HOME_OUT}"
echo ""
echo "Copying outputs to ${HOME_OUT}/ ..."
cp -r "${RUN_DIR}/outputs/"* "${HOME_OUT}/" 2>/dev/null || true
cp "${RUN_DIR}/param_snapshot.xml" "${HOME_OUT}/" 2>/dev/null || true

echo ""
echo "================================================"
echo "Run complete."
echo "  Scratch: ${RUN_DIR}/outputs/"
echo "  Home:    ${HOME_OUT}/"
[[ -n "${PROFILE_DIR}" ]] && echo "  Profile: ${PROFILE_DIR}/hcc_prof.nsys-rep"
echo "================================================"
