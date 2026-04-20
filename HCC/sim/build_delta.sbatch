#!/bin/bash
#SBATCH --job-name=hcc_build
#SBATCH --partition=gpuA100x4
#SBATCH --account=YOUR_ALLOCATION_HERE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/hdd/${USER}/SPQSP_HCC-main/HCC/sim
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

module purge
module load gcc/11.4.0
module load cuda/12.3
module load cmake/3.27

mkdir -p logs

echo "=== Build started: $(date) ==="
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "CUDA: $(nvcc --version | grep release)"

# First-time build fetches FLAMEGPU2, Boost, SUNDIALS from GitHub (~8-15 min)
# Subsequent incremental builds are ~2 min
# Pass NSYS=1 to build with FLAMEGPU2 NVTX markers for Nsight Systems profiling.
BUILD_ARGS=(--cuda-arch 80 -j 4)
[[ "${NSYS:-0}" == "1" ]] && BUILD_ARGS+=(--nvtx)
./build.sh "${BUILD_ARGS[@]}"

echo "=== Build finished: $(date) ==="
