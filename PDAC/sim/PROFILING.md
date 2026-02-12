# GPU Profiling Guide for PDAC Simulation

## Overview

This guide explains how to profile the PDAC simulation's GPU performance and the limitations when running on WSL2 (Windows Subsystem for Linux).

## WSL2 Limitations 🚨

**IMPORTANT**: WSL2 does **NOT** support full CUDA API tracing because:
- The CUDA driver runs in Windows, not Linux
- CUDA calls go through a compatibility layer that blocks profiler interception
- Tools like `nsys` cannot inject hooks into the CUDA runtime

### What DOESN'T Work on WSL2:
❌ Detailed kernel execution timings
❌ CUDA API call traces (cudaMalloc, cudaMemcpy, etc.)
❌ Memory transfer bandwidth analysis
❌ Kernel launch parameter inspection
❌ SM occupancy analysis

### What DOES Work on WSL2:
✅ NVTX range markers (high-level GPU operation tracking)
✅ GPU utilization monitoring via nvidia-smi
✅ CPU profiling and system call tracing
✅ Memory usage tracking
✅ Execution time measurements

## Recommended Profiling Workflow

### Quick Profile (10 steps, ~2 minutes)
```bash
./profile_pdac.sh --quick
```

### Default Profile (200 steps, default simulation)
```bash
./profile_pdac.sh
```

### Custom Profile
```bash
./profile_pdac.sh --steps 50 --output my_profile_results
```

## What Each Method Provides

### 1. Nsight Systems (NVTX Markers)
**File**: `nsys_<timestamp>.nsys-rep`

Shows high-level GPU operations through NVTX markers inserted by CUDA libraries:
- **CUB operations**: DeviceScan, DeviceSort, DeviceReduce
- **Timeline view**: When GPU operations occurred
- **CPU correlation**: Which CPU code triggered GPU work

**Limitations**: Only sees NVTX-instrumented code, not individual kernels.

**View it**:
- Text summary: `cat profiling_results/nsys_stats_<timestamp>.txt`
- GUI: Transfer .nsys-rep to Windows and open with Nsight Systems UI

### 2. GPU Utilization Monitoring
**File**: `gpu_util_<timestamp>.csv`

Real-time GPU metrics captured every second:
- GPU compute utilization (%)
- GPU memory utilization (%)
- Memory used/free (MB)
- Temperature (°C)
- Power draw (W)

**View it**:
```bash
column -t -s, profiling_results/gpu_util_<timestamp>.csv | less
```

**Plot it** (requires gnuplot):
```bash
gnuplot -e "set datafile separator ','; set term dumb; plot 'profiling_results/gpu_util_<timestamp>.csv' using 2 with lines title 'GPU %'"
```

### 3. System Time Profiling
**File**: `time_profile_<timestamp>.txt`

Detailed system-level metrics:
- Wall clock time
- User/system CPU time
- Maximum resident set size (peak memory)
- Page faults
- Context switches
- File I/O operations

### 4. Summary Report
**File**: `profile_summary_<timestamp>.txt`

Human-readable summary combining all metrics:
- Average and peak GPU utilization
- Top GPU operations from NVTX
- Quick performance insights

## Interpreting Results

### Good Performance Indicators:
✅ GPU utilization > 70% during simulation steps
✅ Memory utilization stable (not constantly growing)
✅ CUB::DeviceScan dominant (indicates parallel agent processing)
✅ Low context switch count (< 10,000)

### Performance Red Flags:
🔴 GPU utilization < 30% (CPU bottleneck or small workload)
🔴 Memory utilization increasing over time (memory leak)
🔴 High page fault count (memory pressure)
🔴 Temperature > 85°C (thermal throttling risk)

### Typical Operation Breakdown:
- **DeviceScan::ExclusiveSum** (90-99%): Prefix scans for agent processing
- **DeviceRadixSort** (0-5%): Sorting agents by spatial hash
- **DeviceReduce** (<1%): Population counts, statistics

## Advanced: Adding Custom NVTX Markers

To add your own profiling markers to the code:

```cpp
#include <nvtx3/nvToolsExt.h>

// Mark a region
nvtxRangePush("My Custom Kernel");
myKernel<<<blocks, threads>>>();
cudaDeviceSynchronize();
nvtxRangePop();
```

Then rebuild and the markers will appear in the Nsight Systems timeline.

## For Full Kernel Profiling

If you need detailed kernel-level profiling, you have two options:

### Option 1: Native Linux System
Run on a native Linux machine (not WSL2) with CUDA-capable GPU:
```bash
nsys profile --trace=cuda,nvtx,cudnn,cublas --cuda-memory-usage=true \
    --cuda-graph-trace=node --stats=true ./build/bin/pdac
```

This will capture:
- Individual kernel names and execution times
- Memory transfer sizes and bandwidths
- CUDA API call durations
- Kernel launch parameters
- SM efficiency and occupancy

### Option 2: Windows Nsight Systems
1. Build the PDAC code on Windows with CUDA
2. Run Nsight Systems on Windows
3. Captures full CUDA trace data natively

### Option 3: Nsight Compute for Kernel Analysis
For deep dive into specific kernels (requires native Linux or Windows):
```bash
ncu --set full --target-processes all ./build/bin/pdac
```

## Understanding Current Performance

Based on current NVTX profiles, the simulation is:
1. **GPU-bound** during agent processing (DeviceScan operations)
2. **Using parallel algorithms** effectively (CUB library)
3. **Minimal sorting overhead** (<1% in DeviceRadixSort)

Key bottlenecks to investigate:
- PDE solver CG iterations (not currently NVTX-instrumented)
- Agent state updates (not currently NVTX-instrumented)
- CPU-GPU synchronization points

## Adding More Instrumentation

To get better profiling data on WSL2, add NVTX markers around:
1. PDE solver initialization
2. Each CG iteration
3. Source collection from agents
4. QSP solver steps
5. Agent function dispatch

This will provide a more detailed breakdown even without full kernel tracing.

## Common Issues

**"SKIPPED: does not contain CUDA trace data"**
- Normal on WSL2 - use NVTX markers and GPU utilization instead

**"No NVTX ranges found"**
- CUB library may not have NVTX enabled - add custom markers

**GPU utilization shows 0%**
- Check nvidia-smi works: `nvidia-smi`
- Verify GPU is accessible: `./build/bin/pdac -s 1`

**Profile file is huge (>500 MB)**
- Use shorter runs for profiling: `--steps 20`
- Disable agent/PDE output: `-oa 0 -op 0`

## Contact & Support

For profiling questions specific to PDAC:
- Check CLAUDE.md for architecture details
- Review MEMORY.md for known performance characteristics
- Profile with `--quick` first to verify setup

For Nsight Systems issues:
- NVIDIA Nsight Systems documentation: https://docs.nvidia.com/nsight-systems/
- WSL2 CUDA limitations: https://docs.nvidia.com/cuda/wsl-user-guide/
