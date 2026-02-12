# GPU Profiling Quick Start Guide

## Quick Commands

### Run Default Profile (200 steps, ~10 minutes)
```bash
./profile_pdac.sh
```

### Run Quick Profile (10 steps, ~2 minutes)
```bash
./profile_pdac.sh --quick
```

### Custom Profile
```bash
./profile_pdac.sh --steps 50 --output my_results
```

### View Latest Results
```bash
# Find most recent profiling session
cd profiling_results
ls -lt | head

# View summary (if generated)
cat PROFILING_SUMMARY.md

# View GPU utilization
column -t -s, gpu_util_*.csv | less

# View NVTX stats
cat nsys_stats_*.txt | grep -A 10 "NVTX Range"
```

### Manual Profiling (if script fails)

#### 1. Quick NVTX Profile
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite=true \
  --output=my_profile \
  ./build/bin/pdac -s 20 -oa 0 -op 0
```

#### 2. GPU Utilization Monitoring
```bash
# In one terminal, start monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 1 > gpu_monitor.csv

# In another terminal, run simulation
./build/bin/pdac -s 20 -oa 0 -op 0

# Stop first terminal with Ctrl+C
```

#### 3. Extract Stats from Profile
```bash
nsys stats --report nvtx_sum,osrt_sum my_profile.nsys-rep
```

## What You'll Get on WSL2

✅ **Available**:
- High-level GPU operation timing (NVTX markers)
- GPU utilization percentage over time
- GPU memory usage
- Temperature and power monitoring
- CPU-GPU execution correlation
- System-level performance metrics

❌ **Not Available** (requires native Linux/Windows):
- Individual CUDA kernel execution times
- CUDA API call durations (cudaMalloc, cudaMemcpy, etc.)
- Memory transfer bandwidth analysis
- Kernel launch configuration details
- SM occupancy metrics

## Interpreting Results

### GPU Utilization
- **< 30%**: Small workload or CPU bottleneck
- **30-70%**: Balanced CPU-GPU workload
- **> 70%**: GPU-bound (good for large simulations)

### NVTX Operations
- **DeviceScan::ExclusiveSum**: Should be 90-99% of GPU time (agent processing)
- **DeviceRadixSort**: Should be < 5% (spatial sorting)
- **DeviceReduce**: Should be < 1% (statistics)

### Memory
- **Stable over time**: No memory leaks ✅
- **Growing continuously**: Memory leak ❌
- **Sudden spikes**: Agent population explosions

### Temperature
- **< 60°C**: Excellent
- **60-80°C**: Good
- **> 80°C**: May throttle, check cooling

## Recommended Grid Sizes for Testing

| Grid Size | Voxels | Approx Agents | GPU Util | Time/Step |
|-----------|--------|---------------|----------|-----------|
| 11³ | 1,331 | 100-500 | < 5% | < 100 ms |
| 21³ | 9,261 | 500-5K | 5-15% | ~200 ms |
| 50³ | 125,000 | 5K-100K | 15-40% | ~500 ms |
| 75³ | 421,875 | 50K-500K | 40-70% | ~1-2 s |
| 101³ | 1,030,301 | 100K-1M | 70-95% | ~2-5 s |

## Common Issues

**"SKIPPED: does not contain CUDA trace data"**
- Normal on WSL2 - focus on NVTX markers and GPU utilization

**GPU shows 0% utilization**
- Run `nvidia-smi` to verify GPU is visible
- Check simulation is actually running

**Profile file too large**
- Use shorter runs: `--steps 10`
- Disable output: `-oa 0 -op 0`

**Script hangs at Method 2**
- Background nvidia-smi might fail - press Ctrl+C and re-run

## Adding Your Own Profiling Markers

To add NVTX markers to your code:

```cpp
#include <nvtx3/nvToolsExt.h>

void myFunction() {
    nvtxRangePush("MyFunction");

    // Your code here
    myKernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();

    nvtxRangePop();
}
```

Then rebuild and the markers will show up in the profile!

## For Detailed Kernel Profiling

If you need kernel-level details, use:

### Native Linux: Nsight Compute
```bash
ncu --set full --target-processes all ./build/bin/pdac -s 1
```

### Native Linux: Nsight Systems (full trace)
```bash
nsys profile \
  --trace=cuda,nvtx,cudnn,cublas \
  --cuda-memory-usage=true \
  --cuda-graph-trace=node \
  --stats=true \
  ./build/bin/pdac
```

## Next Steps

1. ✅ Run quick profile to verify setup
2. ✅ Check `profiling_results/PROFILING_SUMMARY.md`
3. ⏭️ Run longer profile (200 steps) for statistics
4. ⏭️ Test with different grid sizes (21³, 50³, 75³)
5. ⏭️ Add custom NVTX markers to PDE solver
6. ⏭️ Profile on native Linux for kernel details

## Documentation

- Full guide: `PROFILING.md`
- Summary of last run: `profiling_results/PROFILING_SUMMARY.md`
- CLAUDE.md: Architecture and performance notes
