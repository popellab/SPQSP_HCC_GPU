# PDAC Simulation - Performance Optimization Analysis

**Date**: February 12, 2026
**Configuration**: 200 ABM steps, 50³ grid, ~600→125K agents
**Total Runtime**: 159.7 seconds (~0.8s/step)
**Platform**: WSL2, CUDA 12.6

---

## Executive Summary

The simulation is currently **CPU-bound with significant synchronization overhead**. GPU utilization is low (14.8% average) with occasional bursts to 96%, indicating frequent CPU-GPU synchronization points. The profiling reveals three major bottlenecks:

1. **Excessive polling overhead** (93% of OS time)
2. **Massive ioctl call count** (3M+ calls, likely GPU status checks)
3. **Low GPU occupancy** (GPU work is only 0.46% of wall clock time)

**Estimated Speedup Potential**: 5-10× with optimizations

---

## Detailed Performance Breakdown

### 1. GPU Utilization Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average GPU Utilization** | 14.8% | LOW - mostly idle |
| **Peak GPU Utilization** | 96% | Brief bursts of activity |
| **GPU Memory Usage** | 2.9 GB stable | No leaks ✅ |
| **GPU Temperature** | 55-59°C | Excellent ✅ |

**GPU Time Budget**:
- Total wall clock time: 159.7s
- Total GPU work (NVTX): 0.728s (0.46% of runtime!)
- **Missing time**: 159.0s (99.54%) is CPU, synchronization, or uninstrumented GPU work

**Key Finding**: GPU is vastly underutilized. Either:
- CPU is the bottleneck (likely), or
- Most GPU work is not instrumented with NVTX markers (PDE solver, agent kernels)

### 2. NVTX Marker Analysis (Instrumented GPU Operations)

| Operation | Time % | Total (ms) | Instances | Avg (μs) | Per Step |
|-----------|--------|------------|-----------|----------|----------|
| **DeviceScan::ExclusiveSum** | 99.9% | 728.0 | 24,400 | 29.8 | 122 |
| **DeviceRadixSort** | 0.1% | 0.43 | 2 | 214.3 | 0.01 |
| **DeviceReduce::Sum** | 0.0% | 0.10 | 2 | 49.1 | 0.01 |
| **DeviceReduce::Max** | 0.0% | 0.05 | 2 | 25.8 | 0.01 |

**Analysis**:
- **122 DeviceScan operations per step** (24,400 / 200 steps)
- Average scan time: 29.8 μs (very fast!)
- **Scans are efficient**, but there are MANY of them
- Sorting and reduction overhead is negligible (<1%)

**Question**: Where is the PDE solver? The implicit CG solver should show up but doesn't have NVTX markers!

### 3. CPU-GPU Synchronization Bottleneck 🔴

| System Call | Time % | Total Time (s) | Calls | Avg (ms) | Interpretation |
|-------------|--------|----------------|-------|----------|----------------|
| **poll()** | 92.9% | 318.8 | 3,199 | 99.7 | GPU sync waits |
| **ioctl()** | 6.9% | 23.6 | 3,043,011 | 0.0078 | GPU status checks |

**Critical Findings**:

1. **Polling Dominates** (93% of OS time):
   - 318.8s spent in poll() out of 343s total OS time
   - 3,199 poll calls ≈ **16 polls per simulation step**
   - Average poll duration: 99.7ms (very long!)
   - **Interpretation**: CPU frequently waits for GPU to finish

2. **Excessive ioctl Calls** (3 million!):
   - 3,043,011 calls in 159s = **19,080 calls per second**
   - This is CUDA runtime checking GPU status constantly
   - Each call is fast (7.8 μs) but adds up to 23.6s total

**Root Cause**: Likely caused by:
- `cudaDeviceSynchronize()` calls after every kernel launch
- FLAME GPU's synchronous kernel execution model
- Lack of kernel fusion or batching

### 4. Population Dynamics

| Step | Cancer Cells | T Cells | Status |
|------|--------------|---------|--------|
| 0 | 515-601 | 49-50 | Initial |
| 50 | 97K-98K | 32 | Exponential growth |
| 100 | 125K | 16 | At capacity |
| 150 | 125K | 8 | Steady state |
| 200 | 125K | 6 | Steady state |

**Observations**:
- Tumor hits max capacity (125K cells) by step 100
- T cells steadily declining (no recruitment, being suppressed/dying)
- After step 100, simulation is just maintaining steady state
- **No new dynamics after step 100** (performance test could be shorter)

### 5. Time Budget Breakdown

**Total Runtime**: 159.7s (99% CPU utilization)

| Component | Estimated Time | % of Total |
|-----------|----------------|------------|
| **Poll (GPU sync)** | 318.8s* | - |
| **ioctl (GPU check)** | 23.6s | 14.8% |
| **GPU compute (NVTX)** | 0.7s | 0.4% |
| **Uninstrumented GPU** | ~50-100s? | ~50%? |
| **CPU compute** | ~30-40s | ~20%? |
| **QSP solver (CPU)** | <1s | <1% |

*Poll time overlaps with other work, can't sum directly

**Key Insight**: Most time is likely in uninstrumented GPU kernels (PDE solver, agent functions) that block CPU via synchronization.

---

## Critical Optimization Targets

### 🔴 **Priority 1: Reduce GPU Synchronization Overhead**

**Problem**: 16 GPU synchronizations per step, each causing 100ms CPU wait

**Root Causes**:
1. FLAME GPU likely calls `cudaDeviceSynchronize()` after each layer
2. PDE solver likely synchronizes after each CG iteration
3. No kernel fusion or asynchronous execution

**Solutions**:

**A. Reduce PDE Solver Synchronization (Est. 5× speedup)**
- **Current**: Likely synchronizing after each CG iteration (10-20 iterations per substrate × 10 substrates = 100-200 syncs per step)
- **Fix**: Remove intermediate synchronizations, only sync at end of full solve
- **File**: `PDAC/pde/pde_solver.cu:solve_implicit_cg()`
- **Change**: Remove `cudaDeviceSynchronize()` from CG loop, add single sync after all substrates solved

```cpp
// BEFORE (suspected):
for (int substrate = 0; substrate < num_substrates; substrate++) {
    for (int iter = 0; iter < max_cg_iters; iter++) {
        cg_step_kernel<<<>>>();
        cudaDeviceSynchronize();  // ❌ REMOVE THIS
    }
    cudaDeviceSynchronize();  // ❌ REMOVE THIS
}

// AFTER:
for (int substrate = 0; substrate < num_substrates; substrate++) {
    for (int iter = 0; iter < max_cg_iters; iter++) {
        cg_step_kernel<<<>>>();
        // No sync - let kernels pipeline
    }
}
cudaDeviceSynchronize();  // ✅ Only sync once at end
```

**B. Use CUDA Events for Precise Profiling (Est. 0× speedup, but critical for finding bottlenecks)**
- Add `cudaEventRecord()` around critical sections
- Measure actual GPU kernel time vs CPU wait time
- **File**: `PDAC/pde/pde_solver.cu`, `PDAC/pde/pde_integration.cu`

**C. Add NVTX Markers to PDE Solver (No speedup, but visibility)**
```cpp
#include <nvtx3/nvToolsExt.h>

void PDESolver::solve_timestep() {
    nvtxRangePush("PDE Solve Total");
    for (int substrate = 0; substrate < num_substrates; substrate++) {
        nvtxRangePush("PDE CG Solver");
        solve_implicit_cg(substrate);
        nvtxRangePop();
    }
    nvtxRangePop();
}
```

### 🟡 **Priority 2: Optimize Agent Processing Pipeline**

**Problem**: 122 DeviceScan operations per step seems high

**Analysis**:
- 200 steps × 122 scans/step = 24,400 total ✓ (matches NVTX count)
- Likely sources:
  - Location broadcasting (per agent type)
  - Neighbor scanning (per agent type)
  - State updates (per agent type)
  - Division intent/execution phases
- **Hypothesis**: Each FLAME GPU message passing operation triggers a scan

**Solutions**:

**A. Profile FLAME GPU Layer Execution**
- Add NVTX markers around each layer in `model_layers.cu`
- Identify which layers trigger most scans
- **File**: `PDAC/sim/model_layers.cu:defineMainModelLayers()`

**B. Reduce Message Passing Overhead**
- Investigate if some layers can be fused
- Check if broadcast + scan can be combined

**C. Increase Batch Size**
- Test with larger grid (75³, 101³) to amortize scan overhead over more agents

### 🟢 **Priority 3: PDE Solver Algorithm Optimization**

**Problem**: Cannot profile PDE solver internals (no NVTX, no kernel visibility)

**Immediate Actions**:
1. **Add NVTX markers** (see Priority 1C)
2. **Add timing printouts** to measure PDE solver contribution:
```cpp
// In pde_integration.cu:solve_pde_step()
auto start = std::chrono::high_resolution_clock::now();
solver_->solve_timestep(pde_timestep);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
if (step % 10 == 0) {
    printf("PDE solve step %d: %ld ms\n", step, duration.count());
}
```

**Long-term Optimizations** (after visibility established):
- Investigate CG iteration count (may be over-solving)
- Test different preconditioners (diagonal, incomplete Cholesky)
- Consider multi-grid methods for faster convergence
- Parallelize over substrates (currently sequential)

### 🟢 **Priority 4: T Cell Dynamics Investigation**

**Biological Issue**: T cells dying off rapidly (50 → 6 over 200 steps)

This is likely a **biological model issue**, not performance, but worth investigating:
- Are T cells dying due to lack of IL-2?
- Is suppression too strong?
- Should recruitment be enabled from QSP model?

**Files**: `PDAC/agents/t_cell.cuh`, `PDAC/qsp/qsp_integration.cu`

---

## Recommended Implementation Plan

### Phase 1: Add Visibility (1-2 hours)
**Goal**: Understand where time is actually spent

1. **Add NVTX markers**:
   - [ ] PDE solver (`pde_solver.cu:solve_timestep`, `solve_implicit_cg`)
   - [ ] Each model layer (`model_layers.cu`)
   - [ ] Source collection (`pde_integration.cu:collect_agent_sources`)

2. **Add timing printouts**:
   - [ ] PDE solve per step
   - [ ] Agent processing per layer
   - [ ] QSP solver per step

3. **Re-profile with markers**:
   ```bash
   ./build.sh
   ./profile_pdac.sh --steps 50
   ```

**Expected Result**: NVTX breakdown showing PDE solver time, layer times

### Phase 2: Remove Synchronization (2-4 hours)
**Goal**: Reduce CPU-GPU sync overhead

1. **Search for unnecessary sync points**:
   ```bash
   grep -r "cudaDeviceSynchronize" PDAC/
   grep -r "cudaStreamSynchronize" PDAC/
   ```

2. **Remove syncs from CG loop** (if present)

3. **Add single sync at layer boundaries** (if needed for correctness)

4. **Profile and measure**:
   ```bash
   ./profile_pdac.sh --steps 50
   # Compare poll() time before/after
   ```

**Expected Speedup**: 3-5× (based on reducing sync overhead)

### Phase 3: Algorithm Tuning (4-8 hours)
**Goal**: Optimize what's left

1. **CG convergence tuning**:
   - Check current tolerance and iteration counts
   - Test if tolerance can be relaxed

2. **Substrate parallelization**:
   - Solve all 10 substrates in parallel using CUDA streams

3. **Kernel fusion**:
   - Combine small kernels where possible

**Expected Speedup**: 1.5-2× additional

---

## Expected Performance After Optimization

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| **Runtime (200 steps)** | 159.7s | 30-50s | Reduce sync |
| **Time per step** | 0.8s | 0.15-0.25s | All optimizations |
| **GPU Utilization** | 14.8% | 40-60% | Better pipelining |
| **Poll overhead** | 93% | <50% | Async execution |

**Conservative Estimate**: 3-5× speedup
**Optimistic Estimate**: 5-10× speedup

---

## Quick Wins (< 1 hour each)

1. **Add PDE solver timing** to see if it's the bottleneck
2. **Add NVTX markers** to make GPU work visible
3. **Test with larger grid** (101³) to see if GPU utilization improves
4. **Profile on native Linux** (if available) to get full kernel visibility

---

## Files to Modify (Priority Order)

1. **PDAC/pde/pde_solver.cu** (CG synchronization, NVTX markers)
2. **PDAC/pde/pde_integration.cu** (timing, NVTX markers)
3. **PDAC/sim/model_layers.cu** (layer NVTX markers)
4. **PDAC/sim/main.cu** (overall timing breakdown)

---

## Next Steps

1. **Immediate** (today):
   - Add NVTX markers to PDE solver
   - Add timing printouts
   - Re-profile

2. **Short-term** (this week):
   - Remove unnecessary synchronizations
   - Profile and measure speedup
   - Test with larger grids

3. **Long-term** (next sprint):
   - Algorithm optimizations (CG tuning, substrate parallelization)
   - Consider GPU-resident agents (remove CPU-GPU transfers)
   - Investigate FLAME GPU kernel fusion options

---

## Questions to Answer via Profiling

1. **Where is PDE solver time?** (Add NVTX to find out)
2. **How many CG iterations per substrate?** (Add counter)
3. **Can we solve substrates in parallel?** (Test with streams)
4. **What triggers 122 scans per step?** (Add layer-level NVTX)
5. **Is QSP solver significant?** (Add timing)

---

## Appendix: Raw Data

### Key Statistics
- **Total simulation steps**: 200
- **Initial agents**: 515 cancer + 50 T cells = 565
- **Final agents**: 124,998 cancer + 6 T cells = 125,004
- **Grid size**: 50³ = 125,000 voxels
- **PDE substrates**: 10
- **Wall clock time**: 159.71s
- **CPU time**: 136.06s user + 22.92s system = 158.98s
- **CPU utilization**: 99% (excellent)

### NVTX Breakdown
- **DeviceScan**: 24,400 calls, 728ms total, 29.8μs average
- **DeviceRadixSort**: 2 calls, 0.43ms total
- **DeviceReduce::Sum**: 2 calls, 0.10ms total
- **DeviceReduce::Max**: 2 calls, 0.05ms total

### System Call Breakdown
- **poll()**: 3,199 calls, 318.8s total, 99.7ms average
- **ioctl()**: 3,043,011 calls, 23.6s total, 7.8μs average
- **system()**: 1 call, 656ms (initialization)
- **fread()**: 196 calls, 280ms (XML loading)
