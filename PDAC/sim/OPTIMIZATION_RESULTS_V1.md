# Optimization Results - Version 1

**Date**: February 12, 2026
**Changes**: Removed unnecessary cudaDeviceSynchronize() calls, added NVTX markers

---

## Performance Comparison (50 Steps)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Wall Clock Time** | ~40s | 26.8s | **1.49× faster** |
| **Time per Step** | 0.8s | 0.54s | **33% faster** |
| **poll() Calls** | 3,199 | 489 | **6.5× fewer** |
| **ioctl() Calls** | 3,043,011 | 610,291 | **5× fewer** |
| **poll() Overhead** | 93% | 88.6% | **4.4% reduction** |
| **GPU Utilization** | 14.8% avg | TBD | Not yet measured |

---

## NVTX Visibility - MAJOR WIN! 🎉

### Before Optimization:
```
DeviceScan::ExclusiveSum    99.9%    728ms    (agent processing)
DeviceRadixSort             0.1%     0.4ms
DeviceReduce                <0.1%    0.1ms
--------------------------------------------------
PDE Solver:                 INVISIBLE!
```

### After Optimization:
```
PDE Solve Total             31.8%    13,284ms    ← NOW VISIBLE!
  ├─ PDE Single Substrate   31.8%    13,282ms    (10 substrates × 50 steps)
  │   ├─ CG Iterations      31.6%    13,181ms    ← WHERE THE TIME GOES!
  │   ├─ CG Setup            0.2%        81ms
  │   └─ CG Finalize         0.0%         5ms
  └─ Collect Agent Sources   4.0%     1,684ms
DeviceScan::ExclusiveSum     0.5%       205ms    (agent processing)
DeviceRadixSort              0.0%         0.3ms
DeviceReduce                 0.0%         0.1ms
```

**Key Finding**: **PDE CG Iterations dominate runtime** (49% of total wall clock time)!

---

## Detailed Breakdown

### What We Can Now See

**PDE Solve Total** (50 steps):
- Total time: 13.3 seconds (49% of 26.8s runtime)
- Per step: 266ms average
- **This was completely invisible before!**

**CG Iterations** (500 total substrates solved):
- Total time: 13.2 seconds
- Per substrate: 26.4ms average
- **Iteration count**: ~1000 iterations per step (100/substrate, hitting max limit)
- **This explains why it's slow** - CG is struggling to converge!

**Collect Agent Sources** (50 steps):
- Total time: 1.7 seconds
- Per step: 33.7ms
- Reasonable overhead

**Agent Processing** (DeviceScan):
- Total time: 205ms (was 728ms for 200 steps, so ~36ms for 50 steps expected)
- Per step: ~4ms
- Very fast ✅

### Synchronization Reduction

**poll() System Calls**:
- Before: 3,199 calls (16 per step)
- After: 489 calls (9.8 per step)
- **Reduction**: 39% fewer polls per step

**ioctl() Calls**:
- Before: 3,043,011 calls (15,215 per step)
- After: 610,291 calls (12,206 per step)
- **Reduction**: 20% fewer ioctls per step

**Why Still High?**
- dot_product() calls cudaDeviceSynchronize() 3× per CG iteration
- With ~1000 CG iterations per step × 3 syncs = 3,000 syncs per step
- Plus syncs from FLAME GPU layers

---

## What We Learned

### 1. PDE Solver IS the Bottleneck (49% of runtime)

Before optimization, we couldn't see this. Now it's crystal clear:
- CG iterations: 13.2s (49% of runtime)
- Agent processing: 0.2s (0.7% of runtime)
- **PDE is 66× more expensive than agent processing!**

### 2. CG Solver Is Not Converging Well

**Iteration counts per substrate**: ~100 (hitting 1000 max limit)

Possible causes:
- Poor initial guess (using C^n as x0 should be good though)
- Stiff problem (large dt × high diffusion coefficient)
- Need preconditioning
- Tolerance too tight (1e-4)

**Next optimization target**: Reduce CG iteration count!

### 3. Synchronization Reduction Helped, But Limited

We removed 6-7 syncs per CG iteration, but dot_product still needs 1 sync per call (3× per iteration), so:
- Before: ~10 syncs/iteration
- After: ~3 syncs/iteration
- **Improvement**: 70% reduction in syncs/iteration ✅

But we're still doing 1000 iterations, so total syncs per step is still high:
- Before: ~10,000 syncs/step
- After: ~3,000 syncs/step

### 4. Next Bottleneck: CG Convergence

**Current performance budget**:
```
26.8s per 50 steps = 0.536s/step
  ├─ 0.266s PDE solve (49%)
  │   ├─ 0.264s CG iterations (99% of PDE time)
  │   ├─ 0.002s CG setup (<1%)
  │   └─ 0.000s CG finalize (<0.1%)
  ├─ 0.034s Source collection (6%)
  ├─ 0.004s Agent processing (1%)
  └─ 0.232s Other/FLAME GPU (44%)
```

**To get to 0.2s/step**, we need to reduce:
1. CG iterations (currently 0.264s/step) → target 0.05s/step (5× faster)
2. Other/FLAME GPU overhead (currently 0.232s/step) → needs investigation

---

## Optimization Success Criteria

✅ **Achieved**:
- [x] Added NVTX markers (PDE solver now visible)
- [x] Reduced synchronization overhead (6.5× fewer polls, 5× fewer ioctls)
- [x] 1.5× overall speedup (40s → 27s for 50 steps)
- [x] PDE timing instrumentation added
- [x] CG iteration counting added

⏳ **Partial**:
- [~] Reduced poll() overhead (93% → 88.6%, only 4.4% reduction)
- [~] Increased GPU utilization (need to measure)

❌ **Not Achieved**:
- [ ] 3-5× speedup target (only got 1.5×)
- [ ] Reduced CG iterations (still hitting 1000 max)

---

## Next Optimization Steps

### Priority 1: Fix CG Convergence (Target: 5× speedup)

**Problem**: 100 iterations per substrate (hitting max limit)

**Solutions to try**:
1. **Relax tolerance**: 1e-4 → 1e-3 (may not affect accuracy much)
2. **Better preconditioner**: Currently none, try diagonal or Jacobi
3. **Reduce max_iters**: If tolerance is too tight, just cap at 20-30 iterations
4. **Check initial guess**: Is C^n actually being used?
5. **Investigate dt × D product**: Very large values → stiff problem

**Code location**: `PDAC/pde/pde_solver.cu:427-428`
```cpp
const int max_iters = 1000;  // ← Try 30-50
const float tolerance = 1e-4f;  // ← Try 1e-3 or 5e-3
```

### Priority 2: Investigate "Other" Overhead (44% of runtime)

**Problem**: 0.232s per step unaccounted for

**Possible sources**:
- FLAME GPU layer execution overhead
- Message passing
- Agent kernel launches
- Additional syncs we haven't found

**Action**: Add NVTX markers to model layers
```cpp
// In model_layers.cu
layer.addHostFunction(my_function).setName("MyFunctionName");
// Add nvtxRangePush/Pop inside each function
```

### Priority 3: Parallelize Substrate Solving

**Current**: Sequential (substrate 0, 1, 2, ..., 9)
**Target**: Parallel using CUDA streams

**Estimated speedup**: 1.5-2× (won't get 10× due to memory bandwidth limits)

**Code changes**:
```cpp
// Create 10 CUDA streams
std::vector<cudaStream_t> streams(10);
for (int i = 0; i < 10; i++) cudaStreamCreate(&streams[i]);

// Solve all substrates in parallel
for (int sub = 0; sub < 10; sub++) {
    // Launch CG solver on streams[sub]
}

// Sync all streams
for (auto& s : streams) cudaStreamSynchronize(s);
```

---

## Validation

### Correctness ✅
- [x] Simulation completes without errors
- [x] Population counts reasonable (~94K cancer cells at step 50)
- [x] T cells declining as expected (49 → 32)
- [x] No NaN or inf values

### Performance ✅
- [x] Faster than before (1.5× speedup)
- [x] NVTX markers working
- [x] CG iteration counts printed
- [x] No memory leaks

---

## Conclusions

### What Worked
1. **NVTX markers** - Critical for visibility, now we can see PDE solver!
2. **Synchronization reduction** - 6.5× fewer polls, measurable impact
3. **Timing instrumentation** - Know exactly where time goes

### What Didn't Work As Expected
1. **Speedup** - Only 1.5× instead of 3-5× target
   - Reason: CG solver convergence is the real bottleneck
2. **poll() overhead** - Only 4.4% reduction (still 88.6% of OS time)
   - Reason: Still have ~3K syncs per step from dot_product calls

### Key Insight
**The bottleneck is NOT synchronization overhead** (though we improved it).
**The bottleneck IS CG convergence** - taking 1000 iterations per step!

If we can reduce CG iterations from 1000 → 200 (5× reduction), we'd get:
- CG time: 13.2s → 2.6s
- Total time: 26.8s → 16.2s
- **Per step**: 0.536s → 0.324s
- **Speedup**: 1.65× additional (2.5× total from baseline)

Combined with substrate parallelization (1.5× more), we'd hit:
- **Per step**: 0.324s → 0.216s
- **Total speedup**: 3.7× from baseline (0.8s → 0.22s)

**This would meet our 3-5× target!**

---

## Files Modified

1. `PDAC/pde/pde_solver.cu`
   - Added NVTX markers
   - Removed 8 unnecessary cudaDeviceSynchronize() calls
   - Added timing and iteration tracking
   - Changed return type to int

2. `PDAC/pde/pde_solver.cuh`
   - Updated function signature

3. `PDAC/pde/pde_integration.cu`
   - Added NVTX markers around source collection

---

## Rollback

If needed, revert with:
```bash
git checkout PDAC/pde/pde_solver.cu PDAC/pde/pde_solver.cuh PDAC/pde/pde_integration.cu
```

---

## Next Session Goals

1. Reduce CG iterations (relax tolerance, reduce max_iters, add preconditioner)
2. Add NVTX markers to model layers (find the "Other" 44%)
3. Profile again and compare
4. If successful, implement substrate parallelization
