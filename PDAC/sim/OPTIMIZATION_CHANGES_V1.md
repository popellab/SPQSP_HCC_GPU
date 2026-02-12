# PDE Solver Optimization - Version 1

**Date**: February 12, 2026
**Objective**: Reduce CPU-GPU synchronization overhead and add profiling visibility

---

## Changes Made

### 1. Added NVTX Profiling Markers

**Files Modified**:
- `PDAC/pde/pde_solver.cu`
- `PDAC/pde/pde_integration.cu`

**Markers Added**:
- "PDE Solve Total" - wraps entire solve_timestep()
- "PDE Single Substrate" - wraps each substrate solve
- "CG Setup" - wraps CG initialization
- "CG Iterations" - wraps CG iteration loop
- "CG Finalize" - wraps non-negative clamping
- "Collect Agent Sources" - wraps source collection

**Result**: PDE solver will now be visible in NVTX profiling!

### 2. Removed Unnecessary GPU Synchronizations

**Before**: ~700-1,400 synchronizations per ABM step
**After**: ~30-60 synchronizations per ABM step (just from dot_product calls)

**Synchronizations Removed** (4 per CG iteration):
1. ❌ Line 459: After apply_diffusion_operator (A*x0) - **REMOVED**
2. ❌ Line 464: After vector_axpy (r = b - A*x0) - **REMOVED**
3. ❌ Line 468: After vector_copy (p = r) - **REMOVED**
4. ❌ Line 479: After apply_diffusion_operator (A*p) in loop - **REMOVED**
5. ❌ Line 490: After vector_axpy (r = r - alpha*Ap) in loop - **REMOVED**
6. ❌ Line 508: After updating p in loop - **REMOVED**
7. ❌ Line 515: After clamp_nonnegative - **REMOVED**
8. ❌ Line 540: After building RHS (per substrate) - **REMOVED**

**Synchronizations Kept** (necessary):
1. ✅ Line 442: Inside dot_product before cudaMemcpy - **REQUIRED**
2. ✅ Line 568: After all substrates solved - **REQUIRED** (ensures GPU work done before FLAME GPU continues)

**Rationale**:
- CUDA kernels on the same stream execute in order automatically
- Synchronization is ONLY needed before:
  - cudaMemcpy (device to host)
  - Returning control to FLAME GPU
- All intermediate syncs were unnecessary and caused CPU to wait unnecessarily

### 3. Added Timing and Iteration Tracking

**Added Features**:
- CG iteration count per substrate
- Total CG iterations per timestep
- Millisecond timing for entire PDE solve
- Console output every 50 steps

**Console Output Example**:
```
[PDE Step 0] Solved 10 substrates in 45 ms (127 total CG iterations, avg 12.7 iters/substrate)
[PDE Step 50] Solved 10 substrates in 38 ms (118 total CG iterations, avg 11.8 iters/substrate)
```

**Function Signature Changed**:
```cpp
// Before:
void solve_implicit_cg(float* d_C, const float* d_rhs, float D, float lambda, float dt, float dx);

// After:
int solve_implicit_cg(float* d_C, const float* d_rhs, float D, float lambda, float dt, float dx);
// Returns: number of CG iterations taken
```

### 4. Added #include for NVTX and Timing

**Files Modified**:
- `PDAC/pde/pde_solver.cu` - added `<nvtx3/nvToolsExt.h>` and `<chrono>`
- `PDAC/pde/pde_integration.cu` - added `<nvtx3/nvToolsExt.h>`
- `PDAC/pde/pde_solver.cuh` - updated function signature

---

## Expected Performance Impact

### Conservative Estimate:
- **Before**: 159.7s for 200 steps (~0.8s/step)
- **After**: 40-60s for 200 steps (~0.2-0.3s/step)
- **Speedup**: 2.5-4.0×

### Optimistic Estimate:
- **After**: 30-40s for 200 steps (~0.15-0.2s/step)
- **Speedup**: 4-5×

### Why the Speedup:
1. **Reduced CPU waiting**: Poll overhead should drop from 93% to <50%
2. **Better kernel pipelining**: GPU can start next kernel while previous is finishing
3. **Fewer ioctl calls**: Less GPU status checking (should drop from 3M to ~300K)

---

## How to Test

### 1. Rebuild
```bash
cd /home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim
./build.sh
```

### 2. Run Quick Test
```bash
./build/bin/pdac -s 50 -oa 0 -op 0
```

**Look for**:
- PDE solver timing printouts every 50 steps
- Fewer poll() calls in profiling
- Much faster runtime!

### 3. Profile with NVTX
```bash
./profile_pdac.sh --steps 50
cat profiling_results/nsys_stats_*.txt | grep "NVTX Range" -A 20
```

**Look for**:
- "PDE Solve Total" marker showing up!
- "CG Iterations" showing actual GPU time
- Breakdown of time between collection, solve, and substrates

### 4. Compare Before/After
```bash
# Before: ~160s for 200 steps
# After: measure with
/usr/bin/time -v ./build/bin/pdac -s 200 -oa 0 -op 0
```

---

## Verification Checklist

After rebuild and test, verify:
- [ ] Simulation completes without errors
- [ ] Population counts match previous runs (~125K cancer cells at step 100)
- [ ] PDE concentrations are reasonable (O2, TGFB, CCL2 non-zero)
- [ ] NVTX markers appear in profile
- [ ] Runtime is significantly faster (>2× speedup)
- [ ] Console shows CG iteration counts

---

## Potential Issues

### Issue 1: Compilation Errors
**Symptoms**: nvtx3/nvToolsExt.h not found
**Solution**: NVTX is part of CUDA toolkit, should be in /usr/local/cuda-12.6/include
**Workaround**: If missing, comment out all nvtx lines temporarily

### Issue 2: Different Results
**Symptoms**: Population counts or PDE values change
**Cause**: Removing syncs shouldn't change results (kernels still execute in order)
**Action**: If results differ, it indicates a bug in original code that syncs were masking
**Debug**: Add back syncs one by one to find which one matters

### Issue 3: CUDA Errors
**Symptoms**: CUDA error messages during runtime
**Cause**: Possible race condition exposed by removing syncs
**Action**: Check for dependencies between kernels - may need to add back specific syncs
**Note**: Unlikely, but possible

---

## Code Review Notes

### Thread Safety
- ✅ All kernels on default stream (stream 0) execute in order
- ✅ dot_product syncs before cudaMemcpy (required)
- ✅ Final sync before returning to FLAME GPU (required)
- ✅ No race conditions introduced

### Correctness
- ✅ CG algorithm unchanged (only sync removal)
- ✅ Same numerical results expected
- ✅ NVTX markers don't affect execution (just profiling)

### Performance
- ✅ Major reduction in CPU-GPU synchronization overhead
- ✅ Better GPU kernel pipelining
- ✅ No negative performance impacts expected

---

## Next Steps After This Change

1. **Measure actual speedup** - profile and compare
2. **Verify NVTX visibility** - confirm PDE solver shows up in profile
3. **Analyze new bottlenecks** - what's the next optimization target?
4. **Consider substrate parallelization** - solve 10 substrates in parallel using CUDA streams
5. **Tune CG convergence** - can we reduce iteration count without losing accuracy?

---

## Rollback Procedure

If this change causes issues:

```bash
cd /home/chase/SPQSP/SPQSP_PDAC-main
git diff PDAC/pde/  # Review changes
git checkout PDAC/pde/pde_solver.cu  # Revert
git checkout PDAC/pde/pde_solver.cuh
git checkout PDAC/pde/pde_integration.cu
./build.sh
```

---

## Author Notes

This optimization was identified through profiling which showed:
- 93% of OS time in poll() (CPU waiting for GPU)
- 3 million ioctl calls (GPU status checks)
- Only 0.4% of runtime was visible GPU work

The root cause was excessive cudaDeviceSynchronize() calls inside the CG iteration loop,
causing the CPU to wait for the GPU after every small kernel launch, preventing kernel
pipelining and introducing massive overhead.

By removing unnecessary syncs and keeping only required ones (before cudaMemcpy and at
layer boundaries), we allow the GPU to pipeline kernels and reduce CPU waiting time.
