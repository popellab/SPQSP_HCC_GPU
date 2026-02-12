# Optimization Quick Start - Action Items

Based on full 200-step profiling run (159.7s runtime)

## 🔴 Critical Findings

### The Bottleneck: CPU-GPU Synchronization
- **93% of time** spent in `poll()` waiting for GPU
- **3 million ioctl calls** checking GPU status
- **16 GPU syncs per simulation step**
- GPU only utilized **14.8%** on average (peaks to 96%)

**Bottom Line**: GPU is FAST but CPU is constantly waiting for it!

---

## 🎯 Top 3 Quick Wins

### 1. Add NVTX Markers to PDE Solver (30 min, 0× speedup but critical visibility)

The PDE solver is invisible in current profiling!

**File**: `PDAC/pde/pde_solver.cu`

**Add at top**:
```cpp
#include <nvtx3/nvToolsExt.h>
```

**Wrap solve_timestep()**:
```cpp
void PDESolver::solve_timestep(float dt) {
    nvtxRangePush("PDE Solve Total");

    for (int substrate_idx = 0; substrate_idx < num_substrates_; substrate_idx++) {
        nvtxRangePush("PDE CG Solver");
        solve_implicit_cg(substrate_idx, dt);
        nvtxRangePop();
    }

    nvtxRangePop();
}
```

**Also wrap in** `PDAC/pde/pde_integration.cu:solve_pde_step()`:
```cpp
void solve_pde_step(...) {
    nvtxRangePush("PDE Step");
    solver_->solve_timestep(pde_timestep);
    nvtxRangePop();
}
```

**Then rebuild and re-profile**:
```bash
./build.sh
./profile_pdac.sh --steps 50
cat profiling_results/nsys_stats_*.txt | grep "NVTX Range" -A 10
```

### 2. Find and Remove Unnecessary cudaDeviceSynchronize() (1-2 hours, 3-5× speedup)

**Search for sync points**:
```bash
grep -rn "cudaDeviceSynchronize" PDAC/pde/
grep -rn "cudaStreamSynchronize" PDAC/pde/
```

**Rule of thumb**:
- ❌ DON'T sync inside CG iteration loop
- ❌ DON'T sync after each substrate
- ✅ DO sync once at end of full PDE solve
- ✅ DO sync before reading results back to CPU

**Expected locations**:
- Inside `solve_implicit_cg()` loop (remove if present)
- After each substrate solve (remove if present)
- Keep only final sync before returning to FLAME GPU

### 3. Add Timing Printouts (15 min, 0× speedup but shows where time goes)

**In** `PDAC/pde/pde_integration.cu:solve_pde_step()`:
```cpp
#include <chrono>

void solve_pde_step(int step, ...) {
    auto t1 = std::chrono::high_resolution_clock::now();

    collect_agent_sources(...);

    auto t2 = std::chrono::high_resolution_clock::now();
    solver_->solve_timestep(pde_timestep);
    auto t3 = std::chrono::high_resolution_clock::now();

    if (step % 10 == 0) {
        auto collect_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        auto solve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        printf("Step %d: collect=%ldms, PDE_solve=%ldms\n", step, collect_ms, solve_ms);
    }
}
```

**Run**:
```bash
./build/bin/pdac -s 50 -oa 0 -op 0 | grep "Step.*collect"
```

---

## 📊 What We Learned from Profiling

### GPU Operations (from NVTX)
- **DeviceScan**: 99.9% of GPU time (24,400 calls, ~30μs each)
  - 122 scans per simulation step (agent processing)
  - These are FAST and efficient ✅
- **Sorting/Reduction**: <1% overhead (negligible)

### Missing from Profile
- **PDE solver kernels**: NOT instrumented with NVTX
- **Agent function kernels**: NOT visible
- **Actual kernel execution time**: Unknown (WSL2 limitation)

### System-Level (from /usr/bin/time)
- **Total runtime**: 159.7s (0.8s per step)
- **CPU utilization**: 99% (good!)
- **Memory**: 405 MB (stable, no leaks)
- **Context switches**: 4,790 voluntary (low, good)

### Population Dynamics
- Tumor grows: 515 → 125K cells (hits capacity at step 100)
- T cells decline: 50 → 6 (dying off, no recruitment)
- After step 100: steady state (no more growth)

---

## 🛠️ Recommended Workflow

### Today (2-3 hours)
1. ✅ Add NVTX markers to PDE solver
2. ✅ Add timing printouts
3. ✅ Rebuild and re-profile:
   ```bash
   ./build.sh
   ./profile_pdac.sh --steps 50
   ```
4. ✅ Analyze new NVTX data to find PDE solver time

### This Week (4-8 hours)
1. ✅ Remove unnecessary `cudaDeviceSynchronize()` calls
2. ✅ Profile and measure speedup
3. ✅ Test with larger grid (101³) to increase GPU load:
   ```bash
   ./build/bin/pdac -s 50 -g 101 -oa 0 -op 0
   ```
4. ✅ Investigate T cell death (biological issue)

### Next Sprint (Optional)
1. Parallelize substrate solving (use CUDA streams)
2. Tune CG solver convergence criteria
3. Add kernel fusion opportunities
4. Profile on native Linux for full kernel visibility

---

## 📈 Expected Speedup

| Optimization | Est. Speedup | Effort | Priority |
|--------------|--------------|--------|----------|
| Remove sync points | 3-5× | 2h | 🔴 HIGH |
| Add NVTX visibility | 0× (insight) | 0.5h | 🔴 HIGH |
| Parallelize substrates | 1.5-2× | 4h | 🟡 MEDIUM |
| CG tuning | 1.2-1.5× | 2h | 🟢 LOW |
| **Total Potential** | **5-10×** | **8-12h** | |

**Target**: 200 steps in 30-50s (currently 160s)

---

## 📁 Files to Modify

### Priority 1 (NVTX + Timing)
- `PDAC/pde/pde_solver.cu` (add NVTX in solve_timestep, solve_implicit_cg)
- `PDAC/pde/pde_integration.cu` (add NVTX and timing in solve_pde_step)

### Priority 2 (Remove Syncs)
- `PDAC/pde/pde_solver.cu` (remove syncs from CG loop)

### Optional (Advanced)
- `PDAC/sim/model_layers.cu` (add NVTX per layer)
- `PDAC/pde/pde_solver.cu` (parallelize substrates with streams)

---

## 🔍 Questions to Answer

Before optimizing, we need to know:
1. **How much time is PDE solver?** (Add NVTX to find out)
2. **How many CG iterations?** (Print iteration count)
3. **Is sync inside CG loop?** (Search for cudaDeviceSynchronize)
4. **Can substrates be parallel?** (Check for dependencies)

**Add this to CG solver** to count iterations:
```cpp
int total_iters = 0;
for (int substrate = 0; substrate < num_substrates; substrate++) {
    int iters = solve_implicit_cg(substrate, dt);
    total_iters += iters;
}
printf("PDE solve: %d total CG iterations\n", total_iters);
```

---

## ⚠️ Important Notes

### WSL2 Limitations
- Cannot see individual CUDA kernel times
- Cannot see memory transfer bandwidths
- **Solution**: Use NVTX markers for high-level timing

### Don't Over-Optimize
- Profile FIRST, optimize SECOND
- Focus on the 80/20 (poll/ioctl overhead is 93% of time)
- Small optimizations (<5% gain) may not be worth complexity

### Preserve Correctness
- Always verify results after optimization
- Compare PDE output before/after
- Check agent counts match

---

## 🚀 Start Here

```bash
# 1. Add NVTX markers (see "Quick Win #1" above)
code PDAC/pde/pde_solver.cu
code PDAC/pde/pde_integration.cu

# 2. Rebuild
./build.sh

# 3. Profile
./profile_pdac.sh --steps 50

# 4. Check results
cat profiling_results/nsys_stats_*.txt | grep "NVTX Range" -A 15

# 5. Look for "PDE" in NVTX markers - you should now see it!
```

Then read `OPTIMIZATION_ANALYSIS.md` for detailed breakdown and next steps.
