# SPQSP HCC GPU — Optimization Plan

**Source capture:** `hcc_prof.nsys-rep` (Delta A100, 280 steps, g=50, `-G 3`)
**Date:** 2026-04-20
**Avg step time:** 90 ms (280 ABM steps in ~25s GPU time)

---

## Per-step budget

| Rank | Component | ms/step | % step | Kernel count/step |
|---|---|---|---|---|
| **1** | `fib_build_density_field_impl` | **30.5** | **34%** | 1 |
| **2** | PDE LOD solver (3 axis sweeps) | **24.3** | **27%** | 360 (12 substeps × 10 substrates × 3 axes) |
| 3 | Aggregate ABM Events | 6.1 | 7% | 1 |
| 4 | QSP Solve (CVODE, CPU) | 4.8 | 5% | 1 |
| 5 | `vascular_state_step` | 3.4 | 4% | 1 |
| 6 | `stepLayer 83` (FLAMEGPU2 internal) | 3.7 | 4% | — |
| | **Top 6 = 81% of step** | | | |

## Memory totals (over 280 steps)

- **Memset: 358 GB / 12k calls** (avg 30 MB, max 1 GB) — **49% of all memory-op time**
- D2H: 2.96 GB (9.4k calls) — simulation output
- H2D: 785 MB (78k tiny calls at 10 KB avg) — FLAMEGPU2 CurveTable updates
- D2D: 40 MB

## CUDA API host-time

- `cudaStreamSynchronize`: **13.3 s (53% of API time)** across 505k calls
- `cudaDeviceSynchronize`: 6.6 s (13.8k calls)
- `cudaLaunchKernel`: 1.9 s (**562k launches** → ~2000/step)
- Sync overhead consumes ~half of host-side runtime.

---

## Priority 1 — `fib_build_density_field` (~25 ms/step potential save)

30.5 ms/step for scattering ~1400 fibroblasts. Dramatically over-budget.

**Hypotheses to confirm from code read:**
- `atomicAdd` contention — fibs in the same region write the same voxel cells.
- Redundant per-slice compute — 2D per-z-slice Nadaraya-Watson re-evaluates kernel for each z.
- Thread-per-agent under-utilization — 1400 threads won't saturate an A100.

**Actions:**
1. Read `agents/fibroblast.cuh` and the `fib_build_density_field_impl` definition.
2. Consider **thread-per-voxel gather** (O(Nvox · neighbors), no atomics) using a fibroblast spatial hash.
3. If keeping scatter: clamp Gaussian at 3σ (not 6σ); reduce z-footprint.

**Expected:** 30 ms → 5–8 ms.

---

## Priority 2 — PDE LOD solver (~10 ms/step potential save)

Individual Thomas kernels are fine (~20 µs). The issue is volume: **360 launches/step**.

**Actions:**
1. **Batch substrates**: one kernel launch handles all 10 substrates (stride loop). Cuts 360 → 36 launches/step. Need SoA-by-substrate memory layout for coalescing.
2. **Fuse source/uptake into x-sweep**: removes one kernel class from the pipeline.
3. Evaluate **cuSPARSE `gtsv2StridedBatch`** as a drop-in for Thomas — A100-tuned.

**Expected:** 24 ms → 12–15 ms.

**Files:** `HCC/pde/pde_integration.cu`, `HCC/pde/pde_solver.*`.

---

## Priority 3 — Memset audit (~5–10 ms/step potential save)

358 GB total memset with max single call 1 GB. Likely includes unnecessary zero-fills between PDE substeps.

**Actions:**
1. Open `hcc_prof.nsys-rep` in Nsight Systems UI, find the largest memsets, identify their call sites.
2. Grep `cudaMemset` / `cudaMemsetAsync` in `HCC/pde/*.cu` — audit each: does the solver require zero-init or will it overwrite?
3. Double-buffer instead of zero-ing where possible.

---

## Priority 4 — Kernel launch overhead (~2–5 ms/step potential save)

562k launches / 28s = 20k launches/sec, ~3-4 µs launch cost each. Priority 2 alone cuts PDE launches 10×. Additional:
- Movement substep interleaving generates 53 move kernels/step — consolidating this is a biology-invariant change.
- **CUDA Graphs**: capture+replay requires static kernel signatures (same args every step). Agent counts change, so likely not viable without restructuring.

---

## Priority 5 — Aggregate ABM Events (6 ms/step, 7%)

Single call taking 6 ms for a ~100k agent reduction. Suspiciously slow.

**Actions:** inspect `aggregate_abm_events` in `core/model_functions.cu`. If CPU-side, port to a GPU reduction.

---

## Deferred (low ROI for now)

- **QSP CVODE (4.8 ms/step)** — single-threaded CPU, hard to parallelize (strong ODE coupling). Not worth attacking first.
- **Agent movement kernels** — each <1% individually. Even 2× speedup on all of them saves only 3–4 ms/step.

---

## Execution order

1. **Priority 1 first** (highest ROI + easiest local change).
2. Priority 2 (PDE batching).
3. Memset audit (Priority 3) via Nsight UI walk.
4. **Re-profile after each change** with `sbatch --export=ALL,NSYS=1 submit.sh -s 280 -g 50 -G 3` to validate savings and catch regressions.
5. Revisit Priority 4/5 once top two are addressed.

---

## Progress (2026-04-20, WSL2 local)

| Phase | File | ms/step before | ms/step after | Delta projection |
|---|---|---|---|---|
| 1 — ECM density separable Gaussian | `agents/fibroblast.cuh` | 15.4 | 8.3 | 30.5 → 8–12 |
| 2 — PDE solver substrate batching | `pde/pde_solver.cu/.cuh` | 24.8 | 6.7 | 24.3 → 6–8 |
| 5 — ABM events aggregate (device counters) | `agents/cancer_cell.cuh`, `pde/pde_integration.cu`, `core/common.cuh` | ~6 (est) | <0.1 | 6 → <0.5 |

**Local wall time 280 steps g=50 G=3:** 22s → 15.8s (−28%).
**Correctness:** voxel-level diff between Phase 2 and Phase 2-rerun equals diff between Phase 1 and Phase 2 → all variation is RNG divergence, not a numerical bug. Final population counts and QSP trajectories stay in distribution across runs.

### Remaining / deferred
- **Priority 3 (memset audit)**: most user-code memsets run once/step, not per-substep. Local reward ~1 ms/step; deferred.
- **Priority 4 (launch overhead)**: Phase 2 already cut PDE launches ~10× (480 → 48/substep group). Movement substep consolidation is biology-invariant; skip.
- Next measurement: re-capture Delta A100 profile (`sbatch --export=ALL,NSYS=1 submit.sh -s 280 -g 50 -G 3`).

## Reference files

- `hcc_prof.nsys-rep` — full timeline, open in Nsight Systems UI
- `hcc_prof_cuda_gpu_kern_sum.csv` — per-kernel GPU time
- `hcc_prof_cuda_gpu_mem_time_sum.csv` / `_mem_size_sum.csv` — memory operations
- `hcc_prof_cuda_api_sum.csv` — host-side CUDA API calls
- `hcc_prof_nvtx_sum.csv` — NVTX phase ranges
