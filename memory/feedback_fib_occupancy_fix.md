---
name: Fibroblast occupancy slot fix (initialization)
description: Fibroblast init must check and write to slot [1], not slot [0] (cancer) in the occupied array
type: feedback
---

During QSP-seeded initialization (`initializeFibroblastsFromQSP` in `PDAC/sim/initialization.cu`), fibroblast chain segments must use the **fibroblast occupancy slot [1]**, not the cancer slot [0].

**Fix applied (Mar 20, 2026):**
- Head voxel check: `occupied[idx][0] != 0 || occupied[idx][1] != 0`
- Segment marking: `occupied[idx][1] = 1`
- Rollback on failed chain: `occupied[idx][1] = 0`
- `findFreeAdjacent`: checks `[0] == 0 && [1] == 0`

**Why:** Previously fibroblasts wrote to slot [0] (same as cancer), contaminating the cancer occupancy array and causing a ~10% overcount of fibroblast chains vs HCC. XML parameters confirmed identical between PDAC and HCC (V_T_CAF = 1.09126e4, V_T_C1 = 4.7e6).

**How to apply:** Whenever occupied array is used in init, slot [0] = cancer/exclusive, slot [1] = fibroblast chains, slot [2] = reserved.
