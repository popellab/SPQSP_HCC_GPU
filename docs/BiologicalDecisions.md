# Biological Decisions & Inconsistencies

This file tracks places where the PDAC GPU model intentionally or unintentionally diverges
from the HCC CPU reference, along with the biological justification (or lack thereof).

Each entry notes: the location, what HCC does, what PDAC does, which is more biologically
correct, and the current status.

---

## T Cell Exhaustion: TReg vs PDL1 Suppression (Independent or Mutually Exclusive?)

**File**: `PDAC/agents/t_cell.cuh` — `tcell_state_step`, inside `if (cell_state == T_CELL_CYT)` block

**HCC behavior** (`TCell.cpp:308`):
```cpp
if (_count_neighbor_Treg > 0) { /* TReg suppression roll */ }
else if (_count_neighbor_all > 0) { /* PDL1 suppression roll */ }
```
TReg suppression and PDL1 suppression are **mutually exclusive**: if any TReg neighbor is
present, the PDL1 check is skipped entirely.

**PDAC behavior** (after revert, currently matches HCC with `else if`):
```cpp
if (neighbor_Treg > 0) { /* TReg suppression roll */ }
else if (neighbor_all > 0) { /* PDL1 suppression roll */ }
```

**Biological argument for double `if` (independent checks)**:
TReg-mediated suppression (IL-10, TGF-β, direct contact) and PD1-PDL1 checkpoint signaling
are completely independent molecular pathways. A cytotoxic T cell surrounded by both TReg
cells and PDL1-expressing cancer cells would face both suppression signals simultaneously.
The `else if` formulation means TReg presence causes the model to ignore PDL1 suppression,
which has no mechanistic justification.

**Current status**: Using `else if` (matches HCC reference) to aid comparison.
**Recommended future change**: Switch to independent `if` / `if` blocks once HCC comparison
is no longer the primary validation target, or update both models simultaneously.

---

## M1 Macrophage IFN-γ Production: Unconditional vs Contact-Dependent

**File**: `PDAC/agents/macrophage.cuh` — `mac_compute_chemical_sources`

**HCC behavior** (`Mac.cpp:40-44`, copy constructor):
```cpp
if (_state == AgentStateEnum::MAC_M1)
    setup_chem_source(_source_IFNg, CHEM_IFN, params.getVal(PARAM_IFN_G_RELEASE));
```
All M1 MACs initialize IFN-γ at full rate immediately on creation. `agent_state_scan` calls
`update_chem_source(_source_IFNg, rate)` only when a cancer neighbor is found — it never resets
to zero when cancer is absent. Net result: **M1 MACs produce IFN-γ unconditionally** once
created, regardless of cancer contact.

**PDAC behavior** (currently matches HCC):
M1 MACs produce IFN-γ unconditionally at full rate.

**Suspected bug in HCC**: The `agent_state_scan` contact-dependent update was likely intended
to gate IFN-γ on cancer contact, but because the copy constructor pre-initializes the source
to full rate and the scan never zeros it when cancer is absent, the contact check has no
practical effect. Biologically, M1 macrophage IFN-γ secretion should require activation
via cancer/pathogen contact.

**Impact**: Unconditional IFN-γ elevates diffuse concentration across the grid, upregulating
PDL1 on cancer cells (via Hill equation in `cancer_cell_state_step`), which increases T cell
exhaustion rates via PDL1-PD1 interaction.

**Current status**: Kept unconditional to match HCC. Flag for future biological validation
and potential fix in both models simultaneously.

---

## T Cell Division Capacity Check: Inverted Occupancy Bug (Fixed Mar 2026)

**File**: `PDAC/agents/t_cell.cuh` — `tcell_divide`; `PDAC/agents/t_reg.cuh` — `treg_divide`

**Bug**: The `has_cancer` boolean used `== 0u` (true when NO cancer present), causing
`MAX_T_PER_VOXEL_WITH_CANCER = 1` to be applied to empty voxels and `MAX_T_PER_VOXEL = 8`
to cancer-occupied voxels — the opposite of intent and of HCC behavior.

**HCC behavior** (`TumorGridVoxel.cpp:64-65`): cancer voxels cap at `PARAM_N_T_VOXEL_C = 1`,
empty voxels cap at `PARAM_N_T_VOXEL = 8`. PDAC values match exactly.

**Fix**: Changed to `> 0u`. Both T cell and TReg divide functions now match HCC.

**Current status**: Fixed.

---

## T Cell Tumble Phase: Weighted Direction vs Uniform Random

**File**: `PDAC/agents/t_cell.cuh` — `tcell_move`, tumble phase (~line 540)

**HCC behavior** (`TCell.cpp:149-162`):
During tumble, picks any open Von Neumann neighbor with **uniform random probability**
(`getOneOpenVoxel` from `getMoveDestinationVoxels`). No directional weighting.

**PDAC behavior**:
Applies a sigma-weighted formula `exp(cos_theta/(σ²)) / exp(1/σ²)` (σ=0.524) that biases
the new direction toward the previous movement direction even during tumble.

**Biological argument**:
The uniform random tumble is the standard run-and-tumble model (e.g., Berg & Brown). The
weighted formula gives PDAC T cells stronger directional memory than the reference model,
causing different spatial clustering behavior.

**Current status**: Fixed — replaced sigma-weighted formula with uniform random pick from all 26 Moore neighbors (matches HCC).

---

