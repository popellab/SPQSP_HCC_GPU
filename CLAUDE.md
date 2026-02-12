# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPQSP PDAC is a GPU-accelerated agent-based model (ABM) with CPU QSP coupling for simulating pancreatic ductal adenocarcinoma (PDAC) tumor microenvironment dynamics. It combines:

- **GPU (FLAME GPU 2)**:
  - Discrete agent-based modeling (cancer cells with functional behavior)
  - Continuous PDE-based chemical diffusion (O2, IFN, IL2, IL10, TGFB, CCL2, ArgI, NO, IL12, VEGFA)
  - Implicit conjugate gradient solver for unconditionally stable diffusion-decay

- **CPU (SUNDIALS CVODE)**:
  - Systemic QSP model for LymphCentral compartment (59+ species ODE system)
  - Drug pharmacokinetics (NIVO, CABO) and pharmacodynamics
  - Immune cell dynamics (recruitment, exhaustion, activation)

- **Integration**:
  - XML-based parameter system for runtime configuration
  - PDE-ABM coupling: chemicals flow from agents to PDE and back
  - QSP-ABM coupling infrastructure (partial implementation)
  - Voxel-based spatial discretization (configurable 10-100 µm resolution)

## Current Implementation Status (Feb 2026)

### ✅ **Fully Functional Components**

**Cancer Cell Agent (CancerCell)**
- ✅ States: Stem, Progenitor, Senescent
- ✅ Movement with random walk
- ✅ Division with cooldown and max division count
- ✅ Senescence upon division limit
- ✅ PDL1 upregulation via IFN-gamma (Hill equation)
- ✅ Hypoxia detection and response
- ✅ Chemical secretion: CCL2, TGFB (stem only), VEGFA
- ✅ Chemical consumption: O2, IFN-gamma
- ✅ Neighbor scanning (cancer, T cell, TReg, MDSC counts)
- ✅ Voxel capacity constraints (1 cancer per voxel)

**PDE Solver**
- ✅ 10 chemicals: O2, IFN, IL2, IL10, TGFB, CCL2, ARGI, NO, IL12, VEGFA
- ✅ Implicit conjugate gradient solver (unconditionally stable, no substeps needed!)
- ✅ Diffusion-decay-reaction equation: ∂C/∂t = D∇²C - λC + S
- ✅ Neumann (no-flux) boundary conditions
- ✅ Source collection from all agents via atomic operations
- ✅ Unit conversion: sources (positive) divided by voxel_volume, sinks (negative) used as-is
- ✅ Configurable grid size, voxel size, timestep
- ✅ CSV output for visualization

**PDE-ABM Coupling**
- ✅ Agents read local chemical concentrations from PDE
- ✅ Agents compute source/sink rates based on state
- ✅ Source collection aggregates contributions from all agents
- ✅ PDE solve updates concentrations for next step
- ✅ **CRITICAL FIX (Feb 11, 2026)**: Fixed double-offset bug in source collection kernel

**QSP Model**
- ✅ LymphCentral ODE system (153 species, 277 parameters)
- ✅ CVODE integration wrapper
- ✅ Parameter loading from XML
- ✅ Time stepping synchronized with ABM
- ✅ Infrastructure for ABM→QSP and QSP→ABM data exchange

**Parameter System**
- ✅ XML-based configuration (param_all_test.xml)
- ✅ 124+ parameters loaded and accessible
- ✅ Type-safe parameter enums (GPUParamFloat, GPUParamInt, GPUParamBool)
- ✅ Environment property population for FLAME GPU
- ✅ Command-line overrides for grid size, steps, initial populations

**Build & Output**
- ✅ CMake build system with CUDA, SUNDIALS, Boost, FLAME GPU
- ✅ Agent output: CSV files with positions, states, properties
- ✅ PDE output: CSV files with spatial chemical distributions
- ✅ Compiles on GPU-enabled systems (CUDA 11.0+)

**T Cell Agent (TCell) - NOW WORKING (Feb 2026)**
- ✅ Agent definitions with properties and state machines
- ✅ Movement with random walk and chemotaxis
- ✅ Neighbor scanning for cancer cells, TRegs, MDSCs
- ✅ T cell killing of cancer cells - functional
- ✅ State transitions triggered by chemicals (IL-2, IFN-γ, etc.)
- ✅ Division mechanics with cooldown
- ✅ Chemical secretion (IL-2, IFN-γ)
- ✅ Death/removal mechanisms
- ✅ Randomized division cooldown (0-100% of base interval)

### 🔄 **Partially Implemented / Non-Functional**

**Immune Cell Agents (TReg, MDSC) - Still Non-Functional**
- ⚠️ Agent definitions exist with properties and state machines
- ⚠️ Basic movement and neighbor scanning implemented
- ❌ **No recruitment function** - cells are initialized but not recruited dynamically
- ❌ **No TReg suppression** - suppression functions defined but not active
- ❌ **No MDSC suppression** - NO/ArgI secretion defined but effects not implemented
- ❌ **No state transitions triggered by chemicals** - cytokine sensing not functional
- ❌ **No death/removal mechanisms** - cells persist indefinitely

**QSP-ABM Bidirectional Coupling**
- ✅ Infrastructure complete (functions, wrappers, data structures)
- ❌ **Species index mapping not implemented** - can't extract specific ODE variables
- ❌ **ABM→QSP feedback not connected** - cancer deaths, T cell counts not updating QSP
- ❌ **QSP→ABM data transfer not active** - drug concentrations not passed to agents
- ❌ **No drug effects on agents** - NIVO/CABO concentrations not affecting PDL1/killing

**Oxygen/Vascularization**
- ⚠️ **O2 decay currently disabled** (decay rate = 0 in XML) to prevent hypoxia
- ✅ O2 diffusion functional but concentration remains stable
- ✅ Cancer cells consume O2 (but doesn't create hypoxia with decay=0)
- ❌ **No vasculature cell type** - proper implementation requires new agent type
- ❌ **No O2 secretion from vessels** - no vessels to secrete from
- ❌ **No VEGFA-dependent angiogenesis** - VEGFA computed but doesn't affect O2
- 📋 **Future**: Add endothelial cell agent type with VEGFA-guided sprouting and O2 release

### ❌ **Not Implemented / Missing**

**Additional Cell Types**
- ❌ T helper cells (CD4+)
- ❌ Macrophages (M1/M2 polarization)
- ❌ Cancer-associated fibroblasts (CAFs)
- ❌ Dendritic cells
- ❌ B cells / antibodies
- ❌ Endothelial cells (vasculature)

**Key Missing Mechanisms**
- ❌ Immune cell recruitment from lymph/blood (QSP should drive this)
- ❌ T cell-cancer cell killing interactions
- ❌ Checkpoint blockade (PD1-PDL1) effects on killing
- ❌ Immunosuppressive effects (TReg, MDSC, IL10, TGFB)
- ❌ Antigen presentation and priming
- ❌ Metabolic competition (glucose, lactate)
- ❌ ECM remodeling
- ❌ Drug delivery and penetration

**Validation & Testing**
- ❌ Unit tests for PDE solver accuracy
- ❌ Comparison with CPU reference implementation
- ❌ Parameter sensitivity analysis
- ❌ Calibration to experimental data

## Architecture

### Directory Structure
```
PDAC/
├── sim/               # Main simulation entry points
│   ├── main.cu        # Entry point, builds model, initializes QSP/PDE
│   ├── model_definition.cu  # FLAME GPU agent/message definitions
│   ├── model_layers.cu      # Simulation layer execution order
│   ├── initialization.cu    # Agent population initialization
│   ├── CMakeLists.txt       # Build configuration
│   └── resource/
│       └── param_all_test.xml  # Parameter configuration file
│
├── abm/               # GPU parameter system
│   └── gpu_param.h/.cu # Type-safe parameter class, XML loading
│
├── core/              # Core simulation infrastructure
│   ├── common.cuh     # Agent types, enums, constants
│   ├── model_functions.cu/cuh  # QSP-ABM coupling functions
│   └── ParamBase.h/.cpp        # Base parameter class
│
├── agents/            # CUDA device functions for agent behavior
│   ├── cancer_cell.cuh    # ✅ Cancer stem/progenitor dynamics (WORKING)
│   ├── t_cell.cuh         # ⚠️ T cell states (PARTIAL - no killing)
│   ├── t_reg.cuh          # ⚠️ TReg suppression (PARTIAL - not active)
│   └── mdsc.cuh           # ⚠️ MDSC suppression (PARTIAL - not active)
│
├── pde/               # Chemical transport solver
│   ├── pde_solver.cu/cuh      # ✅ Implicit CG solver (WORKING)
│   └── pde_integration.cu/cuh # ✅ PDE-ABM coupling (WORKING)
│
└── qsp/               # QSP model integration (CPU)
    ├── LymphCentral_wrapper.h/.cpp  # ⚠️ CVODE wrapper (PARTIAL)
    ├── qsp_integration.cu            # ⚠️ QSP stepping (PARTIAL)
    ├── cvode/
    │   ├── CVODEBase.h/.cpp  # ✅ SUNDIALS interface (WORKING)
    │   └── MolecularModelCVode.h
    └── ode/
        ├── ODE_system.h/.cpp  # ✅ 153 species ODE (WORKING)
        ├── QSPParam.h/.cpp    # ✅ Parameter loader (WORKING)
        └── QSP_enum.h
```

### Simulation Execution Loop (model_layers.cu)

Each ABM timestep (default 600s = 10 min) executes these layers in order:

1. **Update Agent Counts** - Count populations for QSP
2. **Broadcast Locations** - All agents broadcast (x,y,z) via messages
3. **Scan Neighbors** - Agents count neighbors in 26-voxel Moore neighborhood
4. **Read Chemicals from PDE** - Update agent variables with local concentrations
5. **Update Chemical States** - Agents respond to chemicals (PDL1, hypoxia, etc.)
6. **State Transitions** - Agent state machines (division decisions, death, etc.)
7. **Compute Chemical Sources** - Agents calculate production/consumption rates
8. **Write Sources to PDE** - Aggregate sources via atomic operations
9. **Solve PDE** - Implicit CG solver for diffusion-decay-reaction (1 step, no substeps!)
10. **Division** - Two-phase (intent → execute) for cancer, T cells, TRegs
11. **Solve QSP** - Advance ODE system on CPU

**PDE Solver Details:**
- Backward Euler: `(I + dt·λ - dt·D·∇²)C^(n+1) = C^n + dt·S`
- Conjugate gradient with matrix-free operator application
- 7-point stencil for Laplacian (6 neighbors + center)
- Unconditionally stable → no substeps needed, single solve per ABM step!

### Agent Chemical Interactions (Working Examples)

**Cancer Cells:**
```cpp
// Read from PDE
local_O2 = PDE[CHEM_O2][voxel]
local_IFNg = PDE[CHEM_IFN][voxel]

// Compute sources/sinks
O2_uptake = -1e-3  // Negative = consumption
CCL2_release = 2.56e-13  // Positive = secretion
if (cell_state == STEM) {
    TGFB_release = 1.06e-10
    VEGFA_release = 1.27e-12
}

// Write to PDE
PDE_sources[CHEM_O2][voxel] += O2_uptake  // Direct (already concentration/time)
PDE_sources[CHEM_CCL2][voxel] += CCL2_release / voxel_volume  // Convert units
```

**Unit Conversion Logic:**
- **Sinks (negative)**: Uptake rates already in concentration/time → use as-is
- **Sources (positive)**: Release rates in amount/(cell·time) → divide by voxel_volume

## Build & Run

### Requirements
- CUDA Toolkit 11.0+ (tested with CUDA 12.x)
- CMake 3.18+
- C++17 compiler (g++ 7+)
- SUNDIALS 4.0.1 (for QSP CVODE solver)
- Boost 1.70+ (for serialization)
- FLAME GPU 2 v2.0.0-rc.4 (auto-fetched by CMake)

**Installation Paths (customize in CMakeLists.txt):**
- SUNDIALS: `$HOME/lib/sundials-4.0.1`
- Boost: `$HOME/lib/boost_1_70_0`

### Build Commands
```bash
cd PDAC/sim
./build.sh                    # Release build (~8 min first time)
./build.sh --debug            # Debug build with symbols
./build.sh --clean            # Clean and rebuild
```

**Build Options in CMakeLists.txt:**
- `CMAKE_CUDA_ARCHITECTURES`: Default 75;80;86 (Turing/Ampere/Ada)
- `FLAMEGPU_VERSION`: v2.0.0-rc.4

### Run Commands
```bash
./build/bin/pdac                    # Defaults: 50³ grid, 500 steps
./build/bin/pdac -s 200 -g 50       # 200 steps, 50³ grid
./build/bin/pdac -s 10 -g 11        # Quick test: 10 steps, 11³ grid
./build/bin/pdac -oa 1 -op 1        # Enable agent and PDE output
```

**Command-line Options:**
```
-g, --grid-size N       Grid dimensions (overrides XML, default: 50)
-s, --steps N           ABM steps to run (default: 500)
-r, --radius N          Initial tumor radius in voxels (default: 5)
-t, --tcells N          Initial T cell count (default: 50, non-functional)
--tregs N               Initial TReg count (default: 10, non-functional)
--mdscs N               Initial MDSC count (default: 5, non-functional)
-p, --param-file PATH   XML parameter file (default: resource/param_all_test.xml)
-oa, --output-agents    0=no agent output, 1=output agents (default: 1)
-op, --output-pde       0=no PDE output, 1=output PDE (default: 1)
```

**Output Files:**
- `outputs/abm/agents_step_NNNNNN.csv`: Agent positions, states, properties
- `outputs/pde/pde_step_NNNNNN.csv`: Spatial chemical concentrations

### Typical Run Times
- **Small test** (11³, 10 steps): ~1 minute
- **Medium** (50³, 200 steps): ~10-15 minutes
- **Large** (101³, 500 steps): ~1-2 hours

**First run after build**: Takes 5-10 minutes for CUDA initialization - BE PATIENT!

## Key Parameter Values (param_all_test.xml)

### PDE Chemicals
| Chemical | Diffusivity (cm²/s) | Decay Rate (1/s) | Notes |
|----------|---------------------|------------------|-------|
| O2 | 2.8e-5 | **0** | **DISABLED** - decay off to prevent hypoxia |
| IFN-γ | 1e-7 | 6.5e-5 | Slow diffusion |
| IL-2 | 4e-8 | 2.78e-5 | |
| IL-10 | 1.4e-8 | 4.6e-5 | |
| TGF-β | 2.6e-7 | 1.65e-4 | |
| CCL2 | 1.31e-8 | 1.67e-5 | |
| ArgI | 1e-6 | 2e-6 | |
| NO | 3.8e-5 | 1.56e-3 | Fastest diffusion, fastest decay |
| IL-12 | 2.4e-8 | 6.4e-5 | |
| VEGF-A | 2.9e-7 | 1.921e-4 | |

### Cancer Cell Parameters
- **O2 uptake**: 1e-3 (per cell)
- **IFN-γ uptake**: 1e-3 (per cell)
- **CCL2 release**: 2.56e-13 (all cells)
- **TGF-β release**: 1.06e-10 (stem only), 0 (progenitor)
- **VEGF-A release**: 1.27e-12 (stem and progenitor)
- **Hypoxia threshold**: 0.1 (O2 concentration)
- **Division interval**: ~200 ABM steps (progenitor)
- **Max divisions**: ~10 (before senescence)

### Spatial Parameters
- **Voxel size**: 20 µm (default)
- **Grid size**: 50³ voxels (default) = 1 mm³ tissue
- **ABM timestep**: 600 s = 10 min
- **PDE timestep**: 600 s (1 step per ABM step, implicit!)

## Important Code Locations

### Critical Bug Fixes
**CRITICAL (Feb 11, 2026): Double-Offset Bug Fix**
- **File**: `PDAC/pde/pde_solver.cu` line ~190
- **Issue**: Kernel was adding substrate offset twice (worked only for substrate 0)
- **Fix**: Use `source_idx = voxel_idx` (pointer already offset by `get_device_source_ptr`)
- **Impact**: CCL2, TGFB, VEGFA, IFN-γ now work correctly!

### Key Functions to Understand

**Cancer Cell Behavior (agents/cancer_cell.cuh):**
- `cancer_update_chemicals()`: Read PDE, compute PDL1, detect hypoxia
- `cancer_compute_chemical_sources()`: Set CCL2, TGFB, VEGFA, O2 uptake rates
- `cancer_state_step()`: Division countdown, senescence
- `cancer_execute_divide()`: Create daughter cell, update state

**PDE Integration (pde/pde_integration.cu):**
- `update_agent_chemicals()`: Read PDE → agents (all chemicals, all agent types)
- `collect_agent_sources()`: Agents → PDE (reset, then aggregate all sources)
- `solve_pde_step()`: Call `PDESolver::solve_timestep()`
- `initialize_pde_solver()`: Setup grid, diffusion coeffs, decay rates

**PDE Solver (pde/pde_solver.cu):**
- `solve_timestep()`: Main solve loop (for each substrate, build RHS, run CG)
- `solve_implicit_cg()`: Conjugate gradient with matrix-free operator
- `apply_diffusion_operator()`: Compute A·x = (I + dt·λ - dt·D·∇²)·x
- `add_sources_from_agents()`: Kernel to write agent sources (CRITICAL: fixed Feb 11)

**QSP Integration (qsp/LymphCentral_wrapper.cpp):**
- `initialize()`: Setup ODE system, CVODE solver
- `time_step()`: Advance ODE by dt
- `update_from_abm()`: ⚠️ NOT IMPLEMENTED - should receive ABM data
- `get_state_for_abm()`: ⚠️ NOT IMPLEMENTED - should send QSP data

## Known Issues

### Current Bugs / Limitations

**O2 Dynamics (Currently Disabled):**
- O2 decay rate set to 0 in XML (prevents depletion and hypoxia)
- Configuration choice: hypoxia disabled until vasculature is implemented
- Cancer cells still compute O2 uptake but concentration remains stable
- Proper implementation requires vasculature cell type (future work)

**TGF-β Inverted Gradient:**
- TGF-β lower in tumor center than periphery (backwards!)
- Caused by: progenitor cells have `release=0`, only stem cells secrete
- Few stem cells → low TGF-β in tumor
- Expected given current parameter values

**Immune Cells Non-Functional:**
- T cells, TRegs, MDSCs initialize but don't do anything
- No recruitment, no killing, no suppression
- Just random walk until simulation ends
- Major functionality gap!

### Build Issues

**CUDA 12.6 + FLAME GPU:**
- May fail with CUB template errors
- Workaround: Use CUDA 11.8 or 12.0-12.5

**First Run Hangs:**
- Takes 5-10 minutes for CUDA kernel compilation on first run
- Normal behavior, not a bug!
- Subsequent runs are fast

## Development Workflow

### Adding a New Agent Type
1. Define agent properties in `core/common.cuh`
2. Add agent definition in `model_definition.cu::defineXXXAgent()`
3. Create agent functions in `agents/xxx.cuh`
4. Add layers in `model_layers.cu::defineMainModelLayers()`
5. Initialize population in `initialization.cu::initializeAllAgents()`
6. Add parameters to XML and `gpu_param.h`

### Adding a New Chemical
1. Add enum to `pde/pde_solver.cuh::ChemicalType`
2. Add diffusivity/decay to `pde/pde_integration.cu::initialize_pde_solver()`
3. Add XML parameters to `param_all_test.xml::Molecular::biofvm`
4. Add collection in `pde_integration.cu::collect_agent_sources()`
5. Add reading in `pde_integration.cu::update_agent_chemicals()`

### Modifying Parameters
- Edit `resource/param_all_test.xml`
- Rebuild only if adding NEW parameters (need to update `gpu_param.h`)
- Otherwise, just re-run (XML loaded at runtime)

## Next Development Priorities

### High Priority (Core Functionality)
1. **Implement immune cell recruitment** from QSP model
2. **Implement T cell killing** of cancer cells
3. **Connect QSP↔ABM coupling** (species indices, data transfer)
4. **Implement immunosuppression** (TReg, MDSC effects on T cells)

### Medium Priority (Biological Realism)
5. **Add vasculature cell type** (endothelial cells, VEGFA-guided angiogenesis, O2 release)
6. Add macrophages (M1/M2 polarization)
7. Add T helper cells (CD4+)
8. Implement checkpoint blockade (PD1-PDL1 effects)
9. Add metabolic competition (glucose, lactate)
10. Calibrate to experimental data

### Low Priority (Validation & Optimization)
11. Unit tests for PDE solver
12. Performance profiling and optimization
13. Parameter sensitivity analysis
14. Documentation and tutorials

## Debugging Tips

```bash
# Quick iteration with small grid
./build/bin/pdac -g 11 -s 5 -oa 1 -op 1

# Enable CUDA error checking (slower but catches issues)
export CUDA_LAUNCH_BLOCKING=1

# Check parameter loading
./build/bin/pdac 2>&1 | grep -i "parameters loaded"

# Monitor chemical evolution
tail -f outputs/pde/pde_step_*.csv

# Check agent counts over time
for f in outputs/abm/agents_step_*.csv; do
    echo -n "$f: "; tail -n +2 "$f" | wc -l
done

# Visualize PDE in Python
cd python
jupyter notebook SimpleABMPDEvis.ipynb
```

**Common Issues:**
- **All zeros in PDE**: Check source collection is enabled, verify agents exist
- **Simulation hangs**: First run takes 5-10 min (CUDA init), subsequent runs fast
- **Out of memory**: Reduce grid size or number of agents
- **Slow simulation**: Use smaller grid (`-g 21`) or fewer steps (`-s 10`)

## References

- **FLAME GPU 2**: https://github.com/FLAMEGPU/FLAMEGPU2
- **SUNDIALS**: https://computing.llnl.gov/projects/sundials
- **BioFVM**: http://biofvm.org (inspiration for PDE solver)
- **PhysiCell**: http://physicell.org (agent-based model reference)
