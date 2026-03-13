#include "pde_integration.cuh"
#include "../core/common.cuh"
#include "../core/layer_timing.h"
#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <nvtx3/nvToolsExt.h>
#include <cmath>

// Helper: sample normal random variate via Box-Muller (matches HCC getTCellLife / getTCD4Life)
// Returns mean + N(0,1)*sd, clamped to at least 1
static inline int sample_normal_life(float mean, float sd) {
    float u1 = (static_cast<float>(rand()) + 1.0f) / (static_cast<float>(RAND_MAX) + 2.0f);
    float u2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float z  = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    int life = static_cast<int>(mean + z * sd + 0.5f);
    return life > 0 ? life : 1;
}

// ============================================================================
// Layer Timing Globals (defined once here, declared extern in layer_timing.h)
// ============================================================================
namespace PDAC {
    std::vector<LayerTime> g_layer_timings;
    ClockPoint g_checkpoint_t = std::chrono::high_resolution_clock::now();
} // namespace PDAC

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

namespace PDAC {

// ============================================================================
// Global PDE solver instance
// ============================================================================
PDESolver* g_pde_solver = nullptr;
double g_last_pde_ms = 0.0;  // exposed for timing CSV

static RecruitStats g_recruit_stats;

RecruitStats get_last_recruit_stats() { return g_recruit_stats; }

// Flat device array: cancer occupancy per voxel (0 = empty, >0 = cancer present).
// Populated by cancer_write_to_occ_grid, zeroed by zero_occupancy_grid.
// Used by recruitment source-marking kernels to skip tumor-dense voxels.
static unsigned int* d_cancer_occ = nullptr;

// Flat device arrays for ECM (extracellular matrix) and fibroblast density field.
// Replace MacroProperty-based approach to eliminate D2H/H2D copies every step.
// Layout: idx = z * (nx * ny) + y * nx + x  (z-major, x-minor; matches PDE convention)
static float* d_ecm_grid = nullptr;
static float* d_fib_density_field = nullptr;

// ============================================================================
// CUDA Kernel: ECM Grid Update
// Applies decay + fibroblast deposition + saturation clamping per voxel in parallel.
// Called from update_ecm_grid host function after fib_build_density_field runs.
// ============================================================================
__global__ void update_ecm_grid_kernel(
    float* ecm, const float* fib_field,
    int nx, int ny, int nz,
    float voxel_vol_cm3, float decay_rate, float dt,
    float ecm_baseline, float ecm_saturation, float release_rate)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;

    const int idx = tz * (nx * ny) + ty * nx + tx;

    float curr_ecm     = ecm[idx];
    float curr_ecm_amt = curr_ecm * voxel_vol_cm3;

    // Exponential decay
    float decayed = curr_ecm_amt * expf(-decay_rate * dt);

    // Deposition from fibroblast density field (CAF-weighted Gaussian scatter)
    float saturation  = fminf(curr_ecm / ecm_saturation, 1.0f);
    float deposition  = fib_field[idx] * release_rate / 3.0f * (1.0f - saturation);

    float new_ecm = (decayed + deposition) / voxel_vol_cm3;

    // Clamp to [baseline, saturation]
    new_ecm = fmaxf(new_ecm, ecm_baseline);
    new_ecm = fminf(new_ecm, ecm_saturation);

    ecm[idx] = new_ecm;
}

// ============================================================================
// Host Function: Reset PDE Buffers (call before compute_chemical_sources)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(reset_pde_buffers) {
    nvtxRangePush("Reset PDE Buffers");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->reset_sources();
    g_pde_solver->reset_uptakes();
    nvtxRangePop();
}

// ============================================================================
// Host Function: Solve PDE
// ============================================================================

FLAMEGPU_HOST_FUNCTION(solve_pde_step) {
    nvtxRangePush("PDE Solve");
    if (!g_pde_solver) {
        nvtxRangePop();
        return;
    }

    int substeps = FLAMEGPU->environment.getProperty<int>("PARAM_MOLECULAR_STEPS");
    auto pde_t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < substeps; i++) {
        g_pde_solver->solve_timestep();
    }
    auto pde_t1 = std::chrono::high_resolution_clock::now();
    g_last_pde_ms = std::chrono::duration<double, std::milli>(pde_t1 - pde_t0).count();

    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "PDE solved for step " << step << std::endl;
    }
    nvtxRangePop();
}

// ============================================================================
// Host Function: Compute PDE Gradients (call after solve_pde_step)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(compute_pde_gradients) {
    nvtxRangePush("Compute PDE Gradients");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->compute_gradients();
    nvtxRangePop();
}

// ============================================================================
// Initialize/Cleanup
// ============================================================================

void initialize_pde_solver(int grid_x, int grid_y, int grid_z,
                           float voxel_size, float dt_abm, int molecular_steps,
                            const PDAC::GPUParam& gpu_params) {
    PDEConfig config;
    config.nx = grid_x;
    config.ny = grid_y;
    config.nz = grid_z;
    config.num_substrates = NUM_SUBSTRATES;
    config.voxel_size = voxel_size * 1.0e-4f;  // Convert µm to cm
    config.dt_abm = dt_abm;
    config.dt_pde = dt_abm / molecular_steps;
    config.substeps_per_abm = molecular_steps;
    config.boundary_type = 0;  // Neumann (no-flux)
    
    // Set diffusion coefficients (cm²/s) from params file
    config.diffusion_coeffs[CHEM_O2]    = gpu_params.getFloat(PARAM_O2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IFN]   = gpu_params.getFloat(PARAM_IFNG_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL2]   = gpu_params.getFloat(PARAM_IL2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL10]  = gpu_params.getFloat(PARAM_IL10_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_TGFB]  = gpu_params.getFloat(PARAM_TGFB_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_CCL2]  = gpu_params.getFloat(PARAM_CCL2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_ARGI]  = gpu_params.getFloat(PARAM_ARGI_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_NO]    = gpu_params.getFloat(PARAM_NO_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL12]  = gpu_params.getFloat(PARAM_IL12_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_VEGFA] = gpu_params.getFloat(PARAM_VEGFA_DIFFUSIVITY);

    // Set decay rates (1/s) from params file
    config.decay_rates[CHEM_O2]    = gpu_params.getFloat(PARAM_O2_DECAY_RATE);
    config.decay_rates[CHEM_IFN]   = gpu_params.getFloat(PARAM_IFNG_DECAY_RATE);
    config.decay_rates[CHEM_IL2]   = gpu_params.getFloat(PARAM_IL2_DECAY_RATE);
    config.decay_rates[CHEM_IL10]  = gpu_params.getFloat(PARAM_IL10_DECAY_RATE);
    config.decay_rates[CHEM_TGFB]  = gpu_params.getFloat(PARAM_TGFB_DECAY_RATE);
    config.decay_rates[CHEM_CCL2]  = gpu_params.getFloat(PARAM_CCL2_DECAY_RATE);
    config.decay_rates[CHEM_ARGI]  = gpu_params.getFloat(PARAM_ARGI_DECAY_RATE);
    config.decay_rates[CHEM_NO]    = gpu_params.getFloat(PARAM_NO_DECAY_RATE);
    config.decay_rates[CHEM_IL12]  = gpu_params.getFloat(PARAM_IL12_DECAY_RATE);
    config.decay_rates[CHEM_VEGFA] = gpu_params.getFloat(PARAM_VEGFA_DECAY_RATE);
    
    g_pde_solver = new PDESolver(config);
    g_pde_solver->initialize();

    // Set initial O2 concentration, all others start at 0.0
    g_pde_solver->set_initial_concentration(CHEM_O2, 0.673);  // Oxygen starts at 0.673 (amount/mL)

    // Allocate flat cancer occupancy array for recruitment density checks
    int total_voxels = g_pde_solver->get_total_voxels();
    CUDA_CHECK(cudaMalloc(&d_cancer_occ, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_cancer_occ, 0, total_voxels * sizeof(unsigned int)));

    // Allocate ECM and fibroblast density field device arrays.
    // Initialized to 0.0f; update_ecm_grid_kernel clamps to ecm_baseline on first call.
    CUDA_CHECK(cudaMalloc(&d_ecm_grid, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_grid, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fib_density_field, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(float)));

    std::cout << "PDE Solver initialized and coupled to FLAME GPU 2" << std::endl;
}

// Call this after model initialization but before simulation starts
void set_pde_pointers_in_environment(flamegpu::ModelDescription& model) {
    if (!g_pde_solver) {
        std::cerr << "[ERROR] g_pde_solver is NULL!" << std::endl;
        return;
    }

    std::cout << "[DEBUG] Storing PDE device pointers in environment..." << std::endl;

    // Store device pointers as unsigned long long (can be cast back to float*)
    for (int sub = 0; sub < NUM_SUBSTRATES; sub++) {
        std::string concentration_key = "pde_concentration_ptr_" + std::to_string(sub);
        std::string source_key = "pde_source_ptr_" + std::to_string(sub);
        std::string uptake_key = "pde_uptake_ptr_" + std::to_string(sub);

        uintptr_t conc_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_concentration_ptr(sub));
        uintptr_t src_ptr  = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_source_ptr(sub));
        uintptr_t upt_ptr  = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_uptake_ptr(sub));

        model.Environment().newProperty<unsigned long long>(concentration_key, static_cast<unsigned long long>(conc_ptr));
        model.Environment().newProperty<unsigned long long>(source_key,        static_cast<unsigned long long>(src_ptr));
        model.Environment().newProperty<unsigned long long>(uptake_key,        static_cast<unsigned long long>(upt_ptr));
    }

    // Store gradient pointers for chemotaxis substrates (IFN=0, TGFB=1, CCL2=2, VEGFA=3)
    // Naming: pde_grad_IFN_x, pde_grad_IFN_y, pde_grad_IFN_z, pde_grad_TGFB_x, ...
    static const char* grad_names[NUM_GRAD_SUBSTRATES] = {"IFN", "TGFB", "CCL2", "VEGFA"};
    for (int g = 0; g < NUM_GRAD_SUBSTRATES; g++) {
        model.Environment().newProperty<unsigned long long>(
            std::string("pde_grad_") + grad_names[g] + "_x",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_pde_solver->get_device_gradx_ptr(g))));
        model.Environment().newProperty<unsigned long long>(
            std::string("pde_grad_") + grad_names[g] + "_y",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_pde_solver->get_device_grady_ptr(g))));
        model.Environment().newProperty<unsigned long long>(
            std::string("pde_grad_") + grad_names[g] + "_z",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_pde_solver->get_device_gradz_ptr(g))));
    }

    // Store recruitment sources pointer
    uintptr_t recruit_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_recruitment_sources_ptr());
    model.Environment().newProperty<unsigned long long>("pde_recruitment_sources_ptr", static_cast<unsigned long long>(recruit_ptr));

    // Store flat cancer occupancy pointer (for recruitment density checks)
    model.Environment().newProperty<unsigned long long>("cancer_occ_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_cancer_occ)));

    // Store ECM and fibroblast density field pointers (replace MacroProperty approach)
    model.Environment().newProperty<unsigned long long>("ecm_grid_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_grid)));
    model.Environment().newProperty<unsigned long long>("fib_density_field_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_fib_density_field)));

    std::cout << "PDE device pointers stored in FLAME GPU environment" << std::endl;
}

void cleanup_pde_solver() {
    if (g_pde_solver) {
        delete g_pde_solver;
        g_pde_solver = nullptr;
    }
    if (d_cancer_occ) {
        cudaFree(d_cancer_occ);
        d_cancer_occ = nullptr;
    }
    if (d_ecm_grid) {
        cudaFree(d_ecm_grid);
        d_ecm_grid = nullptr;
    }
    if (d_fib_density_field) {
        cudaFree(d_fib_density_field);
        d_fib_density_field = nullptr;
    }
}

// ============================================================================
// Recruitment System Implementation
// ============================================================================

// Helper: check if radius-3 sphere around (x,y,z) is completely filled with cancer.
// Returns true if every in-bounds voxel within radius 3 has cancer present.
__device__ bool is_tumor_dense_r3(
    const unsigned int* d_cancer_occ,
    int x, int y, int z,
    int nx, int ny, int nz)
{
    int sphere_total = 0, sphere_cancer = 0;
    for (int dz = -3; dz <= 3; dz++) {
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                if (dx*dx + dy*dy + dz*dz > 9) continue;
                int cx = x + dx, cy = y + dy, cz = z + dz;
                if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;
                sphere_total++;
                sphere_cancer += (d_cancer_occ[cz*(nx*ny) + cy*nx + cx] > 0u) ? 1 : 0;
            }
        }
    }
    return (sphere_total > 0 && sphere_cancer >= sphere_total);
}

// CUDA kernel to mark MDSC recruitment sources based on CCL2
__global__ void mark_mdsc_sources_kernel(
    int* d_recruitment_sources,
    const float* d_ccl2,
    const unsigned int* d_cancer_occ,
    int nx, int ny, int nz,
    float ec50_ccl2,
    unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Skip voxels where radius-3 sphere is completely filled with cancer
    if (is_tumor_dense_r3(d_cancer_occ, x, y, z, nx, ny, nz)) return;

    int idx = z * (nx * ny) + y * nx + x;

    float ccl2 = d_ccl2[idx];
    float H_CCL2 = ccl2 / (ccl2 + ec50_ccl2);

    // Simple random number generation (thread-local)
    unsigned int rng_state = seed + idx;
    rng_state = rng_state * 1103515245u + 12345u;
    float rand_val = (rng_state & 0x7FFFFFFF) / float(0x7FFFFFFF);

    if (rand_val < H_CCL2) {
        atomicOr(&d_recruitment_sources[idx], 2);  // Set MDSC bit (bit 1)
    }
}

// CUDA kernel to mark macrophage recruitment sources based on CCL2
__global__ void mark_mac_sources_kernel(
    int* d_recruitment_sources,
    const float* d_ccl2,
    const unsigned int* d_cancer_occ,
    int nx, int ny, int nz,
    float ec50_ccl2,
    unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Skip voxels where radius-3 sphere is completely filled with cancer
    if (is_tumor_dense_r3(d_cancer_occ, x, y, z, nx, ny, nz)) return;

    int idx = z * (nx * ny) + y * nx + x;

    float ccl2 = d_ccl2[idx];
    float H_CCL2 = ccl2 / (ccl2 + ec50_ccl2);

    // Simple random number generation (thread-local)
    unsigned int rng_state = seed + idx;
    rng_state = rng_state * 1103515245u + 12345u;
    float rand_val = (rng_state & 0x7FFFFFFF) / float(0x7FFFFFFF);

    if (rand_val < H_CCL2) {
        atomicOr(&d_recruitment_sources[idx], 4);  // Set macrophage bit (bit 2)
    }
}

// Update vasculature count env property (used by vascular_mark_t_sources device function)
FLAMEGPU_HOST_FUNCTION(update_vasculature_count) {
    nvtxRangePush("Update Vas Count");
    int n_vas = static_cast<int>(FLAMEGPU->agent(AGENT_VASCULAR).count());
    FLAMEGPU->environment.setProperty<int>("n_vasculature_total", std::max(1, n_vas));
    nvtxRangePop();
}

// Reset recruitment sources at start of each step
FLAMEGPU_HOST_FUNCTION(reset_recruitment_sources) {
    nvtxRangePush("Reset Recruit Sources");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->reset_recruitment_sources();
    nvtxRangePop();
}

// Mark MDSC sources based on CCL2 concentration
FLAMEGPU_HOST_FUNCTION(mark_mdsc_sources) {
    nvtxRangePush("Mark MDSC Sources");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    int* d_recruitment_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    const float* d_ccl2 = g_pde_solver->get_device_concentration_ptr(CHEM_CCL2);

    // Get parameter
    float ec50_ccl2 = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_EC50_CCL2_REC");

    // Generate random seed from step number
    unsigned int seed = static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 12345u;

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mdsc_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, d_cancer_occ, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
    nvtxRangePop();
}

// Recruit T cells at marked T source voxels
FLAMEGPU_HOST_FUNCTION(recruit_t_cells) {
    nvtxRangePush("Recruit T Cells");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    // Get QSP concentrations and recruitment rates from environment
    float qsp_teff_conc = FLAMEGPU->environment.getProperty<float>("qsp_teff_central");
    float k_teff = FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_RECRUIT_K");
    float p_recruit_teff = std::min(qsp_teff_conc * k_teff, 1.0f);

    float qsp_treg_conc = FLAMEGPU->environment.getProperty<float>("qsp_treg_central");
    float k_treg = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_RECRUIT_K");
    float p_recruit_treg = std::min(qsp_treg_conc * k_treg, 1.0f);

    float qsp_th_conc = FLAMEGPU->environment.getProperty<float>("qsp_th_central");
    float k_th = FLAMEGPU->environment.getProperty<float>("PARAM_TH_RECRUIT_K");
    float p_recruit_th = std::min(qsp_th_conc * k_th, 1.0f);

    // Copy recruitment sources to host
    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent APIs for creating new agents
    auto tcell_api = FLAMEGPU->agent(AGENT_TCELL);
    auto treg_api = FLAMEGPU->agent(AGENT_TREG);

    int teff_recruited = 0;
    int treg_recruited = 0;
    int th_recruited = 0;

    // Count total sources available
    int total_t_sources = 0;
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 1) != 0) total_t_sources++;
    }

    // Scan for T source voxels (bit 0 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 1) == 0) continue;  // Not a T source

        int z = idx / (nx * ny);
        int y = (idx % (nx * ny)) / nx;
        int x = idx % nx;

        // Try to recruit Teff
        if (FLAMEGPU->random.uniform<float>() < p_recruit_teff) {
            // Find empty neighbor voxel for placement (Moore neighborhood)
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            // Create new T cell (simplified initialization)
                            auto new_agent = tcell_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);
                            new_agent.setVariable<int>("cell_state", 0);  // Effector state

                            float lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
                            float lifeSd   = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD");

                            new_agent.setVariable<int>("life", sample_normal_life(lifeMean, lifeSd));

                            new_agent.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL"));
                            new_agent.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_LIMIT"));

                            new_agent.setVariable<float>("IL2_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME"));

                            new_agent.setVariable<int>("tumble", 1);

                            teff_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }

        // Try to recruit Treg
        if (FLAMEGPU->random.uniform<float>() < p_recruit_treg) {
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = treg_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);
                            new_agent.setVariable<int>("cell_state", TCD4_TREG);

                            float lifeMean_treg = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
                            float lifeSd_treg   = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFESPAN_SD");

                            new_agent.setVariable<int>("life", sample_normal_life(lifeMean_treg, lifeSd_treg));

                            new_agent.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_INTERNAL"));
                            new_agent.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_LIMIT"));

                            new_agent.setVariable<float>("TGFB_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME"));

                            new_agent.setVariable<float>("CTLA4", FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG"));

                            new_agent.setVariable<int>("tumble", 1);

                            treg_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
        // Try to recruit TH
        if (FLAMEGPU->random.uniform<float>() < p_recruit_th) {
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent_th = treg_api.newAgent();
                            new_agent_th.setVariable<int>("x", nx_new);
                            new_agent_th.setVariable<int>("y", ny_new);
                            new_agent_th.setVariable<int>("z", nz_new);
                            new_agent_th.setVariable<int>("cell_state", TCD4_TH);

                            float lifeMean_th = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
                            float lifeSd_th   = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFESPAN_SD");

                            new_agent_th.setVariable<int>("life", sample_normal_life(lifeMean_th, lifeSd_th));

                            new_agent_th.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_INTERNAL"));
                            new_agent_th.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_LIMIT"));

                            new_agent_th.setVariable<float>("TGFB_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME"));

                            new_agent_th.setVariable<float>("CTLA4", 0.0f);

                            new_agent_th.setVariable<int>("tumble", 1);

                            th_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    // Stash stats in global for main.cu to read after step()
    g_recruit_stats.teff_rec  = teff_recruited;
    g_recruit_stats.treg_rec  = treg_recruited;
    g_recruit_stats.th_rec    = th_recruited;
    g_recruit_stats.p_teff    = p_recruit_teff;
    g_recruit_stats.p_treg    = p_recruit_treg;
    g_recruit_stats.p_th      = p_recruit_th;
    g_recruit_stats.t_sources = total_t_sources;
    g_recruit_stats.qsp_teff  = FLAMEGPU->environment.getProperty<float>("qsp_teff_central");
    g_recruit_stats.qsp_treg  = FLAMEGPU->environment.getProperty<float>("qsp_treg_central");
    g_recruit_stats.qsp_th    = FLAMEGPU->environment.getProperty<float>("qsp_th_central");

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_TEFF_REC] += teff_recruited;
    counters[ABM_COUNT_TH_REC] += th_recruited;
    counters[ABM_COUNT_TREG_REC] += treg_recruited;

    // Track events: T cell recruitment (CD8 effector) and TREG recruitment
    uint64_t tcell_recruit_ptr = FLAMEGPU->environment.getProperty<uint64_t>("event_tcell_recruit_ptr");
    if (tcell_recruit_ptr != 0 && teff_recruited > 0) {
        unsigned int host_val;
        cudaMemcpy(&host_val, reinterpret_cast<unsigned int*>(tcell_recruit_ptr), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        host_val += teff_recruited;
        cudaMemcpy(reinterpret_cast<unsigned int*>(tcell_recruit_ptr), &host_val, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    // Track TREG recruitment (TREGs are recruited as TH cells, so count as TH recruitment)
    uint64_t th_recruit_ptr = FLAMEGPU->environment.getProperty<uint64_t>("event_th_recruit_ptr");
    if (th_recruit_ptr != 0 && treg_recruited > 0) {
        unsigned int host_val;
        cudaMemcpy(&host_val, reinterpret_cast<unsigned int*>(th_recruit_ptr), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        host_val += treg_recruited;
        cudaMemcpy(reinterpret_cast<unsigned int*>(th_recruit_ptr), &host_val, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    nvtxRangePop();
}

// Recruit MDSCs at marked MDSC source voxels
FLAMEGPU_HOST_FUNCTION(recruit_mdscs) {
    nvtxRangePush("Recruit MDSCs");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Calculate recruitment probability: p = min(concentration * k_recruit, 1.0)
    float p_recruit_mdsc = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_RECRUIT_K");

    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent API for creating new agents
    auto mdsc_api = FLAMEGPU->agent(AGENT_MDSC);
    int mdsc_recruited = 0;

    // Scan for MDSC source voxels (bit 1 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 2) == 0) continue;  // Not an MDSC source

        if (FLAMEGPU->random.uniform<float>() < p_recruit_mdsc) {
            int z = idx / (nx * ny);
            int y = (idx % (nx * ny)) / nx;
            int x = idx % nx;

            // Find empty neighbor voxel
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = mdsc_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);

                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);
                            new_agent.setVariable<int>("tumble", 1);

                            mdsc_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    g_recruit_stats.mdsc_rec = mdsc_recruited;

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_MDSC_REC] += mdsc_recruited;
    nvtxRangePop();
}

// Mark macrophage sources based on CCL2 concentration
FLAMEGPU_HOST_FUNCTION(mark_mac_sources) {
    nvtxRangePush("Mark MAC Sources");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    int* d_recruitment_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    const float* d_ccl2 = g_pde_solver->get_device_concentration_ptr(CHEM_CCL2);

    // Get parameter
    float ec50_ccl2 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_EC50_CCL2_REC");

    // Generate random seed from step number
    unsigned int seed = static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 12345u + 54321u;

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mac_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, d_cancer_occ, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
    nvtxRangePop();
}

// Recruit macrophages at marked macrophage source voxels
FLAMEGPU_HOST_FUNCTION(recruit_macrophages) {
    nvtxRangePush("Recruit MACs");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Calculate recruitment probability
    float p_recruit_mac = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_RECRUIT_K");

    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent API for creating new agents
    auto mac_api = FLAMEGPU->agent(AGENT_MACROPHAGE);
    int mac_recruited = 0;

    // Scan for macrophage source voxels (bit 2 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 4) == 0) continue;  // Not a macrophage source

        if (FLAMEGPU->random.uniform<float>() < p_recruit_mac) {
            int z = idx / (nx * ny);
            int y = (idx % (nx * ny)) / nx;
            int x = idx % nx;

            // Find empty neighbor voxel
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = mac_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);

                            // Recruit as M1 state
                            int cell_state = MAC_M1;

                            // 30% chance to become M2
                            if (FLAMEGPU->random.uniform<float>() < 0.3f) {
                                cell_state = MAC_M2;
                            }
                            new_agent.setVariable<int>("cell_state", cell_state);

                            // Set lifespan
                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_LIFE_MEAN");
                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);
                            new_agent.setVariable<int>("tumble", 1);

                            mac_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    g_recruit_stats.mac_rec = mac_recruited;

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_MAC_REC] += mac_recruited;
    nvtxRangePop();
}

// ============================================================================
// Occupancy Grid: Zero the grid at the start of each step's division phase
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_occupancy_grid) {
    nvtxRangePush("Zero Occ Grid");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ.zero();

    // Also reset the flat cancer occupancy array used by recruitment kernels
    if (d_cancer_occ && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        cudaMemset(d_cancer_occ, 0, total_voxels * sizeof(unsigned int));
    }
    nvtxRangePop();
}

// ============================================================================
// Zero Fibroblast Density Field (reset before scatter)
// Uses cudaMemset on flat device array — no D2H/H2D copy.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_fib_density_field) {
    nvtxRangePush("Zero Fib Density");
    if (d_fib_density_field && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(float));
    }
    nvtxRangePop();
}

// ============================================================================
// ECM Grid: Apply decay, deposition from fibroblast density field, and clamp.
// Replaces the CPU triple-nested loop with a GPU kernel launch.
// No MacroProperty D2H/H2D — operates entirely on device arrays.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(update_ecm_grid) {
    nvtxRangePush("Update ECM Grid");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    float voxel_size_cm  = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_SIZE_CM");
    float decay_rate     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_DECAY_RATE");
    float ecm_baseline   = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_BASELINE");
    float ecm_saturation = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_SATURATION");
    float release_rate   = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_RELEASE_CAF");
    float dt_sec         = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    float dt             = dt_sec / 86400.0f;  // seconds → days
    float voxel_vol_cm3  = voxel_size_cm * voxel_size_cm * voxel_size_cm;

    dim3 block(8, 8, 8);
    dim3 grid((grid_x + 7) / 8, (grid_y + 7) / 8, (grid_z + 7) / 8);
    update_ecm_grid_kernel<<<grid, block>>>(
        d_ecm_grid, d_fib_density_field,
        grid_x, grid_y, grid_z,
        voxel_vol_cm3, decay_rate, dt,
        ecm_baseline, ecm_saturation, release_rate);
    cudaDeviceSynchronize();

    nvtxRangePop();
}

// ============================================================================
// Aggregate ABM Event Counters from Agent States
// Counts cancer cell deaths by cause from agents marked as dead
// ============================================================================
FLAMEGPU_HOST_FUNCTION(aggregate_abm_events) {
    nvtxRangePush("Aggregate ABM Events");
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    auto cc_api = FLAMEGPU->agent("CancerCell");

    // Get population data for iteration
    flamegpu::DeviceAgentVector cc_agents = cc_api.getPopulationData();
    const unsigned int cc_count = cc_agents.size();

    // Count cancer cell deaths by cause from dead agents
    int cc_death_total = 0;
    int cc_death_natural = 0;
    int cc_death_t_kill = 0;
    int cc_death_mac_kill = 0;

    for (unsigned int i = 0; i < cc_count; i++) {
        if (cc_agents[i].getVariable<int>("dead") != 0) {
            cc_death_total++;
            int reason = cc_agents[i].getVariable<int>("death_reason");
            if (reason == 0) {
                cc_death_natural++;
            } else if (reason == 1) {
                cc_death_t_kill++;
            } else if (reason == 2) {
                cc_death_mac_kill++;
            }
        }
    }

    // Update counter MacroProperty with aggregated values
    counters[ABM_COUNT_CC_DEATH] = cc_death_total;
    counters[ABM_COUNT_CC_DEATH_NATURAL] = cc_death_natural;
    counters[ABM_COUNT_CC_DEATH_T_KILL] = cc_death_t_kill;
    counters[ABM_COUNT_CC_DEATH_MAC_KILL] = cc_death_mac_kill;
    nvtxRangePop();
}

// ============================================================================
// Copy ABM Event Counters from MacroProperty to Environment Properties
// Called BEFORE QSP so the ODE model can read accumulated counts this step
// ============================================================================
FLAMEGPU_HOST_FUNCTION(copy_abm_counters_to_environment) {
    nvtxRangePush("Copy ABM Counters");
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");

    // Copy from MacroProperty array to environment properties for QSP access
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death", static_cast<int>(counters[ABM_COUNT_CC_DEATH]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_t_kill", static_cast<int>(counters[ABM_COUNT_CC_DEATH_T_KILL]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_mac_kill", static_cast<int>(counters[ABM_COUNT_CC_DEATH_MAC_KILL]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_natural", static_cast<int>(counters[ABM_COUNT_CC_DEATH_NATURAL]));
    FLAMEGPU->environment.setProperty<int>("ABM_TEFF_REC", static_cast<int>(counters[ABM_COUNT_TEFF_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_TH_REC", static_cast<int>(counters[ABM_COUNT_TH_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_TREG_REC", static_cast<int>(counters[ABM_COUNT_TREG_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_MDSC_REC", static_cast<int>(counters[ABM_COUNT_MDSC_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_MAC_REC", static_cast<int>(counters[ABM_COUNT_MAC_REC]));
    nvtxRangePop();
}

// ============================================================================
// Reset ABM → QSP Event Counters (called at END of each step)
// Clears MacroProperty array for next step's accumulation
// ============================================================================
FLAMEGPU_HOST_FUNCTION(reset_abm_event_counters) {
    nvtxRangePush("Reset ABM Counters");
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");

    // Reset all counter elements to zero
    for (int i = 0; i < ABM_EVENT_COUNTER_SIZE; i++) {
        counters[i] = 0;
    }
    nvtxRangePop();
}

// ============================================================================
// Fibroblast Activation/Expansion: Convert chain to CAF and extend at back.
//
// Matches HCC Fib::agent_state_step + Tumor.cpp expansion logic:
//   - Only the TRUE BACK of a chain (no other cell follows it) can expand.
//   - On activation: back cell + 2 new cells become CAF; entire chain → CAF.
//   - New cells are added behind the back cell (away from the chemotaxis sensor).
//
// Three-phase approach avoids the old crash (DeviceAgentVector + newAgent() conflict):
//   Phase 1: Read all fib data into host memory → close DeviceAgentVector.
//   Phase 2: Create new agents via fib_api.newAgent() (no open vector).
//   Phase 3: Re-open DeviceAgentVector to update cell_state + reset divide_flag.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(fib_execute_divide) {
    nvtxRangePush("Fib Execute Divide");
    auto fib_api = FLAMEGPU->agent(AGENT_FIBROBLAST);
    const unsigned int fib_count = fib_api.count();
    if (fib_count == 0) { nvtxRangePop(); return; }

    const int grid_x    = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y    = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z    = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const float mean_life = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_LIFE_MEAN");

    // --- Phase 1: Read all fib data into host-side struct ---
    struct FibData {
        int x, y, z;
        int my_slot, leader_slot;
        int cell_state, divide_flag;
    };
    std::vector<FibData> all_fibs(fib_count);

    {   // Scoped block: DeviceAgentVector is released before Phase 2
        flamegpu::DeviceAgentVector fib_vec = fib_api.getPopulationData();
        for (unsigned int i = 0; i < fib_count; i++) {
            all_fibs[i].x           = fib_vec[i].getVariable<int>("x");
            all_fibs[i].y           = fib_vec[i].getVariable<int>("y");
            all_fibs[i].z           = fib_vec[i].getVariable<int>("z");
            all_fibs[i].my_slot     = fib_vec[i].getVariable<int>("my_slot");
            all_fibs[i].leader_slot = fib_vec[i].getVariable<int>("leader_slot");
            all_fibs[i].cell_state  = fib_vec[i].getVariable<int>("cell_state");
            all_fibs[i].divide_flag = fib_vec[i].getVariable<int>("divide_flag");
        }
    }   // fib_vec destroyed: any pending writes committed to device

    // --- Build auxiliary data structures ---
    std::unordered_map<int, unsigned int> slot_to_idx;  // my_slot → index in all_fibs
    std::unordered_set<int> all_leader_slots;            // Every leader_slot value in use
    int max_slot = -1;

    for (unsigned int i = 0; i < fib_count; i++) {
        int ms = all_fibs[i].my_slot;
        if (ms >= 0) {
            slot_to_idx[ms] = i;
            if (ms > max_slot) max_slot = ms;
        }
        int ls = all_fibs[i].leader_slot;
        if (ls >= 0) all_leader_slots.insert(ls);
    }
    int next_slot = max_slot + 1;

    auto occ      = FLAMEGPU->environment.getMacroProperty<unsigned int,
                        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    auto fib_pos_x = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_x");
    auto fib_pos_y = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_y");
    auto fib_pos_z = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_z");

    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};

    // Tracks positions claimed this function call (prevent double-placement)
    std::set<std::tuple<int,int,int>> claimed;

    struct ExpansionPlan {
        int n_new;                       // 0, 1, or 2 new cells
        int new_x[2], new_y[2], new_z[2];
        int new_slot[2];
        std::vector<int> chain_slots;    // my_slot of every cell in chain → set to CAF
    };
    std::vector<ExpansionPlan> plans;

    for (unsigned int i = 0; i < fib_count; i++) {
        const auto& fib = all_fibs[i];

        // Must have valid slot, be NORMAL, and have divide_flag set
        if (fib.my_slot < 0 || fib.cell_state != FIB_NORMAL || fib.divide_flag == 0) continue;

        // Must be TRUE BACK of chain: no other cell has leader_slot == my_slot
        if (all_leader_slots.count(fib.my_slot) > 0) continue;

        // Guard slot capacity
        if (next_slot + 1 >= MAX_FIB_SLOTS) continue;

        // --- Find up to 2 adjacent free face-neighbor voxels ---
        int new_x[2], new_y[2], new_z[2];
        int n_found = 0;

        for (int d = 0; d < 6 && n_found < 1; d++) {
            int nx = fib.x + dx6[d], ny = fib.y + dy6[d], nz = fib.z + dz6[d];
            if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
            if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_FIB])    > 0u) continue;
            if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_CANCER]) > 0u) continue;
            if (claimed.count({nx, ny, nz})) continue;
            new_x[0] = nx; new_y[0] = ny; new_z[0] = nz;
            n_found = 1;
        }
        if (n_found == 1) {
            claimed.insert({new_x[0], new_y[0], new_z[0]});
            for (int d = 0; d < 6; d++) {
                int nx = new_x[0]+dx6[d], ny = new_y[0]+dy6[d], nz = new_z[0]+dz6[d];
                if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
                if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_FIB])    > 0u) continue;
                if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_CANCER]) > 0u) continue;
                if (claimed.count({nx, ny, nz})) continue;
                new_x[1] = nx; new_y[1] = ny; new_z[1] = nz;
                n_found = 2;
                claimed.insert({nx, ny, nz});
                break;
            }
        }
        if (n_found == 0) continue;  // No space — skip this activation

        // --- Build expansion plan ---
        ExpansionPlan plan;
        plan.n_new = n_found;
        for (int k = 0; k < n_found; k++) {
            plan.new_x[k] = new_x[k];
            plan.new_y[k] = new_y[k];
            plan.new_z[k] = new_z[k];
        }
        plan.new_slot[0] = next_slot;      // Adjacent to existing back
        plan.new_slot[1] = next_slot + 1;  // Outermost (new back of chain)
        next_slot += n_found;

        // Collect all chain members (from back → front) to convert to CAF
        // Walk leader_slot chain: back cell → middle ... → front (leader_slot==-1)
        plan.chain_slots.push_back(fib.my_slot);
        int follow = fib.leader_slot;
        while (follow >= 0) {
            plan.chain_slots.push_back(follow);
            auto it = slot_to_idx.find(follow);
            if (it == slot_to_idx.end()) break;
            follow = all_fibs[it->second].leader_slot;
        }
        // If chain has a front sensor (leader_slot == -1), it was the last valid slot
        // followed and is already included (the while loop processes it and then stops).

        plans.push_back(std::move(plan));
    }

    if (plans.empty()) { nvtxRangePop(); return; }

    // --- Phase 2: Create new agents (NO DeviceAgentVector open) ---
    for (auto& plan : plans) {
        // Cell adjacent to existing back: leader_slot = parent slot, my_slot = new_slot[0]
        if (plan.n_new >= 1) {
            auto cell1 = fib_api.newAgent();
            cell1.setVariable<int>("x", plan.new_x[0]);
            cell1.setVariable<int>("y", plan.new_y[0]);
            cell1.setVariable<int>("z", plan.new_z[0]);
            cell1.setVariable<int>("cell_state", FIB_CAF);
            cell1.setVariable<int>("my_slot",     plan.new_slot[0]);
            cell1.setVariable<int>("leader_slot", plan.chain_slots[0]);  // points to old back
            cell1.setVariable<int>("divide_flag", 0);
            cell1.setVariable<int>("life", static_cast<int>(mean_life));
            occ[plan.new_x[0]][plan.new_y[0]][plan.new_z[0]][CELL_TYPE_FIB] = 1u;
            fib_pos_x[plan.new_slot[0]] = plan.new_x[0];
            fib_pos_y[plan.new_slot[0]] = plan.new_y[0];
            fib_pos_z[plan.new_slot[0]] = plan.new_z[0];
        }
        // Outermost new cell: new back of chain, my_slot = new_slot[1]
        if (plan.n_new >= 2) {
            auto cell2 = fib_api.newAgent();
            cell2.setVariable<int>("x", plan.new_x[1]);
            cell2.setVariable<int>("y", plan.new_y[1]);
            cell2.setVariable<int>("z", plan.new_z[1]);
            cell2.setVariable<int>("cell_state", FIB_CAF);
            cell2.setVariable<int>("my_slot",     plan.new_slot[1]);
            cell2.setVariable<int>("leader_slot", plan.new_slot[0]);  // points to cell1
            cell2.setVariable<int>("divide_flag", 0);
            cell2.setVariable<int>("life", static_cast<int>(mean_life));
            occ[plan.new_x[1]][plan.new_y[1]][plan.new_z[1]][CELL_TYPE_FIB] = 1u;
            fib_pos_x[plan.new_slot[1]] = plan.new_x[1];
            fib_pos_y[plan.new_slot[1]] = plan.new_y[1];
            fib_pos_z[plan.new_slot[1]] = plan.new_z[1];
        }
    }

    // --- Phase 3: Update existing agents to CAF + reset all divide_flags ---
    // Build set of slots to convert to CAF
    std::unordered_set<int> caf_slots;
    for (auto& plan : plans) {
        for (int s : plan.chain_slots) caf_slots.insert(s);
    }

    {   // Fresh DeviceAgentVector (new agents from Phase 2 appear at end, indices 0..N-1 are stable)
        flamegpu::DeviceAgentVector fib_vec2 = fib_api.getPopulationData();
        for (unsigned int i = 0; i < fib_vec2.size(); i++) {
            int ms = fib_vec2[i].getVariable<int>("my_slot");
            // Reset divide_flag for all cells (re-evaluated each step by fib_state_step)
            if (fib_vec2[i].getVariable<int>("divide_flag") != 0) {
                fib_vec2[i].setVariable<int>("divide_flag", 0);
            }
            // Convert chain members to CAF
            if (ms >= 0 && caf_slots.count(ms)) {
                fib_vec2[i].setVariable<int>("cell_state", FIB_CAF);
            }
        }
    }
    nvtxRangePop();
}

// ============================================================================
// Timing Accessor: Last PDE Solve Time (milliseconds)
// ============================================================================
double get_last_pde_ms() {
    return g_last_pde_ms;
}

// ============================================================================
// Timing Checkpoint Host Functions
//
// These are thin FLAMEGPU host function layers inserted at phase boundaries.
// Each records elapsed wall-clock time since the previous checkpoint.
// Because FLAMEGPU2 fully completes all GPU kernels in a layer before calling
// the next host function, wall-clock accurately captures GPU time per phase.
//
// Phase map (in layer execution order):
//   timing_step_start        → resets the clock (very first layer)
//   [Phase 0: recruitment]
//   timing_after_recruit     → captures recruit time
//   [Phase 1: broadcast + neighbor scan]
//   timing_after_broadcast   → captures broadcast+scan time
//   [reset_pde_buffers + state_transitions + compute_chemical_sources]
//   timing_after_sources     → captures state+sources time
//   [solve_pde  -- internally timed via g_last_pde_ms]
//   timing_after_pde         → captures solve_pde wall time (for cross-check)
//   [compute_pde_gradients]
//   timing_after_gradients   → captures gradients time
//   [Phase 3: ECM]
//   timing_after_ecm         → captures ECM time
//   [Phase 4: occ + movement]
//   timing_after_movement    → captures movement time
//   [Phase 5: division]
//   timing_after_division    → captures division time
//   [Phase 6: QSP -- internally timed via g_last_qsp_ms]
// ============================================================================

FLAMEGPU_HOST_FUNCTION(timing_step_start) {
    nvtxRangePush("Step Start");
    reset_step_timer();
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_recruit) {
    nvtxRangePush("Timing Checkpoint: recruit");
    record_checkpoint("recruit");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_broadcast) {
    nvtxRangePush("Timing Checkpoint: broadcast_scan");
    record_checkpoint("broadcast_scan");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_sources) {
    nvtxRangePush("Timing Checkpoint: state_sources");
    record_checkpoint("state_sources");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_pde) {
    nvtxRangePush("Timing Checkpoint: pde_wall");
    record_checkpoint("pde_wall");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_gradients) {
    nvtxRangePush("Timing Checkpoint: gradients");
    record_checkpoint("gradients");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_ecm) {
    nvtxRangePush("Timing Checkpoint: ecm");
    record_checkpoint("ecm");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_movement) {
    nvtxRangePush("Timing Checkpoint: movement");
    record_checkpoint("movement");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_division) {
    nvtxRangePush("Timing Checkpoint: division");
    record_checkpoint("division");
    nvtxRangePop();
}

} // namespace PDAC