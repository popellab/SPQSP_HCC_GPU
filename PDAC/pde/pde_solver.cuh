#ifndef PDE_SOLVER_CUH
#define PDE_SOLVER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace PDAC {

// Chemical substrate indices (matching CPU HCC implementation)
// Note: NIVO and CABO moved to QSP compartments, 4 new chemicals added
enum ChemicalSubstrate {
    CHEM_O2 = 0,
    CHEM_IFN,       // IFN-gamma
    CHEM_IL2,       // IL-2
    CHEM_IL10,      // IL-10
    CHEM_TGFB,      // TGF-beta
    CHEM_CCL2,      // CCL2
    CHEM_ARGI,      // Arginase I (MDSC produces)
    CHEM_NO,        // Nitric Oxide (MDSC produces)
    CHEM_IL12,      // IL-12 (M1 macrophages produce)
    CHEM_VEGFA,     // VEGF-A (cancer cells produce)
    NUM_SUBSTRATES
};

// Configuration for PDE solver
struct PDEConfig {
    int nx, ny, nz;                    // Grid dimensions
    int num_substrates;                 // Number of chemical species
    float voxel_size;                   // Spatial resolution (cm)
    float dt_abm;                       // ABM timestep (seconds)
    float dt_pde;                       // PDE timestep (seconds)
    int substeps_per_abm;               // PDE substeps per ABM step
    
    // Diffusion coefficients (cm²/s) for each substrate
    float diffusion_coeffs[NUM_SUBSTRATES];
    
    // Decay rates (1/s) for each substrate
    float decay_rates[NUM_SUBSTRATES];
    
    // Boundary conditions (0 = Neumann/no-flux, 1 = Dirichlet)
    int boundary_type;
};

class PDESolver {
public:
    PDESolver(const PDEConfig& config);
    ~PDESolver();
    
    // Initialize solver and allocate memory
    void initialize();
    
    // Run PDE for one ABM timestep (runs substeps internally)
    void solve_timestep();
    
    // Agent-PDE coupling: set source/sink values
    // sources: array of size [num_substrates][nz][ny][nx]
    void set_sources(const float* h_sources, int substrate_idx);
    void add_source_at_voxel(int x, int y, int z, int substrate_idx, float value);
    
    // Agent-PDE coupling: get concentration values
    void get_concentrations(float* h_concentrations, int substrate_idx) const;
    float get_concentration_at_voxel(int x, int y, int z, int substrate_idx) const;
    
    // Direct device pointer access (for FLAME GPU integration)
    float* get_device_concentration_ptr(int substrate_idx);
    float* get_device_source_ptr(int substrate_idx);
    
    // Reset all concentrations to zero
    void reset_concentrations();
    void reset_sources();
    
    // Set uniform initial concentration for a substrate
    void set_initial_concentration(int substrate_idx, float value);
    
    // Utility
    int get_total_voxels() const { return config_.nx * config_.ny * config_.nz; }
    
private:
    PDEConfig config_;

    // Device memory
    float* d_concentrations_current_;   // [num_substrates][nz][ny][nx]
    float* d_concentrations_next_;      // Double buffering (for output)
    float* d_sources_;                   // [num_substrates][nz][ny][nx]

    // CG solver workspace (per voxel, not per substrate)
    float* d_cg_r_;      // Residual vector
    float* d_cg_p_;      // Search direction
    float* d_cg_Ap_;     // A*p product
    float* d_cg_temp_;   // Temporary storage
    float* d_dot_buffer_; // Reduction buffer for dot products
    int cg_reduction_blocks_;

    // Host memory (for transfers)
    float* h_temp_buffer_;

    // Internal indexing
    inline int idx(int x, int y, int z) const {
        return z * (config_.nx * config_.ny) + y * config_.nx + x;
    }

    inline int idx_substrate(int x, int y, int z, int substrate) const {
        return substrate * (config_.nx * config_.ny * config_.nz) + idx(x, y, z);
    }

    // CG solver internals
    void solve_implicit_cg(float* d_C, const float* d_rhs, float D, float lambda, float dt, float dx);

    // Swap current and next buffers
    void swap_buffers();
};

// CUDA kernel declarations
__global__ void diffusion_reaction_kernel(
    const float* __restrict__ C_curr,
    float* __restrict__ C_next,
    const float* __restrict__ sources,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx
);

__global__ void copy_substrate_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
);

__global__ void read_concentrations_at_voxels(
    const float* __restrict__ d_concentrations,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    float* __restrict__ d_agent_concentrations,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz);

__global__ void add_sources_from_agents(
    float* __restrict__ d_sources,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    const float* __restrict__ d_agent_source_rates,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz,
    float voxel_volume);

// CG solver kernels
__global__ void apply_diffusion_operator(
    const float* __restrict__ x,
    float* __restrict__ Ax,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx);

__global__ void vector_axpy(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n);

__global__ void vector_scale(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n);

__global__ void vector_copy(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n);

__global__ void dot_product_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ partial_sums,
    int n);

__global__ void clamp_nonnegative(
    float* __restrict__ x,
    int n);

} // namespace PDAC

#endif // PDE_SOLVER_CUH