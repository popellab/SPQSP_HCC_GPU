#include "pde_solver.cuh"
#include <iostream>
#include <cstring>
#include <cmath>

namespace PDAC {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void diffusion_reaction_kernel(
    const float* __restrict__ C_curr,
    float* __restrict__ C_next,
    const float* __restrict__ sources,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    int idx = z * (nx * ny) + y * nx + x;
    
    float C = C_curr[idx];
    
    // Compute Laplacian using 7-point stencil
    float laplacian = 0.0f;
    float dx2 = dx * dx;
    int neighbor_count = 0;
    
    // X-direction
    if (x > 0) {
        laplacian += (C_curr[idx - 1] - C) / dx2;
        neighbor_count++;
    } else {
        // Neumann BC: zero flux (reflective)
        laplacian += 0.0f;
    }
    
    if (x < nx - 1) {
        laplacian += (C_curr[idx + 1] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    // Y-direction
    if (y > 0) {
        laplacian += (C_curr[idx - nx] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    if (y < ny - 1) {
        laplacian += (C_curr[idx + nx] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    // Z-direction
    if (z > 0) {
        laplacian += (C_curr[idx - nx * ny] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    if (z < nz - 1) {
        laplacian += (C_curr[idx + nx * ny] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    // Reaction-diffusion-decay equation: dC/dt = D*∇²C - λ*C + S
    float diffusion = D * laplacian;
    float decay = -lambda * C;
    float source = sources[idx];
    
    // Forward Euler time integration
    C_next[idx] = C + dt * (diffusion + decay + source);
    
    // Ensure non-negative concentrations
    if (C_next[idx] < 0.0f) {
        C_next[idx] = 0.0f;
    }
}

__global__ void copy_substrate_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Kernel: Read concentration at specific voxel for all agents
__global__ void read_concentrations_at_voxels(
    const float* __restrict__ d_concentrations,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    float* __restrict__ d_agent_concentrations,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;
    
    int x = d_agent_x[agent_idx];
    int y = d_agent_y[agent_idx];
    int z = d_agent_z[agent_idx];
    
    // Bounds check
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
        d_agent_concentrations[agent_idx] = 0.0f;
        return;
    }
    
    // Compute flat index: substrate_offset + z*(nx*ny) + y*nx + x
    int voxel_idx = z * (nx * ny) + y * nx + x;
    int total_voxels = nx * ny * nz;
    int concentration_idx = substrate_idx * total_voxels + voxel_idx;
    
    d_agent_concentrations[agent_idx] = d_concentrations[concentration_idx];
}

// Kernel: Write (add) sources from agents to voxels
__global__ void add_sources_from_agents(
    float* __restrict__ d_sources,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    const float* __restrict__ d_agent_source_rates,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz,
    float voxel_volume)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;
    
    int x = d_agent_x[agent_idx];
    int y = d_agent_y[agent_idx];
    int z = d_agent_z[agent_idx];
    
    // Bounds check
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
        return;
    }
    
    float source_rate = d_agent_source_rates[agent_idx];

    // Skip if no source
    if (source_rate == 0.0f) return;

    // Convert from amount/(cell*time) to concentration/time by dividing by voxel volume
    float concentration_rate = source_rate / voxel_volume;

    // Compute flat index
    int voxel_idx = z * (nx * ny) + y * nx + x;
    int total_voxels = nx * ny * nz;
    int source_idx = substrate_idx * total_voxels + voxel_idx;

    // Atomic add (multiple agents may be in same voxel)
    atomicAdd(&d_sources[source_idx], concentration_rate);
}

__global__ void add_source_kernel(
    float* __restrict__ sources,
    int idx,
    float value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sources[idx] += value;
    }
}

// ============================================================================
// Conjugate Gradient Solver Kernels (Implicit Method)
// ============================================================================

// Apply the implicit diffusion-decay operator: A*x = (I + dt*λ - dt*D*∇²)x
__global__ void apply_diffusion_operator(
    const float* __restrict__ x,
    float* __restrict__ Ax,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int idx = iz * (nx * ny) + iy * nx + ix;
    float x_center = x[idx];

    // Compute Laplacian using 7-point stencil
    float laplacian = 0.0f;
    float dx2 = dx * dx;

    // X-direction
    if (ix > 0) {
        laplacian += (x[idx - 1] - x_center) / dx2;
    }
    if (ix < nx - 1) {
        laplacian += (x[idx + 1] - x_center) / dx2;
    }

    // Y-direction
    if (iy > 0) {
        laplacian += (x[idx - nx] - x_center) / dx2;
    }
    if (iy < ny - 1) {
        laplacian += (x[idx + nx] - x_center) / dx2;
    }

    // Z-direction
    if (iz > 0) {
        laplacian += (x[idx - nx * ny] - x_center) / dx2;
    }
    if (iz < nz - 1) {
        laplacian += (x[idx + nx * ny] - x_center) / dx2;
    }

    // Apply operator: A*x = x + dt*λ*x - dt*D*∇²x
    // (equivalent to: (I + dt*λ - dt*D*∇²)x)
    Ax[idx] = x_center + dt * lambda * x_center - dt * D * laplacian;
}

// Vector addition: y = y + alpha*x
__global__ void vector_axpy(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

// Vector scaling: y = alpha*x
__global__ void vector_scale(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx];
    }
}

// Vector copy: dst = src
__global__ void vector_copy(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Dot product kernel (partial reduction)
__global__ void dot_product_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ partial_sums,
    int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute partial dot product
    float sum = 0.0f;
    if (idx < n) {
        sum = x[idx] * y[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Clamp negative values to zero
__global__ void clamp_nonnegative(float* __restrict__ x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && x[idx] < 0.0f) {
        x[idx] = 0.0f;
    }
}

// ============================================================================
// PDESolver Implementation
// ============================================================================

PDESolver::PDESolver(const PDEConfig& config)
    : config_(config),
      d_concentrations_current_(nullptr),
      d_concentrations_next_(nullptr),
      d_sources_(nullptr),
      d_cg_r_(nullptr),
      d_cg_p_(nullptr),
      d_cg_Ap_(nullptr),
      d_cg_temp_(nullptr),
      d_dot_buffer_(nullptr),
      cg_reduction_blocks_(0),
      h_temp_buffer_(nullptr)
{
    // Validate config
    if (config_.nx <= 0 || config_.ny <= 0 || config_.nz <= 0) {
        throw std::runtime_error("Invalid grid dimensions");
    }
    if (config_.num_substrates <= 0 || config_.num_substrates > NUM_SUBSTRATES) {
        throw std::runtime_error("Invalid number of substrates");
    }
}

PDESolver::~PDESolver() {
    if (d_concentrations_current_) CUDA_CHECK(cudaFree(d_concentrations_current_));
    if (d_concentrations_next_) CUDA_CHECK(cudaFree(d_concentrations_next_));
    if (d_sources_) CUDA_CHECK(cudaFree(d_sources_));
    if (d_cg_r_) CUDA_CHECK(cudaFree(d_cg_r_));
    if (d_cg_p_) CUDA_CHECK(cudaFree(d_cg_p_));
    if (d_cg_Ap_) CUDA_CHECK(cudaFree(d_cg_Ap_));
    if (d_cg_temp_) CUDA_CHECK(cudaFree(d_cg_temp_));
    if (d_dot_buffer_) CUDA_CHECK(cudaFree(d_dot_buffer_));
    if (h_temp_buffer_) delete[] h_temp_buffer_;
}

void PDESolver::initialize() {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    size_t voxel_size = total_voxels * sizeof(float);

    // Allocate device memory for concentration fields
    CUDA_CHECK(cudaMalloc(&d_concentrations_current_, total_size));
    CUDA_CHECK(cudaMalloc(&d_concentrations_next_, total_size));
    CUDA_CHECK(cudaMalloc(&d_sources_, total_size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_concentrations_current_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_concentrations_next_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_sources_, 0, total_size));

    // Allocate CG workspace (per voxel, not per substrate)
    CUDA_CHECK(cudaMalloc(&d_cg_r_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_p_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_Ap_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_temp_, voxel_size));

    // Allocate reduction buffer for dot products
    int threads_per_block = 256;
    cg_reduction_blocks_ = (total_voxels + threads_per_block - 1) / threads_per_block;
    CUDA_CHECK(cudaMalloc(&d_dot_buffer_, cg_reduction_blocks_ * sizeof(float)));

    // Allocate host buffer for transfers
    h_temp_buffer_ = new float[total_voxels];

    float cg_workspace_mb = 5.0f * voxel_size / (1024.0f * 1024.0f);
    std::cout << "PDE Solver initialized (Implicit CG):" << std::endl;
    std::cout << "  Grid: " << config_.nx << "x" << config_.ny << "x" << config_.nz << std::endl;
    std::cout << "  Substrates: " << config_.num_substrates << std::endl;
    std::cout << "  Concentration memory: " << (3 * total_size) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  CG workspace: " << cg_workspace_mb << " MB" << std::endl;
    std::cout << "  PDE timestep: " << config_.dt_pde << " s" << std::endl;
    std::cout << "  Substeps per ABM step: " << config_.substeps_per_abm << std::endl;
}

// Solve implicit system using Conjugate Gradient: A*x = b
// where A = (I + dt*λ - dt*D*∇²)
void PDESolver::solve_implicit_cg(float* d_C, const float* d_rhs, float D, float lambda, float dt, float dx) {
    const int n = config_.nx * config_.ny * config_.nz;
    const int max_iters = 100;  // Usually converges in 10-30 iterations
    const float tolerance = 1e-6f;

    // CUDA grid configuration
    dim3 block_3d(8, 8, 8);
    dim3 grid_3d(
        (config_.nx + block_3d.x - 1) / block_3d.x,
        (config_.ny + block_3d.y - 1) / block_3d.y,
        (config_.nz + block_3d.z - 1) / block_3d.z
    );

    int threads_1d = 256;
    int blocks_1d = (n + threads_1d - 1) / threads_1d;

    // Helper lambda for dot product (with final reduction on host)
    auto dot_product = [&](const float* x, const float* y) -> float {
        dot_product_kernel<<<cg_reduction_blocks_, threads_1d>>>(x, y, d_dot_buffer_, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Final reduction on host
        std::vector<float> h_partial(cg_reduction_blocks_);
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_dot_buffer_,
                              cg_reduction_blocks_ * sizeof(float), cudaMemcpyDeviceToHost));
        float sum = 0.0f;
        for (int i = 0; i < cg_reduction_blocks_; i++) {
            sum += h_partial[i];
        }
        return sum;
    };

    // Initialize: r = b - A*x0 (where x0 = d_C is current concentration)
    apply_diffusion_operator<<<grid_3d, block_3d>>>(d_C, d_cg_Ap_,
                                                     config_.nx, config_.ny, config_.nz,
                                                     D, lambda, dt, dx);
    CUDA_CHECK(cudaDeviceSynchronize());

    // r = b - A*x0
    vector_copy<<<blocks_1d, threads_1d>>>(d_cg_r_, d_rhs, n);
    vector_axpy<<<blocks_1d, threads_1d>>>(d_cg_r_, d_cg_Ap_, -1.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // p = r
    vector_copy<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_r_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float rsold = dot_product(d_cg_r_, d_cg_r_);
    float rs_initial = rsold;

    // CG iteration
    for (int iter = 0; iter < max_iters; iter++) {
        // Ap = A*p
        apply_diffusion_operator<<<grid_3d, block_3d>>>(d_cg_p_, d_cg_Ap_,
                                                         config_.nx, config_.ny, config_.nz,
                                                         D, lambda, dt, dx);
        CUDA_CHECK(cudaDeviceSynchronize());

        // alpha = rsold / (p . Ap)
        float pAp = dot_product(d_cg_p_, d_cg_Ap_);
        float alpha = rsold / (pAp + 1e-30f);  // Add epsilon to avoid division by zero

        // x = x + alpha*p
        vector_axpy<<<blocks_1d, threads_1d>>>(d_C, d_cg_p_, alpha, n);

        // r = r - alpha*Ap
        vector_axpy<<<blocks_1d, threads_1d>>>(d_cg_r_, d_cg_Ap_, -alpha, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check convergence
        float rsnew = dot_product(d_cg_r_, d_cg_r_);
        float residual_norm = sqrtf(rsnew / (rs_initial + 1e-30f));

        if (residual_norm < tolerance) {
            // Converged!
            break;
        }

        // beta = rsnew / rsold
        float beta = rsnew / (rsold + 1e-30f);

        // p = r + beta*p
        vector_scale<<<blocks_1d, threads_1d>>>(d_cg_temp_, d_cg_p_, beta, n);
        vector_copy<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_r_, n);
        vector_axpy<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_temp_, 1.0f, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        rsold = rsnew;
    }

    // Ensure non-negative concentrations (CG can produce small negative values due to roundoff)
    clamp_nonnegative<<<blocks_1d, threads_1d>>>(d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void PDESolver::solve_timestep() {
    int n = config_.nx * config_.ny * config_.nz;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Solve for each substrate independently using implicit CG
    for (int sub = 0; sub < config_.num_substrates; sub++) {
        float D = config_.diffusion_coeffs[sub];
        float lambda = config_.decay_rates[sub];

        // Pointers to this substrate's data
        float* C_curr = d_concentrations_current_ + sub * n;
        float* sources = d_sources_ + sub * n;

        // Build right-hand side: rhs = C^n + dt_abm*S
        // We solve for the entire ABM timestep at once (implicit method is unconditionally stable!)
        vector_copy<<<blocks, threads>>>(d_cg_temp_, C_curr, n);
        vector_axpy<<<blocks, threads>>>(d_cg_temp_, sources, config_.dt_abm, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Solve implicit system: (I + dt*λ - dt*D*∇²)C^(n+1) = rhs
        // using dt = dt_abm (entire ABM timestep, no substeps needed!)
        solve_implicit_cg(C_curr, d_cg_temp_, D, lambda, config_.dt_abm, config_.voxel_size);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void PDESolver::set_sources(const float* h_sources, int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }
    
    int voxels = config_.nx * config_.ny * config_.nz;
    size_t offset = substrate_idx * voxels * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(
        d_sources_ + substrate_idx * voxels,
        h_sources,
        voxels * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

void PDESolver::add_source_at_voxel(int x, int y, int z, int substrate_idx, float value) {
    if (x < 0 || x >= config_.nx || y < 0 || y >= config_.ny || z < 0 || z >= config_.nz) {
        return; // Out of bounds
    }
    
    int voxel_idx = idx(x, y, z);
    int offset = substrate_idx * get_total_voxels() + voxel_idx;
    
    // Atomic add on device (launch simple kernel)
    add_source_kernel<<<1, 1>>>(d_sources_, offset, value);
    CUDA_CHECK(cudaGetLastError());
}

void PDESolver::get_concentrations(float* h_concentrations, int substrate_idx) const {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }
    
    int voxels = config_.nx * config_.ny * config_.nz;
    
    CUDA_CHECK(cudaMemcpy(
        h_concentrations,
        d_concentrations_current_ + substrate_idx * voxels,
        voxels * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

float PDESolver::get_concentration_at_voxel(int x, int y, int z, int substrate_idx) const {
    if (x < 0 || x >= config_.nx || y < 0 || y >= config_.ny || z < 0 || z >= config_.nz) {
        return 0.0f;
    }
    
    int voxel_idx = idx(x, y, z);
    int offset = substrate_idx * get_total_voxels() + voxel_idx;
    
    float value;
    CUDA_CHECK(cudaMemcpy(
        &value,
        d_concentrations_current_ + offset,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    
    return value;
}

float* PDESolver::get_device_concentration_ptr(int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return nullptr;
    }
    return d_concentrations_current_ + substrate_idx * get_total_voxels();
}

float* PDESolver::get_device_source_ptr(int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return nullptr;
    }
    return d_sources_ + substrate_idx * get_total_voxels();
}

void PDESolver::reset_concentrations() {
    int total_voxels = get_total_voxels();
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    CUDA_CHECK(cudaMemset(d_concentrations_current_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_concentrations_next_, 0, total_size));
}

void PDESolver::reset_sources() {
    int total_voxels = get_total_voxels();
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    CUDA_CHECK(cudaMemset(d_sources_, 0, total_size));
}

void PDESolver::set_initial_concentration(int substrate_idx, float value) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }
    
    int voxels = get_total_voxels();
    std::vector<float> init_values(voxels, value);
    
    CUDA_CHECK(cudaMemcpy(
        d_concentrations_current_ + substrate_idx * voxels,
        init_values.data(),
        voxels * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

void PDESolver::swap_buffers() {
    float* temp = d_concentrations_current_;
    d_concentrations_current_ = d_concentrations_next_;
    d_concentrations_next_ = temp;
}

} // namespace PDAC