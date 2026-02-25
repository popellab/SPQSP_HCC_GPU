/**
 * PDE Solver Validation Tests
 *
 * Tests are independent of FLAMEGPU/agents — they drive the PDESolver class
 * directly via its public API (get_device_*_ptr, set_initial_concentration, etc.).
 *
 * Four tests:
 *
 *  1. Background decay only (no diffusion, no agent sources)
 *     D=0, λ>0, S=0, U=0, uniform C0.
 *     LOD Thomas with c1=0 reduces to: each sweep multiplies by 1/(1+c2),
 *     so per full step: C *= 1/(1+dt*λ/3)^3.
 *     Compare to analytical exp(-λ*dt) — LOD implicit Euler error should be small.
 *
 *  2. Agent uptake (exact ODE, no diffusion, no background decay)
 *     D=0, λ=0, S=0, U>0, uniform C0.
 *     Exact ODE: C_new = C0 * exp(-U*dt).
 *     After N steps: C = C0 * exp(-U*N*dt).  Should match to float precision.
 *
 *  3. Source + uptake equilibrium (exact ODE, no diffusion, no background decay)
 *     D=0, λ=0, S>0, U>0, C_init=0.
 *     Steady state: C_ss = S/U.
 *     Run enough steps to converge; verify max error < 1%.
 *
 *  4. Mass conservation with pure diffusion
 *     D>0, λ=0, no sources.  Point source at center, zero elsewhere.
 *     Total mass = sum(C)*V_voxel must be conserved at all times.
 */

#include "pde_solver.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

namespace PDAC {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fill_device_float(float* d_ptr, float value, int n) {
    std::vector<float> h(n, value);
    cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

static bool check(bool pass, const char* label) {
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << "  " << label << "\n";
    return pass;
}

// ---------------------------------------------------------------------------
// Build a minimal PDEConfig
// ---------------------------------------------------------------------------

static PDEConfig make_config(int nx, int ny, int nz,
                             float D, float lambda,
                             float voxel_um = 20.0f, float dt = 600.0f) {
    PDEConfig cfg{};
    cfg.nx = nx; cfg.ny = ny; cfg.nz = nz;
    cfg.num_substrates    = NUM_SUBSTRATES;
    cfg.voxel_size        = voxel_um * 1e-4f;  // µm → cm
    cfg.dt_abm            = dt;
    cfg.dt_pde            = dt;
    cfg.substeps_per_abm  = 1;
    cfg.boundary_type     = 0;

    for (int s = 0; s < NUM_SUBSTRATES; s++) {
        cfg.diffusion_coeffs[s] = D;
        cfg.decay_rates[s]      = lambda;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: Background decay only
// ---------------------------------------------------------------------------
// D=0, λ>0, no sources, uniform initial C0.
// LOD Thomas with c1=0: each 1D sweep multiplies every element by 1/(1+c2)
// where c2 = dt*λ/3. After 3 sweeps (one full step):
//   C_step = C_prev / (1 + dt*λ/3)^3
// After N steps:
//   C_expected = C0 / (1 + dt*λ/3)^(3*N)
// Compare against analytical exp(-λ*N*dt). The implicit-Euler error is O(dt).

static bool test_background_decay() {
    std::cout << "\nTest 1: Background decay (no diffusion, no agents)\n";

    const int  N = 11;
    const float lambda = 6.5e-5f;   // IFN-γ decay rate [1/s]
    const float dt     = 600.0f;    // ABM timestep [s]
    const float C0     = 1.0f;
    const int   steps  = 20;

    PDEConfig cfg = make_config(N, N, N, /*D=*/0.0f, lambda, /*vox_um=*/20.0f, dt);
    PDESolver solver(cfg);
    solver.initialize();

    // Set uniform C0 on substrate 0
    solver.set_initial_concentration(CHEM_O2, C0);

    // Run steps (no sources/uptakes — buffers stay zero from initialize)
    for (int i = 0; i < steps; i++) {
        solver.solve_timestep();
    }

    // Read back substrate 0 and compute mean
    std::vector<float> h_buf(N*N*N);
    solver.get_concentrations(h_buf.data(), CHEM_O2);
    float C_mean = std::accumulate(h_buf.begin(), h_buf.end(), 0.0f) / (N*N*N);

    // LOD implicit-Euler expected value (what the solver should produce exactly)
    float c2         = dt * lambda / 3.0f;
    float factor_per_step = std::pow(1.0f / (1.0f + c2), 3.0f);
    float C_lod      = C0 * std::pow(factor_per_step, (float)steps);

    // Analytical (continuous) reference
    float C_analytic = C0 * std::exp(-lambda * dt * steps);

    float err_vs_lod      = std::fabs(C_mean - C_lod) / C_lod;
    float err_vs_analytic = std::fabs(C_mean - C_analytic) / C_analytic;

    std::cout << "  C0=" << C0 << "  C_mean=" << C_mean
              << "  C_lod=" << C_lod << "  C_analytic=" << C_analytic << "\n";
    std::cout << "  err_vs_lod=" << err_vs_lod
              << "  err_vs_analytic=" << err_vs_analytic << "\n";

    bool p1 = check(err_vs_lod < 1e-5f,
                    "Mean concentration matches LOD implicit-Euler formula");
    bool p2 = check(err_vs_analytic < 0.01f,
                    "Mean concentration within 1% of analytical exp(-λt)");
    bool p3 = check(*std::min_element(h_buf.begin(), h_buf.end()) >= 0.0f,
                    "No negative concentrations");

    return p1 && p2 && p3;
}

// ---------------------------------------------------------------------------
// Test 2: Agent uptake via exact ODE (no diffusion, no background decay)
// ---------------------------------------------------------------------------
// D=0, λ=0, S=0, U>0, uniform C0.
// Exact ODE: dp/dt = -U*p → p(t) = C0 * exp(-U*t)
// After one step with dt:  C = C0 * exp(-U*dt)
// After N steps:           C = C0 * exp(-U*N*dt)

static bool test_agent_uptake() {
    std::cout << "\nTest 2: Agent uptake — exact ODE (no diffusion, no background decay)\n";

    const int  N  = 11;
    const float U  = 1e-3f;    // uptake rate constant [1/s]
    const float dt = 600.0f;
    const float C0 = 1.0f;
    const int  steps = 10;

    PDEConfig cfg = make_config(N, N, N, /*D=*/0.0f, /*lambda=*/0.0f, 20.0f, dt);
    PDESolver solver(cfg);
    solver.initialize();
    solver.set_initial_concentration(CHEM_O2, C0);

    int V = N*N*N;

    for (int i = 0; i < steps; i++) {
        // Set uniform uptake each step (reset between steps, as the real sim does)
        fill_device_float(solver.get_device_uptake_ptr(CHEM_O2), U, V);
        solver.solve_timestep();
        // Re-zero uptake so next step's reset_uptakes is a no-op (we set manually)
        cudaMemset(solver.get_device_uptake_ptr(CHEM_O2), 0, V * sizeof(float));
    }

    std::vector<float> h_buf(V);
    solver.get_concentrations(h_buf.data(), CHEM_O2);
    float C_mean = std::accumulate(h_buf.begin(), h_buf.end(), 0.0f) / V;

    float C_expected = C0 * std::exp(-U * dt * steps);
    float rel_err    = std::fabs(C_mean - C_expected) / C_expected;

    std::cout << "  C0=" << C0 << "  C_mean=" << C_mean
              << "  C_expected=" << C_expected << "  rel_err=" << rel_err << "\n";

    bool p1 = check(rel_err < 1e-4f,
                    "Exact ODE uptake matches exp(-U*N*dt) to float precision");
    bool p2 = check(*std::min_element(h_buf.begin(), h_buf.end()) >= 0.0f,
                    "No negative concentrations");
    return p1 && p2;
}

// ---------------------------------------------------------------------------
// Test 3: Source + uptake steady state
// ---------------------------------------------------------------------------
// D=0, λ=0, S>0, U>0, C_init=0.
// Exact ODE: dp/dt = S - U*p → p_ss = S/U
// Run until convergence, then check.

static bool test_source_uptake_equilibrium() {
    std::cout << "\nTest 3: Source + uptake equilibrium (no diffusion, no background decay)\n";

    const int  N   = 11;
    const float S   = 5e-7f;   // source [conc/s]  (already volume-divided)
    const float U   = 1e-3f;   // uptake [1/s]
    const float dt  = 600.0f;
    const int  steps = 200;    // run long enough to converge

    float C_ss_expected = S / U;

    PDEConfig cfg = make_config(N, N, N, 0.0f, 0.0f, 20.0f, dt);
    PDESolver solver(cfg);
    solver.initialize();
    // C_init = 0 (already zeroed by initialize)

    int V = N*N*N;

    for (int i = 0; i < steps; i++) {
        fill_device_float(solver.get_device_source_ptr(CHEM_CCL2), S, V);
        fill_device_float(solver.get_device_uptake_ptr(CHEM_CCL2), U, V);
        solver.solve_timestep();
        cudaMemset(solver.get_device_source_ptr(CHEM_CCL2), 0, V * sizeof(float));
        cudaMemset(solver.get_device_uptake_ptr(CHEM_CCL2), 0, V * sizeof(float));
    }

    std::vector<float> h_buf(V);
    solver.get_concentrations(h_buf.data(), CHEM_CCL2);
    float C_mean = std::accumulate(h_buf.begin(), h_buf.end(), 0.0f) / V;
    float rel_err = std::fabs(C_mean - C_ss_expected) / C_ss_expected;

    std::cout << "  S=" << S << "  U=" << U << "  C_ss_expected=" << C_ss_expected
              << "  C_mean=" << C_mean << "  rel_err=" << rel_err << "\n";

    bool p1 = check(rel_err < 1e-4f, "Converges to S/U steady state");
    bool p2 = check(*std::min_element(h_buf.begin(), h_buf.end()) >= 0.0f,
                    "No negative concentrations");
    return p1 && p2;
}

// ---------------------------------------------------------------------------
// Test 4: Mass conservation with pure diffusion
// ---------------------------------------------------------------------------
// D>0, λ=0, no sources.  Point source (1 mol) at center voxel.
// Total mass = sum(C) * V_voxel must stay constant.

static bool test_mass_conservation() {
    std::cout << "\nTest 4: Mass conservation with pure diffusion\n";

    const int   N       = 21;
    const float D       = 1e-7f;    // IFN-γ diffusivity [cm²/s]
    const float dt      = 600.0f;
    const float vox_um  = 20.0f;
    const float vox_cm  = vox_um * 1e-4f;
    const int   steps   = 50;

    PDEConfig cfg = make_config(N, N, N, D, /*lambda=*/0.0f, vox_um, dt);
    PDESolver solver(cfg);
    solver.initialize();

    // Place 1 unit at center voxel (concentration = 1/V_voxel so total mass = 1)
    int V     = N*N*N;
    int cx    = N/2, cy = N/2, cz = N/2;
    int center = cz*N*N + cy*N + cx;
    float V_voxel = vox_cm * vox_cm * vox_cm;
    float C_center = 1.0f / V_voxel;  // 1 mol in one voxel

    std::vector<float> h_init(V, 0.0f);
    h_init[center] = C_center;
    cudaMemcpy(solver.get_device_concentration_ptr(CHEM_IFN),
               h_init.data(), V * sizeof(float), cudaMemcpyHostToDevice);

    float mass_init = C_center * V_voxel;  // = 1.0

    float max_mass_err = 0.0f;

    for (int i = 0; i < steps; i++) {
        solver.solve_timestep();

        std::vector<float> h_buf(V);
        solver.get_concentrations(h_buf.data(), CHEM_IFN);
        float total = std::accumulate(h_buf.begin(), h_buf.end(), 0.0f) * V_voxel;
        float err   = std::fabs(total - mass_init) / mass_init;
        max_mass_err = std::max(max_mass_err, err);
    }

    std::cout << "  Initial mass=" << mass_init
              << "  max_mass_error=" << max_mass_err << "\n";

    bool p1 = check(max_mass_err < 1e-4f, "Total mass conserved to 0.01% over 50 steps");

    // Also verify concentration spread away from center
    std::vector<float> h_final(V);
    solver.get_concentrations(h_final.data(), CHEM_IFN);
    float C_neighbor = h_final[center + 1];  // adjacent voxel should have received mass
    bool p2 = check(C_neighbor > 0.0f, "Diffusion spreads mass to neighbors");
    bool p3 = check(*std::min_element(h_final.begin(), h_final.end()) >= 0.0f,
                    "No negative concentrations");

    return p1 && p2 && p3;
}

} // namespace PDAC

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== PDE Solver Validation Tests ===\n";

    int passed = 0, total = 4;

    if (PDAC::test_background_decay())        passed++;
    if (PDAC::test_agent_uptake())            passed++;
    if (PDAC::test_source_uptake_equilibrium()) passed++;
    if (PDAC::test_mass_conservation())       passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
