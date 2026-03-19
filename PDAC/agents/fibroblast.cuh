// Fibroblast / Cancer-Associated Fibroblast (CAF) Agent Behavior Functions
// States: FIB_NORMAL (quiescent fibroblast), FIB_CAF (activated CAF)
// Activation: TGFB-driven NORMAL->CAF transition
//
// Movement: Follower-leader chain model
//   - Cells form chains (HEAD → MIDDLE → ... → TAIL)
//   - HEAD (leader_slot == -1): runs TGFB run-tumble chemotaxis
//   - Followers: move to where their leader WAS last step (caterpillar motion)
//   - MacroProperty arrays fib_pos_x/y/z store snapshot positions
//   - MacroProperty array fib_moved flags which cells moved this step

#ifndef FIBROBLAST_CUH
#define FIBROBLAST_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// Fibroblast: Broadcast location (spatial messaging)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float fx = (x + 0.5f) * voxel_size;
    const float fy = (y + 0.5f) * voxel_size;
    const float fz = (z + 0.5f) * voxel_size;

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_FIB);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("x", x);
    FLAMEGPU->message_out.setVariable<int>("y", y);
    FLAMEGPU->message_out.setVariable<int>("z", z);
    FLAMEGPU->message_out.setLocation(fx, fy, fz);
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write to occupancy grid (exclusive per voxel)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    occ[x][y][z][CELL_TYPE_FIB].exchange(1u);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Compute chemical sources
// (Fibroblasts do not secrete chemicals directly — matching HCC implementation)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write position snapshot to MacroProperty
// Resets fib_moved[my_slot] = 0 for this step
// Must run BEFORE sensor_move and follow_move each step
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_write_pos, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int my_slot = FLAMEGPU->getVariable<int>("my_slot");
    if (my_slot < 0) return flamegpu::ALIVE;  // Uninitialized, skip

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    auto fib_pos_x = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_x");
    auto fib_pos_y = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_y");
    auto fib_pos_z = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_z");
    auto fib_moved = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_moved");

    fib_pos_x[my_slot].exchange(x);
    fib_pos_y[my_slot].exchange(y);
    fib_pos_z[my_slot].exchange(z);
    fib_moved[my_slot].exchange(0);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: HEAD/sensor movement — TGFB run-tumble chemotaxis
// Only runs for cells where leader_slot == -1 (they are the chain front)
// Uses CAS on occ_grid for exclusivity
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_sensor_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int leader_slot = FLAMEGPU->getVariable<int>("leader_slot");
    if (leader_slot != -1) return flamegpu::ALIVE;  // Not a sensor (HEAD), skip

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // ECM based movement probability: higher ECM → more likely to be blocked
    {
        const float* ecm_ptr = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("ecm_grid_ptr"));
        float ECM_density = ecm_ptr[z * (grid_x * grid_y) + y * grid_x + x];
        float ECM_50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50");
        float ECM_sat = ECM_density / (ECM_density + ECM_50);
        if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;
    }

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Use TGFB gradient for chemotaxis — read directly from PDE
    const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_X))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Y))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Z))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const float EC50_grad = 1.0;
    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    int target_x = x;
    int target_y = y;
    int target_z = z;

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {

        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        float dot_product = v_x * grad_x + v_y * grad_y + v_z * grad_z;

        float lambda = 0.0000168;
        if (cell_state == FIB_CAF) {
            float lambda = 0.000168;
        }
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float norm_gradient = std::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        float cos_theta = dot_product / (norm_v * norm_gradient);

        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        float p_tumble = (lambda/2) * (1.0f-cos_theta) * (1-H_grad) * dt;
        p_tumble = 1 - std::exp(-p_tumble);

        if (FLAMEGPU->random.uniform<float>() < p_tumble) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        int tx = x + static_cast<int>(std::round(move_dir_x));
        int ty = y + static_cast<int>(std::round(move_dir_y));
        int tz = z + static_cast<int>(std::round(move_dir_z));

        if (tx < 0 || tx >= grid_x || ty < 0 || ty >= grid_y || tz < 0 || tz >= grid_z) {
            // FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        if (occ[tx][ty][tz][CELL_TYPE_CANCER] == 0u &&
            occ[tx][ty][tz][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
            occ[x][y][z][CELL_TYPE_FIB].exchange(0u);
            FLAMEGPU->setVariable<int>("x", tx);
            FLAMEGPU->setVariable<int>("y", ty);
            FLAMEGPU->setVariable<int>("z", tz);
            const int my_slot = FLAMEGPU->getVariable<int>("my_slot");
            if (my_slot >= 0) {
                auto fib_moved = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_moved");
                fib_moved[my_slot].exchange(1);
            }
        } else {
            // FLAMEGPU->setVariable<int>("tumble", 1);
        }
        return flamegpu::ALIVE;
    }
    // === TUMBLE PHASE (tumble == 1) ===
    // Collect all free Moore neighbors, shuffle, try each until CAS wins.
    else {
        int cand_x[26], cand_y[26], cand_z[26];
        int n_cands = 0;
        for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++) {
            if (di==0 && dj==0 && dk==0) continue;
            int nx = x+di, ny = y+dj, nz = z+dk;
            if (nx<0||nx>=grid_x||ny<0||ny>=grid_y||nz<0||nz>=grid_z) continue;
            if (occ[nx][ny][nz][CELL_TYPE_CANCER] != 0u) continue;
            if (occ[nx][ny][nz][CELL_TYPE_FIB] != 0u) continue;
            cand_x[n_cands] = nx; cand_y[n_cands] = ny; cand_z[n_cands] = nz;
            n_cands++;
        }
        if (n_cands == 0) return flamegpu::ALIVE;
        for (int i = n_cands-1; i > 0; i--) {
            int j = static_cast<int>(FLAMEGPU->random.uniform<float>() * (i+1));
            if (j > i) j = i;
            int tx=cand_x[i]; cand_x[i]=cand_x[j]; cand_x[j]=tx;
            int ty=cand_y[i]; cand_y[i]=cand_y[j]; cand_y[j]=ty;
            int tz=cand_z[i]; cand_z[i]=cand_z[j]; cand_z[j]=tz;
        }
        for (int i = 0; i < n_cands; i++) {
            if (occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
                occ[x][y][z][CELL_TYPE_FIB].exchange(0u);
                FLAMEGPU->setVariable<int>("x", cand_x[i]);
                FLAMEGPU->setVariable<int>("y", cand_y[i]);
                FLAMEGPU->setVariable<int>("z", cand_z[i]);
                FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(cand_x[i]-x));
                FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(cand_y[i]-y));
                FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(cand_z[i]-z));
                FLAMEGPU->setVariable<int>("tumble", 0);
                const int my_slot = FLAMEGPU->getVariable<int>("my_slot");
                if (my_slot >= 0) {
                    auto fib_moved = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_moved");
                    fib_moved[my_slot].exchange(1);
                }
                break;
            }
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Follower movement — moves to leader's snapshot position if leader moved
// Runs in separate layers to propagate movement through chain
// Layer 0: cells whose leader is the HEAD move (1 step from HEAD)
// Layer 1: cells whose leader moved in layer 0 move (2 steps from HEAD)
// Uses CAS on occ_grid for correctness with cross-chain conflicts
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_follow_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int leader_slot = FLAMEGPU->getVariable<int>("leader_slot");
    if (leader_slot == -1) return flamegpu::ALIVE;  // HEAD/sensor, handled by fib_sensor_move

    const int my_slot = FLAMEGPU->getVariable<int>("my_slot");
    if (my_slot < 0) return flamegpu::ALIVE;

    // Check if our leader moved this step
    auto fib_moved = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_moved");
    if (static_cast<int>(fib_moved[leader_slot]) == 0) return flamegpu::ALIVE;  // Leader didn't move

    // Get leader's snapshot position (from fib_write_pos at start of step)
    auto fib_pos_x = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_x");
    auto fib_pos_y = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_y");
    auto fib_pos_z = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_z");

    const int target_x = static_cast<int>(fib_pos_x[leader_slot]);
    const int target_y = static_cast<int>(fib_pos_y[leader_slot]);
    const int target_z = static_cast<int>(fib_pos_z[leader_slot]);

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    if (target_x == x && target_y == y && target_z == z) return flamegpu::ALIVE;  // Already there

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // CAS to claim the target voxel (leader should have vacated it)
    if (occ[target_x][target_y][target_z][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
        // Claimed successfully: release old voxel, update position
        occ[x][y][z][CELL_TYPE_FIB].exchange(0u);
        FLAMEGPU->setVariable<int>("x", target_x);
        FLAMEGPU->setVariable<int>("y", target_y);
        FLAMEGPU->setVariable<int>("z", target_z);

        // Signal to our followers that we moved
        fib_moved[my_slot].exchange(1);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: State step — TGFB-driven activation (NORMAL -> CAF) and lifespan
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {

    const int cell_state   = FLAMEGPU->getVariable<int>("cell_state");
    const int my_slot     = FLAMEGPU->getVariable<int>("my_slot");

    // Allow ALL normal cells with valid slots to evaluate TGFB-driven activation.
    // This includes isolated sensors (leader_slot==-1) and chain-back cells (leader_slot>=0).
    // fib_execute_divide (host) filters for true back-of-chain cells before expanding.
    if (my_slot < 0 || cell_state != FIB_NORMAL) {
        return flamegpu::ALIVE;
    }

    const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax_ss = FLAMEGPU->getVariable<int>("x");
    const int ay_ss = FLAMEGPU->getVariable<int>("y");
    const int az_ss = FLAMEGPU->getVariable<int>("z");
    const int voxel_ss = az_ss * ny_ss*nx_ss + ay_ss * nx_ss + ax_ss;
    const float TGFB = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_TGFB))[voxel_ss];
    const float ec50     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
    const float caf_act  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_ACTIVATION");
    const float activation = caf_act * 5 * (1 + TGFB / (TGFB + ec50));
    const float p_div  = 1.0f - expf(-activation);

    if (FLAMEGPU->random.uniform<float>() < p_div) {
        FLAMEGPU->setVariable<int>("divide_flag", 1);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Build Gaussian density field for ECM deposition
// Each fibroblast scatters a 3D Gaussian kernel (radius=10, sigma=3) to the
// fib_density_field MacroProperty. CAFs contribute with scale=1.0, normals 0.5.
// TGFB amplification is pre-multiplied at the fibroblast's voxel.
// update_ecm_grid host fn then uses this field to deposit ECM.
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_build_density_field, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cx = FLAMEGPU->getVariable<int>("x");
    const int cy = FLAMEGPU->getVariable<int>("y");
    const int cz = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Matching HCC: pure Gaussian scatter with CAF/normal scale only.
    // TGFB amplification is applied per-voxel in update_ecm_grid_kernel, not here.
    const float scale = (cell_state == FIB_CAF) ? 1.0f : 0.5f;

    // Gaussian parameters matching HCC (radius=10, sigma=3)
    const int radius = 10;
    const float variance = 9.0f;  // sigma^2 = 3^2
    // normalizer = 1 / (sigma^3 * sqrt(2*pi)) = 1 / (27 * 2.5066) ≈ 0.014784
    const float normalizer = 0.014784f;

    float* field_ptr = reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("fib_density_field_ptr"));

    for (int dx = -radius; dx <= radius; dx++) {
        const int nx = cx + dx;
        if (nx < 0 || nx >= grid_x) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            const int ny = cy + dy;
            if (ny < 0 || ny >= grid_y) continue;
            for (int dz = -radius; dz <= radius; dz++) {
                const int nz = cz + dz;
                if (nz < 0 || nz >= grid_z) continue;
                float dist_sq = static_cast<float>(dx*dx + dy*dy + dz*dz);
                float kernel_val = scale * normalizer * expf(-dist_sq / (2.0f * variance));
                atomicAdd(&field_ptr[nz * (grid_x * grid_y) + ny * grid_x + nx], kernel_val);
            }
        }
    }

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif  // FIBROBLAST_CUH
