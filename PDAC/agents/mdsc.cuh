#ifndef FLAMEGPU_TNBC_MDSC_CUH
#define FLAMEGPU_TNBC_MDSC_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// Von Neumann mask: first 6 bits correspond to face neighbors (indices 0-5)
// Directions 0-5 are: {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
constexpr unsigned int VON_NEUMANN_MASK_MDSC = 0x3Fu;  // binary: 00111111

// Helper function to get Moore neighborhood direction for MDSC
__device__ __forceinline__ void get_moore_direction_mdsc(int idx, int& dx, int& dy, int& dz) {
    const int dirs[26][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
        {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
        {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
        {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}
    };
    dx = dirs[idx][0];
    dy = dirs[idx][1];
    dz = dirs[idx][2];
}

// Helper: Hill equation for PD1-PDL1 suppression
__device__ __forceinline__ float hill_equation_mdsc(float x, float k50, float n) {
    if (x <= 0.0f) return 0.0f;
    const float xn = powf(x, n);
    const float kn = powf(k50, n);
    return xn / (kn + xn);
}

// MDSC agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(mdsc_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", 0);  // MDSCs have single state
    FLAMEGPU->message_out.setVariable<float>("PDL1", FLAMEGPU->getVariable<float>("PDL1_syn"));
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    return flamegpu::ALIVE;
}

// MDSC agent function: Scan neighbors and cache available voxels
// MDSCs check for voxels without other MDSCs (1 MDSC per voxel max)
FLAMEGPU_AGENT_FUNCTION(mdsc_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const float my_pos_x = (my_x + 0.5f) * voxel_size;
    const float my_pos_y = (my_y + 0.5f) * voxel_size;
    const float my_pos_z = (my_z + 0.5f) * voxel_size;

    int cancer_count = 0;
    int tcell_count = 0;
    int treg_count = 0;
    int mdsc_count = 0;

    // Track which neighbor voxels have MDSCs (for movement availability)
    bool neighbor_blocked[26] = {false};
    int neighbor_tcells[26] = {0};

    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");

        const int dx = msg_x - my_x;
        const int dy = msg_y - my_y;
        const int dz = msg_z - my_z;

        // Moore neighborhood (excluding self)
        if (abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1 && !(dx == 0 && dy == 0 && dz == 0)) {
            const int agent_type = msg.getVariable<int>("agent_type");

            // Find direction index
            int dir_idx = -1;
            for (int i = 0; i < 26; i++) {
                int ddx, ddy, ddz;
                get_moore_direction_mdsc(i, ddx, ddy, ddz);
                if (ddx == dx && ddy == dy && ddz == dz) {
                    dir_idx = i;
                    break;
                }
            }

            if (dir_idx >= 0) {
                if (agent_type == CELL_TYPE_CANCER) {
                    cancer_count++;
                } else if (agent_type == CELL_TYPE_T) {
                    tcell_count++;
                    neighbor_tcells[dir_idx]++;
                } else if (agent_type == CELL_TYPE_TREG) {
                    treg_count++;
                } else if (agent_type == CELL_TYPE_MDSC) {
                    mdsc_count++;
                    neighbor_blocked[dir_idx] = true;
                }
            }
        }
    }

    // Build available_neighbors mask (voxels without MDSC - since 1 MDSC per voxel max)
    unsigned int available_neighbors = 0;
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction_mdsc(i, dx, dy, dz);
        int nx = my_x + dx;
        int ny = my_y + dy;
        int nz = my_z + dz;

        if (is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) {
            // MDSC can move to voxel only if no other MDSC is there
            if (!neighbor_blocked[i]) {
                available_neighbors |= (1u << i);
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_Tcell_count", tcell_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_MDSC_count", mdsc_count);
    FLAMEGPU->setVariable<unsigned int>("available_neighbors", available_neighbors);

    return flamegpu::ALIVE;
}

// MDSC agent function: State step (life countdown only - MDSCs don't divide)
FLAMEGPU_AGENT_FUNCTION(mdsc_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    return flamegpu::ALIVE;
}

// Helper: Compare two agents for priority (lower wins)
__device__ __forceinline__ bool has_higher_priority_mdsc(unsigned int id1, int sx1, int sy1, int sz1,
                                                          unsigned int id2, int sx2, int sy2, int sz2) {
    if (id1 != id2) return id1 < id2;
    if (sx1 != sx2) return sx1 < sx2;
    if (sy1 != sy2) return sy1 < sy2;
    return sz1 < sz2;
}

// MDSC agent function: Select movement target and broadcast intent
// Phase 1 of two-phase conflict resolution
// Uses cached available_neighbors mask from scan phase
FLAMEGPU_AGENT_FUNCTION(mdsc_select_move_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // Clear previous intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");

    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
        FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->message_out.setVariable<int>("target_x", -1);
        FLAMEGPU->message_out.setVariable<int>("target_y", -1);
        FLAMEGPU->message_out.setVariable<int>("target_z", -1);
        FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
        FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
        FLAMEGPU->message_out.setVariable<int>("source_z", my_z);
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
        return flamegpu::ALIVE;
    }

    //TODO: Add ECM Check for movement probability
    float ECM_sat = 0.2;

    int target_x = -1, target_y = -1, target_z = -1;
    int intent_action = INTENT_NONE;

    if (FLAMEGPU->random.uniform<float>() < ECM_sat) {
        // Use cached available_neighbors mask, restricted to Von Neumann (6 face neighbors)
        const unsigned int available_all = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
        const unsigned int available = available_all & VON_NEUMANN_MASK_MDSC;
        int num_available = __popc(available);

        if (num_available > 0) {
            int selected = FLAMEGPU->random.uniform<int>(0, num_available - 1);
            int count = 0;
            // Only iterate over Von Neumann directions (indices 0-5)
            for (int i = 0; i < 6; i++) {
                if (available & (1u << i)) {
                    if (count == selected) {
                        int dx, dy, dz;
                        get_moore_direction_mdsc(i, dx, dy, dz);
                        target_x = my_x + dx;
                        target_y = my_y + dy;
                        target_z = my_z + dz;
                        intent_action = INTENT_MOVE;
                        break;
                    }
                    count++;
                }
            }

            FLAMEGPU->setVariable<int>("intent_action", intent_action);
            FLAMEGPU->setVariable<int>("target_x", target_x);
            FLAMEGPU->setVariable<int>("target_y", target_y);
            FLAMEGPU->setVariable<int>("target_z", target_z);
        }
    }

    // Broadcast intent message with source position for conflict resolution
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("intent_action", intent_action);
    FLAMEGPU->message_out.setVariable<int>("target_x", target_x);
    FLAMEGPU->message_out.setVariable<int>("target_y", target_y);
    FLAMEGPU->message_out.setVariable<int>("target_z", target_z);
    FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
    FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
    FLAMEGPU->message_out.setVariable<int>("source_z", my_z);

    if (intent_action != INTENT_NONE) {
        FLAMEGPU->message_out.setLocation(
            (target_x + 0.5f) * voxel_size,
            (target_y + 0.5f) * voxel_size,
            (target_z + 0.5f) * voxel_size
        );
    } else {
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
    }

    return flamegpu::ALIVE;
}

// MDSC agent function: Execute movement if won conflict
// Phase 2 of two-phase conflict resolution
// Since 1 MDSC per voxel, only highest priority agent wins
FLAMEGPU_AGENT_FUNCTION(mdsc_execute_move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int intent_action = FLAMEGPU->getVariable<int>("intent_action");

    if (intent_action != INTENT_MOVE) {
        return flamegpu::ALIVE;
    }

    const int target_x = FLAMEGPU->getVariable<int>("target_x");
    const int target_y = FLAMEGPU->getVariable<int>("target_y");
    const int target_z = FLAMEGPU->getVariable<int>("target_z");
    const unsigned int my_id = FLAMEGPU->getID();
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    const float target_pos_x = (target_x + 0.5f) * voxel_size;
    const float target_pos_y = (target_y + 0.5f) * voxel_size;
    const float target_pos_z = (target_z + 0.5f) * voxel_size;

    // Check if any other MDSC/Cancer with higher priority also wants this voxel
    bool can_move = true;
    for (const auto& msg : FLAMEGPU->message_in(target_pos_x, target_pos_y, target_pos_z)) {
        const int msg_target_x = msg.getVariable<int>("target_x");
        const int msg_target_y = msg.getVariable<int>("target_y");
        const int msg_target_z = msg.getVariable<int>("target_z");

        if (msg_target_x == target_x && msg_target_y == target_y && msg_target_z == target_z) {
            const int msg_agent_type = msg.getVariable<int>("agent_type");
            const unsigned int msg_id = msg.getVariable<unsigned int>("agent_id");
            const int msg_intent = msg.getVariable<int>("intent_action");
            const int msg_src_x = msg.getVariable<int>("source_x");
            const int msg_src_y = msg.getVariable<int>("source_y");
            const int msg_src_z = msg.getVariable<int>("source_z");

            // Only 1 MDSC per voxel - higher priority wins
            if ((msg_agent_type == CELL_TYPE_MDSC) && msg_intent == INTENT_MOVE) {
                // Skip self
                if (msg_id == my_id) {
                    continue;
                }
                // Check if other agent has higher priority
                if (has_higher_priority_mdsc(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                              my_id, my_x, my_y, my_z)) {
                    can_move = false;
                    break;
                }
            }
        }
    }

    if (can_move) {
        FLAMEGPU->setVariable<int>("x", target_x);
        FLAMEGPU->setVariable<int>("y", target_y);
        FLAMEGPU->setVariable<int>("z", target_z);
    }

    // Clear intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}

// Occupancy Grid
FLAMEGPU_AGENT_FUNCTION(mdsc_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ[x][y][z][CELL_TYPE_MDSC].exchange(1u);  // Exclusive (MAX_MDSC_PER_VOXEL = 1)
    return flamegpu::ALIVE;
}

// MDSC agent function: Update chemicals from PDE
FLAMEGPU_AGENT_FUNCTION(mdsc_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // ========== READ CHEMICAL CONCENTRATIONS FROM AGENT VARIABLES ==========
    // These were already set by the host function update_agent_chemicals in layer 6
    float local_IFNg = FLAMEGPU->getVariable<float>("local_IFNg");
    
    // ========== COMPUTE DERIVED STATES ==========

    const float IFNg_PDL1_EC50 = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_HALF");
    const float IFNg_PDL1_hill = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_N");
    float H_IFNg = hill_equation_mdsc(local_IFNg, IFNg_PDL1_EC50, IFNg_PDL1_hill);
    const float PDL1_syn_max = FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX");
    float minPDL1 = PDL1_syn_max * H_IFNg;

    float PDL1_current = FLAMEGPU->getVariable<float>("PDL1_syn");
    if (PDL1_current < minPDL1) {
        FLAMEGPU->setVariable<float>("PDL1_syn", minPDL1);
    }
    
    return flamegpu::ALIVE;
}

// MDSC agent function: Compute chemical sources
FLAMEGPU_AGENT_FUNCTION(mdsc_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");
    
    // Dead cells don't produce
    if (dead == 1) {
        FLAMEGPU->setVariable<float>("ArgI_release_rate", 0.0f);
        FLAMEGPU->setVariable<float>("NO_release_rate", 0.0f);
        FLAMEGPU->setVariable<float>("CCL2_uptake_rate", 0.0f);
        return flamegpu::ALIVE;
    }

    const float ArgI_release = FLAMEGPU->environment.getProperty<float>("PARAM_ARGI_RELEASE");
    FLAMEGPU->setVariable<float>("ArgI_release_rate", ArgI_release);

    const float NO_release = FLAMEGPU->environment.getProperty<float>("PARAM_NO_RELEASE");
    FLAMEGPU->setVariable<float>("NO_release_rate", NO_release);

    const float CCL2_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_UPTAKE");
    FLAMEGPU->setVariable<float>("CCL2_uptake_rate", -CCL2_uptake);
    
    return flamegpu::ALIVE;
}

// Single-phase MDSC movement using occupancy grid.
// Replaces two-phase select_move_target + execute_move.
// MDSCs are exclusive per voxel (CAS) but can share voxels with cancer cells.
FLAMEGPU_AGENT_FUNCTION(mdsc_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    // ECM saturation: 20% chance to skip movement this step
    if (FLAMEGPU->random.uniform<float>() < 0.2f) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // Von Neumann neighbor offsets
    const int ddx[6] = {-1, 1,  0, 0,  0, 0};
    const int ddy[6] = { 0, 0, -1, 1,  0, 0};
    const int ddz[6] = { 0, 0,  0, 0, -1, 1};

    // Build candidate list: neighbors with no MDSC (MDSCs can share with cancer)
    int cands[6];
    int n_cands = 0;
    for (int i = 0; i < 6; i++) {
        int nx = x + ddx[i];
        int ny = y + ddy[i];
        int nz = z + ddz[i];
        if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y || nz < 0 || nz >= size_z) continue;
        if (occ[nx][ny][nz][CELL_TYPE_MDSC] == 0u) {
            cands[n_cands++] = i;
        }
    }
    if (n_cands == 0) return flamegpu::ALIVE;

    // Fisher-Yates shuffle for random candidate order
    for (int i = n_cands - 1; i > 0; i--) {
        int j = FLAMEGPU->random.uniform<int>(0, i);
        int tmp = cands[i]; cands[i] = cands[j]; cands[j] = tmp;
    }

    // Try candidates in shuffled order; CAS to atomically claim new voxel
    for (int i = 0; i < n_cands; i++) {
        int idx = cands[i];
        int nx = x + ddx[idx];
        int ny = y + ddy[idx];
        int nz = z + ddz[idx];
        unsigned int old = occ[nx][ny][nz][CELL_TYPE_MDSC].CAS(0u, 1u);
        if (old == 0u) {
            // Won the voxel — release old and update position
            occ[x][y][z][CELL_TYPE_MDSC].exchange(0u);
            FLAMEGPU->setVariable<int>("x", nx);
            FLAMEGPU->setVariable<int>("y", ny);
            FLAMEGPU->setVariable<int>("z", nz);
            break;
        }
    }
    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_MDSC_CUH