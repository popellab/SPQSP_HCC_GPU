#ifndef __CancerVCT_QSPPARAM__
#define __CancerVCT_QSPPARAM__

#include "../../core/ParamBase.h"
#include <string>
#include <map>

namespace CancerVCT {

// ============================================================================
// ENUM-DRIVEN PARAMETER SYSTEM
// ============================================================================
// Key insight: The enum ORDER is the source of truth for indices.
// Add new parameters ANYWHERE in the enum - the XML reader will assign
// them the correct index based on their position in the enum.
//
// Benefits:
// - Adding a parameter doesn't require updating all other indices
// - Parameters are grouped logically, not by Param.cpp order
// - Enum order = Vector index automatically
// - No hardcoded index values needed!

enum QSPParamFloat {
    //simulation settings
    QSP_SIM_START,
    QSP_SIM_STEP,
    QSP_SIM_N_STEP,
    QSP_SIM_TOL_REL,
    QSP_SIM_TOL_ABS,
    //compartments
    QSP_V_C,
    QSP_V_P,
    QSP_V_T,
    QSP_V_LN,
    QSP_V_e,
    QSP_A_e,
    QSP_A_s,
    QSP_syn_T_C1,
    QSP_syn_T_APC,
    QSP_syn_M_C,
    //species
    QSP_V_C_nT0,
    QSP_V_C_T0,
    QSP_V_C_nT1,
    QSP_V_C_T1,
    QSP_V_C_aPD1,
    QSP_V_C_aPDL1,
    QSP_V_C_aCTLA4,
    QSP_V_C_Th,
    QSP_V_C_cabozantinib,
    QSP_V_C_A_site1,
    QSP_V_C_A_site2,
    QSP_V_P_nT0,
    QSP_V_P_T0,
    QSP_V_P_nT1,
    QSP_V_P_T1,
    QSP_V_P_aPD1,
    QSP_V_P_aPDL1,
    QSP_V_P_aCTLA4,
    QSP_V_P_Th,
    QSP_V_P_cabozantinib,
    QSP_V_T_C_x,
    QSP_V_T_T1_exh,
    QSP_V_T_Th_exh,
    QSP_V_T_C1,
    QSP_V_T_K,
    QSP_V_T_c_vas,
    QSP_V_T_C2,
    QSP_V_T_T0,
    QSP_V_T_T1,
    QSP_V_T_IFNg,
    QSP_V_T_APC,
    QSP_V_T_mAPC,
    QSP_V_T_P0,
    QSP_V_T_P1,
    QSP_V_T_aPD1,
    QSP_V_T_aPDL1,
    QSP_V_T_aCTLA4,
    QSP_V_T_Th,
    QSP_V_T_TGFb,
    QSP_V_T_MDSC,
    QSP_V_T_NO,
    QSP_V_T_ArgI,
    QSP_V_T_CCL2,
    QSP_V_T_cabozantinib,
    QSP_V_T_Mac_M1,
    QSP_V_T_Mac_M2,
    QSP_V_T_IL12,
    QSP_V_T_IL10,
    QSP_V_T_Fib,
    QSP_V_T_CAF,
    QSP_V_T_ECM,

    QSP_V_LN_nT0,
    QSP_V_LN_aT0,
    QSP_V_LN_T0,
    QSP_V_LN_IL2,
    QSP_V_LN_nT1,
    QSP_V_LN_aT1,
    QSP_V_LN_T1,
    QSP_V_LN_APC,
    QSP_V_LN_mAPC,
    QSP_V_LN_aPD1,
    QSP_V_LN_aPDL1,
    QSP_V_LN_aCTLA4,
    QSP_V_LN_aTh,
    QSP_V_LN_Th,
    QSP_V_LN_cabozantinib,

    QSP_V_e_P0,
    QSP_V_e_p0,
    QSP_V_e_P1,
    QSP_V_e_p1,

    QSP_A_e_M1,
    QSP_A_e_M1p0,
    QSP_A_e_M1p1,

    QSP_A_s_M1,
    QSP_A_s_M1p0,
    QSP_A_s_M1p1,

    QSP_syn_T_C1_PD1_PDL1,
    QSP_syn_T_C1_PD1_PDL2,
    QSP_syn_T_C1_PD1,
    QSP_syn_T_C1_PDL1,
    QSP_syn_T_C1_PDL2,
    QSP_syn_T_C1_PD1_aPD1,
    QSP_syn_T_C1_PD1_aPD1_PD1,
    QSP_syn_T_C1_PDL1_aPDL1,
    QSP_syn_T_C1_PDL1_aPDL1_PDL1,
    QSP_syn_T_C1_TPDL1,
    QSP_syn_T_C1_TPDL1_aPDL1,
    QSP_syn_T_C1_TPDL1_aPDL1_TPDL1,
    QSP_syn_T_C1_CD28_CD80,
    QSP_syn_T_C1_CD28_CD80_CD28,
    QSP_syn_T_C1_CD28_CD86,
    QSP_syn_T_C1_CD80_CTLA4,
    QSP_syn_T_C1_CD80_CTLA4_CD80,
    QSP_syn_T_C1_CTLA4_CD80_CTLA4,
    QSP_syn_T_C1_CD80_CTLA4_CD80_CTLA4,
    QSP_syn_T_C1_CD86_CTLA4,
    QSP_syn_T_C1_CD86_CTLA4_CD86,
    QSP_syn_T_C1_PDL1_CD80,
    QSP_syn_T_C1_PDL1_CD80_CD28,
    QSP_syn_T_C1_PDL1_CD80_CTLA4,
    QSP_syn_T_C1_CD28,
    QSP_syn_T_C1_CTLA4,
    QSP_syn_T_C1_CD80,
    QSP_syn_T_C1_CD80m,
    QSP_syn_T_C1_CD86,
    QSP_syn_T_C1_CTLA4_aCTLA4,
    QSP_syn_T_C1_CTLA4_aCTLA4_CTLA4,

    QSP_syn_T_APC_PD1_PDL1,
    QSP_syn_T_APC_PD1_PDL2,
    QSP_syn_T_APC_PD1,
    QSP_syn_T_APC_PDL1,
    QSP_syn_T_APC_PDL2,
    QSP_syn_T_APC_PD1_aPD1,
    QSP_syn_T_APC_PD1_aPD1_PD1,
    QSP_syn_T_APC_PDL1_aPDL1,
    QSP_syn_T_APC_PDL1_aPDL1_PDL1,
    QSP_syn_T_APC_TPDL1,
    QSP_syn_T_APC_TPDL1_aPDL1,
    QSP_syn_T_APC_TPDL1_aPDL1_TPDL1,
    QSP_syn_T_APC_CD28_CD80,
    QSP_syn_T_APC_CD28_CD80_CD28,
    QSP_syn_T_APC_CD28_CD86,
    QSP_syn_T_APC_CD80_CTLA4,
    QSP_syn_T_APC_CD80_CTLA4_CD80,
    QSP_syn_T_APC_CTLA4_CD80_CTLA4,
    QSP_syn_T_APC_CD80_CTLA4_CD80_CTLA4,
    QSP_syn_T_APC_CD86_CTLA4,
    QSP_syn_T_APC_CD86_CTLA4_CD86,
    QSP_syn_T_APC_PDL1_CD80,
    QSP_syn_T_APC_PDL1_CD80_CD28,
    QSP_syn_T_APC_PDL1_CD80_CTLA4,
    QSP_syn_T_APC_CD28,
    QSP_syn_T_APC_CTLA4,
    QSP_syn_T_APC_CD80,
    QSP_syn_T_APC_CD80m,
    QSP_syn_T_APC_CD86,
    QSP_syn_T_APC_CTLA4_aCTLA4,
    QSP_syn_T_APC_CTLA4_aCTLA4_CTLA4,

    QSP_syn_M_C_CD47,
    QSP_syn_M_C_SIRPa,
    QSP_syn_M_C_CD47_SIRPa,
    QSP_syn_M_C_PDL1_total,
    QSP_syn_M_C_PDL2_total,
    QSP_syn_M_C_PD1_PDL1,
    QSP_syn_M_C_PD1_PDL2,
    QSP_syn_M_C_PD1,
    QSP_syn_M_C_PDL1,
    QSP_syn_M_C_PDL2,
    QSP_syn_M_C_PD1_aPD1,
    QSP_syn_M_C_PD1_aPD1_PD1,
    QSP_syn_M_C_PDL1_aPDL1,
    QSP_syn_M_C_PDL1_aPDL1_PDL1,
    QSP_syn_M_C_PDL1_CD80,
    QSP_syn_M_C_CD80,
    QSP_syn_M_C_CD80m,

    QSP_k_cell_clear,
    QSP_cell,
    QSP_day,
    QSP_vol_cell,
    QSP_vol_Tcell,
    QSP_Ve_T,
    QSP_C_total,
    QSP_T_total,
    QSP_T_total_LN,
    QSP_R_Tcell,
    QSP_Tregs_,
    QSP_H_APC,
    QSP_H_mAPC,
    QSP_H_APCh,
    QSP_H_PD1_C1,
    QSP_H_PD1_APC,
    QSP_H_CD28_C1,
    QSP_H_CD28_APC,
    QSP_k_C1_growth,
    QSP_C_max,
    QSP_k_C1_death,
    QSP_k_C1_therapy,
    QSP_initial_tumour_diameter,
    QSP_k_K_g,
    QSP_k_K_d,
    QSP_k_vas_Csec,
    QSP_k_vas_deg,
    QSP_c_vas_50,
    QSP_k_C2_growth,
    QSP_k_C2_death,
    QSP_k_C2_therapy,
    QSP_div_T0,
    QSP_n_T0_clones,
    QSP_q_nT0_LN_in,
    QSP_q_T0_LN_out,
    QSP_q_nT0_LN_out,
    QSP_k_T0_act,
    QSP_k_T0_pro,
    QSP_k_T0_death,
    QSP_q_T0_P_in,
    QSP_q_T0_P_out,
    QSP_q_T0_T_in,
    QSP_q_nT0_P_in,
    QSP_q_nT0_P_out,
    QSP_Q_nT0_thym,
    QSP_k_nT0_pro,
    QSP_K_nT0_pro,
    QSP_k_nT0_death,
    QSP_k_IL2_deg,
    QSP_k_IL2_cons,
    QSP_k_IL2_sec,
    QSP_IL2_50,
    QSP_IL2_50_Treg,
    QSP_N0,
    QSP_N_costim,
    QSP_N_IL2_CD8,
    QSP_N_IL2_CD4,
    QSP_k_Treg,
    QSP_H_P0,
    QSP_N_aT,
    QSP_N_aT0,
    QSP_div_T1,
    QSP_n_T1_clones,
    QSP_q_nT1_LN_in,
    QSP_q_T1_LN_out,
    QSP_q_nT1_LN_out,
    QSP_k_T1_act,
    QSP_k_T1_pro,
    QSP_k_T1_death,
    QSP_q_T1_P_in,
    QSP_q_T1_P_out,
    QSP_q_T1_T_in,
    QSP_q_nT1_P_in,
    QSP_q_nT1_P_out,
    QSP_Q_nT1_thym,
    QSP_k_nT1_pro,
    QSP_K_nT1_pro,
    QSP_k_nT1_death,
    QSP_k_T1,
    QSP_k_C_T1,
    QSP_k_Tcell_ECM,
    QSP_H_ECM_T_mot,
    QSP_H_ECM_T_exh,
    QSP_k_IFNg_Tsec,
    QSP_H_P1,
    QSP_K_T_C,
    QSP_K_T_Treg,
    QSP_k_APC_mat,
    QSP_k_APC_mig,
    QSP_k_APC_death,
    QSP_k_mAPC_death,
    QSP_APC0_T,
    QSP_APC0_LN,
    QSP_n_sites_APC,
    QSP_kin,
    QSP_kout,
    QSP_k_P0_up,
    QSP_k_xP0_deg,
    QSP_k_P0_deg,
    QSP_k_p0_deg,
    QSP_k_P0_on,
    QSP_k_P0_d1,
    QSP_p0_50,
    QSP_P0_C1,
    QSP_P0_C2,
    QSP_A_syn,
    QSP_A_Tcell,
    QSP_A_cell,
    QSP_A_APC,
    QSP_k_M1p0_TCR_on,
    QSP_k_M1p0_TCR_off,
    QSP_k_M1p0_TCR_p,
    QSP_phi_M1p0_TCR,
    QSP_N_M1p0_TCR,
    QSP_TCR_p0_tot,
    QSP_pTCR_p0_MHC_tot,
    QSP_k_P1_up,
    QSP_k_xP1_deg,
    QSP_k_P1_deg,
    QSP_k_p1_deg,
    QSP_k_P1_on,
    QSP_k_P1_d1,
    QSP_p1_50,
    QSP_P1_C1,
    QSP_P1_C2,
    QSP_k_M1p1_TCR_on,
    QSP_k_M1p1_TCR_off,
    QSP_k_M1p1_TCR_p,
    QSP_phi_M1p1_TCR,
    QSP_N_M1p1_TCR,
    QSP_TCR_p1_tot,
    QSP_pTCR_p1_MHC_tot,
    QSP_q_P_aPD1,
    QSP_q_T_aPD1,
    QSP_q_LN_aPD1,
    QSP_q_LD_aPD1,
    QSP_k_cl_aPD1,
    QSP_gamma_C_aPD1,
    QSP_gamma_P_aPD1,
    QSP_gamma_T_aPD1,
    QSP_gamma_LN_aPD1,
    QSP_q_P_aPDL1,
    QSP_q_T_aPDL1,
    QSP_q_LN_aPDL1,
    QSP_q_LD_aPDL1,
    QSP_k_cl_aPDL1,
    QSP_gamma_C_aPDL1,
    QSP_gamma_P_aPDL1,
    QSP_gamma_T_aPDL1,
    QSP_gamma_LN_aPDL1,
    QSP_k_cln_aPDL1,
    QSP_Kc_aPDL1,
    QSP_q_P_aCTLA4,
    QSP_q_T_aCTLA4,
    QSP_q_LN_aCTLA4,
    QSP_q_LD_aCTLA4,
    QSP_k_cl_aCTLA4,
    QSP_gamma_C_aCTLA4,
    QSP_gamma_P_aCTLA4,
    QSP_gamma_T_aCTLA4,
    QSP_gamma_LN_aCTLA4,
    QSP_kon_PD1_PDL1,
    QSP_k_out_PDL1,
    QSP_k_in_PDL1,
    QSP_r_PDL1_IFNg,
    QSP_kon_PD1_PDL2,
    QSP_kon_PD1_aPD1,
    QSP_kon_PDL1_aPDL1,
    QSP_kon_CD28_CD80,
    QSP_kon_CD28_CD86,
    QSP_kon_CTLA4_CD80,
    QSP_kon_CTLA4_CD86,
    QSP_kon_CD80_PDL1,
    QSP_kon_CTLA4_aCTLA4,
    QSP_kon_CD80_CD80,
    QSP_koff_PD1_PDL1,
    QSP_koff_PD1_PDL2,
    QSP_koff_PD1_aPD1,
    QSP_koff_PDL1_aPDL1,
    QSP_koff_CD28_CD80,
    QSP_koff_CD28_CD86,
    QSP_koff_CTLA4_CD80,
    QSP_koff_CTLA4_CD86,
    QSP_koff_CD80_PDL1,
    QSP_koff_CTLA4_aCTLA4,
    QSP_koff_CD80_CD80,
    QSP_Chi_PD1_aPD1,
    QSP_Chi_PDL1_aPDL1,
    QSP_Chi_CTLA4_aCTLA4,
    QSP_PD1_50,
    QSP_n_PD1,
    QSP_CD28_CD8X_50,
    QSP_n_CD28_CD8X,
    QSP_T_PD1_total,
    QSP_T_CD28_total,
    QSP_T_CTLA4_syn,
    QSP_T_PDL1_total,
    QSP_C1_PDL1_base,
    QSP_r_PDL2C1,
    QSP_C1_CD80_total,
    QSP_C1_CD86_total,
    QSP_syn_T_C1_PDL1_total,
    QSP_syn_T_C1_PDL2_total,
    QSP_APC_PDL1_base,
    QSP_r_PDL2APC,
    QSP_APC_CD80_total,
    QSP_APC_CD86_total,
    QSP_syn_T_APC_PDL1_total,
    QSP_syn_T_APC_PDL2_total,
    QSP_k_Th_act,
    QSP_k_Th_Treg,
    QSP_k_TGFb_Tsec,
    QSP_k_TGFb_deg,
    QSP_TGFb_50,
    QSP_TGFb_50_Teff,
    QSP_Kc_rec,
    QSP_TGFbase,
    QSP_k_IFNg_Thsec,
    QSP_k_IFNg_deg,
    QSP_IFNg_50_ind,
    QSP_H_TGFb,
    QSP_H_TGFb_Teff,
    QSP_N_aTh,
    QSP_k_CCL2_sec,
    QSP_k_CCL2_deg,
    QSP_CCL2_50,
    QSP_k_MDSC_rec,
    QSP_k_MDSC_death,
    QSP_k_NO_deg,
    QSP_k_ArgI_deg,
    QSP_k_NO_sec,
    QSP_k_ArgI_sec,
    QSP_ArgI_50_Teff,
    QSP_NO_50_Teff,
    QSP_ArgI_50_Treg,
    QSP_H_NO,
    QSP_H_ArgI_Teff,
    QSP_H_ArgI_Treg,
    QSP_H_MDSC,
    QSP_k_a1_cabozantinib,
    QSP_k_a2_cabozantinib,
    QSP_k_cln_cabozantinib,
    QSP_Kc_cabozantinib,
    QSP_lagP1_cabozantinib,
    QSP_lagP2_cabozantinib,
    QSP_F_cabozantinib,
    QSP_q_P_cabozantinib,
    QSP_q_T_cabozantinib,
    QSP_q_LN_cabozantinib,
    QSP_q_LD_cabozantinib,
    QSP_gamma_C_cabozantinib,
    QSP_gamma_P_cabozantinib,
    QSP_gamma_T_cabozantinib,
    QSP_gamma_LN_cabozantinib,
    QSP_IC50_MET,
    QSP_IC50_RET,
    QSP_IC50_AXL,
    QSP_IC50_VEGFR2,
    QSP_k_C_resist,
    QSP_k_K_cabo,
    QSP_R_cabo,
    QSP_H_therapy_cabo,
    QSP_k_Mac_rec,
    QSP_k_Mac_death,
    QSP_k_TGFb_Msec,
    QSP_k_vas_Msec,
    QSP_k_IL12_sec,
    QSP_k_IL12_Msec,
    QSP_k_IL12_deg,
    QSP_k_IL10_sec,
    QSP_k_IL10_deg,
    QSP_k_M2_pol,
    QSP_k_M1_pol,
    QSP_IL10_50,
    QSP_IL12_50,
    QSP_IFNg_50,
    QSP_k_M1_phago,
    QSP_vol_Mcell,
    QSP_kon_CD47_SIRPa,
    QSP_koff_CD47_SIRPa,
    QSP_SIRPa_50,
    QSP_n_SIRPa,
    QSP_H_Mac_C,
    QSP_H_PD1_M,
    QSP_H_SIRPa,
    QSP_C_CD47,
    QSP_M_PD1_total,
    QSP_M_SIRPa,
    QSP_A_Mcell,
    QSP_IL10_50_phago,
    QSP_K_Mac_C,
    QSP_M_total,
    QSP_H_IL10,
    QSP_H_IL10_phago,
    QSP_H_IL12,
    QSP_k_fib_rec,
    QSP_k_fib_const,
    QSP_k_caf_tran,
    QSP_k_ECM_fib_sec,
    QSP_k_ECM_CAF_sec,
    QSP_k_ECM_deg,
    QSP_ECM_base,
    QSP_ECM_max,
    QSP_ECM_level,
    QSP_ECM_MW,
    QSP_ECM_density,
    QSP_k_fib_death,
    QSP_k_CAF_death,
    QSP_ECM_50_T_exh,
    QSP_ECM_50_T_mot,
    QSP_vol_Fibcell,
    QSP_vol_CAFcell,



    QSP_PARAM_FLOAT_COUNT       // Sentinel value = total count
};

enum QSPParamInt {
    // Add integer parameters if needed
    QSP_PARAM_INT_COUNT = 0
};

// ============================================================================
// XML PATH MAPPINGS - Maps enum value to XML path
// ============================================================================
// Add one entry per enum value in the SAME ORDER as the enum definition.
// When you add a parameter to the enum, add a corresponding XML path here.
// The XML reader will use the enum position to determine vector index.

extern const char* QSP_PARAM_FLOAT_XML_PATHS[];

// ============================================================================
// QSPPARAM CLASS - ENUM-DRIVEN
// ============================================================================

class QSPParam : public SP_QSP_IO::ParamBase {
public:
    QSPParam();
    ~QSPParam() {}

    // ========================================================================
    // TYPE-SAFE ENUM-BASED ACCESSORS
    // ========================================================================
    // Enum value automatically maps to vector index
    // Usage: double ifng = qsp_params.getFloat(QSP_V_T_IFNg);
    //        -> Automatically uses index 9 based on enum order

    inline double getFloat(QSPParamFloat idx) const {
        if (idx >= QSP_PARAM_FLOAT_COUNT) return 0.0;
        return _paramFloat[idx];  // idx IS the vector index!
    }

    inline int getInt(QSPParamInt idx) const {
        if (idx >= QSP_PARAM_INT_COUNT) return 0;
        return _paramInt[idx];
    }

    inline double getVal(int n) const {
        QSPParamFloat idx = QSPParamFloat(n);  // Convert int to enum (assuming valid)
        return _paramFloat[idx];  // idx IS the vector index!
    }

    // ========================================================================
    // CONVENIENCE METHODS
    // ========================================================================

    inline double getCompartmentVolume_Central() const {
        return getFloat(QSP_V_C);
    }
    inline double getCompartmentVolume_Tumor() const {
        return getFloat(QSP_V_T);
    }
    inline double getCompartmentVolume_LymphNode() const {
        return getFloat(QSP_V_LN);
    }

    inline double getTumorTCell_Naive() const {
        return getFloat(QSP_V_T_T0);
    }
    inline double getTumorTCell_Activated() const {
        return getFloat(QSP_V_T_T1);
    }
    inline double getTumorTCell_Cytotoxic() const {
        return getFloat(QSP_V_T_C1);
    }

    inline double getTumorCytokine_IFNg() const {
        return getFloat(QSP_V_T_IFNg);
    }
    /*
    inline double getTumorCytokine_IL2() const {
        return getFloat(QSP_V_T_IL2);
    } 
    */
    inline double getTumorCytokine_IL10() const {
        return getFloat(QSP_V_T_IL10);
    }
    /*
    inline double getTumorCytokine_TGFB() const {
        return getFloat(QSP_V_T_TGFB);
    }
    */
    inline double getTumorCytokine_IL12() const {
        return getFloat(QSP_V_T_IL12);
    }

    inline double getTumorImmune_MDSC() const {
        return getFloat(QSP_V_T_MDSC);
    }
    inline double getTumorImmune_ArgI() const {
        return getFloat(QSP_V_T_ArgI);
    }
    inline double getTumorImmune_NO() const {
        return getFloat(QSP_V_T_NO);
    }
    /*
    inline double getDrug_NIVO_Tumor() const {
        return getFloat(QSP_V_T_NIVO);
    }
    inline double getDrug_CABO_Tumor() const {
        return getFloat(QSP_V_T_CABO);
    }
    */

private:
    virtual void setupParam() override;
    virtual void processInternalParams() override;
    virtual bool readParamsFromXml(std::string inFileName) override;
};

} // namespace CancerVCT

#endif
