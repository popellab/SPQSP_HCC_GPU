#include "QSPParam.h"
#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

// ============================================================================
// XML PATH MAPPINGS - Maps enum value to XML path
// ============================================================================
// Add one entry per enum value in the SAME ORDER as the enum definition.
// When you add a parameter to the enum, add a corresponding XML path here.
// The XML reader will use the enum position to determine vector index.

const char* CancerVCT::QSP_PARAM_FLOAT_XML_PATHS[] = {
    //simulation settings
    "Param.QSP.simulation.start",//0
    "Param.QSP.simulation.step",//1
    "Param.QSP.simulation.n_step",//2
    "Param.QSP.simulation.tol_rel",//3
    "Param.QSP.simulation.tol_abs",//4
    //compartments
    "Param.QSP.init_value.Compartment.V_C",//5|5
    "Param.QSP.init_value.Compartment.V_P",//6|6
    "Param.QSP.init_value.Compartment.V_T",//7|7
    "Param.QSP.init_value.Compartment.V_LN",//8|8
    "Param.QSP.init_value.Compartment.V_e",//9|9
    "Param.QSP.init_value.Compartment.A_e",//10|10
    "Param.QSP.init_value.Compartment.A_s",//11|11
    "Param.QSP.init_value.Compartment.syn_T_C1",//12|12
    "Param.QSP.init_value.Compartment.syn_T_APC",//13|13
    "Param.QSP.init_value.Compartment.syn_M_C",//14|14
    //species
    "Param.QSP.init_value.Species.V_C_nT0",//15|15
    "Param.QSP.init_value.Species.V_C_T0",//16|16
    "Param.QSP.init_value.Species.V_C_nT1",//17|17
    "Param.QSP.init_value.Species.V_C_T1",//18|18
    "Param.QSP.init_value.Species.V_C_aPD1",//19|19
    "Param.QSP.init_value.Species.V_C_aPDL1",//20|20
    "Param.QSP.init_value.Species.V_C_aCTLA4",//21|21
    "Param.QSP.init_value.Species.V_C_Th",//22|22
    "Param.QSP.init_value.Species.V_C_cabozantinib",//23|23
    "Param.QSP.init_value.Species.V_C_A_site1",//24|24
    "Param.QSP.init_value.Species.V_C_A_site2",//25|25
    "Param.QSP.init_value.Species.V_P_nT0",//26|26
    "Param.QSP.init_value.Species.V_P_T0",//27|27
    "Param.QSP.init_value.Species.V_P_nT1",//28|28
    "Param.QSP.init_value.Species.V_P_T1",//29|29
    "Param.QSP.init_value.Species.V_P_aPD1",//30|30
    "Param.QSP.init_value.Species.V_P_aPDL1",//31|31
    "Param.QSP.init_value.Species.V_P_aCTLA4",//32|32
    "Param.QSP.init_value.Species.V_P_Th",//33|33
    "Param.QSP.init_value.Species.V_P_cabozantinib",//34|34
    "Param.QSP.init_value.Species.V_T_C_x",//35|35
    "Param.QSP.init_value.Species.V_T_T1_exh",//36|36
    "Param.QSP.init_value.Species.V_T_Th_exh",//37|37
    "Param.QSP.init_value.Species.V_T_C1",//38|38
    "Param.QSP.init_value.Species.V_T_K",//39|39
    "Param.QSP.init_value.Species.V_T_c_vas",//40|40
    "Param.QSP.init_value.Species.V_T_C2",//41|41
    "Param.QSP.init_value.Species.V_T_T0",//42|42
    "Param.QSP.init_value.Species.V_T_T1",//43|43
    "Param.QSP.init_value.Species.V_T_IFNg",//44|44
    "Param.QSP.init_value.Species.V_T_APC",//45|45
    "Param.QSP.init_value.Species.V_T_mAPC",//46|46
    "Param.QSP.init_value.Species.V_T_P0",//47|47
    "Param.QSP.init_value.Species.V_T_P1",//48|48
    "Param.QSP.init_value.Species.V_T_aPD1",//49|49
    "Param.QSP.init_value.Species.V_T_aPDL1",//50|50
    "Param.QSP.init_value.Species.V_T_aCTLA4",//51|51
    "Param.QSP.init_value.Species.V_T_Th",//52|52
    "Param.QSP.init_value.Species.V_T_TGFb",//53|53
    "Param.QSP.init_value.Species.V_T_MDSC",//54|54
    "Param.QSP.init_value.Species.V_T_NO",//55|55
    "Param.QSP.init_value.Species.V_T_ArgI",//56|56
    "Param.QSP.init_value.Species.V_T_CCL2",//57|57
    "Param.QSP.init_value.Species.V_T_cabozantinib",//58|58
    "Param.QSP.init_value.Species.V_T_Mac_M1",//59|59
    "Param.QSP.init_value.Species.V_T_Mac_M2",//60|60
    "Param.QSP.init_value.Species.V_T_IL12",//61|61
    "Param.QSP.init_value.Species.V_T_IL10",//62|62
    "Param.QSP.init_value.Species.V_T_Fib",//63|63
    "Param.QSP.init_value.Species.V_T_CAF",//64|64
    "Param.QSP.init_value.Species.V_T_ECM",//65|65
    "Param.QSP.init_value.Species.V_LN_nT0",//66|66
    "Param.QSP.init_value.Species.V_LN_aT0",//67|67
    "Param.QSP.init_value.Species.V_LN_T0",//68|68
    "Param.QSP.init_value.Species.V_LN_IL2",//69|69
    "Param.QSP.init_value.Species.V_LN_nT1",//70|70
    "Param.QSP.init_value.Species.V_LN_aT1",//71|71
    "Param.QSP.init_value.Species.V_LN_T1",//72|72
    "Param.QSP.init_value.Species.V_LN_APC",//73|73
    "Param.QSP.init_value.Species.V_LN_mAPC",//74|74
    "Param.QSP.init_value.Species.V_LN_aPD1",//75|75
    "Param.QSP.init_value.Species.V_LN_aPDL1",//76|76
    "Param.QSP.init_value.Species.V_LN_aCTLA4",//77|77
    "Param.QSP.init_value.Species.V_LN_aTh",//78|78
    "Param.QSP.init_value.Species.V_LN_Th",//79|79
    "Param.QSP.init_value.Species.V_LN_cabozantinib",//80|80
    "Param.QSP.init_value.Species.V_e_P0",//81|81
    "Param.QSP.init_value.Species.V_e_p0",//82|82
    "Param.QSP.init_value.Species.V_e_P1",//83|83
    "Param.QSP.init_value.Species.V_e_p1",//84|84
    "Param.QSP.init_value.Species.A_e_M1",//85|85
    "Param.QSP.init_value.Species.A_e_M1p0",//86|86
    "Param.QSP.init_value.Species.A_e_M1p1",//87|87
    "Param.QSP.init_value.Species.A_s_M1",//88|88
    "Param.QSP.init_value.Species.A_s_M1p0",//89|89
    "Param.QSP.init_value.Species.A_s_M1p1",//90|90
    "Param.QSP.init_value.Species.syn_T_C1_PD1_PDL1",//91|91
    "Param.QSP.init_value.Species.syn_T_C1_PD1_PDL2",//92|92
    "Param.QSP.init_value.Species.syn_T_C1_PD1",//93|93
    "Param.QSP.init_value.Species.syn_T_C1_PDL1",//94|94
    "Param.QSP.init_value.Species.syn_T_C1_PDL2",//95|95
    "Param.QSP.init_value.Species.syn_T_C1_PD1_aPD1",//96|96
    "Param.QSP.init_value.Species.syn_T_C1_PD1_aPD1_PD1",//97|97
    "Param.QSP.init_value.Species.syn_T_C1_PDL1_aPDL1",//98|98
    "Param.QSP.init_value.Species.syn_T_C1_PDL1_aPDL1_PDL1",//99|99
    "Param.QSP.init_value.Species.syn_T_C1_TPDL1",//100|100
    "Param.QSP.init_value.Species.syn_T_C1_TPDL1_aPDL1",//101|101
    "Param.QSP.init_value.Species.syn_T_C1_TPDL1_aPDL1_TPDL1",//102|102
    "Param.QSP.init_value.Species.syn_T_C1_CD28_CD80",//103|103
    "Param.QSP.init_value.Species.syn_T_C1_CD28_CD80_CD28",//104|104
    "Param.QSP.init_value.Species.syn_T_C1_CD28_CD86",//105|105
    "Param.QSP.init_value.Species.syn_T_C1_CD80_CTLA4",//106|106
    "Param.QSP.init_value.Species.syn_T_C1_CD80_CTLA4_CD80",//107|107
    "Param.QSP.init_value.Species.syn_T_C1_CTLA4_CD80_CTLA4",//108|108
    "Param.QSP.init_value.Species.syn_T_C1_CD80_CTLA4_CD80_CTLA4",//109|109
    "Param.QSP.init_value.Species.syn_T_C1_CD86_CTLA4",//110|110
    "Param.QSP.init_value.Species.syn_T_C1_CD86_CTLA4_CD86",//111|111
    "Param.QSP.init_value.Species.syn_T_C1_PDL1_CD80",//112|112
    "Param.QSP.init_value.Species.syn_T_C1_PDL1_CD80_CD28",//113|113
    "Param.QSP.init_value.Species.syn_T_C1_PDL1_CD80_CTLA4",//114|114
    "Param.QSP.init_value.Species.syn_T_C1_CD28",//115|115
    "Param.QSP.init_value.Species.syn_T_C1_CTLA4",//116|116
    "Param.QSP.init_value.Species.syn_T_C1_CD80",//117|117
    "Param.QSP.init_value.Species.syn_T_C1_CD80m",//118|118
    "Param.QSP.init_value.Species.syn_T_C1_CD86",//119|119
    "Param.QSP.init_value.Species.syn_T_C1_CTLA4_aCTLA4",//120|120
    "Param.QSP.init_value.Species.syn_T_C1_CTLA4_aCTLA4_CTLA4",//121|121
    "Param.QSP.init_value.Species.syn_T_APC_PD1_PDL1",//122|122
    "Param.QSP.init_value.Species.syn_T_APC_PD1_PDL2",//123|123
    "Param.QSP.init_value.Species.syn_T_APC_PD1",//124|124
    "Param.QSP.init_value.Species.syn_T_APC_PDL1",//125|125
    "Param.QSP.init_value.Species.syn_T_APC_PDL2",//126|126
    "Param.QSP.init_value.Species.syn_T_APC_PD1_aPD1",//127|127
    "Param.QSP.init_value.Species.syn_T_APC_PD1_aPD1_PD1",//128|128
    "Param.QSP.init_value.Species.syn_T_APC_PDL1_aPDL1",//129|129
    "Param.QSP.init_value.Species.syn_T_APC_PDL1_aPDL1_PDL1",//130|130
    "Param.QSP.init_value.Species.syn_T_APC_TPDL1",//131|131
    "Param.QSP.init_value.Species.syn_T_APC_TPDL1_aPDL1",//132|132
    "Param.QSP.init_value.Species.syn_T_APC_TPDL1_aPDL1_TPDL1",//133|133
    "Param.QSP.init_value.Species.syn_T_APC_CD28_CD80",//134|134
    "Param.QSP.init_value.Species.syn_T_APC_CD28_CD80_CD28",//135|135
    "Param.QSP.init_value.Species.syn_T_APC_CD28_CD86",//136|136
    "Param.QSP.init_value.Species.syn_T_APC_CD80_CTLA4",//137|137
    "Param.QSP.init_value.Species.syn_T_APC_CD80_CTLA4_CD80",//138|138
    "Param.QSP.init_value.Species.syn_T_APC_CTLA4_CD80_CTLA4",//139|139
    "Param.QSP.init_value.Species.syn_T_APC_CD80_CTLA4_CD80_CTLA4",//140|140
    "Param.QSP.init_value.Species.syn_T_APC_CD86_CTLA4",//141|141
    "Param.QSP.init_value.Species.syn_T_APC_CD86_CTLA4_CD86",//142|142
    "Param.QSP.init_value.Species.syn_T_APC_PDL1_CD80",//143|143
    "Param.QSP.init_value.Species.syn_T_APC_PDL1_CD80_CD28",//144|144
    "Param.QSP.init_value.Species.syn_T_APC_PDL1_CD80_CTLA4",//145|145
    "Param.QSP.init_value.Species.syn_T_APC_CD28",//146|146
    "Param.QSP.init_value.Species.syn_T_APC_CTLA4",//147|147
    "Param.QSP.init_value.Species.syn_T_APC_CD80",//148|148
    "Param.QSP.init_value.Species.syn_T_APC_CD80m",//149|149
    "Param.QSP.init_value.Species.syn_T_APC_CD86",//150|150
    "Param.QSP.init_value.Species.syn_T_APC_CTLA4_aCTLA4",//151|151
    "Param.QSP.init_value.Species.syn_T_APC_CTLA4_aCTLA4_CTLA4",//152|152
    "Param.QSP.init_value.Species.syn_M_C_CD47",//153|153
    "Param.QSP.init_value.Species.syn_M_C_SIRPa",//154|154
    "Param.QSP.init_value.Species.syn_M_C_CD47_SIRPa",//155|155
    "Param.QSP.init_value.Species.syn_M_C_PDL1_total",//156|156
    "Param.QSP.init_value.Species.syn_M_C_PDL2_total",//157|157
    "Param.QSP.init_value.Species.syn_M_C_PD1_PDL1",//158|158
    "Param.QSP.init_value.Species.syn_M_C_PD1_PDL2",//159|159
    "Param.QSP.init_value.Species.syn_M_C_PD1",//160|160
    "Param.QSP.init_value.Species.syn_M_C_PDL1",//161|161
    "Param.QSP.init_value.Species.syn_M_C_PDL2",//162|162
    "Param.QSP.init_value.Species.syn_M_C_PD1_aPD1",//163|163
    "Param.QSP.init_value.Species.syn_M_C_PD1_aPD1_PD1",//164|164
    "Param.QSP.init_value.Species.syn_M_C_PDL1_aPDL1",//165|165
    "Param.QSP.init_value.Species.syn_M_C_PDL1_aPDL1_PDL1",//166|166
    "Param.QSP.init_value.Species.syn_M_C_PDL1_CD80",//167|167
    "Param.QSP.init_value.Species.syn_M_C_CD80",//168|168
    "Param.QSP.init_value.Species.syn_M_C_CD80m",//169|169
    //parameters
    "Param.QSP.init_value.Parameter.k_cell_clear",//170|170
    "Param.QSP.init_value.Parameter.cell",//171|171
    "Param.QSP.init_value.Parameter.day",//172|172
    "Param.QSP.init_value.Parameter.vol_cell",//173|173
    "Param.QSP.init_value.Parameter.vol_Tcell",//174|174
    "Param.QSP.init_value.Parameter.Ve_T",//175|175
    "Param.QSP.init_value.Parameter.C_total",//176|176
    "Param.QSP.init_value.Parameter.T_total",//177|177
    "Param.QSP.init_value.Parameter.T_total_LN",//178|178
    "Param.QSP.init_value.Parameter.R_Tcell",//179|179
    "Param.QSP.init_value.Parameter.Tregs_",//180|180
    "Param.QSP.init_value.Parameter.H_APC",//181|181
    "Param.QSP.init_value.Parameter.H_mAPC",//182|182
    "Param.QSP.init_value.Parameter.H_APCh",//183|183
    "Param.QSP.init_value.Parameter.H_PD1_C1",//184|184
    "Param.QSP.init_value.Parameter.H_PD1_APC",//185|185
    "Param.QSP.init_value.Parameter.H_CD28_C1",//186|186
    "Param.QSP.init_value.Parameter.H_CD28_APC",//187|187
    "Param.QSP.init_value.Parameter.k_C1_growth",//188|188
    "Param.QSP.init_value.Parameter.C_max",//189|189
    "Param.QSP.init_value.Parameter.k_C1_death",//190|190
    "Param.QSP.init_value.Parameter.k_C1_therapy",//191|191
    "Param.QSP.init_value.Parameter.initial_tumour_diameter",//192|192
    "Param.QSP.init_value.Parameter.k_K_g",//193|193
    "Param.QSP.init_value.Parameter.k_K_d",//194|194
    "Param.QSP.init_value.Parameter.k_vas_Csec",//195|195
    "Param.QSP.init_value.Parameter.k_vas_deg",//196|196
    "Param.QSP.init_value.Parameter.c_vas_50",//197|197
    "Param.QSP.init_value.Parameter.k_C2_growth",//198|198
    "Param.QSP.init_value.Parameter.k_C2_death",//199|199
    "Param.QSP.init_value.Parameter.k_C2_therapy",//200|200
    "Param.QSP.init_value.Parameter.div_T0",//201|201
    "Param.QSP.init_value.Parameter.n_T0_clones",//202|202
    "Param.QSP.init_value.Parameter.q_nT0_LN_in",//203|203
    "Param.QSP.init_value.Parameter.q_T0_LN_out",//204|204
    "Param.QSP.init_value.Parameter.q_nT0_LN_out",//205|205
    "Param.QSP.init_value.Parameter.k_T0_act",//206|206
    "Param.QSP.init_value.Parameter.k_T0_pro",//207|207
    "Param.QSP.init_value.Parameter.k_T0_death",//208|208
    "Param.QSP.init_value.Parameter.q_T0_P_in",//209|209
    "Param.QSP.init_value.Parameter.q_T0_P_out",//210|210
    "Param.QSP.init_value.Parameter.q_T0_T_in",//211|211
    "Param.QSP.init_value.Parameter.q_nT0_P_in",//212|212
    "Param.QSP.init_value.Parameter.q_nT0_P_out",//213|213
    "Param.QSP.init_value.Parameter.Q_nT0_thym",//214|214
    "Param.QSP.init_value.Parameter.k_nT0_pro",//215|215
    "Param.QSP.init_value.Parameter.K_nT0_pro",//216|216
    "Param.QSP.init_value.Parameter.k_nT0_death",//217|217
    "Param.QSP.init_value.Parameter.k_IL2_deg",//218|218
    "Param.QSP.init_value.Parameter.k_IL2_cons",//219|219
    "Param.QSP.init_value.Parameter.k_IL2_sec",//220|220
    "Param.QSP.init_value.Parameter.IL2_50",//221|221
    "Param.QSP.init_value.Parameter.IL2_50_Treg",//222|222
    "Param.QSP.init_value.Parameter.N0",//223|223
    "Param.QSP.init_value.Parameter.N_costim",//224|224
    "Param.QSP.init_value.Parameter.N_IL2_CD8",//225|225
    "Param.QSP.init_value.Parameter.N_IL2_CD4",//226|226
    "Param.QSP.init_value.Parameter.k_Treg",//227|227
    "Param.QSP.init_value.Parameter.H_P0",//228|228
    "Param.QSP.init_value.Parameter.N_aT",//229|229
    "Param.QSP.init_value.Parameter.N_aT0",//230|230
    "Param.QSP.init_value.Parameter.div_T1",//231|231
    "Param.QSP.init_value.Parameter.n_T1_clones",//232|232
    "Param.QSP.init_value.Parameter.q_nT1_LN_in",//233|233
    "Param.QSP.init_value.Parameter.q_T1_LN_out",//234|234
    "Param.QSP.init_value.Parameter.q_nT1_LN_out",//235|235
    "Param.QSP.init_value.Parameter.k_T1_act",//236|236
    "Param.QSP.init_value.Parameter.k_T1_pro",//237|237
    "Param.QSP.init_value.Parameter.k_T1_death",//238|238
    "Param.QSP.init_value.Parameter.q_T1_P_in",//239|239
    "Param.QSP.init_value.Parameter.q_T1_P_out",//240|240
    "Param.QSP.init_value.Parameter.q_T1_T_in",//241|241
    "Param.QSP.init_value.Parameter.q_nT1_P_in",//242|242
    "Param.QSP.init_value.Parameter.q_nT1_P_out",//243|243
    "Param.QSP.init_value.Parameter.Q_nT1_thym",//244|244
    "Param.QSP.init_value.Parameter.k_nT1_pro",//245|245
    "Param.QSP.init_value.Parameter.K_nT1_pro",//246|246
    "Param.QSP.init_value.Parameter.k_nT1_death",//247|247
    "Param.QSP.init_value.Parameter.k_T1",//248|248
    "Param.QSP.init_value.Parameter.k_C_T1",//249|249
    "Param.QSP.init_value.Parameter.k_Tcell_ECM",//250|250
    "Param.QSP.init_value.Parameter.H_ECM_T_mot",//251|251
    "Param.QSP.init_value.Parameter.H_ECM_T_exh",//252|252
    "Param.QSP.init_value.Parameter.k_IFNg_Tsec",//253|253
    "Param.QSP.init_value.Parameter.H_P1",//254|254
    "Param.QSP.init_value.Parameter.K_T_C",//255|255
    "Param.QSP.init_value.Parameter.K_T_Treg",//256|256
    "Param.QSP.init_value.Parameter.k_APC_mat",//257|257
    "Param.QSP.init_value.Parameter.k_APC_mig",//258|258
    "Param.QSP.init_value.Parameter.k_APC_death",//259|259
    "Param.QSP.init_value.Parameter.k_mAPC_death",//260|260
    "Param.QSP.init_value.Parameter.APC0_T",//261|261
    "Param.QSP.init_value.Parameter.APC0_LN",//262|262
    "Param.QSP.init_value.Parameter.n_sites_APC",//263|263
    "Param.QSP.init_value.Parameter.kin",//264|264
    "Param.QSP.init_value.Parameter.kout",//265|265
    "Param.QSP.init_value.Parameter.k_P0_up",//266|266
    "Param.QSP.init_value.Parameter.k_xP0_deg",//267|267
    "Param.QSP.init_value.Parameter.k_P0_deg",//268|268
    "Param.QSP.init_value.Parameter.k_p0_deg",//269|269
    "Param.QSP.init_value.Parameter.k_P0_on",//270|270
    "Param.QSP.init_value.Parameter.k_P0_d1",//271|271
    "Param.QSP.init_value.Parameter.p0_50",//272|272
    "Param.QSP.init_value.Parameter.P0_C1",//273|273
    "Param.QSP.init_value.Parameter.P0_C2",//274|274
    "Param.QSP.init_value.Parameter.A_syn",//275|275
    "Param.QSP.init_value.Parameter.A_Tcell",//276|276
    "Param.QSP.init_value.Parameter.A_cell",//277|277
    "Param.QSP.init_value.Parameter.A_APC",//278|278
    "Param.QSP.init_value.Parameter.k_M1p0_TCR_on",//279|279
    "Param.QSP.init_value.Parameter.k_M1p0_TCR_off",//280|280
    "Param.QSP.init_value.Parameter.k_M1p0_TCR_p",//281|281
    "Param.QSP.init_value.Parameter.phi_M1p0_TCR",//282|282
    "Param.QSP.init_value.Parameter.N_M1p0_TCR",//283|283
    "Param.QSP.init_value.Parameter.TCR_p0_tot",//284|284
    "Param.QSP.init_value.Parameter.pTCR_p0_MHC_tot",//285|285
    "Param.QSP.init_value.Parameter.k_P1_up",//286|286
    "Param.QSP.init_value.Parameter.k_xP1_deg",//287|287
    "Param.QSP.init_value.Parameter.k_P1_deg",//288|288
    "Param.QSP.init_value.Parameter.k_p1_deg",//289|289
    "Param.QSP.init_value.Parameter.k_P1_on",//290|290
    "Param.QSP.init_value.Parameter.k_P1_d1",//291|291
    "Param.QSP.init_value.Parameter.p1_50",//292|292
    "Param.QSP.init_value.Parameter.P1_C1",//293|293
    "Param.QSP.init_value.Parameter.P1_C2",//294|294
    "Param.QSP.init_value.Parameter.k_M1p1_TCR_on",//295|295
    "Param.QSP.init_value.Parameter.k_M1p1_TCR_off",//296|296
    "Param.QSP.init_value.Parameter.k_M1p1_TCR_p",//297|297
    "Param.QSP.init_value.Parameter.phi_M1p1_TCR",//298|298
    "Param.QSP.init_value.Parameter.N_M1p1_TCR",//299|299
    "Param.QSP.init_value.Parameter.TCR_p1_tot",//300|300
    "Param.QSP.init_value.Parameter.pTCR_p1_MHC_tot",//301|301
    "Param.QSP.init_value.Parameter.q_P_aPD1",//302|302
    "Param.QSP.init_value.Parameter.q_T_aPD1",//303|303
    "Param.QSP.init_value.Parameter.q_LN_aPD1",//304|304
    "Param.QSP.init_value.Parameter.q_LD_aPD1",//305|305
    "Param.QSP.init_value.Parameter.k_cl_aPD1",//306|306
    "Param.QSP.init_value.Parameter.gamma_C_aPD1",//307|307
    "Param.QSP.init_value.Parameter.gamma_P_aPD1",//308|308
    "Param.QSP.init_value.Parameter.gamma_T_aPD1",//309|309
    "Param.QSP.init_value.Parameter.gamma_LN_aPD1",//310|310
    "Param.QSP.init_value.Parameter.q_P_aPDL1",//311|311
    "Param.QSP.init_value.Parameter.q_T_aPDL1",//312|312
    "Param.QSP.init_value.Parameter.q_LN_aPDL1",//313|313
    "Param.QSP.init_value.Parameter.q_LD_aPDL1",//314|314
    "Param.QSP.init_value.Parameter.k_cl_aPDL1",//315|315
    "Param.QSP.init_value.Parameter.gamma_C_aPDL1",//316|316
    "Param.QSP.init_value.Parameter.gamma_P_aPDL1",//317|317
    "Param.QSP.init_value.Parameter.gamma_T_aPDL1",//318|318
    "Param.QSP.init_value.Parameter.gamma_LN_aPDL1",//319|319
    "Param.QSP.init_value.Parameter.k_cln_aPDL1",//320|320
    "Param.QSP.init_value.Parameter.Kc_aPDL1",//321|321
    "Param.QSP.init_value.Parameter.q_P_aCTLA4",//322|322
    "Param.QSP.init_value.Parameter.q_T_aCTLA4",//323|323
    "Param.QSP.init_value.Parameter.q_LN_aCTLA4",//324|324
    "Param.QSP.init_value.Parameter.q_LD_aCTLA4",//325|325
    "Param.QSP.init_value.Parameter.k_cl_aCTLA4",//326|326
    "Param.QSP.init_value.Parameter.gamma_C_aCTLA4",//327|327
    "Param.QSP.init_value.Parameter.gamma_P_aCTLA4",//328|328
    "Param.QSP.init_value.Parameter.gamma_T_aCTLA4",//329|329
    "Param.QSP.init_value.Parameter.gamma_LN_aCTLA4",//330|330
    "Param.QSP.init_value.Parameter.kon_PD1_PDL1",//331|331
    "Param.QSP.init_value.Parameter.k_out_PDL1",//332|332
    "Param.QSP.init_value.Parameter.k_in_PDL1",//333|333
    "Param.QSP.init_value.Parameter.r_PDL1_IFNg",//334|334
    "Param.QSP.init_value.Parameter.kon_PD1_PDL2",//335|335
    "Param.QSP.init_value.Parameter.kon_PD1_aPD1",//336|336
    "Param.QSP.init_value.Parameter.kon_PDL1_aPDL1",//337|337
    "Param.QSP.init_value.Parameter.kon_CD28_CD80",//338|338
    "Param.QSP.init_value.Parameter.kon_CD28_CD86",//339|339
    "Param.QSP.init_value.Parameter.kon_CTLA4_CD80",//340|340
    "Param.QSP.init_value.Parameter.kon_CTLA4_CD86",//341|341
    "Param.QSP.init_value.Parameter.kon_CD80_PDL1",//342|342
    "Param.QSP.init_value.Parameter.kon_CTLA4_aCTLA4",//343|343
    "Param.QSP.init_value.Parameter.kon_CD80_CD80",//344|344
    "Param.QSP.init_value.Parameter.koff_PD1_PDL1",//345|345
    "Param.QSP.init_value.Parameter.koff_PD1_PDL2",//346|346
    "Param.QSP.init_value.Parameter.koff_PD1_aPD1",//347|347
    "Param.QSP.init_value.Parameter.koff_PDL1_aPDL1",//348|348
    "Param.QSP.init_value.Parameter.koff_CD28_CD80",//349|349
    "Param.QSP.init_value.Parameter.koff_CD28_CD86",//350|350
    "Param.QSP.init_value.Parameter.koff_CTLA4_CD80",//351|351
    "Param.QSP.init_value.Parameter.koff_CTLA4_CD86",//352|352
    "Param.QSP.init_value.Parameter.koff_CD80_PDL1",//353|353
    "Param.QSP.init_value.Parameter.koff_CTLA4_aCTLA4",//354|354
    "Param.QSP.init_value.Parameter.koff_CD80_CD80",//355|355
    "Param.QSP.init_value.Parameter.Chi_PD1_aPD1",//356|356
    "Param.QSP.init_value.Parameter.Chi_PDL1_aPDL1",//357|357
    "Param.QSP.init_value.Parameter.Chi_CTLA4_aCTLA4",//358|358
    "Param.QSP.init_value.Parameter.PD1_50",//359|359
    "Param.QSP.init_value.Parameter.n_PD1",//360|360
    "Param.QSP.init_value.Parameter.CD28_CD8X_50",//361|361
    "Param.QSP.init_value.Parameter.n_CD28_CD8X",//362|362
    "Param.QSP.init_value.Parameter.T_PD1_total",//363|363
    "Param.QSP.init_value.Parameter.T_CD28_total",//364|364
    "Param.QSP.init_value.Parameter.T_CTLA4_syn",//365|365
    "Param.QSP.init_value.Parameter.T_PDL1_total",//366|366
    "Param.QSP.init_value.Parameter.C1_PDL1_base",//367|367
    "Param.QSP.init_value.Parameter.r_PDL2C1",//368|368
    "Param.QSP.init_value.Parameter.C1_CD80_total",//369|369
    "Param.QSP.init_value.Parameter.C1_CD86_total",//370|370
    "Param.QSP.init_value.Parameter.syn_T_C1_PDL1_total",//371|371
    "Param.QSP.init_value.Parameter.syn_T_C1_PDL2_total",//372|372
    "Param.QSP.init_value.Parameter.APC_PDL1_base",//373|373
    "Param.QSP.init_value.Parameter.r_PDL2APC",//374|374
    "Param.QSP.init_value.Parameter.APC_CD80_total",//375|375
    "Param.QSP.init_value.Parameter.APC_CD86_total",//376|376
    "Param.QSP.init_value.Parameter.syn_T_APC_PDL1_total",//377|377
    "Param.QSP.init_value.Parameter.syn_T_APC_PDL2_total",//378|378
    "Param.QSP.init_value.Parameter.k_Th_act",//379|379
    "Param.QSP.init_value.Parameter.k_Th_Treg",//380|380
    "Param.QSP.init_value.Parameter.k_TGFb_Tsec",//381|381
    "Param.QSP.init_value.Parameter.k_TGFb_deg",//382|382
    "Param.QSP.init_value.Parameter.TGFb_50",//383|383
    "Param.QSP.init_value.Parameter.TGFb_50_Teff",//384|384
    "Param.QSP.init_value.Parameter.Kc_rec",//385|385
    "Param.QSP.init_value.Parameter.TGFbase",//386|386
    "Param.QSP.init_value.Parameter.k_IFNg_Thsec",//387|387
    "Param.QSP.init_value.Parameter.k_IFNg_deg",//388|388
    "Param.QSP.init_value.Parameter.IFNg_50_ind",//389|389
    "Param.QSP.init_value.Parameter.H_TGFb",//390|390
    "Param.QSP.init_value.Parameter.H_TGFb_Teff",//391|391
    "Param.QSP.init_value.Parameter.N_aTh",//392|392
    "Param.QSP.init_value.Parameter.k_CCL2_sec",//393|393
    "Param.QSP.init_value.Parameter.k_CCL2_deg",//394|394
    "Param.QSP.init_value.Parameter.CCL2_50",//395|395
    "Param.QSP.init_value.Parameter.k_MDSC_rec",//396|396
    "Param.QSP.init_value.Parameter.k_MDSC_death",//397|397
    "Param.QSP.init_value.Parameter.k_NO_deg",//398|398
    "Param.QSP.init_value.Parameter.k_ArgI_deg",//399|399
    "Param.QSP.init_value.Parameter.k_NO_sec",//400|400
    "Param.QSP.init_value.Parameter.k_ArgI_sec",//401|401
    "Param.QSP.init_value.Parameter.ArgI_50_Teff",//402|402
    "Param.QSP.init_value.Parameter.NO_50_Teff",//403|403
    "Param.QSP.init_value.Parameter.ArgI_50_Treg",//404|404
    "Param.QSP.init_value.Parameter.H_NO",//405|405
    "Param.QSP.init_value.Parameter.H_ArgI_Teff",//406|406
    "Param.QSP.init_value.Parameter.H_ArgI_Treg",//407|407
    "Param.QSP.init_value.Parameter.H_MDSC",//408|408
    "Param.QSP.init_value.Parameter.k_a1_cabozantinib",//409|409
    "Param.QSP.init_value.Parameter.k_a2_cabozantinib",//410|410
    "Param.QSP.init_value.Parameter.k_cln_cabozantinib",//411|411
    "Param.QSP.init_value.Parameter.Kc_cabozantinib",//412|412
    "Param.QSP.init_value.Parameter.lagP1_cabozantinib",//413|413
    "Param.QSP.init_value.Parameter.lagP2_cabozantinib",//414|414
    "Param.QSP.init_value.Parameter.F_cabozantinib",//415|415
    "Param.QSP.init_value.Parameter.q_P_cabozantinib",//416|416
    "Param.QSP.init_value.Parameter.q_T_cabozantinib",//417|417
    "Param.QSP.init_value.Parameter.q_LN_cabozantinib",//418|418
    "Param.QSP.init_value.Parameter.q_LD_cabozantinib",//419|419
    "Param.QSP.init_value.Parameter.gamma_C_cabozantinib",//420|420
    "Param.QSP.init_value.Parameter.gamma_P_cabozantinib",//421|421
    "Param.QSP.init_value.Parameter.gamma_T_cabozantinib",//422|422
    "Param.QSP.init_value.Parameter.gamma_LN_cabozantinib",//423|423
    "Param.QSP.init_value.Parameter.IC50_MET",//424|424
    "Param.QSP.init_value.Parameter.IC50_RET",//425|425
    "Param.QSP.init_value.Parameter.IC50_AXL",//426|426
    "Param.QSP.init_value.Parameter.IC50_VEGFR2",//427|427
    "Param.QSP.init_value.Parameter.k_C_resist",//428|428
    "Param.QSP.init_value.Parameter.k_K_cabo",//429|429
    "Param.QSP.init_value.Parameter.R_cabo",//430|430
    "Param.QSP.init_value.Parameter.H_therapy_cabo",//431|431
    "Param.QSP.init_value.Parameter.k_Mac_rec",//432|432
    "Param.QSP.init_value.Parameter.k_Mac_death",//433|433
    "Param.QSP.init_value.Parameter.k_TGFb_Msec",//434|434
    "Param.QSP.init_value.Parameter.k_vas_Msec",//435|435
    "Param.QSP.init_value.Parameter.k_IL12_sec",//436|436
    "Param.QSP.init_value.Parameter.k_IL12_Msec",//437|437
    "Param.QSP.init_value.Parameter.k_IL12_deg",//438|438
    "Param.QSP.init_value.Parameter.k_IL10_sec",//439|439
    "Param.QSP.init_value.Parameter.k_IL10_deg",//440|440
    "Param.QSP.init_value.Parameter.k_M2_pol",//441|441
    "Param.QSP.init_value.Parameter.k_M1_pol",//442|442
    "Param.QSP.init_value.Parameter.IL10_50",//443|443
    "Param.QSP.init_value.Parameter.IL12_50",//444|444
    "Param.QSP.init_value.Parameter.IFNg_50",//445|445
    "Param.QSP.init_value.Parameter.k_M1_phago",//446|446
    "Param.QSP.init_value.Parameter.vol_Mcell",//447|447
    "Param.QSP.init_value.Parameter.kon_CD47_SIRPa",//448|448
    "Param.QSP.init_value.Parameter.koff_CD47_SIRPa",//449|449
    "Param.QSP.init_value.Parameter.SIRPa_50",//450|450
    "Param.QSP.init_value.Parameter.n_SIRPa",//451|451
    "Param.QSP.init_value.Parameter.H_Mac_C",//452|452
    "Param.QSP.init_value.Parameter.H_PD1_M",//453|453
    "Param.QSP.init_value.Parameter.H_SIRPa",//454|454
    "Param.QSP.init_value.Parameter.C_CD47",//455|455
    "Param.QSP.init_value.Parameter.M_PD1_total",//456|456
    "Param.QSP.init_value.Parameter.M_SIRPa",//457|457
    "Param.QSP.init_value.Parameter.A_Mcell",//458|458
    "Param.QSP.init_value.Parameter.IL10_50_phago",//459|459
    "Param.QSP.init_value.Parameter.K_Mac_C",//460|460
    "Param.QSP.init_value.Parameter.M_total",//461|461
    "Param.QSP.init_value.Parameter.H_IL10",//462|462
    "Param.QSP.init_value.Parameter.H_IL10_phago",//463|463
    "Param.QSP.init_value.Parameter.H_IL12",//464|464
    "Param.QSP.init_value.Parameter.k_fib_rec",//465|465
    "Param.QSP.init_value.Parameter.k_fib_const",//466|466
    "Param.QSP.init_value.Parameter.k_caf_tran",//467|467
    "Param.QSP.init_value.Parameter.k_ECM_fib_sec",//468|468
    "Param.QSP.init_value.Parameter.k_ECM_CAF_sec",//469|469
    "Param.QSP.init_value.Parameter.k_ECM_deg",//470|470
    "Param.QSP.init_value.Parameter.ECM_base",//471|471
    "Param.QSP.init_value.Parameter.ECM_max",//472|472
    "Param.QSP.init_value.Parameter.ECM_level",//473|473
    "Param.QSP.init_value.Parameter.ECM_MW",//474|474
    "Param.QSP.init_value.Parameter.ECM_density",//475|475
    "Param.QSP.init_value.Parameter.k_fib_death",//476|476
    "Param.QSP.init_value.Parameter.k_CAF_death",//477|477
    "Param.QSP.init_value.Parameter.ECM_50_T_exh",//478|478
    "Param.QSP.init_value.Parameter.ECM_50_T_mot",//479|479
    "Param.QSP.init_value.Parameter.vol_Fibcell",//480|480
    "Param.QSP.init_value.Parameter.vol_CAFcell",//481|481

};

// Compile-time check: ensure array size matches enum count
static_assert(sizeof(CancerVCT::QSP_PARAM_FLOAT_XML_PATHS) / sizeof(CancerVCT::QSP_PARAM_FLOAT_XML_PATHS[0])
              == CancerVCT::QSP_PARAM_FLOAT_COUNT,
              "QSP_PARAM_FLOAT_XML_PATHS array size must match QSP_PARAM_FLOAT_COUNT");

namespace CancerVCT {

QSPParam::QSPParam() : ParamBase() {
    setupParam();
}

void QSPParam::setupParam() {
    // Initialize parameter vectors to match enum count
    _paramFloat.resize(QSP_PARAM_FLOAT_COUNT, 0.0);
    _paramInt.resize(QSP_PARAM_INT_COUNT, 0);

    // Create parameter descriptions based on enum order
    // Each entry in QSP_PARAM_FLOAT_XML_PATHS corresponds to an enum value
    // The POSITION in the array = the ENUM VALUE = the VECTOR INDEX

    for (int i = 0; i < QSP_PARAM_FLOAT_COUNT; ++i) {
        // For each enum position, add description
        _paramDesc.push_back({
            QSP_PARAM_FLOAT_XML_PATHS[i],  // XML path
            "",                             // units (can be extended)
            "pos"                           // positive number validation
        });
    }

    std::cout << "QSPParam setup: " << QSP_PARAM_FLOAT_COUNT << " parameters configured" << std::endl;
}

void QSPParam::processInternalParams() {
    // Any internal parameter processing after XML loading
    // (validation, unit conversions, etc.)
}

// ============================================================================
// OVERRIDE: XML PARSING - ENUM-DRIVEN APPROACH
// ============================================================================
// The key difference: Instead of reading parameters in XML order and storing
// at sequential indices, we read by XML path and store at enum-defined indices.

bool QSPParam::readParamsFromXml(std::string inFileName) {
    std::cout << "Loading QSP parameters from: " << inFileName << std::endl;

    try {
        // Load XML file
        boost::property_tree::ptree pt;
        boost::property_tree::read_xml(inFileName, pt);

        // ====================================================================
        // KEY STEP: Read parameters by XML PATH, store at ENUM INDEX
        // ====================================================================
        // For each enum value (0, 1, 2, ...), we know:
        // 1. The XML path from QSP_PARAM_FLOAT_XML_PATHS[enum_value]
        // 2. The vector index is EXACTLY the enum_value
        // 3. No need for any external index mapping!

        for (int enum_idx = 0; enum_idx < QSP_PARAM_FLOAT_COUNT; ++enum_idx) {
            std::string xml_path = QSP_PARAM_FLOAT_XML_PATHS[enum_idx];

            try {
                // Read value from XML using the path
                double value = pt.get<double>(xml_path);

                // Store at vector index = enum
                _paramFloat[enum_idx] = value;

                // std::cout << "✓ Parameter[" << enum_idx << "] = " << xml_path
                //           << " : " << value << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "✗ Failed to load parameter at index " << enum_idx
                          << " (" << xml_path << "): " << e.what() << std::endl;
                throw;
            }
        }

        std::cout << "✓ All " << QSP_PARAM_FLOAT_COUNT << " QSP parameters loaded successfully"
                  << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "ERROR reading QSP parameters from XML: " << e.what() << std::endl;
        return false;
    }
}

} // namespace CancerVCT
