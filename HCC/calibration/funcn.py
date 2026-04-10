"""Functional Cell Neighborhood (FunCN) score and 5-pair summary.

For each target cell at position (xT, yT, zT), the FunCN score w.r.t. a
source cell population S is a Gaussian-kernel density:

    f(T) = sum_{s in S} (1 / (2*pi*sigma^2)) * exp(-||T - s||^2 / (2 sigma^2))

Distances are computed in microns (voxel indices scaled by voxel_size). Sigma
defaults to 10 um per the FunCN definition. For "X to X" pairs (fib/fib,
treg/treg), the target cell itself is excluded from the source set.

The summary vector used for calibration is (p25, p50, p75) of the per-target
score distribution for each of five pairs, concatenated -> 15 values.

Pair naming convention: pair "(A, B)" = "influence of source B on target A".
The five pairs below follow the user's spec ("X to Y" = source X, target Y).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .abm_reader import (
    COL_STATE,
    COL_TYPE,
    T_CANCER,
    T_FIB,
    T_MAC,
    T_TCELL,
    T_TREG,
    TREG_TH,
    TREG_TREG,
    select,
)

DEFAULT_SIGMA_UM = 10.0
DEFAULT_VOXEL_UM = 20.0

# Order is fixed — must match between ground-truth and ABC runs.
PAIR_NAMES = (
    "CD4Th_to_tumor",     # source=CD4Th, target=tumor
    "CD8_to_tumor",       # source=CD8, target=tumor
    "fib_to_fib",         # source=fib,  target=fib  (self excluded)
    "treg_to_treg",       # source=treg, target=treg (self excluded)
    "CD8_to_mac",         # source=CD8,  target=mac
)

QUANTILES = (0.25, 0.50, 0.75)


def funcn_scores(
    target_xyz: np.ndarray,
    source_xyz: np.ndarray,
    sigma_um: float = DEFAULT_SIGMA_UM,
    exclude_self: bool = False,
) -> np.ndarray:
    """Return one FunCN score per target cell.

    Inputs are (N, 3) float arrays in microns. If exclude_self=True the diagonal
    (each target paired with itself when target is also in source) is removed.
    """
    if target_xyz.shape[0] == 0 or source_xyz.shape[0] == 0:
        return np.zeros((target_xyz.shape[0],), dtype=np.float64)

    two_sigma_sq = 2.0 * sigma_um * sigma_um
    norm = 1.0 / (2.0 * np.pi * sigma_um * sigma_um)

    diff = target_xyz[:, None, :] - source_xyz[None, :, :]  # (T, S, 3)
    d2 = np.einsum("tsi,tsi->ts", diff, diff)                # (T, S)
    w = norm * np.exp(-d2 / two_sigma_sq)                    # (T, S)

    if exclude_self:
        # Remove the self-pair by subtracting the self-weight (which is `norm`).
        # This works whenever target and source sets are identical.
        if target_xyz.shape == source_xyz.shape and np.array_equal(target_xyz, source_xyz):
            w_sum = w.sum(axis=1) - norm
        else:
            w_sum = w.sum(axis=1)
    else:
        w_sum = w.sum(axis=1)

    return w_sum


@dataclass
class FunCNResult:
    per_pair: dict[str, np.ndarray]   # pair_name -> per-target score array
    summary: np.ndarray               # shape (15,) float64, see QUANTILES order


def _xyz_um(agents: np.ndarray, type_id: int, state: int | None, voxel_um: float) -> np.ndarray:
    xyz = select(agents, type_id, state)
    return xyz * voxel_um if xyz.size else xyz.reshape(0, 3)


def compute_funcn(
    agents: np.ndarray,
    sigma_um: float = DEFAULT_SIGMA_UM,
    voxel_um: float = DEFAULT_VOXEL_UM,
) -> FunCNResult:
    """Compute all 5 FunCN score arrays and the (15,) summary vector."""
    tumor = _xyz_um(agents, T_CANCER, None, voxel_um)
    cd8   = _xyz_um(agents, T_TCELL,  None, voxel_um)
    cd4th = _xyz_um(agents, T_TREG,   TREG_TH,  voxel_um)
    treg  = _xyz_um(agents, T_TREG,   TREG_TREG, voxel_um)
    fib   = _xyz_um(agents, T_FIB,    None, voxel_um)
    mac   = _xyz_um(agents, T_MAC,    None, voxel_um)

    per_pair: dict[str, np.ndarray] = {
        "CD4Th_to_tumor": funcn_scores(tumor, cd4th, sigma_um),
        "CD8_to_tumor":   funcn_scores(tumor, cd8,   sigma_um),
        "fib_to_fib":     funcn_scores(fib,   fib,   sigma_um, exclude_self=True),
        "treg_to_treg":   funcn_scores(treg,  treg,  sigma_um, exclude_self=True),
        "CD8_to_mac":     funcn_scores(mac,   cd8,   sigma_um),
    }

    summary = summarize(per_pair)
    return FunCNResult(per_pair=per_pair, summary=summary)


def summarize(per_pair: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concat (p25, p50, p75) for each pair in PAIR_NAMES order -> (15,).

    Empty target populations yield zeros (so the sim still returns a vector).
    """
    out = np.zeros(len(PAIR_NAMES) * len(QUANTILES), dtype=np.float64)
    for i, name in enumerate(PAIR_NAMES):
        scores = per_pair[name]
        base = i * len(QUANTILES)
        if scores.size == 0:
            continue
        out[base : base + len(QUANTILES)] = np.quantile(scores, QUANTILES)
    return out


def summary_labels() -> list[str]:
    labels = []
    for name in PAIR_NAMES:
        for q in QUANTILES:
            labels.append(f"{name}_p{int(q*100):02d}")
    return labels
