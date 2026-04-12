"""Functional Cell Neighborhood (FunCN) score — 2D per-z-slice version.

Matches the reference R implementation (spatstat::Smooth, Nadaraya-Watson
kernel smoother).  For each z-slice the interaction weight of source type S
on target type T is:

    w_S(i) = sum_{j != i, j in S} K(d_ij) / sum_{j != i} K(d_ij)

where K is a 2D Gaussian kernel with bandwidth sigma (default 10 um) and d_ij
is the 2D Euclidean distance.  The per-slice interaction matrix entry (T, S) is
the mean of w_S(i) over all cells i of type T.  The final summary averages
these matrices across all z-slices.

Coordinates are converted to microns as (voxel_index + 1) * voxel_size, with
optional small jitter (sd=2 um) to match the reference preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .abm_reader import (
    COL_STATE,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    T_CANCER,
    T_FIB,
    T_MAC,
    T_MDSC,
    T_TCELL,
    T_TREG,
    T_VAS,
    TREG_TH,
    TREG_TREG,
)

DEFAULT_SIGMA_UM = 20.0
DEFAULT_VOXEL_UM = 20.0

# Cell type labels used in the interaction matrix, matching the reference R code.
CELL_TYPES = (
    "Tumor",
    "CD8T",
    "CD4T",
    "Tregs",
    "MDSCs",
    "Macrophages",
    "Fibroblasts",
    "Endothelials",
)

# The 5 (reference, weight) pairs used for calibration (from the paper).
# Name convention: "Reference_{target}_Weight_{source}"
PAIR_NAMES = (
    "Reference_CD4T_Weight_Tumor",
    "Reference_CD8T_Weight_Tumor",
    "Reference_Fibroblasts_Weight_Fibroblasts",
    "Reference_Tregs_Weight_Tregs",
    "Reference_CD8T_Weight_Macrophages",
)


def _classify_cells(agents: np.ndarray) -> np.ndarray:
    """Return a string-label array (length N) classifying each agent row."""
    n = agents.shape[0]
    labels = np.empty(n, dtype=object)
    labels[:] = "Other"
    for i in range(n):
        t = agents[i, COL_TYPE]
        s = agents[i, COL_STATE]
        if t == T_CANCER:
            labels[i] = "Tumor"
        elif t == T_TCELL:
            labels[i] = "CD8T"
        elif t == T_TREG:
            if s == TREG_TH:
                labels[i] = "CD4T"
            elif s == TREG_TREG:
                labels[i] = "Tregs"
        elif t == T_MDSC:
            labels[i] = "MDSCs"
        elif t == T_MAC:
            labels[i] = "Macrophages"
        elif t == T_FIB:
            labels[i] = "Fibroblasts"
        elif t == T_VAS:
            labels[i] = "Endothelials"
    return labels


def _interaction_matrix_for_slice(
    xy: np.ndarray,
    labels: np.ndarray,
    sigma_um: float,
) -> np.ndarray | None:
    """Compute the 8x8 Nadaraya-Watson interaction matrix for one z-slice.

    Returns an (8, 8) float64 array where entry [t, s] is the mean kernel-
    weighted proportion of type s around cells of type t, or None if the slice
    has no usable cells.
    """
    n = xy.shape[0]
    if n < 2:
        return None

    # Pairwise squared distances (2D)
    diff = xy[:, None, :] - xy[None, :, :]          # (N, N, 2)
    d2 = np.einsum("ijk,ijk->ij", diff, diff)        # (N, N)

    # Gaussian kernel weights, zero on the diagonal (leave-one-out)
    two_sigma_sq = 2.0 * sigma_um * sigma_um
    K = np.exp(-d2 / two_sigma_sq)                    # (N, N)
    np.fill_diagonal(K, 0.0)

    # Build one-hot mark matrix (N, 8) for the cell types we care about
    type_to_idx = {name: i for i, name in enumerate(CELL_TYPES)}
    marks = np.zeros((n, len(CELL_TYPES)), dtype=np.float64)
    for i in range(n):
        idx = type_to_idx.get(labels[i])
        if idx is not None:
            marks[i, idx] = 1.0

    # Nadaraya-Watson: smoothed(i, s) = sum_j K(i,j)*marks(j,s) / sum_j K(i,j)
    denom = K.sum(axis=1, keepdims=True)              # (N, 1)
    denom = np.maximum(denom, 1e-300)                  # avoid div-by-zero
    smoothed = (K @ marks) / denom                     # (N, 8)

    # Build interaction matrix: for each target type t, mean of smoothed scores
    mat = np.zeros((len(CELL_TYPES), len(CELL_TYPES)), dtype=np.float64)
    for t_idx, t_name in enumerate(CELL_TYPES):
        mask = labels == t_name
        if mask.sum() == 0:
            continue
        mat[t_idx, :] = smoothed[mask].mean(axis=0)

    return mat


@dataclass
class FunCNResult:
    interaction_matrix: np.ndarray   # (8, 8) averaged across z-slices
    summary: np.ndarray              # (7,) — one mean value per calibration pair
    per_slice: list[np.ndarray | None]  # per-z interaction matrices (for diagnostics)


def compute_funcn(
    agents: np.ndarray,
    sigma_um: float = DEFAULT_SIGMA_UM,
    voxel_um: float = DEFAULT_VOXEL_UM,
    jitter_sd: float = 2.0,
    rng_seed: int | None = 42,
) -> FunCNResult:
    """Compute 2D per-z-slice FunCN interaction matrix and calibration summary.

    Matches the reference R implementation:
      1. For each z-slice, convert voxel coords to microns with jitter.
      2. Compute Nadaraya-Watson smoothed interaction matrix (8x8).
      3. Average across z-slices.
      4. Extract the 7 calibration pairs.
    """
    rng = np.random.default_rng(rng_seed)
    labels = _classify_cells(agents)

    # Identify unique z-slices
    z_vals = np.unique(agents[:, COL_Z])

    slice_matrices: list[np.ndarray | None] = []
    valid_matrices: list[np.ndarray] = []

    for z in z_vals:
        zmask = agents[:, COL_Z] == z
        slice_agents = agents[zmask]
        slice_labels = labels[zmask]

        # Skip slices with only one cell type (matches reference: "all Tumor" skip)
        unique_types = set(slice_labels)
        unique_types.discard("Other")
        if len(unique_types) <= 1:
            slice_matrices.append(None)
            continue

        # Convert to microns: (index + 1) * voxel_size + jitter
        xy = np.column_stack([
            (slice_agents[:, COL_X].astype(np.float64) + 1) * voxel_um
            + rng.normal(0, jitter_sd, size=slice_agents.shape[0]),
            (slice_agents[:, COL_Y].astype(np.float64) + 1) * voxel_um
            + rng.normal(0, jitter_sd, size=slice_agents.shape[0]),
        ])

        mat = _interaction_matrix_for_slice(xy, slice_labels, sigma_um)
        slice_matrices.append(mat)
        if mat is not None:
            valid_matrices.append(mat)

    # Average across z-slices
    if valid_matrices:
        avg_matrix = np.mean(valid_matrices, axis=0)
    else:
        avg_matrix = np.zeros((len(CELL_TYPES), len(CELL_TYPES)), dtype=np.float64)

    # Extract the 7 calibration summary values
    type_to_idx = {name: i for i, name in enumerate(CELL_TYPES)}
    summary = np.zeros(len(PAIR_NAMES), dtype=np.float64)
    for k, pair_name in enumerate(PAIR_NAMES):
        # Parse "Reference_{target}_Weight_{source}"
        parts = pair_name.split("_")
        # Reference_X_Weight_Y  or  Reference_X_Weight_Y (multi-word types handled)
        ref_idx = 0   # "Reference"
        weight_idx = parts.index("Weight")
        target = "_".join(parts[1:weight_idx])
        source = "_".join(parts[weight_idx + 1:])
        t_idx = type_to_idx[target]
        s_idx = type_to_idx[source]
        summary[k] = avg_matrix[t_idx, s_idx]

    return FunCNResult(
        interaction_matrix=avg_matrix,
        summary=summary,
        per_slice=slice_matrices,
    )


def summary_labels() -> list[str]:
    return list(PAIR_NAMES)
