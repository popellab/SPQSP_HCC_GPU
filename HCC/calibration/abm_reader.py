"""Reader for .abm.lz4 snapshots produced by the HCC GPU model.

Format (see HCC/sim/main.cu:441):
    Bytes 0-3   magic "ABM1"
    Bytes 4-7   n_agents (int32)
    Bytes 8-11  n_cols (int32, == 8)
    Bytes 12-15 raw_bytes (int32)
    Bytes 16-19 comp_bytes (int32)
    Bytes 20+   LZ4-compressed int32 payload, shape (n_agents, n_cols)

Columns: [type_id, agent_id, x, y, z, cell_state, life, extra]
Type IDs: 0 CANCER, 1 TCELL (CD8), 2 TREG, 3 MDSC, 4 MAC, 5 FIB, 6 VAS
"""
from __future__ import annotations

import struct
from pathlib import Path

import lz4.block
import numpy as np

# Column indices
COL_TYPE = 0
COL_ID = 1
COL_X = 2
COL_Y = 3
COL_Z = 4
COL_STATE = 5
COL_LIFE = 6
COL_EXTRA = 7

# Type IDs
T_CANCER = 0
T_TCELL = 1
T_TREG = 2
T_MDSC = 3
T_MAC = 4
T_FIB = 5
T_VAS = 6

# TREG sub-states
TREG_TH = 0
TREG_TREG = 1


def read_abm_lz4(path: str | Path) -> np.ndarray:
    """Return an (N, 8) int32 array of agents from a .abm.lz4 file."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ABM1":
            raise ValueError(f"{path}: bad magic {magic!r}")
        n_agents, n_cols, raw_sz, _comp_sz = struct.unpack("<4i", f.read(16))
        payload = f.read()

    if n_agents == 0:
        return np.zeros((0, n_cols), dtype=np.int32)

    raw = lz4.block.decompress(payload, uncompressed_size=raw_sz)
    arr = np.frombuffer(raw, dtype=np.int32).reshape(n_agents, n_cols)
    return arr


def find_final_presim_abm(outputs_dir: str | Path) -> Path:
    """Return the path to the highest-index presim ABM snapshot."""
    outputs_dir = Path(outputs_dir)
    presim_abm = outputs_dir / "presim" / "abm"
    files = sorted(presim_abm.glob("agents_presim_*.abm.lz4"))
    if not files:
        raise FileNotFoundError(f"No presim ABM snapshots under {presim_abm}")
    return files[-1]


def select(agents: np.ndarray, type_id: int, state: int | None = None) -> np.ndarray:
    """Return (M, 3) float coordinates for all agents of the given type/state."""
    mask = agents[:, COL_TYPE] == type_id
    if state is not None:
        mask &= agents[:, COL_STATE] == state
    return agents[mask][:, [COL_X, COL_Y, COL_Z]].astype(np.float32)
