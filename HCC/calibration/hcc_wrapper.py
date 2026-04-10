"""Python wrapper around the HCC GPU binary.

One call = one simulation run in an isolated working directory. The main loop
is skipped (`-s 0`); only the presim phase runs, and we return the final presim
ABM snapshot as an (N, 8) int32 numpy array.

Binary path resolution: $HCC_BINARY, else <repo>/HCC/sim/build/bin/hcc.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from . import abm_reader
from .xml_override import apply_overrides

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "HCC" / "sim" / "build" / "bin" / "hcc"
DEFAULT_BASE_XML = REPO_ROOT / "HCC" / "sim" / "resource" / "param_all_test.xml"


def get_binary() -> Path:
    env = os.environ.get("HCC_BINARY")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"HCC_BINARY={env} does not exist")
        return p
    if DEFAULT_BINARY.exists():
        return DEFAULT_BINARY
    raise FileNotFoundError(
        f"HCC binary not found. Set HCC_BINARY or build {DEFAULT_BINARY}"
    )


@dataclass
class RunResult:
    agents: np.ndarray          # (N, 8) int32
    workdir: Path               # containing outputs/
    xml_path: Path              # modified XML used for this run
    stdout: str
    stderr: str


def run_hcc(
    overrides: Mapping[str, float | int | str] | None = None,
    *,
    base_xml: str | Path = DEFAULT_BASE_XML,
    seed: int = 12345,
    grid: int | None = None,
    workdir: str | Path | None = None,
    keep_workdir: bool = False,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Run one presim-only HCC simulation and return its final ABM snapshot.

    Parameters
    ----------
    overrides : dict of dotted XML paths -> value, e.g. {"ABM.TCell.moveSteps": 42}
    base_xml  : template XML to modify
    seed      : simulation seed (picked to disambiguate output files)
    grid      : optional grid override (maps to -g)
    workdir   : if None, a tempdir is created and cleaned up unless keep_workdir=True
    extra_args: additional CLI args passed verbatim to the binary
    """
    binary = get_binary()
    base_xml = Path(base_xml)
    if not base_xml.exists():
        raise FileNotFoundError(f"base_xml not found: {base_xml}")

    cleanup = False
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="hcc_calib_"))
        cleanup = not keep_workdir
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        xml_out = workdir / "params.xml"
        apply_overrides(base_xml, overrides or {}, xml_out)

        cmd: list[str | Path] = [
            binary,
            "-p", xml_out,
            "-s", "0",
            "--seed", str(seed),
        ]
        if grid is not None:
            cmd += ["-g", str(grid)]
        if extra_args:
            cmd += list(extra_args)

        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"hcc exited {proc.returncode}\n--- stderr ---\n{proc.stderr[-2000:]}"
            )

        final = abm_reader.find_final_presim_abm(workdir / "outputs")
        agents = abm_reader.read_abm_lz4(final)

        return RunResult(
            agents=agents,
            workdir=workdir,
            xml_path=xml_out,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
