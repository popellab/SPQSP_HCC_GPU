"""End-to-end calibration driver: ground truth + PyABC identifiability check.

Usage:
    # 1. Generate ground truth summary from the base XML (one simulation)
    python -m HCC.calibration.run_pyabc ground-truth \
        --base-xml HCC/sim/resource/param_all_test.xml \
        --out HCC/calibration/ground_truth.json \
        --seed 12345

    # 2. Run PyABC identifiability over the 3 moveSteps parameters
    python -m HCC.calibration.run_pyabc abc \
        --target HCC/calibration/ground_truth.json \
        --db sqlite:///HCC/calibration/abc.db \
        --population-size 50 --max-populations 4

Calibration params (priors are uniform, centered on the base XML values):
    ABM.TCell.moveSteps  (default 53)
    ABM.Fib.moveSteps    (default 16)
    ABM.Mac.moveSteps    (default  6)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .funcn import compute_funcn, summary_labels
from .hcc_wrapper import DEFAULT_BASE_XML, run_hcc

# Parameter names and their priors, keyed by XML dot-path.
PARAM_PATHS = {
    "TCell_moveSteps": "ABM.TCell.moveSteps",
    "Fib_moveSteps":   "ABM.Fib.moveSteps",
    "Mac_moveSteps":   "ABM.Mac.moveSteps",
}

# (lower, upper) bounds for the uniform prior on each parameter.
PARAM_BOUNDS = {
    "TCell_moveSteps": (10, 120),
    "Fib_moveSteps":   (4, 40),
    "Mac_moveSteps":   (1, 20),
}


def _params_to_overrides(params: dict[str, Any]) -> dict[str, int]:
    """PyABC parameters (floats) -> integer XML overrides keyed by dot-path."""
    out: dict[str, int] = {}
    for key, path in PARAM_PATHS.items():
        if key not in params:
            continue
        out[path] = int(round(float(params[key])))
    return out


def simulate_summary(
    params: dict[str, Any],
    base_xml: Path,
    seed: int,
    grid: int | None,
    keep_workdir: bool,
) -> np.ndarray:
    """Run HCC with these params and return the 5-value FunCN summary."""
    overrides = _params_to_overrides(params)
    result = run_hcc(
        overrides=overrides,
        base_xml=base_xml,
        seed=seed,
        grid=grid,
        keep_workdir=keep_workdir,
    )
    return compute_funcn(result.agents).summary


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth mode
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ground_truth(args: argparse.Namespace) -> int:
    base_xml = Path(args.base_xml)
    t0 = time.time()
    summary = simulate_summary(
        params={},  # no overrides — use base XML verbatim
        base_xml=base_xml,
        seed=args.seed,
        grid=args.grid,
        keep_workdir=args.keep_workdir,
    )
    dt = time.time() - t0

    labels = summary_labels()
    payload = {
        "base_xml": str(base_xml.resolve()),
        "seed": args.seed,
        "grid": args.grid,
        "labels": labels,
        "summary": summary.tolist(),
        "wall_time_s": dt,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Ground-truth written to {out} ({dt:.1f}s)")
    for name, val in zip(labels, summary):
        print(f"  {name:>24s} = {val:.6g}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# PyABC mode
# ─────────────────────────────────────────────────────────────────────────────

def cmd_abc(args: argparse.Namespace) -> int:
    try:
        import pyabc
    except ImportError:
        print("pyabc is not installed. pip install pyabc", file=sys.stderr)
        return 2

    target_payload = json.loads(Path(args.target).read_text())
    target_vec = np.asarray(target_payload["summary"], dtype=np.float64)
    base_xml = Path(target_payload.get("base_xml", str(DEFAULT_BASE_XML)))
    grid = target_payload.get("grid")

    # Per-seed differently from the ground truth run so stochastic runs differ.
    base_seed = args.seed_base

    def model(params: dict[str, Any]) -> dict[str, Any]:
        # Unique-ish seed per call so duplicate param samples don't collide.
        seed = (base_seed + int(abs(hash(tuple(sorted(params.items())))) % 1_000_000)) & 0x7FFFFFFF
        try:
            summ = simulate_summary(
                params=params,
                base_xml=base_xml,
                seed=seed,
                grid=grid,
                keep_workdir=False,
            )
        except Exception:
            traceback.print_exc()
            summ = np.full_like(target_vec, np.nan)
        return {"summary": summ}

    def distance(x: dict[str, Any], x0: dict[str, Any]) -> float:
        a = np.asarray(x["summary"], dtype=np.float64)
        b = np.asarray(x0["summary"], dtype=np.float64)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return float("inf")
        return float(np.mean((a - b) ** 2))

    prior_dict = {
        key: pyabc.RV("uniform", lo, hi - lo)
        for key, (lo, hi) in PARAM_BOUNDS.items()
    }
    prior = pyabc.Distribution(**prior_dict)

    # Force single-process sampling: the HCC binary owns the GPU, so PyABC's
    # default multicore sampler would launch concurrent hcc processes all
    # fighting for the same device and OOM/crash.
    sampler = pyabc.sampler.SingleCoreSampler()

    abc = pyabc.ABCSMC(
        models=model,
        parameter_priors=prior,
        distance_function=distance,
        population_size=args.population_size,
        sampler=sampler,
    )
    abc.new(args.db, {"summary": target_vec})
    history = abc.run(
        minimum_epsilon=args.min_epsilon,
        max_nr_populations=args.max_populations,
    )
    print(f"ABC finished. History id={history.id}, n_populations={history.max_t + 1}")
    print(f"Database: {args.db}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("ground-truth", help="Run HCC once and save FunCN target vector")
    g.add_argument("--base-xml", default=str(DEFAULT_BASE_XML))
    g.add_argument("--out", default="HCC/calibration/ground_truth.json")
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--grid", type=int, default=None)
    g.add_argument("--keep-workdir", action="store_true")
    g.set_defaults(func=cmd_ground_truth)

    a = sub.add_parser("abc", help="Run PyABC against a saved target")
    a.add_argument("--target", default="HCC/calibration/ground_truth.json")
    a.add_argument("--db", default="sqlite:///HCC/calibration/abc.db")
    a.add_argument("--population-size", type=int, default=100)
    a.add_argument("--max-populations", type=int, default=5)
    a.add_argument("--min-epsilon", type=float, default=0.0)
    a.add_argument("--seed-base", type=int, default=20260410)
    a.set_defaults(func=cmd_abc)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
