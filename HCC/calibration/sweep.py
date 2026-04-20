"""Parameter sweep: moveSteps (TCell, Fib, Mac) → 5 FunCN scores.

Design: 5×5×5 full-factorial grid = 125 combinations, 5 random seeds each.
Total: 625 presim simulations.  Each SLURM array task runs one combo (5 seeds
sequentially on the same GPU) and writes a per-task CSV to RESULTS_DIR.

Commands
--------
    # Print the full design (125 rows) — no simulation:
    python -m HCC.calibration.sweep grid

    # Run one combo by 0-indexed task ID (called from submit_sweep.sh):
    python -m HCC.calibration.sweep run --task 0 --results-dir /path/to/results

    # Merge all per-task CSVs into sweep_results.csv:
    python -m HCC.calibration.sweep merge --results-dir /path/to/results
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .funcn import compute_funcn, summary_labels, PAIR_NAMES
from .hcc_wrapper import DEFAULT_BASE_XML, run_hcc
from .run_pyabc import PARAM_PATHS

# ── Sweep design ──────────────────────────────────────────────────────────────

# 5 evenly-spaced integer values within each parameter's prior bounds.
TCELL_VALUES: list[int] = [10, 38, 65, 93, 120]   # ABM.TCell.moveSteps in [10, 120]
FIB_VALUES:   list[int] = [4,  13, 22, 31, 40]    # ABM.Fib.moveSteps   in [4,  40]
MAC_VALUES:   list[int] = [1,  6,  11, 15, 20]    # ABM.Mac.moveSteps   in [1,  20]

# 5 fixed seeds per combination (chosen to not overlap with ground-truth seed 12345).
SEEDS: list[int] = [1001, 2002, 3003, 4004, 5005]

CSV_COLUMNS = (
    ["task_id", "tcell_move_steps", "fib_move_steps", "mac_move_steps", "seed"]
    + list(PAIR_NAMES)
    + ["wall_time_s", "error"]
)


def build_grid() -> list[tuple[int, int, int]]:
    """Return all 125 (tcell, fib, mac) combinations in stable order."""
    return list(itertools.product(TCELL_VALUES, FIB_VALUES, MAC_VALUES))


def task_combo(task_id: int) -> tuple[int, int, int]:
    """Return the (tcell, fib, mac) combo for a given 0-indexed task ID."""
    grid = build_grid()
    if not 0 <= task_id < len(grid):
        raise ValueError(f"task_id {task_id} out of range [0, {len(grid) - 1}]")
    return grid[task_id]


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_grid(args: argparse.Namespace) -> int:
    """Print the full 125-row design to stdout."""
    grid = build_grid()
    print(f"{'task_id':>8}  {'tcell':>6}  {'fib':>5}  {'mac':>4}")
    print("-" * 32)
    for i, (t, f, m) in enumerate(grid):
        print(f"{i:>8}  {t:>6}  {f:>5}  {m:>4}")
    print(f"\nTotal combinations: {len(grid)}  ×  {len(SEEDS)} seeds = {len(grid) * len(SEEDS)} runs")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run all seeds for one combo and write a per-task CSV."""
    task_id = args.task
    base_xml = Path(args.base_xml)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tcell, fib, mac = task_combo(task_id)
    overrides = {
        PARAM_PATHS["TCell_moveSteps"]: tcell,
        PARAM_PATHS["Fib_moveSteps"]:   fib,
        PARAM_PATHS["Mac_moveSteps"]:   mac,
    }

    out_csv = results_dir / f"sweep_task_{task_id:04d}.csv"

    print(f"Task {task_id}: TCell={tcell}, Fib={fib}, Mac={mac}, seeds={SEEDS}")
    print(f"Output: {out_csv}")

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        t0 = time.time()
        error_msg = ""
        summary = [float("nan")] * len(PAIR_NAMES)
        try:
            result = run_hcc(
                overrides=overrides,
                base_xml=base_xml,
                seed=seed,
                grid=args.grid if args.grid else None,
                keep_workdir=False,
            )
            funcn = compute_funcn(result.agents)
            summary = funcn.summary.tolist()
        except Exception:
            error_msg = traceback.format_exc(limit=3).strip().splitlines()[-1]
            print(f"  seed={seed} FAILED: {error_msg}", file=sys.stderr)

        wall = time.time() - t0
        row: dict[str, Any] = {
            "task_id":          task_id,
            "tcell_move_steps": tcell,
            "fib_move_steps":   fib,
            "mac_move_steps":   mac,
            "seed":             seed,
            "wall_time_s":      round(wall, 2),
            "error":            error_msg,
        }
        for name, val in zip(PAIR_NAMES, summary):
            row[name] = val

        rows.append(row)
        status = "OK" if not error_msg else "FAIL"
        vals = "  ".join(f"{v:.4f}" for v in summary) if not error_msg else "n/a"
        print(f"  seed={seed}  [{status}]  {vals}  ({wall:.1f}s)")

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if not r["error"])
    print(f"\nTask {task_id} done: {n_ok}/{len(SEEDS)} seeds succeeded → {out_csv}")
    return 0 if n_ok > 0 else 1


def cmd_merge(args: argparse.Namespace) -> int:
    """Combine all per-task CSVs into a single sweep_results.csv."""
    results_dir = Path(args.results_dir)
    per_task = sorted(results_dir.glob("sweep_task_*.csv"))
    if not per_task:
        print(f"No sweep_task_*.csv files found in {results_dir}", file=sys.stderr)
        return 1

    out_csv = results_dir / "sweep_results.csv"
    total_rows = 0
    with out_csv.open("w", newline="") as fh:
        writer: csv.DictWriter | None = None
        for path in per_task:
            with path.open(newline="") as src:
                reader = csv.DictReader(src)
                for row in reader:
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    total_rows += 1

    n_tasks = len(per_task)
    print(f"Merged {n_tasks} task files ({total_rows} rows) → {out_csv}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("grid", help="Print the 125-row design grid and exit")

    r = sub.add_parser("run", help="Run one task (all 5 seeds for one combo)")
    r.add_argument("--task", type=int, required=True,
                   help="0-indexed task ID in [0, 124]")
    r.add_argument("--results-dir", default="sweep_results",
                   help="Directory to write per-task CSV (default: sweep_results/)")
    r.add_argument("--base-xml", default=str(DEFAULT_BASE_XML))
    r.add_argument("--grid", type=int, default=None,
                   help="Override grid size (-g flag passed to hcc binary)")

    m = sub.add_parser("merge", help="Merge all per-task CSVs into sweep_results.csv")
    m.add_argument("--results-dir", default="sweep_results")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dispatch = {"grid": cmd_grid, "run": cmd_run, "merge": cmd_merge}
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
