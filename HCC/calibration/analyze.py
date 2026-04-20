"""Plot diagnostics and posteriors for a finished PyABC calibration run.

Reads a PyABC SQLite history and the ground_truth.json produced by
`run_pyabc ground-truth`, then writes PNGs into an output directory.

Usage:
    python -m HCC.calibration.analyze \
        --db sqlite:///HCC/calibration/abc.db \
        --target HCC/calibration/ground_truth.json \
        --out HCC/calibration/figures

Plots produced:
    epsilon.png              — acceptance threshold ε per population
    ess.png                  — effective sample size per population
    samples.png              — model evaluations per population (and cumulative)
    acceptance_rate.png      — accepted particles / proposed samples
    posterior_marginals.png  — 1D posterior histogram per movement parameter, with
                               the base-XML (ground-truth) value marked
    posterior_corner.png     — pairwise scatter of the posterior (identifiability)
    funcn_distance_by_pair.png — weighted median + IQR of per-pair squared error
                                 across populations, one line per FunCN pair
    posterior_summary.txt    — mean / median / 95% CI per parameter, per population

With --ppc N (requires the HCC binary):
    posterior_predictive.png  — per-pair box/jitter of N simulated summaries
                                drawn from the final posterior, with target marked
    posterior_predictive.json — raw params + summaries + target for those runs
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .funcn import PAIR_NAMES, summary_labels
from .run_pyabc import PARAM_BOUNDS, PARAM_PATHS, simulate_summary


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth parameter values (from the base XML stored in ground_truth.json)
# ─────────────────────────────────────────────────────────────────────────────

def load_truth_params(base_xml: Path) -> dict[str, float]:
    root = ET.parse(str(base_xml)).getroot()
    out: dict[str, float] = {}
    for key, dotted in PARAM_PATHS.items():
        node = root
        for p in dotted.split("."):
            node = node.find(p)
            if node is None:
                raise KeyError(f"Missing {dotted} in {base_xml}")
        out[key] = float(node.text)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Weighted statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return 0.0
    wn = w / s
    return float(1.0 / np.sum(wn * wn))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float("nan")
    return float(np.interp(q * total, cw, v))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _extract_summary(stat: Any, expected_len: int) -> np.ndarray:
    """pyabc stores sum-stats as dicts; be tolerant to key/type variations."""
    if isinstance(stat, dict):
        for key in ("summary", "sumstat", "summary_stats"):
            if key in stat:
                return np.asarray(stat[key], dtype=np.float64).ravel()
        # fall-through: single-value dict
        if len(stat) == 1:
            (v,) = stat.values()
            return np.asarray(v, dtype=np.float64).ravel()
    arr = np.asarray(stat, dtype=np.float64).ravel()
    if arr.size == expected_len:
        return arr
    raise ValueError(f"Cannot coerce sum-stat to length {expected_len}: {type(stat)}")


def load_history(db_url: str, run_id: int | None):
    try:
        import pyabc
    except ImportError as e:
        print("pyabc is not installed. pip install pyabc", file=sys.stderr)
        raise SystemExit(2) from e
    if run_id is None:
        return pyabc.History(db_url)
    return pyabc.History(db_url, _id=run_id)


# ─────────────────────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_epsilon(pops, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pops["t"], pops["epsilon"], marker="o")
    ax.set_xlabel("Population t")
    ax.set_ylabel("ε (acceptance threshold)")
    ax.set_yscale("log")
    ax.set_title("ABC-SMC epsilon decay")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ess(history, pops, out: Path) -> None:
    ess = []
    for t in pops["t"]:
        _, w = history.get_distribution(m=0, t=int(t))
        ess.append(_ess(np.asarray(w)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pops["t"], ess, marker="o", color="tab:green")
    ax.set_xlabel("Population t")
    ax.set_ylabel("Effective sample size")
    ax.set_title("Posterior ESS per population")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_samples(pops, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(pops["t"], pops["samples"], color="tab:blue")
    axes[0].set_xlabel("Population t")
    axes[0].set_ylabel("Proposed samples")
    axes[0].set_title("Model evaluations per population")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(pops["t"], np.cumsum(pops["samples"]), marker="o", color="tab:red")
    axes[1].set_xlabel("Population t")
    axes[1].set_ylabel("Cumulative samples")
    axes[1].set_title("Cumulative model evaluations")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_acceptance_rate(pops, out: Path) -> None:
    rate = np.asarray(pops["particles"], dtype=np.float64) / np.maximum(1, np.asarray(pops["samples"], dtype=np.float64))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pops["t"], rate, marker="o", color="tab:purple")
    ax.set_xlabel("Population t")
    ax.set_ylabel("Accepted / proposed")
    ax.set_title("Acceptance rate per population")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_posteriors(history, truth: dict[str, float], out: Path) -> None:
    t_final = history.max_t
    df, w = history.get_distribution(m=0, t=t_final)
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    w = w / s if s > 0 else w

    keys = list(PARAM_BOUNDS.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        lo, hi = PARAM_BOUNDS[key]
        vals = np.asarray(df[key].values, dtype=np.float64)
        ax.hist(vals, bins=20, range=(lo, hi), weights=w, color="tab:blue",
                edgecolor="black", alpha=0.7, density=True)
        # Prior reference
        ax.axhline(1.0 / (hi - lo), color="gray", ls=":", label="prior")
        # Ground truth
        if key in truth:
            ax.axvline(truth[key], color="red", lw=2, label=f"truth={truth[key]:.0f}")
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.set_title(f"posterior (t={t_final})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_corner(history, truth: dict[str, float], out: Path) -> None:
    t_final = history.max_t
    df, w = history.get_distribution(m=0, t=t_final)
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    w = w / s if s > 0 else w

    keys = list(PARAM_BOUNDS.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            ax = axes[i, j]
            if i == j:
                vals = np.asarray(df[ki].values, dtype=np.float64)
                lo, hi = PARAM_BOUNDS[ki]
                ax.hist(vals, bins=20, range=(lo, hi), weights=w,
                        color="tab:blue", edgecolor="black", alpha=0.7, density=True)
                if ki in truth:
                    ax.axvline(truth[ki], color="red", lw=1.5)
            elif i > j:
                from scipy.stats import gaussian_kde
                x = np.asarray(df[kj].values, dtype=np.float64)
                y = np.asarray(df[ki].values, dtype=np.float64)
                xlo, xhi = PARAM_BOUNDS[kj]
                ylo, yhi = PARAM_BOUNDS[ki]
                try:
                    kde = gaussian_kde(np.vstack([x, y]), weights=w)
                    gx = np.linspace(xlo, xhi, 60)
                    gy = np.linspace(ylo, yhi, 60)
                    GX, GY = np.meshgrid(gx, gy)
                    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
                    ax.contourf(GX, GY, Z, levels=8, cmap="Blues")
                    ax.contour(GX, GY, Z, levels=8, colors="steelblue",
                               linewidths=0.4, alpha=0.6)
                except np.linalg.LinAlgError:
                    # Degenerate distribution — fall back to scatter
                    ax.scatter(x, y, s=6, alpha=0.4, color="tab:blue", edgecolors="none")
                if kj in truth and ki in truth:
                    ax.axvline(truth[kj], color="red", lw=1, alpha=0.7)
                    ax.axhline(truth[ki], color="red", lw=1, alpha=0.7)
                ax.set_xlim(xlo, xhi)
                ax.set_ylim(ylo, yhi)
            else:
                ax.set_visible(False)

            if i == n - 1:
                ax.set_xlabel(kj)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(ki)
            elif j != 0:
                ax.set_yticklabels([])
    fig.suptitle("Posterior pairwise (identifiability)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_funcn_distance_by_pair(history, target_vec: np.ndarray, pops, out: Path) -> None:
    """Per-pair squared-error contribution vs population.

    Each pair is a single value (mean interaction weight), so the per-particle
    error is just (sim - target)^2 for that pair index.
    """
    n_pairs = len(PAIR_NAMES)
    t_values = list(pops["t"])

    medians = np.full((len(t_values), n_pairs), np.nan)
    lowers = np.full((len(t_values), n_pairs), np.nan)
    uppers = np.full((len(t_values), n_pairs), np.nan)

    for ti, t in enumerate(t_values):
        try:
            w_list, stats = history.get_weighted_sum_stats(t=int(t))
        except Exception:
            continue
        if not stats:
            continue
        w_arr = np.asarray(w_list, dtype=np.float64)
        vecs = np.stack([_extract_summary(s, target_vec.size) for s in stats])  # (P, 5)
        sq = (vecs - target_vec[None, :]) ** 2                                   # (P, 5)
        for p in range(n_pairs):
            per_particle = sq[:, p]                                              # (P,)
            medians[ti, p] = _weighted_quantile(per_particle, w_arr, 0.5)
            lowers[ti, p]  = _weighted_quantile(per_particle, w_arr, 0.25)
            uppers[ti, p]  = _weighted_quantile(per_particle, w_arr, 0.75)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))
    for p, name in enumerate(PAIR_NAMES):
        ax.plot(t_values, medians[:, p], marker="o", color=colors[p], label=name)
        ax.fill_between(t_values, lowers[:, p], uppers[:, p], color=colors[p], alpha=0.15)
    ax.set_xlabel("Population t")
    ax.set_ylabel("per-pair squared error (weighted IQR)")
    ax.set_yscale("log")
    ax.set_title("FunCN per-pair distance over populations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_posterior_predictive(
    history,
    target_vec: np.ndarray,
    n: int,
    base_xml: Path,
    seed_base: int,
    grid: int | None,
    rng_seed: int,
) -> dict[str, Any]:
    """Sample N particles from the final posterior (weighted) and resimulate.

    Returns a dict with 'params' (list of dicts), 'summaries' (N_ok, 5 array),
    and 'failed' (count of runs that errored).
    """
    t_final = history.max_t
    df, w = history.get_distribution(m=0, t=int(t_final))
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        raise RuntimeError("Final population has zero total weight")
    w = w / s

    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(df), size=n, replace=True, p=w)

    all_summaries: list[np.ndarray] = []
    all_params: list[dict[str, float]] = []
    failed = 0
    for i, row_idx in enumerate(idx):
        row = df.iloc[int(row_idx)]
        params = {k: float(row[k]) for k in PARAM_BOUNDS.keys() if k in row}
        try:
            summ = simulate_summary(
                params=params,
                base_xml=base_xml,
                seed=(seed_base + i) & 0x7FFFFFFF,
                grid=grid,
                keep_workdir=False,
            )
        except Exception as e:
            print(f"  [ppc] run {i} failed ({params}): {e}", file=sys.stderr)
            failed += 1
            continue
        if not np.all(np.isfinite(summ)):
            failed += 1
            continue
        all_params.append(params)
        all_summaries.append(summ)
        print(f"  [ppc] {i+1}/{n} done", file=sys.stderr)

    if not all_summaries:
        raise RuntimeError("All posterior predictive runs failed")

    return {
        "params": all_params,
        "summaries": np.stack(all_summaries),
        "failed": failed,
        "target": target_vec,
    }


def plot_posterior_predictive(ppc: dict[str, Any], out: Path) -> None:
    summaries = ppc["summaries"]            # (N_ok, 5)
    target = np.asarray(ppc["target"], dtype=np.float64)
    n_pairs = len(PAIR_NAMES)

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * n_pairs), 6))
    positions = np.arange(n_pairs) + 1
    bp = ax.boxplot(
        [summaries[:, p] for p in range(n_pairs)],
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.8)
    # Individual PPC draws as faint jitter
    for p in range(n_pairs):
        jitter = np.random.default_rng(p).uniform(-0.12, 0.12, size=summaries.shape[0])
        ax.scatter(np.full(summaries.shape[0], positions[p]) + jitter,
                   summaries[:, p],
                   s=16, alpha=0.35, color="tab:blue", edgecolors="none")
    # Target
    ax.scatter(positions, target, marker="X", s=180, color="red",
               edgecolors="black", linewidths=0.8, zorder=5, label="target")
    short_names = [n.replace("Reference_", "").replace("_Weight_", "\n→ ") for n in PAIR_NAMES]
    ax.set_xticks(positions)
    ax.set_xticklabels(short_names, fontsize=13)
    ax.set_ylabel("Interaction weight", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"Posterior predictive check  "
        f"(N={summaries.shape[0]}, failed={ppc['failed']})",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_ppc_dump(ppc: dict[str, Any], out: Path) -> None:
    payload = {
        "params": ppc["params"],
        "summaries": ppc["summaries"].tolist(),
        "target": np.asarray(ppc["target"]).tolist(),
        "labels": summary_labels(),
        "failed": int(ppc["failed"]),
    }
    out.write_text(json.dumps(payload, indent=2))


def write_posterior_summary(history, truth: dict[str, float], out: Path) -> None:
    lines: list[str] = []
    keys = list(PARAM_BOUNDS.keys())
    lines.append(f"{'pop':>4s} " + " ".join(
        f"{k:>28s}" for k in keys
    ))
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=int(t))
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        cells = []
        for k in keys:
            v = np.asarray(df[k].values, dtype=np.float64)
            mean = float(np.sum(w * v))
            lo = _weighted_quantile(v, w, 0.025)
            hi = _weighted_quantile(v, w, 0.975)
            cells.append(f"{mean:7.2f} [{lo:6.2f},{hi:6.2f}]")
        lines.append(f"{t:>4d} " + " ".join(f"{c:>28s}" for c in cells))

    lines.append("")
    lines.append("truth:")
    for k in keys:
        lines.append(f"  {k} = {truth.get(k, float('nan'))}")

    out.write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="sqlite:///HCC/calibration/abc.db")
    p.add_argument("--target", default="HCC/calibration/ground_truth.json")
    p.add_argument("--out", default="HCC/calibration/figures")
    p.add_argument("--run-id", type=int, default=None,
                   help="PyABC run id (default: latest in db)")
    p.add_argument("--ppc", type=int, default=0, metavar="N",
                   help="Run N posterior-predictive HCC simulations (needs GPU binary). "
                        "Default 0 = skip.")
    p.add_argument("--ppc-seed-base", type=int, default=98765,
                   help="Base seed for PPC simulations (each draw uses base + i)")
    p.add_argument("--ppc-rng-seed", type=int, default=0,
                   help="Seed for the weighted resampling of particles")
    p.add_argument("--ppc-json", default=None, metavar="PATH",
                   help="Re-plot posterior_predictive.png from an existing "
                        "posterior_predictive.json (skips simulation, no GPU needed).")
    args = p.parse_args(argv)

    target_payload = json.loads(Path(args.target).read_text())
    target_vec = np.asarray(target_payload["summary"], dtype=np.float64)
    base_xml = Path(target_payload["base_xml"])
    truth = load_truth_params(base_xml)

    history = load_history(args.db, args.run_id)
    pops = history.get_all_populations()
    # Drop the "init" pre-population row if present (t == -1) so plots start at 0.
    pops = pops[pops["t"] >= 0].reset_index(drop=True)
    if len(pops) == 0:
        print("No populations found in history.", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_epsilon(pops, out_dir / "epsilon.png")
    plot_ess(history, pops, out_dir / "ess.png")
    plot_samples(pops, out_dir / "samples.png")
    plot_acceptance_rate(pops, out_dir / "acceptance_rate.png")
    plot_posteriors(history, truth, out_dir / "posterior_marginals.png")
    plot_corner(history, truth, out_dir / "posterior_corner.png")
    plot_funcn_distance_by_pair(history, target_vec, pops, out_dir / "funcn_distance_by_pair.png")
    write_posterior_summary(history, truth, out_dir / "posterior_summary.txt")

    if args.ppc_json:
        print(f"Re-plotting posterior predictive from {args.ppc_json} ...")
        raw = json.loads(Path(args.ppc_json).read_text())
        ppc_replot = {
            "params": raw["params"],
            "summaries": np.asarray(raw["summaries"], dtype=np.float64),
            "target": np.asarray(raw["target"], dtype=np.float64),
            "failed": int(raw.get("failed", 0)),
        }
        plot_posterior_predictive(ppc_replot, out_dir / "posterior_predictive.png")
    elif args.ppc > 0:
        grid = target_payload.get("grid")
        print(f"Running posterior predictive check: {args.ppc} simulations...")
        ppc = run_posterior_predictive(
            history=history,
            target_vec=target_vec,
            n=args.ppc,
            base_xml=base_xml,
            seed_base=args.ppc_seed_base,
            grid=grid,
            rng_seed=args.ppc_rng_seed,
        )
        plot_posterior_predictive(ppc, out_dir / "posterior_predictive.png")
        write_ppc_dump(ppc, out_dir / "posterior_predictive.json")

    print(f"Wrote {len(list(out_dir.glob('*.png')))} PNGs + summary to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
