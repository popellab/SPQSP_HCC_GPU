"""Sweep analysis: moveSteps sensitivity on FunCN scores.

Produces a single PDF with three figure panels:

  Fig 1 — Main effects: mean ± 1 SD of each FunCN score vs. each move-step
           parameter value, marginalizing over the other two parameters and all
           seeds.  Flat lines → that parameter doesn't drive the score.

  Fig 2 — Variance decomposition: for each FunCN score, fraction of
           between-group variance attributable to TCell, Fib, Mac moveSteps,
           and seed.  Identifies which knob (or noise) explains the most spread.

  Fig 3 — Seed noise vs. parameter spread: for each combo (125 points), x is
           the combo-level mean across seeds, y is the combo-level SD across
           seeds.  Points near the diagonal → seed noise is comparable to the
           signal range.

Usage
-----
    python -m HCC.calibration.sweep_analysis \
        --results-dir HCC/calibration/results/17644242 \
        [--out sweep_analysis.pdf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ─────────────────────────────────────────────────────────────────

SCORE_COLS = [
    "Reference_CD4T_Weight_Tumor",
    "Reference_CD8T_Weight_Tumor",
    "Reference_Fibroblasts_Weight_Fibroblasts",
    "Reference_Tregs_Weight_Tregs",
    "Reference_CD8T_Weight_Macrophages",
]

# Short labels for the scores (axis titles / legends)
SCORE_LABELS = [
    "CD4T ← Tumor",
    "CD8T ← Tumor",
    "Fib ← Fib",
    "Tregs ← Tregs",
    "CD8T ← Mac",
]

PARAM_COLS  = ["tcell_move_steps", "fib_move_steps", "mac_move_steps"]
PARAM_LABELS = ["TCell moveSteps", "Fib moveSteps", "Mac moveSteps"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(results_dir: Path) -> pd.DataFrame:
    csv = results_dir / "sweep_results.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Expected merged CSV at {csv}")
    df = pd.read_csv(csv)
    # Drop any rows with errors or NaN scores
    df = df[df["error"].isna() | (df["error"] == "")]
    df = df.dropna(subset=SCORE_COLS)
    print(f"Loaded {len(df)} valid rows from {csv}")
    return df


def between_group_var(df: pd.DataFrame, group_col: str, score_col: str) -> float:
    """Variance of group means (marginalizing over everything else)."""
    group_means = df.groupby(group_col)[score_col].mean()
    return float(group_means.var(ddof=1))


# ── Figure 1: Main effects ────────────────────────────────────────────────────

def plot_main_effects(df: pd.DataFrame, ax_grid: np.ndarray) -> None:
    """3 rows (params) × 5 cols (scores), mean ± 1 SD."""
    colors = plt.cm.tab10.colors

    for row, (param, plabel) in enumerate(zip(PARAM_COLS, PARAM_LABELS)):
        for col, (score, slabel) in enumerate(zip(SCORE_COLS, SCORE_LABELS)):
            ax = ax_grid[row, col]
            grouped = df.groupby(param)[score]
            means = grouped.mean()
            stds  = grouped.std(ddof=1)
            xs = means.index.values

            ax.errorbar(xs, means.values, yerr=stds.values,
                        fmt="o-", capsize=4, color=colors[col],
                        linewidth=1.5, markersize=5, elinewidth=1)
            ax.set_xlim(xs[0] - (xs[-1] - xs[0]) * 0.08,
                        xs[-1] + (xs[-1] - xs[0]) * 0.08)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.5)

            if row == 0:
                ax.set_title(slabel, fontsize=8, fontweight="bold")
            if col == 0:
                ax.set_ylabel(plabel, fontsize=8)
            if row == len(PARAM_COLS) - 1:
                ax.set_xlabel("Parameter value", fontsize=7)


# ── Figure 2: Variance decomposition ─────────────────────────────────────────

def plot_variance_decomp(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Stacked bar: fraction of between-group variance per factor per FunCN score."""
    factors      = PARAM_COLS + ["seed"]
    factor_labels = PARAM_LABELS + ["Seed"]
    colors       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    frac_matrix = np.zeros((len(SCORE_COLS), len(factors)))
    for s_idx, score in enumerate(SCORE_COLS):
        vars_ = np.array([between_group_var(df, f, score) for f in factors])
        total = vars_.sum()
        if total > 0:
            frac_matrix[s_idx] = vars_ / total

    x = np.arange(len(SCORE_COLS))
    bottoms = np.zeros(len(SCORE_COLS))
    for f_idx, (flabel, color) in enumerate(zip(factor_labels, colors)):
        vals = frac_matrix[:, f_idx]
        ax.bar(x, vals, bottom=bottoms, label=flabel, color=color, width=0.6,
               edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_LABELS, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Fraction of between-group variance", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(title="Factor", fontsize=8, title_fontsize=8, loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_title("Variance decomposition by factor", fontsize=10, fontweight="bold")


# ── Figure 3: Seed noise vs parameter spread ──────────────────────────────────

def plot_noise_vs_spread(df: pd.DataFrame, ax_row: np.ndarray) -> None:
    """Per-combo mean vs SD across seeds, one subplot per FunCN score."""
    combo_stats = (
        df.groupby(["tcell_move_steps", "fib_move_steps", "mac_move_steps"])[SCORE_COLS]
        .agg(["mean", "std"])
    )
    # combo_stats columns are MultiIndex (score, stat)
    colors = plt.cm.tab10.colors

    for col, (score, slabel) in enumerate(zip(SCORE_COLS, SCORE_LABELS)):
        ax = ax_row[col]
        means = combo_stats[score]["mean"].values
        stds  = combo_stats[score]["std"].fillna(0).values

        ax.scatter(means, stds, s=18, alpha=0.65, color=colors[col], linewidths=0)

        # Reference line: SD = 10% of signal range
        xmin, xmax = means.min(), means.max()
        signal_range = xmax - xmin if xmax > xmin else 1e-6
        ax.axhline(0.1 * signal_range, color="gray", linestyle="--",
                   linewidth=0.8, label="10% of range")

        ax.set_xlabel("Combo mean", fontsize=8)
        ax.set_ylabel("Seed SD", fontsize=8)
        ax.set_title(slabel, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(True, linewidth=0.4, alpha=0.5)
        if col == 0:
            ax.legend(fontsize=7)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="HCC/calibration/results/17644242",
                   help="Directory containing sweep_results.csv")
    p.add_argument("--out", default=None,
                   help="Output PDF path (default: <results-dir>/sweep_analysis.pdf)")
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    df = load(results_dir)

    out_path = Path(args.out) if args.out else results_dir / "sweep_analysis.pdf"

    n_scores = len(SCORE_COLS)
    n_params = len(PARAM_COLS)

    # ── Layout ────────────────────────────────────────────────────────────────
    # 3 figures as separate pages in a single PDF
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_path) as pdf:

        # Fig 1: Main effects (3 rows × 5 cols)
        fig1, axes1 = plt.subplots(n_params, n_scores,
                                   figsize=(3.2 * n_scores, 2.8 * n_params),
                                   constrained_layout=True)
        fig1.suptitle(
            "Main effects: FunCN score vs. moveSteps parameter\n"
            "(mean ± 1 SD, marginalised over other params & seeds)",
            fontsize=11, fontweight="bold"
        )
        plot_main_effects(df, axes1)
        pdf.savefig(fig1, dpi=150)
        plt.close(fig1)

        # Fig 2: Variance decomposition (single axes)
        fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)
        plot_variance_decomp(df, ax2)
        pdf.savefig(fig2, dpi=150)
        plt.close(fig2)

        # Fig 3: Seed noise vs parameter spread (1 row × 5 cols)
        fig3, axes3 = plt.subplots(1, n_scores,
                                   figsize=(3.2 * n_scores, 3.8),
                                   constrained_layout=True)
        fig3.suptitle(
            "Seed noise vs. parameter spread\n"
            "(each point = one (TCell, Fib, Mac) combo; y = SD across 5 seeds)",
            fontsize=11, fontweight="bold"
        )
        plot_noise_vs_spread(df, axes3)
        pdf.savefig(fig3, dpi=150)
        plt.close(fig3)

    print(f"Saved: {out_path}")

    # ── Print a quick text summary ────────────────────────────────────────────
    print("\n=== Quick summary ===")
    print(f"{'Score':<42}  {'Total range':>11}  {'Mean seed SD':>12}  {'Noise/range':>11}")
    print("-" * 82)
    combo_stats = (
        df.groupby(["tcell_move_steps", "fib_move_steps", "mac_move_steps"])[SCORE_COLS]
        .agg(["mean", "std"])
    )
    for score, slabel in zip(SCORE_COLS, SCORE_LABELS):
        combo_means = combo_stats[score]["mean"]
        combo_stds  = combo_stats[score]["std"].fillna(0)
        sig_range   = combo_means.max() - combo_means.min()
        mean_sd     = combo_stds.mean()
        ratio       = mean_sd / sig_range if sig_range > 1e-9 else float("nan")
        print(f"{slabel:<42}  {sig_range:>11.4f}  {mean_sd:>12.4f}  {ratio:>10.1%}")


if __name__ == "__main__":
    main()
