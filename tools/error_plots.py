"""Logical-error-rate plots with 95% Wilson error bars, from the Sinter stats in out/.
(Written by Claude 2, 2026-11-21.)

    python tools/error_plots.py                      # all plots -> out/error_plots/
    python tools/error_plots.py --verify out/verify  # also overlay a fresh run on the stored data

Plots
-----
ler_circuit.png        per-round logical error rate vs p, circuit noise, one panel per distance
ratio_vs_mwpm.png      each decoder's logical error rate divided by PyMatching's (d = 5, 7)
lambda.png             error suppression Lambda = eps_5 / eps_7 vs p
ler_code_capacity.png  per-shot logical error rate vs p, code capacity, d = 5, 9, 13
verify.png             (--verify) a fresh run vs the stored numbers it should reproduce
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import sinter

from qec import analysis

# Library baselines are neutral gray; our decoders take categorical slots in fixed order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "pymatching":            dict(label="PyMatching (baseline)", color="#6f6e69", ls="-",  marker="s"),
    "pymatching_correlated": dict(label="PyMatching corr. (baseline)", color="#6f6e69", ls="--", marker="D"),
    "union_find":            dict(label="Union-Find (ours)", color="#2a78d6", ls="-", marker="o"),
    "union_find_weighted":   dict(label="UF weighted (ours)", color="#eb6834", ls="-", marker="^"),
    "bp_uf":                 dict(label="BP+UF (ours)", color="#1baf7a", ls="-", marker="v"),
    "bp_osd":                dict(label="BP+OSD (ours)", color="#6250d6", ls="-", marker="P"),
    "belief_matching":       dict(label="Belief-matching (baseline)", color="#6f6e69", ls=":", marker="X"),
}
ORDER = list(STYLE)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
})


def load(name: str) -> dict[str, dict[int, list[analysis.Point]]]:
    """{decoder: {d: [Point sorted by p]}} for one out/<name>/stats.csv."""
    stats = sinter.read_stats_from_csv_files(str(ROOT / "out" / name / "stats.csv"))
    pts = analysis.aggregate(stats, series_keys=("decoder",))
    return {k: dict(v) for k, v in analysis.by_series(pts).items()}


def val(pt, per_round):
    return (pt.per_round, pt.per_round_lo, pt.per_round_hi) if per_round else (pt.rate, pt.lo, pt.hi)


def errbar(ax, pts, per_round, style, label=True, **kw):
    pts = [pt for pt in pts if pt.errors > 0]
    if not pts:
        return
    xs = [pt.p for pt in pts]
    y, lo, hi = zip(*(val(pt, per_round) for pt in pts))
    yerr = [[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]]
    s = {**style, **kw}
    ax.errorbar(xs, y, yerr=yerr, color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"] if label else None, capsize=2.5, elinewidth=1,
                markeredgecolor=SURFACE, markeredgewidth=0.8, alpha=s.get("alpha", 1))


CIRC_TICKS = [0.002, 0.003, 0.005, 0.008, 0.012]
CC_TICKS = [0.08, 0.10, 0.12, 0.15, 0.18]


def loglog(ax, xticks=CIRC_TICKS):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="minor", alpha=0.35)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100 * x:g}%"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_ler(data, path, title, per_round, distances, xticks=CIRC_TICKS):
    fig, axes = plt.subplots(1, len(distances), figsize=(4.6 * len(distances), 4.4), sharey=True)
    for ax, d in zip(axes, distances):
        for dec in ORDER:
            if dec in data and d in data[dec]:
                errbar(ax, data[dec][d], per_round, STYLE[dec])
        loglog(ax, xticks)
        ax.set_title(f"d = {d}", fontsize=11)
        ax.set_xlabel("physical error rate p")
    axes[0].set_ylabel("logical error rate " + ("per round" if per_round else "per shot"))
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title + "   (bars: 95% Wilson interval)", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ratio(a, b):
    """a/b with a conservative interval from the two Wilson intervals."""
    (ya, la, ha), (yb, lb, hb) = a, b
    return ya / yb, la / hb, ha / max(lb, 1e-300)


def plot_ratio(data, path, distances=(5, 7), ref="pymatching"):
    fig, axes = plt.subplots(1, len(distances), figsize=(5.2 * len(distances), 4.4), sharey=True)
    for ax, d in zip(axes, distances):
        refs = {pt.p: pt for pt in data[ref][d] if pt.errors > 0}
        for dec in ORDER:
            if dec == ref or dec not in data or d not in data[dec]:
                continue
            rows = [(pt.p, ratio(val(pt, True), val(refs[pt.p], True)))
                    for pt in data[dec][d] if pt.errors > 0 and pt.p in refs]
            if not rows:
                continue
            xs = [p for p, _ in rows]
            y = [r[0] for _, r in rows]
            yerr = [[r[0] - r[1] for _, r in rows], [r[2] - r[0] for _, r in rows]]
            s = STYLE[dec]
            ax.errorbar(xs, y, yerr=yerr, color=s["color"], ls=s["ls"], marker=s["marker"],
                        label=s["label"], capsize=2.5, elinewidth=1,
                        markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.axhline(1, color=INK2, lw=1)
        ax.text(min(refs), 1.03, " = PyMatching", color=INK2, fontsize=8, va="bottom")
        loglog(ax)
        ax.set_title(f"d = {d}", fontsize=11)
        ax.set_xlabel("physical error rate p")
    axes[0].set_ylabel("logical errors relative to PyMatching\n(below 1 = fewer errors)")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Accuracy relative to standard MWPM, circuit noise", fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lambda(data, path, d_small=5, d_large=7):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for dec in ORDER:
        if dec not in data or d_small not in data[dec] or d_large not in data[dec]:
            continue
        big = {pt.p: pt for pt in data[dec][d_large] if pt.errors > 0}
        rows = [(pt.p, ratio(val(pt, True), val(big[pt.p], True)))
                for pt in data[dec][d_small] if pt.errors > 0 and pt.p in big]
        if not rows:
            continue
        xs = [p for p, _ in rows]
        y = [r[0] for _, r in rows]
        yerr = [[r[0] - r[1] for _, r in rows], [r[2] - r[0] for _, r in rows]]
        s = STYLE[dec]
        ax.errorbar(xs, y, yerr=yerr, color=s["color"], ls=s["ls"], marker=s["marker"],
                    label=s["label"], capsize=2.5, elinewidth=1,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.axhline(1, color=INK2, lw=1)
    ax.text(0.0021, 0.96, "Λ = 1: no gain from a bigger code (threshold)", color=INK2, fontsize=8, va="top")
    loglog(ax)
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel(f"Λ = ε(d={d_small}) / ε(d={d_large})")
    ax.set_title("Error suppression from d = 5 to 7 (higher is better)", fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_verify(fresh, stored, path):
    decs = [d for d in ORDER if d in fresh]
    dists = sorted({d for dec in decs for d in fresh[dec]})
    fig, axes = plt.subplots(1, len(dists), figsize=(4.8 * len(dists), 4.4), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], dists):
        for dec in decs:
            s = STYLE[dec]
            if dec in stored and d in stored[dec]:
                errbar(ax, stored[dec][d], True, s, label=False, alpha=0.35, ls=":")
            errbar(ax, fresh[dec].get(d, []), True, s)
        loglog(ax)
        ax.set_title(f"d = {d}", fontsize=11)
        ax.set_xlabel("physical error rate p")
    axes[0][0].set_ylabel("logical error rate per round")
    h, l = axes[0][-1].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK2, ls=":", alpha=0.5, marker="o"))
    l.append("stored result (out/ours_circuit)")
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Fresh run today (solid) vs stored results (faded)   (bars: 95% Wilson)", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/error_plots")
    ap.add_argument("--verify", default=None, help="out/<name> folder of a fresh circuit-noise run")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    circ = load("ours_circuit")
    plot_ler(circ, out / "ler_circuit.png", "Circuit-level noise, d rounds, Z memory", True, [3, 5, 7])
    plot_ratio(circ, out / "ratio_vs_mwpm.png")
    plot_lambda(circ, out / "lambda.png")
    cc = load("ours_code_capacity")
    plot_ler(cc, out / "ler_code_capacity.png", "Code-capacity noise (both logicals)", False, [5, 9, 13], CC_TICKS)
    if args.verify:
        plot_verify(load(Path(args.verify).name), circ, out / "verify.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
