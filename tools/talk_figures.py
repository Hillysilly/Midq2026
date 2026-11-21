"""Slide-sized figures for the talk (out/talk/*.png), from out/ours_circuit and out/metrics/report.json.

    .venv\\Scripts\\python tools\\talk_figures.py

Colours: our decoders orange, library baselines navy (they differ strongly in lightness, so they
stay distinguishable in grayscale and for colour-blind viewers); every mark is also labelled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qec import analysis  # noqa: E402

PAPER, INK, SOFT, GRID = "#F7F5EF", "#14213D", "#4A5568", "#DDD8CC"
OURS, BASE, BLUE = "#C8661A", "#14213D", "#5B8FC9"
LABEL = {"pymatching": "PyMatching", "pymatching_correlated": "PyMatching correlated",
         "bp_uf": "BP+UF (ours)", "bp_osd": "BP+OSD (ours)", "union_find_weighted": "UF weighted (ours)",
         "union_find": "Union-Find (ours)", "mwpm": "MWPM (ours)", "sparse_blossom": "Sparse Blossom (ours)", "greedy_matching": "Greedy (ours)",
         "greedy_matching_2opt": "Greedy+2-opt (ours)", "bposd": "BP+OSD (ldpc library)", "bp": "BP alone (ours)"}
OUT = ROOT / "out" / "talk"

plt.rcParams.update({"font.size": 22, "font.family": "DejaVu Sans", "axes.edgecolor": SOFT,
                     "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.facecolor": PAPER, "figure.facecolor": PAPER, "savefig.facecolor": PAPER})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)


def accuracy_bars(d: int = 7, p: float = 0.004) -> None:
    pts = analysis.aggregate(sinter.read_stats_from_csv_files(str(ROOT / "out/ours_circuit/stats.csv")))
    rows = []
    for pt in pts:
        dec = pt.series.split()[1]
        if pt.d == d and abs(pt.p - p) < 1e-12:
            rows.append((dec, pt.per_round * 1e4, pt.per_round_lo * 1e4, pt.per_round_hi * 1e4))
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ys = range(len(rows))
    for y, (dec, v, lo, hi) in zip(ys, rows):
        ours = dec not in ("pymatching", "pymatching_correlated")
        ax.barh(y, v, height=0.62, color=OURS if ours else BASE)
        ax.plot([lo, hi], [y, y], color=SOFT, lw=2)
        ax.text(hi + 0.25, y, f"{v:.1f}", va="center", color=INK, fontsize=22)
    ax.set_yticks(list(ys), [LABEL[r[0]] for r in rows], fontsize=24, color=INK)
    ax.set_xlabel(f"logical errors per 10,000 rounds   (d = {d}, circuit noise p = {p:.1%}; lower is better)")
    ax.tick_params(axis="y", length=0)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png", dpi=100)
    plt.close(fig)


def pareto() -> None:
    rep = json.loads((ROOT / "out/metrics/report.json").read_text())
    rows = [r for r in rep["rows"] if r["decoder"] != "bp"]
    offs = {"bp_uf": (-16, -4), "bp_osd": (14, 8), "mwpm": (14, 4), "bposd": (-14, 26),
            "pymatching": (16, -6), "pymatching_correlated": (16, -2), "union_find": (16, 6),
            "union_find_weighted": (16, -6), "greedy_matching": (16, -4), "greedy_matching_2opt": (16, -4)}
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.axvline(1.0, color=SOFT, ls="--", lw=2)
    ax.text(1.12, 33, "1 µs: one round on a\nsuperconducting chip", color=SOFT, fontsize=20, va="top")
    for r in rows:
        ours = r["ours"]
        x, y = r["us_per_round"], r["ler_per_round"] * 1e4
        ax.scatter(x, y, s=260, marker="o" if ours else "s", color=OURS if ours else BASE,
                   edgecolor=PAPER, linewidth=2, zorder=3)
        ha = "right" if offs.get(r["decoder"], (1, 0))[0] < 0 else "left"
        ax.annotate(LABEL[r["decoder"]], (x, y), textcoords="offset points", xytext=offs.get(r["decoder"], (14, 4)),
                    fontsize=20, color=INK, ha=ha, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.2, 1500)
    ax.set_xlabel(f"decode time per round, µs (one CPU thread, d = {rep['d']})")
    ax.set_ylabel("logical errors per 10,000 rounds")
    ax.set_ylim(3.5, 40)
    ax.set_yticks([4, 6, 10, 20, 35], ["4", "6", "10", "20", "35"])
    ax.minorticks_off()
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=100)
    plt.close(fig)


def depth() -> None:
    rep = json.loads((ROOT / "out/metrics/report.json").read_text())
    dv = rep["depth_vs_d"]
    ds = dv["d"]
    fig, ax = plt.subplots(figsize=(16, 7.2))
    series = [("bp_uf", OURS, "BP+UF: BP iterations + UF rounds"), ("union_find_weighted", BLUE, "UF weighted: growth rounds"),
              ("union_find", BASE, "Union-Find: growth rounds")]
    for key, col, lab in series:
        ax.plot(ds, dv[key], "-o", color=col, lw=3, ms=12, mec=PAPER, mew=2, label=lab)
        ax.text(ds[-1] + 0.15, dv[key][-1], lab.split(":")[0], color=INK, fontsize=22, va="center")
    ax.set_xlim(2.6, 11.2)
    ax.set_xticks(ds, [f"d = {d}\n({d * d * 2 - 1} qubits)" for d in ds])
    ax.set_ylabel("sequential steps per shot (mean)")
    ax.legend(frameon=False, fontsize=20, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "depth.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_bars()
    pareto()
    depth()
    print("wrote", OUT)
