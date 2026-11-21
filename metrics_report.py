"""One table and three plots with every metric for every decoder (for RESULTS.md and the talk).

    python metrics_report.py                 # d=5, circuit noise p=0.3%, single-threaded timing
    python metrics_report.py --p 0.005 --threads 12

Writes out/metrics/: report.md, report.json, pareto.png (accuracy vs speed),
depth.png (parallel depth vs distance) and, if out/ours_circuit/stats.csv exists,
curves.png (logical error rate vs p for d = 3, 5, 7).

Timing is single-threaded by default so the comparison with PyMatching (single-threaded C++) is
fair. Our numba kernels also run shots on all cores (`--threads`), which is throughput, not latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qec import BASELINES, OURS, detector_error_model, make_decoder, memory_circuit
from qec.metrics import (fault_correction_rate, latency, logical_error_rate, parallel_depth,
                         sample)

LABELS = {
    "union_find": "Union-Find", "union_find_weighted": "UF weighted", "mwpm": "MWPM (ours)",
    "greedy_matching": "Greedy", "greedy_matching_2opt": "Greedy+2opt", "lookup": "Lookup (d=3)",
    "bp": "BP", "bp_osd": "BP+OSD (ours)", "bp_uf": "BP+UF (ours)",
    "pymatching": "PyMatching", "pymatching_correlated": "PyMatching corr.",
    "bposd": "BP+OSD (ldpc)", "belief_matching": "Belief-matching",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", type=int, default=5)
    ap.add_argument("--p", type=float, default=3e-3)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="out/metrics")
    args = ap.parse_args()
    try:
        import numba
        numba.set_num_threads(args.threads)
    except ImportError:
        pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d, p = args.d, args.p
    circ = memory_circuit(d, p)
    dem = detector_error_model(circ)
    dets, obs = sample(circ, args.shots, seed=2026)
    names = [n for n in OURS if n != "lookup"] + ["pymatching", "pymatching_correlated", "bposd"]
    rows = []
    for name in names:
        dec = make_decoder(name, dem)
        slow = name in ("mwpm", "greedy_matching", "greedy_matching_2opt", "bposd")
        shots = min(args.shots, 3000) if slow else args.shots
        r = logical_error_rate(dec, dets[:shots], obs[:shots])
        t = latency(dec, dets[:min(shots, 3000)], single_shots=100)
        w1 = fault_correction_rate(dec, dem, 1)
        w2 = fault_correction_rate(dec, dem, 2, max_samples=1000, seed=1)
        depth = parallel_depth(dec, dets[:2000])
        row = dict(decoder=name, ours=name in OURS, shots=shots, errors=r.errors,
                   ler_per_round=r.per_round(d), ler_ci=[x for x in r.ci],
                   w1_fail=w1.rate, w2_fail=w2.rate,
                   us_per_shot=t.batch_us_per_shot, us_per_round=t.batch_us_per_shot / d,
                   depth_mean=depth["mean"] if depth else None, depth_max=depth["max"] if depth else None)
        rows.append(row)
        print(f"{name:22s} LER/round {row['ler_per_round']:.2e}  {row['us_per_shot']:8.1f} us/shot  "
              f"w2 fail {row['w2_fail']:.4f}  depth {row['depth_mean']}")

    (out / "report.json").write_text(json.dumps(dict(d=d, p=p, threads=args.threads, rows=rows), indent=1))

    # ---- markdown table
    lines = [f"Circuit-level noise p = {p:.2%}, d = {d}, {d} rounds, Z memory. Timing: {args.threads} thread(s).", "",
             "| decoder | ours? | logical error / round | 1-fault fail | 2-fault fail | µs / shot | µs / round | parallel depth (mean / max) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["ler_per_round"]):
        dep = "–" if r["depth_mean"] is None else f"{r['depth_mean']:.1f} / {r['depth_max']:.0f}"
        lines.append(f"| {LABELS.get(r['decoder'], r['decoder'])} | {'yes' if r['ours'] else 'baseline'} | "
                     f"{r['ler_per_round']:.2e} ({r['errors']}/{r['shots']}) | {r['w1_fail']:.1%} | {r['w2_fail']:.1%} | "
                     f"{r['us_per_shot']:.1f} | {r['us_per_round']:.2f} | {dep} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- Pareto: accuracy vs time
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for r in rows:
        ax.scatter(r["us_per_round"], r["ler_per_round"], s=60,
                   marker="o" if r["ours"] else "s", color="tab:blue" if r["ours"] else "tab:gray")
        ax.annotate(LABELS.get(r["decoder"], r["decoder"]), (r["us_per_round"], r["ler_per_round"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(1.0, color="tab:red", ls="--", lw=1)
    ax.text(1.05, ax.get_ylim()[1] if False else max(r["ler_per_round"] for r in rows), " 1 µs cycle",
            color="tab:red", fontsize=8, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"decode time per round (µs, {args.threads} thread)")
    ax.set_ylabel("logical error rate per round")
    ax.set_title(f"Accuracy vs speed: d={d}, circuit noise p={p:.1%}\n(circles: ours, squares: library baselines)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out / "pareto.png", dpi=150); plt.close(fig)

    # ---- parallel depth vs distance
    fig, ax = plt.subplots(figsize=(6, 4))
    depth_rows = {}
    for name in ["union_find", "union_find_weighted", "bp", "bp_uf"]:
        ys = []
        for dd in (3, 5, 7, 9):
            c = memory_circuit(dd, p)
            dts, _ = sample(c, 1000, seed=9)
            ys.append(parallel_depth(make_decoder(name, detector_error_model(c)), dts)["mean"])
        depth_rows[name] = ys
        ax.plot([3, 5, 7, 9], ys, "o-", label=LABELS[name])
    ax.set_xlabel("distance d (d rounds)"); ax.set_ylabel("mean sequential steps per shot")
    ax.set_title(f"Parallel depth if every edge/check updates at once (p={p:.1%})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "depth.png", dpi=150); plt.close(fig)
    rep = json.loads((out / "report.json").read_text())
    rep["depth_vs_d"] = {"d": [3, 5, 7, 9], **depth_rows}
    (out / "report.json").write_text(json.dumps(rep, indent=1))

    # ---- curves from the sweep, if it has run
    stats = Path("out/ours_circuit/stats.csv")
    if stats.exists():
        import sinter
        from qec import analysis
        pts = analysis.aggregate(sinter.read_stats_from_csv_files(str(stats)))
        series = analysis.by_series(pts)
        show = [s for s in series if any(k in s for k in ("pymatching ", "bp_uf", "bp_osd", "union_find_weighted"))]
        fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.6), sharey=True)
        for ax, s in zip(np.atleast_1d(axes), show):
            for dd, lst in sorted(series[s].items()):
                lst = sorted(lst, key=lambda q: q.p)
                ax.errorbar([q.p for q in lst], [q.per_round for q in lst],
                            yerr=[[q.per_round - q.per_round_lo for q in lst], [q.per_round_hi - q.per_round for q in lst]],
                            marker="o", ms=3, label=f"d={dd}")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            dec = s.split()[1]
            ax.set_title(LABELS.get(dec, dec), fontsize=10); ax.set_xlabel("physical p")
        np.atleast_1d(axes)[0].set_ylabel("logical error / round"); np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "curves.png", dpi=150); plt.close(fig)
    print(f"wrote {out}/report.md, report.json, pareto.png, depth.png")


if __name__ == "__main__":
    main()
