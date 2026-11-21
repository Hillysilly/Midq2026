"""Benchmark codes x noise models x decoders, with threshold estimates.

Examples
--------
    # Reproduce the review's code-capacity thresholds (Fig. 27): MWPM ~14-15%, UF ~11-12%
    python bench.py --name code_capacity --noise code-capacity --ps 0.08 0.10 0.12 0.14 0.16 0.18 \\
        --decoders pymatching union_find

    # Circuit-level noise, both memory bases
    python bench.py --name circuit --noise circuit --basis both

    # Biased noise: CSS vs XZZX
    python bench.py --name bias100 --noise circuit --bias 100 --variants css xzzx --basis both

Each run writes to out/<name>/: stats.csv (resumable), plot.png and summary.txt.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter

from qec import analysis
from qec.decoders import sinter_decoders
from qec.noise import NoiseModel
from qec.surface_code import SurfaceCode

NOISE_PRESETS = {
    "code-capacity": lambda p, bias: NoiseModel.code_capacity(p, bias),
    "phenomenological": lambda p, bias: NoiseModel.phenomenological(p, bias=bias),
    "circuit": lambda p, bias: NoiseModel.circuit(p, bias),
    "si1000": lambda p, bias: NoiseModel.si1000(p),
}

DEFAULT_PS = {
    "code-capacity": [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
    "phenomenological": [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055],
    "circuit": [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012],
    "si1000": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008],
}


def build_tasks(args) -> list[sinter.Task]:
    tasks = []
    bases = ["Z", "X"] if args.basis == "both" else [args.basis.upper()]
    for variant in args.variants:
        for d in args.distances:
            code = SurfaceCode(d, variant=variant)
            for p in args.ps:
                noise = NOISE_PRESETS[args.noise](p, args.bias)
                meta = {"variant": variant, "noise": args.noise, "bias": args.bias, "d": d, "p": p}
                if args.noise == "code-capacity":
                    tasks.append(sinter.Task(circuit=code.memory_circuit_ideal(noise),
                                             json_metadata={**meta, "r": 1, "basis": "ZX"}))
                elif args.noise == "phenomenological":
                    r = args.rounds or d
                    tasks.append(sinter.Task(circuit=code.memory_circuit_ideal(noise, rounds=r),
                                             json_metadata={**meta, "r": r, "basis": "ZX"}))
                else:
                    r = args.rounds or d
                    for b in bases:
                        tasks.append(sinter.Task(circuit=code.memory_circuit(noise, rounds=r, basis=b),
                                                 json_metadata={**meta, "r": r, "basis": b}))
    return tasks


def plot(points, path: Path, title: str, per_round: bool) -> None:
    series = analysis.by_series(points)
    n = len(series)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False, sharey=True)
    for ax, (name, curves) in zip(axes.flat, sorted(series.items())):
        for d in sorted(curves):
            pts = [pt for pt in curves[d] if pt.errors > 0]
            if not pts:
                continue
            xs = [pt.p for pt in pts]
            ys = [pt.per_round if per_round else pt.rate for pt in pts]
            lo = [pt.per_round_lo if per_round else pt.lo for pt in pts]
            hi = [pt.per_round_hi if per_round else pt.hi for pt in pts]
            line, = ax.plot(xs, ys, marker="o", label=f"d={d}")
            ax.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("physical error rate p")
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0][0].set_ylabel("logical error rate" + (" per round" if per_round else " per shot"))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(points, per_round: bool) -> str:
    lines = []
    for series, pairs in analysis.thresholds(points, per_round).items():
        crosses = [c for _, _, c in pairs if c]
        txt = ", ".join(f"d{a}/d{b}: {c:.4g}" if c else f"d{a}/d{b}: no crossing" for a, b, c in pairs)
        est = f"~{sum(crosses) / len(crosses):.4g}" if crosses else "not bracketed by the p range"
        lines.append(f"{series}\n  threshold {est}   ({txt})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output folder name under out/")
    ap.add_argument("--noise", choices=list(NOISE_PRESETS), default="circuit")
    ap.add_argument("--bias", type=float, default=0.5, help="eta = pZ/(pX+pY); 0.5 = depolarizing")
    ap.add_argument("--variants", nargs="+", default=["css"], choices=["css", "xzzx"])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--ps", type=float, nargs="+", default=None)
    ap.add_argument("--rounds", type=int, default=None, help="default: d rounds")
    ap.add_argument("--basis", default="Z", choices=["Z", "X", "z", "x", "both"],
                    help="memory basis for circuit-level noise ('both' sums the Z and X failure rates)")
    ap.add_argument("--decoders", nargs="+", default=["pymatching"])
    ap.add_argument("--max-shots", type=int, default=200_000)
    ap.add_argument("--max-errors", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    args.ps = args.ps or DEFAULT_PS[args.noise]

    out = Path("out") / args.name
    out.mkdir(parents=True, exist_ok=True)
    sinter.collect(
        num_workers=args.workers,
        tasks=build_tasks(args),
        decoders=args.decoders,
        custom_decoders=sinter_decoders(),
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        save_resume_filepath=str(out / "stats.csv"),
        print_progress=True,
    )
    # Plot everything in the folder, including decoders added by earlier runs with the same --name.
    stats = sinter.read_stats_from_csv_files(str(out / "stats.csv"))
    per_round = args.noise != "code-capacity"
    points = analysis.aggregate(stats)
    title = f"{args.noise} noise" + (f", bias eta={args.bias:g}" if args.bias != 0.5 else "")
    plot(points, out / "plot.png", title, per_round)
    text = summarize(points, per_round)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
