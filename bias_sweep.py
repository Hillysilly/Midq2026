"""Threshold vs noise bias for the CSS and XZZX surface codes (code-capacity noise, MWPM).

    python bias_sweep.py

As eta = pZ/(pX+pY) grows, the CSS code's threshold *drops*, because one decoding sub-problem
gets all the errors. The XZZX code's threshold *rises* toward 50% (review §6, Bonilla Ataides et
al. 2021). The same qubits and decoder are used for both; only the stabilizers are deformed.

Outputs go to out/bias_sweep/: stats.csv (resumable), plot.png and thresholds.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sinter

from qec import analysis
from qec.noise import NoiseModel
from qec.surface_code import SurfaceCode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--biases", type=float, nargs="+", default=[0.5, 1, 3, 10, 30, 100, 1000])
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 9, 13])
    ap.add_argument("--max-shots", type=int, default=50_000)
    ap.add_argument("--max-errors", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    out = Path("out") / "bias_sweep"
    out.mkdir(parents=True, exist_ok=True)

    ps = list(np.round(np.geomspace(0.04, 0.48, 16), 4))
    tasks = []
    for variant in ("css", "xzzx"):
        for d in args.distances:
            code = SurfaceCode(d, variant=variant)
            for bias in args.biases:
                for p in ps:
                    tasks.append(sinter.Task(
                        circuit=code.memory_circuit_ideal(NoiseModel.code_capacity(float(p), bias)),
                        json_metadata={"variant": variant, "bias": bias, "d": d, "p": float(p), "r": 1},
                    ))
    stats = sinter.collect(
        num_workers=args.workers, tasks=tasks, decoders=["pymatching"],
        max_shots=args.max_shots, max_errors=args.max_errors,
        save_resume_filepath=str(out / "stats.csv"), print_progress=True,
    )
    points = analysis.aggregate(stats, series_keys=("variant", "bias"))
    result: dict[str, dict[str, float | None]] = {"css": {}, "xzzx": {}}
    for series, pairs in analysis.thresholds(points, per_round_rates=False).items():
        variant, bias = series.split(" bias=")
        crosses = [c for _, _, c in pairs if c]
        result[variant][bias] = float(np.mean(crosses)) if crosses else None
    (out / "thresholds.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for variant, colour in (("css", "tab:blue"), ("xzzx", "tab:red")):
        xs = [float(b) for b, v in result[variant].items() if v]
        ys = [v for v in result[variant].values() if v]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color=colour, label=variant.upper())
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("bias eta = pZ / (pX + pY)")
    ax.set_ylabel("code-capacity threshold (MWPM)")
    ax.set_title("Biased noise hurts CSS and helps XZZX")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=150)
    print(json.dumps(result, indent=2))
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
