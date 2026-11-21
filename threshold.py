"""Threshold sweep: logical error rate vs physical error rate for several distances and decoders.

    .venv\\Scripts\\python threshold.py                          # MWPM, d=3,5,7
    .venv\\Scripts\\python threshold.py --decoders pymatching union_find --max-shots 20000

The curves for different distances cross at the threshold. Below it, a larger
code gives a lower logical error rate. With this circuit-level noise model,
expect the curves to cross near p ~ 0.7-1%. The review's 10-18% thresholds are for
code-capacity noise (data errors only), which is much easier.

Results go to out/threshold_stats.csv (an interrupted run resumes from it) and
out/threshold.png.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter

from qec import UnionFindSinterDecoder, memory_circuit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012])
    ap.add_argument("--decoders", nargs="+", default=["pymatching"], help="pymatching and/or union_find")
    ap.add_argument("--max-shots", type=int, default=100_000)
    ap.add_argument("--max-errors", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    out = Path("out")
    out.mkdir(exist_ok=True)

    tasks = [
        sinter.Task(
            circuit=memory_circuit(d, p, rounds=d),
            json_metadata={"d": d, "p": p, "r": d},
        )
        for d in args.distances
        for p in args.ps
    ]

    stats = sinter.collect(
        num_workers=args.workers,
        tasks=tasks,
        decoders=args.decoders,
        custom_decoders={"union_find": UnionFindSinterDecoder()},
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        save_resume_filepath=str(out / "threshold_stats.csv"),
        print_progress=True,
    )

    decoders = sorted({s.decoder for s in stats})
    fig, axes = plt.subplots(1, len(decoders), figsize=(6 * len(decoders), 5), squeeze=False, sharey=True)
    for ax, dec in zip(axes[0], decoders):
        sinter.plot_error_rate(
            ax=ax,
            stats=[s for s in stats if s.decoder == dec],
            x_func=lambda s: s.json_metadata["p"],
            group_func=lambda s: f"d={s.json_metadata['d']}",
            failure_units_per_shot_func=lambda s: s.json_metadata["r"],
        )
        ax.set_title(f"Rotated surface code memory, {dec}")
        ax.set_xlabel("physical error rate p")
        ax.loglog()
        ax.grid(which="both", alpha=0.3)
        ax.legend()
    axes[0][0].set_ylabel("logical error rate per round")
    fig.tight_layout()
    fig.savefig(out / "threshold.png", dpi=150)
    print(f"Saved {(out / 'threshold.png').resolve()}")


if __name__ == "__main__":
    main()
