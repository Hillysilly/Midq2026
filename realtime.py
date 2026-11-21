"""Real-time decoding experiment: sliding windows, buffer size and the backlog problem.

    python realtime.py                       # d = 3, 5, 7, 30 rounds, p = 0.4%
    python realtime.py --round-time-us 1.0   # superconducting-style 1 us syndrome cycle

Panel 1: logical error rate vs buffer size. Small buffers commit bad guesses near the window
         edge, and a buffer of about d rounds recovers the accuracy of decoding everything at once.
Panel 2: decoder latency per window vs the time the hardware takes to produce that window.
         If decoding is slower, the backlog grows without bound (review Appendix A).

Outputs go to out/realtime/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qec import NoiseModel, SurfaceCode, detector_error_model, make_decoder
from qec.analysis import wilson
from qec.sliding_window import SlidingWindowDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--commit", type=int, default=None, help="commit rounds per window (default d)")
    ap.add_argument("--buffers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 7])
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--round-time-us", type=float, default=1.0, help="hardware time per syndrome round")
    args = ap.parse_args()
    out = Path("out") / "realtime"
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    lines = [f"Sliding-window MWPM, {args.rounds} rounds, circuit noise p={args.p}\n"]
    latency = {}
    results: dict = {"p": args.p, "rounds": args.rounds, "round_time_us": args.round_time_us, "distances": {}}
    for d in args.distances:
        circuit = SurfaceCode(d).memory_circuit(NoiseModel.circuit(args.p), rounds=args.rounds)
        dem = detector_error_model(circuit)
        dets, obs = circuit.compile_detector_sampler().sample(args.shots, separate_observables=True)
        glob = np.any(make_decoder("pymatching", dem).decode_batch(dets) != obs, axis=1)
        g_rate = glob.mean()
        commit = args.commit or d
        rates, lo, hi = [], [], []
        for b in args.buffers:
            sw = SlidingWindowDecoder(dem, commit=commit, buffer=b)
            fails = int(np.any(sw.decode_batch(dets) != obs, axis=1).sum())
            rates.append(fails / args.shots)
            l, h = wilson(fails, args.shots)
            lo.append(l)
            hi.append(h)
        line, = ax1.plot(args.buffers, rates, marker="o", label=f"d={d} windowed (commit={commit})")
        ax1.fill_between(args.buffers, lo, hi, alpha=0.2, color=line.get_color())
        ax1.axhline(g_rate, color=line.get_color(), ls="--", lw=1, label=f"d={d} global")
        lines.append(f"d={d}: global {g_rate:.4f}; " +
                     ", ".join(f"buffer {b}: {r:.4f}" for b, r in zip(args.buffers, rates)))

        # Latency: single-shot decode time of one interior window (commit=d, buffer=d).
        sw = SlidingWindowDecoder(dem, commit=d, buffer=d)
        m, _, _, _, window_dets = sw.windows[min(1, sw.num_windows - 1)]
        sample = (dets[:2000] * window_dets)[:, : m.num_detectors]
        t = time.perf_counter()
        for row in sample:
            m.decode(row)
        per_window_us = 1e6 * (time.perf_counter() - t) / len(sample)
        latency[d] = per_window_us
        produce_us = d * args.round_time_us
        results["distances"][d] = {"commit": commit, "buffers": args.buffers, "rates": rates, "lo": lo, "hi": hi,
                                   "global": float(g_rate), "latency_us": per_window_us}
        lines.append(f"     latency per window {per_window_us:.1f} us vs {produce_us:.1f} us to produce "
                     f"{d} rounds -> {'keeps up' if per_window_us < produce_us else 'BACKLOG GROWS'}")

    ax1.set_xlabel("buffer rounds")
    ax1.set_ylabel(f"logical error rate ({args.rounds} rounds)")
    ax1.set_yscale("log")
    ax1.set_title("Windowed decoding needs a buffer of about d rounds")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7)

    ds = sorted(latency)
    ax2.bar([str(d) for d in ds], [latency[d] for d in ds], label="MWPM time per window (this laptop)")
    ax2.plot([str(d) for d in ds], [d * args.round_time_us for d in ds], "r-o",
             label=f"time to produce d rounds @ {args.round_time_us} us/round")
    ax2.set_yscale("log")
    ax2.set_xlabel("code distance d (window = 2d rounds, commit d)")
    ax2.set_ylabel("microseconds")
    ax2.set_title("Can the decoder keep up?")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=150)
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(text)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
