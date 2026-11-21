"""Decoder speed vs accuracy: microseconds per shot and logical error rate, per decoder and distance.

    python speed.py                              # circuit noise p = 0.3%, d = 3, 5, 7, 9
    python speed.py --decoders pymatching union_find --distances 3 5 7 9 11

Run it on an otherwise idle machine: timings are single-threaded wall-clock time.
Outputs go to out/speed/ (plot.png, results.json, summary.txt).
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

SLOW = {"bposd", "belief_matching"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--decoders", nargs="+",
                    default=["pymatching", "pymatching_correlated", "union_find", "union_find_weighted", "belief_matching"])
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--slow-shots", type=int, default=2_000, help="shots for BP-based decoders")
    args = ap.parse_args()
    out = Path("out") / "speed"
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[int, dict[str, float]]] = {dec: {} for dec in args.decoders}
    lines = [f"circuit noise p={args.p}, d rounds, Z memory, single thread\n",
             f"{'decoder':22s} {'d':>3} {'us/shot':>10} {'us/round':>9} {'logical err':>12}"]
    for d in args.distances:
        circuit = SurfaceCode(d).memory_circuit(NoiseModel.circuit(args.p), rounds=d)
        dem = detector_error_model(circuit)
        dets, obs = circuit.compile_detector_sampler(seed=d).sample(args.shots, separate_observables=True)
        for dec in args.decoders:
            n = args.slow_shots if dec in SLOW else args.shots
            decoder = make_decoder(dec, dem)
            decoder.decode_batch(dets[:10])  # warm-up
            t = time.perf_counter()
            pred = decoder.decode_batch(dets[:n])
            us = 1e6 * (time.perf_counter() - t) / n
            ler = float(np.any(pred != obs[:n], axis=1).mean())
            results[dec][d] = {"us_per_shot": us, "us_per_round": us / d, "ler": ler, "shots": n}
            lines.append(f"{dec:22s} {d:>3} {us:>10.1f} {us / d:>9.2f} {ler:>12.4f}")
            print(lines[-1], flush=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for dec, by_d in results.items():
        ds = sorted(by_d)
        a1.plot(ds, [by_d[d]["us_per_round"] for d in ds], "o-", label=dec)
        a2.scatter([by_d[d]["us_per_round"] for d in ds], [max(by_d[d]["ler"], 1e-5) for d in ds], label=dec)
        for d in ds:
            a2.annotate(f"d={d}", (by_d[d]["us_per_round"], max(by_d[d]["ler"], 1e-5)), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    a1.axhline(1.0, color="red", ls="--", lw=1, label="1 us/round (superconducting cycle)")
    a1.set_yscale("log")
    a1.set_xlabel("code distance d")
    a1.set_ylabel("decode time per round (us, this laptop)")
    a1.set_title("Decoder cost grows with distance")
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=7)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("decode time per round (us)")
    a2.set_ylabel("logical error rate (d rounds)")
    a2.set_title("Accuracy vs speed (down-left is better)")
    a2.grid(alpha=0.3, which="both")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=150)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
