"""Decode-only speed table: µs per shot for each decoder, distance and thread count (markdown output).

    .venv\\Scripts\\python tools\\decode_speed.py --p 0.001 --distances 3 5 7 9

Shots are sampled once per distance and decoded with `decode_batch` after a warm-up call, so the
times exclude Stim sampling and numba compilation. Numba decoders are timed on 1 and on all threads;
PyMatching doesn't release Python's GIL, so threads don't change it (one call = one core).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qec import detector_error_model, make_decoder, memory_circuit  # noqa: E402

SLOW = {"sparse_blossom": 3000, "mwpm": 1000, "bp_uf": 3000, "bp_uf_local": 10000, "bp_osd": 1000}
NUMBA = {"union_find", "union_find_weighted", "bp", "bp_uf", "bp_uf_local"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=1e-3)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--decoders", nargs="+",
                    default=["pymatching", "pymatching_correlated", "union_find", "union_find_weighted", "sparse_blossom", "mwpm"])
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--max-d-slow", type=int, default=7, help="largest distance for the slow pure-Python decoders")
    args = ap.parse_args()
    import numba
    all_threads = os.cpu_count() or 1
    rows: dict[str, dict[int, str]] = {}
    for d in args.distances:
        c = memory_circuit(d, args.p)
        dem = detector_error_model(c)
        n = args.shots if d <= 7 else args.shots // 4
        dets = c.compile_detector_sampler(seed=2026).sample(n, separate_observables=True)[0]
        for name in args.decoders:
            if name in SLOW and d > args.max_d_slow:
                continue
            dec = make_decoder(name, dem)
            shots = dets[: SLOW.get(name, n)]
            dec.decode_batch(shots[:50])
            cells = []
            for th in ([1, all_threads] if name in NUMBA else [1]):
                numba.set_num_threads(th)
                t = time.perf_counter()
                dec.decode_batch(shots)
                us = 1e6 * (time.perf_counter() - t) / len(shots)
                cells.append(f"{us:.2f}" if us < 10 else f"{us:.1f}" if us < 100 else f"{us:,.0f}")
            numba.set_num_threads(all_threads)
            rows.setdefault(name, {})[d] = " / ".join(cells)
            print(f"d={d} {name}: {rows[name][d]} us/shot", flush=True)
    print(f"\n| decoder (µs per shot{', 1 / ' + str(all_threads) + ' threads where shown'}) | " + " | ".join(f"d={d}" for d in args.distances) + " |")
    print("|---|" + "---|" * len(args.distances))
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(r.get(d, "–") for d in args.distances) + " |")


if __name__ == "__main__":
    main()
