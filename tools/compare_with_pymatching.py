"""Same-shot comparison of one of our decoders against PyMatching.

    .venv\\Scripts\\python tools\\compare_with_pymatching.py --decoder sparse_blossom
    .venv\\Scripts\\python tools\\compare_with_pymatching.py --decoder mwpm --distances 3 5 --shots 1000

Both decoders see exactly the same sampled shots, so any difference is the decoder's, not sampling noise.
(Separate Sinter runs draw different shots per decoder, so their logical error rates differ by chance even
for identical decoders.) For exact-matching decoders it also compares the total matching weight.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pymatching

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qec import detector_error_model, make_decoder, memory_circuit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="sparse_blossom")
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--ps", type=float, nargs="+", default=[1e-3, 3e-3, 5e-3, 1e-2])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    print(f"| d | p | shots | same prediction | logical errors (ours / PyMatching) | max matching-weight excess |")
    print("|---|---|---|---|---|---|")
    for d in args.distances:
        for p in args.ps:
            c = memory_circuit(d, p)
            dem = detector_error_model(c)
            ours = make_decoder(args.decoder, dem)
            m = pymatching.Matching.from_detector_error_model(dem)
            dets, obs = c.compile_detector_sampler(seed=args.seed).sample(args.shots, separate_observables=True)
            a = np.asarray(ours.decode_batch(dets), dtype=bool)
            b = m.decode_batch(dets).astype(bool)
            excess = "–"
            if hasattr(ours, "matching_weight"):
                worst = 0.0
                for row in dets:
                    _, w_pm = m.decode(row, return_weight=True)
                    worst = max(worst, (ours.matching_weight(row) - w_pm) / max(w_pm, 1e-9))
                excess = f"{worst:.1e}"
            print(f"| {d} | {p:.1%} | {args.shots} | {(a == b).all(1).sum()}/{args.shots} | "
                  f"{(a != obs).any(1).sum()} / {(b != obs).any(1).sum()} | {excess} |", flush=True)


if __name__ == "__main__":
    main()
