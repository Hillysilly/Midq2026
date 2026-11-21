"""Quickstart: simulate one noisy surface-code memory experiment and decode it two ways.

    .venv\\Scripts\\python quickstart.py --distance 5 --p 0.003 --shots 20000

Pipeline (the two PDFs in resources/ cover each step):
    Stim circuit -> detector error model (hypergraph) -> sampled detection events
    -> decoder (MWPM via PyMatching, or our Union-Find) -> predicted logical flips
    -> compare with the actual flips -> logical error rate
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pymatching

from qec import UnionFindDecoder, detector_error_model, memory_circuit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", "-d", type=int, default=5)
    ap.add_argument("--rounds", "-r", type=int, default=None, help="default: same as distance")
    ap.add_argument("--p", type=float, default=0.003, help="physical error rate")
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--basis", choices=["x", "z"], default="z")
    ap.add_argument("--no-uf", action="store_true", help="skip the (slower, pure Python) Union-Find decoder")
    ap.add_argument("--diagrams", action="store_true", help="write SVG diagrams to out/")
    args = ap.parse_args()

    rounds = args.rounds or args.distance
    circuit = memory_circuit(args.distance, args.p, rounds, args.basis)
    dem = detector_error_model(circuit)
    print(f"Rotated surface code d={args.distance}, {rounds} rounds, p={args.p}, basis={args.basis.upper()}")
    print(f"  qubits={circuit.num_qubits}  detectors={circuit.num_detectors}  "
          f"observables={circuit.num_observables}  error mechanisms={dem.num_errors}")

    if args.diagrams:
        out = Path("out")
        out.mkdir(exist_ok=True)
        (out / "detslice.svg").write_text(str(circuit.diagram("detslice-svg")), encoding="utf-8")
        (out / "matchgraph.svg").write_text(str(circuit.diagram("matchgraph-svg")), encoding="utf-8")
        (out / "timeline.svg").write_text(str(circuit.diagram("timeline-svg")), encoding="utf-8")
        print(f"  diagrams written to {out.resolve()}")

    sampler = circuit.compile_detector_sampler()
    dets, actual = sampler.sample(args.shots, separate_observables=True)
    print(f"  mean detection events per shot: {dets.sum(axis=1).mean():.2f}")

    def report(name: str, predicted: np.ndarray, seconds: float) -> None:
        errors = int(np.any(predicted != actual, axis=1).sum())
        rate = errors / args.shots
        per_round = 1 - (1 - rate) ** (1 / rounds)
        print(f"  {name:<12} logical errors {errors:>6}/{args.shots}  "
              f"rate={rate:.2e}  per-round={per_round:.2e}  {1e6 * seconds / args.shots:7.1f} us/shot")

    matcher = pymatching.Matching.from_detector_error_model(dem)
    t = time.perf_counter()
    report("MWPM", matcher.decode_batch(dets).astype(bool), time.perf_counter() - t)

    if not args.no_uf:
        uf = UnionFindDecoder(dem)
        t = time.perf_counter()
        report("Union-Find", uf.decode_batch(dets), time.perf_counter() - t)

    report("no decoding", np.zeros_like(actual), 0.0)


if __name__ == "__main__":
    main()
