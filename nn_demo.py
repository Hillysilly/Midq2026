"""Train the numpy MLP decoder and compare it with MWPM on fresh test data.

    python nn_demo.py                     # d=3, circuit noise p=0.5%
    python nn_demo.py -d 5 --train-shots 4000000 --hidden 512 256

An NN can learn correlations MWPM ignores (like Y errors), so at small d it can match or beat plain
MWPM. Watch how the training data needed grows with d. That scaling is the real challenge.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from qec import NoiseModel, SurfaceCode, detector_error_model, make_decoder
from qec.analysis import wilson
from qec.nn_decoder import MLPDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--distance", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--train-shots", type=int, default=2_000_000)
    ap.add_argument("--test-shots", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    args = ap.parse_args()

    circuit = SurfaceCode(args.distance).memory_circuit(NoiseModel.circuit(args.p), rounds=args.rounds)
    print(f"d={args.distance}, p={args.p}, {circuit.num_detectors} detectors; training MLP{tuple(args.hidden)} "
          f"on {args.train_shots:,} shots")
    t = time.perf_counter()
    nn = MLPDecoder.train_on_circuit(circuit, hidden=tuple(args.hidden), train_shots=args.train_shots, epochs=args.epochs)
    print(f"trained in {time.perf_counter() - t:.0f}s")

    dets, obs = circuit.compile_detector_sampler(seed=99).sample(args.test_shots, separate_observables=True)
    dem = detector_error_model(circuit)
    print(f"\ntest set: {args.test_shots:,} fresh shots")
    for name, dec in [("neural net (MLP)", nn), ("MWPM", make_decoder("pymatching", dem)),
                      ("MWPM correlated", make_decoder("pymatching_correlated", dem))]:
        t = time.perf_counter()
        fails = int(np.any(dec.decode_batch(dets) != obs, axis=1).sum())
        us = 1e6 * (time.perf_counter() - t) / args.test_shots
        lo, hi = wilson(fails, args.test_shots)
        print(f"  {name:18s} logical error {fails / args.test_shots:.4f}  [{lo:.4f}, {hi:.4f}]   {us:.1f} us/shot")


if __name__ == "__main__":
    main()
