"""Resource estimator: how big must the surface code be?

Fits eps(d, p) = A * (p / p_th) ** ((d + 1) / 2) to benchmark data, then projects the
distance and qubit count needed to hit a target logical error rate.

    python estimate.py --stats out/circuit/stats.csv --p 0.001 --target 1e-12 --logical-qubits 100

Without --stats it falls back to the textbook rule of thumb A = 0.1, p_th = 1%
(Fowler et al. 2012).

The usual target is total failure probability below about 1 / (number of logical-qubit-rounds in
the algorithm). For example, 100 logical qubits x 1e9 rounds needs about 1e-11 to 1e-12 per
qubit-round.
"""
from __future__ import annotations

import argparse

import sinter

from qec import analysis


def qubits_per_patch(d: int) -> int:
    """Rotated surface code: d^2 data + d^2 - 1 ancilla qubits."""
    return 2 * d * d - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", help="stats.csv written by bench.py / threshold.py")
    ap.add_argument("--series", help="which series in the stats to fit (default: the first one)")
    ap.add_argument("--p", type=float, nargs="+", default=[1e-3, 2e-3, 5e-3])
    ap.add_argument("--target", type=float, default=1e-12, help="logical error rate per qubit per round")
    ap.add_argument("--logical-qubits", type=int, default=100)
    ap.add_argument("--routing-overhead", type=float, default=2.0,
                    help="extra patches for lattice-surgery routing / magic-state factories (x)")
    args = ap.parse_args()

    if args.stats:
        stats = sinter.read_stats_from_csv_files(args.stats)
        points = analysis.aggregate(stats)
        names = sorted({pt.series for pt in points})
        series = args.series or names[0]
        fit = analysis.fit_scaling(points, series)
        if fit is None:
            raise SystemExit(f"not enough data to fit {series!r}; available: {names}")
        A, p_th = fit
        print(f"Fitted '{series}': eps(d,p) = {A:.3g} * (p / {p_th:.4g}) ** ((d+1)/2)")
        lam = analysis.lambda_factor(points, min(pt.p for pt in points if pt.series == series))
        if lam.get(series):
            print("Measured Lambda = eps_d / eps_{d+2}: " + ", ".join(f"d={d}: {v:.2f}" for d, v in lam[series]))
    else:
        A, p_th = 0.1, 0.01
        print(f"Rule of thumb: eps(d,p) = {A} * (p / {p_th}) ** ((d+1)/2)")

    print(f"\nTarget: {args.target:g} logical errors per qubit-round, {args.logical_qubits} logical qubits, "
          f"x{args.routing_overhead:g} routing/factory overhead\n")
    print(f"{'p':>8} {'Lambda':>8} {'d':>4} {'qubits/patch':>13} {'total physical qubits':>22}")
    for p in args.p:
        lam = (p_th / p) if p < p_th else float("nan")
        d = analysis.project(A, p_th, p, args.target)
        if d is None:
            print(f"{p:>8.4g} {lam:>8.2f}  -- not reachable (p too close to or above threshold)")
            continue
        per = qubits_per_patch(d)
        total = int(per * args.logical_qubits * args.routing_overhead)
        print(f"{p:>8.4g} {lam:>8.2f} {d:>4} {per:>13,} {total:>22,}")
    print("\nLambda = p_th / p is how much each +2 in distance suppresses errors. Halving p doubles it.")


if __name__ == "__main__":
    main()
