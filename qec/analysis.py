"""Turn raw Sinter statistics into the numbers people care about.

* logical error rate per shot and per round, with confidence intervals
* threshold estimates (where curves for consecutive distances cross)
* the error-suppression factor Lambda = eps_d / eps_{d+2} (Google's headline metric)
* projections: what distance / how many qubits for a target logical error rate
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    """One aggregated data point: a (series, distance, p) combination."""

    series: str
    d: int
    p: float
    shots: int
    errors: int
    rounds: int
    rate: float  # failure probability per shot (summed over bases if several)
    lo: float
    hi: float

    @property
    def per_round(self) -> float:
        return per_round(self.rate, self.rounds)

    @property
    def per_round_lo(self) -> float:
        return per_round(self.lo, self.rounds)

    @property
    def per_round_hi(self) -> float:
        return per_round(self.hi, self.rounds)


def per_round(p_shot: float, rounds: int) -> float:
    """Convert a per-shot failure probability into a per-round one."""
    p_shot = min(max(p_shot, 0.0), 0.5)
    return 0.5 * (1 - (1 - 2 * p_shot) ** (1 / max(rounds, 1)))


def wilson(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: an error bar that behaves well even with 0 errors."""
    if shots == 0:
        return 0.0, 1.0
    ph = errors / shots
    denom = 1 + z * z / shots
    centre = (ph + z * z / (2 * shots)) / denom
    half = z * math.sqrt(ph * (1 - ph) / shots + z * z / (4 * shots * shots)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def aggregate(stats, series_keys=("variant", "decoder", "noise", "bias")) -> list[Point]:
    """Group Sinter TaskStats by series, distance and p; sum the error rates over memory bases.

    Memory experiments in the Z and X bases each catch one kind of logical error. Their sum
    approximates the chance that *either* logical observable fails.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for s in stats:
        md = dict(s.json_metadata)
        md["decoder"] = s.decoder
        series = " ".join(f"{md[k]}" if k in ("variant", "decoder") else f"{k}={md[k]}"
                          for k in series_keys if k in md)
        groups[(series, md["d"], md["p"])].append((s, md))
    points = []
    for (series, d, p), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rate = lo = hi = 0.0
        shots = errors = 0
        rounds = items[0][1].get("r", 1)
        for s, _ in items:
            e = s.errors
            n = s.shots - s.discards
            if n == 0:
                continue
            lo_i, hi_i = wilson(e, n)
            rate += e / n
            lo += lo_i
            hi += hi_i
            shots += n
            errors += e
        points.append(Point(series, d, p, shots, errors, rounds, rate, lo, min(hi, 1.0)))
    return points


def by_series(points: list[Point]) -> dict[str, dict[int, list[Point]]]:
    out: dict[str, dict[int, list[Point]]] = defaultdict(lambda: defaultdict(list))
    for pt in points:
        out[pt.series][pt.d].append(pt)
    for s in out.values():
        for lst in s.values():
            lst.sort(key=lambda x: x.p)
    return out


def crossing(a: list[Point], b: list[Point], per_round_rates: bool = True) -> float | None:
    """p where curves a (smaller d) and b (larger d) cross, by log-log linear interpolation."""
    pa = {pt.p: pt for pt in a if pt.errors > 0}
    pb = {pt.p: pt for pt in b if pt.errors > 0}
    ps = sorted(set(pa) & set(pb))
    val = (lambda pt: pt.per_round) if per_round_rates else (lambda pt: pt.rate)
    diffs = [(p, math.log(val(pb[p])) - math.log(val(pa[p]))) for p in ps]
    for (p1, d1), (p2, d2) in zip(diffs, diffs[1:]):
        if d1 < 0 <= d2:
            f = -d1 / (d2 - d1)
            return math.exp(math.log(p1) + f * (math.log(p2) - math.log(p1)))
    return None


def thresholds(points: list[Point], per_round_rates: bool = True) -> dict[str, list[tuple[int, int, float | None]]]:
    """For each series: (d_small, d_large, crossing p) for consecutive distance pairs."""
    out = {}
    for series, curves in by_series(points).items():
        ds = sorted(curves)
        out[series] = [(d1, d2, crossing(curves[d1], curves[d2], per_round_rates)) for d1, d2 in zip(ds, ds[1:])]
    return out


def lambda_factor(points: list[Point], p: float) -> dict[str, list[tuple[int, float]]]:
    """Error suppression Lambda_d = eps_d / eps_{d+2} per series at physical error rate p."""
    out = {}
    for series, curves in by_series(points).items():
        eps = {d: next((pt.per_round for pt in lst if math.isclose(pt.p, p) and pt.errors > 0), None)
               for d, lst in curves.items()}
        ds = sorted(d for d, e in eps.items() if e)
        out[series] = [(d, eps[d] / eps[d + 2]) for d in ds if d + 2 in eps and eps.get(d + 2)]
    return out


def fit_scaling(points: list[Point], series: str) -> tuple[float, float] | None:
    """Fit eps(d, p) = A * (p / p_th) ** ((d + 1) / 2) to the below-threshold points.

    Returns (A, p_th) for use in `project`. Points with fewer than 5 errors are ignored.
    """
    rows = [(pt.d, pt.p, pt.per_round) for pt in points if pt.series == series and pt.errors >= 5]
    if len(rows) < 3:
        return None
    # log eps = log A + (d+1)/2 * (log p - log p_th)  -> linear least squares in (log A, log p_th)
    X = np.array([[1.0, -(d + 1) / 2] for d, p, _ in rows])
    y = np.array([math.log(e) - (d + 1) / 2 * math.log(p) for d, p, e in rows])
    (logA, logpth), *_ = np.linalg.lstsq(X, y, rcond=None)
    return math.exp(logA), math.exp(logpth)


def project(A: float, p_th: float, p: float, target: float, max_d: int = 101) -> int | None:
    """Smallest odd distance with per-round logical error rate below `target` at physical rate p."""
    for d in range(3, max_d + 1, 2):
        if A * (p / p_th) ** ((d + 1) / 2) <= target:
            return d
    return None
