"""Decoder metrics: everything you'd want to know about a decoder, as plain functions.

Every function takes a decoder made by `make_decoder(name, dem)` (anything with
`decode_batch(dets) -> predicted observable flips`). `tests/test_metrics.py` calls them for each
of our decoders, and `metrics_report.py` turns them into a table and plots.

Accuracy
    logical_error_rate        fraction of shots where the decoder's prediction is wrong (+ Wilson CI)
    per_round                 the same, normalised to one round of syndrome extraction
    fault_correction_rate     fraction of all weight-t fault combinations corrected
                              (t <= (d-1)/2 must give 100% for a distance-preserving decoder)
    min_failing_weight        smallest number of faults the decoder fails on (sampled):
                              the decoder's *effective distance* is 2x this minus 1 (or so)
    threshold                 physical error rate where curves for two distances cross
    lambda_factor             error suppression eps_d / eps_{d+2} (Google reports ~2.1)

Speed
    latency                   wall-clock per shot: batched throughput and single-shot p50/p99
    keeps_up                  does throughput beat the syndrome cycle (1 us on superconducting chips)?

Parallelism
    parallel_depth            the decoder's sequential step count per shot: growth rounds for
                              Union-Find, iterations for BP, 1 for a lookup table. If each step
                              runs in O(1) on parallel hardware, this is the hardware latency in
                              "clock cycles", and how it grows with d tells you how it scales.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import stim

from .analysis import per_round as _per_round
from .analysis import wilson
from .dem_graph import DemGraph


# --------------------------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------------------------- #

@dataclass
class Rate:
    errors: int
    shots: int

    @property
    def rate(self) -> float:
        return self.errors / self.shots if self.shots else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.errors, self.shots)

    def per_round(self, rounds: int) -> float:
        return _per_round(self.rate, rounds)

    def __str__(self) -> str:
        lo, hi = self.ci
        return f"{self.rate:.2e} [{lo:.1e}, {hi:.1e}] ({self.errors}/{self.shots})"


def sample(circuit: stim.Circuit, shots: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Detection events and true observable flips for `shots` noisy runs of `circuit`."""
    return circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)


def logical_error_rate(decoder, dets: np.ndarray, obs: np.ndarray) -> Rate:
    """A shot fails if the prediction differs from the truth on any logical observable."""
    pred = np.asarray(decoder.decode_batch(dets), dtype=bool)
    return Rate(int((pred != obs).any(axis=1).sum()), len(obs))


def per_round(p_shot: float, rounds: int) -> float:
    return _per_round(p_shot, rounds)


def _fault_table(dem: stim.DetectorErrorModel | DemGraph):
    g = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
    H = g.H.toarray().astype(bool)   # (detectors, mechanisms)
    L = g.L.toarray().astype(bool)   # (observables, mechanisms)
    return H, L


def fault_correction_rate(decoder, dem: stim.DetectorErrorModel | DemGraph, weight: int = 1,
                          max_samples: int = 2000, seed: int = 0) -> Rate:
    """Inject exactly `weight` distinct fault mechanisms, decode, count corrections.

    weight 1 enumerates every mechanism; higher weights draw `max_samples` random combinations.
    Returns a Rate whose `errors` are the *failures*.
    """
    H, L = _fault_table(dem)
    n = H.shape[1]
    if weight == 1:
        combos = np.arange(n)[:, None]
    else:
        rng = np.random.default_rng(seed)
        combos = np.array([rng.choice(n, weight, replace=False) for _ in range(max_samples)])
    dets = np.zeros((len(combos), H.shape[0]), dtype=bool)
    obs = np.zeros((len(combos), L.shape[0]), dtype=bool)
    for k in range(combos.shape[1]):
        dets ^= H[:, combos[:, k]].T
        obs ^= L[:, combos[:, k]].T
    return logical_error_rate(decoder, dets, obs)


def min_failing_weight(decoder, dem: stim.DetectorErrorModel | DemGraph, max_weight: int = 3,
                       samples_per_weight: int = 2000, seed: int = 0) -> int | None:
    """Smallest fault weight with at least one (sampled) failure, or None if none up to max_weight.

    A distance-d decoder must correct every combination of up to (d-1)/2 faults.
    """
    for w in range(1, max_weight + 1):
        if fault_correction_rate(decoder, dem, w, samples_per_weight, seed).errors > 0:
            return w
    return None


def threshold(rates: dict[int, dict[float, float]]) -> float | None:
    """Crossing point of the two smallest distances' curves, by log-log interpolation.

    `rates[d][p]` = logical error rate (use per-round rates for circuit-level noise).
    Returns None if the curves don't cross inside the sampled range.
    """
    ds = sorted(rates)
    if len(ds) < 2:
        return None
    a, b = rates[ds[0]], rates[ds[1]]
    ps = sorted(p for p in set(a) & set(b) if a[p] > 0 and b[p] > 0)
    diffs = [(p, math.log(b[p]) - math.log(a[p])) for p in ps]
    for (p1, d1), (p2, d2) in zip(diffs, diffs[1:]):
        if d1 < 0 <= d2:
            f = -d1 / (d2 - d1)
            return math.exp(math.log(p1) + f * (math.log(p2) - math.log(p1)))
    return None


def lambda_factor(rates_by_d: dict[int, float]) -> dict[int, float]:
    """Lambda_d = eps_d / eps_{d+2}. Above 1 means a bigger code helps; ~2 means each +2 in
    distance halves the logical error rate."""
    return {d: rates_by_d[d] / rates_by_d[d + 2] for d in sorted(rates_by_d)
            if d + 2 in rates_by_d and rates_by_d[d + 2] > 0}


# --------------------------------------------------------------------------------------------- #
# speed
# --------------------------------------------------------------------------------------------- #

@dataclass
class Timing:
    batch_us_per_shot: float     # throughput: decode everything in one call
    single_us_p50: float         # latency: one shot per call (includes Python call overhead)
    single_us_p99: float

    def as_dict(self) -> dict:
        return asdict(self)


def latency(decoder, dets: np.ndarray, single_shots: int = 200, warmup: int = 20) -> Timing:
    decoder.decode_batch(dets[:warmup])  # JIT compile / cache warm-up
    t = time.perf_counter()
    decoder.decode_batch(dets)
    batch = (time.perf_counter() - t) / len(dets) * 1e6
    singles = []
    for row in dets[:single_shots]:
        t = time.perf_counter()
        decoder.decode_batch(row[None, :])
        singles.append((time.perf_counter() - t) * 1e6)
    return Timing(batch, float(np.percentile(singles, 50)), float(np.percentile(singles, 99)))


def keeps_up(timing: Timing, rounds_per_shot: int, cycle_us: float = 1.0) -> bool:
    """Throughput test: can the decoder process syndrome data as fast as the chip produces it?"""
    return timing.batch_us_per_shot / rounds_per_shot <= cycle_us


# --------------------------------------------------------------------------------------------- #
# parallelism
# --------------------------------------------------------------------------------------------- #

def parallel_depth(decoder, dets: np.ndarray) -> dict | None:
    """Sequential steps per shot when every independent update runs at once.

    * Union-Find: growth rounds (all edges grow simultaneously each round).
    * BP: iterations (all checks, then all variables, update simultaneously).
    * BP+UF / BP+OSD: BP iterations plus the second stage's rounds (OSD reported as rank, since
      Gaussian elimination is sequential in the number of pivots).
    * Lookup table: 1 (one memory read).
    * Matching (blossom / greedy): None, since there's no fixed parallel round structure.
    Returns {"mean", "p99", "max"} over the shots, or None.
    """
    from .bp import BPDecoder, BPOSDDecoder, BPUnionFindDecoder
    from .bp_local import LocalBPUnionFindDecoder
    from .lookup import LookupTableDecoder
    from .union_find import UnionFindDecoder

    dets = np.atleast_2d(dets)
    if isinstance(decoder, UnionFindDecoder):
        decoder.decode_batch(dets)
        steps = decoder.last_batch_rounds.astype(float)
    elif isinstance(decoder, BPUnionFindDecoder):
        res = decoder.bp.run(dets)
        steps = res.iterations.astype(float)
        todo = np.flatnonzero(~res.converged)
        if len(todo):
            decoder.decode_batch(dets[todo])
            steps[todo] += decoder.uf.last_batch_rounds
    elif isinstance(decoder, LocalBPUnionFindDecoder):
        decoder.decode_batch(dets)
        steps = (decoder.last_iterations + decoder.last_uf_rounds).astype(float)
    elif isinstance(decoder, BPOSDDecoder):
        res = decoder.bp.run(dets)
        steps = res.iterations.astype(float) + np.where(res.converged, 0, decoder.osd.rank)
    elif isinstance(decoder, BPDecoder):
        steps = decoder.bp.run(dets).iterations.astype(float)
    elif isinstance(decoder, LookupTableDecoder):
        steps = np.ones(len(dets))
    else:
        return None
    return {"mean": float(steps.mean()), "p99": float(np.percentile(steps, 99)), "max": float(steps.max())}


# --------------------------------------------------------------------------------------------- #
# everything at once
# --------------------------------------------------------------------------------------------- #

@dataclass
class DecoderReport:
    decoder: str
    d: int
    p: float
    rounds: int
    shots: int
    errors: int
    ler_per_shot: float
    ler_ci: tuple[float, float]
    ler_per_round: float
    single_fault_failures: int
    single_faults: int
    batch_us_per_shot: float
    single_us_p50: float
    parallel_depth_mean: float | None
    parallel_depth_max: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(name: str, circuit: stim.Circuit, d: int, p: float, rounds: int, shots: int = 5000,
             seed: int = 0, dem: stim.DetectorErrorModel | None = None) -> DecoderReport:
    """Run every metric for decoder `name` on `circuit`."""
    from .circuits import detector_error_model
    from .decoders import make_decoder

    dem = dem if dem is not None else detector_error_model(circuit)
    dec = make_decoder(name, dem)
    dets, obs = sample(circuit, shots, seed)
    t = latency(dec, dets, single_shots=min(200, shots))
    r = logical_error_rate(dec, dets, obs)
    sf = fault_correction_rate(dec, dem, 1)
    depth = parallel_depth(dec, dets[: min(shots, 2000)])
    return DecoderReport(
        name, d, p, rounds, r.shots, r.errors, r.rate, r.ci, r.per_round(rounds),
        sf.errors, sf.shots, t.batch_us_per_shot, t.single_us_p50,
        depth["mean"] if depth else None, depth["max"] if depth else None)
