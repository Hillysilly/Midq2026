"""Test functions that *calculate decoder metrics* for every decoder we wrote.

Each test computes a metric (logical error rate, fault-tolerance, threshold, latency, parallel
depth), records it with `metrics_log` (printed as a table at the end of `pytest`, and saved to
out/metrics/test_metrics.json), and asserts what it must satisfy.

Run:  .venv\\Scripts\\python -m pytest tests/test_metrics.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qec import NoiseModel, SurfaceCode, detector_error_model, make_decoder, memory_circuit
from qec.metrics import (Rate, fault_correction_rate, keeps_up, lambda_factor, latency,
                         logical_error_rate, min_failing_weight, parallel_depth, sample, threshold)

# Our decoders, minus the ones that are deliberately weak (bp alone, greedy) where noted.
DISTANCE_PRESERVING = ["union_find", "union_find_weighted", "mwpm", "sparse_blossom", "bp_osd", "bp_uf", "bp_uf_local"]
ALL_OURS = DISTANCE_PRESERVING + ["greedy_matching", "greedy_matching_2opt", "bp", "lookup"]

P_CIRCUIT = 2e-3


def _circuit(d: int, p: float = P_CIRCUIT):
    c = memory_circuit(d, p)
    return c, detector_error_model(c)


# --------------------------------------------------------------------------------------------- #
# 1. logical error rate and its suppression with distance (circuit-level noise)
# --------------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", DISTANCE_PRESERVING)
def test_logical_error_rate_drops_with_distance(name, metrics_log):
    rates = {}
    for d, shots in ((3, 6000), (5, 6000)):
        c, dem = _circuit(d)
        dets, obs = sample(c, shots, seed=100 + d)
        r = logical_error_rate(make_decoder(name, dem), dets, obs)
        rates[d] = r.per_round(d)
        metrics_log(metric=f"logical error rate (circuit p={P_CIRCUIT})", decoder=name, d=d,
                    per_shot=r.rate, per_round=rates[d], errors=r.errors, shots=r.shots)
    lam = lambda_factor(rates)
    metrics_log(metric="Lambda = eps_3/eps_5", decoder=name, Lambda=lam.get(3, float("nan")))
    assert rates[5] < rates[3], f"{name}: no suppression from d=3 to d=5 at p={P_CIRCUIT}"


@pytest.mark.parametrize("name", ["bp", "greedy_matching", "greedy_matching_2opt", "lookup"])
def test_logical_error_rate_weak_decoders(name, metrics_log):
    """Recorded for comparison; the only requirement is that they beat doing nothing."""
    c, dem = _circuit(3)
    dets, obs = sample(c, 6000, seed=103)
    r = logical_error_rate(make_decoder(name, dem), dets, obs)
    raw = Rate(int(obs.any(axis=1).sum()), len(obs))  # "decoder" that never corrects
    metrics_log(metric=f"logical error rate (circuit p={P_CIRCUIT})", decoder=name, d=3,
                per_shot=r.rate, per_round=r.per_round(3), errors=r.errors, shots=r.shots)
    assert r.rate < raw.rate


# --------------------------------------------------------------------------------------------- #
# 2. fault tolerance: which fault weights get corrected
# --------------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", DISTANCE_PRESERVING + ["greedy_matching", "greedy_matching_2opt"])
def test_fault_weight_correction_d5(name, metrics_log):
    """At d=5 a distance-preserving decoder corrects every combination of 1 or 2 faults.

    Two documented exceptions:
    * greedy matching is *not* distance-preserving (it strands defects; ~2% of 2-fault errors fail);
    * BP-based decoders (bp_osd, bp_uf) fail a few 2-fault errors (~0.1-0.3%) when BP oscillates
      and its posteriors rank the true fault columns hundreds of places down. They still have the
      lowest overall logical error rates of our decoders: distance-preserving != most accurate.
    """
    c, dem = _circuit(5, 1e-3)
    dec = make_decoder(name, dem)
    w1 = fault_correction_rate(dec, dem, weight=1)
    w2 = fault_correction_rate(dec, dem, weight=2, max_samples=1500, seed=1)
    w3 = fault_correction_rate(dec, dem, weight=3, max_samples=1500, seed=1)
    metrics_log(metric="fault-weight correction, d=5 (failure fraction)", decoder=name,
                weight1=w1.rate, weight2=w2.rate, weight3=w3.rate)
    assert w1.errors == 0
    if name.startswith("greedy"):
        assert w2.errors > 0  # the finding: greedy loses distance
    elif name.startswith("bp"):
        assert w2.rate <= 0.005
    else:
        assert w2.errors == 0


@pytest.mark.parametrize("name", ["union_find_weighted", "mwpm", "sparse_blossom", "greedy_matching"])
def test_min_failing_weight(name, metrics_log):
    c, dem = _circuit(5, 1e-3)
    w = min_failing_weight(make_decoder(name, dem), dem, max_weight=3, samples_per_weight=1500)
    metrics_log(metric="smallest failing fault weight, d=5", decoder=name, min_failing_weight=w)
    assert w is not None and w >= (2 if name.startswith("greedy") else 3)


# --------------------------------------------------------------------------------------------- #
# 3. threshold (code-capacity noise: cheap, and comparable with the review's Fig. 27)
# --------------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,lo,hi", [
    ("union_find", 0.09, 0.17),
    ("union_find_weighted", 0.09, 0.17),
    ("mwpm", 0.11, 0.17),
    ("sparse_blossom", 0.11, 0.17),
    ("bp_osd", 0.11, 0.20),
    ("bp_uf", 0.11, 0.20),
    ("bp_uf_local", 0.11, 0.20),
])
def test_code_capacity_threshold(name, lo, hi, metrics_log):
    ps = [0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
    rates: dict[int, dict[float, float]] = {}
    for d in (3, 5):
        code = SurfaceCode(d)
        rates[d] = {}
        for p in ps:
            c = code.memory_circuit_ideal(NoiseModel.code_capacity(p))
            dem = detector_error_model(c)
            dets, obs = sample(c, 1500 if name in ("mwpm", "sparse_blossom") else 3000, seed=int(p * 1000) + d)
            rates[d][p] = logical_error_rate(make_decoder(name, dem), dets, obs).rate
    th = threshold(rates)
    metrics_log(metric="code-capacity threshold (d3/d5 crossing)", decoder=name,
                threshold=th if th is not None else float("nan"))
    assert th is not None and lo <= th <= hi


# --------------------------------------------------------------------------------------------- #
# 4. speed
# --------------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL_OURS + ["pymatching"])
def test_latency(name, metrics_log):
    d = 3 if name == "lookup" else 5
    c, dem = _circuit(d, 1e-3)
    dets, _ = sample(c, 300 if name in ("mwpm", "sparse_blossom") else 2000, seed=7)
    t = latency(make_decoder(name, dem), dets, single_shots=100)
    metrics_log(metric="decode time (circuit p=0.1%, d rounds)", decoder=name, d=d,
                batch_us_per_shot=t.batch_us_per_shot, us_per_round=t.batch_us_per_shot / d,
                single_p50_us=t.single_us_p50, keeps_up_1us=keeps_up(t, d))
    assert math.isfinite(t.batch_us_per_shot) and t.batch_us_per_shot > 0


# --------------------------------------------------------------------------------------------- #
# 5. parallelism
# --------------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["union_find", "union_find_weighted", "bp", "bp_uf", "bp_uf_local"])
def test_parallel_depth_scaling(name, metrics_log):
    """Sequential steps per shot if every independent update runs simultaneously. For the
    decoders to be useful in hardware, this must grow much slower than the code size (~d^3)."""
    depth = {}
    for d in (3, 5, 7):
        c, dem = _circuit(d, 3e-3)
        dets, _ = sample(c, 1000, seed=8)
        depth[d] = parallel_depth(make_decoder(name, dem), dets)
        metrics_log(metric="parallel depth (circuit p=0.3%)", decoder=name, d=d, **depth[d])
    if name.startswith("bp"):
        # BP converges less often on bigger codes, so its *mean* depth grows, but the worst case
        # is fixed by max_iter (30): constant-time in hardware, at the cost of accuracy.
        assert depth[7]["max"] <= 30 + (2 * 7 * 7 if name.startswith("bp_uf") else 0)
    else:
        growth = depth[7]["mean"] / max(depth[3]["mean"], 1e-9)
        volume_growth = 7 ** 3 / 3 ** 3  # detectors grow like d^3
        assert growth < volume_growth / 3


# --------------------------------------------------------------------------------------------- #
# 6. the metric functions themselves
# --------------------------------------------------------------------------------------------- #

def test_metric_math():
    r = Rate(5, 1000)
    lo, hi = r.ci
    assert lo < 0.005 < hi
    assert math.isclose(r.per_round(1), 0.005)
    assert r.per_round(10) < 0.005 / 9  # per-round rate is ~1/rounds of the per-shot rate
    # synthetic curves crossing at p = 0.1: rate_d = (p / 0.1) ** ((d + 1) / 2) * 0.1
    curves = {d: {p: 0.1 * (p / 0.1) ** ((d + 1) / 2) for p in (0.05, 0.08, 0.12, 0.15)} for d in (3, 5)}
    assert math.isclose(threshold(curves), 0.1, rel_tol=1e-6)
    assert lambda_factor({3: 1e-2, 5: 5e-3, 7: 2.5e-3}) == {3: 2.0, 5: 2.0}
