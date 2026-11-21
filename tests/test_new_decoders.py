"""Correctness checks for the decoders we wrote from scratch (qec/bp.py, matching.py, lookup.py,
the numba Union-Find kernel and the DemGraph they all share)."""
from __future__ import annotations

import numpy as np
import pymatching
import pytest

from qec import detector_error_model, make_decoder, memory_circuit
from qec.bp import MinSumBP
from qec.dem_graph import DemGraph, obs_mask_to_bits
from qec.metrics import fault_correction_rate, sample
from qec.union_find import UnionFindDecoder


@pytest.fixture(scope="module", params=[3, 5])
def setup(request):
    d = request.param
    c = memory_circuit(d, 5e-3)
    dem = detector_error_model(c)
    return d, c, dem, DemGraph.from_dem(dem)


def test_dem_graph_matches_pymatching(setup):
    """Our graph builder gives the same edges and weights as PyMatching's."""
    d, c, dem, g = setup
    m = pymatching.Matching.from_detector_error_model(dem)
    theirs = {}
    for u, v, a in m.edges():
        theirs[(u, g.boundary if v is None else v)] = a["weight"]
    ours = {(int(u), int(v)): w for u, v, w in zip(g.edge_u, g.edge_v, g.edge_weights)}
    norm = lambda k: (min(k), max(k))
    theirs = {norm(k): w for k, w in theirs.items()}
    ours = {norm(k): w for k, w in ours.items()}
    assert theirs.keys() == ours.keys()
    assert np.allclose([ours[k] for k in theirs], [theirs[k] for k in theirs], rtol=1e-6)


def test_our_mwpm_equals_pymatching(setup):
    """Exact MWPM is unique up to ties, so our pipeline must agree with PyMatching's."""
    d, c, dem, g = setup
    dets, obs = sample(c, 400 if d == 5 else 1500, seed=11)
    ours = make_decoder("mwpm", g).decode_batch(dets)
    theirs = pymatching.Matching.from_detector_error_model(dem).decode_batch(dets).astype(bool)
    assert (ours == theirs).all(axis=1).mean() >= 0.995


@pytest.mark.parametrize("name", ["union_find", "union_find_weighted", "mwpm", "sparse_blossom", "greedy_matching",
                                  "greedy_matching_2opt", "bp_osd", "bp_uf", "bp_uf_local", "lookup"])
def test_single_faults_corrected(setup, name):
    d, c, dem, g = setup
    if name == "lookup" and d > 3:
        pytest.skip("lookup table only built for d=3")
    r = fault_correction_rate(make_decoder(name, g), g, weight=1)
    assert r.errors == 0, f"{name} failed {r.errors}/{r.shots} single faults"


def test_numba_bp_matches_numpy(setup):
    d, c, dem, g = setup
    dets, _ = sample(c, 200, seed=4)
    a = MinSumBP(g.H, g.priors, backend="numba").run(dets)
    b = MinSumBP(g.H, g.priors, backend="numpy").run(dets)
    assert np.array_equal(a.converged, b.converged)
    assert np.array_equal(a.iterations, b.iterations)
    assert np.allclose(a.llr, b.llr, atol=1e-8)


@pytest.mark.parametrize("weighted", [False, True])
def test_numba_union_find_matches_python(setup, weighted):
    """The compiled edge-parallel kernel and the readable Python version are the same algorithm.
    They can break peeling ties differently (different spanning tree), so allow a tiny mismatch."""
    d, c, dem, g = setup
    dets, obs = sample(c, 1000, seed=6)
    uf = UnionFindDecoder(g, weighted=weighted)
    fast = uf.decode_batch(dets)
    slow = obs_mask_to_bits(np.array([uf.decode(r) for r in dets]), uf.num_observables)
    assert (fast == slow).all(axis=1).mean() >= 0.98
    fast_err = (fast != obs).any(axis=1).sum()
    slow_err = (slow != obs).any(axis=1).sum()
    assert abs(int(fast_err) - int(slow_err)) <= max(5, 0.3 * slow_err)


def test_bp_uf_uses_bp_when_converged():
    c = memory_circuit(3, 1e-3)
    dem = detector_error_model(c)
    dec = make_decoder("bp_uf", dem)
    dets, obs = sample(c, 500, seed=2)
    dec.decode_batch(dets)
    assert dec.last.converged.mean() > 0.9  # at low p, BP alone usually finds a valid answer


def test_sparse_blossom_matches_pymatching_weight(setup):
    """Sparse Blossom is exact MWPM: its total matching weight must equal PyMatching's on every shot
    (up to the 1/4096 weight rounding both implementations use)."""
    from qec.sparse_blossom import SparseBlossomDecoder

    d, c, dem, g = setup
    dets, _ = sample(c, 300 if d == 5 else 800, seed=21)
    sb = SparseBlossomDecoder(g)
    m = pymatching.Matching.from_detector_error_model(dem)
    for row in dets:
        _, w_pm = m.decode(row, return_weight=True)
        assert abs(sb.matching_weight(row) - w_pm) <= 2e-3 * max(w_pm, 1e-9)


def test_sparse_blossom_handles_blossoms():
    """High noise forces blossom formation, shattering and zero-radius events; weights must still match."""
    from qec.sparse_blossom import SparseBlossomDecoder

    c = memory_circuit(7, 1.2e-2)
    dem = detector_error_model(c)
    sb = SparseBlossomDecoder(dem)
    m = pymatching.Matching.from_detector_error_model(dem)
    dets, _ = sample(c, 150, seed=5)
    seen = {"blossom": 0, "shatter": 0, "zero_radius": 0}
    for row in dets:
        _, w_pm = m.decode(row, return_weight=True)
        assert abs(sb.matching_weight(row) - w_pm) <= 2e-3 * max(w_pm, 1e-9)
        for k in seen:
            seen[k] += sb.last_stats[k]
    assert seen["blossom"] > 0 and seen["zero_radius"] > 0


def test_local_bp_uf_close_to_global():
    """Local BP+UF restricts BP to the lit detectors' neighbourhood; on the same shots its predictions should
    almost always match global BP+UF."""
    from qec.bp import BPUnionFindDecoder
    from qec.bp_local import LocalBPUnionFindDecoder

    c = memory_circuit(7, 3e-3)
    dem = detector_error_model(c)
    dets, _ = sample(c, 2000, seed=9)
    glob = BPUnionFindDecoder(dem, max_iter=10).decode_batch(dets)
    for radius in (1, 2):
        loc = LocalBPUnionFindDecoder(dem, radius=radius, max_iter=10).decode_batch(dets)
        assert (loc == glob).all(axis=1).mean() >= 0.995
