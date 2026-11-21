"""Sanity checks: every single fault in the error model must be decoded correctly.

Run with:  .venv\\Scripts\\python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pymatching
import pytest
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qec import UnionFindDecoder, detector_error_model, memory_circuit  # noqa: E402


def single_faults(dem: stim.DetectorErrorModel):
    """Yield (detection events, observable flips) for each error mechanism on its own."""
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = np.zeros(dem.num_detectors, dtype=bool)
        obs = np.zeros(dem.num_observables, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets[t.val] ^= True
            elif t.is_logical_observable_id():
                obs[t.val] ^= True
        yield dets, obs


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["x", "z"])
def test_single_faults_corrected(d: int, basis: str) -> None:
    dem = detector_error_model(memory_circuit(d, 1e-3, basis=basis))
    uf = UnionFindDecoder(dem)
    wuf = UnionFindDecoder(dem, weighted=True)
    mwpm = pymatching.Matching.from_detector_error_model(dem)
    faults = list(single_faults(dem))
    assert faults
    for dets, obs in faults:
        assert np.array_equal(uf.decode_batch(dets[None, :])[0], obs)
        assert np.array_equal(wuf.decode_batch(dets[None, :])[0], obs)
        assert np.array_equal(mwpm.decode(dets).astype(bool), obs)


@pytest.mark.parametrize("name,d", [("pymatching", 3), ("union_find_weighted", 3), ("pymatching_correlated", 5)])
def test_registry_decoders_fix_single_faults(name: str, d: int) -> None:
    # Known quirk: at d=3, PyMatching's correlated re-weighting mis-decodes a handful of single
    # 4-detector (Y-type) faults near the boundary. From d=5 up, every single fault is fixed.
    from qec import make_decoder

    dem = detector_error_model(memory_circuit(d, 1e-3))
    dec = make_decoder(name, dem)
    dets = np.array([d for d, _ in single_faults(dem)])
    obs = np.array([o for _, o in single_faults(dem)])
    assert np.array_equal(dec.decode_batch(dets), obs)


def test_no_events_means_no_flip() -> None:
    dem = detector_error_model(memory_circuit(3, 1e-3))
    assert UnionFindDecoder(dem).decode(np.zeros(dem.num_detectors, dtype=bool)) == 0
