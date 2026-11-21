"""All the decoders in one place, through two interfaces.

1. `make_decoder(name, dem)` returns an object with
   `decode_batch(dets: bool[shots, num_detectors]) -> bool[shots, num_observables]`.
   Use it in your own scripts and notebooks.
2. `SINTER_DECODERS` maps the same names to Sinter decoders, for parallel threshold sweeps.

| name                 | algorithm (review section)                         | speed     |
|----------------------|----------------------------------------------------|-----------|
| pymatching           | Minimum-weight perfect matching, Sparse Blossom §5.1 | fast (C++) |
| pymatching_correlated| Two-pass MWPM that uses the X/Z correlation of Y errors (§6) | fast (C++) |
| union_find           | Union-Find, unweighted §5.2                         | slow (Python) |
| union_find_weighted  | Union-Find, weighted growth                         | slow (Python) |
| bposd                | Belief propagation + ordered statistics §5.3        | slow      |
| belief_matching      | BP to re-weight the graph, then MWPM (§5.3, §6)     | slow      |
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pymatching
import sinter
import stim

from .union_find import UnionFindDecoder, UnionFindSinterDecoder


class _Matching:
    def __init__(self, dem: stim.DetectorErrorModel, correlated: bool = False):
        self.correlated = correlated
        self.m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        return self.m.decode_batch(dets, enable_correlations=self.correlated).astype(bool)


class CorrelatedMatchingSinterDecoder(sinter.Decoder):
    """PyMatching with `enable_correlations=True`: two-pass matching that uses the X/Z correlation
    of Y errors (review §6, eq. 22). About 2-3x slower, and noticeably more accurate.

    Known quirk: at d=3 a few single Y-type faults are mis-decoded, so the circuit-level fault
    distance is effectively below 3. Use d >= 5 for fair comparisons."""

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _CompiledCorrelated(dem)


class _CompiledCorrelated(sinter.CompiledDecoder):
    def __init__(self, dem: stim.DetectorErrorModel):
        self.m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        return self.m.decode_batch(bit_packed_detection_event_data, bit_packed_shots=True,
                                   bit_packed_predictions=True, enable_correlations=True)


def _bposd(dem: stim.DetectorErrorModel):
    import stimbposd
    return _Wrap(stimbposd.BPOSD(dem, max_bp_iters=20, osd_order=0))


def _belief_matching(dem: stim.DetectorErrorModel):
    import beliefmatching
    return _Wrap(beliefmatching.BeliefMatching(dem, max_bp_iters=20))


class _Wrap:
    def __init__(self, inner):
        self.inner = inner

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.decode_batch(dets)).astype(bool)


_FACTORIES: dict[str, Callable[[stim.DetectorErrorModel], object]] = {
    "pymatching": _Matching,
    "pymatching_correlated": lambda dem: _Matching(dem, correlated=True),
    "union_find": lambda dem: UnionFindDecoder(dem),
    "union_find_weighted": lambda dem: UnionFindDecoder(dem, weighted=True),
    "bposd": _bposd,
    "belief_matching": _belief_matching,
}

DECODER_NAMES = list(_FACTORIES)


def make_decoder(name: str, dem: stim.DetectorErrorModel):
    if name not in _FACTORIES:
        raise KeyError(f"unknown decoder {name!r}; choose from {DECODER_NAMES}")
    return _FACTORIES[name](dem)


def sinter_decoders() -> dict:
    """Custom decoders for `sinter.collect(custom_decoders=...)`; 'pymatching' is built into Sinter."""
    out: dict = {
        "pymatching_correlated": CorrelatedMatchingSinterDecoder(),
        "union_find": UnionFindSinterDecoder(),
        "union_find_weighted": UnionFindSinterDecoder(weighted=True),
    }
    try:
        import stimbposd
        out["bposd"] = stimbposd.SinterDecoder_BPOSD(max_bp_iters=20, osd_order=0)
    except ImportError:
        pass
    try:
        import beliefmatching
        out["belief_matching"] = beliefmatching.BeliefMatchingSinterDecoder()
    except ImportError:
        pass
    return out
