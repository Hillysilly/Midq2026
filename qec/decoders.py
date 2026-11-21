"""All the decoders in one place, through two interfaces.

1. `make_decoder(name, dem)` returns an object with
   `decode_batch(dets: bool[shots, num_detectors]) -> bool[shots, num_observables]`.
   Use it in your own scripts and notebooks.
2. `sinter_decoders()` maps the same names to Sinter decoders, for parallel threshold sweeps.

**Written by us, from scratch** (no decoding library inside; see `OURS`):

| name                 | algorithm (review section)                                   | file |
|----------------------|--------------------------------------------------------------|------|
| union_find           | Union-Find, unweighted §5.2 (numba, edge-parallel growth)    | union_find.py |
| union_find_weighted  | Union-Find with weighted growth                              | union_find.py |
| mwpm                 | MWPM: Dijkstra + boundary copies + blossom (networkx) §5.1   | matching.py |
| sparse_blossom       | Exact MWPM grown on the detector graph (PyMatching's algorithm) | sparse_blossom.py |
| greedy_matching      | Greedy matching by savings                                   | matching.py |
| greedy_matching_2opt | Greedy matching + 2-opt local search                         | matching.py |
| lookup               | Lookup table, most-likely error up to weight 2 (small codes) | lookup.py |
| bp                   | Min-sum belief propagation §5.3 (numba)                      | bp.py |
| bp_osd               | BP + our own OSD-0 over GF(2) §5.3                            | bp.py |
| bp_uf                | BP posteriors as per-shot weights for Union-Find ("belief-find") | bp.py |

**Library baselines** (to compare against):

| name                 | algorithm                                                   |
|----------------------|-------------------------------------------------------------|
| pymatching           | PyMatching Sparse Blossom MWPM (C++)                        |
| pymatching_correlated| PyMatching two-pass correlated MWPM (§6)                    |
| bposd                | stimbposd / ldpc BP+OSD                                     |
| belief_matching      | beliefmatching (BP re-weighting + PyMatching)               |
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pymatching
import sinter
import stim

from .bp import BPDecoder, BPOSDDecoder, BPUnionFindDecoder
from .lookup import LookupTableDecoder
from .matching import GreedyMatchingDecoder, MWPMDecoder
from .sparse_blossom import SparseBlossomDecoder
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
    "mwpm": MWPMDecoder,
    "sparse_blossom": SparseBlossomDecoder,
    "greedy_matching": GreedyMatchingDecoder,
    "greedy_matching_2opt": lambda dem: GreedyMatchingDecoder(dem, local_search=True),
    "lookup": LookupTableDecoder,
    "bp": BPDecoder,
    "bp_osd": BPOSDDecoder,
    "bp_uf": BPUnionFindDecoder,
    "bposd": _bposd,
    "belief_matching": _belief_matching,
}

DECODER_NAMES = list(_FACTORIES)
OURS = ["union_find", "union_find_weighted", "mwpm", "sparse_blossom", "greedy_matching", "greedy_matching_2opt",
        "lookup", "bp", "bp_osd", "bp_uf"]
BASELINES = ["pymatching", "pymatching_correlated", "bposd", "belief_matching"]


class _CompiledGeneric(sinter.CompiledDecoder):
    def __init__(self, dec, num_detectors: int):
        self.dec = dec
        self.num_detectors = num_detectors

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little",
                             count=self.num_detectors).astype(bool)
        return np.packbits(self.dec.decode_batch(dets), axis=1, bitorder="little")


class RegistrySinterDecoder(sinter.Decoder):
    """Any `make_decoder` name as a Sinter decoder."""

    def __init__(self, name: str):
        self.name = name

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _CompiledGeneric(make_decoder(self.name, dem), dem.num_detectors)


def make_decoder(name: str, dem: stim.DetectorErrorModel):
    if name not in _FACTORIES:
        raise KeyError(f"unknown decoder {name!r}; choose from {DECODER_NAMES}")
    return _FACTORIES[name](dem)


def sinter_decoders() -> dict:
    """Custom decoders for `sinter.collect(custom_decoders=...)`; 'pymatching' is built into Sinter."""
    out: dict = {
        "pymatching_correlated": CorrelatedMatchingSinterDecoder(),
    }
    for name in OURS:
        out[name] = RegistrySinterDecoder(name)
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
