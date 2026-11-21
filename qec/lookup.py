"""Lookup-table decoder: precompute the most likely correction for every likely syndrome.

The simplest decoder there is, and the one real hardware used first (it is a ROM read, so it
takes one clock cycle). We enumerate every combination of up to `max_weight` error mechanisms
from the detector error model, compute the syndrome each produces, and for each syndrome keep the
combination with the highest probability. At decode time, the syndrome is the key.

* Within its table, it's the most-likely-error decoder (better than MWPM, which only sees a graph).
* The table grows like n^w (n mechanisms), so it's only practical for small codes: d = 3 at
  circuit level (219 mechanisms -> ~24k entries for w = 2). That's the classic trade-off: perfect
  speed, exponential memory.
* Syndromes not in the table go to a `fallback` decoder (default: our Union-Find).
"""
from __future__ import annotations

import itertools

import numpy as np
import stim

from .dem_graph import DemGraph, obs_mask_to_bits
from .union_find import UnionFindDecoder


class LookupTableDecoder:
    def __init__(self, dem: stim.DetectorErrorModel | DemGraph, max_weight: int = 2,
                 fallback=None, max_entries: int = 5_000_000):
        g = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.graph = g
        self.num_observables = g.num_observables
        n = g.num_mechanisms
        Hc, Lc = g.H.tocsc(), g.L.tocsc()
        syn = [sum(1 << int(i) for i in Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]) for j in range(n)]
        obs = [sum(1 << int(i) for i in Lc.indices[Lc.indptr[j]:Lc.indptr[j + 1]]) for j in range(n)]
        logodds = np.log(g.priors / (1 - g.priors)).tolist()  # log P(combo) up to a constant

        table: dict[int, tuple[float, int]] = {0: (0.0, 0)}
        for w in range(1, max_weight + 1):
            for combo in itertools.combinations(range(n), w):
                s, o, lp = 0, 0, 0.0
                for j in combo:
                    s ^= syn[j]
                    o ^= obs[j]
                    lp += logodds[j]
                best = table.get(s)
                if best is None or lp > best[0]:
                    table[s] = (lp, o)
            if len(table) > max_entries:
                raise MemoryError(f"lookup table exceeded {max_entries} entries at weight {w}")
        self.table = {s: o for s, (lp, o) in table.items()}
        self.fallback = fallback if fallback is not None else UnionFindDecoder(g, weighted=True)
        self.last_hit_rate = 1.0

    @staticmethod
    def _key(row: np.ndarray) -> int:
        return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.atleast_2d(dets)
        masks = np.zeros(len(dets), dtype=np.int64)
        miss = []
        for i, row in enumerate(dets):
            o = self.table.get(self._key(row))
            if o is None:
                miss.append(i)
            else:
                masks[i] = o
        out = obs_mask_to_bits(masks, self.num_observables)
        if miss:
            out[miss] = self.fallback.decode_batch(dets[miss])
        self.last_hit_rate = 1 - len(miss) / max(1, len(dets))
        return out
