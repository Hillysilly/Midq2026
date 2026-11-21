"""Minimum-weight perfect matching (MWPM) and greedy matching, on our own decoding graph.

Review §5.1. The pipeline, for one shot:

1. **Defects**: the detectors that fired.
2. **Shortest paths**: Dijkstra from every defect over the detector graph (edge weight
   log((1-p)/p)), giving the cost of explaining any two defects by one chain of errors, and the
   cost of connecting each defect to the boundary.
3. **Syndrome graph**: a complete graph on the defects. Each defect also gets its own private
   copy of the boundary, and boundary copies connect to each other at zero cost, so any number
   of defects can be matched to the boundary (the standard construction from Fowler/Dennis et al.).
4. **Matching**:
   * `MWPMDecoder`: the exact minimum-weight perfect matching (Edmonds' blossom algorithm, here
     networkx's implementation; everything around it is ours).
   * `GreedyMatchingDecoder`: repeatedly take the pairing that saves the most cost over sending both
     defects to the boundary. Not optimal, but simple, O(k^2 log k), and easy to put in hardware.
5. **Correction**: walk each matched shortest path and XOR the observables its edges flip.

PyMatching's Sparse Blossom does all of this far faster by never building the complete graph,
but the answer (the matching) is the same, which `tests/test_new_decoders.py` checks.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import stim

from .dem_graph import DemGraph, obs_mask_to_bits


class _MatchingBase:
    def __init__(self, dem: stim.DetectorErrorModel | DemGraph):
        g = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.graph = g
        self.num_observables = g.num_observables
        w = np.maximum(g.edge_weights, 1e-9)  # Dijkstra needs positive weights
        n = g.num_nodes
        self.csgraph = sp.csr_matrix(
            (np.concatenate([w, w]), (np.concatenate([g.edge_u, g.edge_v]), np.concatenate([g.edge_v, g.edge_u]))),
            shape=(n, n))
        self.edge_of = {}
        for e, (u, v) in enumerate(zip(g.edge_u.tolist(), g.edge_v.tolist())):
            self.edge_of[(u, v)] = e
            self.edge_of[(v, u)] = e

    def _paths(self, defects: list[int]):
        dist, pred = dijkstra(self.csgraph, directed=False, indices=defects, return_predecessors=True)
        return dist, pred

    def _path_obs(self, pred_row: np.ndarray, target: int) -> int:
        """XOR of observable masks along the shortest path ending at `target`."""
        mask, x = 0, target
        eobs = self.graph.edge_obs
        while pred_row[x] >= 0:
            px = int(pred_row[x])
            mask ^= int(eobs[self.edge_of[(px, x)]])
            x = px
        return mask

    def _pairs(self, defects: list[int], dist: np.ndarray) -> list[tuple[int, int | None]]:
        raise NotImplementedError

    def decode(self, detection_events: np.ndarray) -> int:
        defects = np.flatnonzero(detection_events).tolist()
        if not defects:
            return 0
        dist, pred = self._paths(defects)
        B = self.graph.boundary
        mask = 0
        for i, j in self._pairs(defects, dist):
            mask ^= self._path_obs(pred[i], B if j is None else defects[j])
        return mask

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.atleast_2d(dets)
        return obs_mask_to_bits(np.array([self.decode(r) for r in dets], dtype=np.int64), self.num_observables)


class MWPMDecoder(_MatchingBase):
    """Exact MWPM: Dijkstra + boundary copies + blossom (networkx)."""

    def _pairs(self, defects, dist):
        k = len(defects)
        B = self.graph.boundary
        G = nx.Graph()
        # maximise (BIG - cost) with max cardinality == minimise total cost over perfect matchings
        big = 1.0 + float(np.nanmax(np.where(np.isfinite(dist), dist, 0))) * 2
        for i in range(k):
            db = dist[i, B]
            if np.isfinite(db):
                G.add_edge(i, k + i, weight=big - db)
            for j in range(i + 1, k):
                dij = dist[i, defects[j]]
                if np.isfinite(dij):
                    G.add_edge(i, j, weight=big - dij)
                G.add_edge(k + i, k + j, weight=big)  # boundary copies pair up for free
        matching = nx.max_weight_matching(G, maxcardinality=True)
        pairs = []
        for a, b in matching:
            a, b = min(a, b), max(a, b)
            if b < k:
                pairs.append((a, b))
            elif a < k:  # defect a matched to its boundary copy
                pairs.append((a, None))
        return pairs


class GreedyMatchingDecoder(_MatchingBase):
    """Greedy matching by *savings*. Start with every defect matched to the boundary. Pairing
    defects i and j instead saves  dist(i, B) + dist(j, B) - dist(i, j).  Take pairs in order of
    largest saving while the saving is positive and both defects are still free.

    (Naively taking the cheapest link first is much worse: a defect next to the boundary grabs
    it, and its partner from the same error chain is forced to the far boundary.)

    Pure greedy is *not* distance-preserving: at d = 5 about 1.4% of two-fault errors fail
    (it grabs one attractive pair and strands two defects at the boundary). With
    `local_search=True` we then run 2-opt: for any two pairs (a,b),(c,d) try (a,c),(b,d) and
    (a,d),(b,c) (a "pair" may be a defect-boundary link, or the empty boundary-boundary pair),
    and keep any swap that lowers the total cost, until none does. Each pass is O(k^2) and the
    swaps for disjoint pairs are independent, so it parallelises like the greedy step.
    """

    def __init__(self, dem, local_search: bool = False):
        super().__init__(dem)
        self.local_search = local_search

    def _pairs(self, defects, dist):
        k = len(defects)
        B = self.graph.boundary
        db = [dist[i, B] for i in range(k)]
        cand = []
        for i in range(k):
            for j in range(i + 1, k):
                dij = dist[i, defects[j]]
                if np.isfinite(dij):
                    cand.append((db[i] + db[j] - dij, i, j))
        cand.sort(key=lambda t: -t[0])
        free = [True] * k
        pairs = []
        for saving, i, j in cand:
            if saving <= 0:
                break
            if free[i] and free[j]:
                free[i] = free[j] = False
                pairs.append((i, j))
        pairs.extend((i, None) for i in range(k) if free[i])
        if self.local_search:
            pairs = self._two_opt(pairs, defects, dist)
        return pairs

    def _two_opt(self, pairs, defects, dist):
        B = self.graph.boundary

        def cost(x, y):
            if x is None and y is None:
                return 0.0
            if y is None:
                return dist[x, B]
            if x is None:
                return dist[y, B]
            return dist[x, defects[y]]

        # Shortest paths may pass through the boundary node, so dist(x, y) <= dist(x,B) + dist(y,B):
        # pairing two boundary-matched defects with each other never costs more and gives the
        # same correction. Doing it first lets 2-opt reach moves that would otherwise need a
        # zero-gain intermediate step.
        at_b = [x for x, y in pairs if y is None]
        pairs = [p for p in pairs if p[1] is not None]
        pairs += [(at_b[i], at_b[i + 1]) for i in range(0, len(at_b) - 1, 2)]
        if len(at_b) % 2:
            pairs.append((at_b[-1], None))
        pairs = [tuple(p) for p in pairs] + [(None, None)]
        improved = True
        while improved:
            improved = False
            for a in range(len(pairs)):
                for b in range(a + 1, len(pairs)):
                    (w, x), (y, z) = pairs[a], pairs[b]
                    now = cost(w, x) + cost(y, z)
                    for p1, p2 in (((w, y), (x, z)), ((w, z), (x, y))):
                        if cost(*p1) + cost(*p2) < now - 1e-9:
                            pairs[a], pairs[b] = p1, p2
                            now = cost(*p1) + cost(*p2)
                            improved = True
                            (w, x), (y, z) = pairs[a], pairs[b]
            # keep exactly one empty boundary-boundary pair to swap with
            pairs = [p for p in pairs if p != (None, None)] + [(None, None)]
        out = []
        for x, y in pairs:
            if x is None and y is None:
                continue
            if x is None:
                x, y = y, None
            out.append((x, y) if y is None or x < y else (y, x))
        return out
