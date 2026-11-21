"""Union-Find decoder (Delfosse & Nickerson, arXiv:1709.06218) in pure Python.

The decoder runs on the same detector graph PyMatching uses, built from a Stim
detector error model, so the two can be compared on equal terms.

* `weighted=False` (default): the original algorithm. Every edge needs two half-edge growth
  steps no matter its probability. Fast and simple, but less accurate than MWPM (review §5.2).
* `weighted=True`: edges with lower probability (higher weight log((1-p)/p)) take
  proportionally more growth steps, so clusters grow faster along likely errors. This narrows
  much of the gap to MWPM (Huang, Newman & Brown, arXiv:2004.04693).

The algorithm has two steps:
  1. Grow clusters: each cluster with an odd number of detection events grows
     by half an edge per round. When an edge becomes fully grown, the clusters
     at its two ends merge. A cluster stops growing once its parity is even or
     it touches the boundary.
  2. Peel: build a spanning tree over each cluster's fully grown edges, then
     strip leaves off the tree. Each leaf that holds a detection event puts its
     parent edge into the correction. The XOR of the logical observables on the
     correction edges is the prediction.
"""
from __future__ import annotations

import numpy as np
import pymatching
import sinter
import stim


class UnionFindDecoder:
    def __init__(self, dem: stim.DetectorErrorModel, weighted: bool = False, resolution: float = 4.0):
        """
        Args:
            dem: detector error model (from `qec.detector_error_model`).
            weighted: grow edges at a speed set by their weight (see module docstring).
            resolution: growth steps for the lightest edge when weighted. Too coarse (2) makes
                ties between "pair up" and "both go to the boundary" that peeling breaks badly,
                and some single faults then fail at d=3. 4 fixes this, and higher is only slower.
        """
        matching = pymatching.Matching.from_detector_error_model(dem)
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.boundary = self.num_detectors  # single virtual boundary node

        self.edge_u: list[int] = []
        self.edge_v: list[int] = []
        self.edge_obs: list[int] = []  # bitmask of observables each edge flips
        weights: list[float] = []
        for u, v, attrs in matching.edges():
            mask = 0
            for f in attrs.get("fault_ids", ()):
                mask ^= 1 << f
            self.edge_u.append(u)
            self.edge_v.append(self.boundary if v is None else v)
            self.edge_obs.append(mask)
            weights.append(attrs.get("weight", 1.0))

        # Growth steps needed to fully grow each edge.
        if weighted and weights:
            positive = [w for w in weights if w > 0]
            w_min = min(positive) if positive else 1.0
            self.edge_cap = [max(1, round(resolution * max(w, w_min) / w_min)) for w in weights]
        else:
            self.edge_cap = [2] * len(weights)

        self.adj: list[list[tuple[int, int]]] = [[] for _ in range(self.num_detectors + 1)]
        for e, (u, v) in enumerate(zip(self.edge_u, self.edge_v)):
            self.adj[u].append((e, v))
            self.adj[v].append((e, u))

    def decode(self, detection_events: np.ndarray) -> int:
        """Return a bitmask of predicted logical-observable flips for one shot."""
        defects = np.flatnonzero(detection_events).tolist()
        if not defects:
            return 0
        B, adj, eu, ev, cap = self.boundary, self.adj, self.edge_u, self.edge_v, self.edge_cap

        # Disjoint-set forest over vertices that clusters have reached.
        parent: dict[int, int] = {}
        members: dict[int, list[int]] = {}
        parity: dict[int, int] = {}
        has_boundary: dict[int, bool] = {}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def add(x: int) -> None:
            if x not in parent:
                parent[x] = x
                members[x] = [x]
                parity[x] = 0
                has_boundary[x] = x == B

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if len(members[ra]) < len(members[rb]):
                ra, rb = rb, ra
            parent[rb] = ra
            members[ra].extend(members.pop(rb))
            parity[ra] ^= parity.pop(rb)
            has_boundary[ra] |= has_boundary.pop(rb)

        for d in defects:
            add(d)
            parity[d] = 1

        # Step 1: grow clusters.
        support: dict[int, int] = {}  # edge -> growth so far; full when it reaches cap[e]
        active = set(defects)
        while active:
            grown: list[int] = []
            progressed = False
            for r in active:
                seen: set[int] = set()
                for x in members[r]:
                    for e, _ in adj[x]:
                        if e in seen:
                            continue
                        seen.add(e)
                        s = support.get(e, 0)
                        if s < cap[e]:
                            progressed = True
                            support[e] = s + 1
                            if s + 1 == cap[e]:
                                grown.append(e)
            for e in grown:
                add(eu[e])
                add(ev[e])
                union(eu[e], ev[e])
            if not progressed:  # odd cluster with nowhere to grow (should not happen)
                break
            active = {find(r) for r in active}
            active = {r for r in active if parity[r] and not has_boundary[r]}

        # Step 2: peel a spanning forest of the fully grown edges.
        erasure_adj: dict[int, list[tuple[int, int]]] = {}
        for e, s in support.items():
            if s >= cap[e]:
                erasure_adj.setdefault(eu[e], []).append((e, ev[e]))
                erasure_adj.setdefault(ev[e], []).append((e, eu[e]))

        defect_set = set(defects)
        prediction = 0
        visited: set[int] = set()
        for r in {find(x) for x in parent}:
            start = B if has_boundary[r] else r
            order = [start]
            tree_edge: dict[int, tuple[int, int]] = {}
            visited.add(start)
            i = 0
            while i < len(order):
                x = order[i]
                i += 1
                for e, y in erasure_adj.get(x, ()):
                    if y not in visited:
                        visited.add(y)
                        tree_edge[y] = (e, x)
                        order.append(y)
            for x in reversed(order[1:]):
                if x in defect_set:
                    e, px = tree_edge[x]
                    prediction ^= self.edge_obs[e]
                    defect_set.discard(x)
                    defect_set ^= {px}
        return prediction

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode a (shots, num_detectors) bool array -> (shots, num_observables) bool array."""
        out = np.zeros((detection_events.shape[0], self.num_observables), dtype=bool)
        for i, row in enumerate(detection_events):
            mask = self.decode(row)
            for k in range(self.num_observables):
                out[i, k] = (mask >> k) & 1
        return out


class _CompiledUnionFind(sinter.CompiledDecoder):
    def __init__(self, decoder: UnionFindDecoder):
        self.decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            bitorder="little",
            count=self.decoder.num_detectors,
        ).astype(bool)
        predictions = self.decoder.decode_batch(dets)
        return np.packbits(predictions, axis=1, bitorder="little")


class UnionFindSinterDecoder(sinter.Decoder):
    """Lets Sinter use the decoder: `custom_decoders={"union_find": UnionFindSinterDecoder()}`."""

    def __init__(self, weighted: bool = False):
        self.weighted = weighted

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _CompiledUnionFind(UnionFindDecoder(dem, weighted=self.weighted))
