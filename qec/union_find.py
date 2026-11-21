"""Union-Find decoder (Delfosse & Nickerson, arXiv:1709.06218) in pure Python.

The decoder runs on our own detector graph (`qec/dem_graph.py`), which has exactly the same
edges and weights as PyMatching's, so the two can be compared on equal terms.

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
import sinter
import stim

from .dem_graph import DemGraph, obs_mask_to_bits

try:
    import numba as _nb
except ImportError:  # pragma: no cover
    _nb = None


def _uf_one(dets, s, cap, eu, ev, eobs, adj_ptr, adj_edge, adj_nbr, B,
            parent, size, parity, has_b, reached, support, root, newly, visited, tree_e, tree_p, order, defect):
    """Decode shot `s` using caller-provided scratch arrays. Returns (observable bitmask, growth rounds)."""
    N = B + 1
    M = eu.size
    n_def = 0
    for x in range(N):
        parent[x] = x
        size[x] = 1
        parity[x] = False
        has_b[x] = False
        reached[x] = False
        visited[x] = False
        tree_e[x] = -1
        tree_p[x] = -1
        defect[x] = False
    has_b[B] = True
    for x in range(B):
        if dets[s, x]:
            parity[x] = True
            reached[x] = True
            defect[x] = True
            n_def += 1
    if n_def == 0:
        return 0, 0
    for e in range(M):
        support[e] = 0
    rounds = 0
    while True:
        # roots and activity, frozen for this round
        for x in range(N):
            r = x
            while parent[r] != r:
                r = parent[r]
            root[x] = r
        any_active = False
        for x in range(N):
            if reached[x] and root[x] == x and parity[x] and not has_b[x]:
                any_active = True
                break
        if not any_active:
            break
        rounds += 1
        n_new = 0
        progressed = False
        for e in range(M):  # every edge independently
            if support[e] >= cap[e]:
                continue
            u = eu[e]
            v = ev[e]
            inc = 0
            ru = -1
            if reached[u]:
                ru = root[u]
                if parity[ru] and not has_b[ru]:
                    inc += 1
            if reached[v]:
                rv = root[v]
                if rv != ru and parity[rv] and not has_b[rv]:
                    inc += 1
            if inc > 0:
                progressed = True
                support[e] = min(cap[e], support[e] + inc)
                if support[e] == cap[e]:
                    newly[n_new] = e
                    n_new += 1
        if not progressed:
            break
        for i in range(n_new):  # merge clusters joined by fully grown edges
            e = newly[i]
            u = eu[e]
            v = ev[e]
            reached[u] = True
            reached[v] = True
            ru = u
            while parent[ru] != ru:
                ru = parent[ru]
            rv = v
            while parent[rv] != rv:
                rv = parent[rv]
            if ru == rv:
                continue
            if size[ru] < size[rv]:
                tmp = ru
                ru = rv
                rv = tmp
            parent[rv] = ru
            size[ru] += size[rv]
            parity[ru] ^= parity[rv]
            has_b[ru] |= has_b[rv]


    # peel: BFS forest over grown edges (boundary component first), then leaves inwards
    n_ord = 0
    for start_i in range(N + 1):
        start = B if start_i == 0 else start_i - 1
        if not reached[start] or visited[start]:
            continue
        visited[start] = True
        order[n_ord] = start
        head = n_ord
        n_ord += 1
        while head < n_ord:
            x = order[head]
            head += 1
            for k in range(adj_ptr[x], adj_ptr[x + 1]):
                e = adj_edge[k]
                y = adj_nbr[k]
                if support[e] >= cap[e] and not visited[y]:
                    visited[y] = True
                    tree_e[y] = e
                    tree_p[y] = x
                    order[n_ord] = y
                    n_ord += 1
    for x in range(B):
        defect[x] = dets[s, x]
    pred = 0
    for i in range(n_ord - 1, -1, -1):
        x = order[i]
        if tree_e[x] >= 0 and defect[x]:
            pred ^= eobs[tree_e[x]]
            defect[x] = False
            defect[tree_p[x]] ^= True
    return pred, rounds


def _uf_kernel_py(dets, caps, per_shot, eu, ev, eobs, adj_ptr, adj_edge, adj_nbr, B):
    """Union-Find over a batch of shots, written in the *synchronous, edge-parallel* form.

    Each growth round, every edge looks at the clusters at its two ends (as they were at the start
    of the round) and grows by one step per active neighbouring cluster. All edges can do this at
    the same time, which is how a hardware (FPGA/ASIC) Union-Find works. Then the fully grown
    edges merge their clusters. Rounds repeat until no cluster is odd and boundary-free. Peeling
    is a BFS forest over the grown edges, stripped from the leaves inwards.

    Returns (observable bitmask per shot, growth rounds per shot).
    """
    S = dets.shape[0]
    N = B + 1
    M = eu.size
    out = np.zeros(S, dtype=np.int64)
    rounds_out = np.zeros(S, dtype=np.int64)
    block = 256
    n_blocks = (S + block - 1) // block
    for blk in _prange(n_blocks):
        # scratch arrays: allocated once per block of shots (not once per shot), reset inside _uf_one
        parent = np.empty(N, dtype=np.int64)
        size = np.empty(N, dtype=np.int64)
        parity = np.empty(N, dtype=np.bool_)
        has_b = np.empty(N, dtype=np.bool_)
        reached = np.empty(N, dtype=np.bool_)
        support = np.empty(M, dtype=np.int64)
        root = np.empty(N, dtype=np.int64)
        newly = np.empty(M, dtype=np.int64)
        visited = np.empty(N, dtype=np.bool_)
        tree_e = np.empty(N, dtype=np.int64)
        tree_p = np.empty(N, dtype=np.int64)
        order = np.empty(N, dtype=np.int64)
        defect = np.empty(N, dtype=np.bool_)
        s0 = np.int64(blk) * block
        s1 = min(S, s0 + block)
        for s in range(s0, s1):
            row = s if per_shot else 0
            pred, rounds = _uf_one(dets, s, caps[row], eu, ev, eobs, adj_ptr, adj_edge, adj_nbr, B,
                                   parent, size, parity, has_b, reached, support, root, newly,
                                   visited, tree_e, tree_p, order, defect)
            out[s] = pred
            rounds_out[s] = rounds
    return out, rounds_out


if _nb is not None:
    _prange = _nb.prange
    _uf_one = _nb.njit(cache=True)(_uf_one)
    _uf_kernel = _nb.njit(parallel=True, cache=True)(_uf_kernel_py)
else:  # pragma: no cover
    _prange = range
    _uf_kernel = _uf_kernel_py



class UnionFindDecoder:
    def __init__(self, dem: stim.DetectorErrorModel | DemGraph, weighted: bool = False,
                 resolution: float = 4.0):
        """
        Args:
            dem: detector error model (from `qec.detector_error_model`), or a prebuilt `DemGraph`.
            weighted: grow edges at a speed set by their weight (see module docstring).
            resolution: growth steps for the lightest edge when weighted. Too coarse (2) makes
                ties between "pair up" and "both go to the boundary" that peeling breaks badly,
                and some single faults then fail at d=3. 4 fixes this, and higher is only slower.
        """
        g = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.graph = g
        self.num_detectors = g.num_detectors
        self.num_observables = g.num_observables
        self.boundary = g.boundary  # single virtual boundary node
        self.resolution = resolution
        self.edge_u: list[int] = g.edge_u.tolist()
        self.edge_v: list[int] = g.edge_v.tolist()
        self.edge_obs: list[int] = g.edge_obs.tolist()  # bitmask of observables each edge flips
        self.adj = g.adj
        weights = g.edge_weights
        self.w_ref = float(weights[weights > 0].min()) if (weights > 0).any() else 1.0
        # Growth steps needed to fully grow each edge.
        self.edge_cap = self.caps_from_weights(weights) if weighted else [2] * len(weights)
        self.last_growth_rounds = 0  # parallel depth of the last decode (see qec/metrics.py)
        self.backend = "numba" if _nb is not None else "python"
        # flat arrays for the compiled kernel
        nbrs = [(e, y) for x in range(g.num_nodes) for e, y in self.adj[x]]
        self._adj_ptr = np.concatenate([[0], np.cumsum([len(a) for a in self.adj])]).astype(np.int64)
        self._adj_edge = np.array([e for e, _ in nbrs], dtype=np.int64)
        self._adj_nbr = np.array([y for _, y in nbrs], dtype=np.int64)
        self._eu = g.edge_u.astype(np.int64)
        self._ev = g.edge_v.astype(np.int64)
        self._eobs = g.edge_obs.astype(np.int64)
        self.last_batch_rounds = np.zeros(0, dtype=np.int64)

    def caps_from_weights(self, weights: np.ndarray) -> list[int]:
        """Edge weight -> number of growth steps. The lightest static edge takes `resolution` steps."""
        w = np.maximum(np.asarray(weights, dtype=float), 0.0)
        return np.maximum(1, np.rint(self.resolution * w / self.w_ref)).astype(int).tolist()

    def decode(self, detection_events: np.ndarray, edge_cap: list[int] | None = None) -> int:
        """Return a bitmask of predicted logical-observable flips for one shot.

        `edge_cap` overrides the growth steps per edge for this shot only (BP+UF uses it to feed
        in per-shot weights from belief propagation)."""
        defects = np.flatnonzero(detection_events).tolist()
        self.last_growth_rounds = 0
        if not defects:
            return 0
        B, adj, eu, ev = self.boundary, self.adj, self.edge_u, self.edge_v
        cap = self.edge_cap if edge_cap is None else edge_cap

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
            self.last_growth_rounds += 1
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

    def decode_batch(self, detection_events: np.ndarray, edge_caps: np.ndarray | None = None) -> np.ndarray:
        """Decode a (shots, num_detectors) bool array -> (shots, num_observables) bool array.

        Uses the compiled edge-parallel kernel when numba is installed (same answers as `decode`,
        checked in tests/test_decoders.py). `edge_caps` may be (num_edges,) or (shots, num_edges).
        Growth rounds per shot are left in `self.last_batch_rounds`.
        """
        dets = np.ascontiguousarray(np.atleast_2d(detection_events), dtype=np.bool_)
        caps = np.atleast_2d(np.asarray(self.edge_cap if edge_caps is None else edge_caps, dtype=np.int64))
        if self.backend == "python":
            masks, rounds = [], []
            for i, row in enumerate(dets):
                masks.append(self.decode(row, edge_cap=caps[min(i, len(caps) - 1)].tolist()))
                rounds.append(self.last_growth_rounds)
            self.last_batch_rounds = np.array(rounds, dtype=np.int64)
            return obs_mask_to_bits(np.array(masks, dtype=np.int64), self.num_observables)
        # One row of caps for every shot unless per-shot caps were given (BP+UF). Copying the row
        # once per shot cost ~2 us/shot and 3 GB of memory at d = 7 before this was fixed.
        per_shot = caps.shape[0] > 1
        if per_shot and caps.shape[0] != len(dets):
            raise ValueError("edge_caps must have one row, or one row per shot")
        masks, rounds = _uf_kernel(dets, np.ascontiguousarray(caps), per_shot, self._eu, self._ev, self._eobs,
                                   self._adj_ptr, self._adj_edge, self._adj_nbr, self.boundary)
        self.last_batch_rounds = rounds
        return obs_mask_to_bits(masks, self.num_observables)


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
