"""Turn a Stim detector error model (DEM) into the two structures decoders need.

Written from scratch (no PyMatching), so every decoder of ours that uses it runs on our own code.

A DEM is a list of independent error mechanisms. Each one has a probability, the detectors it
flips and the logical observables it flips. With `decompose_errors=True`, Stim also splits each
mechanism into graph-like pieces ("components") that flip at most two detectors each, separated
by `^`. For example, a Y error on a data qubit = (its X part) ^ (its Z part).

We keep both views:

* **Check matrices** (for belief propagation / OSD / lookup tables):
  `H[detector, mechanism]` and `L[observable, mechanism]` over GF(2), plus a prior probability per
  mechanism. Mechanisms with identical (detectors, observables) are merged.
* **Decoding graph** (for matching / Union-Find): nodes are detectors plus one virtual boundary
  node. Each component is an edge (u, v), or (u, boundary) if it flips a single detector.
  Parallel edges are merged: two independent ways to flip the same pair combine as
  p = p1(1-p2) + p2(1-p1). Edge weight = log((1-p)/p), so a lower-probability edge is "longer".

`col_edges[j]` lists the graph edges that mechanism j decomposes into. Belief-matching style
decoders use it to push BP's soft output (per mechanism) back onto graph edges (per edge).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import stim


def xor_prob(p: np.ndarray | float, q: np.ndarray | float):
    """Probability that exactly one of two independent events happens."""
    return p * (1 - q) + q * (1 - p)


@dataclass
class DemGraph:
    num_detectors: int
    num_observables: int
    # check-matrix view (one column per merged mechanism)
    H: sp.csr_matrix                 # (num_detectors, n) uint8
    L: sp.csr_matrix                 # (num_observables, n) uint8
    priors: np.ndarray               # (n,) float
    col_edges: list[list[int]]       # graph edges making up each mechanism
    # graph view
    edge_u: np.ndarray               # (m,) int
    edge_v: np.ndarray               # (m,) int, boundary = num_detectors
    edge_p: np.ndarray               # (m,) float
    edge_obs: np.ndarray             # (m,) int bitmask of observables
    adj: list[list[tuple[int, int]]] = field(repr=False)  # node -> [(edge, neighbour)]

    @property
    def boundary(self) -> int:
        return self.num_detectors

    @property
    def num_nodes(self) -> int:
        return self.num_detectors + 1

    @property
    def edge_weights(self) -> np.ndarray:
        p = np.clip(self.edge_p, 1e-12, 0.5 - 1e-12)
        return np.log((1 - p) / p)

    @property
    def num_mechanisms(self) -> int:
        return self.H.shape[1]

    def edge_probs_from_mechanisms(self, mech_p: np.ndarray) -> np.ndarray:
        """Map per-mechanism probabilities (shots, n) to per-edge probabilities (shots, m).

        Each edge flips if an odd number of the mechanisms containing it fire, so the
        probabilities combine with `xor_prob`, exactly as when the graph was first built.
        """
        mech_p = np.atleast_2d(mech_p)
        out = np.zeros((mech_p.shape[0], len(self.edge_u)))
        for j, edges in enumerate(self.col_edges):
            for e in edges:
                out[:, e] = xor_prob(out[:, e], mech_p[:, j])
        return out

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "DemGraph":
        nd, no = dem.num_detectors, dem.num_observables
        B = nd
        cols: dict[tuple, int] = {}          # (dets, obs) -> column index
        priors: list[float] = []
        col_edges: list[set[int]] = []
        edges: dict[tuple[int, int], int] = {}  # (u, v) -> edge index
        edge_p: list[float] = []
        edge_obs: list[int] = []
        edge_obs_p: list[float] = []         # prob of the mechanism that set edge_obs (keep likeliest)

        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            if p <= 0:
                continue
            # split into graph-like components at the `^` separators
            comps: list[tuple[list[int], int]] = [([], 0)]
            for t in inst.targets_copy():
                if t.is_separator():
                    comps.append(([], 0))
                elif t.is_relative_detector_id():
                    comps[-1][0].append(t.val)
                elif t.is_logical_observable_id():
                    dets, mask = comps[-1]
                    comps[-1] = (dets, mask ^ (1 << t.val))

            # mechanism = XOR of its components
            det_set: set[int] = set()
            obs_mask = 0
            these_edges: list[int] = []
            for dets, mask in comps:
                det_set ^= set(dets)
                obs_mask ^= mask
                if len(dets) == 0:
                    continue  # pure-observable piece: cannot be seen, cannot be decoded
                if len(dets) > 2:
                    raise ValueError("DEM is not decomposed into graph-like pieces; "
                                     "use qec.detector_error_model(circuit)")
                u, v = (dets[0], B) if len(dets) == 1 else sorted(dets)
                key = (u, v)
                if key not in edges:
                    edges[key] = len(edge_p)
                    edge_p.append(0.0)
                    edge_obs.append(mask)
                    edge_obs_p.append(p)
                e = edges[key]
                edge_p[e] = xor_prob(edge_p[e], p)
                if p > edge_obs_p[e]:
                    edge_obs[e], edge_obs_p[e] = mask, p
                these_edges.append(e)

            if not det_set:
                continue
            key = (tuple(sorted(det_set)), obs_mask)
            if key not in cols:
                cols[key] = len(priors)
                priors.append(0.0)
                col_edges.append(set())
            j = cols[key]
            priors[j] = xor_prob(priors[j], p)
            col_edges[j].update(these_edges)

        n = len(priors)
        rows_h, cols_h, rows_l, cols_l = [], [], [], []
        for (dets, mask), j in cols.items():
            rows_h.extend(dets)
            cols_h.extend([j] * len(dets))
            for k in range(no):
                if mask >> k & 1:
                    rows_l.append(k)
                    cols_l.append(j)
        H = sp.csr_matrix((np.ones(len(rows_h), np.uint8), (rows_h, cols_h)), shape=(nd, n))
        L = sp.csr_matrix((np.ones(len(rows_l), np.uint8), (rows_l, cols_l)), shape=(no, n))

        eu = np.array([k[0] for k in edges], dtype=np.int64)
        ev = np.array([k[1] for k in edges], dtype=np.int64)
        adj: list[list[tuple[int, int]]] = [[] for _ in range(nd + 1)]
        for e, (u, v) in enumerate(zip(eu.tolist(), ev.tolist())):
            adj[u].append((e, v))
            adj[v].append((e, u))
        return cls(nd, no, H, L, np.array(priors), [sorted(s) for s in col_edges],
                   eu, ev, np.array(edge_p), np.array(edge_obs, dtype=np.int64), adj)


def obs_mask_to_bits(masks: np.ndarray, num_observables: int) -> np.ndarray:
    """int bitmasks (shots,) -> bool array (shots, num_observables)."""
    masks = np.asarray(masks, dtype=np.int64)
    return ((masks[:, None] >> np.arange(num_observables)) & 1).astype(bool)
