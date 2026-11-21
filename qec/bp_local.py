"""Local BP + Union-Find: belief-find that only does work near the lit detectors.

Why: global BP+UF (`qec/bp.py`) updates every Tanner-graph edge every iteration (5,004 at d = 5,
17,156 at d = 7) even when only one or two detectors are lit. Its cost per message update is already
compiled-code speed (~5 ns), so the gap to PyMatching is the *amount* of work, not the language.
Sparse Blossom is fast because it only touches the graph near lit detectors; this does the same for BP.

Per shot, in one compiled kernel (shots spread over CPU cores, scratch arrays reused per block of shots):

1. **Region.** Breadth-first search from the lit detectors over the check graph (two checks are
   neighbours if some fault flips both), `radius` hops deep. These are the *active* checks; the faults
   touching them are the *active* variables.
2. **Local BP.** Normalised min-sum runs only on active checks and variables. A far-away check is
   unlit and surrounded by faults at their prior, so its message to an active variable is fixed at the
   value it would have after one iteration ("not flipped", strength = scaled smallest prior LLR).
   Convergence is checked on the active checks *and* the unlit checks just outside the region.
3. **Converged:** BP's answer gives the prediction. **Not converged:** each active fault's posterior
   replaces its prior in the probability of the graph edges it decomposes into (edges combine as
   1 - 2p = product of (1 - 2p_j)), those edges get new Union-Find growth costs, and weighted Union-Find
   runs on this shot with them. All other edges keep their prior-based costs.

Results and speed vs global BP+UF: `results/` and `docs/DECODERS.md`.
"""
from __future__ import annotations

import numpy as np
import stim

from .bp import BPUnionFindDecoder, MinSumBP
from .dem_graph import DemGraph, obs_mask_to_bits
from .union_find import UnionFindDecoder, _nb, _prange, _uf_one


def _bpuf_local_py(dets, prior, row_ptr, var_of, check_of, col_ptr, col_msgs, rfix, col_obs,
                   ce_ptr, ce_idx, prod_static, cap0, w_ref, resolution, radius, max_iter, scaling,
                   eu, ev, eobs, adj_ptr, adj_edge, adj_nbr, B):
    S, m = dets.shape
    n = prior.size
    nnz = var_of.size
    M = eu.size
    N = B + 1
    out = np.zeros(S, dtype=np.int64)
    iters = np.zeros(S, dtype=np.int64)
    conv = np.zeros(S, dtype=np.bool_)
    nact = np.zeros(S, dtype=np.int64)
    ufr = np.zeros(S, dtype=np.int64)
    block = 64
    n_blocks = (S + block - 1) // block
    for blk in _prange(n_blocks):
        act_c = np.zeros(m, dtype=np.bool_)
        seen_c = np.zeros(m, dtype=np.bool_)
        act_v = np.zeros(n, dtype=np.bool_)
        clist = np.empty(m, dtype=np.int64)
        blist = np.empty(m, dtype=np.int64)
        depth = np.empty(m, dtype=np.int64)
        vlist = np.empty(n, dtype=np.int64)
        q = np.zeros(nnz)
        r = np.zeros(nnz)
        total = np.zeros(n)
        touched = np.zeros(M, dtype=np.bool_)
        tlist = np.empty(M, dtype=np.int64)
        prod = np.empty(M)
        caps = cap0.copy()
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
            # ---- 1. region: BFS over the check graph from the lit checks
            nc = 0
            for c in range(m):
                if dets[s, c]:
                    clist[nc] = c
                    nc += 1
                    seen_c[c] = True
                    act_c[c] = True
                    depth[c] = 0
            if nc == 0:
                conv[s] = True
                continue
            head = 0
            while head < nc:
                c = clist[head]
                head += 1
                if depth[c] >= radius:
                    continue
                for k in range(row_ptr[c], row_ptr[c + 1]):
                    v = var_of[k]
                    for kk in range(col_ptr[v], col_ptr[v + 1]):
                        c2 = check_of[col_msgs[kk]]
                        if not seen_c[c2]:
                            seen_c[c2] = True
                            act_c[c2] = True
                            depth[c2] = depth[c] + 1
                            clist[nc] = c2
                            nc += 1
            nv = 0
            for i in range(nc):
                c = clist[i]
                for k in range(row_ptr[c], row_ptr[c + 1]):
                    v = var_of[k]
                    q[k] = prior[v]
                    if not act_v[v]:
                        act_v[v] = True
                        vlist[nv] = v
                        nv += 1
            # every message into an active variable: 0 from active checks (updated each iteration),
            # the fixed far-away value from inactive ones (never updated). Then the iteration loops
            # below need no "is this check active?" branch.
            for i in range(nv):
                v = vlist[i]
                for kk in range(col_ptr[v], col_ptr[v + 1]):
                    k = col_msgs[kk]
                    r[k] = 0.0 if act_c[check_of[k]] else rfix[k]
            nb = 0
            for i in range(nv):
                v = vlist[i]
                for kk in range(col_ptr[v], col_ptr[v + 1]):
                    c2 = check_of[col_msgs[kk]]
                    if not seen_c[c2]:
                        seen_c[c2] = True
                        blist[nb] = c2
                        nb += 1
            nact[s] = nc

            # ---- 2. local min-sum BP
            ok = False
            it = 0
            while it < max_iter:
                it += 1
                for i in range(nc):
                    c = clist[i]
                    a = row_ptr[c]
                    b = row_ptr[c + 1]
                    sgn = dets[s, c]
                    min1 = np.inf
                    min2 = np.inf
                    arg = -1
                    for k in range(a, b):
                        x = q[k]
                        if x < 0:
                            sgn = not sgn
                            x = -x
                        if x < min1:
                            min2 = min1
                            min1 = x
                            arg = k
                        elif x < min2:
                            min2 = x
                    for k in range(a, b):
                        mag = min2 if k == arg else min1
                        if mag > 50.0:
                            mag = 50.0
                        if sgn != (q[k] < 0):
                            r[k] = -scaling * mag
                        else:
                            r[k] = scaling * mag
                for i in range(nv):
                    v = vlist[i]
                    t = prior[v]
                    for kk in range(col_ptr[v], col_ptr[v + 1]):
                        t += r[col_msgs[kk]]
                    total[v] = t
                for i in range(nc):  # variable -> check messages, in contiguous row order
                    c = clist[i]
                    for k in range(row_ptr[c], row_ptr[c + 1]):
                        q[k] = total[var_of[k]] - r[k]
                ok = True
                for i in range(nc):
                    c = clist[i]
                    par = dets[s, c]
                    for k in range(row_ptr[c], row_ptr[c + 1]):
                        if total[var_of[k]] < 0:
                            par = not par
                    if par:
                        ok = False
                        break
                if ok:
                    for i in range(nb):
                        c = blist[i]
                        par = False
                        for k in range(row_ptr[c], row_ptr[c + 1]):
                            v = var_of[k]
                            if act_v[v] and total[v] < 0:
                                par = not par
                        if par:
                            ok = False
                            break
                if ok:
                    break
            iters[s] = it
            conv[s] = ok

            # ---- 3. converged: BP's answer; otherwise beliefs -> local edge costs -> Union-Find
            if ok:
                pred = 0
                for i in range(nv):
                    v = vlist[i]
                    if total[v] < 0:
                        pred ^= col_obs[v]
                out[s] = pred
            else:
                nt = 0
                for i in range(nv):
                    v = vlist[i]
                    tv = min(max(total[v], -50.0), 50.0)
                    post = 1.0 / (1.0 + np.exp(tv))
                    pr = 1.0 / (1.0 + np.exp(prior[v]))
                    ratio = (1.0 - 2.0 * post) / (1.0 - 2.0 * pr)
                    for j in range(ce_ptr[v], ce_ptr[v + 1]):
                        e = ce_idx[j]
                        if not touched[e]:
                            touched[e] = True
                            tlist[nt] = e
                            nt += 1
                            prod[e] = prod_static[e]
                        prod[e] *= ratio
                for i in range(nt):
                    e = tlist[i]
                    pe = 0.5 * (1.0 - prod[e])
                    if pe < 1e-9:
                        pe = 1e-9
                    if pe > 0.5 - 1e-9:
                        pe = 0.5 - 1e-9
                    w = np.log((1.0 - pe) / pe)
                    cv = np.int64(np.rint(resolution * w / w_ref))
                    caps[e] = cv if cv > 1 else 1
                pred, rounds = _uf_one(dets, s, caps, eu, ev, eobs, adj_ptr, adj_edge, adj_nbr, B,
                                       parent, size, parity, has_b, reached, support, root, newly,
                                       visited, tree_e, tree_p, order, defect)
                out[s] = pred
                ufr[s] = rounds
                for i in range(nt):
                    e = tlist[i]
                    caps[e] = cap0[e]
                    touched[e] = False
            # ---- reset scratch for the next shot
            for i in range(nc):
                act_c[clist[i]] = False
                seen_c[clist[i]] = False
            for i in range(nb):
                seen_c[blist[i]] = False
            for i in range(nv):
                act_v[vlist[i]] = False
    return out, iters, conv, nact, ufr


_bpuf_local = _nb.njit(parallel=True, cache=True)(_bpuf_local_py) if _nb is not None else None


class LocalBPUnionFindDecoder:
    """Belief-find with BP restricted to a neighbourhood of the lit detectors (see module docstring).

    Args:
        radius: check-graph hops around each lit detector that BP runs on.
        max_iter: BP iterations before handing over to Union-Find (Union-Find cleans up anyway, so fewer
            are needed than for BP on its own).
    """

    def __init__(self, dem: stim.DetectorErrorModel | DemGraph, radius: int = 1, max_iter: int = 10,
                 scaling: float = 0.625, resolution: float = 4.0):
        g = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.graph = g
        self.num_observables = g.num_observables
        self.radius, self.max_iter, self.scaling, self.resolution = radius, max_iter, scaling, resolution
        bp = MinSumBP(g.H, g.priors, max_iter, scaling)
        self.uf = UnionFindDecoder(g, weighted=True, resolution=resolution)
        self._row_ptr = bp.H.indptr.astype(np.int64)
        self._var_of = bp.var_of
        self._check_of = bp.check_of.astype(np.int64)
        self._col_ptr, self._col_msgs = bp.col_ptr, bp.col_msgs
        self._prior = bp.prior_llr.astype(np.float64)
        # message a far-away, unlit check sends: as after one iteration with every other fault at its prior
        rfix = np.zeros(len(self._var_of))
        for c in range(bp.m):
            a, b = self._row_ptr[c], self._row_ptr[c + 1]
            pri = self._prior[self._var_of[a:b]]
            for k in range(a, b):
                others = np.delete(pri, k - a)
                rfix[k] = min(50.0, scaling * (others.min() if len(others) else 50.0))
        self._rfix = rfix
        Lc = g.L.tocsc()
        self._col_obs = np.array([sum(1 << int(i) for i in Lc.indices[Lc.indptr[j]:Lc.indptr[j + 1]])
                                  for j in range(g.num_mechanisms)], dtype=np.int64)
        self._ce_ptr = np.concatenate([[0], np.cumsum([len(c) for c in g.col_edges])]).astype(np.int64)
        self._ce_idx = np.array([e for c in g.col_edges for e in c], dtype=np.int64)
        self._prod_static = 1.0 - 2.0 * np.clip(g.edge_p, 0.0, 0.5)
        self._cap0 = np.asarray(self.uf.edge_cap, dtype=np.int64)
        self._fallback = None if _nb is not None else BPUnionFindDecoder(g, max_iter, scaling, resolution)
        self.last_iterations = self.last_converged = self.last_active = self.last_uf_rounds = None

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        if self._fallback is not None:  # pragma: no cover - numba missing
            return self._fallback.decode_batch(dets)
        dets = np.ascontiguousarray(np.atleast_2d(dets), dtype=np.bool_)
        u = self.uf
        out, it, conv, nact, ufr = _bpuf_local(
            dets, self._prior, self._row_ptr, self._var_of, self._check_of, self._col_ptr, self._col_msgs,
            self._rfix, self._col_obs, self._ce_ptr, self._ce_idx, self._prod_static, self._cap0,
            float(u.w_ref), float(self.resolution), int(self.radius), int(self.max_iter), float(self.scaling),
            u._eu, u._ev, u._eobs, u._adj_ptr, u._adj_edge, u._adj_nbr, u.boundary)
        self.last_iterations, self.last_converged, self.last_active, self.last_uf_rounds = it, conv, nact, ufr
        return obs_mask_to_bits(out, self.num_observables)
