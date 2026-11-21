"""Belief propagation and the decoders built on it, written from scratch in numpy.

Review §5.3. Three decoders live here:

* `BPDecoder`: min-sum belief propagation on the detector error model's check matrix H.
  Every message update is independent, so one iteration is a handful of array operations over
  all (check, mechanism) pairs of all shots at once. That is the "parallel" part: on hardware,
  each Tanner-graph edge gets its own little processor and an iteration takes constant time.
  On its own, BP is a poor surface-code decoder: the code is highly *degenerate* (many
  equivalent corrections), so the messages often oscillate and never settle on a valid answer.
* `BPOSDDecoder`: when BP does not converge, ordered-statistics decoding (OSD-0) takes BP's
  soft output, sorts mechanisms from most to least likely, and solves H e = s over GF(2) using
  only the most likely independent columns. Always returns a valid correction.
* `BPUnionFindDecoder` ("belief-find"): uses BP's soft output as per-shot edge weights for
  weighted Union-Find. BP sees hyperedges (e.g. the Y-error correlation between the X and Z
  graphs) that the graph decoder can't, and UF keeps the near-linear, local-growth structure.
  Same idea as belief-matching (Higgott et al. 2023), with UF instead of MWPM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from .dem_graph import DemGraph
from .union_find import UnionFindDecoder

try:  # numba is optional; without it we fall back to the (slower) numpy implementation
    import numba as _nb
except ImportError:  # pragma: no cover
    _nb = None

if _nb is not None:
    @_nb.njit(parallel=True, cache=True)
    def _bp_kernel(synd, prior, row_ptr, var_of, col_ptr, col_msgs, max_iter, scaling):
        S, m = synd.shape
        n, nnz = prior.size, var_of.size
        llr_out = np.empty((S, n))
        conv = np.zeros(S, dtype=np.bool_)
        iters = np.zeros(S, dtype=np.int64)
        for s in _nb.prange(S):  # shots are independent: one per core
            q = np.empty(nnz)
            r = np.zeros(nnz)
            total = prior.copy()
            for k in range(nnz):
                q[k] = prior[var_of[k]]
            nonzero = False
            for c in range(m):
                if synd[s, c]:
                    nonzero = True
                    break
            if not nonzero:
                llr_out[s] = total
                conv[s] = True
                continue
            for it in range(1, max_iter + 1):
                # check -> variable: every check is independent (parallel in hardware)
                for c in range(m):
                    a, b = row_ptr[c], row_ptr[c + 1]
                    sgn = synd[s, c]
                    min1 = np.inf
                    min2 = np.inf
                    arg = -1
                    for k in range(a, b):
                        v = q[k]
                        if v < 0:
                            sgn ^= True
                            v = -v
                        if v < min1:
                            min2 = min1
                            min1 = v
                            arg = k
                        elif v < min2:
                            min2 = v
                    for k in range(a, b):
                        mag = min2 if k == arg else min1
                        if mag > 50.0:
                            mag = 50.0
                        if sgn ^ (q[k] < 0):
                            r[k] = -scaling * mag
                        else:
                            r[k] = scaling * mag
                # variable update: every variable is independent too
                for v in range(n):
                    t = prior[v]
                    for idx in range(col_ptr[v], col_ptr[v + 1]):
                        t += r[col_msgs[idx]]
                    total[v] = t
                for k in range(nnz):
                    q[k] = total[var_of[k]] - r[k]
                iters[s] = it
                ok = True
                for c in range(m):
                    par = synd[s, c]
                    for k in range(row_ptr[c], row_ptr[c + 1]):
                        if total[var_of[k]] < 0:
                            par ^= True
                    if par:
                        ok = False
                        break
                if ok:
                    conv[s] = True
                    break
            llr_out[s] = total
        return llr_out, conv, iters



@dataclass
class BPResult:
    error: np.ndarray        # (shots, n) bool hard decision
    llr: np.ndarray          # (shots, n) posterior log-likelihood ratios log(P(no error)/P(error))
    converged: np.ndarray    # (shots,) bool: H @ error == syndrome
    iterations: np.ndarray   # (shots,) int: iterations used (the parallel depth)

    @property
    def posterior(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(np.clip(self.llr, -50, 50)))


class MinSumBP:
    """Normalised min-sum BP for syndrome decoding on a sparse GF(2) matrix.

    Messages live on the nonzeros of H, ordered by check (row). Two backends compute the same thing:
    "numba" (compiled loops, shots spread over CPU cores) and "numpy" (every message of every shot
    updated at once with whole-array operations; slower on a CPU because of temporaries).
    """

    def __init__(self, H, priors: np.ndarray, max_iter: int = 30, scaling: float = 0.625,
                 backend: str = "numba"):
        self.backend = backend if _nb is not None else "numpy"
        H = H.tocsr()
        H.sort_indices()
        self.H = H
        self.m, self.n = H.shape
        self.max_iter = max_iter
        self.scaling = scaling
        self.prior_llr = np.log((1 - priors) / priors)
        row_len = np.diff(H.indptr)
        if (row_len == 0).any():
            raise ValueError("H has an empty row (a detector no error can flip)")
        self.starts = H.indptr[:-1]
        self.check_of = np.repeat(np.arange(self.m), row_len)  # check index for each message
        self.var_of = H.indices.astype(np.int64)                # variable index for each message
        # sparse (nnz, n) matrix that sums messages into their variables
        import scipy.sparse as sp
        nnz = len(self.var_of)
        self.gather = sp.csr_matrix((np.ones(nnz), (np.arange(nnz), self.var_of)), shape=(nnz, self.n))
        self.HT = H.T.tocsr().astype(np.int32)
        # message indices grouped by variable (for the numba kernel's variable update)
        self.col_msgs = np.argsort(self.var_of, kind="stable").astype(np.int64)
        self.col_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.var_of, minlength=self.n))]).astype(np.int64)

    def _syndrome(self, e: np.ndarray) -> np.ndarray:
        return (e.astype(np.int32) @ self.HT) % 2  # (shots, m)

    def run(self, syndrome: np.ndarray) -> BPResult:
        syndrome = np.atleast_2d(syndrome).astype(bool)
        if self.backend == "numba":
            llr, conv, iters = _bp_kernel(syndrome, self.prior_llr, self.H.indptr.astype(np.int64),
                                          self.var_of, self.col_ptr, self.col_msgs,
                                          self.max_iter, float(self.scaling))
            return BPResult(llr < 0, llr, conv, iters)
        return self._run_numpy(syndrome)

    def _run_numpy(self, syndrome: np.ndarray) -> BPResult:
        """Reference implementation: the same algorithm as whole-array operations, which is how
        the per-edge parallelism looks on a GPU. Used to cross-check the numba kernel."""
        S = syndrome.shape[0]
        llr_out = np.tile(self.prior_llr, (S, 1))
        err = np.zeros((S, self.n), dtype=bool)
        conv = np.zeros(S, dtype=bool)
        iters = np.zeros(S, dtype=np.int64)

        # trivial shots: empty syndrome -> no error, converged at iteration 0
        conv[~syndrome.any(axis=1)] = True
        act = np.flatnonzero(~conv)
        q = np.tile(self.prior_llr[self.var_of], (len(act), 1))  # variable -> check messages
        s_act = syndrome[act]
        starts, check_of, var_of = self.starts, self.check_of, self.var_of

        for it in range(1, self.max_iter + 1):
            if len(act) == 0:
                break
            # ---- check -> variable (min-sum), all checks and shots in parallel ----
            a = np.abs(q)
            neg = q < 0
            parity = (np.add.reduceat(neg, starts, axis=1) % 2).astype(bool) ^ s_act
            min1 = np.minimum.reduceat(a, starts, axis=1)
            is_min = a == min1[:, check_of]
            n_min = np.add.reduceat(is_min, starts, axis=1)
            min2 = np.minimum.reduceat(np.where(is_min, np.inf, a), starts, axis=1)
            excl = np.where(is_min & (n_min[:, check_of] == 1), min2[:, check_of], min1[:, check_of])
            excl = np.minimum(excl, 50.0)  # degree-1 checks give inf: cap it
            sign = np.where(parity[:, check_of] ^ neg, -1.0, 1.0)
            r = self.scaling * sign * excl

            # ---- variable update ----
            total = self.prior_llr + (self.gather.T @ r.T).T  # (shots, n)
            q = total[:, var_of] - r

            e = total < 0
            ok = (self._syndrome(e) == s_act).all(axis=1)
            idx = act
            llr_out[idx] = total
            err[idx] = e
            iters[idx] = it
            if ok.any():
                conv[idx[ok]] = True
                keep = ~ok
                act, q, s_act = act[keep], q[keep], s_act[keep]
        return BPResult(err, llr_out, conv, iters)


def _logicals(g: DemGraph, e: np.ndarray) -> np.ndarray:
    return ((e.astype(np.int32) @ g.L.T.astype(np.int32)) % 2).astype(bool)


class BPDecoder:
    """Belief propagation alone. Fast and fully parallel, but often fails to converge on the
    surface code (degeneracy + short loops). Mainly here as the first stage of BP+OSD / BP+UF."""

    def __init__(self, dem: stim.DetectorErrorModel | DemGraph, max_iter: int = 30, scaling: float = 0.625):
        self.graph = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.bp = MinSumBP(self.graph.H, self.graph.priors, max_iter, scaling)
        self.last: BPResult | None = None

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        self.last = self.bp.run(dets)
        return _logicals(self.graph, self.last.error)


class GF2Solver:
    """Ordered-statistics decoding (OSD) over GF(2) with an incremental XOR basis.

    Columns of H are stored as Python ints (bit i = detector i), so XORing two columns is one
    big-int operation. We insert columns from most to least likely; each insert reduces the column
    by the basis built so far, remembering *which original columns* were combined (another int,
    bit j = column j). Once the basis spans the column space of H (size = rank H), reducing the
    syndrome gives OSD-0's correction: the unique solution using only the most-likely independent
    columns ("pivots").

    OSD-0 alone is not distance-preserving (it trusts the ordering completely). Order-w
    *combination sweep* (OSD-CS) also tries adding each of the first `lam` non-pivot columns, and
    pairs among the first `lam2` of them; each such choice fixes the pivots' values, giving another
    valid correction. We keep the one with the smallest total weight sum(log((1-p)/p)).
    """

    def __init__(self, g: DemGraph):
        Hc = g.H.tocsc()
        self.cols = [sum(1 << int(i) for i in Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]])
                     for j in range(Hc.shape[1])]
        Lc = g.L.tocsc()
        self.col_obs = [sum(1 << int(i) for i in Lc.indices[Lc.indptr[j]:Lc.indptr[j + 1]])
                        for j in range(Lc.shape[1])]
        self.weights = np.log((1 - g.priors) / g.priors).tolist()
        self.rank = len(self._basis(range(len(self.cols)), full=True)[0])

    def _basis(self, order, full: bool = False):
        basis: dict[int, tuple[int, int]] = {}  # leading bit -> (vector, columns combined)
        pivots: set[int] = set()
        for j in order:
            v, c = self.cols[j], 1 << j
            while v:
                h = v.bit_length() - 1
                b = basis.get(h)
                if b is None:
                    basis[h] = (v, c)
                    pivots.add(j)
                    break
                v ^= b[0]
                c ^= b[1]
            if not full and len(basis) == self.rank:
                break
        return basis, pivots

    @staticmethod
    def _reduce(basis, v: int) -> int:
        c = 0
        while v:
            v_, c_ = basis[v.bit_length() - 1]  # KeyError would mean v is outside the column space
            v ^= v_
            c ^= c_
        return c

    def _bits(self, mask: int):
        while mask:
            low = mask & -mask
            yield low.bit_length() - 1
            mask ^= low

    def _cost(self, mask: int) -> float:
        w = self.weights
        return sum(w[j] for j in self._bits(mask))

    def _obs(self, mask: int) -> int:
        o = 0
        for j in self._bits(mask):
            o ^= self.col_obs[j]
        return o

    def solve(self, syndrome_bits: np.ndarray, order: np.ndarray, lam: int = 0, lam2: int = 0) -> int:
        """Observable bitmask of the OSD correction for one shot (OSD-0 if lam == lam2 == 0)."""
        s = int.from_bytes(np.packbits(syndrome_bits, bitorder="little").tobytes(), "little")
        order = order.tolist()
        basis, pivots = self._basis(order)
        best = self._reduce(basis, s)
        if lam == 0 and lam2 == 0:
            return self._obs(best)
        best_cost = self._cost(best)
        others = [j for j in order if j not in pivots][:max(lam, lam2)]
        # each non-pivot column t, if flipped, forces the pivots to change by reduce(col t)
        delta = [self._reduce(basis, self.cols[t]) ^ (1 << t) for t in others]
        cands = [best ^ dt for dt in delta[:lam]]
        cands += [best ^ delta[a] ^ delta[b] for a in range(min(lam2, len(delta)))
                  for b in range(a + 1, min(lam2, len(delta)))]
        for cand in cands:
            c = self._cost(cand)
            if c < best_cost:
                best, best_cost = cand, c
        return self._obs(best)


class BPOSDDecoder:
    """BP, then OSD for the shots where BP did not converge (review §5.3).

    Default is OSD-CS with lam = 30 single flips and pairs among the first lam2 = 10 (55 candidates):
    enough to correct every 2-fault combination at d = 5, which OSD-0 (lam = lam2 = 0) misses.
    """

    def __init__(self, dem: stim.DetectorErrorModel | DemGraph, max_iter: int = 30, scaling: float = 0.625,
                 lam: int = 30, lam2: int = 10):
        self.graph = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.bp = MinSumBP(self.graph.H, self.graph.priors, max_iter, scaling)
        self.osd = GF2Solver(self.graph)
        self.lam, self.lam2 = lam, lam2
        self.last: BPResult | None = None

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        res = self.bp.run(dets)
        self.last = res
        out = _logicals(self.graph, res.error)
        no = self.graph.num_observables
        for i in np.flatnonzero(~res.converged):
            order = np.argsort(res.llr[i], kind="stable")  # most likely error first
            mask = self.osd.solve(dets[i], order, self.lam, self.lam2)
            out[i] = [(mask >> k) & 1 for k in range(no)]
        return out


class BPUnionFindDecoder:
    """Belief-find: BP's posteriors become per-shot weights for weighted Union-Find.

    If BP converges we keep its answer (it already satisfies the syndrome); otherwise each
    edge's posterior probability -> weight log((1-p)/p) -> growth steps, and UF runs with them.
    """

    def __init__(self, dem: stim.DetectorErrorModel | DemGraph, max_iter: int = 30,
                 scaling: float = 0.625, resolution: float = 4.0, trust_converged: bool = True):
        self.graph = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.bp = MinSumBP(self.graph.H, self.graph.priors, max_iter, scaling)
        self.uf = UnionFindDecoder(self.graph, weighted=True, resolution=resolution)
        self.trust_converged = trust_converged
        self.last: BPResult | None = None

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        res = self.bp.run(dets)
        self.last = res
        out = _logicals(self.graph, res.error)
        todo = np.flatnonzero(~res.converged) if self.trust_converged else np.flatnonzero(dets.any(axis=1))
        if len(todo) == 0:
            return out
        p_edge = self.graph.edge_probs_from_mechanisms(res.posterior[todo])
        p_edge = np.clip(p_edge, 1e-9, 0.5 - 1e-9)
        w = np.log((1 - p_edge) / p_edge)
        caps = np.maximum(1, np.rint(self.uf.resolution * w / self.uf.w_ref)).astype(np.int64)
        out[todo] = self.uf.decode_batch(dets[todo], edge_caps=caps)
        return out
