"""Sparse Blossom: exact minimum-weight perfect matching grown directly on the detector graph.

Our from-scratch Python implementation of the algorithm behind PyMatching 2 (Higgott & Gidney 2023,
arXiv:2303.15933). Same answers as `MWPMDecoder` (exact MWPM), but it never computes all-pairs
shortest paths or builds the complete graph between lit detectors.

How it works (one shot):

* **Regions.** Every lit detector starts a region with radius 0. Edmonds' dual variables *are* the
  region radii. Growing regions (slope +1) spread over the detector graph's edges at unit speed; a node
  is covered when the region's radius reaches it (exact integer edge weights, so times are exact).
* **Events** go through a priority queue ordered by time:
  - a region reaches an empty node  -> the node joins the region;
  - two regions touch               -> a matching step (below);
  - a region reaches the boundary   -> augment to the boundary;
  - a shrinking region loses a node, a shrinking blossom reaches radius 0 (shatter it), or a
    shrinking single-detector region reaches radius 0 (its neighbours touch through it).
* **Matching steps** are Edmonds' blossom algorithm on the regions:
  - two growing regions from different trees touch -> **augment** (match them, flip both tree paths);
  - a growing region touches a matched pair         -> **grow the alternating tree** (the touched
    region becomes inner and shrinks, its partner becomes outer and grows);
  - two growing regions of the same tree touch      -> an odd cycle: **form a blossom**, which then
    grows as one region;
  - an inner blossom shrinks to radius 0            -> **shatter** it back into its children.
* When nothing grows, every region is matched. Blossoms are expanded to get the pairs of detectors,
  and each pair's correction is a shortest path found by a small local Dijkstra search.

The dual-variable bookkeeping: a covered node v stores `base[v]`, so that its *local radius* (how far
the covering region extends past v) is `base[v] + y(top region of v)`. Only top-level regions change
radius, so blossom formation and shattering only shift `base` by the frozen child radius.
"""
from __future__ import annotations

import heapq

import numpy as np
import stim

from .dem_graph import DemGraph, obs_mask_to_bits

_B = "boundary"  # sentinel partner for regions matched to the boundary
_TOL = 1e-6


class _Region:
    __slots__ = ("y0", "t0", "slope", "shell", "src", "children", "cycle", "bparent", "match", "medge",
                 "tparent", "tedge", "tchildren", "alive")

    def __init__(self, src: int = -1):
        self.y0 = 0.0
        self.t0 = 0.0
        self.slope = 0
        self.shell: list[int] = []        # nodes this region grew into while top-level, in order
        self.src = src                    # detector, for a single-detector region
        self.children = None              # blossom: child regions around an odd cycle
        self.cycle = None                 # blossom: cycle[i] = (detector in children[i], detector in children[i+1])
        self.bparent = None               # enclosing blossom
        self.match = None                 # partner region, _B, or None
        self.medge = None                 # (detector in self, detector in partner or -1)
        self.tparent = None               # alternating-tree parent
        self.tedge = None                 # (detector in self, detector in tparent)
        self.tchildren: list[_Region] = []
        self.alive = True


class SparseBlossomDecoder:
    """Exact MWPM decoder using our Sparse Blossom implementation (pure Python)."""

    def __init__(self, dem: stim.DetectorErrorModel | DemGraph, resolution: int = 2 ** 12):
        g = dem if isinstance(dem, DemGraph) else DemGraph.from_dem(dem)
        self.graph = g
        self.num_observables = g.num_observables
        n = g.num_detectors
        self.n = n
        w = np.maximum(g.edge_weights, 1e-9)
        # Even integer weights (as PyMatching does), so every event happens at an exact time.
        wi = 2 * np.maximum(1, np.rint(w / w.max() * resolution)).astype(np.int64)
        self.nbr: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        self.fadj: list[list[tuple[int, float, int]]] = [[] for _ in range(n + 1)]
        for e, (u, v) in enumerate(zip(g.edge_u.tolist(), g.edge_v.tolist())):
            ww, wf, ob = float(wi[e]), float(w[e]), int(g.edge_obs[e])
            if v == g.boundary:
                self.nbr[u].append((-1, ww))
            else:
                self.nbr[u].append((v, ww))
                self.nbr[v].append((u, ww))
            self.fadj[u].append((v, wf, ob))
            self.fadj[v].append((u, wf, ob))
        self.last_events = 0
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ matching
    def match(self, detection_events: np.ndarray) -> list[tuple[int, int]]:
        """Return the MWPM as (detector, detector) pairs, with -1 meaning the boundary."""
        defects = np.flatnonzero(detection_events).tolist()
        if not defects:
            return []
        n, nbr = self.n, self.nbr
        top: list = [None] * n
        base = [0.0] * n
        src = [-1] * n
        heap: list = []
        seq = 0
        now = 0.0
        trivial: dict[int, _Region] = {}
        stats = {"grow_tree": 0, "augment": 0, "to_boundary": 0, "blossom": 0, "shatter": 0, "zero_radius": 0}

        def y(R):
            return R.y0 + R.slope * (now - R.t0)

        def setslope(R, s):
            R.y0 = y(R)
            R.t0 = now
            R.slope = s

        def local(v):
            return base[v] + y(top[v])

        def nodes_of(R):
            out, stack = [], [R]
            while stack:
                X = stack.pop()
                if X.children is None:
                    out.append(X.src)
                else:
                    stack.extend(X.children)
                out.extend(X.shell)
            return out

        def push(t, kind, a, b):
            nonlocal seq
            seq += 1
            heapq.heappush(heap, (t, seq, kind, a, b))

        def edge_time(a, b, w):
            Ta = top[a]
            if b == -1:
                if Ta is None or Ta.slope <= 0:
                    return None
                return now + (w - local(a)) / Ta.slope
            Tb = top[b]
            if Ta is None and Tb is None:
                return None
            if Ta is None:
                a, b, Ta, Tb = b, a, Tb, Ta
            if Tb is None:
                if Ta.slope <= 0:
                    return None
                return now + (w - local(a)) / Ta.slope
            if Ta is Tb:
                return None
            s = Ta.slope + Tb.slope
            if s <= 0:
                return None
            return now + (w - local(a) - local(b)) / s

        def schedule_node(v):
            for (u, w) in nbr[v]:
                t = edge_time(v, u, w)
                if t is not None:
                    push(max(t, now), 0, v, (u, w))

        def shrink_time(R):
            if not R.alive or R.bparent is not None or R.slope >= 0:
                return None
            if R.shell:
                return now + local(R.shell[-1])
            return now + y(R)

        def schedule_region(R):
            for v in nodes_of(R):
                schedule_node(v)
            t = shrink_time(R)
            if t is not None:
                push(max(t, now), 1, R, None)

        def set_match(A, B, e):
            A.match, A.medge = B, e
            if B is not _B:
                B.match, B.medge = A, (e[1], e[0])

        def root(R):
            while R.tparent is not None:
                R = R.tparent
            return R

        def augment(A):
            cur = A
            while cur.tparent is not None:
                I = cur.tparent
                O = I.tparent
                set_match(I, O, I.tedge)
                cur = O

        def dissolve(rt):
            regs, stack = [], [rt]
            while stack:
                X = stack.pop()
                stack.extend(X.tchildren)
                X.tparent, X.tedge, X.tchildren = None, None, []
                setslope(X, 0)
                regs.append(X)
            for X in regs:
                schedule_region(X)

        def path_up(R):
            out = [R]
            while R.tparent is not None:
                R = R.tparent
                out.append(R)
            return out

        def form_blossom(A, B, e):
            stats["blossom"] += 1
            pa, pb = path_up(A), path_up(B)
            ida = {id(X): i for i, X in enumerate(pa)}
            ib = next(i for i, X in enumerate(pb) if id(X) in ida)
            lca = pb[ib]
            down = list(reversed(pa[:ida[id(lca)] + 1]))  # lca ... A
            up = pb[:ib]                                     # B ... child of lca
            cyc = down + up
            edges = [(down[i + 1].tedge[1], down[i + 1].tedge[0]) for i in range(len(down) - 1)]
            edges.append(e)
            edges.extend(X.tedge for X in up)
            Bl = _Region()
            Bl.children, Bl.cycle = cyc, edges
            Bl.slope, Bl.y0, Bl.t0 = 1, 0.0, now
            Bl.tparent, Bl.tedge = lca.tparent, lca.tedge
            if lca.tparent is not None:
                lst = lca.tparent.tchildren
                lst[next(i for i, X in enumerate(lst) if X is lca)] = Bl
            Bl.match, Bl.medge = lca.match, lca.medge
            if lca.match is not None and lca.match is not _B:
                lca.match.match = Bl
            members = {id(X) for X in cyc}
            for X in cyc:
                for ch in X.tchildren:
                    if id(ch) not in members:
                        ch.tparent = Bl
                        Bl.tchildren.append(ch)
            for X in cyc:
                setslope(X, 0)
                X.bparent = Bl
                X.tparent, X.tedge, X.tchildren = None, None, []
                yx = X.y0
                for v in nodes_of(X):
                    base[v] += yx
                    top[v] = Bl
            schedule_region(Bl)

        def grow_tree(A, B, e):
            stats["grow_tree"] += 1
            Bp = B.match
            B.tparent, B.tedge = A, (e[1], e[0])
            A.tchildren.append(B)
            Bp.tparent, Bp.tedge = B, Bp.medge
            B.tchildren = [Bp]
            setslope(B, -1)
            setslope(Bp, 1)
            schedule_region(B)
            schedule_region(Bp)

        def collide(A, B, e):
            if A.slope < B.slope:
                A, B, e = B, A, (e[1], e[0])
            if B.slope == 1:
                ra, rb = root(A), root(B)
                if ra is rb:
                    form_blossom(A, B, e)
                else:
                    stats["augment"] += 1
                    set_match(A, B, e)
                    augment(A)
                    augment(B)
                    dissolve(ra)
                    dissolve(rb)
            elif B.slope == 0:
                if B.match is _B:
                    stats["augment"] += 1
                    ra = root(A)
                    set_match(A, B, e)
                    augment(A)
                    dissolve(ra)
                    schedule_region(B)
                else:
                    grow_tree(A, B, e)
            else:  # pragma: no cover - outer/inner contact never fires an event
                raise RuntimeError("collision with a shrinking region")

        def to_boundary(A, x):
            stats["to_boundary"] += 1
            ra = root(A)
            set_match(A, _B, (x, -1))
            augment(A)
            dissolve(ra)

        def shatter(Bl):
            stats["shatter"] += 1
            P = Bl.tparent
            xin, yout = Bl.tedge
            C = Bl.tchildren[0]
            yin, zc = Bl.medge
            ch, cyc = Bl.children, Bl.cycle
            m = len(ch)

            def child_of(x):
                X = trivial[x]
                while X.bparent is not Bl:
                    X = X.bparent
                return X

            idx = {id(X): i for i, X in enumerate(ch)}
            ip, ic = idx[id(child_of(xin))], idx[id(child_of(yin))]
            fwd = (ic - ip) % m
            if fwd % 2 == 0:
                path = [(ip + k) % m for k in range(fwd + 1)]
                pedges = [cyc[(ip + k) % m] for k in range(fwd)]
                rest0, nrest = (ic + 1) % m, m - fwd - 1
            else:
                bk = m - fwd
                path = [(ip - k) % m for k in range(bk + 1)]
                pedges = [(cyc[(ip - k - 1) % m][1], cyc[(ip - k - 1) % m][0]) for k in range(bk)]
                rest0, nrest = (ip + 1) % m, m - bk - 1
            for X in ch:
                X.bparent = None
                yx = y(X)
                X.y0, X.t0 = yx, now
                for v in nodes_of(X):
                    base[v] -= yx
                    top[v] = X
            q = [ch[i] for i in path]
            q[0].tparent, q[0].tedge = P, (xin, yout)
            lst = P.tchildren
            lst[next(i for i, X in enumerate(lst) if X is Bl)] = q[0]
            for k in range(len(q) - 1):
                a_, b_ = pedges[k]
                q[k + 1].tparent, q[k + 1].tedge = q[k], (b_, a_)
                q[k].tchildren = [q[k + 1]]
                if k % 2 == 0:
                    set_match(q[k], q[k + 1], (a_, b_))
            last = q[-1]
            last.tchildren = [C]
            set_match(last, C, (yin, zc))
            C.tparent, C.tedge = last, (zc, yin)
            for k, X in enumerate(q):
                setslope(X, -1 if k % 2 == 0 else 1)
            for k in range(0, nrest, 2):
                i1, i2 = (rest0 + k) % m, (rest0 + k + 1) % m
                X, Y = ch[i1], ch[i2]
                set_match(X, Y, cyc[i1])
                for Z in (X, Y):
                    Z.tparent, Z.tedge, Z.tchildren = None, None, []
                    setslope(Z, 0)
            Bl.alive = False
            for X in ch:
                schedule_region(X)

        def trivial_zero(R):
            stats["zero_radius"] += 1
            s = R.src
            P, C = R.tparent, R.tchildren[0]
            p = c = -1
            for (u, w) in nbr[s]:
                if u == -1 or top[u] is None:
                    continue
                if p < 0 and top[u] is P and local(u) >= w - _TOL:
                    p = u
                if c < 0 and top[u] is C and local(u) >= w - _TOL:
                    c = u
            if p < 0 or c < 0:  # pragma: no cover
                raise RuntimeError("degenerate region with no touching neighbours")
            collide(P, C, (src[p], src[c]))

        # ---- initialise: one growing region per lit detector
        for x in defects:
            R = _Region(x)
            R.slope = 1
            trivial[x] = R
            top[x], base[x], src[x] = R, 0.0, x
        for x in defects:
            schedule_node(x)

        events = 0
        while heap:
            t, _, kind, a, b = heapq.heappop(heap)
            if kind == 0:
                u, w = b
                tt = edge_time(a, u, w)
                if tt is None or abs(tt - t) > 1e-6:
                    continue
                now = max(now, t)
                events += 1
                Ta = top[a]
                if u == -1:
                    to_boundary(Ta, src[a])
                elif Ta is None or top[u] is None:
                    if Ta is None:
                        a, u, Ta = u, a, top[u]
                    top[u], base[u], src[u] = Ta, -y(Ta), src[a]
                    Ta.shell.append(u)
                    schedule_node(u)
                else:
                    collide(Ta, top[u], (src[a], src[u]))
            else:
                R = a
                tt = shrink_time(R)
                if tt is None or abs(tt - t) > 1e-6:
                    continue
                now = max(now, t)
                events += 1
                if R.shell:
                    v = R.shell.pop()
                    top[v], src[v] = None, -1
                    for (u, w) in nbr[v]:
                        if u != -1 and top[u] is not None:
                            schedule_node(u)
                    t2 = shrink_time(R)
                    if t2 is not None:
                        push(max(t2, now), 1, R, None)
                elif R.children is not None:
                    shatter(R)
                else:
                    trivial_zero(R)
        self.last_events = events
        self.last_stats = stats

        # ---- expand blossoms into detector pairs
        pairs: list[tuple[int, int]] = []

        def expand(R, x):
            while R.children is not None:
                ch, cyc = R.children, R.cycle
                m = len(ch)
                X = trivial[x]
                while X.bparent is not R:
                    X = X.bparent
                j = next(i for i, Z in enumerate(ch) if Z is X)
                for k in range(1, m, 2):
                    i1, i2 = (j + k) % m, (j + k + 1) % m
                    a_, b_ = cyc[i1]
                    pairs.append((a_, b_))
                    expand(ch[i1], a_)
                    expand(ch[i2], b_)
                R = X

        tops, seen = [], set()
        for R in trivial.values():
            while R.bparent is not None:
                R = R.bparent
            if id(R) not in seen:
                seen.add(id(R))
                tops.append(R)
        done = set()
        for T in tops:
            if id(T) in done:
                continue
            if T.match is None:  # pragma: no cover
                raise RuntimeError("unmatched region at the end")
            x, yv = T.medge
            pairs.append((x, yv))
            expand(T, x)
            done.add(id(T))
            if T.match is not _B:
                expand(T.match, yv)
                done.add(id(T.match))
        return pairs

    # ------------------------------------------------------------------ corrections
    def _path_obs(self, x: int, target: int) -> tuple[int, float]:
        """Local Dijkstra from detector x to `target` (-1 = boundary). Returns (observable mask, length)."""
        B = self.n
        tgt = B if target == -1 else target
        dist = {x: 0.0}
        obs = {x: 0}
        heap = [(0.0, x)]
        while heap:
            d, v = heapq.heappop(heap)
            if v == tgt:
                return obs[v], d
            if d > dist[v]:
                continue
            for (u, w, ob) in self.fadj[v]:
                nd = d + w
                if nd < dist.get(u, float("inf")):
                    dist[u] = nd
                    obs[u] = obs[v] ^ ob
                    heapq.heappush(heap, (nd, u))
        raise RuntimeError("no path")  # pragma: no cover

    def decode(self, detection_events: np.ndarray) -> int:
        mask = 0
        for x, yv in self.match(detection_events):
            mask ^= self._path_obs(x, yv)[0]
        return mask

    def matching_weight(self, detection_events: np.ndarray) -> float:
        return sum(self._path_obs(x, yv)[1] for x, yv in self.match(detection_events))

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.atleast_2d(dets)
        return obs_mask_to_bits(np.array([self.decode(r) for r in dets], dtype=np.int64), self.num_observables)
