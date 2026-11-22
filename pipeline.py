import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
import sinter
from typing import List
import os
import math
import networkx
import heapq
import time
import ctypes
import shutil
import subprocess


class graph_neighbor():
    def __init__(self, nname, prob, obs_mask=0):
        self.nname = nname          # index of the neighbouring node (num_dets = boundary)
        self.prob = prob            # probability that this edge flips
        self.obs_mask = obs_mask    # bitmask of logical observables this edge flips
        self.weight = math.log((1 - prob) / prob)


class sim_graph():
    """Matching graph built from a (decomposed) detector error model.

    Nodes 0..size-1 are detectors, node `size` is the virtual boundary.
    """

    def __init__(self, size, dem):
        self.size = size
        self.boundary = size
        edges = self.parse_input(size, dem)
        self.adj = self.construct_graph(edges, size)
        self.location = dem.get_detector_coordinates()

    def parse_input(self, size, dem):
        edges = {}
        for inst in dem.flattened():
            if inst.type != "error":
                continue  # detector / logical_observable lines carry no edges
            prob = inst.args_copy()[0]
            if prob <= 0:
                continue
            for group in inst.target_groups():
                dets = [t.val for t in group if t.is_relative_detector_id()]
                obs_mask = 0
                for t in group:
                    if t.is_logical_observable_id():
                        obs_mask ^= 1 << t.val
                if len(dets) == 1:
                    key = (dets[0], self.boundary)
                elif len(dets) == 2:
                    key = (min(dets), max(dets))
                else:
                    continue  # not graphlike (shouldn't happen with decompose_errors=True)
                edges.setdefault(key, []).append((prob, obs_mask))
        return edges

    def construct_graph(self, edges, size):
        adj = [[] for _ in range(size + 1)]
        for (u, v), contributions in edges.items():
            # Combine independent mechanisms on the same edge: the edge flips
            # if an odd number of them fire.
            p = 0.0
            for p_i, _ in contributions:
                p = p * (1 - p_i) + p_i * (1 - p)
            if p <= 0 or p >= 0.5:
                continue
            # Parallel mechanisms should agree on observables; take the most likely one's.
            obs_mask = max(contributions)[1]
            adj[u].append(graph_neighbor(v, p, obs_mask))
            adj[v].append(graph_neighbor(u, p, obs_mask))
        return adj


def dijkstra(this_error, errors, adj, boundary):
    """Shortest paths from `this_error` to every node in `errors` and to the boundary.

    Returns {node: (distance, obs_mask_along_path)} for the targets that are reachable.
    The boundary is treated as a sink: paths never pass *through* it.
    """
    targets = set(errors)
    targets.add(boundary)
    targets.discard(this_error)

    dist = {this_error: 0.0}
    done = set()
    found = {}
    pq = [(0.0, this_error, 0)]
    while pq and len(found) < len(targets):
        d, node, obs = heapq.heappop(pq)
        if node in done:
            continue
        done.add(node)
        if node in targets:
            found[node] = (d, obs)
        if node == boundary:
            continue
        for nb in adj[node]:
            nd = d + nb.weight
            if nb.nname not in done and nd < dist.get(nb.nname, math.inf):
                dist[nb.nname] = nd
                heapq.heappush(pq, (nd, nb.nname, obs ^ nb.obs_mask))
    return found


def mwpm(errors, apsp, boundary_dist):
    """Minimum-weight perfect matching on the standard boundary-augmented graph.

    Each defect i gets a boundary copy ('b', i). Edges:
      i -- j        weight apsp[i][j]
      i -- ('b', i) weight boundary_dist[i]
      ('b', i) -- ('b', j) weight 0
    Returns a list of pairs (i, j) of defect indices, with j = None meaning "matched to boundary".
    """
    k = len(errors)
    G = networkx.Graph()
    edges = []
    for i in range(k):
        if boundary_dist[i] < math.inf:
            edges.append((i, ('b', i), boundary_dist[i]))
        for j in range(i + 1, k):
            if apsp[i][j] < math.inf:
                edges.append((i, j, apsp[i][j]))
            edges.append((('b', i), ('b', j), 0.0))
    # Every perfect matching has exactly k edges, so maximising (C - w) among
    # max-cardinality matchings is the same as minimising w.
    C = 1.0 + max((w for _, _, w in edges), default=0.0)
    for u, v, w in edges:
        G.add_edge(u, v, weight=C - w)
    matching = networkx.max_weight_matching(G, maxcardinality=True)

    pairs = []
    for u, v in matching:
        u_real, v_real = isinstance(u, int), isinstance(v, int)
        if u_real and v_real:
            pairs.append((u, v))
        elif u_real:
            pairs.append((u, None))
        elif v_real:
            pairs.append((v, None))
        # boundary-copy <-> boundary-copy edges mean nothing physically
    return pairs


# ---------------------------------------------------------------------------
# Sparse blossom (Higgott & Gidney 2023, arXiv:2303.15933)
#
# Same answer as dijkstra + mwpm above (exact minimum-weight matching), but it
# never computes all-pairs shortest paths. Every lit detector starts a region
# whose radius is its Edmonds dual variable. Regions grow (+1), shrink (-1) or
# are frozen (0); a priority queue says when the next thing happens:
#   * a region reaches an empty node        -> the node joins the region
#   * two regions touch                     -> one of Edmonds' steps:
#       - growing hits growing, other tree  -> augment (match them)
#       - growing hits a frozen matched pair -> the pair joins the tree
#                                              (hit region shrinks, its partner grows)
#       - growing hits growing, same tree   -> odd cycle: form a blossom
#   * a region reaches the boundary         -> match it to the boundary
#   * a shrinking region loses a node, a shrinking blossom hits radius 0
#     (shatter it), or a shrinking single detector hits radius 0 (its
#     neighbours in the tree now touch through it)
# When nothing is growing, every region is matched.
#
# Bookkeeping: a covered node v stores base[v] so that "how far its region
# reaches past v" is base[v] + y(top-level region of v). Only top-level regions
# change radius, so forming/shattering a blossom just shifts base[].
# ---------------------------------------------------------------------------

_BOUNDARY = "boundary"   # partner marker for a region matched to the boundary
_TOL = 1e-6


class _Region:
    __slots__ = ("y0", "t0", "slope", "shell", "src", "children", "cycle", "bparent", "match",
                 "medge", "tparent", "tedge", "tchildren", "alive")

    def __init__(self, src=-1):
        self.y0 = 0.0             # radius at time t0
        self.t0 = 0.0
        self.slope = 0            # +1 growing, -1 shrinking, 0 frozen
        self.shell = []           # nodes this region grew into (in order) while top-level
        self.src = src            # the detector, for a single-detector region
        self.children = None      # blossom: child regions around the odd cycle
        self.cycle = None         # blossom: cycle[i] = (detector in children[i], detector in children[i+1])
        self.bparent = None       # blossom that contains this region
        self.match = None         # partner region, _BOUNDARY, or None
        self.medge = None         # (detector in self, detector in partner or -1)
        self.tparent = None       # alternating-tree parent
        self.tedge = None         # (detector in self, detector in tparent)
        self.tchildren = []
        self.alive = True


def integer_adjacency(graph, resolution=2 ** 12):
    """Neighbour lists for sparse blossom: [(neighbour or -1 for boundary, weight)].

    Like PyMatching, weights are rescaled to even integers so every event happens
    at an exact time (no floating-point near-misses between colliding regions).
    """
    wmax = max((nb.weight for nbs in graph.adj[:graph.size] for nb in nbs), default=1.0)
    nbr = []
    for u in range(graph.size):
        nbr.append([(-1 if nb.nname == graph.boundary else nb.nname,
                     float(2 * max(1, round(nb.weight / wmax * resolution))))
                    for nb in graph.adj[u]])
    return nbr


def sparse_blossom_match(defects, nbr, n):
    """Minimum-weight matching of the lit detectors `defects`.

    Returns (detector, detector) pairs; the second entry is -1 for "matched to boundary".
    """
    top = [None] * n          # top-level region covering each node
    base = [0.0] * n
    src = [-1] * n            # which lit detector's region reached each node
    heap = []
    seq = 0
    now = 0.0
    trivial = {}              # lit detector -> its single-detector region

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
        """When will the edge a-b (b = -1: boundary) be fully covered?"""
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
        if B is not _BOUNDARY:
            B.match, B.medge = A, (e[1], e[0])

    def root(R):
        while R.tparent is not None:
            R = R.tparent
        return R

    def augment(A):
        """Flip matched/unmatched along the tree path from A up to its root."""
        cur = A
        while cur.tparent is not None:
            inner = cur.tparent
            outer = inner.tparent
            set_match(inner, outer, inner.tedge)
            cur = outer

    def dissolve(rt):
        """Break up a whole alternating tree: everything becomes a frozen matched pair."""
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
        pa, pb = path_up(A), path_up(B)
        ida = {id(X): i for i, X in enumerate(pa)}
        ib = next(i for i, X in enumerate(pb) if id(X) in ida)
        lca = pb[ib]
        down = list(reversed(pa[:ida[id(lca)] + 1]))   # lca ... A
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
        if lca.match is not None and lca.match is not _BOUNDARY:
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
        partner = B.match
        B.tparent, B.tedge = A, (e[1], e[0])
        A.tchildren.append(B)
        partner.tparent, partner.tedge = B, partner.medge
        B.tchildren = [partner]
        setslope(B, -1)
        setslope(partner, 1)
        schedule_region(B)
        schedule_region(partner)

    def collide(A, B, e):
        if A.slope < B.slope:
            A, B, e = B, A, (e[1], e[0])
        if B.slope == 1:
            ra, rb = root(A), root(B)
            if ra is rb:
                form_blossom(A, B, e)
            else:
                set_match(A, B, e)
                augment(A)
                augment(B)
                dissolve(ra)
                dissolve(rb)
        elif B.slope == 0:
            if B.match is _BOUNDARY:
                ra = root(A)
                set_match(A, B, e)
                augment(A)
                dissolve(ra)
                schedule_region(B)
            else:
                grow_tree(A, B, e)
        else:
            raise RuntimeError("collision with a shrinking region")

    def to_boundary(A, x):
        ra = root(A)
        set_match(A, _BOUNDARY, (x, -1))
        augment(A)
        dissolve(ra)

    def shatter(Bl):
        """A shrinking blossom reached radius 0: put its children back into the tree."""
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
        """A shrinking single detector hit radius 0: its tree parent and child touch through it."""
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
        if p < 0 or c < 0:
            raise RuntimeError("degenerate region with no touching neighbours")
        collide(P, C, (src[p], src[c]))

    # ---- start: one growing region per lit detector
    for x in defects:
        R = _Region(x)
        R.slope = 1
        trivial[x] = R
        top[x], base[x], src[x] = R, 0.0, x
    for x in defects:
        schedule_node(x)

    # ---- event loop (stale events are skipped by re-checking their time)
    while heap:
        t, _, kind, a, b = heapq.heappop(heap)
        if kind == 0:                                    # an edge became fully covered
            u, w = b
            tt = edge_time(a, u, w)
            if tt is None or abs(tt - t) > _TOL:
                continue
            now = max(now, t)
            Ta = top[a]
            if u == -1:
                to_boundary(Ta, src[a])
            elif Ta is None or top[u] is None:           # grew into an empty node
                if Ta is None:
                    a, u, Ta = u, a, top[u]
                top[u], base[u], src[u] = Ta, -y(Ta), src[a]
                Ta.shell.append(u)
                schedule_node(u)
            else:
                collide(Ta, top[u], (src[a], src[u]))
        else:                                            # a shrinking region hit something
            R = a
            tt = shrink_time(R)
            if tt is None or abs(tt - t) > _TOL:
                continue
            now = max(now, t)
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

    # ---- expand blossoms into detector pairs
    pairs = []

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
        if T.match is None:
            raise RuntimeError("unmatched region at the end")
        x, yv = T.medge
        pairs.append((x, yv))
        expand(T, x)
        done.add(id(T))
        if T.match is not _BOUNDARY:
            expand(T.match, yv)
            done.add(id(T.match))
    return pairs


def path_obs(x, target, adj, boundary):
    """Observable mask along a shortest path x -> target (never passing through the boundary)."""
    dist = {x: 0.0}
    done = set()
    pq = [(0.0, x, 0)]
    while pq:
        d, node, obs = heapq.heappop(pq)
        if node in done:
            continue
        done.add(node)
        if node == target:
            return obs
        if node == boundary:
            continue
        for nb in adj[node]:
            nd = d + nb.weight
            if nb.nname not in done and nd < dist.get(nb.nname, math.inf):
                dist[nb.nname] = nd
                heapq.heappush(pq, (nd, nb.nname, obs ^ nb.obs_mask))
    raise RuntimeError("no path")


def dijkstra_all(source, adj, boundary):
    """Full single-source Dijkstra (no early stop), boundary treated as a sink.

    Returns numpy arrays (dist, obs) over every node, so the result can be cached and
    reused for any later set of targets.
    """
    n = len(adj)
    dist = [math.inf] * n
    obs = [0] * n
    done = [False] * n
    dist[source] = 0.0
    pq = [(0.0, source, 0)]
    while pq:
        d, node, o = heapq.heappop(pq)
        if done[node]:
            continue
        done[node] = True
        obs[node] = o
        if node == boundary:
            continue
        for nb in adj[node]:
            nd = d + nb.weight
            if not done[nb.nname] and nd < dist[nb.nname]:
                dist[nb.nname] = nd
                heapq.heappush(pq, (nd, nb.nname, o ^ nb.obs_mask))
    return np.array(dist), np.array(obs, dtype=np.int64)


def direct_edge_obs(graph):
    """{(u, v): obs_mask} for every edge that is guaranteed to be a shortest path.

    Any path of two or more edges weighs at least 2 * (lightest edge), so an edge no
    heavier than that is always a shortest path between its endpoints. (At p = 0.1%
    circuit noise that is every edge: weights run ~5.2 to ~8.2.)
    """
    weights = [nb.weight for nbs in graph.adj for nb in nbs]
    limit = 2 * min(weights, default=0.0)
    table = {}
    for u in range(graph.size):
        for nb in graph.adj[u]:
            if nb.weight <= limit:
                table[(min(u, nb.nname), max(u, nb.nname))] = nb.obs_mask
    return table


# ---------------------------------------------------------------------------
# Compiled sparse blossom: blossom.cpp (a C++ port of sparse_blossom_match), built
# into blossom.dll with MinGW g++ and called through ctypes. No Python headers or
# Visual Studio needed; the DLL only exposes plain C functions.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_BLOSSOM_SRC = os.path.join(_HERE, "blossom.cpp")
_BLOSSOM_DLL = os.path.join(_HERE, "blossom.dll")
_BLOSSOM_LIBS = {}   # dll path -> loaded library


def build_blossom(src=_BLOSSOM_SRC, dll=_BLOSSOM_DLL, flags=("-O2",)):
    """Compile a sparse blossom C++ source into a DLL with MinGW g++."""
    gxx = shutil.which("g++") or r"C:\msys64\mingw64\bin\g++.exe"
    env = dict(os.environ, PATH=os.path.dirname(gxx) + os.pathsep + os.environ.get("PATH", ""))
    subprocess.run([gxx, *flags, "-std=c++17", "-shared", "-static", "-o", dll, src],
                   check=True, env=env)


def load_compiled_blossom(dll=_BLOSSOM_DLL):
    """Load a blossom DLL. The default one is (re)built from blossom.cpp if missing or stale."""
    if dll in _BLOSSOM_LIBS:
        return _BLOSSOM_LIBS[dll]
    if dll == _BLOSSOM_DLL and (not os.path.exists(_BLOSSOM_DLL)
                                or os.path.getmtime(_BLOSSOM_DLL) < os.path.getmtime(_BLOSSOM_SRC)):
        build_blossom()
    lib = ctypes.CDLL(dll)
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    u64p = np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    lib.sb_create.argtypes = [ctypes.c_int, ctypes.c_int, i32p, i32p, f64p, f64p, u64p]
    lib.sb_create.restype = ctypes.c_void_p
    lib.sb_destroy.argtypes = [ctypes.c_void_p]
    lib.sb_decode_batch.argtypes = [ctypes.c_void_p, u8p, ctypes.c_int, ctypes.c_int, u64p]
    lib.sb_decode_batch.restype = ctypes.c_int
    lib.sb_last_error.restype = ctypes.c_char_p
    if hasattr(lib, "sb_decode_batch_mt"):          # (older step builds don't have it)
        lib.sb_decode_batch_mt.argtypes = [ctypes.c_void_p, u8p, ctypes.c_int, ctypes.c_int, u64p, ctypes.c_int]
        lib.sb_decode_batch_mt.restype = ctypes.c_int
    if hasattr(lib, "sb_set_correlations"):
        lib.sb_set_correlations.argtypes = [ctypes.c_void_p, ctypes.c_int, i32p, i32p, f64p]
        lib.sb_set_correlations.restype = ctypes.c_int
    _BLOSSOM_LIBS[dll] = lib
    return lib


def compiled_edge_arrays(graph, resolution=2 ** 12):
    """The graph as flat edge arrays for sb_create: one entry per edge (boundary = -1),
    with integer matching weights (as integer_adjacency) and float weights / obs masks."""
    eu, ev, wf, ob = [], [], [], []
    for u in range(graph.size):
        for nb in graph.adj[u]:
            if nb.nname == graph.boundary or u < nb.nname:
                eu.append(u)
                ev.append(-1 if nb.nname == graph.boundary else nb.nname)
                wf.append(nb.weight)
                ob.append(nb.obs_mask)
    wf = np.array(wf, dtype=np.float64)
    wmax = wf.max() if len(wf) else 1.0
    wint = (2 * np.maximum(1, np.rint(wf / wmax * resolution))).astype(np.float64)
    return (len(eu), np.array(eu, np.int32), np.array(ev, np.int32), wint, wf,
            np.array(ob, np.uint64))


def compiled_correlation_rules(graph, dem, resolution=2 ** 12):
    """Implied-weight rules for correlated matching, following PyMatching's definitions.

    For every pair of edges (components) e, f of the same decomposed error mechanism:
        joint[e][f] = XOR-accumulated probability of the mechanisms containing both,
        joint[e][e] = XOR-accumulated probability of every mechanism containing e.
    If e is used by the first matching, f may be lowered to the weight of
        q(f|e) = min(0.5, joint[e][f] / joint[e][e]),   w = log((1 - q) / q).
    Returns (nrules, rule_e, rule_f, rule_w) with edge indices as in compiled_edge_arrays and
    weights in the same integer scale.
    """
    keys = []                                   # same order as compiled_edge_arrays
    for u in range(graph.size):
        for nb in graph.adj[u]:
            if nb.nname == graph.boundary or u < nb.nname:
                keys.append((u, nb.nname))
    index = {k: i for i, k in enumerate(keys)}
    wmax = max((nb.weight for u in range(graph.size) for nb in graph.adj[u]), default=1.0)

    def bxor(a, b):
        return a * (1 - b) + b * (1 - a)

    joint = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        if p <= 0:
            continue
        comps = []
        for group in inst.target_groups():
            dets = [t.val for t in group if t.is_relative_detector_id()]
            if len(dets) == 1:
                comps.append((dets[0], graph.boundary))
            elif len(dets) == 2:
                comps.append((min(dets), max(dets)))
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i], comps[j]
                row_a, row_b = joint.setdefault(a, {}), joint.setdefault(b, {})
                row_a[b] = bxor(row_a.get(b, 0.0), p)
                row_b[a] = bxor(row_b.get(a, 0.0), p)
        for c in comps:
            row = joint.setdefault(c, {})
            row[c] = bxor(row.get(c, 0.0), p)

    re, rf, rw = [], [], []
    for e, row in joint.items():
        marginal = row.get(e, 0.0)
        if marginal <= 0 or e not in index:
            continue
        for f, pj in row.items():
            if f == e or f not in index:
                continue
            q = min(0.5, pj / marginal)
            if q <= 0:
                continue
            w = math.log((1 - q) / q)
            re.append(index[e])
            rf.append(index[f])
            rw.append(2.0 * max(1.0, float(np.rint(w / wmax * resolution))))
    return (len(re), np.array(re, np.int32), np.array(rf, np.int32), np.array(rw, np.float64))


class MyCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, dem: stim.DetectorErrorModel, method="apsp",
                 direct_edge=True, pair_cache=False, source_cache=False, threads=1,
                 correlated=False):
        """method="apsp":    Dijkstra from every lit detector + networkx matching.
                             source_cache: keep each detector's full Dijkstra across shots.
        method="blossom": sparse blossom matching, then each pair's obs parity from
                             direct_edge: the single edge joining the pair, if there is one;
                             pair_cache:  a remembered earlier search for that pair;
                             otherwise a fresh shortest-path search.
        method="compiled": the same sparse blossom + direct edge, in C++ (blossom.cpp).
                             threads: split each batch over this many C++ threads. Leave at 1
                             under sinter, which already runs one worker process per core.
                             correlated: two-pass correlated matching, as PyMatching's
                             enable_correlations=True (see compiled_correlation_rules).
        """
        self.threads = threads
        self.correlated = correlated
        self.num_dets = dem.num_detectors
        self.num_obs = dem.num_observables
        self.dem = dem
        self.graph = sim_graph(self.num_dets, dem)
        self.counts = {}   # how each lookup was answered (for the ablation)
        self.method = method
        self._handle = None
        if method == "compiled":
            self._init_compiled()
            self.decode = self.decode_one_compiled
        elif method == "blossom":
            self.int_adj = integer_adjacency(self.graph)
            self.edge_obs = direct_edge_obs(self.graph) if direct_edge else None
            self.pair_cache = {} if pair_cache else None
            self.decode = self.decode_one_blossom
        else:
            self.source_cache = {} if source_cache else None
            self.decode = self.decode_one

    def _init_compiled(self, dll=_BLOSSOM_DLL):
        # Hand the C++ side one entry per edge, with the same integer weights as
        # integer_adjacency() and the float weights / obs masks of sim_graph.
        self._lib = load_compiled_blossom(dll)
        self._handle = self._lib.sb_create(self.num_dets, *compiled_edge_arrays(self.graph))
        if not self._handle:
            raise RuntimeError(self._lib.sb_last_error().decode())
        if self.correlated:
            rules = compiled_correlation_rules(self.graph, self.dem)
            if self._lib.sb_set_correlations(self._handle, *rules) != 0:
                raise RuntimeError(self._lib.sb_last_error().decode())

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.sb_destroy(self._handle)
            self._handle = None

    def decode_masks_compiled(self, bit_packed: np.ndarray) -> np.ndarray:
        """Observable masks (uint64, one per shot) for bit-packed shots, decoded in C++."""
        data = np.ascontiguousarray(bit_packed, dtype=np.uint8)
        out = np.zeros(data.shape[0], dtype=np.uint64)
        if self.threads > 1:
            rc = self._lib.sb_decode_batch_mt(self._handle, data, data.shape[0], data.shape[1], out, self.threads)
        else:
            rc = self._lib.sb_decode_batch(self._handle, data, data.shape[0], data.shape[1], out)
        if rc != 0:
            raise RuntimeError(self._lib.sb_last_error().decode())
        return out

    def decode_one_compiled(self, shot: np.ndarray) -> np.ndarray:
        mask = int(self.decode_masks_compiled(np.packbits(shot, bitorder="little")[None, :])[0])
        return np.array([(mask >> b) & 1 for b in range(self.num_obs)], dtype=bool)

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,   # uint8, shape (shots, ceil(num_dets / 8))
    ) -> np.ndarray:                                    # uint8, shape (shots, ceil(num_obs / 8))
        if self.method == "compiled":
            # the whole batch goes to C++ in one call
            masks = self.decode_masks_compiled(bit_packed_detection_event_data)
            pred = ((masks[:, None] >> np.arange(self.num_obs, dtype=np.uint64)) & np.uint64(1)).astype(bool)
            return np.packbits(pred, axis=1, bitorder="little")

        # unpack to bool: shape (shots, num_dets)
        dets = np.unpackbits(bit_packed_detection_event_data,
                             axis=1, bitorder="little")[:, :self.num_dets].astype(bool)

        pred = np.zeros((dets.shape[0], self.num_obs), dtype=bool)
        for i, shot in enumerate(dets):
            pred[i] = self.decode(shot)

        return np.packbits(pred, axis=1, bitorder="little")

    def decode_one(self, shot: np.ndarray) -> np.ndarray:
        # shot: bool array of length num_dets (True = lit detector)
        # return: bool array of length num_obs (True = flip that logical)
        errors = np.flatnonzero(shot).tolist()
        k = len(errors)
        if k == 0:
            return np.zeros(self.num_obs, dtype=bool)

        adj = self.graph.adj
        boundary = self.graph.boundary

        apsp = np.full((k, k), math.inf)
        apsp_obs = np.zeros((k, k), dtype=np.int64)
        boundary_dist = np.full(k, math.inf)
        boundary_obs = np.zeros(k, dtype=np.int64)
        index = {e: idx for idx, e in enumerate(errors)}

        cache = self.source_cache
        for i, e in enumerate(errors):
            if cache is not None:
                # Same distances as the early-stopping search below, but computed for
                # the whole graph once per detector and then reused by every later shot.
                if e in cache:
                    self.counts["cached source"] = self.counts.get("cached source", 0) + 1
                else:
                    self.counts["new source"] = self.counts.get("new source", 0) + 1
                    cache[e] = dijkstra_all(e, adj, boundary)
                dist, obs = cache[e]
                apsp[i] = dist[errors]
                apsp_obs[i] = obs[errors]
                apsp[i][i] = math.inf
                boundary_dist[i] = dist[boundary]
                boundary_obs[i] = obs[boundary]
                continue
            found = dijkstra(e, errors, adj, boundary)
            for node, (d, obs) in found.items():
                if node == boundary:
                    boundary_dist[i] = d
                    boundary_obs[i] = obs
                else:
                    j = index[node]
                    apsp[i][j] = d
                    apsp_obs[i][j] = obs

        obs_flip = 0
        for i, j in mwpm(errors, apsp, boundary_dist):
            obs_flip ^= int(boundary_obs[i]) if j is None else int(apsp_obs[i][j])
        return np.array([(obs_flip >> b) & 1 for b in range(self.num_obs)], dtype=bool)

    def decode_one_blossom(self, shot: np.ndarray) -> np.ndarray:
        # Same output as decode_one, but the matching comes from sparse blossom.
        # Sparse blossom only says *which* detectors pair up, so the obs parity
        # of each pair is read off one shortest path afterwards.
        errors = np.flatnonzero(shot).tolist()
        obs_flip = 0
        boundary = self.graph.boundary
        counts = self.counts
        for x, y in sparse_blossom_match(errors, self.int_adj, self.num_dets):
            target = boundary if y == -1 else y
            key = (min(x, target), max(x, target))    # paths are symmetric
            if self.edge_obs is not None and key in self.edge_obs:
                how, o = "direct edge", self.edge_obs[key]
            elif self.pair_cache is not None and key in self.pair_cache:
                how, o = "cached pair", self.pair_cache[key]
            else:
                how, o = "search", path_obs(x, target, self.graph.adj, boundary)
                if self.pair_cache is not None:
                    self.pair_cache[key] = o
            counts[how] = counts.get(how, 0) + 1
            obs_flip ^= o
        return np.array([(obs_flip >> b) & 1 for b in range(self.num_obs)], dtype=bool)


class MyDecoder(sinter.Decoder):
    def __init__(self, method="apsp", **options):
        self.method = method      # "apsp" (dijkstra + networkx), "blossom" (sparse blossom) or "compiled" (C++)
        self.options = options    # direct_edge / pair_cache / source_cache, see MyCompiledDecoder
        if method == "compiled":
            load_compiled_blossom()   # build the DLL once here, before sinter starts its workers

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return MyCompiledDecoder(dem, method=self.method, **self.options)


def main():
    a = run_sinter_benchmark(drange=[3, 5])
    analyze_surface_code_stats(a)
    compare_decoders(a)


# Benchmarking
def make_circuit(d, noise):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=d * 3,
        distance=d,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )


def run_sinter_benchmark(mydecfn=None, noise=1e-3, drange=[3, 5, 7, 9],
                         max_shots=2_000, max_errors=100, fast_max_shots=1_000_000):
    """Sample logical error rates with sinter. The Python decoders get max_shots; the fast ones
    (C++ and PyMatching) get fast_max_shots, so their error rates have usable error bars."""
    if mydecfn is None:
        mydecfn = MyDecoder()
    surface_code_tasks = [
        sinter.Task(
            circuit=make_circuit(d, noise),
            json_metadata={'d': d, 'r': d * 3, 'p': noise},
        )
        for d in drange
    ]
    custom = {"mine": mydecfn, "mine_blossom": MyDecoder(method="blossom"),
              "mine_compiled": MyDecoder(method="compiled"),
              "mine_compiled_corr": MyDecoder(method="compiled", correlated=True),
              "pymatching_corr": PyMatchingCorrelated()}

    collected_surface_code_stats: List[sinter.TaskStats] = []
    for decoders, shots in ((["mine", "mine_blossom"], max_shots),
                            (["mine_compiled", "mine_compiled_corr", "pymatching", "pymatching_corr"],
                             fast_max_shots)):
        collected_surface_code_stats += sinter.collect(
            num_workers=os.cpu_count(),
            tasks=surface_code_tasks,
            decoders=decoders,
            custom_decoders=custom,
            max_shots=shots,
            max_errors=max_errors,
            print_progress=False,
        )
    return collected_surface_code_stats


def analyze_surface_code_stats(collected_surface_code_stats):
    for stats in sorted(collected_surface_code_stats,
                        key=lambda s: (s.json_metadata['d'], s.decoder)):
        d = stats.json_metadata['d']
        if not stats.errors:
            print(f"[{stats.decoder}] Didn't see any errors for d={d} ({stats.shots} shots)")
            continue
        per_shot = stats.errors / stats.shots
        per_round = sinter.shot_error_rate_to_piece_error_rate(per_shot, pieces=stats.json_metadata['r'])
        print(f"[{stats.decoder}] d:{d}, shots {stats.shots}, errors {stats.errors}, per round {per_round:.3e}")


class PyMatchingCorrelated(sinter.Decoder):
    """PyMatching with enable_correlations=True, as a sinter decoder."""
    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _PyMatchingCorrelatedCompiled(dem)


class _PyMatchingCorrelatedCompiled(sinter.CompiledDecoder):
    def __init__(self, dem):
        self.pm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        return self.pm.decode_batch(bit_packed_detection_event_data, bit_packed_shots=True,
                                    bit_packed_predictions=True, enable_correlations=True)


# name: (label, color, marker, linestyle, MyCompiledDecoder options, or None for pymatching).
# Colour = algorithm family, so the Python and C++ sparse blossom share green; dotted = correlated.
DECODER_STYLE = {
    "mine":               ("mine (Dijkstra + networkx)",    "#2a78d6", "o", "-",  dict(method="apsp")),
    "mine_blossom":       ("mine (sparse blossom, Python)", "#1baf7a", "D", "--", dict(method="blossom")),
    "mine_compiled":      ("mine (sparse blossom, C++)",    "#1baf7a", "P", "-",  dict(method="compiled")),
    "mine_compiled_corr": ("mine (C++, correlated)",        "#1baf7a", "X", ":",
                           dict(method="compiled", correlated=True)),
    "pymatching":         ("pymatching",                    "#eb6834", "s", "-",  None),
    "pymatching_corr":    ("pymatching (correlated)",       "#eb6834", "^", ":",  None),
}


def _reference(name):
    """The pymatching variant a decoder of ours is compared against."""
    return "pymatching_corr" if DECODER_STYLE[name][4].get("correlated") else "pymatching"


def compare_decoders(collected_surface_code_stats, timing_shots=1_000, timing_repeats=7,
                     out_path="decoder_comparison.png", min_plot_shots=10_000):
    """Summary table + plots: mine (every matching method) vs pymatching.

    Accuracy comes from the sinter stats (binomial intervals at likelihood ratio 1000,
    i.e. wider than 95%, converted to per-round rates); the accuracy panel leaves out decoders
    with fewer than min_plot_shots shots (the slow Python ones), whose error bars say nothing;
    their predictions are compared with pymatching's on the timing shots instead. Timing re-decodes the *same*
    freshly sampled shots with every decoder on one core, so it measures decoding only
    (no sampling, no worker overhead), and reports how often each of ours agrees with
    pymatching on those shots (correlated variants against pymatching's correlated mode).
    """
    by_key = {(s.json_metadata['d'], s.decoder): s for s in collected_surface_code_stats}
    ds = sorted({d for d, _ in by_key})
    decoders = [n for n in DECODER_STYLE if any(dec == n for _, dec in by_key)]

    # --- accuracy ---
    print(f"\n{'d':>3} {'decoder':>13} {'shots':>8} {'errors':>6} {'LER/round':>10}  {'interval (per round)':>23}")
    ler = {}
    for d in ds:
        for dec in decoders:
            s = by_key.get((d, dec))
            if s is None:
                continue
            fit = sinter.fit_binomial(num_shots=s.shots, num_hits=s.errors, max_likelihood_factor=1000)
            lo, best, hi = (sinter.shot_error_rate_to_piece_error_rate(x, pieces=s.json_metadata['r'])
                            for x in (fit.low, fit.best, fit.high))
            ler[(d, dec)] = (lo, best, hi)
            print(f"{d:>3} {dec:>13} {s.shots:>8} {s.errors:>6} {best:>10.2e}  [{lo:.2e}, {hi:.2e}]")

    # --- timing on identical shots ---
    ours = [n for n, st in DECODER_STYLE.items() if st[4] is not None]
    times = {name: [] for name in DECODER_STYLE}
    print(f"\n{'d':>3} {'events/shot':>11} " + " ".join(f"{n + ' us':>18}" for n in DECODER_STYLE)
          + "   time / pymatching's, agreement with it")
    for d in ds:
        circuit = make_circuit(d, by_key[(d, decoders[0])].json_metadata['p'])
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, _ = circuit.compile_detector_sampler(seed=d).sample(
            timing_shots, separate_observables=True, bit_packed=True)
        decode = {n: MyCompiledDecoder(dem, **DECODER_STYLE[n][4]).decode_shots_bit_packed for n in ours}
        for name, corr in (("pymatching", False), ("pymatching_corr", True)):
            pm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=corr)
            decode[name] = (lambda pm, corr: lambda bit_packed_detection_event_data: pm.decode_batch(
                bit_packed_detection_event_data, bit_packed_shots=True, bit_packed_predictions=True,
                enable_correlations=corr))(pm, corr)

        preds = {}
        for name, fn in decode.items():
            fn(bit_packed_detection_event_data=dets[:20])      # warm-up: first-call setup is not timed
            slow = name in ("mine", "mine_blossom")
            runs = []
            for _ in range(1 if slow else timing_repeats):     # median of several runs for the fast ones
                t0 = time.perf_counter()
                preds[name] = fn(bit_packed_detection_event_data=dets)
                runs.append((time.perf_counter() - t0) / timing_shots * 1e6)
            times[name].append(float(np.median(runs)))

        events = np.unpackbits(dets, axis=1, bitorder="little")[:, :dem.num_detectors].sum(1).mean()
        agree = {n: np.all(preds[n] == preds[_reference(n)], axis=1).mean() for n in ours}
        print(f"{d:>3} {events:>11.1f} " + " ".join(f"{times[n][-1]:>18.2f}" for n in DECODER_STYLE) + "   "
              + "  ".join(f"{n}: {times[n][-1] / times[_reference(n)][-1]:.2f}x/{agree[n]:.2%}"
                          for n in ours))

    # --- plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    plotted = [dec for dec in decoders
               if all(by_key[(d, dec)].shots >= min_plot_shots for d in ds if (d, dec) in by_key)]
    for k, dec in enumerate(plotted):
        label, color, marker, ls, _ = DECODER_STYLE[dec]
        offset = (k - (len(plotted) - 1) / 2) * 0.1
        seen = [(d, *ler[(d, dec)]) for d in ds if (d, dec) in ler and ler[(d, dec)][1] > 0]
        unseen = [(d, ler[(d, dec)][2]) for d in ds if (d, dec) in ler and ler[(d, dec)][1] == 0]
        if seen:
            x, lo, best, hi = map(np.array, zip(*seen))
            ax1.errorbar(x + offset, best, yerr=[best - lo, hi - best], fmt=marker, ls=ls,
                         ms=7, lw=2, capsize=4, color=color, label=label)
        if unseen:  # zero errors observed: only an upper bound is known
            x, hi = map(np.array, zip(*unseen))
            ax1.plot(x + offset, hi, "v", ms=9, mfc="none", mew=2, color=color,
                     label=None if seen else label)
    if any(ler[(d, dec)][1] == 0 for d in ds for dec in plotted if (d, dec) in ler):
        ax1.plot([], [], "v", ms=9, mfc="none", mew=2, color="#52514e", label="0 errors seen (upper bound)")
    ax1.set_yscale("log")
    ax1.set_ylabel("logical error rate per round")
    ax1.set_title(f"Accuracy (bars = likelihood interval; up to {max(s.shots for s in collected_surface_code_stats):,}"
                  " shots per point)\nPython decoders not shown: too few shots for useful bars", fontsize=10)

    for name, (label, color, marker, ls, _) in DECODER_STYLE.items():
        ax2.plot(ds, times[name], marker=marker, ls=ls, ms=7, lw=2, color=color, label=label)
    for name in ours:
        opts = DECODER_STYLE[name][4]
        if opts["method"] != "compiled":
            pos = dict(xytext=(0, 7), ha="center", va="bottom")
        elif opts.get("correlated"):   # the C++ lines run close to pymatching's: left / below-right
            pos = dict(xytext=(-8, 2), ha="right", va="bottom")
        else:
            pos = dict(xytext=(7, -7), ha="left", va="top")
        for d, a, b in zip(ds, times[name], times[_reference(name)]):
            r = a / b
            ax2.annotate(f"{r:.0f}x" if r >= 10 else (f"{r:.1f}x" if r >= 1 else f"{r:.2f}x"), (d, a),
                         textcoords="offset points", fontsize=8.5, color="#52514e", **pos)
    ax2.set_yscale("log")
    all_t = [t for ts in times.values() for t in ts]
    ax2.set_ylim(min(all_t) / 3, max(all_t) * 3)   # room for the labels at both ends
    ax2.set_ylabel("decode time per shot (us, 1 core)")
    ax2.set_title(f"Speed ({timing_shots} identical shots, median of {timing_repeats} runs)\n"
                  "labels = time / pymatching's (correlated vs pymatching correlated)", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xticks(ds)
        ax.set_xlabel("code distance d")
        ax.set_xlim(min(ds) - 0.6, max(ds) + (0.9 if ax is ax2 else 0.6))   # room for right-hand labels
        ax.grid(True, which="major", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")
    return fig


ABLATION_VARIANTS = {   # name: MyCompiledDecoder options
    "networkx":                       dict(method="apsp"),
    "networkx + source cache":        dict(method="apsp", source_cache=True),
    "blossom":                        dict(method="blossom", direct_edge=False, pair_cache=False),
    "blossom + direct edge":          dict(method="blossom", direct_edge=True, pair_cache=False),
    "blossom + pair cache":           dict(method="blossom", direct_edge=False, pair_cache=True),
    "blossom + direct edge + cache":  dict(method="blossom", direct_edge=True, pair_cache=True),
}


def run_ablation(drange=(3, 5, 7, 9), shots=1_000, noise=1e-3, variants=None):
    """Time each decoder variant on the same shots, one core.

    Every variant decodes three independent batches of `shots` with the same decoder
    object: "cold" is the first batch (caches start empty), "warm" the average of the
    other two (caches already filled). Variants without a cache should time the same
    on both, so their cold/warm gap shows the timing noise.
    Also checks every variant against pymatching and reports how lookups were answered.
    """
    variants = variants or ABLATION_VARIANTS
    rows = []
    for d in drange:
        circuit = make_circuit(d, noise)
        dem = circuit.detector_error_model(decompose_errors=True)
        sampler = circuit.compile_detector_sampler(seed=100 + d)
        batches = [sampler.sample(shots, separate_observables=True, bit_packed=True)[0] for _ in range(3)]
        pm = pymatching.Matching.from_detector_error_model(dem)

        t0 = time.perf_counter()
        ref = [pm.decode_batch(b, bit_packed_shots=True, bit_packed_predictions=True) for b in batches]
        t_pm = (time.perf_counter() - t0) / (3 * shots) * 1e6
        rows.append(dict(d=d, variant="pymatching", cold=t_pm, warm=t_pm, agree=1.0, counts={}))

        for name, opts in variants.items():
            dec = MyCompiledDecoder(dem, **opts)
            t, agree = [], []
            for b, r in zip(batches, ref):
                t0 = time.perf_counter()
                pred = dec.decode_shots_bit_packed(bit_packed_detection_event_data=b)
                t.append((time.perf_counter() - t0) / shots * 1e6)
                agree.append(np.all(pred == r, axis=1).mean())
            warm = (t[1] + t[2]) / 2
            rows.append(dict(d=d, variant=name, cold=t[0], warm=warm, agree=min(agree),
                             counts=dict(dec.counts)))
            print(f"d={d} {name:<31} cold {t[0]:>10.1f} us  warm {warm:>10.1f} us  "
                  f"agree {min(agree):.1%}  {dec.counts}", flush=True)
    return rows


# Colour = decoder family (same colours as compare_decoders), marker/line = variant.
ABLATION_STYLE = {   # name: (color, marker, linestyle, baseline it is compared against)
    "networkx":                      ("#2a78d6", "o", "--", None),
    "networkx + source cache":       ("#2a78d6", "o", "-",  "networkx"),
    "blossom":                       ("#1baf7a", "D", "--", None),
    "blossom + direct edge":         ("#1baf7a", "^", "-",  "blossom"),
    "blossom + pair cache":          ("#1baf7a", "v", "-",  "blossom"),
    "blossom + direct edge + cache": ("#1baf7a", "*", "-",  "blossom"),
    "pymatching":                    ("#eb6834", "s", "-",  None),
}


def plot_ablation(rows, out_path="cache_ablation.png"):
    """Plot run_ablation() rows: time per shot (left) and speedup from each optimization (right).

    Right panel: each optimization relative to its own decoder without it. Filled markers =
    warm (cache already full), hollow = cold (first batch, cache starting empty).
    """
    by = {(r["d"], r["variant"]): r for r in rows}
    ds = sorted({r["d"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    for name, (color, marker, ls, _) in ABLATION_STYLE.items():
        pts = [(d, by[(d, name)]["warm"]) for d in ds if (d, name) in by]
        if pts:
            x, t = zip(*pts)
            ax1.plot(x, t, marker=marker, ls=ls, ms=8 if marker != "*" else 11, lw=2,
                     color=color, label=name)
    ax1.set_yscale("log")
    ax1.set_ylabel("decode time per shot (us, 1 core, warm)")
    ax1.set_title("Time per shot (dashed = no optimization)")

    for name, (color, marker, ls, base) in ABLATION_STYLE.items():
        if base is None:
            continue
        pts = [(d, by[(d, base)]["warm"] / by[(d, name)]["warm"], by[(d, base)]["cold"] / by[(d, name)]["cold"])
               for d in ds if (d, name) in by and (d, base) in by]
        if not pts:
            continue
        x, warm, cold = map(np.array, zip(*pts))
        ms = 8 if marker != "*" else 11
        ax2.plot(x, warm, marker=marker, ls="-", ms=ms, lw=2, color=color,
                 label=f"{name.replace(base + ' + ', '')}  (vs {base})")
        ax2.plot(x, cold, marker=marker, ls="none", ms=ms, mfc="none", mew=1.8, color=color)
    ax2.axhline(1.0, color="#52514e", lw=1, ls=":")
    ax2.plot([], [], "o", ms=8, color="#52514e", label="filled = warm (cache full)")
    ax2.plot([], [], "o", ms=8, mfc="none", mew=1.8, color="#52514e", label="hollow = cold (cache empty)")
    ax2.set_yscale("log")
    ax2.set_yticks([1, 1.5, 2, 3, 4, 5])
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax2.minorticks_off()
    ax2.set_ylabel("speedup over the same decoder without it")
    ax2.set_title("What each optimization buys")

    for ax in (ax1, ax2):
        ax.set_xticks(ds)
        ax.set_xlabel("code distance d")
        ax.set_xlim(min(ds) - 0.6, max(ds) + 0.6)
        ax.grid(True, which="major", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")
    return fig


def plot_agreement(ds=(3, 5, 7), ps=(1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2), shots=200_000,
                   out_path=None, correlated=False):
    """Logical error rate of our compiled decoder vs PyMatching, decoded on the SAME shots.

    Both decoders see identical detection events at every point, so the curves can only differ
    where the two predictions differ (ties between equal-weight matchings). correlated=True
    compares both decoders' correlated modes (saved to correlated_agreement.png).
    """
    if out_path is None:
        out_path = "correlated_agreement.png" if correlated else "agreement.png"
    mode = " (correlated)" if correlated else ""

    colors = {3: "#6da7ec", 5: "#2a78d6", 7: "#1c5cab", 9: "#104281"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    total = disagree = 0
    for d in ds:
        ours_ler, pm_ler = [], []
        for p in ps:
            circuit = make_circuit(d, p)
            dem = circuit.detector_error_model(decompose_errors=True)
            dets, obs = circuit.compile_detector_sampler(seed=int(1e6 * p) + d).sample(
                shots, separate_observables=True, bit_packed=True)
            truth = np.unpackbits(obs, axis=1, bitorder="little")[:, 0].astype(bool)
            ours = (MyCompiledDecoder(dem, method="compiled", correlated=correlated)
                    .decode_masks_compiled(dets) & 1).astype(bool)
            pm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated).decode_batch(
                dets, bit_packed_shots=True, enable_correlations=correlated)[:, 0].astype(bool)
            ours_ler.append((ours != truth).mean())
            pm_ler.append((pm != truth).mean())
            total += shots
            disagree += int((ours != pm).sum())
        c = colors.get(d, "#2a78d6")
        ax.plot(ps, ours_ler, "o-", color=c, lw=2, ms=6, label=f"ours{mode}, d = {d}")
        ax.plot(ps, pm_ler, "o--", color=c, lw=1.5, ms=12, mfc="none", mew=1.5, label=f"PyMatching{mode}, d = {d}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate (per shot)")
    ax.set_title(f"Same shots, both decoders{mode} ({shots:,} shots per point)")
    ax.text(0.02, 0.97, f"predictions agree on {1 - disagree / total:.4%} of {total:,} shots\n"
            f"({disagree} differ)", transform=ax.transAxes, va="top", fontsize=9, color="#52514e")
    ax.grid(True, which="major", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"{disagree}/{total} predictions differ; saved plot to {out_path}")
    return fig


if __name__ == "__main__":
    main()

"""
My impelmentation, I think I understand this part
claude implemented APSP and the matching algo cuz its not really qc

It also made my code cleaner


import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
import sinter
from typing import List
imp
impor
import scipy
import re
import networkx
import heapq


pattern = re.compile(
    r"detector\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*D(\d+)"
)

class graph_neighbor():
    def __init__(self, nname, prob):
        self.nname = nname
        self.prob = prob

class sim_graph():
    def __init__(self, size, dem):
        adj_mat, location = self.parse_input(size,dem)
        self.adj_mat = self.construct_graph(adj_mat, size)



    def parse_input(self, size, dem):
        adj_list = ()
        adj_mat = [[]] * size * size
        location = [None] * size

        for line in dem.flattened():
            if line[0] == "e":
                data  = line.split(" ")
                prob = float(data[0][6:-1])
                all_nodes = []
                cur_pair = []
                for i in range(1,len(data)):
                    if(data[i] == "^"):
                        all_nodes.append(cur_pair)
                    else:
                        cur_pair.append(data[i])
                all_nodes.append(cur_pair)
                
                for pair in all_nodes:
                    if(len(pair) != 2): 
                        continue
                    idx1 = int(pair[0][1:])
                    idx2 = int(pair[1][1:])
                    adj_mat[size * idx1 + idx2].append(prob)
                    adj_mat[size * idx2 + idx1].append(prob)
            elif line[0] == "d":
                m = pattern.search(line)
                a, b, c = map(float, m.groups()[:3])   # 2.0, 4.0, 8.0
                d = int(m.group(4))                    # 65
                location[d] = (a,b,c)
            else:
                print("parse error")
        return adj_mat, location

    def construct_graph(self, adj_mat,size):
        adj_new = np.zeros(adj_mat, size)
        for x in range(size):
            for y in range(size):
                p1 = adj_new[x,y]
                p2 = adj_mat[x*size, y]
                adj_new[x,y] = p1(1-p2) + p2(1-p1)
        return adj_new

# claude's formula:
# p = 0.0
# for p_i in contributions:
#     p = p * (1 - p_i) + p_i * (1 - p)

class MyCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, dem: stim.DetectorErrorModel):
        self.num_dets = dem.num_detectors
        self.num_obs = dem.num_observables
        self.dem = dem
        dists = [3,5,7,9,11]
        for d in dists:
            if(self.num_dets == 3 * d *(d * d - 1)):
                self.dist = d
                break
        self.graph = sim_graph(self.num_dets, dem)



    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,   # uint8, shape (shots, ceil(num_dets / 8))
    ) -> np.ndarray:                                    # uint8, shape (shots, ceil(num_obs / 8))
        # unpack to bool: shape (shots, num_dets)
        dets = np.unpackbits(bit_packed_detection_event_data,
                             axis=1, bitorder="little")[:, :self.num_dets].astype(bool)

        pred = np.zeros((dets.shape[0], self.num_obs), dtype=bool)
        for i, shot in enumerate(dets):
            pred[i] = self.decode_one(shot)            # your decoder: detectors -> observable flips

        return np.packbits(pred, axis=1, bitorder="little")

    def decode_one(self, shot: np.ndarray) -> np.ndarray:
        # shot: bool array of length num_dets (True = lit detector)
        # return: bool array of length num_obs (True = flip that logical)
        measurements_per = self.dist * self.dist - 1

        self.graph.adj_mat

        errors = [i for i, val in enumerate(shot) if val]
        apsp = -1 * np.ones_like((len(errors), len(errors)))
        for error in errors:


        

        coords = self.dem.get_detector_coordinates()  
        print(f"{len(coords)}, {self.num_dets}")
        return np.zeros(self.num_obs, dtype=bool)

def dijkstra(this_error, errors, adj):
    d = dict()
    for idx, e in enumerate(errors):
        d[e] = idx
    path = [-1 ] * len(errors)
    pq = []
    heapq.push(pq,(0, this_error))
    while(len(pq) > 0):
        priority, task = heapq.heappop(pq)
        if(path[d[task]] != -1):
            continue
            



class MyDecoder(sinter.Decoder):
    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return MyCompiledDecoder(dem)


def main():
    a = run_sinter_benchmark(drange = [3])
    analyze_surface_code_stats(a)



    pass




# Benchmarking
def run_sinter_benchmark(mydecfn = MyDecoder(), noise=1e-3, drange = [3, 5, 7, 9]):
    surface_code_tasks = [
        sinter.Task(
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=d * 3,
                distance=d,
                after_clifford_depolarization=noise,
                after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise,
            ),
            json_metadata={'d': d, 'r': d * 3, 'p': noise},
        )
        for d in drange
    ]

    collected_surface_code_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=surface_code_tasks,
        decoders=["mine", "pymatching"],          # names to run
        custom_decoders={"mine": mydecfn},        
        max_shots=1,
        max_errors=1,
        print_progress=False,
    )
    return collected_surface_code_stats

def analyze_surface_code_stats(collected_surface_code_stats):
    for stats in collected_surface_code_stats:
        d = stats.json_metadata['d']
        if not stats.errors:
            print(f"Didn't see any errors for d={d}")
            continue
        per_shot = stats.errors / stats.shots
        per_round = sinter.shot_error_rate_to_piece_error_rate(per_shot, pieces=stats.json_metadata['r'])
        print(f"d:{d}, per round {per_round}")

if __name__ == "__main__":
    main()  per_round = sinter.shot_error_rate_to_piece_error_rate(per_shot, pieces=stats.json_metadata['r'])
        print(f"d:{d}, per round {per_round}")

if __name__ == "__main__":
    main()
    """
