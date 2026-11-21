"""Fault-injection probe for greedy matching (written by Claude 2, 2026-11-21).

Injects k faults bunched near each other (each within 2 hops of the first in the fault graph) and
reports how often greedy, greedy+2-opt and exact MWPM fail. A distance-d decoder must correct every
set of k <= (d-1)/2 faults. For greedy+2-opt failures it also counts how many pairs differ from
MWPM's matching (2-opt can only repair differences of 2 pairs in one move).

    python tools/greedy_fault_probe.py
"""
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from qec import memory_circuit, detector_error_model
from qec.matching import GreedyMatchingDecoder, MWPMDecoder
from qec.metrics import _fault_table

def norm(pairs):
    return {frozenset((i, -1 if j is None else j)) for i, j in pairs}

def total(pairs, defects, dist, B):
    return sum(dist[i, B] if j is None else dist[i, defects[j]] for i, j in pairs)

for d, n in [(5, 3000), (7, 3000), (9, 1500)]:
    dem = detector_error_model(memory_circuit(d, 0.003))
    H, L = _fault_table(dem)
    A = sp.csr_matrix(H.T.astype(np.int8)) @ sp.csr_matrix(H.astype(np.int8))   # faults sharing a detector
    A2 = (A @ A).tocsr()
    decs = {"greedy": GreedyMatchingDecoder(dem), "greedy+2opt": GreedyMatchingDecoder(dem, local_search=True),
            "MWPM": MWPMDecoder(dem)}
    rng = np.random.default_rng(7)
    for k in range(2, (d - 1) // 2 + 1):
        fails = Counter(); cyc = Counter()
        for _ in range(n):
            f0 = rng.integers(H.shape[1])
            near = A2.indices[A2.indptr[f0]:A2.indptr[f0 + 1]]
            near = near[near != f0]
            if len(near) < k - 1:
                continue
            c = np.concatenate([[f0], rng.choice(near, k - 1, replace=False)])
            det = np.logical_xor.reduce(H[:, c], axis=1)
            obs = int(np.logical_xor.reduce(L[:, c], axis=1) @ (1 << np.arange(L.shape[0])))
            for name, dec in decs.items():
                if dec.decode(det) != obs:
                    fails[name] += 1
                    if name == "greedy+2opt":
                        g = decs["greedy+2opt"]; defects = np.flatnonzero(det).tolist()
                        dist, _ = g._paths(defects)
                        cyc[len(norm(g._pairs(defects, dist)) ^ norm(decs["MWPM"]._pairs(defects, dist))) // 2] += 1
        print(f"d={d}, {k} bunched faults ({n} samples): " +
              ", ".join(f"{nm} {100*fails[nm]/n:.2f}%" for nm in decs) +
              (f"   | 2opt failures by pairs that differ from MWPM: {dict(sorted(cyc.items()))}" if cyc else ""))
