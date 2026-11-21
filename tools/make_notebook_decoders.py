"""Generate notebooks/02_decoders_from_scratch.ipynb.

    .venv\\Scripts\\python tools\\make_notebook_decoders.py           # write the notebook
    .venv\\Scripts\\python tools\\make_notebook_decoders.py --check   # also run every code cell
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_notebook import ROOT  # noqa: E402

CELLS: list[tuple[str, str]] = [
    ("md", """# Decoders from scratch

Notebook 1 used PyMatching as a black box. Here we open the box: every decoder below was written by us
(`qec/union_find.py`, `qec/matching.py`, `qec/bp.py`, `qec/lookup.py`), and we'll watch each one work
on the same shots.

1. turn a noisy circuit into a **decoding graph**,
2. **Union-Find**: grow, merge, peel,
3. **matching**: exact (blossom) vs greedy,
4. **belief propagation**, and why it needs a partner,
5. **measure** every decoder: accuracy, fault tolerance, speed, parallel depth.

Background: `docs/DECODERS.md`. Run top to bottom with Shift+Enter."""),
    ("code", """import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
from qec import memory_circuit, detector_error_model, make_decoder, OURS
from qec.dem_graph import DemGraph
from qec.metrics import sample, logical_error_rate, fault_correction_rate, latency, parallel_depth"""),
    ("md", """## 1. From circuit to graph

A distance-3 surface code, 3 rounds of noisy syndrome extraction, every operation failing with
probability 0.5%. Stim's **detector error model** lists every way a single fault can happen. Our
`DemGraph` turns it into:

* `H`, `L`: which detectors and which logical observable each fault flips (for BP and lookup tables),
* a weighted graph: detectors are nodes, faults are edges (for matching and Union-Find)."""),
    ("code", """circuit = memory_circuit(3, 0.005)
dem = detector_error_model(circuit)
g = DemGraph.from_dem(dem)
print("detectors:", g.num_detectors, " fault mechanisms:", g.num_mechanisms, " graph edges:", len(g.edge_u))
print("most likely edge: p =", g.edge_p.max().round(5), " least likely: p =", g.edge_p.min().round(6))
print("edge weight = log((1-p)/p): from", g.edge_weights.min().round(2), "to", g.edge_weights.max().round(2))"""),
    ("md", """## 2. Some noisy shots

Each shot gives the **detection events** (which detectors fired) and the truth: did the logical qubit
actually flip? A decoder only sees the first and must guess the second."""),
    ("code", """dets, obs = sample(circuit, 5000, seed=1)
i = int(np.flatnonzero(dets.sum(axis=1) == 4)[0])     # pick a shot with 4 detection events
shot = dets[i]
print("shot", i, "fired detectors:", np.flatnonzero(shot), " logical flipped:", bool(obs[i, 0]))"""),
    ("md", """## 3. Union-Find

Each odd cluster grows by half an edge per round. When an edge fills, the clusters at its ends merge.
Even clusters stop. Then peeling picks the correction. `last_growth_rounds` counts the rounds, which is
the decoder's *parallel depth*: on a chip, every edge grows at the same time, so each round takes one step."""),
    ("code", """uf = make_decoder("union_find", g)
pred = uf.decode(shot)                       # the readable Python version
print("UF predicts logical flip:", bool(pred & 1), " growth rounds:", uf.last_growth_rounds)

wuf = make_decoder("union_find_weighted", g)  # likely edges grow faster
print("weighted UF predicts:", bool(wuf.decode(shot) & 1), " growth rounds:", wuf.last_growth_rounds)"""),
    ("md", """## 4. Matching: exact vs greedy

`mwpm` runs Dijkstra from every fired detector, gives each one a private boundary copy, and finds the
exact cheapest pairing (blossom). `greedy_matching` takes the pairings that save the most first.
Let's look at the pairs each one picks: `None` means "matched to the boundary"."""),
    ("code", """mw = make_decoder("mwpm", g)
gr = make_decoder("greedy_matching", g)
defects = np.flatnonzero(shot).tolist()
dist, pred_tree = mw._paths(defects)
print("MWPM pairs:  ", [(defects[a], None if b is None else defects[b]) for a, b in mw._pairs(defects, dist)])
print("greedy pairs:", [(defects[a], None if b is None else defects[b]) for a, b in gr._pairs(defects, dist)])

import pymatching
pm = pymatching.Matching.from_detector_error_model(dem)
ours, theirs = mw.decode_batch(dets[:1000]), pm.decode_batch(dets[:1000]).astype(bool)
print("our MWPM agrees with PyMatching on", (ours == theirs).all(axis=1).mean() * 100, "% of 1000 shots")"""),
    ("md", """## 5. Belief propagation

BP passes messages between detectors and faults until its best guess explains every detection event,
or it gives up after 30 iterations. On the surface code it often gives up: many different corrections
are equally good, and BP can't decide between them."""),
    ("code", """bp = make_decoder("bp", g)
bp.decode_batch(dets)
res = bp.last
print(f"BP converged on {res.converged.mean():.0%} of shots, average {res.iterations.mean():.1f} iterations")
top = np.argsort(res.llr[i])[:5]
print("shot", i, "- the 5 faults BP thinks most likely, with P(fault):", np.round(res.posterior[i][top], 3))"""),
    ("md", """When BP doesn't converge, its beliefs still carry information. **BP+OSD** sorts faults by belief
and solves the linear system using the most likely ones. **BP+UF** turns the beliefs into this shot's edge
weights for Union-Find."""),
    ("md", """## 6. Measure everything

The same functions `pytest` uses (`qec/metrics.py`). Compare with the table at the end of
`pytest tests/test_metrics.py`."""),
    ("code", """circuit5 = memory_circuit(5, 0.003)
dem5 = detector_error_model(circuit5)
d5, o5 = sample(circuit5, 3000, seed=2)
print(f"{'decoder':22s} {'logical error':>14s} {'1-fault fails':>14s} {'us/shot':>8s} {'depth':>6s}")
for name in ["union_find", "union_find_weighted", "greedy_matching", "bp", "bp_osd", "bp_uf", "pymatching"]:
    dec = make_decoder(name, dem5)
    r = logical_error_rate(dec, d5, o5)
    f1 = fault_correction_rate(dec, dem5, 1)
    t = latency(dec, d5[:1000], single_shots=20)
    dep = parallel_depth(dec, d5[:1000])
    print(f"{name:22s} {r.rate:14.4f} {f1.errors:>14d} {t.batch_us_per_shot:8.1f} {'-' if dep is None else round(dep['mean'], 1):>6}")"""),
    ("md", """## Things to try

* Raise `p` to 0.008 and rerun the last cell. Which decoders fall apart first?
* `fault_correction_rate(dec, dem5, 2)`: which decoders fix *every* two-fault error at d = 5?
  (Greedy doesn't. Why not? Look at section 4.)
* `make_decoder("lookup", dem)` at d = 3: the fastest possible decoder. Why can't we use it at d = 7?
* Write your own decoder: any class with `decode_batch(dets) -> predictions`. Register it in
  `qec/decoders.py` and the tests and benchmarks pick it up."""),
]


def main() -> None:
    import json

    from make_notebook import build as _build  # reuse the cell -> JSON conversion

    import make_notebook
    make_notebook.CELLS, saved = CELLS, make_notebook.CELLS
    try:
        nb = _build()
    finally:
        make_notebook.CELLS = saved
    path = ROOT / "notebooks" / "02_decoders_from_scratch.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", path)
    if "--check" in sys.argv:
        import os

        import matplotlib
        matplotlib.use("Agg")
        os.chdir(path.parent)
        ns: dict = {}
        for kind, src in CELLS:
            if kind == "code":
                exec(compile(src, "<cell>", "exec"), ns)  # noqa: S102 - our own notebook source
        print("all code cells ran")


if __name__ == "__main__":
    main()
