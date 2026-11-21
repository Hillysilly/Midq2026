# Our decoders: how they work, what they cost, how they parallelise

Everything on this page is implemented from scratch in `qec/` (no decoding library inside).
PyMatching, ldpc/stimbposd and beliefmatching are only used as **baselines** to compare against.
Measured numbers are in [`RESULTS.md`](../RESULTS.md) §7. `pytest` recomputes the metrics.

| decoder | file | idea in one line | cost per shot | parallel structure |
|---|---|---|---|---|
| Union-Find | `union_find.py` | grow clusters around defects until each is even, then peel a spanning tree | ~O(n α(n)) | all edges grow at once each round; few rounds |
| UF weighted | `union_find.py` | same, but likely edges grow faster | same | same, a few more rounds |
| MWPM | `matching.py` | shortest paths + exact minimum-weight perfect matching (blossom) | O(k³) for k defects | hard to parallelise (blossom is sequential) |
| Sparse Blossom | `sparse_blossom.py` | the same exact matching, grown directly on the detector graph (PyMatching's algorithm) | ~linear in practice | sequential events |
| Greedy (+2-opt) | `matching.py` | pair the defects that save the most, then fix with local swaps | O(k² log k) | swaps on disjoint pairs run in parallel |
| Lookup table | `lookup.py` | precompute the most likely correction for every likely syndrome | O(1) | one memory read |
| BP | `bp.py` | pass probability messages between detectors and error mechanisms | O(iters · nnz) | every message updates at once |
| BP+OSD | `bp.py` | BP, then Gaussian elimination on the most likely columns | O(n r²) for OSD | BP part parallel, OSD sequential |
| **BP+UF** | `bp.py` | BP's beliefs become Union-Find's edge weights, per shot | BP + UF | both stages parallel |
| Local BP+UF | `bp_local.py` | the same, with BP run only near the lit detectors, fused into one kernel | ~4–5× less than BP+UF | same, and scales better across threads |

`n` = error mechanisms, `k` = defects (detectors that fired), `r` = rank of H (≈ number of detectors).

---

## The shared foundation: from Stim to a graph (`dem_graph.py`)

Stim gives a **detector error model (DEM)**: a list of independent error mechanisms, each with a
probability, the detectors it flips and the logical observables it flips. We build two views of it:

* **Check matrices** `H` (detector × mechanism) and `L` (observable × mechanism) over GF(2), for BP,
  OSD and the lookup table. A shot's syndrome is `s = H e`, and the decoder must guess `L e`.
* **Decoding graph**: detectors are nodes, plus one boundary node. Each mechanism that flips one or two
  detectors is an edge (Stim splits the others, such as Y errors, into two edges). Edge weight
  `log((1-p)/p)`: the less likely an error, the "longer" the edge.

Check: our graph has exactly PyMatching's edges and weights (`tests/test_new_decoders.py`).

---

## Union-Find (Delfosse & Nickerson 2017; review §5.2)

1. Every defect starts as a cluster with odd parity.
2. **Grow**: every odd cluster that doesn't touch the boundary grows by half an edge in every direction.
   When an edge becomes fully grown, the clusters at its ends **merge** (union-find data structure,
   hence the name). Merging two odd clusters makes an even one, which stops growing.
3. **Peel**: inside each final cluster, build a spanning tree and strip leaves. A leaf that is a defect
   puts its edge into the correction and passes its defect status to its parent.

*Weighted* growth: an edge needs `≈ 4·w/w_min` steps instead of 2, so likely edges fill first.
This closes most of the gap to MWPM.

**Parallelism.** Our compiled kernel (`_uf_kernel_py`) is written in the *synchronous* form a chip
would use. In each round, every edge independently looks at its two end clusters and grows, and then
the merges happen. The number of rounds is the hardware latency. At circuit level, p = 0.3%, that is
about 1–2 rounds (unweighted) or 2–6 (weighted) from d = 3 to 9, while the graph grows like d³.
Rounds depend on the *size of the biggest error cluster*, not the code size.

## MWPM (Dennis et al. 2002, Fowler; review §5.1)

1. Dijkstra from every defect gives the cost of every defect pair, and of every defect to the boundary.
2. Build the **syndrome graph**: defects plus one private boundary copy each (copies connect to each
   other at zero cost, so any number of defects can go to the boundary).
3. Minimum-weight perfect matching on it (Edmonds' blossom algorithm; we call networkx for this one step).
4. XOR the observables along each matched path.

It agrees with PyMatching on **100%** of shots (tested), and is ~500× slower, which is why PyMatching's
Sparse Blossom (no explicit syndrome graph) matters.

## Sparse Blossom (Higgott & Gidney 2023; the algorithm inside PyMatching 2)

The same exact MWPM, without ever computing all-pairs distances. Our implementation is
`qec/sparse_blossom.py` (pure Python, ~450 lines).

1. Every lit detector starts a **region** of radius 0. The radii are Edmonds' dual variables.
2. Growing regions spread along the detector graph's edges at unit speed (integer edge weights, so every
   event happens at an exact time). A priority queue holds the next event on every edge:
   - a region **reaches an empty node**: the node joins it;
   - **two regions touch**: a matching step;
   - a region **reaches the boundary**: it matches to the boundary.
3. Matching steps are Edmonds' blossom algorithm on regions:
   - two growing regions from different alternating trees touch: **augment** (match them, flip both tree paths);
   - a growing region touches a matched pair: **grow the tree** (the touched region becomes inner and
     shrinks, its partner becomes outer and grows);
   - two growing regions of the same tree touch: an odd cycle, so **form a blossom** that grows as one region;
   - an inner blossom shrinks to radius 0: **shatter** it back into its children;
   - an inner single-detector region shrinks to radius 0: its two tree neighbours now touch through it,
     which forms a blossom.
4. When nothing grows, every region is matched. Blossoms are expanded into detector pairs, and each pair's
   correction is a shortest path from a small local Dijkstra search.

It finds a matching of exactly PyMatching's weight on every shot we compared (thousands, up to d = 9, including
1.2% noise where blossoms form and shatter constantly; `tests/test_new_decoders.py`). It is 3–6× faster than our
textbook MWPM, and the gap grows with d. It is ~130× slower than PyMatching at d = 7: that's pure Python vs
tuned C++ running the same algorithm.

How it relates to Union-Find: both grow regions around lit detectors. Union-Find stops as soon as every
cluster has even parity and guesses a correction inside each cluster; Sparse Blossom keeps exact radii and
runs Edmonds' steps when regions touch, which is what makes its answer optimal.

**Greedy matching** replaces step 3 with "start with everyone at the boundary, then take pairings in
order of how much they save". It is fast and simple but **not distance-preserving**: it grabs one
attractive pair and strands two defects. About 2% of all two-fault errors fail at d = 5, so its logical
error rate *doesn't fall* with distance. **2-opt** (swap partners between two pairs whenever that
lowers the total cost) cuts the failures by 3×, but still not to zero. That is a neat demonstration of
why the exact blossom algorithm is worth its cost.

## Lookup table

Enumerate every combination of up to 2 error mechanisms, compute its syndrome, and keep the most
probable combination per syndrome. At d = 3 (circuit level) this is 13k entries, covers 97% of shots at
p = 0.5%, and is **more accurate than MWPM** (it sees hyperedges and exact probabilities). The table
grows like n^w, so it's hopeless beyond d = 3. It's the classic speed/memory trade-off.

## Belief propagation (review §5.3)

Messages flow on the Tanner graph between detectors (checks) and mechanisms (variables):

* check → variable (min-sum): "given what the other mechanisms on me are doing, and my syndrome bit,
  how likely is it that *you* flipped?" That's the sign product × the minimum magnitude, scaled by 0.625.
* variable → check: the prior plus everything the *other* checks said.

After each iteration, the hard decision `llr < 0` is checked against the syndrome. Every message in an
iteration is independent: the numpy version (`backend="numpy"`) literally updates all of them with
whole-array operations, and the numba version (same results to 1e-14) spreads shots over cores.

On the surface code, BP alone is poor: the code is **degenerate** (many equivalent corrections) and
has short loops, so BP often oscillates. At d = 5 it converges on only ~40% of shots at p = 0.5%.
So BP is used as a *first stage*.

### BP+OSD

When BP doesn't converge, sort mechanisms from most to least likely (by BP's output) and solve
`H e = s` using only the most likely linearly independent columns (OSD-0). Our solver keeps an
incremental XOR basis in Python big ints (each column is one int), so elimination is a few hundred
big-int XORs. **Combination sweep (OSD-CS)** then tries flipping each of the next 30 columns and pairs
among the next 10, and keeps the lowest-weight valid answer. This lowered d = 5 logical errors by ~25%
compared with OSD-0.

### BP+UF ("belief-find", Higgott et al. 2023)

Run BP. If it converges, done. If not, turn each mechanism's posterior into per-edge probabilities
(an edge flips if an odd number of its mechanisms do), then weights, then Union-Find growth steps, and
run weighted UF **with this shot's weights**. BP contributes what a graph decoder can't see (the X/Z
correlation of Y errors, hyperedges), and UF contributes the guaranteed valid, local, near-linear
correction.

* Both stages are parallel: BP per message, UF per edge.
* It's the most accurate of our decoders, beating PyMatching at d = 5 and 7 (see RESULTS §7).
* It was introduced in the same paper as belief-matching (Higgott et al. 2023), with UF in place of MWPM, so it keeps
  UF's hardware-friendliness.

### Local BP+UF (`bp_local.py`)

Global BP+UF spends almost all its time in BP, and BP updates every Tanner-graph edge every iteration even when
only one or two detectors are lit. Its cost per update is already compiled speed (~5 ns), so the fix is less work,
not a faster language:

* run BP only on checks within 1 hop of a lit detector (far checks send a fixed "not flipped" message);
* stop BP after 10 iterations instead of 30 (Union-Find cleans up the rest);
* turn beliefs into edge costs and run Union-Find in the same compiled kernel, with no per-shot allocations.

It is 3–5× faster than global BP+UF on one thread and 7–9× on 12 threads (d = 5–11, p = 0.1%), with the same
accuracy from d = 7 up. At d = 5 the 1-hop region costs 1.7× in accuracy (a 2-hop region fixes that but is no
faster at that size). Numbers: `results/2026-11-21_1559_p1e-3_distance_sweep.md`, *Local BP+UF*.

**Caveat we found**: BP-based decoders aren't strictly distance-preserving. When BP oscillates, its
posteriors can rank the true error columns hundreds of places down. A few two-fault errors at d = 5
(0.07% for BP+UF, 0.27% for BP+OSD) then fail even though plain UF fixes them. Yet their overall logical
error rate is the lowest: *distance-preserving* and *most accurate* are different things.

---

## The metrics (`qec/metrics.py`, `tests/test_metrics.py`)

| metric | function | what it tells you |
|---|---|---|
| logical error rate (+ Wilson CI), per shot and per round | `logical_error_rate`, `Rate.per_round` | accuracy |
| Λ = ε_d / ε_{d+2} | `lambda_factor` | how much a bigger code helps |
| threshold | `threshold` | p below which bigger codes win |
| k-fault correction rate | `fault_correction_rate` | is it distance-preserving? |
| smallest failing fault weight | `min_failing_weight` | effective distance |
| µs per shot (batched), single-shot p50/p99 | `latency` | throughput vs latency |
| keeps up with a 1 µs cycle? | `keeps_up` | real-time feasibility |
| parallel depth (mean/p99/max) | `parallel_depth` | hardware latency in "clock cycles" |

`pytest` runs them all and prints a table at the end. `python metrics_report.py` makes the
plots in `out/metrics/`.
