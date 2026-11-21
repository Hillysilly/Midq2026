# Logical error rate at p = 0.1%, by decoder and distance

**Run:** 2026-11-21, 15:59 CST · raw data: `out/p1e-3/stats.csv` · plot: `out/p1e-3/plot.png`

**Updated 2026-11-21, 16:30 CST:** added our Sparse Blossom decoder, and a decode-only speed table measured after
fixing a Union-Find slowdown (see *Decode-only speed*).

**Updated 2026-11-21, 17:11 CST:** added local BP+UF (`bp_uf_local`; see *Local BP+UF*).

**Setup:** rotated CSS surface code, Z-basis memory, d rounds per shot, uniform circuit-level depolarizing
noise p = 0.001 (every gate, reset and measurement), same Stim circuits for every decoder. Each point ran
until 100 logical errors or its shot cap (20M for the fast decoders, 1–2M for BP+UF, 0.1–1M for the others).
Sinter workers single-threaded (`NUMBA_NUM_THREADS=1`).

Reproduce (bash, from the repo root; resumes from `out/p1e-3/stats.csv` if present):

```bash
export NUMBA_NUM_THREADS=1
B=".venv/Scripts/python bench.py --name p1e-3 --noise circuit --ps 0.001 --max-errors 100"
$B --distances 3 5 7 9 11 --decoders pymatching --max-shots 20000000
$B --distances 3 5 7 9 --decoders pymatching_correlated union_find union_find_weighted --max-shots 20000000
$B --distances 3 5 7 --decoders bp_uf --max-shots 2000000
$B --distances 9 --decoders bp_uf --max-shots 1000000
$B --distances 3 5 --decoders bp_osd mwpm greedy_matching greedy_matching_2opt bp --max-shots 1000000
$B --distances 7 --decoders bp_osd --max-shots 300000
$B --distances 9 --decoders bp_osd --max-shots 100000
$B --distances 3 --decoders lookup --max-shots 2000000
$B --distances 3 5 --decoders sparse_blossom --max-shots 1000000
$B --distances 7 --decoders sparse_blossom --max-shots 200000
$B --distances 3 5 7 --decoders bp_uf_local --max-shots 2000000
$B --distances 9 --decoders bp_uf_local --max-shots 1000000

# decode-only speed tables (CPU otherwise idle)
.venv/Scripts/python tools/decode_speed.py --p 0.001 --distances 3 5 7 9 11
.venv/Scripts/python tools/decode_speed.py --p 0.001 --distances 3 5 7 9 11 --decoders pymatching_correlated bp_uf bp_uf_local --max-d-slow 11
```

(The d = 11 points for correlated PyMatching and both Union-Finds come from an earlier, stopped run with
the same settings.)

## Logical error rate per round

| decoder | d=3 | d=5 | d=7 | d=9 | d=11 | Λ (per +2 in d) | µs/shot at d=5 (see Speed) |
|---|---|---|---|---|---|---|---|
| **PyMatching correlated** (baseline) | 2.8e-4 | **1.8e-5** | **1.2e-6** | **6.7e-8**ᵃ | <1e-7 (0 errors) | **15–18** | 4.1 |
| **BP+UF** (ours) | 3.0e-4 | 2.3e-5 | 2.1e-6 | 2.2e-7ᵇ | – | 10–13 | 420 |
| Local BP+UF (ours) | 2.8e-4 | 4.0e-5 | 2.5e-6 | 1.1e-7ᵇ (1 error) | – | 7–16 | 50ᵈ |
| PyMatching (baseline) | 2.7e-4 | 2.7e-5 | 2.2e-6 | 1.9e-7 | 3.6e-8ᵃ | 10–12 | 1.9 |
| BP+OSD (ours) | 2.5e-4 | 2.8e-5 | 3.3e-6ᵇ | <4e-6 (0 errors) | – | 8–9 | 570 |
| MWPM (ours) | 2.6e-4 | 2.9e-5 | – | – | – | 9 | 750 |
| Sparse Blossom (ours) | 2.3e-4 | 2.6e-5 | 7e-7ᵇ (1 error) | – | – | 9 | 181 |
| UF weighted (ours) | 3.5e-4 | 3.4e-5 | 3.8e-6 | 3.4e-7 | – | 9–11 | 14 |
| Union-Find (ours) | 3.2e-4 | 3.5e-5 | 5.9e-6 | 6.1e-7 | 8.5e-8ᵇ | 6–10 | 8.6 |
| Lookup table (ours) | **2.2e-4** | – | – | – | – | – | – |
| Greedy + 2-opt (ours) | 2.6e-4 | 1.6e-4 | – | – | – | 1.6 | 370 |
| Greedy (ours) | 3.0e-4 | 4.3e-4 | – | – | – | 0.7 (gets worse) | 360 |
| BP alone (ours) | 3.6e-3 | 4.5e-3 | – | – | – | 0.8 (gets worse) | 600 |

Most points have 100+ logical errors (±20% at 95% confidence).
ᵃ 8–35 errors: roughly ±30–60%. ᵇ 1–7 errors: order of magnitude only.
"–" = not run at that size (too slow: the slow decoders, BP+OSD at d=11; impossible: lookup beyond d=3).
ᵈ decode-only, one thread (its Sinter time is inflated by numba compiling the kernel in every worker).
Λ = ε_d / ε_{d+2}, the range over the distance pairs with enough errors. µs/shot is one CPU thread inside a
Sinter worker.

## Speed

Time per **round** in µs (time per shot in brackets), from Sinter's own accounting: CPU time per shot for
**Stim sampling + decoding**, one thread per worker, 11 workers sharing the laptop's cores. Same runs as the
table above. The Union-Find rows were measured **before** the fix described below.

| decoder | d=3 | d=5 | d=7 | d=9 | d=11 |
|---|---|---|---|---|---|
| Stim sampling alone (measured separately) | 0.03 (0.1) | 0.05 (0.25) | 0.11 (0.8) | 0.18 (1.6) | 0.25 (2.8) |
| PyMatching | **0.09** (0.3) | **0.39** (1.9) | **0.73** (5.1) | **1.3** (12) | **2.0** (22) |
| PyMatching correlated | 0.31 (0.9) | 0.82 (4.1) | 1.9 (13) | 3.5 (31) | 5.7 (63) |
| Union-Find (ours) | 2.6 (7.9) | 1.7 (8.6) | 4.3 (30) | 8.8 (79) | 14 (150) |
| UF weighted (ours) | 2.8 (8.3) | 2.8 (14) | 8.1 (57) | 17 (154) | 26 (290) |
| Lookup table (ours) | 13 (40)ᶜ | – | – | – | – |
| BP+UF (ours) | 15 (46) | 84 (420) | 515 (3,605) | 1,960 (17,600) | – |
| BP+OSD (ours) | 15 (44) | 113 (566) | 660 (4,620) | 2,430 (21,900) | – |
| Greedy (ours) | 22 (66) | 72 (359) | – | – | – |
| Greedy + 2-opt (ours) | 24 (71) | 74 (368) | – | – | – |
| BP alone (ours) | 42 (125) | 121 (605) | – | – | – |
| MWPM (ours) | 46 (139) | 151 (754) | – | – | – |
| Sparse Blossom (ours) | 9 (27) | 36 (181) | 91 (636) | – | – |

ᶜ Almost all Python overhead (packing each shot into a dictionary key); the table read itself is one memory
access, which in hardware is a single clock cycle.

What it shows:

* **Only PyMatching keeps up with a 1 µs syndrome cycle on one core**, up to about d = 9 (1.1 µs per round once
  sampling is subtracted). Correlated PyMatching keeps up to d = 5. Everything else needs several cores,
  parallel windows or hardware.
* **Union-Find (before the fix) was 4–7× slower than PyMatching per round from d = 5 up.** Part of that was a
  wrapper bug (next section). The rest is structural: the edge-parallel kernel scans every edge each growth
  round, so on one CPU core it pays for the whole graph even when only a few checks are lit. On parallel
  hardware that scan is one step per round.
* **The BP hybrids are 100–1,500× slower than PyMatching on a CPU**, and their time per shot grows
  quickly with d (bigger graph × more iterations, because BP converges less often). Nearly all of it is BP, the
  part that parallelises on a GPU or FPGA.
* **Our MWPM costs ~400× PyMatching at d = 5** for the same answers: the price of building the full graph
  between lit checks and using a general-purpose blossom implementation instead of Sparse Blossom.
* **Our Sparse Blossom is 4–5× faster than our textbook MWPM** under Sinter for identical answers, but ~100×
  slower than PyMatching's C++ version of the same algorithm.
* These numbers run higher than the quiet single-thread benchmark in `out/metrics/report.md` (for example
  Union-Find 8.6 vs 5.6 µs per shot at d = 5), because 11 workers compete for cores and sampling is included.

## Decode-only speed (after the Union-Find fix)

`tools/decode_speed.py`: µs per shot, decoding only (no sampling), CPU otherwise idle, same p = 0.1% circuits.
Numba decoders show **1 thread / 12 threads**. PyMatching doesn't release Python's GIL, so extra threads
don't change it (measured: 1.84 → 1.70 µs at d = 7 with 12 threads).

| decoder | d=3 | d=5 | d=7 | d=9 | d=11 |
|---|---|---|---|---|---|
| PyMatching (C++) | 0.12 | 0.63 | 1.89 | 4.07 | 7.80 |
| PyMatching correlated (C++) | 0.28 | 1.66 | 4.94 | 11.9 | 23.6 |
| Union-Find (ours, numba) | 0.36 / **0.07** | 2.40 / **0.43** | 8.98 / **1.83** | 22.5 / 4.67 | 46.0 / 10.1 |
| UF weighted (ours, numba) | 0.45 / 0.09 | 4.35 / 0.85 | 19.5 / 4.20 | 50.5 / 11.6 | 104 / 23.0 |
| Sparse Blossom (ours, Python) | 17.6 | 74.7 | 248 | – | – |
| MWPM (ours, Python + networkx) | 56.4 | 310 | 1,405 | – | – |

**The Union-Find fix.** Before calling the compiled kernel, the wrapper copied the per-edge growth limits
into one row per shot (about 3 GB of memory at d = 7), and the kernel allocated 13 working arrays for every
shot inside the parallel loop. Now the limits are passed once (per-shot rows only for BP+UF), and each block
of 256 shots shares one set of working arrays. Predictions are unchanged (all tests pass). At d = 7:

| Union-Find, d = 7, p = 0.1% | 1 thread | 12 threads |
|---|---|---|
| before | 15.5 µs | 7.4 µs (2.1× scaling) |
| after | **9.0 µs** | **1.83 µs** (4.9× scaling) |

What it shows:

* **Multithreaded Union-Find now matches PyMatching's throughput at d = 7** (1.83 vs 1.89 µs per shot) and beats
  it at d = 3 and 5 (0.07 vs 0.12, 0.43 vs 0.63). Per core, PyMatching is still 4–6× faster.
* **Sparse Blossom vs our textbook MWPM:** 3–6× faster for the same matching, and the gap grows with d
  (3.2× at d = 3, 5.7× at d = 7), because it only grows regions around lit detectors instead of running a
  shortest-path search from every lit detector over the whole graph.
* **Sparse Blossom vs PyMatching:** the same algorithm (identical matching weights on every shot we compared),
  ~130× slower at d = 7. That's the cost of pure Python against tuned C++, not of the algorithm.

## Local BP+UF

Global BP+UF was 80–560× slower than correlated PyMatching. Profiling showed BP already ran at compiled speed
(~5 ns per message update), so porting to C++ would gain maybe 2–3×. The real cost was the amount of work:
global BP updated every Tanner-graph edge (5,004 at d = 5, 17,156 at d = 7) every iteration, even with fewer
than 2 lit detectors, and 20–40% of shots ran all 30 iterations. `qec/bp_local.py` cuts the work three ways,
in one compiled kernel:

1. **BP only near the lit detectors**: checks within 1 hop (in the check graph) of a lit detector; far-away
   checks send a fixed "not flipped" message.
2. **10 BP iterations instead of 30**: Union-Find cleans up after BP anyway.
3. **Fused with Union-Find**: the beliefs → edge-cost step (a Python loop taking ~20% of the time) now happens
   inside the kernel, for the touched edges only, and there are no per-shot memory allocations.

Decode-only speed, µs per shot (1 thread / 12 threads), p = 0.1%:

| decoder | d=3 | d=5 | d=7 | d=9 | d=11 |
|---|---|---|---|---|---|
| PyMatching correlated (C++, 1 core) | 0.31 | 1.74 | 4.99 | 11.5 | 23.3 |
| BP+UF, global (30 iterations) | 11.3 / 9.15 | 159 / 80.3 | 1,037 / 370 | 3,907 / 1,463 | 10,284 / 5,589 |
| **BP+UF, local** | 3.90 / **0.72** | 50.3 / **9.43** | 264 / **52.7** | 856 / **183** | 2,202 / **640** |
| speed-up, local over global | 2.9× / 13× | 3.2× / 8.5× | 3.9× / 7.0× | 4.6× / 8.0× | 4.7× / 8.7× |

Where the speed-up comes from (same shots, one thread): capping BP at 10 iterations gives ~2.2× by itself
(d = 7, p = 0.3%: 2,519 → 1,131 µs) at no accuracy cost; locality plus fusing gives another 1.3× at d = 7 and
2.1× at d = 11. The gain grows with d because a 1-hop region is a smaller share of a bigger graph (43% of the
checks at d = 7, 21% at d = 11). The local kernel also scales better across threads (5× on 12 threads vs
~2.8× for the global version).

Accuracy (table at the top): same as global BP+UF at d = 7 (2.5e-6 vs 2.1e-6, intervals overlap), but **1.7×
worse at d = 5** (4.0e-5 vs 2.3e-5): at small d, a 1-hop region cuts off useful information. A 2-hop region
restores d = 5 accuracy (164 vs 158 errors in a 60k-shot same-shot test at p = 0.3%) but covers most of the
graph at these sizes, so it's no faster than global.

Gap to correlated PyMatching: from 208× (d = 7) and 441× (d = 11) on one thread down to 53× and 95×; with 12
threads, 10× at d = 7 and 27× at d = 11. Circuit-level noise links each detector to many others (space, time and
hyperedges), so BP neighbourhoods are large; closing the rest needs hardware parallelism, a cheaper BP schedule,
or running BP only on clusters where it matters.

## Takeaways

* **Correlated PyMatching is clearly best at low noise:** ~1.8× fewer errors than plain PyMatching or BP+UF at
  d = 7, and the steepest suppression (Λ ≈ 15–18).
* **BP+UF's edge shrinks at low p:** ~15% better than PyMatching at d = 5, level at d = 7 (at p = 0.4% it was
  1.8× better). BP helps most when errors are dense enough for correlations to matter.
* **Our MWPM and Sparse Blossom agree with PyMatching** at d = 3 and 5 (within error bars), as expected: all three
  are exact MWPM.
* **Weighted UF stays within ~1.3–1.8× of PyMatching** up to d = 9 at ~7× its per-shot time. Plain UF falls
  further behind with size (2.7× at d = 7, 3.1× at d = 9).
* **The lookup table is the most accurate decoder at d = 3** (~20% below matching): it uses exact fault
  probabilities, including the correlated ones matching ignores.
* **Greedy matching and BP alone don't suppress errors with distance** (Λ ≤ 1.6), consistent with their
  2-fault failures in `tests/test_metrics.py`.
* **Λ ≈ 10 for matching at p = 0.1%:** each +2 in distance cuts logical errors ~10×; PyMatching goes from
  2.7e-5 per round at d = 5 to ~4e-8 at d = 11.
* Our Union-Find kernel's time per shot still grows with code size even at low p (2.4 µs at d = 5, 46 µs at
  d = 11 on one thread), because the edge-parallel form scans every edge each round. A CPU version that only
  touches edges next to growing clusters would close most of the remaining gap to PyMatching.
