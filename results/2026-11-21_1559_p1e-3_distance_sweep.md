# Logical error rate at p = 0.1%, by decoder and distance

**Run:** 2026-11-21, 15:59 CST · raw data: `out/p1e-3/stats.csv` · plot: `out/p1e-3/plot.png`

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
```

(The d = 11 points for correlated PyMatching and both Union-Finds come from an earlier, stopped run with
the same settings.)

## Logical error rate per round

| decoder | d=3 | d=5 | d=7 | d=9 | d=11 | Λ (per +2 in d) | µs/shot at d=5 (see Speed) |
|---|---|---|---|---|---|---|---|
| **PyMatching correlated** (baseline) | 2.8e-4 | **1.8e-5** | **1.2e-6** | **6.7e-8**ᵃ | <1e-7 (0 errors) | **15–18** | 4.1 |
| **BP+UF** (ours) | 3.0e-4 | 2.3e-5 | 2.1e-6 | 2.2e-7ᵇ | – | 10–13 | 420 |
| PyMatching (baseline) | 2.7e-4 | 2.7e-5 | 2.2e-6 | 1.9e-7 | 3.6e-8ᵃ | 10–12 | 1.9 |
| BP+OSD (ours) | 2.5e-4 | 2.8e-5 | 3.3e-6ᵇ | <4e-6 (0 errors) | – | 8–9 | 570 |
| MWPM (ours) | 2.6e-4 | 2.9e-5 | – | – | – | 9 | 750 |
| UF weighted (ours) | 3.5e-4 | 3.4e-5 | 3.8e-6 | 3.4e-7 | – | 9–11 | 14 |
| Union-Find (ours) | 3.2e-4 | 3.5e-5 | 5.9e-6 | 6.1e-7 | 8.5e-8ᵇ | 6–10 | 8.6 |
| Lookup table (ours) | **2.2e-4** | – | – | – | – | – | – |
| Greedy + 2-opt (ours) | 2.6e-4 | 1.6e-4 | – | – | – | 1.6 | 370 |
| Greedy (ours) | 3.0e-4 | 4.3e-4 | – | – | – | 0.7 (gets worse) | 360 |
| BP alone (ours) | 3.6e-3 | 4.5e-3 | – | – | – | 0.8 (gets worse) | 600 |

Most points have 100+ logical errors (±20% at 95% confidence).
ᵃ 8–35 errors: roughly ±30–60%. ᵇ 1–7 errors: order of magnitude only.
"–" = not run at that size (too slow: the slow decoders, BP+OSD at d=11; impossible: lookup beyond d=3).
Λ = ε_d / ε_{d+2}, the range over the distance pairs with enough errors. µs/shot is one CPU thread inside a
Sinter worker.

## Speed

Time per **round** in µs (time per shot in brackets), from Sinter's own accounting: CPU time per shot for
**Stim sampling + decoding**, one thread per worker, 11 workers sharing the laptop's cores. Same runs as the
table above.

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

ᶜ Almost all Python overhead (packing each shot into a dictionary key); the table read itself is one memory
access, which in hardware is a single clock cycle.

What it shows:

* **Only PyMatching keeps up with a 1 µs syndrome cycle on one core**, up to about d = 9 (1.1 µs per round once
  sampling is subtracted). Correlated PyMatching keeps up to d = 5. Everything else needs several cores,
  parallel windows or hardware.
* **Our Union-Find is 4–7× slower than PyMatching per round from d = 5 up**, and the gap widens with size
  (4.4× at d = 5, 7× at d = 11). At d = 3 fixed per-call overhead dominates, so the ratio there (~29×) says
  little about the algorithm. Its edge-parallel kernel scans every edge each growth round, so on one CPU core it pays for the
  whole graph even when only a few checks are lit. On parallel hardware that scan is one step per round.
* **The BP hybrids are 100–1,500× slower than PyMatching on a CPU**, and their time per shot grows
  quickly with d (bigger graph × more iterations, because BP converges less often). Nearly all of it is BP, the
  part that parallelises on a GPU or FPGA.
* **Our MWPM costs ~400× PyMatching at d = 5** for the same answers: the price of building the full graph
  between lit checks and using a general-purpose blossom implementation instead of Sparse Blossom.
* These numbers run higher than the quiet single-thread benchmark in `out/metrics/report.md` (for example
  Union-Find 8.6 vs 5.6 µs per shot at d = 5), because 11 workers compete for cores and sampling is included.

## Takeaways

* **Correlated PyMatching is clearly best at low noise:** ~1.8× fewer errors than plain PyMatching or BP+UF at
  d = 7, and the steepest suppression (Λ ≈ 15–18).
* **BP+UF's edge shrinks at low p:** ~15% better than PyMatching at d = 5, level at d = 7 (at p = 0.4% it was
  1.8× better). BP helps most when errors are dense enough for correlations to matter.
* **Our MWPM agrees with PyMatching** at d = 3 and 5 (within error bars), as expected.
* **Weighted UF stays within ~1.3–1.8× of PyMatching** up to d = 9 at ~7× its per-shot time. Plain UF falls
  further behind with size (2.7× at d = 7, 3.1× at d = 9).
* **The lookup table is the most accurate decoder at d = 3** (~20% below matching): it uses exact fault
  probabilities, including the correlated ones matching ignores.
* **Greedy matching and BP alone don't suppress errors with distance** (Λ ≤ 1.6), consistent with their
  2-fault failures in `tests/test_metrics.py`.
* **Λ ≈ 10 for matching at p = 0.1%:** each +2 in distance cuts logical errors ~10×; PyMatching goes from
  2.7e-5 per round at d = 5 to ~4e-8 at d = 11.
* Our Union-Find kernel's time per shot grows with code size even at low p (8.6 µs at d = 5, ~150 µs at
  d = 11), because the edge-parallel form scans every edge each round. A CPU version that only touches edges
  next to growing clusters would fix that.
