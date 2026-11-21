# Judges' brief: what to say, what to claim, what to expect

Talking points and Q&A preparation for the 10-minute talk, collected from everything we measured and
discussed. Numbers point to where they came from, so anyone on the team can check them before quoting.

* Results: [`RESULTS.md`](../RESULTS.md) §7, [`results/`](../results/) (timestamped sweeps)
* How each decoder works: [`docs/DECODERS.md`](DECODERS.md) · interactive: [`docs/walkthroughs.html`](walkthroughs.html)
* Slides: the "Decoding the Surface Code" deck (claude.ai artifact; share link from its Share menu)

---

## 1. The judging criteria, and our answer to each

| criterion | our answer | evidence |
|---|---|---|
| **Number of implemented algorithms** | 10 decoders written by us, no decoding library inside: Union-Find, weighted UF, MWPM, Sparse Blossom, greedy matching (+2-opt), lookup table, BP, BP+OSD, BP+UF, local BP+UF | `qec/`, registry `qec.OURS` |
| **Quality of the code** | one shared graph builder, one decoder interface, 107 tests, metrics computed by `pytest`, every result reproducible from a command | `tests/`, `README.md` |
| **Parallelizability** | measured, not just claimed: *parallel depth* (sequential steps if every independent update runs at once) for UF and BP; multithreaded UF matches PyMatching's throughput at d = 7 | `qec/metrics.py::parallel_depth`, `results/` |
| **Novelty** | see §3: what is ours vs what we re-implemented | |
| **Presentation** | deck + interactive walkthroughs + this brief | |

## 2. Claims we can make (and the numbers behind them)

Circuit-level noise unless stated, "per round" logical error rates, same Stim circuits for every decoder.

1. **Our MWPM and our Sparse Blossom give the same answers as PyMatching.** MWPM: identical predictions on every
   shot tested. Sparse Blossom: identical total matching weight on every shot compared (thousands, d ≤ 9, high
   noise with blossoms forming and shattering). *(tests in `tests/test_new_decoders.py`)*
2. **BP+UF ties correlated PyMatching at moderate noise and beats standard MWPM by 1.8×**: d = 7, p = 0.4%:
   3.5 vs 3.5 vs 6.2 per 10,000 rounds. *(`out/ours_circuit`, RESULTS §7.2)*
3. **Our BP+OSD beats the ldpc library's BP+OSD**: 6.1e-4 vs 1.0e-3 per round at d = 5, p = 0.3%, and 2× faster
   (the library number has only 15 errors, so ±50%). *(RESULTS §7.1)*
4. **The lookup table beats both PyMatching modes at d = 3**: 2.2e-4 vs 2.7e-4 (plain) and 2.8e-4 (correlated)
   per round at p = 0.1%. *(`results/2026-11-21_1559_p1e-3_distance_sweep.md`)*
5. **With 12 threads, our Union-Find matches PyMatching's throughput at d = 7** (1.83 vs 1.89 µs per shot) and
   beats it at d = 3 and 5. PyMatching gets nothing from threads in one process. *(same results file)*
6. **Union-Find's parallel depth barely grows with the code**: 0.7 → 2.1 growth rounds from d = 3 to 9 while the
   code grows 9×. *(RESULTS §7.5)*
7. **Local BP+UF is 3–5× faster than global BP+UF on one thread and 7–9× on 12 threads** (d = 5–11, p = 0.1%), same
   accuracy from d = 7 up. The lesson: our BP was already at compiled speed (~5 ns per update), so the win came from
   doing less work, not from changing language. *(results file, "Local BP+UF")*
8. **Our stored results reproduce**: an independent rerun by a second agent matched all 60 overlapping points
   within 95% intervals. *(`results/2026-11-21_1704_claude2_verification_and_greedy_scaling.md`)*
9. **Distance-preserving ≠ most accurate**: Union-Find and MWPM fix every 2-fault error at d = 5; the BP hybrids
   (lowest error rates overall) miss 0.07–0.3%; greedy matching misses 2%, so its error rate *rises* with d.
   *(`tests/test_metrics.py::test_fault_weight_correction_d5`)*

## 3. What is ours and what isn't (say this before a judge does)

| re-implemented from the literature (credit it) | our own contribution |
|---|---|
| Union-Find (Delfosse & Nickerson 2017), weighted growth (Huang, Newman & Brown 2020) | the synchronous **edge-parallel UF kernel** and the parallel-depth metric |
| MWPM / blossom (Edmonds; Dennis et al.; Fowler) | the shared DEM → graph/check-matrix builder (identical to PyMatching's graph) |
| **Sparse Blossom** (Higgott & Gidney 2023, the algorithm inside PyMatching) | a from-scratch Python implementation, verified weight-for-weight against PyMatching |
| BP and BP+OSD (Panteleev & Kalachev 2019; Roffe et al. 2020) | our GF(2) solver on Python big-int bit vectors with combination sweep, which beats the ldpc library's OSD-0 here |
| **BP+UF = "belief-find"** (Higgott et al. 2023, same paper as belief-matching) | **local BP+UF**: BP restricted to the lit detectors' neighbourhood, fused with UF in one compiled kernel |
| greedy matching, lookup tables (standard) | 2-opt + boundary pre-pairing for greedy; the finding that greedy loses distance |
| correlated matching (Fowler 2013; PyMatching's `enable_correlations`) | the fault-weight metric that exposed distance-preserving ≠ most accurate |

Do **not** call BP+UF a new algorithm. It's belief-find; our parts are the implementation, the local version and the
measurements.

## 4. Claims to avoid

* "We beat PyMatching." Only true against *plain* PyMatching at moderate noise (BP+UF), at d = 3 (lookup table), and
  on multithreaded throughput (UF). Correlated PyMatching is 1.3–1.8× more accurate than BP+UF at p = 0.1%, and
  PyMatching is 4–6× faster per core than our UF.
- "Our decoder is faster than PyMatching" without saying *throughput with 12 threads*. Per call, PyMatching wins.
- "The threshold is X%" without the noise model. Ours: 14.4% code capacity, 4.7% phenomenological, ~0.7% per shot
  / ~1.2% per round circuit level. They differ by 20×.
* Results at d ≥ 9 with < 30 logical errors (they're flagged in the results docs).

## 5. Questions judges may ask, with answers

**How many errors does it take to cause a logical error?** About (d+1)/2, not d. A full row of d errors is the
smallest *undetectable* error, but the decoder picks the shortest explanation: if 3 of the 5 qubits in a row are
flipped (d = 5), the shortest explanation of the one lit check is the other 2, so error + correction completes the
row. This is why the logical error rate scales like p^((d+1)/2) and each +2 in distance needs one more fault (Λ).

**If one row fails, why aren't the other rows a backup?** The rows aren't copies. The patch stores *one* logical qubit,
and every row is the same logical X operator (they differ by stabilizers). The logical Z readout is a column parity,
which every row crosses once, so one bad row flips every column. Two bad rows cancel (each column crossed twice):
what matters is the parity of boundary-to-boundary crossings, not a majority vote.

**What would an X measurement show after a row of X errors?** Nothing changes: X errors commute with X checks and
with an X-basis readout. The row flips logical Z only. That's why there are Z-basis and X-basis memory experiments,
each blind to the other's failure, and why biased-noise results must run both.

**Is PyMatching MWPM? What's Sparse Blossom?** MWPM is the problem (cheapest pairing); blossom variants are
algorithms that solve it exactly. PyMatching 2 runs Sparse Blossom in C++: regions grow on the detector graph and
blossom steps happen when they touch, with no all-pairs distances. It's single-threaded and doesn't release Python's
GIL (we measured no gain from threads); parallelism comes from separate processes.

**What does correlated PyMatching do?** Two exact matchings: match, raise the likelihood of edges that come from the
same multi-detector fault as the chosen edges (e.g. the Z half of a Y error whose X half was used), match again.
Each pass is still blossom; the result is no longer the min-weight matching of the original graph. Cost ~2.6×;
known quirk: a few single faults fail at d = 3.

**Can its two passes run in parallel?** Not for one shot: pass 2 needs pass 1's answer. But shots are independent,
a shot splits into far-apart clusters that can each be matched on its own core (both passes), and pass 2 only needs
redoing on clusters with re-weighted edges. BP-based re-weighting removes the chain entirely.

**Why is BP+UF so much slower than correlated PyMatching? Would C++ fix it?** No. Our BP already costs ~5 ns per
message update (compiled speed). Global BP updated every Tanner-graph edge (5,004 at d = 5) every iteration even
with < 2 lit detectors, and 20–40% of shots ran the full 30 iterations. The gap is the amount of work, so the fix
is algorithmic: **local BP** (only near lit detectors) and fewer iterations. See §6.

**Do your decoders work in real time?** The sweeps decode all d rounds of a shot at once. Real systems decode
overlapping windows; our sliding-window study (PyMatching inside each window) shows a look-ahead of about d
rounds matches global decoding exactly, and no look-ahead costs 1.4–2.5×. Windows can be decoded in parallel, so
throughput scales with cores for any decoder. *(RESULTS §6)*

**Why does greedy matching get worse with distance?** It grabs the most attractive pair and strands two defects;
~2% of 2-fault errors fail at d = 5, so the effective distance stays ~2. 2-opt local search cuts that to 0.7%, but
the remaining failures need 3–5 pairs re-matched at once, which pairwise swaps can't find, so greedy+2-opt still
doesn't improve from d = 5 to 7 (Claude 2's scaling study in `results/`).

**"Are you beating PyMatching at high error rates?"** Plain PyMatching, yes: at d = 7, p ≥ 0.3%, BP+UF makes
0.6–0.8× its errors. Correlated PyMatching, no: it ties or beats BP+UF everywhere, and is 1.8× better at p = 0.1%.
Say which PyMatching.

**Shot vs round?** A shot is one full memory experiment: prepare, d rounds of syndrome measurement, read out. We
report per-round rates (per-shot rate converted with 1 − 2ε_round = (1 − 2ε_shot)^(1/d)) so different d compare fairly.

**What's the state of the art?** Matching (Sparse Blossom, correlated, Fusion/Micro Blossom on FPGA), Union-Find
hardware decoders (Helios, Riverlane's Collision Clustering), BP hybrids (BP+OSD, BP+LSD, belief-matching, IBM's
Relay-BP), search (Google's Tesseract), neural networks (AlphaQubit, best on real hardware data), tensor networks
(near-optimal, slow). Our ten decoders cover the matching, Union-Find and BP families.

## 6. Speed story in one slide

| step | d = 7, p = 0.1%, µs per shot |
|---|---|
| PyMatching (C++, 1 core) | 1.9 |
| our Union-Find, 1 thread → 12 threads (after fixing a 3 GB per-batch copy) | 9.0 → **1.8** |
| our Sparse Blossom (pure Python) vs our textbook MWPM | 248 vs 1,405 |
| global BP+UF → local BP+UF, 1 thread | 1,037 → 264 |
| global BP+UF → local BP+UF, 12 threads | 370 → **52.7** |
| correlated PyMatching (C++, 1 core), for reference | 5.0 |

## 7. Honest limitations

* All decoding here is on a laptop CPU; parallel-depth numbers stand in for hardware latency.
* BP+OSD degrades at d = 7 and high p (its combination sweep is too narrow there).
* Statistics at d ≥ 9 and p = 0.1% are thin for the slow decoders.
* Most sweeps are Z-basis memory under unbiased noise (fine for symmetric noise; biased-noise runs used both bases).
