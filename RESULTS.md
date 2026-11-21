# Results

Everything here comes from `run_experiments.ps1`, `run_followups.ps1` and `run_new_decoders.ps1` (Stim + Sinter, 12-core laptop).
**Our own decoders are in §7.**
Plots live in `out/<experiment>/plot.png`, raw counts in `out/<experiment>/stats.csv`.
The same numbers drive the charts in [`docs/explainer.html`](docs/explainer.html).

Error bars are 95% Wilson intervals. Each threshold is where the curves for consecutive distances
cross. Two pairs of distances are listed so you can see how stable the estimate is.

## 1. We reproduce the review's code-capacity thresholds

Rotated CSS surface code, depolarizing noise on data qubits only, perfect measurements, both logical
observables tracked. Distances 5, 9 and 13, up to 100k shots or 500 errors per point.

| decoder | ours (d5/d9, d9/d13) | review, Fig. 27 |
|---|---|---|
| MWPM (PyMatching) | **14.4%** (14.4, 14.5) | 14.0% |
| MWPM, correlated | **16.6%** (17.2, 15.9) | not compared |
| Union-Find | ~13.9% (12.9, 14.9) | 11.6% |
| Union-Find, weighted | ~13.5% (14.2, 12.8) | – |
| BP+OSD-0 / Tensor network | not swept (too slow in Python) | 13.9% / 18.5% |

Takeaways:
* Our MWPM agrees with the review to within statistical and finite-size error.
* Correlated matching, which uses the fact that a Y error is an X and a Z at once, gives about +2 points of
  threshold for 2–3× the decode time. It's the easiest accuracy win available.
* The Union-Find crossings move around a lot between distance pairs. Treat them as ~13–14%. The review's
  11.6% comes from a different UF variant and different distances.

## 2. Thresholds depend on the noise model

MWPM, CSS code:

| noise model | threshold | experiment |
|---|---|---|
| code capacity | 14.4% (per shot) | `out/code_capacity` |
| phenomenological (data + measurement errors, q = p) | 4.7% (per round) | `out/phenomenological` |
| circuit level (every operation fails with prob p) | **0.7%** per shot (d rounds); 1.3% per round | `out/circuit_decoders` |

Quoting "the" surface-code threshold without naming the noise model is meaningless. These three
differ by about 20×.

Per shot vs per round: a shot of a distance-d experiment runs d rounds. The curves cross at a different p
depending on whether you compare failure per shot (the convention in most papers, giving ~0.7% here) or per
round (~1.3%). Say which one you mean.

Circuit-level thresholds for every decoder (crossings of d3/d5 and d5/d7):

| decoder | per shot | per round |
|---|---|---|
| MWPM | 0.64%, 0.71% | 1.18%, 1.42% |
| MWPM, correlated | 0.83%, 0.91% | 1.36%, 1.31% |
| Union-Find | 0.52%, 0.54% | 1.03%, 0.89% |
| Union-Find, weighted | 0.79%, 0.71% | 1.49%, 1.11% |
| Belief-matching (d ≤ 5 only) | 0.75% | – |

## 3. Decoder shoot-out at circuit level

d = 3, 5, 7 with d rounds, Z-basis memory. Logical error rate per round at p = 0.3%:

| decoder | d = 3 | d = 5 | d = 7 | µs/shot at d=5 (`speed.py`) |
|---|---|---|---|---|
| MWPM | 2.3e-3 | 6.9e-4 | 2.2e-4 | 1.6 |
| MWPM, correlated | 2.5e-3 | **4.9e-4** | **1.2e-4** | 4.5 |
| Union-Find | 2.7e-3 | 1.1e-3 | 4.4e-4 | 60 (Python) |
| Union-Find, weighted | 2.9e-3 | 7.6e-4 | 2.4e-4 | 75 (Python) |
| Belief-matching | 2.0e-3 | **5.0e-4** | – | 1,735 |

* **Correlated MWPM is the best value:** at d = 7 it has about half the logical error rate of MWPM, for 2.8× the time.
  Belief-matching is as accurate at d = 5 but about 400× slower.
* Plain Union-Find is the least accurate. Weighting its growth recovers most of the gap to MWPM.
* Speed is single-threaded, batched, on this laptop (`out/speed/summary.txt`). Our Union-Find is pure Python, so
  its wall-clock time says nothing about the algorithm, which is near-linear.
* Correlated MWPM is slightly *worse* than MWPM at d = 3. PyMatching's correlated pass mis-decodes a few
  single Y-type faults at that size (see the note in `qec/decoders.py`). From d = 5 up it wins clearly.

### Neural-network baseline (`nn_demo.py`, circuit noise p = 0.5%, 200k fresh test shots)

| | d = 3 (MLP 256-128, 2M training shots) | d = 5 (MLP 512-256, 3M training shots) |
|---|---|---|
| neural net | **1.53%** | 4.10% |
| MWPM | 1.75% | 1.38% |
| MWPM, correlated | 1.67% | **1.03%** |

A plain MLP **beats both matching decoders at d = 3**, because it learns the correlations MWPM ignores. But it's
**3× worse than MWPM at d = 5**, because it doesn't know the lattice geometry and the number of possible syndromes
explodes. Geometry-aware models (CNNs over space-time, transformers like AlphaQubit) are how ML decoders
scale. See `docs/PROJECT_IDEAS.md`, idea 6.

## 4. Biased noise: the XZZX code wins big

**Code capacity, threshold vs bias η = pZ/(pX+pY)** (`out/bias_sweep`):

| η | 0.5 | 1 | 3 | 10 | 30 | 100 | 1000 |
|---|---|---|---|---|---|---|---|
| CSS | 15.0% | 15.0% | 11.1% | 9.8% | 9.6% | 9.8% | 9.6% |
| XZZX | 16.1% | 16.6% | 20.4% | 25.3% | 34.9% | 44.8% | 46.9% |

Same qubits, same decoder. Only the stabilizers change (a Hadamard on every other qubit). The XZZX threshold
heads toward the 50% hashing bound, as in Bonilla Ataides et al. (2021).

**Circuit level, η = 100, Z- and X-basis memory failures added** (`out/bias100`):

| | threshold (per shot / per round) | d = 7 at p = 0.2% | d = 7 at p = 0.5% |
|---|---|---|---|
| CSS | 0.6–0.8% / 1.0–1.5% | 8.7e-5 per round | 2.7e-3 per round |
| XZZX | **1.8% / 2.6%** | **8.6e-6** (10× lower) | **2.1e-4** (13× lower) |

At η = 100 the XZZX code more than doubles the circuit-level threshold and cuts logical errors by an
order of magnitude at the same qubit count. Our two-qubit noise model is "bias-preserving" (ZI, IZ, ZZ
dominate). The literature says that with ordinary CNOTs, which turn Z errors into X errors, the advantage
shrinks. We haven't tested that yet, and it would make a good extension (add a non-bias-preserving
`gate2` channel to `NoiseModel`). This is why cat qubits with bias-preserving gates are the natural fit for XZZX.

Watch out: a CSS code run as a *Z-basis* memory looks excellent under Z-biased noise, because it can't see
Z errors at all. Always run `--basis both`.

## 5. Realistic superconducting noise (SI1000) and Λ

SI1000 noise (measurement 5p, gates p, idle p/10). Error suppression factor Λ = ε_d / ε_{d+2}:

| p | Λ (MWPM), d3→5, d5→7 | Λ (correlated), d3→5, d5→7 |
|---|---|---|
| 0.10% | 8.2, 8.8 | 10.0, 10.4 |
| 0.15% | 4.9, 5.8 | 7.3, 7.1 |
| 0.20% | 4.2, 3.7 | 5.1, 5.0 |
| 0.30% | 2.7, 2.5 | 3.7, 2.9 |

Google's 2024 below-threshold experiment reported Λ ≈ 2.1. In this model that matches p ≈ 0.3%. Each
halving of the physical error rate roughly doubles Λ.

**Resource projection** (`estimate.py`, target 1e-12 per logical qubit per round, 100 logical qubits, ×2 for
routing and factories):

| p | uniform circuit noise: d / physical qubits | SI1000: d / physical qubits |
|---|---|---|
| 0.05% | 15 / 90k | 17 / 115k |
| 0.1% | 19 / 144k | 23 / 211k |
| 0.2% | 27 / 291k | 35 / 490k |

## 6. Real-time (sliding-window) decoding

30-round memory, circuit noise p = 0.4%, commit = d rounds per window (`out/realtime`):

| buffer (rounds) | 0 | 1 | 2 | 3 | 5 | 7 | global |
|---|---|---|---|---|---|---|---|
| d = 3 | 0.143 | 0.104 | 0.102 | 0.101 | 0.101 | 0.101 | 0.101 |
| d = 5 | 0.087 | 0.060 | 0.051 | 0.048 | 0.047 | 0.047 | 0.047 |
| d = 7 | 0.051 | 0.036 | 0.027 | 0.023 | 0.021 | 0.021 | 0.021 |

(Logical error rate over 30 rounds.)

* With **no buffer**, the logical error rate is 1.4–2.5× higher: the decoder commits to bad guesses at the window edge.
* A buffer of **about d rounds** recovers the accuracy of decoding everything at once.
* **Throughput vs latency:** batched PyMatching decodes d = 5 at 0.32 µs per round on one core, faster than a
  superconducting chip's 1 µs cycle. A *single* windowed call from Python takes 8–40 µs, mostly Python
  overhead, so `realtime.py` reports "backlog grows". Real systems use C++/FPGA decoders and decode windows in
  parallel. That gap is a good hackathon topic (`docs/PROJECT_IDEAS.md`, idea 8).

## 7. Our own decoders (the challenge entry)

Eight decoders written from scratch (`docs/DECODERS.md`), compared with the library baselines on the
same Stim circuits. Reproduce with `run_new_decoders.ps1`, `metrics_report.py` and `pytest`.

### 7.1 Every metric at one operating point (`out/metrics/report.md`)

| decoder | ours? | logical error / round | 1-fault fail | 2-fault fail | µs / shot | µs / round | parallel depth (mean / max) |
|---|---|---|---|---|---|---|---|
| PyMatching corr. | baseline | 4.21e-04 (42/20000) | 0.0% | 0.0% | 4.7 | 0.93 | – |
| BP+UF (ours) | yes | 5.61e-04 (56/20000) | 0.0% | 0.1% | 394.9 | 78.99 | 13.2 / 42 |
| BP+OSD (ours) | yes | 6.11e-04 (61/20000) | 0.0% | 0.3% | 763.5 | 152.69 | 57.2 / 150 |
| PyMatching | baseline | 6.92e-04 (69/20000) | 0.0% | 0.0% | 1.7 | 0.34 | – |
| MWPM (ours) | yes | 7.35e-04 (11/3000) | 0.0% | 0.0% | 1008.9 | 201.78 | – |
| UF weighted | yes | 7.82e-04 (78/20000) | 0.0% | 0.0% | 10.1 | 2.03 | 4.2 / 12 |
| BP+OSD (ldpc) | baseline | 1.00e-03 (15/3000) | 0.1% | 1.3% | 1672.8 | 334.56 | – |
| Union-Find | yes | 1.10e-03 (110/20000) | 0.0% | 0.0% | 5.6 | 1.12 | 1.5 / 4 |
| Greedy+2opt | yes | 1.75e-03 (26/3000) | 0.0% | 0.7% | 324.2 | 64.84 | – |
| Greedy | yes | 3.38e-03 (50/3000) | 0.0% | 2.1% | 302.7 | 60.53 | – |
| BP | yes | 1.76e-02 (1640/20000) | 3.3% | 8.9% | 336.6 | 67.32 | 12.5 / 30 |

Circuit-level noise p = 0.3%, d = 5, 5 rounds, Z memory, **one CPU thread** for every decoder (fair
against single-threaded PyMatching). Parallel depth = sequential steps if every independent update runs
at once (UF growth rounds, BP iterations; OSD counted as rank(H) steps). Slow decoders used 3000 shots,
so their error bars are wide (see the counts in brackets).

* **BP+UF is our most accurate decoder**, ahead of PyMatching's standard MWPM and behind only its
  correlated mode at d = 5. At d = 7 it ties correlated MWPM (7.2).
* **Our MWPM = PyMatching:** identical predictions on every shot tested (`test_our_mwpm_equals_pymatching`).
  It's ~600× slower: the price of an explicit syndrome graph plus networkx's pure-Python blossom.
* **Our BP+OSD beats the ldpc library's** (`bposd`, OSD-0, 20 iterations): 6.1e-4 vs 1.0e-3 per round and 2×
  faster, thanks to the combination sweep. (The library number has only 15 errors, so ±50%.)
* **Union-Find on one thread: 5.6 µs per shot**, 3.3× PyMatching's C++. That's ~1 µs per round, which
  keeps up with a superconducting chip.

### 7.2 Logical error rate vs p, d = 3, 5, 7 (`out/ours_circuit`)

Per-round logical error rate, circuit-level noise, 40k shots or 200 errors per point:

| decoder | d=7, p=0.3% | d=7, p=0.4% | d=7, p=0.5% | threshold (per round, d5/d7) |
|---|---|---|---|---|
| PyMatching correlated | 1.1e-4 | **3.5e-4** | **8.5e-4** | > 1.2% (not bracketed) |
| **BP+UF (ours)** | **1.1e-4** | **3.5e-4** | 1.1e-3 | > 1.2% (not bracketed) |
| BP+OSD (ours) | 2.1e-4 | 6.1e-4 | 1.9e-3 | 0.69% (OSD struggles at d = 7, high p) |
| PyMatching | 1.9e-4 | 6.2e-4 | 1.4e-3 | 1.17% (d3/d5) |
| UF weighted (ours) | 2.9e-4 | 7.2e-4 | 1.8e-3 | 1.06% |
| Union-Find (ours) | 4.6e-4 | 1.3e-3 | 2.8e-3 | 0.82% |

At d = 7, p = 0.4%, BP+UF makes **1.8× fewer logical errors than standard MWPM** and ties correlated
MWPM. BP+OSD's d = 7 curve crosses d = 5 early: at high p, BP rarely converges, and OSD-CS with λ = 30
doesn't search widely enough. A larger λ or a better BP would help.

### 7.3 Code-capacity thresholds (`out/ours_code_capacity`, d = 5/9/13, compare review Fig. 27)

| decoder | ours (d5/d9, d9/d13) | review |
|---|---|---|
| **BP+UF (ours)** | **16.3%** (16.3, not crossed) | – |
| BP+OSD (ours) | 15.4% (16.0, 14.8) | 13.9% (BP+OSD) |
| PyMatching | 14.6% (13.9, 15.3) | 14.0% |
| UF weighted (ours) | 13.9% (13.3, 14.5) | – |
| Union-Find (ours) | 12.4% (11.9, 12.8) | 11.6% |

### 7.4 Fault tolerance (`tests/test_metrics.py::test_fault_weight_correction_d5`)

Fraction of random k-fault combinations that fail at d = 5 (1500 samples each; k = 1 is exhaustive):

| decoder | 1 fault | 2 faults | 3 faults |
|---|---|---|---|
| Union-Find, UF weighted, MWPM | 0 | **0** | 0.8–1.3% |
| BP+UF | 0 | 0.07% | 0.8% |
| BP+OSD | 0 | 0.27% | 2.7% |
| Greedy+2-opt | 0 | 0.7% | 3.7% |
| Greedy | 0 | 2.1% | 6.9% |

**Distance-preserving ≠ most accurate.** The BP hybrids have the lowest error rates but miss a few
two-fault errors, because when BP oscillates its posteriors rank the true fault columns hundreds of
places down (we traced individual cases: ranks 394–482 of 1677). Blending BP's output with the prior
made things worse. Greedy matching loses distance outright, so its error rate doesn't fall with d.

### 7.5 Parallel depth vs distance (`out/metrics/depth.png`, p = 0.3%)

| decoder | d = 3 | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| Union-Find (growth rounds) | 0.7 | 1.6 | 1.9 | 2.1 |
| UF weighted | 1.8 | 4.3 | 5.2 | 5.9 |
| BP (iterations, cap 30) | 4.2 | 12.9 | 21.6 | 27.2 |
| BP+UF | 4.3 | 13.6 | 23.1 | 29.7 |

The code grows 9× in qubits (27× in detectors) from d = 3 to 9; Union-Find's depth grows 3×. BP's
grows because it converges less often on bigger codes, but it's capped, so the worst case is constant.

## 8. Sanity checks (all in `tests/`, 89 tests)

* Every single fault in the detector error model is corrected by MWPM and both Union-Find variants (d = 3, 5)
  and by correlated MWPM (d = 5).
* Our hand-built circuits reach the full circuit-level distance (d) for CSS and XZZX, square and rectangular,
  in both bases. Hook errors are oriented correctly.
* Our circuits' logical error rates match Stim's built-in generator within statistical noise.
* The in-browser decoder in the explainer fixes every error of weight ≤ (d−1)/2 and matches PyMatching's
  logical error rate (checked with QuickJS).
* Sliding-window decoding with one giant window is bit-identical to global decoding.
* Our decoding graph has exactly PyMatching's edges and weights; our MWPM makes identical predictions;
  numba BP agrees with numpy BP to 1e-14; numba Union-Find agrees with the Python reference on ≥ 98% of shots
  (the rest are equal-weight spanning-tree ties).

