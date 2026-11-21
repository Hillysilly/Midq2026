# Claude 2: reproduction check, error plots and why greedy matching stops scaling

**Author:** Claude 2 (a second Claude Code session, separate from the agent working on BP+UF)
**Run:** 2026-11-21, 14:30–17:04 CST · Windows 11, Python 3.11.9, packages from `requirements.txt`

## TL;DR

* **The stored results reproduce.** A fresh circuit-noise run of 5 decoders matches `out/ours_circuit` on
  all 60 overlapping points (95% intervals overlap).
* **Tests:** 89 passed, 1 skipped (by design: the lookup table only exists for d = 3).
* **New error-bar plots** for every stored circuit and code-capacity sweep: `out/error_plots/`.
* **Greedy + 2-opt doesn't improve with distance:** at d = 7 it has the same logical error rate as at d = 5
  (p = 0.2–0.5%), while PyMatching improves 2–7×. It fails on small clusters of 2–3 faults at every size,
  and fixing them needs 3–5 pairs re-matched at once, which 2-opt can't do.
* **Caveat for the headline:** BP+UF ties correlated PyMatching at p ≥ 0.3%, but at p = 0.1% correlated
  PyMatching makes ~1.8× fewer errors (see `2026-11-21_1559_p1e-3_distance_sweep.md`). Quote the p.

## 1. Setup and tests

There was no `.venv`, and the `py` launcher in the README isn't installed on this laptop. So:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q          # 89 passed, 1 skipped in 105 s
```

Running pytest rewrites `out/metrics/test_metrics.json`. That caused the merge conflict on that file. It was
resolved by keeping the newer side (`34b8809`, the one with Sparse Blossom's records).

## 2. Reproduction check (`out/verify_circuit/`)

```powershell
.\.venv\Scripts\python bench.py --name verify_circuit --noise circuit --distances 3 5 7 `
    --ps 0.003 0.005 0.008 0.012 `
    --decoders pymatching pymatching_correlated union_find union_find_weighted bp_uf `
    --max-shots 20000 --max-errors 150
```

All 60 (decoder, d, p) points have 95% Wilson intervals that overlap the stored `out/ours_circuit` points.
The largest ratios (fresh/stored) were 0.75–1.29, all at p = 0.3%, where counts are smallest. Overlay plot:
`out/error_plots/verify.png`.

## 3. Error plots (`out/error_plots/`, `tools/error_plots.py`)

95% Wilson error bars. Our decoders are in colour, library baselines in gray.

| file | shows |
|---|---|
| `ler_circuit.png` | per-round logical error rate vs p, circuit noise, d = 3 / 5 / 7 panels |
| `ratio_vs_mwpm.png` | each decoder's error rate ÷ PyMatching's (below 1 = better), d = 5 and 7 |
| `lambda.png` | Λ = ε(d=5)/ε(d=7) vs p; Λ = 1 marks the threshold |
| `ler_code_capacity.png` | per-shot logical error rate, code capacity, d = 5 / 9 / 13 |
| `verify.png` | the fresh run on top of the stored one |

Regenerate: `.\.venv\Scripts\python tools\error_plots.py --verify out/verify_circuit`

What they show:
* **BP+UF tracks correlated PyMatching** at d = 7 for p = 0.3–1.2% (ratio to PyMatching ≈ 0.6–0.8).
* **Plain UF is ~1.5–2× worse than PyMatching at d = 7**; weighted UF is ~1–1.5× worse.
* **BP+OSD's Λ drops below 1 at p ≈ 0.7–0.8%**, the early threshold noted in RESULTS §7.2.
* **Code capacity:** BP+UF and BP+OSD pull clearly ahead at d = 9 and 13.

## 4. Greedy matching stops improving after d = 5

### 4.1 Simulation (`out/greedy_scaling/`)

```powershell
$env:NUMBA_NUM_THREADS=1
.\.venv\Scripts\python bench.py --name greedy_scaling --noise circuit --distances 3 5 7 `
    --ps 0.002 0.003 0.005 --decoders pymatching greedy_matching greedy_matching_2opt `
    --max-shots 30000 --max-errors 150
```

Logical error rate per round (errors / shots in brackets):

| p | decoder | d = 3 | d = 5 | d = 7 |
|---|---|---|---|---|
| 0.2% | PyMatching | 1.1e-3 (100/30k) | 2.1e-4 (31/30k) | 2.9e-5 (6/30k)ᵃ |
| | Greedy | 9.9e-4 (89/30k) | 1.5e-3 (185/25k) | 1.1e-3 (172/23k) |
| | **Greedy + 2-opt** | 1.1e-3 (95/30k) | 6.6e-4 (99/30k) | **6.9e-4** (144/30k) |
| 0.3% | PyMatching | 2.3e-3 (151/22k) | 6.8e-4 (101/30k) | 1.9e-4 (40/30k) |
| | Greedy | 2.9e-3 (173/20k) | 3.1e-3 (187/12k) | 2.7e-3 (176/9.5k) |
| | **Greedy + 2-opt** | 2.2e-3 (172/26k) | 1.9e-3 (175/19k) | **2.0e-3** (178/13k) |
| 0.5% | PyMatching | 5.6e-3 (154/9.4k) | 2.9e-3 (164/12k) | 1.4e-3 (157/17k) |
| | Greedy | 6.6e-3 (165/8.5k) | 9.2e-3 (206/4.6k) | 1.2e-2 (180/2.3k) |
| | **Greedy + 2-opt** | 6.9e-3 (198/9.8k) | 7.6e-3 (191/5.2k) | **7.3e-3** (195/4.0k) |

ᵃ 6 errors: order of magnitude only.

* **Greedy + 2-opt ties PyMatching at d = 3**, is ~3× worse at d = 5 and stays flat from d = 5 to 7. At
  p = 0.2%, d = 7 it's ~24× worse than PyMatching, though that PyMatching point rests on 6 errors.
* Plain greedy gets *worse* with distance at p = 0.5%.
* The "thresholds" in `out/greedy_scaling/summary.txt` (0.36%, 0.41%) are crossings of nearly flat curves.
  Don't quote them as thresholds.

### 4.2 Why: bunched-fault injection (`tools/greedy_fault_probe.py`)

Inject k faults that sit close together (each within 2 hops of the first in the fault graph), circuit
noise p = 0.3% weights, and check which decoders correct them. Seed 7.

| code | bunched faults | Greedy fails | Greedy + 2-opt fails | MWPM fails |
|---|---|---|---|---|
| d = 5 | 2 | 4.2% | 1.5% | 0 |
| d = 7 | 2 | 0.07% | 0.07% | 0 |
| d = 7 | 3 | 1.6% | 0.8% | 0 |
| d = 9 | 3 | 0.27% | 0.13% | 0 |
| d = 9 | 4 | 1.2% | 0.6% | 0 |

(3000 samples per row at d = 5 and 7, 1500 at d = 9.) At d = 3 it corrects all 219 single faults, which is
all d = 3 requires.

* **Greedy keeps failing on small fault clusters at every size.** At d = 7 it fails some 2-fault and 3-fault
  sets; MWPM's smallest failing set is about (d+1)/2 faults. A bigger code adds more places where a small
  bad cluster can happen, which cancels the gain. So the error rate stays flat.
* **Every failure is a genuinely worse matching, not a tie.** In all 25 failures from random (unbunched)
  2-fault sets at d = 5, greedy + 2-opt's total weight was strictly higher than MWPM's.
* **2-opt can't reach the fix.** Going from the 2-opt pairing to the MWPM pairing means re-pairing 3–5 pairs
  at once (d = 5: 45 of 45 failures needed 3–4 pairs; d = 7, 3 faults: 24 of 24 needed 3–5). 2-opt only
  swaps partners between 2 pairs, and every such swap from there raises the cost, so it stops at a local
  minimum. Blossom's augmenting paths re-pair any number of pairs in one step. That's exactly what greedy lacks.
* **It's not mainly about the boundary.** In most failures greedy and MWPM send the same number of defects
  to the boundary (d = 5: 42 of 45; d = 7: 31 of 37). The pairings among defects are what's wrong.

**Recommendation:** don't invest further in greedy as a contender. A k-opt search would fix more cases but
costs more and still doesn't guarantee distance, while Union-Find already guarantees it at ~5 µs/shot
(greedy + 2-opt: ~330 µs/shot in Python). Keep greedy in the talk as the "why exact matching matters" example.

## 5. Suggested next approaches (besides BP+UF, which the other agent has)

1. **Correlated two-pass Union-Find.** Decode, raise the probability of edges that come from the same fault
   as the chosen edges (the parts of Y-type faults), then decode again. This is the trick behind correlated
   PyMatching, which leads at p = 0.1%. It keeps UF's parallelism and distance. `docs/PROJECT_IDEAS.md` #5.
2. **Pick the most likely answer from an ensemble** (Shutty et al. 2024, "Harmony"). Run UF weighted, BP+UF and
   UF with perturbed weights; keep the correction with the highest probability under the error model. It
   should cover BP+UF's 2-fault misses. Needs decoders to return their fault sets, not just the logical flip.
3. **Localised statistics decoding (LSD)** in place of OSD (Hillmann et al. 2024), to fix BP+OSD's d = 7
   collapse and its long serial OSD step. `ldpc` 2.4.1 should include `BpLsdDecoder` as a baseline.
4. **Union-Find that only touches active edges** on CPU (already noted in the p = 0.1% results). It overlaps
   with `qec/union_find.py` changes the other agent is making, so coordinate first.
5. **Quick runs:** our decoders under biased noise (XZZX, η = 100) and SI1000. Only PyMatching has been
   run on those so far.

## Files added by Claude 2

* `tools/error_plots.py`, `tools/greedy_fault_probe.py`
* `out/error_plots/`, `out/verify_circuit/`, `out/greedy_scaling/`
* this file
