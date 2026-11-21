# midq2026: surface-code decoders, built from scratch

Our entry for the **Northrop Grumman decoder-implementation challenge**. We implemented **ten
decoding algorithms ourselves** (no decoding library inside), test functions that compute decoder
metrics, and a benchmark harness that compares them against the standard libraries on
[Stim](https://github.com/quantumlib/Stim)-generated syndromes. We follow the decoder review
[arXiv:2307.14989](https://arxiv.org/abs/2307.14989).

| our decoder | file | one line |
|---|---|---|
| Union-Find (plain, weighted) | `qec/union_find.py` | grow clusters, merge, peel; compiled edge-parallel kernel |
| MWPM | `qec/matching.py` | Dijkstra + boundary copies + blossom; matches PyMatching on 100% of shots |
| Sparse Blossom | `qec/sparse_blossom.py` | PyMatching's algorithm, from scratch: exact MWPM grown on the detector graph |
| Greedy matching (+2-opt) | `qec/matching.py` | fast pairing by savings; shows why exact matching matters |
| Lookup table | `qec/lookup.py` | most-likely correction per syndrome, up to 2 faults (d = 3) |
| Belief propagation | `qec/bp.py` | min-sum on the Tanner graph; numba and whole-array numpy versions |
| BP+OSD | `qec/bp.py` | BP, then our own GF(2) ordered-statistics decoding with combination sweep |
| **BP+UF** | `qec/bp.py` | BP's beliefs become per-shot Union-Find weights: our most accurate, and parallel |
| Local BP+UF | `qec/bp_local.py` | BP only near the lit detectors, fused with UF in one kernel: 3–9× faster than BP+UF |

**Start here:** [`docs/DECODERS.md`](docs/DECODERS.md) (how each works, cost, parallelism),
[`RESULTS.md`](RESULTS.md) §7 (measured numbers) and
[`notebooks/02_decoders_from_scratch.ipynb`](notebooks/02_decoders_from_scratch.ipynb) (watch each decoder work).

**Headline results** (circuit-level noise, same Stim circuits for every decoder):
* **BP+UF** (belief-find, Higgott et al. 2023; our implementation on our own BP and Union-Find) ties PyMatching's best (correlated) mode at
  d = 7 and makes 1.8× fewer logical errors than standard MWPM. Both stages are parallel.
* **Union-Find** runs at ~1 µs per round on one CPU thread (PyMatching: 0.34 µs), with 2–6 parallel growth
  rounds from d = 3 to 9, and never loses code distance.
* Our **BP+OSD** beats the ldpc library's BP+OSD (6.1e-4 vs 1.0e-3 per round at d = 5, 2× faster).
* Our **MWPM** makes identical predictions to PyMatching on every shot tested.
* Our **Sparse Blossom** (PyMatching's own algorithm, written from scratch in Python) finds matchings of exactly
  PyMatching's weight on every shot we compared, 3–6× faster than our textbook MWPM.
* With 12 threads our **Union-Find** matches single-call PyMatching's throughput at d = 7 and beats it at d = 3 and 5.
* **Local BP+UF** runs BP only near the lit detectors: 3–5× faster than global BP+UF on one thread and 7–9× on
  12 threads, with the same accuracy from d = 7 up.
* Judges' Q&A prep and the claims we can and can't make: [`docs/JUDGES_BRIEF.md`](docs/JUDGES_BRIEF.md).
* **Distance-preserving ≠ most accurate:** the BP hybrids miss 0.1–0.3% of two-fault errors at d = 5;
  greedy matching misses 2%, so bigger codes don't help it.

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q              # ~90 tests incl. decoder metrics, ~1 min
.\.venv\Scripts\python metrics_report.py         # metrics table + plots in out/metrics/
```

`pytest` doesn't just pass/fail. `tests/test_metrics.py` *computes* logical error rates, Λ,
thresholds, fault-weight correction, latency and parallel depth for every decoder, prints them as
a table at the end of the run, and saves them to `out/metrics/test_metrics.json`.

---

## The rest of the toolkit

Simulate noisy surface codes, decode them several ways, and measure thresholds. All of it is
built on [Stim](https://github.com/quantumlib/Stim), [PyMatching](https://github.com/oscarhiggott/PyMatching)
and [Sinter](https://pypi.org/project/sinter/), following the decoder review [arXiv:2307.14989](https://arxiv.org/abs/2307.14989) and Fowler's surface-code notes (not redistributed here).

**Reading order for new teammates**

0. [`docs/DECODERS.md`](docs/DECODERS.md): our decoders, for the talk, then
   [`notebooks/02_decoders_from_scratch.ipynb`](notebooks/02_decoders_from_scratch.ipynb) to watch them work.
1. [`docs/explainer.html`](docs/explainer.html): open it in a browser. An interactive primer: click errors
   onto a surface code, decode them, and see our results. No background needed.
2. [`notebooks/01_first_surface_code.ipynb`](notebooks/01_first_surface_code.ipynb): build, simulate and decode
   your first code in about 15 cells (`pip install notebook`, or open it in VS Code).
3. [`RESULTS.md`](RESULTS.md): what we measured, with the numbers to quote.
4. [`docs/PROJECT_IDEAS.md`](docs/PROJECT_IDEAS.md): what to build, ranked by effort, plus a pitch skeleton.
5. [`docs/CODE_TOUR.md`](docs/CODE_TOUR.md): read before changing anything in `qec/`.

## Setup (Windows, Python 3.10+)

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q          # ~90 tests, ~1 min
```

## What's here

| File | What it does |
|---|---|
| `qec/surface_code.py` | Our own rotated surface code: **CSS or XZZX**, square or **rectangular** (dx ≠ dz), hook-safe CNOT order. Builds circuit-level memory experiments and ideal-measurement (code-capacity / phenomenological) ones. |
| `qec/noise.py` | `NoiseModel`: code capacity, phenomenological, circuit level, SI1000, plus **biased noise** (η = pZ/(pX+pY)). |
| `qec/circuits.py` | Stim's built-in generated circuits, and `detector_error_model()`, the one helper every decoder uses. |
| `qec/decoders.py` | One registry: `make_decoder(name, dem)` for scripts, `sinter_decoders()` for sweeps. `OURS` lists our decoders, `BASELINES` the library ones. |
| `qec/dem_graph.py` | Stim detector error model → check matrices (for BP/OSD/lookup) and decoding graph (for matching/UF). |
| `qec/bp.py`, `qec/matching.py`, `qec/lookup.py` | Our BP, BP+OSD, BP+UF, MWPM, greedy matching and lookup-table decoders. |
| `qec/metrics.py` | Decoder metrics: logical error rate, Λ, threshold, fault-weight correction, latency, parallel depth. |
| `metrics_report.py` | Every metric for every decoder in one table, plus accuracy-vs-speed and parallel-depth plots. |
| `run_new_decoders.ps1` | Threshold sweeps of our decoders vs the library baselines. |
| `qec/union_find.py` | Union-Find decoder written from scratch (unweighted and weighted): readable Python version plus a compiled edge-parallel kernel. |
| `qec/sliding_window.py` | Sliding-window decoder for real-time decoding studies. |
| `qec/analysis.py` | Error bars, threshold crossings, Λ, scaling fits, distance projections. |
| `quickstart.py` | One experiment, decoded two ways, with optional SVG circuit diagrams. |
| `bench.py` | **The main tool.** Threshold sweeps over code × noise × decoder, with plots and threshold estimates. |
| `threshold.py` | Simpler threshold demo using Stim's generated circuits. |
| `bias_sweep.py` | Threshold vs bias, CSS vs XZZX. |
| `realtime.py` | Sliding-window accuracy vs buffer size, plus decoder latency vs syndrome rate. |
| `estimate.py` | Physical-qubit budget for a target logical error rate. |
| `speed.py` | Decoder speed vs accuracy (µs per shot and per round, by distance). |
| `qec/nn_decoder.py`, `nn_demo.py` | Neural-network (MLP) decoder in pure numpy, trained on Stim samples and compared with MWPM. |
| `run_experiments.ps1`, `run_followups.ps1` | Re-run every experiment behind `RESULTS.md`. |
| `tools/export_explainer_data.py` | Copies results into the interactive explainer. |
| `notebooks/01_first_surface_code.ipynb` | A step-by-step beginner notebook. |
| `notebooks/02_decoders_from_scratch.ipynb` | Each of our decoders on real shots, then every metric (regenerate with `tools/make_notebook_decoders.py --check`). |
| `tools/talk_figures.py` | Slide-sized figures for the talk (`out/talk/`). |

## Common commands

```powershell
$py = ".\.venv\Scripts\python"

& $py quickstart.py -d 5 --p 0.003 --diagrams                        # one run + SVGs in out/
& $py bench.py --name cc --noise code-capacity --decoders pymatching union_find
& $py bench.py --name circ --noise circuit --basis both               # both memory bases
& $py bench.py --name xzzx --noise circuit --bias 100 --variants css xzzx --basis both
& $py bench.py --name si --noise si1000 --distances 3 5 7 9
& $py realtime.py --distances 3 5 7
& $py estimate.py --stats out\circuit_decoders\stats.csv --p 0.001 0.002 --target 1e-12
```

`bench.py` options: `--noise {code-capacity,phenomenological,circuit,si1000}`, `--bias`,
`--variants css xzzx`, `--distances`, `--ps`, `--rounds`, `--basis {Z,X,both}`,
`--decoders` (any name from `qec.DECODER_NAMES`, e.g. `pymatching bp_uf bp_osd union_find_weighted`),
`--max-shots`, `--max-errors`, `--workers`. Results go to `out/<name>/`, and an interrupted run
picks up where it stopped.

## Using it from Python

```python
from qec import SurfaceCode, NoiseModel, detector_error_model, make_decoder

code = SurfaceCode(5, variant="xzzx")                 # or SurfaceCode(dx=3, dz=7)
print(code.ascii())                                   # quick picture of the layout
circuit = code.memory_circuit(NoiseModel.circuit(1e-3, bias=10), rounds=5, basis="Z")
dem = detector_error_model(circuit)
dets, obs = circuit.compile_detector_sampler().sample(10_000, separate_observables=True)
pred = make_decoder("pymatching", dem).decode_batch(dets)
print("logical error rate:", (pred != obs).any(axis=1).mean())
```

## Adding your own decoder

1. Write a class whose constructor takes a `stim.DetectorErrorModel`, with
   `decode_batch(dets: bool[shots, n_det]) -> bool[shots, n_obs]`. Copy the pattern in
   `qec/union_find.py`.
2. Register it in `_FACTORIES` in `qec/decoders.py` and add its name to `OURS`; Sinter gets it
   automatically through `RegistrySinterDecoder`.
3. Add its name to the parametrised tests in `tests/test_new_decoders.py` (every single fault must be
   corrected) and `tests/test_metrics.py` (metrics).
4. `bench.py --decoders pymatching your_decoder` gives a head-to-head plot.

## Conventions

* Data qubit (i, j) sits at coordinate (2i+1, 2j+1); checks sit at even coordinates. X checks are where
  (a+b) is even. `Z_L` is the top row (weight dz) and `X_L` is the left column (weight dx).
* `memory_circuit_ideal` tracks **both** logical observables (obs 0 = Z_L, obs 1 = X_L) using a
  perfect reference qubit. That's the metric behind the review's code-capacity numbers.
* Per-round logical error rate = ½(1 − (1 − 2·p_shot)^(1/rounds)), the same formula Sinter uses.
