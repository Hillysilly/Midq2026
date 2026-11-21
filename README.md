# midq2026 — surface-code QEC toolkit

Simulate noisy surface codes, decode them several ways, and measure thresholds. All of it is
built on [Stim](https://github.com/quantumlib/Stim), [PyMatching](https://github.com/oscarhiggott/PyMatching)
and [Sinter](https://pypi.org/project/sinter/), following the decoder review [arXiv:2307.14989](https://arxiv.org/abs/2307.14989) and Fowler's surface-code notes (not redistributed here).

**Reading order for new teammates**

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
.\.venv\Scripts\python -m pytest -q          # 24 tests, ~10 s
```

## What's here

| File | What it does |
|---|---|
| `qec/surface_code.py` | Our own rotated surface code: **CSS or XZZX**, square or **rectangular** (dx ≠ dz), hook-safe CNOT order. Builds circuit-level memory experiments and ideal-measurement (code-capacity / phenomenological) ones. |
| `qec/noise.py` | `NoiseModel`: code capacity, phenomenological, circuit level, SI1000, plus **biased noise** (η = pZ/(pX+pY)). |
| `qec/circuits.py` | Stim's built-in generated circuits, and `detector_error_model()`, the one helper every decoder uses. |
| `qec/decoders.py` | One registry: `make_decoder(name, dem)` for scripts, `sinter_decoders()` for sweeps. MWPM, Union-Find, weighted Union-Find, BP+OSD, belief-matching. |
| `qec/union_find.py` | Union-Find decoder written from scratch (unweighted and weighted). |
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
`--decoders pymatching union_find union_find_weighted bposd belief_matching`,
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
2. Register it in `_FACTORIES` in `qec/decoders.py`, and wrap it for Sinter the way
   `UnionFindSinterDecoder` is wrapped.
3. Add a single-fault test in `tests/test_decoders.py`: every single fault must be corrected.
4. `bench.py --decoders pymatching your_decoder` gives a head-to-head plot.

## Conventions

* Data qubit (i, j) sits at coordinate (2i+1, 2j+1); checks sit at even coordinates. X checks are where
  (a+b) is even. `Z_L` is the top row (weight dz) and `X_L` is the left column (weight dx).
* `memory_circuit_ideal` tracks **both** logical observables (obs 0 = Z_L, obs 1 = X_L) using a
  perfect reference qubit. That's the metric behind the review's code-capacity numbers.
* Per-round logical error rate = ½(1 − (1 − 2·p_shot)^(1/rounds)), the same formula Sinter uses.
