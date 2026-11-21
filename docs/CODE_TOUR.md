# Code tour: how the toolkit works

Read this before changing `qec/`. It assumes you've skimmed `docs/explainer.html`
(chapters 3–6), so stabilizers, detectors and decoding aren't new.

## The pipeline in one picture

```
SurfaceCode(d, variant)          layout: which qubit is where, which checks touch which qubits
        │  .memory_circuit(noise, rounds, basis)
        ▼
stim.Circuit                     gates + noise channels + DETECTOR / OBSERVABLE_INCLUDE annotations
        │  detector_error_model(circuit)
        ▼
stim.DetectorErrorModel (DEM)    list of "error(p) D3 D7 L0" lines: every fault, what it sets off
        │                                   │
        │ compile_detector_sampler()        │ make_decoder(name, dem)
        ▼                                   ▼
detection events + true flips  ──►  decoder.decode_batch(dets) ──► predicted flips
                                                                     │
                                         logical error rate = mean(predicted != true)
```

Every experiment in the repo is this loop. Sinter (`bench.py`) runs it in parallel across
many (circuit, decoder) pairs and keeps statistics.

## 1. The layout (`qec/surface_code.py`, `SurfaceCode.__init__`)

* Data qubit (i, j) sits at coordinate (2i+1, 2j+1). Checks sit at even coordinates (2a, 2b).
* Checkerboard rule: a check is X-type when (a + b) is even.
* Bulk checks touch 4 qubits. Edge checks touch 2, and we keep only X-type on top/bottom and
  Z-type on left/right. That gives exactly n − 1 checks for n data qubits (asserted in the code).
* `Check.schedule` is the order in which a check touches its qubits: X checks go TL, TR, BL, BR and Z checks
  go TL, BL, TR, BR. This "N/Z" order keeps hook errors from shortening the distance, and
  `tests/test_surface_code.py::test_circuit_level_distance` proves it with Stim's
  `shortest_graphlike_error()`.
* XZZX: `self.deformed[q]` marks data qubits with a Hadamard applied. `physical(frame_pauli, q)` maps
  "the X of this qubit's frame" to the real Pauli. That one function is the whole XZZX code.

Try `print(SurfaceCode(5, variant="xzzx").ascii())`.

## 2. The circuit (`memory_circuit`)

Per round:

```
PAULI noise on data         (noise.data)    errors that happened between rounds
RX ancillas + Z_ERROR       (noise.reset)   prepare |+>, a bad reset flips it
4 layers of CX / CZ         (noise.gate2)   CX if the check applies X to that qubit, CZ if Z
MX ancillas (flip prob)     (noise.measure) read the check value
DETECTOR lines                               compare with the previous round
```

The detector rules are the key part:

* **Round 0:** only checks whose type matches the preparation basis are deterministic. Preparing |0⟩
  makes the Z checks +1, while X checks come out random. So round 0 has a detector only for those checks.
* **Later rounds:** `rec[-m+k]` XOR `rec[-2m+k]`, i.e. the same check now vs. last round.
* **Final data readout:** measure every data qubit in the basis. For each check of the matching type, the
  product of its data measurements must equal its last ancilla result. That's one more layer of detectors.
* **Observable:** the product of the data measurements along the logical operator.

`memory_circuit_ideal` skips ancillas: it measures each stabilizer directly with `MPP` (a perfect
or noisy multi-qubit Pauli measurement). That's how the code-capacity and phenomenological models are
built. It tracks **both** logicals by entangling with a perfect reference qubit (Z_L·Z_R and X_L·X_R
commute, while Z_L and X_L alone do not).

## 3. Noise (`qec/noise.py`)

`NoiseModel` is just six probabilities plus a bias. `append_1q` / `append_2q` write either Stim's
`DEPOLARIZE1/2` (when bias = 0.5) or `PAULI_CHANNEL_1/2` (biased). Biased two-qubit channels need
`approximate_disjoint_errors=True` when building the DEM. `detector_error_model()` always sets it, and so
does Sinter.

**To add a hardware-specific model:** add a classmethod preset (see `si1000`). To add a
new *kind* of fault (e.g. leakage), add a field and one `append` call in `memory_circuit`.

## 4. The detector error model

`detector_error_model(circuit)` asks Stim to push every possible fault through the circuit and record
which detectors and observables it flips. `decompose_errors=True` splits Y-type faults that set
off 4 detectors into two graph edges (an X part and a Z part) joined by `^`. Matching decoders need
plain edges. Correlated matching uses the `^` to remember that the two parts go together.

Useful one-liners:

```python
dem = detector_error_model(circuit)
print(dem.num_detectors, dem.num_errors)
print(len(circuit.shortest_graphlike_error()))        # circuit-level distance
circuit.diagram("matchgraph-svg")                     # picture of the decoding graph
circuit.diagram("detslice-with-ops-svg", tick=range(0, 8))
```

## 5. Decoders (`qec/decoders.py` and friends)

Every decoder has the same shape: constructed from a DEM (or a prebuilt `DemGraph`), with
`decode_batch(bool[shots, detectors]) -> bool[shots, observables]`. `docs/DECODERS.md` explains the
algorithms; this is where the code lives.

* **`dem_graph.py`**: `DemGraph.from_dem(dem)` is the shared foundation. It reads `dem.flattened()`,
  splits each error at `^` separators into graph-like components (edges), merges parallel edges with
  `xor_prob`, and also builds `H`, `L` and `priors` for the hypergraph view. `col_edges[j]` maps each
  mechanism to its graph edges (BP+UF uses it).
* **`union_find.py`**: `decode()` is the readable reference (dicts, cluster member lists).
  `_uf_kernel_py` is the same algorithm in the synchronous edge-parallel form, compiled with numba
  (`parallel=True` spreads shots over cores). `decode_batch(dets, edge_caps=...)` takes per-shot growth
  caps, and `last_batch_rounds` holds the growth rounds (parallel depth).
* **`bp.py`**: `MinSumBP` with two backends. `"numpy"` updates all messages with `np.ufunc.reduceat`
  per check, and `"numba"` (`_bp_kernel`) loops. They agree to 1e-14 (tested). `GF2Solver` is the OSD:
  an XOR basis keyed by leading bit, with each vector carrying the set of columns it combines (a
  Python int used as a bit set). `BPUnionFindDecoder` maps posteriors to edges with
  `edge_probs_from_mechanisms`, then to caps.
* **`matching.py`**: scipy Dijkstra from each defect, a syndrome graph with boundary copies, and networkx
  `max_weight_matching(maxcardinality=True)` on `BIG − distance`. `_path_obs` walks the predecessor
  array to XOR observable masks. `GreedyMatchingDecoder(local_search=True)` adds 2-opt.
* **`lookup.py`**: dict from syndrome (packed to a Python int) to observable mask; misses go to
  weighted UF.
* **pymatching / pymatching_correlated / bposd / belief_matching**: thin wrappers around the libraries,
  used as baselines only.
* **`sliding_window.py`**: splits detectors by their time coordinate into windows, builds one small matching
  graph per window (future edges become boundary edges, past ones are dropped), commits only
  corrections that start in the commit region, and flips the affected later detectors.

**Sinter adapters:** `RegistrySinterDecoder(name)` wraps any registry decoder (bit-unpacking with
`np.unpackbits(..., bitorder="little")`). Every name in `OURS` is registered automatically.

## 6. Statistics (`qec/analysis.py`)

* `aggregate(stats)` groups Sinter results by series (variant / decoder / noise / bias), distance and p.
  It **adds up** the Z- and X-basis failure rates when both were run.
* `wilson()` gives error bars that behave well with 0 errors, so never plot a bare rate without one.
* `thresholds()` finds where the curves for consecutive distances cross (log-log interpolation).
  The estimates are noisy. Check that d5/d9 and d9/d13 roughly agree before quoting a number.
* `fit_scaling()` and `project()` drive `estimate.py`.

## 7. Tests (`tests/`)

* `test_decoders.py`, `test_surface_code.py`, `test_nn.py`: the original checks. Every single fault is
  corrected, circuit-level distance equals code distance, and the sliding window with one giant window is
  bit-identical to global decoding.
* `test_new_decoders.py`: our graph equals PyMatching's, our MWPM equals PyMatching's on random shots,
  every decoder fixes every single fault, numba BP == numpy BP, and numba UF ≈ Python UF.
* `test_metrics.py`: **computes decoder metrics** (logical error rate, Λ, fault-weight correction,
  threshold, latency, parallel depth) via `qec/metrics.py` and records them with the `metrics_log`
  fixture. `conftest.py` prints them as a table after the run and saves `out/metrics/test_metrics.json`.

Run `.venv\Scripts\python -m pytest -q` after any change to `qec/` (~1 minute).

## Gotchas we hit (so you don't)

0. **Numba kernels compile on first call** (a few seconds, then cached in `__pycache__`). Time decoders
   after a warm-up call, which `metrics.latency` does. Set `numba.set_num_threads(1)` for a fair latency
   comparison with single-threaded PyMatching.

1. **Z-basis memory hides Z errors.** Under biased noise, a CSS code looks amazing in a Z-basis memory.
   Use `--basis both`.
2. **Z_L and X_L can't both be observables** in one circuit (they anticommute). Use the reference-qubit trick
   or run two circuits.
3. **Coarse weighted-UF growth** (2 steps for the lightest edge) caused ties that broke single-fault
   correction at d = 3. Resolution 4 fixed it.
4. **Belief-matching / BP-OSD are slow** (milliseconds per shot at d=7). Cap their `--max-shots`.
5. **Thresholds depend on the noise model.** 14% (code capacity), ~4% (phenomenological) and ~0.7–1%
   (circuit level) are all "the surface-code threshold". Always say which one you mean.
