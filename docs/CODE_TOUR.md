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

## 5. Decoders (`qec/decoders.py`, `qec/union_find.py`, `qec/sliding_window.py`)

Every decoder has the same shape: constructed from a DEM, with `decode_batch(bool[shots, detectors]) ->
bool[shots, observables]`.

* **pymatching / pymatching_correlated**: thin wrappers around PyMatching.
* **union_find**: builds its graph from PyMatching's `edges()` so it sees the exact same graph.
  `decode()` runs (1) cluster growth with a disjoint-set forest (`find` / `union`, union by size,
  path halving), stopping when every cluster has even parity or touches the boundary, then (2) peeling
  a BFS spanning tree from the leaves inward. Weighted mode gives each edge `cap = round(4·w/w_min)` growth
  steps. The comments explain why 4 and not 2.
* **sliding_window**: splits detectors by their time coordinate into windows, builds one small matching
  graph per window (future edges become boundary edges, past ones are dropped), commits only
  corrections that start in the commit region, and flips the affected later detectors.

**Sinter adapters:** a `sinter.Decoder` subclass whose `compile_decoder_for_dem(dem)` returns a
`sinter.CompiledDecoder` with `decode_shots_bit_packed`. See `UnionFindSinterDecoder` and
`CorrelatedMatchingSinterDecoder`. Unpack with `np.unpackbits(..., bitorder="little")`.

## 6. Statistics (`qec/analysis.py`)

* `aggregate(stats)` groups Sinter results by series (variant / decoder / noise / bias), distance and p.
  It **adds up** the Z- and X-basis failure rates when both were run.
* `wilson()` gives error bars that behave well with 0 errors, so never plot a bare rate without one.
* `thresholds()` finds where the curves for consecutive distances cross (log-log interpolation).
  The estimates are noisy. Check that d5/d9 and d9/d13 roughly agree before quoting a number.
* `fit_scaling()` and `project()` drive `estimate.py`.

## 7. Tests (`tests/`)

The checks that matter most:

* every single fault is corrected (UF, weighted UF, MWPM; correlated MWPM from d=5),
* circuit-level distance equals the code distance for CSS and XZZX, square and rectangular,
* sliding window with one giant window gives *bit-identical* results to global decoding.

Run `.venv\Scripts\python -m pytest -q` after any change to `qec/`. It takes a few seconds.

## Gotchas we hit (so you don't)

1. **Z-basis memory hides Z errors.** Under biased noise, a CSS code looks amazing in a Z-basis memory.
   Use `--basis both`.
2. **Z_L and X_L can't both be observables** in one circuit (they anticommute). Use the reference-qubit trick
   or run two circuits.
3. **Coarse weighted-UF growth** (2 steps for the lightest edge) caused ties that broke single-fault
   correction at d = 3. Resolution 4 fixed it.
4. **Belief-matching / BP-OSD are slow** (milliseconds per shot at d=7). Cap their `--max-shots`.
5. **Thresholds depend on the noise model.** 14% (code capacity), ~4% (phenomenological) and ~0.7–1%
   (circuit level) are all "the surface-code threshold". Always say which one you mean.
