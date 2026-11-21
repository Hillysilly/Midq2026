# Hackathon project ideas

Each idea lists what already exists in this repo, the missing piece, and what a convincing
demo looks like. Ordered roughly from "done in an afternoon" to "ambitious".

What judges usually reward in QEC projects: **a clear claim, backed by a plot with error bars, that
compares against a fair baseline** (same circuits, same noise, same shots). `bench.py` gives you all
three almost for free. Spend the saved time on the idea and the story.

---

## Tier 1: quick wins (a few hours)

### 1. "Bias is a feature": XZZX vs CSS on realistic qubits
* **Have:** `SurfaceCode(variant="xzzx")`, biased `NoiseModel`, `bias_sweep.py`, `bench.py --basis both`.
* **Add:** realistic bias values for a specific platform (cat qubits η ≈ 100–1000; spin qubits
  η ≈ 10–100), and rectangular XZZX codes (`dx != dz`) sized to the bias.
* **Demo:** qubits needed to reach 1e-9 for CSS vs XZZX vs rectangular XZZX on one platform.
  `estimate.py` does the projection.

### 2. Decoder shoot-out with a real-time angle
* **Have:** MWPM, UF, weighted UF, BP+OSD and belief-matching through one interface.
* **Add:** timing per shot vs distance (`decode_batch` with `time.perf_counter`), and accuracy vs
  speed on one scatter plot.
* **Demo:** "which decoder would you put on an FPGA?" A Pareto front of accuracy vs latency.

### 3. Hardware-calibrated noise model
* **Have:** `NoiseModel` with separate gate / measurement / reset / idle strengths.
* **Add:** a published device's numbers (Google Willow, IBM Heron or a neutral-atom paper) as a preset.
* **Demo:** "on device X the threshold and Λ would be …", plus which error source matters most
  (scale one parameter at a time).

---

## Tier 2: solid projects (a day)

### 4. Better Union-Find
* **Have:** readable pure-Python UF in `qec/union_find.py`, with single-fault tests.
* **Add:** any of: growing the smallest cluster first (Delfosse–Nickerson), a "peeling from
  the boundary last" tweak, or rewriting the hot loop in numba / Rust for honest timing.
* **Demo:** threshold and speed vs PyMatching. Closing the accuracy gap while staying near-linear is a real result.

### 5. Correlated X/Z decoding (review §6, eq. 22)
* **Have:** `pymatching_correlated`, PyMatching's built-in two-pass correlated matching. It already
  cuts circuit-level logical errors by about 25% and raises the code-capacity threshold a lot (see RESULTS.md).
* **Add:** the same idea for Union-Find (re-weight Z edges after decoding X, per the review's eq. 22),
  or iterate it (the "recursive" variant the review cites), or combine it with XZZX under bias.
* **Demo:** correlated UF vs plain UF vs correlated MWPM: accuracy and speed.

### 6. Neural-network decoder
* **Have:** `qec/nn_decoder.py`, a numpy MLP baseline (`nn_demo.py`). At d = 3 it already **beats MWPM**
  (1.53% vs 1.75% logical error at p = 0.5%). At d = 5 a plain MLP falls well behind (see RESULTS.md):
  it doesn't know the lattice geometry and needs far more data.
* **Add:** a geometry-aware model in PyTorch: a 3D CNN over (x, y, round) using the detector coordinates
  from `dem.get_detector_coordinates()`, or a recurrent/transformer model over rounds (the AlphaQubit idea).
  Or a hybrid: let the NN re-weight MWPM edges.
* **Demo:** NN vs MWPM vs correlated MWPM at d = 3, 5, 7, plus inference time. Be honest about how it scales.

### 7. Leakage / erasure-aware decoding (review §4.5)
* **Have:** the noise hooks in `qec/noise.py`.
* **Add:** heralded erasures: a qubit whose error location is *known*. Set those edges' weight to ~0
  (MWPM) or pre-grow them (UF already handles erasures naturally).
* **Demo:** how much a herald raises the threshold. This is the pitch behind erasure qubits
  (neutral atoms, dual-rail transmons).

---

## Tier 3: ambitious

### 8. Real-time decoding system study
* **Have:** `qec/sliding_window.py`, latency measurement in `realtime.py`.
* **Add:** parallel windows (decode windows 1, 3, 5 at once and patch the seams; Skoric et al. 2023),
  and a simulated backlog queue with a target syndrome rate.
* **Demo:** the minimum decoder speed or core count needed for d = 25 at 1 µs/round.

### 9. Lattice surgery CNOT in Stim
* **Have:** the layout and syndrome-extraction code in `SurfaceCode` (Fowler §XI, Fig. 11).
* **Add:** two patches, a merge region measured for d rounds, then a split, with logical observables for the
  joint XX / ZZ measurement.
* **Demo:** the logical error rate of a two-patch merge vs a plain memory experiment.

### 10. Resource estimate for a real algorithm
* **Have:** `estimate.py`, and fitted A and p_th from our own data.
* **Add:** T-count and logical-qubit count for a target algorithm, magic-state factory overhead,
  and the reaction-time limit.
* **Demo:** "N physical qubits and H hours at p = 1e-3", with a sensitivity plot.

---

## Pitch skeleton (5 minutes)

1. **Problem (30 s):** 1e-3 physical vs 1e-12 needed. Show the threshold plot.
2. **Idea (60 s):** the one change we made, explained with the explainer's surface-code widget.
3. **Evidence (2 min):** one or two plots with error bars and a fair baseline. State the noise model.
4. **So what (60 s):** turn it into qubits or time saved (`estimate.py`).
5. **Limits and next steps (30 s):** what we didn't model.
