"""Generate notebooks/01_first_surface_code.ipynb (keeps the notebook diff-friendly to regenerate).

    .venv\\Scripts\\python tools\\make_notebook.py           # write the notebook
    .venv\\Scripts\\python tools\\make_notebook.py --check   # also run every code cell as a smoke test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md", """# Your first surface code

In this notebook you will:

1. build a surface code and look at it,
2. turn it into a noisy circuit with **Stim**,
3. sample thousands of noisy runs ("shots"),
4. decode them with **PyMatching** (minimum-weight perfect matching),
5. measure the **logical error rate** and watch it fall as the code grows.

No QEC background needed. If a word is unfamiliar, check the glossary in `docs/explainer.html`.
Run the cells top to bottom with Shift+Enter."""),
    ("code", """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))   # so `import qec` works from notebooks/

import numpy as np
import matplotlib.pyplot as plt
from qec import SurfaceCode, NoiseModel, detector_error_model, make_decoder"""),
    ("md", """## 1. Build a code

A distance-3 rotated surface code has 3×3 **data qubits** (`o`) and 8 **checks** (`X` and `Z`).
Each check is measured by its own **measure qubit**. `X` checks catch Z errors, and `Z` checks
catch X errors."""),
    ("code", """code = SurfaceCode(3)
print(code.describe())
print(code.ascii())"""),
    ("md", """## 2. Make it noisy

A **memory experiment**: prepare logical |0⟩, measure all checks for `rounds` rounds, then read out
every data qubit and ask whether the logical value survived.

`NoiseModel.circuit(p)` lets *every* operation fail with probability `p`: gates, resets,
measurements, and data qubits between rounds. That's the realistic "circuit-level" model."""),
    ("code", """p = 0.003
circuit = code.memory_circuit(NoiseModel.circuit(p), rounds=3, basis="Z")
print("qubits:", circuit.num_qubits, " detectors:", circuit.num_detectors, " observables:", circuit.num_observables)
print(str(circuit)[:900], "...")"""),
    ("md", """A **detector** is a check compared with itself one round earlier. With no errors it is always 0.
Seeing a 1 is a **detection event**.

Stim can also draw the circuit. In Jupyter this renders as an SVG (it can be big):"""),
    ("code", """circuit.diagram("timeslice-svg")"""),
    ("md", """## 3. The detector error model

Stim works out, for every possible fault, which detectors it sets off and whether it flips the
logical qubit. That list is the **detector error model** (DEM): exactly what a decoder needs."""),
    ("code", """dem = detector_error_model(circuit)
print("number of distinct error mechanisms:", dem.num_errors)
for line in str(dem).splitlines()[:8]:
    print(line)
print("...")
print("smallest number of faults that cause an undetected logical error:", len(circuit.shortest_graphlike_error()))"""),
    ("md", """`error(0.002) D0 D3` reads: "with probability 0.002, detectors 0 and 3 fire together."
`L0` means the fault also flips the logical observable. The last line is the
**code distance** seen at circuit level. It's 3, so no single fault, or pair of faults, can
silently break the logical qubit."""),
    ("md", "## 4. Sample and decode"),
    ("code", """sampler = circuit.compile_detector_sampler()
dets, actual = sampler.sample(20_000, separate_observables=True)
print("detector data shape:", dets.shape, "   mean detection events per shot:", dets.sum(1).mean().round(2))

decoder = make_decoder("pymatching", dem)
predicted = decoder.decode_batch(dets)
fails = (predicted != actual).any(axis=1)
print(f"logical error rate: {fails.mean():.4f}   (without decoding it would be {actual.mean():.4f})")"""),
    ("md", """## 5. Bigger codes, fewer errors (below threshold)

Repeat for distances 3, 5 and 7 and a few physical error rates. Below the threshold
(~0.7–1% for this noise model) the bigger code wins, and it wins by more as `p` shrinks."""),
    ("code", """def logical_error_rate(d, p, shots=20_000, decoder="pymatching", variant="css", bias=0.5):
    c = SurfaceCode(d, variant=variant).memory_circuit(NoiseModel.circuit(p, bias=bias), rounds=d)
    dem = detector_error_model(c)
    dets, obs = c.compile_detector_sampler().sample(shots, separate_observables=True)
    return (make_decoder(decoder, dem).decode_batch(dets) != obs).any(axis=1).mean()

ps = [0.002, 0.004, 0.006, 0.008, 0.012]
for d in (3, 5, 7):
    rates = [logical_error_rate(d, p) for p in ps]
    plt.loglog(ps, np.maximum(rates, 1e-5), "o-", label=f"d={d}")
plt.xlabel("physical error rate p"); plt.ylabel("logical error rate (d rounds)")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.title("Curves cross at the threshold")
plt.show()"""),
    ("md", """## 6. Your turn

Each exercise is a one-line change:

* **Decoders:** pass `decoder="union_find"` or `"union_find_weighted"` to `logical_error_rate`. How much worse is Union-Find?
* **Biased noise:** set `bias=100` (mostly Z errors). Then try `variant="xzzx"` too. Remember to check the X basis
  as well: `memory_circuit(..., basis="X")`.
* **Rectangular codes:** `SurfaceCode(dx=3, dz=7)` protects Z errors better than X errors. When would you want that?
* **Measurement-heavy noise:** use `NoiseModel.si1000(p)`, which is closer to superconducting hardware.

For proper threshold plots with error bars, use the command line:
`python bench.py --name mine --noise circuit --decoders pymatching union_find`."""),
    ("code", """# Example: XZZX vs CSS at strong Z bias, adding both memory bases
for variant in ("css", "xzzx"):
    total = 0
    for basis in ("Z", "X"):
        c = SurfaceCode(5, variant=variant).memory_circuit(NoiseModel.circuit(0.005, bias=100), basis=basis)
        dem = detector_error_model(c)
        dets, obs = c.compile_detector_sampler().sample(20_000, separate_observables=True)
        total += (make_decoder("pymatching", dem).decode_batch(dets) != obs).any(axis=1).mean()
    print(f"{variant:5s} d=5, eta=100, p=0.5%: logical error rate (Z+X) = {total:.4f}")"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    path = ROOT / "notebooks" / "01_first_surface_code.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print("wrote", path)
    if "--check" in sys.argv:
        import os

        import matplotlib
        matplotlib.use("Agg")
        os.chdir(path.parent)
        ns: dict = {}
        for kind, src in CELLS:
            if kind == "code":
                exec(compile(src, "<cell>", "exec"), ns)  # noqa: S102 - our own notebook source
        print("all code cells ran")


if __name__ == "__main__":
    main()
