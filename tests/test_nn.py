"""The numpy MLP decoder learns something useful on a small code."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qec import NoiseModel, SurfaceCode  # noqa: E402
from qec.nn_decoder import MLPDecoder  # noqa: E402


def test_mlp_beats_no_decoding():
    circuit = SurfaceCode(3).memory_circuit(NoiseModel.circuit(0.005), rounds=3)
    nn = MLPDecoder.train_on_circuit(circuit, hidden=(128,), train_shots=300_000, val_shots=5_000, epochs=2,
                                     batch=256, verbose=False)
    dets, obs = circuit.compile_detector_sampler(seed=5).sample(20_000, separate_observables=True)
    pred = nn.decode_batch(dets)
    assert pred.shape == obs.shape
    nn_err = np.any(pred != obs, axis=1).mean()
    raw_err = np.any(obs, axis=1).mean()
    assert nn_err < 0.5 * raw_err
