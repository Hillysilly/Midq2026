"""Checks for the hand-built surface code, the noise models and the sliding-window decoder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qec import NoiseModel, SurfaceCode, detector_error_model, make_decoder  # noqa: E402
from qec.noise import pauli_probs, two_qubit_pauli_probs  # noqa: E402
from qec.sliding_window import SlidingWindowDecoder  # noqa: E402


def graph_distance(circuit) -> int:
    return len(circuit.shortest_graphlike_error())


@pytest.mark.parametrize("variant", ["css", "xzzx"])
@pytest.mark.parametrize("dx,dz", [(3, 3), (5, 5), (3, 5), (5, 3)])
def test_circuit_level_distance(variant, dx, dz):
    """Hook-safe CNOT order: a circuit-level fault distance equal to the code distance."""
    code = SurfaceCode(dx=dx, dz=dz, variant=variant)
    noise = NoiseModel.circuit(1e-3)
    # A Z-basis memory fails through X-type logicals (weight dx); an X-basis one through Z-type ones (dz).
    assert graph_distance(code.memory_circuit(noise, basis="Z")) == dx
    assert graph_distance(code.memory_circuit(noise, basis="X")) == dz


@pytest.mark.parametrize("variant", ["css", "xzzx"])
def test_ideal_circuits(variant):
    code = SurfaceCode(5, variant=variant)
    assert graph_distance(code.memory_circuit_ideal(NoiseModel.code_capacity(0.05))) == 5
    assert graph_distance(code.memory_circuit_ideal(NoiseModel.phenomenological(0.02), rounds=3)) == 5


def test_layout_counts():
    code = SurfaceCode(dx=3, dz=7)
    assert code.num_data == 21 and len(code.checks) == 20
    assert len(code.logical_z) == 7 and len(code.logical_x) == 3


def test_noise_probabilities():
    assert np.allclose(pauli_probs(0.3), (0.1, 0.1, 0.1))
    px, py, pz = pauli_probs(0.3, bias=10)
    assert np.isclose(px + py + pz, 0.3) and np.isclose(pz / (px + py), 10)
    assert np.allclose(two_qubit_pauli_probs(0.15), [0.01] * 15)
    assert np.isclose(sum(two_qubit_pauli_probs(0.15, bias=100)), 0.15)


def test_biased_noise_builds():
    circuit = SurfaceCode(3, variant="xzzx").memory_circuit(NoiseModel.circuit(1e-3, bias=100))
    assert detector_error_model(circuit).num_detectors == circuit.num_detectors


def test_sliding_window_matches_global_with_one_window():
    circuit = SurfaceCode(3).memory_circuit(NoiseModel.circuit(0.005), rounds=9)
    dem = detector_error_model(circuit)
    dets, obs = circuit.compile_detector_sampler(seed=1).sample(3000, separate_observables=True)
    glob = make_decoder("pymatching", dem).decode_batch(dets)
    one = SlidingWindowDecoder(dem, commit=100, buffer=0).decode_batch(dets)
    assert np.array_equal(glob, one)


def test_sliding_window_with_buffer_is_close_to_global():
    circuit = SurfaceCode(3).memory_circuit(NoiseModel.circuit(0.005), rounds=12)
    dem = detector_error_model(circuit)
    dets, obs = circuit.compile_detector_sampler(seed=2).sample(5000, separate_observables=True)
    glob = np.any(make_decoder("pymatching", dem).decode_batch(dets) != obs, axis=1).mean()
    win = np.any(SlidingWindowDecoder(dem, commit=3, buffer=3).decode_batch(dets) != obs, axis=1).mean()
    assert win <= glob * 1.3 + 0.005
