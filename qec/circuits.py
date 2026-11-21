"""Noisy surface-code memory circuits built with Stim.

A "memory experiment" (Fowler, Sec. VIII): initialise every data qubit, run
`rounds` rounds of stabilizer measurement, then measure every data qubit and
check whether the logical observable flipped.
"""
from __future__ import annotations

import stim


def memory_circuit(
    distance: int,
    p: float,
    rounds: int | None = None,
    basis: str = "z",
) -> stim.Circuit:
    """Rotated surface-code memory experiment under uniform circuit-level noise.

    Every noise channel uses the same strength `p`:
      * depolarizing noise after each 1- and 2-qubit Clifford gate,
      * depolarizing noise on data qubits at the start of each round,
      * bit-flip before each measurement and after each reset.

    Args:
        distance: code distance d (odd, >= 3).
        p: physical error probability.
        rounds: rounds of stabilizer measurement (default: d).
        basis: "z" keeps |0_L> and measures Z_L; "x" keeps |+_L> and measures X_L.
    """
    if basis not in ("x", "z"):
        raise ValueError("basis must be 'x' or 'z'")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds if rounds is not None else distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Detector error model (Fowler's "detector hypergraph").

    `decompose_errors=True` splits hyperedges (e.g. a Y error that sets off 4
    detectors) into graph edges, so matching-style decoders can use the model.
    `approximate_disjoint_errors=True` is needed for biased PAULI_CHANNEL_1/2 noise. Sinter
    uses the same flags, so results from both paths agree.
    """
    return circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
