"""Noise models (review §4) that can be injected into the surface-code circuits.

Three standard levels, from easiest to most realistic:

* **code capacity**: Pauli errors on the data qubits once; stabilizers are measured perfectly.
* **phenomenological**: data errors every round, and every stabilizer measurement can report the
  wrong bit with probability q.
* **circuit level**: every reset, gate, idle step and measurement in the real syndrome-extraction
  circuit can fail. This is what hardware actually looks like.

Biased noise: the bias is eta = pZ / (pX + pY) (review §4.2). eta = 0.5 is depolarizing
(pX = pY = pZ = p/3); eta -> infinity is pure dephasing (only Z errors).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import stim

DEPOLARIZING = 0.5


def pauli_probs(p: float, bias: float = DEPOLARIZING) -> tuple[float, float, float]:
    """(pX, pY, pZ) with total p and bias eta = pZ / (pX + pY)."""
    if bias == float("inf"):
        return 0.0, 0.0, p
    px = p / (2 * (1 + bias))
    return px, px, p * bias / (1 + bias)


def two_qubit_pauli_probs(p: float, bias: float = DEPOLARIZING) -> list[float]:
    """Probabilities for Stim's PAULI_CHANNEL_2 (15 entries in IX, IY, IZ, XI, ... ZZ order).

    Each Pauli pair P1⊗P2 gets weight r(P1)·r(P2), with r(X) = r(Y) = 1 and r(I) = r(Z) = 2·eta.
    At eta = 0.5 every weight is equal, which is the two-qubit depolarizing channel. At large eta
    the pairs IZ, ZI and ZZ dominate, like a bias-preserving CNOT.
    """
    if bias == float("inf"):
        r = {"I": 1.0, "X": 0.0, "Y": 0.0, "Z": 1.0}
    else:
        r = {"I": 2 * bias, "X": 1.0, "Y": 1.0, "Z": 2 * bias}
    pairs = [a + b for a, b in itertools.product("IXYZ", repeat=2) if a + b != "II"]
    weights = [r[a] * r[b] for a, b in pairs]
    total = sum(weights)
    return [p * w / total for w in weights]


@dataclass(frozen=True)
class NoiseModel:
    """Error probabilities for each place a fault can happen.

    data:     Pauli channel on every data qubit at the start of each round
    measure:  flip probability of each measurement result
    reset:    probability that a reset produces the wrong state
    gate1:    Pauli channel after each single-qubit gate
    gate2:    two-qubit Pauli channel after each two-qubit gate
    idle:     Pauli channel on qubits sitting idle during a gate layer
    bias:     eta = pZ/(pX+pY) for all Pauli channels (0.5 = depolarizing)
    """

    data: float = 0.0
    measure: float = 0.0
    reset: float = 0.0
    gate1: float = 0.0
    gate2: float = 0.0
    idle: float = 0.0
    bias: float = DEPOLARIZING

    # ---- presets ---------------------------------------------------------------------------
    @classmethod
    def code_capacity(cls, p: float, bias: float = DEPOLARIZING) -> "NoiseModel":
        return cls(data=p, bias=bias)

    @classmethod
    def phenomenological(cls, p: float, q: float | None = None, bias: float = DEPOLARIZING) -> "NoiseModel":
        return cls(data=p, measure=p if q is None else q, bias=bias)

    @classmethod
    def circuit(cls, p: float, bias: float = DEPOLARIZING, idle: bool = False) -> "NoiseModel":
        """Uniform circuit-level noise of strength p, the same as Stim's generated circuits."""
        return cls(data=p, measure=p, reset=p, gate1=p, gate2=p, idle=p if idle else 0.0, bias=bias)

    @classmethod
    def si1000(cls, p: float) -> "NoiseModel":
        """Superconducting-inspired model (Gidney et al.): measurement is the noisiest step."""
        return cls(data=2 * p, measure=5 * p, reset=2 * p, gate1=p / 10, gate2=p, idle=p / 10)

    def scaled(self, factor: float) -> "NoiseModel":
        return replace(
            self,
            data=self.data * factor,
            measure=self.measure * factor,
            reset=self.reset * factor,
            gate1=self.gate1 * factor,
            gate2=self.gate2 * factor,
            idle=self.idle * factor,
        )

    # ---- helpers to append noise to a Stim circuit -----------------------------------------
    def append_1q(self, circuit: stim.Circuit, targets: list[int], p: float) -> None:
        if p <= 0 or not targets:
            return
        if self.bias == DEPOLARIZING:
            circuit.append("DEPOLARIZE1", targets, p)
        else:
            circuit.append("PAULI_CHANNEL_1", targets, pauli_probs(p, self.bias))

    def append_2q(self, circuit: stim.Circuit, pairs: list[int], p: float) -> None:
        if p <= 0 or not pairs:
            return
        if self.bias == DEPOLARIZING:
            circuit.append("DEPOLARIZE2", pairs, p)
        else:
            circuit.append("PAULI_CHANNEL_2", pairs, two_qubit_pauli_probs(p, self.bias))
