"""A rotated surface code we build ourselves, so every detail can be changed.

Stim's `stim.Circuit.generated(...)` is great, but it only offers CSS codes with uniform
depolarizing noise. This module adds:

* rectangular codes (dx != dz), useful under biased noise (review §6),
* the XZZX code (a Clifford-deformed surface code, review §6) that thrives under Z-biased noise,
* any `NoiseModel`: code capacity, phenomenological, circuit level, biased, SI1000.

Layout (x to the right, y downward). Data qubit (i, j) sits at coordinate (2i+1, 2j+1) for
0 <= i < width, 0 <= j < height. Checks sit at even coordinates (2a, 2b) in a checkerboard:
X-type when a+b is even, Z-type otherwise.

    * Z_L = Z on the top row of data qubits; its weight is `width`, so dz = width.
    * X_L = X on the left column of data qubits; its weight is `height`, so dx = height.
    * The top and bottom edges keep only weight-2 X checks; the left and right edges keep only
      weight-2 Z checks.

XZZX variant: data qubits with i+j odd are conjugated by Hadamard (X <-> Z), so every bulk
check becomes X Z Z X. The "frame" of a qubit tells you which physical Pauli plays the role of
logical-frame X or Z.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import stim

from .noise import NoiseModel

# Order in which a check touches its (up to four) data qubits. Choosing it well keeps hook errors
# (one ancilla fault -> two data errors) perpendicular to the logical operator of the same type,
# so they don't reduce the code distance (Fowler §VI).
#   X checks: TL, TR, BL, BR  -> last two (BL, BR) form a horizontal pair  ⟂ X_L (vertical)
#   Z checks: TL, BL, TR, BR  -> last two (TR, BR) form a vertical pair    ⟂ Z_L (horizontal)
_OFFSETS = {"TL": (-1, -1), "TR": (1, -1), "BL": (-1, 1), "BR": (1, 1)}
_ORDER = {"X": ("TL", "TR", "BL", "BR"), "Z": ("TL", "BL", "TR", "BR")}


@dataclass
class Check:
    index: int
    coord: tuple[int, int]
    basis: str  # CSS type before any deformation: 'X' or 'Z'
    schedule: list[int | None]  # data qubit touched in each of the 4 CNOT layers (None = idle)
    paulis: dict[int, str] = field(default_factory=dict)  # data qubit -> physical Pauli it applies

    @property
    def data(self) -> list[int]:
        return [q for q in self.schedule if q is not None]


class SurfaceCode:
    """Rotated surface code layout with `dx` x `dz` data qubits.

    Args:
        distance: shorthand for dx = dz = distance.
        dx: X distance (height); errors of X type need dx flips to cause a logical error.
        dz: Z distance (width).
        variant: "css" (standard) or "xzzx".
    """

    def __init__(self, distance: int | None = None, *, dx: int | None = None, dz: int | None = None,
                 variant: str = "css"):
        dx = dx if dx is not None else distance
        dz = dz if dz is not None else distance
        if dx is None or dz is None:
            raise ValueError("give distance, or both dx and dz")
        if dx < 1 or dz < 1 or dx % 2 == 0 or dz % 2 == 0:
            raise ValueError("dx and dz must be odd positive integers")
        if variant not in ("css", "xzzx"):
            raise ValueError("variant must be 'css' or 'xzzx'")
        self.dx, self.dz, self.variant = dx, dz, variant
        self.width, self.height = dz, dx

        # Data qubits.
        self.data_coords: list[tuple[int, int]] = []
        coord_to_data: dict[tuple[int, int], int] = {}
        for j in range(self.height):
            for i in range(self.width):
                coord_to_data[(2 * i + 1, 2 * j + 1)] = len(self.data_coords)
                self.data_coords.append((2 * i + 1, 2 * j + 1))
        n = len(self.data_coords)
        # Frame: True means this data qubit is Hadamard-deformed (X <-> Z).
        self.deformed = [variant == "xzzx" and ((x - 1) // 2 + (y - 1) // 2) % 2 == 1
                         for x, y in self.data_coords]

        # Checks (ancilla qubits are numbered after the data qubits).
        self.checks: list[Check] = []
        for b in range(self.height + 1):
            for a in range(self.width + 1):
                basis = "X" if (a + b) % 2 == 0 else "Z"
                neighbours = {name: coord_to_data.get((2 * a + ox, 2 * b + oy))
                              for name, (ox, oy) in _OFFSETS.items()}
                weight = sum(q is not None for q in neighbours.values())
                if weight < 2:
                    continue
                if weight == 2:
                    on_top_bottom = b in (0, self.height)
                    if on_top_bottom and basis != "X":
                        continue
                    if not on_top_bottom and basis != "Z":
                        continue
                schedule = [neighbours[name] for name in _ORDER[basis]]
                chk = Check(len(self.checks), (2 * a, 2 * b), basis, schedule)
                chk.paulis = {q: self.physical(basis, q) for q in chk.data}
                self.checks.append(chk)
        assert len(self.checks) == n - 1, "a rotated surface code has n-1 independent checks"
        self.num_data = n
        self.ancilla = [n + c.index for c in self.checks]

        # Logical operators in the code frame.
        self.logical_z = [coord_to_data[(2 * i + 1, 1)] for i in range(self.width)]
        self.logical_x = [coord_to_data[(1, 2 * j + 1)] for j in range(self.height)]

    # -------------------------------------------------------------------------------------
    def physical(self, frame_pauli: str, q: int) -> str:
        """Physical Pauli that plays the role of `frame_pauli` on data qubit q."""
        if self.deformed[q] and frame_pauli in "XZ":
            return "Z" if frame_pauli == "X" else "X"
        return frame_pauli

    def logical(self, basis: str) -> list[tuple[int, str]]:
        support = self.logical_z if basis == "Z" else self.logical_x
        return [(q, self.physical(basis, q)) for q in support]

    @property
    def num_qubits(self) -> int:
        return self.num_data + len(self.checks)

    def describe(self) -> str:
        return (f"{self.variant.upper()} rotated surface code dx={self.dx} dz={self.dz}: "
                f"{self.num_data} data + {len(self.checks)} ancilla qubits")

    def ascii(self) -> str:
        """Small text picture: data qubits as 'o' (or 'h' if deformed), checks as X / Z."""
        W, H = 2 * self.width + 1, 2 * self.height + 1
        grid = [[" "] * W for _ in range(H)]
        for q, (x, y) in enumerate(self.data_coords):
            grid[y][x] = "h" if self.deformed[q] else "o"
        for c in self.checks:
            x, y = c.coord
            grid[y][x] = c.basis
        return "\n".join(" ".join(row).rstrip() for row in grid)

    # -------------------------------------------------------------------------------------
    def _coords(self, circuit: stim.Circuit) -> None:
        for q, (x, y) in enumerate(self.data_coords):
            circuit.append("QUBIT_COORDS", [q], [x, y])
        for c in self.checks:
            circuit.append("QUBIT_COORDS", [self.ancilla[c.index]], list(c.coord))

    def _pauli_product(self, items: list[tuple[int, str]]) -> list[stim.GateTarget]:
        targets: list[stim.GateTarget] = []
        for k, (q, p) in enumerate(items):
            if k:
                targets.append(stim.target_combiner())
            targets.append({"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}[p](q))
        return targets

    # ---- ideal-measurement circuits (code capacity / phenomenological) --------------------
    def memory_circuit_ideal(self, noise: NoiseModel, rounds: int = 1) -> stim.Circuit:
        """Memory experiment where stabilizers are measured directly (MPP), not via ancillas.

        Covers code-capacity noise (rounds=1, measure=0) and phenomenological noise. Both
        logical observables are tracked (obs 0 = Z_L, obs 1 = X_L), so a shot fails if either
        flips. That is the metric behind the review's threshold tables (11.6% for UF up to
        18.5% for TN).
        """
        c = stim.Circuit()
        self._coords(c)
        all_data = list(range(self.num_data))
        c.append("R", [q for q in all_data if not self.deformed[q]])
        c.append("RX", [q for q in all_data if self.deformed[q]])

        def measure_checks(p: float) -> None:
            for chk in self.checks:
                c.append("MPP", self._pauli_product(sorted(chk.paulis.items())), p if p > 0 else None)

        # Z_L and X_L anticommute, so they can't both be measured deterministically. Instead,
        # entangle the code with a perfect reference qubit R and track Z_L·Z_R and X_L·X_R,
        # which commute (a logical Bell pair).
        ref = self.num_qubits

        def measure_logicals() -> None:
            c.append("MPP", self._pauli_product(self.logical("Z") + [(ref, "Z")]))
            c.append("MPP", self._pauli_product(self.logical("X") + [(ref, "X")]))

        measure_checks(0)  # perfect reference round fixes every stabilizer's sign
        measure_logicals()
        m = len(self.checks)
        noisy_rounds = rounds if noise.measure > 0 else 1
        for t in range(noisy_rounds):
            c.append("TICK")
            noise.append_1q(c, all_data, noise.data)
            measure_checks(noise.measure)
            for k, chk in enumerate(self.checks):
                prev = -(2 * m + 2) + k if t == 0 else -2 * m + k
                c.append("DETECTOR", [stim.target_rec(-m + k), stim.target_rec(prev)],
                         [*chk.coord, t])
        if noise.measure > 0:  # a final perfect round, so measurement errors are always caught
            c.append("TICK")
            measure_checks(0)
            for k, chk in enumerate(self.checks):
                c.append("DETECTOR", [stim.target_rec(-m + k), stim.target_rec(-2 * m + k)],
                         [*chk.coord, noisy_rounds])
        measure_logicals()
        total = c.num_measurements
        start = m  # index of the first Z_L measurement
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(start - total), stim.target_rec(-2)], 0)
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(start + 1 - total), stim.target_rec(-1)], 1)
        return c

    # ---- full circuit-level syndrome extraction ------------------------------------------
    def memory_circuit(self, noise: NoiseModel, rounds: int | None = None, basis: str = "Z") -> stim.Circuit:
        """Memory experiment with ancilla-based syndrome extraction (Fowler §III, VIII).

        Every ancilla starts in |+>, applies CX (an X action) or CZ (a Z action) to its data
        qubits in the hook-safe order, and is measured in the X basis. That is equivalent to
        the textbook H/CNOT circuits, and the same code handles both CSS and XZZX.

        Args:
            noise: where and how strongly faults happen.
            rounds: rounds of stabilizer measurement (default max(dx, dz)).
            basis: "Z" prepares |0_L> and measures Z_L; "X" prepares |+_L> and measures X_L.
        """
        basis = basis.upper()
        rounds = rounds if rounds is not None else max(self.dx, self.dz)
        c = stim.Circuit()
        self._coords(c)
        data = list(range(self.num_data))
        anc = self.ancilla
        m = len(self.checks)

        # Prepare every data qubit in the frame eigenstate of `basis`.
        z_prep = [q for q in data if self.physical(basis, q) == "Z"]
        x_prep = [q for q in data if self.physical(basis, q) == "X"]
        c.append("R", z_prep)
        c.append("RX", x_prep)
        if noise.reset > 0:
            if z_prep:
                c.append("X_ERROR", z_prep, noise.reset)
            if x_prep:
                c.append("Z_ERROR", x_prep, noise.reset)

        for t in range(rounds):
            c.append("TICK")
            noise.append_1q(c, data, noise.data)
            c.append("RX", anc)
            if noise.reset > 0:
                c.append("Z_ERROR", anc, noise.reset)
            if noise.idle > 0:
                noise.append_1q(c, data, noise.idle)
            for layer in range(4):
                c.append("TICK")
                cx, cz, busy = [], [], set()
                for chk in self.checks:
                    q = chk.schedule[layer]
                    if q is None:
                        continue
                    (cx if chk.paulis[q] == "X" else cz).extend([anc[chk.index], q])
                    busy.update((anc[chk.index], q))
                if cx:
                    c.append("CX", cx)
                if cz:
                    c.append("CZ", cz)
                noise.append_2q(c, cx + cz, noise.gate2)
                idle = [q for q in range(self.num_qubits) if q not in busy]
                noise.append_1q(c, idle, noise.idle)
            c.append("TICK")
            c.append("MX", anc, noise.measure if noise.measure > 0 else None)
            for k, chk in enumerate(self.checks):
                if t == 0:
                    if chk.basis == basis:
                        c.append("DETECTOR", [stim.target_rec(-m + k)], [*chk.coord, t])
                else:
                    c.append("DETECTOR", [stim.target_rec(-m + k), stim.target_rec(-2 * m + k)],
                             [*chk.coord, t])

        # Read out every data qubit in the frame basis.
        c.append("TICK")
        mz = [q for q in data if self.physical(basis, q) == "Z"]
        mx = [q for q in data if self.physical(basis, q) == "X"]
        order = mz + mx
        if mz:
            c.append("M", mz, noise.measure if noise.measure > 0 else None)
        if mx:
            c.append("MX", mx, noise.measure if noise.measure > 0 else None)
        nd = len(order)
        pos = {q: k for k, q in enumerate(order)}
        rec = lambda q: stim.target_rec(-nd + pos[q])  # noqa: E731
        for k, chk in enumerate(self.checks):
            if chk.basis == basis:
                c.append("DETECTOR", [rec(q) for q in chk.data] + [stim.target_rec(-nd - m + k)],
                         [*chk.coord, rounds])
        c.append("OBSERVABLE_INCLUDE", [rec(q) for q, _ in self.logical(basis)], 0)
        return c


def memory(distance: int, noise: NoiseModel, *, rounds: int | None = None, basis: str = "Z",
           variant: str = "css", dx: int | None = None, dz: int | None = None) -> stim.Circuit:
    """Convenience wrapper: circuit-level memory experiment."""
    return SurfaceCode(distance, dx=dx, dz=dz, variant=variant).memory_circuit(noise, rounds, basis)
