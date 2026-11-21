"""Sliding-window decoding: how a real-time decoder handles an endless stream of syndromes.

A real quantum computer never stops producing syndrome data (about one round per microsecond on
superconducting hardware). You can't wait for the experiment to end before decoding: unprocessed
data piles up (the "backlog problem", review Appendix A). Instead, decode a window of rounds:

    rounds:   0 1 2 3 4 5 6 7 8 9 ...
    window 1 [commit  ][buffer ]
    window 2           [commit  ][buffer ]
    window 3                     [commit  ][buffer ] ...

Within each window the inner decoder (MWPM here) finds a correction. We only **commit** the edges
that start in the commit region. Edges that cross into the buffer flip ("toggle") those buffer
detectors so the next window sees the consequence. Edges that leave the window toward the
future become boundary edges, since the future is unknown. The buffer protects against
committing a wrong guess near the cut. A buffer of about d rounds gives the same accuracy as
decoding everything at once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymatching
import scipy.sparse as sp
import stim


@dataclass
class _Edge:
    in_window: tuple[int, ...]  # 1 or 2 detectors inside the window
    all_dets: tuple[int, ...]  # every detector this error flips (window + future)
    obs: int  # observable bitmask
    p: float


def _graphlike_errors(dem: stim.DetectorErrorModel):
    """(detectors, observable mask, probability) for every graphlike component of every error."""
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets: list[int] = []
        obs = 0
        for t in inst.targets_copy() + [stim.target_separator()]:
            if t.is_separator():
                if dets:
                    yield tuple(sorted(set(d for d in dets if dets.count(d) % 2))), obs, p
                dets, obs = [], 0
            elif t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs ^= 1 << t.val


class SlidingWindowDecoder:
    """Decode a long memory experiment window by window.

    Args:
        dem: detector error model of the full experiment (needs detector time coordinates).
        commit: rounds committed per window.
        buffer: extra look-ahead rounds per window (decoded but not committed).
    """

    def __init__(self, dem: stim.DetectorErrorModel, commit: int, buffer: int):
        if commit < 1 or buffer < 0:
            raise ValueError("need commit >= 1 and buffer >= 0")
        self.commit, self.buffer = commit, buffer
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        coords = dem.get_detector_coordinates()
        self.det_time = np.array([int(round(coords[k][-1])) for k in range(self.num_detectors)])
        t_max = int(self.det_time.max())
        errors = list(_graphlike_errors(dem))

        self.windows = []  # (matching, commit_matrix_obs, commit_matrix_dets, window detector mask)
        t0 = 0
        while t0 <= t_max:
            last = t0 + commit + buffer > t_max  # final window commits everything
            t1 = t_max + 1 if last else t0 + commit + buffer
            t_commit = t1 if last else t0 + commit
            self.windows.append(self._build_window(errors, t0, t1, t_commit))
            if last:
                break
            t0 += commit

    def _build_window(self, errors, t0: int, t1: int, t_commit: int):
        time = self.det_time
        merged: dict[tuple[int, ...], _Edge] = {}
        for dets, obs, p in errors:
            if not dets or p <= 0:
                continue
            if any(time[d] < t0 for d in dets):
                continue  # touches the committed past: already handled by earlier windows
            inside = tuple(d for d in dets if time[d] < t1)
            if not inside or len(inside) > 2:
                continue
            key = inside
            e = merged.get(key)
            if e is None:
                merged[key] = _Edge(inside, dets, obs, p)
            else:  # parallel edges: combine probabilities, keep the likelier error's effect
                p_comb = e.p * (1 - p) + p * (1 - e.p)
                if p > e.p:
                    merged[key] = _Edge(inside, dets, obs, p_comb)
                else:
                    e.p = p_comb
        edges = list(merged.values())
        m = pymatching.Matching()
        m.ensure_num_fault_ids(len(edges))
        for k, e in enumerate(edges):
            w = math.log((1 - e.p) / e.p) if e.p < 0.5 else 0.0
            if len(e.in_window) == 1:
                m.add_boundary_edge(e.in_window[0], fault_ids={k}, weight=w, error_probability=e.p)
            else:
                m.add_edge(e.in_window[0], e.in_window[1], fault_ids={k}, weight=w, error_probability=e.p)

        committed = [k for k, e in enumerate(edges) if min(time[d] for d in e.in_window) < t_commit]
        obs_mat = np.zeros((len(edges), self.num_observables), dtype=np.uint8)
        rows, cols = [], []
        for k in committed:
            for o in range(self.num_observables):
                obs_mat[k, o] = (edges[k].obs >> o) & 1
            for d in edges[k].all_dets:
                rows.append(k)
                cols.append(d)
        det_mat = sp.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                                shape=(len(edges), self.num_detectors))
        commit_mask = np.zeros(len(edges), dtype=np.uint8)
        commit_mask[committed] = 1
        window_dets = (time >= t0) & (time < t1)
        return m, commit_mask, obs_mat, det_mat, window_dets

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        syndrome = dets.astype(np.uint8).copy()
        prediction = np.zeros((dets.shape[0], self.num_observables), dtype=np.uint8)
        for m, commit_mask, obs_mat, det_mat, window_dets in self.windows:
            local = (syndrome * window_dets)[:, : m.num_detectors]  # the window sees only its detectors
            corr = m.decode_batch(local.astype(bool))[:, : len(commit_mask)].astype(np.uint8)
            corr &= commit_mask
            prediction ^= (corr.astype(np.int64) @ obs_mat.astype(np.int64) % 2).astype(np.uint8)
            syndrome ^= (sp.csr_matrix(corr) @ det_mat).toarray().astype(np.uint8) % 2
        return prediction.astype(bool)

    @property
    def num_windows(self) -> int:
        return len(self.windows)
