"""A tiny neural-network decoder in pure numpy: a starting point for ML-decoder projects.

Decoding as classification: input = detection events (one 0/1 per detector), output = probability
that each logical observable flipped. Stim gives unlimited labelled training data for free.

The model is a multilayer perceptron (MLP) with ReLU hidden layers and a sigmoid output, trained with
Adam on binary cross-entropy. It's fine for d = 3 and 5. Bigger codes need models that know the geometry
(CNNs, transformers, e.g. Google's AlphaQubit), and that's a great hackathon direction.

    from qec.nn_decoder import MLPDecoder
    nn = MLPDecoder.train_on_circuit(circuit, hidden=(256, 128), train_shots=2_000_000)
    pred = nn.decode_batch(dets)
"""
from __future__ import annotations

import time

import numpy as np
import stim


class MLPDecoder:
    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = (256, 128), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0, np.sqrt(2 / a), size=(a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.n_in, self.n_out = n_in, n_out

    # ---- forward / backward ------------------------------------------------------------
    def _forward(self, x: np.ndarray):
        acts = [x]
        h = x
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if k < len(self.W) - 1 else z
            acts.append(h)
        return acts  # last entry = logits

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x.astype(np.float32))[-1]

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        out = np.empty((dets.shape[0], self.n_out), dtype=bool)
        for s in range(0, dets.shape[0], 65536):
            out[s:s + 65536] = self.logits(dets[s:s + 65536]) > 0
        return out

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int = 3, batch: int = 1024, lr: float = 1e-3,
            x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, verbose: bool = True) -> list[float]:
        """Adam on binary cross-entropy. Returns the validation error rate after each epoch."""
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        history = []
        n = x.shape[0]
        rng = np.random.default_rng(1)
        for ep in range(epochs):
            t0 = time.perf_counter()
            order = rng.permutation(n)
            step_lr = lr * (0.5 ** ep)  # simple decay
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                xb = x[idx].astype(np.float32)
                yb = y[idx].astype(np.float32)
                acts = self._forward(xb)
                # d(BCE)/d(logits) = sigmoid(z) - y
                g = (1 / (1 + np.exp(-acts[-1])) - yb) / len(idx)
                grads_W, grads_b = [], []
                for k in range(len(self.W) - 1, -1, -1):
                    grads_W.append(acts[k].T @ g)
                    grads_b.append(g.sum(0))
                    if k:
                        g = (g @ self.W[k].T) * (acts[k] > 0)
                grads = list(reversed(grads_W)) + list(reversed(grads_b))
                t += 1
                params = self.W + self.b
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m[i] = b1 * m[i] + (1 - b1) * gr
                    v[i] = b2 * v[i] + (1 - b2) * gr * gr
                    p -= step_lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)
            if x_val is not None:
                err = float(np.any(self.decode_batch(x_val) != y_val, axis=1).mean())
                history.append(err)
                if verbose:
                    print(f"  epoch {ep + 1}/{epochs}: validation logical error {err:.4f} "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
        return history

    @classmethod
    def train_on_circuit(cls, circuit: stim.Circuit, *, hidden=(256, 128), train_shots: int = 1_000_000,
                         val_shots: int = 50_000, epochs: int = 3, batch: int = 1024, seed: int = 0,
                         verbose: bool = True) -> "MLPDecoder":
        sampler = circuit.compile_detector_sampler(seed=seed)
        x, y = sampler.sample(train_shots, separate_observables=True)
        xv, yv = circuit.compile_detector_sampler(seed=seed + 1).sample(val_shots, separate_observables=True)
        model = cls(x.shape[1], y.shape[1], hidden, seed)
        model.fit(x, y, epochs=epochs, batch=batch, x_val=xv, y_val=yv, verbose=verbose)
        return model
