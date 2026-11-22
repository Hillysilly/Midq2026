"""Ablation of the C++ sparse blossom: every optimization step vs PyMatching.

Each step is a snapshot of blossom.cpp in blossom_steps/ (cumulative: step N = step N-1 plus one
change). For every distance this script samples one set of shots and then, for every step:
  * checks its predictions against PyMatching on every shot (they must agree, up to exact ties),
  * times it, interleaved with PyMatching in several rounds, and reports medians.

Laptop timings are noisy (hybrid P/E cores, power limits), so the process is pinned to one core
at high priority, decoders are timed in alternating rounds, and every number is also given as a
ratio to PyMatching measured in the same rounds.

Correlated mode (--correlated): the steps in blossom_steps/correlated/ are run with correlated
matching switched on and compared with PyMatching's enable_correlations=True. Exact agreement is
not expected there (when pass 1 has several equally short paths, the two decoders may reweight
different edges), so the check is an agreement rate on high-noise shots instead of tie weights.

Usage:  python compiled_ablation.py                      (all steps)
        python compiled_ablation.py 0 3 5                (only those step numbers)
        python compiled_ablation.py --correlated [0 1]   (correlated-matching steps)
"""
import ctypes
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import pymatching

import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(HERE, "blossom_steps")
BUILD_DIR = os.path.join(STEPS_DIR, "build")
SHOTS = {3: 200_000, 5: 100_000, 7: 40_000, 9: 20_000}
ROUNDS = 7
NOISE = 1e-3
CORRELATED = False


def set_mode(correlated):
    """Plain steps live in blossom_steps/, correlated-matching steps in blossom_steps/correlated/."""
    global CORRELATED, STEPS_DIR, BUILD_DIR
    CORRELATED = correlated
    STEPS_DIR = os.path.join(HERE, "blossom_steps", "correlated") if correlated else os.path.join(HERE, "blossom_steps")
    BUILD_DIR = os.path.join(STEPS_DIR, "build")


def make_handle(lib, dem, graph, arrays, rules=None):
    h = lib.sb_create(dem.num_detectors, *arrays)
    if rules is not None and lib.sb_set_correlations(h, *rules) != 0:
        raise RuntimeError(lib.sb_last_error().decode())
    return h


def check_correlated(num, dll, cases=((5, 5e-3, 20000), (7, 5e-3, 5000), (9, 3e-3, 2000)), min_agree=0.995):
    """Correlated steps: high-noise agreement with PyMatching's correlated mode, and error counts."""
    lib = P.load_compiled_blossom(dll)
    ok = True
    for d, p, n in cases:
        circuit = P.make_circuit(d, p)
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, obs = circuit.compile_detector_sampler(seed=78).sample(n, separate_observables=True, bit_packed=True)
        dets = np.ascontiguousarray(dets)
        truth = np.unpackbits(obs, axis=1, bitorder="little")[:, 0].astype(np.uint64)
        graph = P.sim_graph(dem.num_detectors, dem)
        h = make_handle(lib, dem, graph, P.compiled_edge_arrays(graph), P.compiled_correlation_rules(graph, dem))
        out = np.zeros(n, dtype=np.uint64)
        if lib.sb_decode_batch(h, dets, n, dets.shape[1], out) != 0:
            raise RuntimeError(f"step {num}: {lib.sb_last_error().decode()}")
        lib.sb_destroy(h)
        pm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        ref = pm.decode_batch(dets, bit_packed_shots=True, enable_correlations=True)[:, 0].astype(np.uint64)
        agree = float(((out & 1) == ref).mean())
        ok &= agree >= min_agree
        print(f"  step {num:02d} high noise d={d} p={p}: agree {agree:.3%} with PyMatching-correlated, "
              f"errors {int(((out & 1) != truth).sum())} vs {int((ref != truth).sum())}"
              f" -> {'ok' if agree >= min_agree else 'LOW AGREEMENT'}")
    return ok


def pin_to_one_core(cpu=2):
    """Windows: run on a single performance core at high priority (less timing noise)."""
    if os.name != "nt":
        return
    k32 = ctypes.windll.kernel32
    h = k32.GetCurrentProcess()
    k32.SetProcessAffinityMask(h, 1 << cpu)
    k32.SetPriorityClass(h, 0x00000080)  # HIGH_PRIORITY_CLASS


def step_sources(selected=None):
    files = sorted(glob.glob(os.path.join(STEPS_DIR, "step*.cpp")))
    out = []
    for f in files:
        num = int(os.path.basename(f)[4:6])
        if selected is None or num in selected:
            out.append((num, os.path.basename(f)[7:-4], f))
    return out


def build(src):
    """Build a step's DLL. A step may set its own compiler flags with a first line
    '// BUILD-FLAGS: ...' (default -O2), e.g. to test -O3 on an unchanged source."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    dll = os.path.join(BUILD_DIR, os.path.basename(src)[:-4] + ".dll")
    with open(src, encoding="utf-8") as f:
        first = f.readline()
    flags = first.split(":", 1)[1].split() if first.startswith("// BUILD-FLAGS:") else ["-O2"]
    if not os.path.exists(dll) or os.path.getmtime(dll) < os.path.getmtime(src):
        P.build_blossom(src, dll, flags=flags)
    return dll


def path_length(graph, x, target):
    """Float shortest-path length x -> target (boundary is a sink), for weight checks."""
    import heapq, math
    B = graph.boundary
    dist, done, pq = {x: 0.0}, set(), [(0.0, x)]
    while pq:
        d, v = heapq.heappop(pq)
        if v in done:
            continue
        done.add(v)
        if v == target:
            return d
        if v == B:
            continue
        for nb in graph.adj[v]:
            nd = d + nb.weight
            if nd < dist.get(nb.nname, math.inf):
                dist[nb.nname] = nd
                heapq.heappush(pq, (nd, nb.nname))
    return math.inf


def check_high_noise(num, dll, cases=((5, 1.2e-2, 3000), (7, 1.2e-2, 1000), (9, 8e-3, 400))):
    """Blossoms, shattering and zero-radius events only show up at high noise. Every shot where
    the step disagrees with PyMatching must be an exact tie (same total matching weight)."""
    lib = P.load_compiled_blossom(dll)
    lib.sb_last_pairs.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(np.int32), ctypes.c_int]
    ok = True
    for d, p, n in cases:
        circuit = P.make_circuit(d, p)
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, _ = circuit.compile_detector_sampler(seed=77).sample(n, separate_observables=True, bit_packed=True)
        dets = np.ascontiguousarray(dets)
        graph = P.sim_graph(dem.num_detectors, dem)
        h = lib.sb_create(dem.num_detectors, *P.compiled_edge_arrays(graph))
        out = np.zeros(n, dtype=np.uint64)
        if lib.sb_decode_batch(h, dets, n, dets.shape[1], out) != 0:
            raise RuntimeError(f"step {num}: {lib.sb_last_error().decode()}")
        pm = pymatching.Matching.from_detector_error_model(dem)
        ref = pm.decode_batch(dets, bit_packed_shots=True)[:, 0].astype(np.uint64)
        bad = np.flatnonzero((out & 1) != ref)
        worst = 0.0
        buf = np.zeros(4 * dem.num_detectors + 8, dtype=np.int32)
        for s in bad:
            one = np.zeros(1, dtype=np.uint64)
            lib.sb_decode_batch(h, dets[s:s + 1], 1, dets.shape[1], one)
            k = lib.sb_last_pairs(h, buf, len(buf) // 2)
            w = sum(path_length(graph, int(buf[2 * i]), graph.boundary if buf[2 * i + 1] == -1 else int(buf[2 * i + 1]))
                    for i in range(k))
            row = np.unpackbits(dets[s], bitorder="little")[:dem.num_detectors]
            _, w_pm = pm.decode(row, return_weight=True)
            worst = max(worst, abs(w - w_pm) / max(w_pm, 1e-9))
        lib.sb_destroy(h)
        tie_ok = worst < 1e-3   # both decoders round weights to integers; gaps ~1e-5 are rounding
        ok &= tie_ok
        print(f"  step {num:02d} high noise d={d} p={p}: {len(bad)}/{n} disagree, worst weight gap "
              f"{worst:.1e} -> {'all ties (within weight rounding)' if tie_ok else 'NOT TIES: BUG'}")
    return ok


STEP_LABELS = {   # short x-axis labels for the plot
    0: "C++ port", 1: "one event per node", 2: "reuse memory", 3: "flat layout", 4: "hot loop",
    5: "no-alloc dissolve", 6: "no-alloc pair_obs", 7: "lazy reschedule", 8: "small-shot shortcuts",
    9: "no division", 10: "radix heap", 11: "flat pair_obs", 12: "threads API",
}
CORR_STEP_LABELS = {   # correlated-matching steps (blossom_steps/correlated/)
    0: "two-pass port", 1: "fast direct-edge test", 2: "skip pass 2 (certificate)", 3: "1-defect shortcut",
}
D_COLORS = {3: "#6da7ec", 5: "#2a78d6", 7: "#1c5cab", 9: "#104281"}   # light -> dark = small -> large d
PM_COLOR = "#eb6834"


def plot_ablation(results_path=None, out_path=None):
    """Plot an ablation run: time relative to PyMatching per step (left) and the speedup each
    step gave over the previous one (right)."""
    import matplotlib.pyplot as plt
    results_path = results_path or os.path.join(STEPS_DIR, "results_all.json")
    out_path = out_path or os.path.join(HERE, "correlated_ablation.png" if CORRELATED else "compiled_ablation.png")
    labels_for = CORR_STEP_LABELS if CORRELATED else STEP_LABELS
    ref_name = "PyMatching (correlated)" if CORRELATED else "PyMatching"
    with open(results_path) as f:
        rows = json.load(f)
    ds = sorted({r["d"] for r in rows})
    steps = sorted({r["step"] for r in rows})
    by = {(r["d"], r["step"]): r for r in rows}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.35, 1]))

    # left: absolute time per shot. PyMatching gets its own dashed reference line per distance
    # (it does not change with our steps, but it does grow with d).
    labels = []   # [y, y_label, text, color]
    for d in ds:
        color = D_COLORS.get(d, "#2a78d6")
        us = [by[(d, s)]["us"] for s in steps]
        pm = statistics.median(by[(d, s)]["pm_us"] for s in steps)
        ax1.plot([steps[0], steps[-1]], [pm, pm], ls="--", lw=1.8, color=PM_COLOR, zorder=1)
        ax1.plot(steps, us, "o-", ms=6, lw=2, color=color, label=f"ours, d = {d}", zorder=3)
        labels.append([us[-1], us[-1], f"ours d={d}: {us[-1]:.2f} us  ({pm / us[-1]:.1f}x faster)", color])
        labels.append([pm, pm, f"{ref_name} d={d}: {pm:.2f} us", PM_COLOR])
    ax1.plot([], [], ls="--", lw=1.8, color=PM_COLOR, label=f"{ref_name} (one line per d)")
    # right-hand labels: spread apart (in log space) any that would overlap
    labels.sort(key=lambda r: r[0])
    for i in range(1, len(labels)):
        if np.log10(labels[i][1]) - np.log10(labels[i - 1][1]) < 0.075:
            labels[i][1] = labels[i - 1][1] * 10 ** 0.075
    for y, ypos, text, color in labels:
        ax1.text(steps[-1] + 0.35, ypos, text, color=color if color == PM_COLOR else "#0b0b0b",
                 va="center", fontsize=8)
    ax1.set_yscale("log")
    ax1.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.minorticks_off()
    ax1.set_ylabel("decode time per shot (us, 1 core)")
    ax1.set_title(("Correlated" if CORRELATED else "C++") + f" sparse blossom, step by step, vs {ref_name}")
    ax1.set_xlim(steps[0] - 0.4, steps[-1] + 5.2)

    # right: speedup of each step over the previous one, grouped by distance
    width = 0.8 / len(ds)
    for i, d in enumerate(ds):
        gains = [by[(d, s - 1)]["us"] / by[(d, s)]["us"] for s in steps[1:]]
        x = np.array(steps[1:]) + (i - (len(ds) - 1) / 2) * width
        ax2.bar(x, np.array(gains) - 1, bottom=1, width=width * 0.92, color=D_COLORS.get(d, "#2a78d6"),
                label=f"d = {d}", zorder=3)
    ax2.axhline(1.0, color="#52514e", lw=1)
    ax2.axhspan(0.95, 1.05, color="#e6e5e1", zorder=1)
    ax2.text(steps[-1] + 0.6, 1.0, "noise\n(±5%)", fontsize=8, color="#52514e", va="center")
    ax2.set_ylabel("speedup over the previous step")
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax2.set_title("What each step bought")
    ax2.set_xlim(steps[1] - 0.6, steps[-1] + 1.6)

    for ax, xs in ((ax1, steps), (ax2, steps[1:])):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s:02d} {labels_for.get(s, '')}" for s in xs], fontsize=8,
                           rotation=40, ha="right", rotation_mode="anchor")
        ax.grid(True, axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="lower left" if (CORRELATED and ax is ax1) else "upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")
    return fig


def main(selected=None, correlated=False):
    set_mode(correlated)
    pin_to_one_core()
    steps = step_sources(selected)
    dlls = {num: build(src) for num, _, src in steps}
    for num, _, _ in steps:
        passed = check_correlated(num, dlls[num]) if correlated else check_high_noise(num, dlls[num])
        if not passed:
            raise SystemExit(f"step {num} fails the high-noise check")
    results = []
    for d, shots in SHOTS.items():
        circuit = P.make_circuit(d, NOISE)
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, _ = circuit.compile_detector_sampler(seed=1000 + d).sample(
            shots, separate_observables=True, bit_packed=True)
        dets = np.ascontiguousarray(dets)
        graph = P.sim_graph(dem.num_detectors, dem)
        arrays = P.compiled_edge_arrays(graph)
        rules = P.compiled_correlation_rules(graph, dem) if correlated else None
        pm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)
        ref = pm.decode_batch(dets, bit_packed_shots=True, enable_correlations=correlated)[:, 0].astype(np.uint64)

        handles = {}
        for num, name, _ in steps:
            lib = P.load_compiled_blossom(dlls[num])
            h = make_handle(lib, dem, graph, arrays, rules)
            out = np.zeros(shots, dtype=np.uint64)
            if lib.sb_decode_batch(h, dets, shots, dets.shape[1], out) != 0:
                raise RuntimeError(f"step {num}: {lib.sb_last_error().decode()}")
            handles[num] = (lib, h, (out & 1) == ref)

        times = {num: [] for num, _, _ in steps}
        times["pm"] = []
        out = np.zeros(shots, dtype=np.uint64)
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            pm.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True,
                            enable_correlations=correlated)
            times["pm"].append((time.perf_counter() - t0) / shots * 1e6)
            for num, _, _ in steps:
                lib, h, _ = handles[num]
                t0 = time.perf_counter()
                lib.sb_decode_batch(h, dets, shots, dets.shape[1], out)
                times[num].append((time.perf_counter() - t0) / shots * 1e6)
        t_pm = statistics.median(times["pm"])
        print(f"\nd={d}  ({shots} shots, {ROUNDS} rounds)   pymatching {t_pm:.3f} us/shot")
        prev = None
        for num, name, _ in steps:
            lib, h, agree = handles[num]
            t = statistics.median(times[num])
            vs_prev = f"{prev / t:5.2f}x" if prev else "     -"
            print(f"  step {num:02d} {name:<28} {t:8.3f} us/shot   {t / t_pm:5.2f}x pymatching"
                  f"   {vs_prev} vs previous   agree {agree.mean():.4%}")
            results.append(dict(d=d, step=num, name=name, us=t, pm_us=t_pm,
                                agree=float(agree.mean()), rounds=times[num]))
            prev = t
            lib.sb_destroy(h)
    tag = "all" if selected is None else "_".join(map(str, selected))
    with open(os.path.join(STEPS_DIR, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=1)
    if selected is None:
        plot_ablation()


if __name__ == "__main__":
    args = sys.argv[1:]
    corr = "--correlated" in args
    nums = [int(a) for a in args if a != "--correlated"]
    main(nums or None, correlated=corr)
