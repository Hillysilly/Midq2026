"""Collect experiment results from out/ into the JSON embedded in docs/explainer.html.

    .venv\\Scripts\\python tools\\export_explainer_data.py

Run it after run_experiments.ps1. Missing experiments are skipped, and their charts show
"Data pending".
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sinter  # noqa: E402

from qec import analysis  # noqa: E402

OUT = ROOT / "out"
HTML = ROOT / "docs" / "explainer.html"


def load(name: str):
    path = OUT / name / "stats.csv"
    if not path.exists():
        return None
    return analysis.aggregate(sinter.read_stats_from_csv_files(str(path)))


def curves(points, series_contains: str, per_round: bool):
    out: dict[str, list] = {}
    for series, by_d in analysis.by_series(points).items():
        if series_contains not in series:
            continue
        for d, pts in sorted(by_d.items()):
            rows = [[pt.p, pt.per_round if per_round else pt.rate,
                     pt.per_round_lo if per_round else pt.lo,
                     pt.per_round_hi if per_round else pt.hi] for pt in pts if pt.errors > 0]
            out[str(d)] = [[round(x, 8) for x in r] for r in rows]
    return out


def threshold_of(points, series_contains: str, per_round: bool):
    for series, pairs in analysis.thresholds(points, per_round).items():
        if series_contains in series:
            crosses = [c for _, _, c in pairs if c]
            return round(sum(crosses) / len(crosses), 5) if crosses else None
    return None


def main() -> None:
    data: dict = {"thresholds": {}, "curves": {}, "decoders": {}}
    for name, per_round in (("code_capacity", False), ("phenomenological", True), ("circuit_decoders", True)):
        pts = load(name)
        if not pts:
            continue
        key = "circuit" if name == "circuit_decoders" else name
        data["thresholds"][key] = threshold_of(pts, "css pymatching ", per_round)
        if per_round:  # the per-shot crossing (d rounds per shot) is the convention most papers quote
            data.setdefault("thresholds_per_shot", {})[key] = threshold_of(pts, "css pymatching ", False)
        data["curves"][key] = curves(pts, "css pymatching ", per_round)  # trailing space: not "_correlated"
        if name == "code_capacity":
            for dec in ("pymatching", "pymatching_correlated", "union_find", "union_find_weighted"):
                data["decoders"][dec] = threshold_of(pts, f"css {dec} ", per_round)
        if name == "circuit_decoders":
            dc = {}
            for series, by_d in analysis.by_series(pts).items():
                if 5 in by_d:
                    dec = series.split()[1]
                    dc[dec] = [[pt.p, pt.per_round, pt.per_round_lo, pt.per_round_hi] for pt in by_d[5] if pt.errors > 0]
            order = ["pymatching", "pymatching_correlated", "union_find", "union_find_weighted", "belief_matching"]
            data["decoder_curves"] = {k: dc[k] for k in order if k in dc}
            data["circuit_thresholds"] = {}
            for series, pairs in analysis.thresholds(pts, True).items():
                crosses = [c for _, _, c in pairs if c]
                data["circuit_thresholds"][series.split()[1]] = (
                    round(sum(crosses) / len(crosses), 5) if crosses else None)

    sweep = OUT / "bias_sweep" / "thresholds.json"
    if sweep.exists():
        data["bias_sweep"] = json.loads(sweep.read_text(encoding="utf-8"))

    pts = load("bias100")
    if pts:
        data["bias100"] = {v: curves(pts, f"{v} pymatching ", True) for v in ("css", "xzzx")}
        target_p = 0.005
        rates = {}
        for pt in pts:
            if pt.d == 7 and abs(pt.p - target_p) < 1e-9:
                rates[pt.series.split()[0]] = pt.per_round
        if "css" in rates and "xzzx" in rates and rates["xzzx"] > 0:
            data["bias100_note"] = (f"Both memory bases summed, MWPM, η = 100. At p = 0.5% and d = 7, XZZX fails "
                                    f"{rates['xzzx']:.1e} per round vs {rates['css']:.1e} for CSS "
                                    f"({rates['css'] / rates['xzzx']:.0f}× lower).")

    speed = OUT / "speed" / "results.json"
    if speed.exists():
        data["speed"] = json.loads(speed.read_text(encoding="utf-8"))

    si = load("si1000")
    if si:
        data["si1000_lambda"] = {s: v for s, v in analysis.lambda_factor(si, 0.001).items()}

    rt = OUT / "realtime" / "results.json"
    if rt.exists():
        data["realtime"] = json.loads(rt.read_text(encoding="utf-8"))["distances"]

    html = HTML.read_text(encoding="utf-8")
    blob = json.dumps(data, separators=(",", ":"))
    html, n = re.subn(r"/\*DATA_START\*/.*?/\*DATA_END\*/", lambda _: f"/*DATA_START*/{blob}/*DATA_END*/", html, flags=re.S)
    if n != 1:
        raise SystemExit("data marker not found in explainer.html")
    HTML.write_text(html, encoding="utf-8")
    (OUT / "explainer_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Embedded {len(blob):,} bytes of data; sections: {sorted(data)}")
    print("thresholds:", data["thresholds"], "decoders:", data["decoders"])


if __name__ == "__main__":
    main()
