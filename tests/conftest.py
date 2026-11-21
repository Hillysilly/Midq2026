"""Shared pytest setup: a `metrics_log` fixture that test functions use to record decoder metrics.

At the end of the run, everything recorded is printed as a table and saved to
out/metrics/test_metrics.json, so `pytest` doubles as a decoder benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_RECORDS: list[dict] = []


@pytest.fixture
def metrics_log():
    def record(**row):
        _RECORDS.append(row)
    return record


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _RECORDS:
        return
    tr = terminalreporter
    tr.section("decoder metrics (computed by the tests)")
    by_metric: dict[str, list[dict]] = {}
    for r in _RECORDS:
        by_metric.setdefault(r.get("metric", "?"), []).append(r)
    for metric, rows in by_metric.items():
        tr.write_line(f"\n[{metric}]")
        keys = [k for k in rows[0] if k != "metric"]
        for r in rows:
            for k in r:
                if k not in keys and k != "metric":
                    keys.append(k)
        tr.write_line("  " + "  ".join(f"{k:>14s}" for k in keys))
        for r in rows:
            cells = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    v = f"{v:.3g}"
                cells.append(f"{str(v):>14s}")
            tr.write_line("  " + "  ".join(cells))
    out = ROOT / "out" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(_RECORDS, indent=1, default=str))
    tr.write_line(f"\nsaved {len(_RECORDS)} records to out/metrics/test_metrics.json")
