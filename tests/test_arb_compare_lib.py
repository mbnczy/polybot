"""
tests/test_arb_compare_lib.py
─────────────────────────────
Unit tests for the A/B comparison stats lib (append / aggregate / format / prune).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# arb_compare_lib lives in scripts/ (added to path the same way compare_monitor does)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import arb_compare_lib as lib   # noqa: E402


def test_append_and_read_roundtrip(tmp_path):
    f = str(tmp_path / "stats.jsonl")
    lib.append_window(f, "main", ts=1000.0, duration_s=2.5, edge_bps=80.0)
    lib.append_window(f, "refactored", ts=1001.0, duration_s=4.0, edge_bps=150.0)
    recs = lib.read_windows(f)
    assert len(recs) == 2
    assert recs[0]["label"] == "main" and recs[0]["duration_s"] == 2.5


def test_read_missing_file_is_empty():
    assert lib.read_windows("/nonexistent/path/stats.jsonl") == []


def test_read_since_filters_old(tmp_path):
    f = str(tmp_path / "s.jsonl")
    lib.append_window(f, "main", ts=100.0, duration_s=1.0, edge_bps=10.0)
    lib.append_window(f, "main", ts=500.0, duration_s=1.0, edge_bps=10.0)
    assert len(lib.read_windows(f, since_ts=200.0)) == 1


def test_aggregate_per_label():
    windows = [
        {"label": "main", "ts": 1, "duration_s": 2.0, "edge_bps": 100.0},
        {"label": "main", "ts": 2, "duration_s": 4.0, "edge_bps": 200.0},
        {"label": "refactored", "ts": 3, "duration_s": 6.0, "edge_bps": 300.0},
    ]
    agg = lib.aggregate(windows)
    assert agg["main"]["count"] == 2
    assert agg["main"]["avg_duration_s"] == pytest.approx(3.0)
    assert agg["main"]["max_duration_s"] == pytest.approx(4.0)
    assert agg["main"]["avg_edge_bps"] == pytest.approx(150.0)
    assert agg["refactored"]["count"] == 1
    assert agg["refactored"]["avg_duration_s"] == pytest.approx(6.0)


def test_aggregate_empty():
    assert lib.aggregate([]) == {}


def test_format_summary_contains_both_labels():
    agg = lib.aggregate([
        {"label": "main", "ts": 1, "duration_s": 2.0, "edge_bps": 100.0},
        {"label": "refactored", "ts": 2, "duration_s": 4.0, "edge_bps": 200.0},
    ])
    out = lib.format_summary(agg, day_label="2026-06-14", window_hours=24)
    assert "DAILY ARB SUMMARY" in out
    assert "[main]" in out and "[refactored]" in out
    assert "avg_dur" in out and "freq" in out


def test_format_summary_empty():
    out = lib.format_summary({}, day_label="2026-06-14")
    assert "no arbitrage windows" in out


def test_seconds_until_next_utc_midnight():
    now = datetime(2026, 6, 14, 23, 0, 0, tzinfo=timezone.utc)   # 1h before midnight
    assert lib.seconds_until_next_utc_midnight(now) == pytest.approx(3600.0, abs=1.0)


def test_prune_drops_old(tmp_path):
    f = str(tmp_path / "s.jsonl")
    now = datetime.now(timezone.utc).timestamp()
    lib.append_window(f, "main", ts=now - 10 * 86400, duration_s=1.0, edge_bps=10.0)  # 10d old
    lib.append_window(f, "main", ts=now, duration_s=1.0, edge_bps=10.0)               # fresh
    lib.prune(f, keep_days=7.0)
    assert len(lib.read_windows(f)) == 1
