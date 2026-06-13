"""
tests/test_arb_duration.py
──────────────────────────
Unit tests for ArbDurationTracker (how long a market stays in an arb window).
"""
from __future__ import annotations

import pytest

from strategy.arb_duration import ArbDurationTracker, ArbWindow


def test_open_then_close_returns_duration():
    t = ArbDurationTracker()
    assert t.update("c", in_arb=True,  ts=100.0, edge_bps=120.0) is None   # opens
    assert t.update("c", in_arb=True,  ts=101.0, edge_bps=150.0) is None   # still open
    w = t.update("c", in_arb=False, ts=103.5)                              # closes
    assert isinstance(w, ArbWindow)
    assert w.condition_id == "c"
    assert w.duration_s == pytest.approx(3.5)
    assert w.peak_edge_bps == pytest.approx(150.0)   # peak across the window
    assert w.ticks == 2
    assert w.still_open is False


def test_no_arb_tick_without_open_window_is_noop():
    t = ArbDurationTracker()
    assert t.update("c", in_arb=False, ts=10.0) is None
    assert t.open_count == 0


def test_peak_edge_and_maker_flag_track_the_max():
    t = ArbDurationTracker()
    t.update("c", in_arb=True, ts=0.0, edge_bps=50.0,  is_maker=False)
    t.update("c", in_arb=True, ts=1.0, edge_bps=200.0, is_maker=True)   # new peak (maker)
    t.update("c", in_arb=True, ts=2.0, edge_bps=80.0,  is_maker=False)
    w = t.update("c", in_arb=False, ts=3.0)
    assert w.peak_edge_bps == pytest.approx(200.0)
    assert w.is_maker_peak is True
    assert w.ticks == 3


def test_multiple_markets_independent():
    t = ArbDurationTracker()
    t.update("a", in_arb=True, ts=0.0, edge_bps=10.0)
    t.update("b", in_arb=True, ts=1.0, edge_bps=20.0)
    assert t.open_count == 2
    wa = t.update("a", in_arb=False, ts=5.0)
    assert wa.condition_id == "a" and wa.duration_s == pytest.approx(5.0)
    assert t.open_count == 1                      # b still open
    wb = t.update("b", in_arb=False, ts=4.0)
    assert wb.condition_id == "b" and wb.duration_s == pytest.approx(3.0)


def test_reopen_after_close_starts_fresh():
    t = ArbDurationTracker()
    t.update("c", in_arb=True,  ts=0.0, edge_bps=100.0)
    t.update("c", in_arb=False, ts=2.0)
    t.update("c", in_arb=True,  ts=10.0, edge_bps=30.0)   # new window
    w = t.update("c", in_arb=False, ts=11.0)
    assert w.duration_s == pytest.approx(1.0)
    assert w.peak_edge_bps == pytest.approx(30.0)


def test_min_duration_filters_flicker():
    t = ArbDurationTracker(min_duration_s=1.0)
    t.update("c", in_arb=True, ts=0.0, edge_bps=100.0)
    # closes after 0.2s — below the 1.0s floor → not reported
    assert t.update("c", in_arb=False, ts=0.2) is None
    # a longer window IS reported
    t.update("d", in_arb=True, ts=0.0, edge_bps=100.0)
    w = t.update("d", in_arb=False, ts=2.0)
    assert w is not None and w.duration_s == pytest.approx(2.0)


def test_flush_reports_open_windows_as_still_open():
    t = ArbDurationTracker()
    t.update("a", in_arb=True, ts=0.0, edge_bps=40.0)
    t.update("b", in_arb=True, ts=1.0, edge_bps=60.0)
    windows = t.flush(ts=5.0)
    assert {w.condition_id for w in windows} == {"a", "b"}
    assert all(w.still_open for w in windows)
    a = next(w for w in windows if w.condition_id == "a")
    assert a.duration_s == pytest.approx(5.0)
    assert t.open_count == 0   # cleared after flush
