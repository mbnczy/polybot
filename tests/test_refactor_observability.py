"""
tests/test_refactor_observability.py
─────────────────────────────────────
Tests for the refactor-and-observability branch:

  • R1  _within_quality_band shared helper
  • R2  TICK_SIZE single source of truth
  • R3  MarketScorer._is_expired + _days_to_close floor
  • F1  observability metrics (FEEDS_PRUNED increments on prune)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from prometheus_client import REGISTRY

from strategy.arbitrage import _within_quality_band, TICK_SIZE
from core.scanner import MarketScorer


# ═══════════════════════════════════════════════════════════════════════════
# R1 — shared quality-band helper
# ═══════════════════════════════════════════════════════════════════════════

class TestQualityBand:
    def test_both_in_band(self):
        assert _within_quality_band(0.47, 0.50, 0.05, 0.95) is True

    def test_yes_below(self):
        assert _within_quality_band(0.02, 0.97, 0.05, 0.95) is False

    def test_no_above(self):
        assert _within_quality_band(0.50, 0.96, 0.05, 0.95) is False

    def test_edges_inclusive(self):
        assert _within_quality_band(0.05, 0.95, 0.05, 0.95) is True


# ═══════════════════════════════════════════════════════════════════════════
# R2 — TICK_SIZE single source of truth
# ═══════════════════════════════════════════════════════════════════════════

def test_tick_size_single_source():
    import core.clob_client as cc
    # clob_client must re-export the exact same object from arbitrage.
    assert cc.TICK_SIZE == TICK_SIZE
    assert cc.TICK_SIZE == 0.001


# ═══════════════════════════════════════════════════════════════════════════
# R3 — expiry handling
# ═══════════════════════════════════════════════════════════════════════════

def _iso(days_from_now: float) -> str:
    dt = datetime.now(timezone.utc) + timedelta(days=days_from_now)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class TestExpiry:
    def test_days_to_close_floors_at_one(self):
        assert MarketScorer._days_to_close({"endDate": _iso(1 / 48)}) == 1.0  # 30 min
        assert MarketScorer._days_to_close({"endDate": _iso(-5)}) == 1.0      # expired
        assert MarketScorer._days_to_close({}) == 1.0                         # missing
        assert MarketScorer._days_to_close({"endDate": "nope"}) == 1.0        # bad

    def test_days_to_close_passthrough_future(self):
        assert 9.99 < MarketScorer._days_to_close({"endDate": _iso(10)}) < 10.01

    def test_is_expired_true_only_for_past(self):
        assert MarketScorer._is_expired({"endDate": _iso(-1)}) is True
        assert MarketScorer._is_expired({"endDate": _iso(1)}) is False

    def test_is_expired_false_for_missing_or_bad(self):
        # Can't tell → not expired (Gamma already filters closed markets).
        assert MarketScorer._is_expired({}) is False
        assert MarketScorer._is_expired({"endDate": "not-a-date"}) is False

    def test_score_not_inflated_near_expiry(self):
        """A market 30 min from close must not get an exploded score: the
        sqrt(days) divisor is floored via _days_to_close==1.0, so a near-expiry
        market scores the same as a 1-day one (not blown up by /0.02)."""
        common = dict(category="crypto", volume24hr=5_000.0, liquidityNum=8_000.0,
                      bestBid=0.47, bestAsk=0.52, outcomePrices='["0.52", "0.50"]')
        near = MarketScorer().score({**common, "endDate": _iso(1 / 48)})   # 30 min
        one_day = MarketScorer().score({**common, "endDate": _iso(1.0)})
        assert near == pytest.approx(one_day, rel=1e-6)   # floored, not exploded
        assert near > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# F1 — FEEDS_PRUNED metric increments on prune
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_feeds_pruned_metric(monkeypatch):
    from core.scanner import FeedRegistry

    async def _idle(self):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return
    monkeypatch.setattr("core.ws_feed.MarketFeed.run", _idle, raising=True)

    before = REGISTRY.get_sample_value("polly_feeds_pruned_total") or 0.0

    q: asyncio.Queue = asyncio.Queue()
    reg = FeedRegistry(queue=q, max_feeds=5)
    await reg.add_market("c1", "y1", "n1")
    await reg.add_market("c2", "y2", "n2")
    feed1, _ = reg._feeds["c1"]
    monkeypatch.setattr(feed1, "idle_seconds", lambda now=None: 9999.0)

    pruned = await reg.prune_stale(max_idle_s=600)
    assert pruned == ["c1"]

    after = REGISTRY.get_sample_value("polly_feeds_pruned_total") or 0.0
    assert after - before == pytest.approx(1.0)
    await reg.stop_all()
