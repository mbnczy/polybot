"""
tests/test_efficiency_reliability.py
────────────────────────────────────
Unit tests for the efficiency-and-reliability feature set:

  • clob_client.classify_fills        — leg-fill reconciliation states
  • DutchBookPricer / ArbDetector     — extreme-price + min-real-edge guards
  • FeedRegistry                      — global cap + stale-feed pruning
  • MarketScanner                     — 24h-volume liquidity floor
  • MarketFeed                        — tick dedup + idle_seconds liveness
  • TelegramNotifier.send_arb_detected— per-market cooldown + min-bps throttle
"""
from __future__ import annotations

import asyncio
import os

import pytest

from core.clob_client import classify_fills, _resp_filled
from strategy.arbitrage import ArbDetector, DutchBookPricer


# ═══════════════════════════════════════════════════════════════════════════
# classify_fills — leg reconciliation
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyFills:
    def test_both_matched(self):
        assert classify_fills({"status": "matched"}, {"status": "matched"}) == "both"

    def test_both_paper(self):
        assert classify_fills({"status": "paper"}, {"status": "paper"}) == "both"

    def test_yes_only(self):
        assert classify_fills({"status": "matched"}, {"status": "unmatched"}) == "yes_only"

    def test_no_only(self):
        assert classify_fills({"status": "cancelled"}, {"status": "filled"}) == "no_only"

    def test_none(self):
        assert classify_fills({"status": "unmatched"}, {"status": "live"}) == "none"

    def test_none_on_missing_or_garbage(self):
        assert classify_fills(None, None) == "none"
        assert classify_fills({}, {"foo": "bar"}) == "none"

    def test_resp_filled_case_insensitive(self):
        assert _resp_filled({"status": "MATCHED"}) is True
        assert _resp_filled({"status": "Filled"}) is True
        assert _resp_filled({"status": "live"}) is False


# ═══════════════════════════════════════════════════════════════════════════
# Signal-quality guards
# ═══════════════════════════════════════════════════════════════════════════

class TestExtremePriceGuard:
    def test_maker_rejects_near_resolved(self):
        """yes_ask far below band → near-resolved market → no signal."""
        p = DutchBookPricer(desired_net_margin=0.005)
        # 0.019 + 0.98 combined < 1.0 would otherwise fire on the rebate path
        sig = p.evaluate_maker("c", "y", "n", yes_ask=0.019, no_ask=0.98,
                               maker_rebate=0.01)
        assert sig is None

    def test_maker_accepts_contested(self):
        p = DutchBookPricer(desired_net_margin=0.005)
        sig = p.evaluate_maker("c", "y", "n", yes_ask=0.47, no_ask=0.50,
                               maker_rebate=0.01)
        assert sig is not None
        assert sig.is_maker_signal

    def test_taker_rejects_extreme(self):
        d = ArbDetector(desired_net_margin=0.005, default_fee_rate=0.0)
        sig = d.evaluate("c", "y", "n", yes_ask=0.97, no_ask=0.02, fee_rate=0.0)
        assert sig is None   # 0.97 > 0.95 hi-band

    def test_custom_band(self):
        p = DutchBookPricer(desired_net_margin=0.005, extreme_lo=0.30, extreme_hi=0.70)
        # 0.25 is now outside the tighter band
        assert p.evaluate_maker("c", "y", "n", 0.25, 0.50, maker_rebate=0.01) is None


class TestMinRealEdgeGuard:
    def test_off_by_default_allows_rebate_entry(self):
        """min_real_edge=0 (off): combined slightly >1.0 still fires via rebate."""
        p = DutchBookPricer(desired_net_margin=0.005, min_real_edge=0.0)
        # combined asks 0.504+0.504 = 1.008 (real_edge negative); 1.44% rebate
        # funds it: effective ≈ 0.9915 → maker_net_edge ≈ 85 bps > 0.5% margin.
        sig = p.evaluate_maker("c", "y", "n", yes_ask=0.504, no_ask=0.504,
                               maker_rebate=0.0144)
        assert sig is not None   # rebate-funded entry permitted when guard off

    def test_on_blocks_rebate_only_entry(self):
        p = DutchBookPricer(desired_net_margin=0.005, min_real_edge=0.002)
        # real_edge = 1-(0.504+0.504) = -0.008 < 0.002 → blocked
        sig = p.evaluate_maker("c", "y", "n", yes_ask=0.504, no_ask=0.504,
                               maker_rebate=0.0144)
        assert sig is None

    def test_on_allows_genuine_gap(self):
        p = DutchBookPricer(desired_net_margin=0.005, min_real_edge=0.002)
        # real_edge = 1-(0.47+0.50) = 0.03 ≥ 0.002 → allowed
        sig = p.evaluate_maker("c", "y", "n", yes_ask=0.47, no_ask=0.50,
                               maker_rebate=0.01)
        assert sig is not None


# ═══════════════════════════════════════════════════════════════════════════
# FeedRegistry — cap + pruning  (MarketFeed.run is patched to a no-op)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def _no_network_feed(monkeypatch):
    """Stop MarketFeed.run from opening real WebSockets during registry tests."""
    async def _idle(self):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return
    monkeypatch.setattr("core.ws_feed.MarketFeed.run", _idle, raising=True)


@pytest.mark.asyncio
async def test_feed_registry_global_cap(_no_network_feed):
    from core.scanner import FeedRegistry
    q = asyncio.Queue()
    reg = FeedRegistry(queue=q, max_feeds=2)
    assert await reg.add_market("c1", "y1", "n1") is True
    assert await reg.add_market("c2", "y2", "n2") is True
    # Third exceeds cap → rejected
    assert await reg.add_market("c3", "y3", "n3") is False
    assert reg.active_count == 2
    # Idempotent re-add of existing returns True without growing
    assert await reg.add_market("c1", "y1", "n1") is True
    assert reg.active_count == 2
    await reg.stop_all()


@pytest.mark.asyncio
async def test_feed_registry_prune_stale(_no_network_feed, monkeypatch):
    from core.scanner import FeedRegistry
    q = asyncio.Queue()
    reg = FeedRegistry(queue=q, max_feeds=5)
    await reg.add_market("c1", "y1", "n1")
    await reg.add_market("c2", "y2", "n2")

    # Force c1's feed to look stale, c2 fresh.
    feed1, _ = reg._feeds["c1"]
    feed2, _ = reg._feeds["c2"]
    monkeypatch.setattr(feed1, "idle_seconds", lambda now=None: 9999.0)
    monkeypatch.setattr(feed2, "idle_seconds", lambda now=None: 1.0)

    pruned = await reg.prune_stale(max_idle_s=600)
    assert pruned == ["c1"]
    assert reg.active_count == 1
    assert "c2" in reg.condition_ids
    await reg.stop_all()


@pytest.mark.asyncio
async def test_feed_registry_prune_disabled(_no_network_feed):
    from core.scanner import FeedRegistry
    q = asyncio.Queue()
    reg = FeedRegistry(queue=q)
    await reg.add_market("c1", "y1", "n1")
    assert await reg.prune_stale(max_idle_s=0) == []   # disabled
    await reg.stop_all()


# ═══════════════════════════════════════════════════════════════════════════
# MarketScanner — volume floor
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_scanner_volume_floor(_no_network_feed):
    from core.scanner import FeedRegistry, MarketScanner
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _mkt(cid, vol):
        return {
            "conditionId": cid,
            "clobTokenIds": [f"{cid}_Y", f"{cid}_N"],
            "category": "crypto",
            "volume24hr": vol,
            "endDate": future,
            "active": True, "closed": False,
            # V2 scoring fields so the market is admittable (above floors).
            "liquidityNum": 8_000.0, "bestBid": 0.47, "bestAsk": 0.52,
            "outcomePrices": '["0.52", "0.50"]',
        }

    markets = [_mkt("0xhigh", 100_000), _mkt("0xlow", 100)]
    added: list[str] = []

    q = asyncio.Queue()
    reg = FeedRegistry(queue=q, max_feeds=10)

    async def _cb(cid, y, n):
        added.append(cid)
        return await reg.add_market(cid, y, n)

    scanner = MarketScanner(
        on_market_added=_cb, scan_interval=9999, max_feeds=10,
        feed_registry=reg, min_volume_24h=1_000,
    )

    async def _fake_fetch(self):
        return markets
    import core.scanner as sc
    sc.MarketScanner._fetch_all_active = _fake_fetch  # type: ignore

    await scanner._scan_once_inner()
    assert "0xhigh" in added
    assert "0xlow" not in added   # below the 1_000 floor
    await reg.stop_all()


# ═══════════════════════════════════════════════════════════════════════════
# MarketFeed — dedup + idle_seconds
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_marketfeed_dedup_and_liveness():
    from core.ws_feed import MarketFeed
    q: asyncio.Queue = asyncio.Queue()
    feed = MarketFeed("Y", "N", "0xc", q)

    # Seed both legs and push once.
    feed._best_ask["Y"] = 0.47
    feed._best_ask["N"] = 0.50
    feed._maybe_push_tick()
    assert q.qsize() == 1

    # Identical asks → deduped (no new tick).
    feed._maybe_push_tick()
    assert q.qsize() == 1

    # Changed ask → new tick.
    feed._best_ask["Y"] = 0.46
    feed._maybe_push_tick()
    assert q.qsize() == 2

    # idle_seconds resets on push (~0) and grows with the supplied clock.
    assert feed.idle_seconds(feed._last_tick_monotonic) == pytest.approx(0.0, abs=1e-6)
    assert feed.idle_seconds(feed._last_tick_monotonic + 5.0) == pytest.approx(5.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# TelegramNotifier.send_arb_detected — throttling
# ═══════════════════════════════════════════════════════════════════════════

def _make_notifier(monkeypatch, cooldown="60", min_bps="100"):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test:token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("ARB_ALERT_COOLDOWN_S", cooldown)
    monkeypatch.setenv("ARB_ALERT_MIN_BPS", min_bps)
    from telemetry.telegram import TelegramNotifier
    n = TelegramNotifier()
    fired: list[str] = []
    n._fire = lambda text, parse_mode=None: fired.append(text)  # type: ignore
    return n, fired


class TestArbDetectedThrottle:
    def test_min_bps_floor(self, monkeypatch):
        n, fired = _make_notifier(monkeypatch, min_bps="100")
        n.send_arb_detected("0xA", 0.995, 0.0050, True, 0.50, 0.495)  # 50 bps < 100
        assert fired == []
        n.send_arb_detected("0xA", 0.98, 0.0200, True, 0.50, 0.48)    # 200 bps
        assert len(fired) == 1 and "ARB DETECTED" in fired[0]

    def test_per_market_cooldown(self, monkeypatch):
        n, fired = _make_notifier(monkeypatch, cooldown="60", min_bps="0")
        n.send_arb_detected("0xA", 0.98, 0.02, True, 0.5, 0.48)
        n.send_arb_detected("0xA", 0.97, 0.03, True, 0.5, 0.47)   # cooldown
        assert len(fired) == 1
        n.send_arb_detected("0xB", 0.96, 0.04, False, 0.52, 0.44) # other market
        assert len(fired) == 2

    def test_cooldown_expiry(self, monkeypatch):
        import time
        n, fired = _make_notifier(monkeypatch, cooldown="60", min_bps="0")
        n.send_arb_detected("0xA", 0.98, 0.02, True, 0.5, 0.48)
        n._last_arb_alert["0xA"] = time.monotonic() - 61
        n.send_arb_detected("0xA", 0.98, 0.02, True, 0.5, 0.48)
        assert len(fired) == 2

    def test_disabled_when_no_token(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        from telemetry.telegram import TelegramNotifier
        n = TelegramNotifier()
        fired: list[str] = []
        n._fire = lambda text, parse_mode=None: fired.append(text)  # type: ignore
        n.send_arb_detected("0xA", 0.98, 0.02, True, 0.5, 0.48)
        assert fired == []   # notifier disabled → no-op
