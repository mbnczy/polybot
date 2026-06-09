"""
tests/test_latency_optimization.py
──────────────────────────────────
Tests for the latency-optimization branch:

  • L2  FeeEngine.peek_taker_fee / MakerRebateEngine.peek_maker_rebate
        (sync cache peek, no fetch, TTL-aware)
  • L1  MarketScanner on_admit hook fires with (condition_id, market) so caches
        can be pre-warmed before the first tick
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest
from prometheus_client import REGISTRY

from strategy.arbitrage import FeeEngine, MakerRebateEngine, _CACHE_TTL


# ═══════════════════════════════════════════════════════════════════════════
# L2 — synchronous cache peeks
# ═══════════════════════════════════════════════════════════════════════════

class TestPeekFee:
    def test_miss_returns_none(self):
        assert FeeEngine().peek_taker_fee("0xnope") is None

    def test_hit_after_prime(self):
        fe = FeeEngine()
        fe.prime_cache("0xA", 0.012)
        assert fe.peek_taker_fee("0xA") == 0.012

    def test_expired_entry_returns_none(self):
        fe = FeeEngine()
        fe.prime_cache("0xA", 0.012)
        # Age the entry past the TTL.
        fe._cache["0xA"] = (0.012, time.monotonic() - _CACHE_TTL - 1)
        assert fe.peek_taker_fee("0xA") is None

    def test_peek_does_not_fetch(self):
        """peek must never hit the network — even on a miss it returns None fast."""
        fe = FeeEngine()
        t0 = time.monotonic()
        assert fe.peek_taker_fee("0xmiss") is None
        assert time.monotonic() - t0 < 0.05


class TestPeekRebate:
    def test_miss_returns_none(self):
        assert MakerRebateEngine().peek_maker_rebate("0xnope") is None

    def test_hit_after_prime_category(self):
        re = MakerRebateEngine()
        re.prime_cache("0xA", category_slug="crypto")
        assert re.peek_maker_rebate("0xA") == 0.0144

    def test_hit_after_prime_explicit_rate(self):
        re = MakerRebateEngine()
        re.prime_cache("0xA", rebate_rate=0.009)
        assert re.peek_maker_rebate("0xA") == 0.009

    def test_expired_entry_returns_none(self):
        re = MakerRebateEngine()
        re.prime_cache("0xA", rebate_rate=0.009)
        re._cache["0xA"] = (0.009, time.monotonic() - _CACHE_TTL - 1)
        assert re.peek_maker_rebate("0xA") is None


# ═══════════════════════════════════════════════════════════════════════════
# L1 — scanner on_admit pre-warm hook
# ═══════════════════════════════════════════════════════════════════════════

def _future(days: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.asyncio
async def test_on_admit_fires_and_prewarms(monkeypatch):
    from core.scanner import FeedRegistry, MarketScanner
    import core.scanner as sc

    # Don't open real WebSockets.
    async def _idle(self):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return
    monkeypatch.setattr("core.ws_feed.MarketFeed.run", _idle, raising=True)

    market = {
        "conditionId": "0xcrypto",
        "clobTokenIds": ["0xcrypto_Y", "0xcrypto_N"],
        "category": "crypto",
        "volume24hr": 100_000.0,
        "endDate": _future(10),
        "feeRate": 0.015,
    }

    async def _fake_fetch(self):
        return [market]
    monkeypatch.setattr(sc.MarketScanner, "_fetch_all_active", _fake_fetch, raising=True)

    fee_engine    = FeeEngine()
    rebate_engine = MakerRebateEngine()
    admitted: list[tuple[str, dict]] = []

    def _prewarm(condition_id: str, m: dict) -> None:
        admitted.append((condition_id, m))
        rebate_engine.prime_cache(condition_id, category_slug=str(m.get("category") or ""))
        from strategy.arbitrage import _normalise_fee
        fr = _normalise_fee(m.get("feeRate"))
        if fr is not None:
            fee_engine.prime_cache(condition_id, fr)

    q: asyncio.Queue = asyncio.Queue()
    reg = MarketScanner(
        on_market_added=FeedRegistry(queue=q, max_feeds=10).add_market,
        scan_interval=9999, max_feeds=10, on_admit=_prewarm,
    )
    # Use a real registry so add succeeds.
    real_reg = FeedRegistry(queue=q, max_feeds=10)
    reg._on_market_added = real_reg.add_market
    reg._registry = real_reg

    await reg._scan_once_inner()

    # on_admit fired with the market dict …
    assert admitted and admitted[0][0] == "0xcrypto"
    # … and the caches are now warm BEFORE any tick (peek hits synchronously).
    assert rebate_engine.peek_maker_rebate("0xcrypto") == 0.0144
    assert fee_engine.peek_taker_fee("0xcrypto") == 0.015
    await real_reg.stop_all()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0 — per-hop instrumentation
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ws_parse_metric_observes():
    """_dispatch must record a WS_PARSE_SECONDS sample per parsed message."""
    from core.ws_feed import MarketFeed

    before = REGISTRY.get_sample_value("polly_ws_parse_seconds_count") or 0.0
    feed = MarketFeed("Y", "N", "0xc", asyncio.Queue())
    await feed._dispatch(
        '{"event_type":"book","asset_id":"Y","asks":[{"price":"0.47","size":"10"}]}'
    )
    after = REGISTRY.get_sample_value("polly_ws_parse_seconds_count") or 0.0
    assert after - before == pytest.approx(1.0)


def test_orjson_jsondecodeerror_is_caught():
    """orjson.JSONDecodeError subclasses json.JSONDecodeError, so the existing
    _dispatch except clause still catches malformed frames."""
    import json
    from core.ws_feed import _loads
    with pytest.raises(json.JSONDecodeError):
        _loads("{not json")
