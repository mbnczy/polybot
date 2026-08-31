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
        # V2 scoring fields so the market scores > 0 and is admitted.
        "liquidityNum": 8_000.0, "bestBid": 0.47, "bestAsk": 0.52,
        "outcomePrices": '["0.52", "0.50"]',
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
    feed._dispatch(   # _dispatch is synchronous (queue push has no await)
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


# ═══════════════════════════════════════════════════════════════════════════
# L3 — market-meta cache (tick size + neg-risk) on the order hot path
#
# The V2 SDK re-fetches /tick-size and /neg-risk from the CLOB on every order
# build and again on every post, with no cache of its own (~140 ms of blocking
# network I/O per signature, measured).  Both values are immutable per market
# and already present in the Gamma dict the scanner fetches, so admission can
# prime them for free.
# ═══════════════════════════════════════════════════════════════════════════

from decimal import Decimal  # noqa: E402

import core.clob_client as cc  # noqa: E402


GAMMA_MARKET = {
    "conditionId": "0xmeta",
    "clobTokenIds": '["tok_YES", "tok_NO"]',   # Gamma ships this as a JSON string
    "orderPriceMinTickSize": 0.001,
    "negRisk": True,
}


@pytest.fixture(autouse=True)
def _clean_meta_cache():
    cc.clear_market_meta_cache()
    yield
    cc.clear_market_meta_cache()


class TestPrimeMarketMeta:
    def test_primes_both_legs_from_gamma_dict(self):
        assert cc.prime_market_meta(GAMMA_MARKET) == 2
        assert cc.peek_market_meta("tok_YES") == (Decimal("0.001"), True)
        assert cc.peek_market_meta("tok_NO")  == (Decimal("0.001"), True)

    def test_accepts_list_token_ids(self):
        assert cc.prime_market_meta(
            {"clobTokenIds": ["a", "b"], "orderPriceMinTickSize": "0.01"}
        ) == 2
        assert cc.peek_market_meta("a") == (Decimal("0.01"), False)

    def test_tick_size_has_no_binary_float_noise(self):
        # Decimal(0.001) from a float would carry ...00000000208 tails.
        cc.prime_market_meta(GAMMA_MARKET)
        tick, _ = cc.peek_market_meta("tok_YES")
        assert str(tick) == "0.001"

    def test_miss_returns_none(self):
        assert cc.peek_market_meta("nope") is None

    @pytest.mark.parametrize("bad", [
        {},                                                    # no fields
        {"clobTokenIds": '["a"]'},                             # no tick size
        {"orderPriceMinTickSize": 0.01},                       # no tokens
        {"clobTokenIds": "not-json", "orderPriceMinTickSize": 0.01},
        {"clobTokenIds": '["a"]', "orderPriceMinTickSize": "abc"},
        {"clobTokenIds": '["a"]', "orderPriceMinTickSize": 0},  # non-positive
    ])
    def test_malformed_input_is_a_noop_not_an_error(self, bad):
        assert cc.prime_market_meta(bad) == 0
        assert cc.market_meta_cache_size() == 0


class TestSdkFetchersUseCache:
    """The patched SDK fetchers must serve primed values without network I/O."""

    def test_patch_is_installed(self):
        assert cc._META_CACHE_INSTALLED is True

    def test_cache_hit_skips_the_network(self):
        from polymarket._internal.actions.orders import limit, place

        cc.prime_market_meta(GAMMA_MARKET)
        # A ctx that raises if anything reaches the network.
        class _Boom:
            def __getattr__(self, _):
                raise AssertionError("network call on a cache hit")

        assert limit.fetch_tick_size_sync(_Boom(), token_id="tok_YES") == Decimal("0.001")
        assert limit.fetch_neg_risk_sync(_Boom(), token_id="tok_YES") is True
        # place.py posts orders and re-fetches neg-risk — must hit cache too.
        assert place.fetch_neg_risk_sync(_Boom(), token_id="tok_NO") is True

    def test_miss_falls_through_and_backfills(self, monkeypatch):
        from polymarket._internal.actions.orders import limit

        calls: list[str] = []

        def _orig(ctx, *, token_id):
            calls.append(token_id)
            return Decimal("0.01")

        # Re-wrap a fresh original so we can observe fall-through.
        monkeypatch.setattr(cc, "_META_CACHE_INSTALLED", False, raising=True)
        monkeypatch.setattr(
            "polymarket._internal.actions.orders.market_data.fetch_tick_size_sync",
            _orig, raising=True,
        )
        cc._install_market_meta_cache()

        assert limit.fetch_tick_size_sync(object(), token_id="cold") == Decimal("0.01")
        assert calls == ["cold"]                      # fetched once …
        assert limit.fetch_tick_size_sync(object(), token_id="cold") == Decimal("0.01")
        assert calls == ["cold"]                      # … then served from cache

    def test_install_is_idempotent(self):
        before = cc.market_meta_cache_size()
        cc._install_market_meta_cache()
        cc._install_market_meta_cache()
        assert cc._META_CACHE_INSTALLED is True
        assert cc.market_meta_cache_size() == before
