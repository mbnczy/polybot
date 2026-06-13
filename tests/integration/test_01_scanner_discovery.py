"""Step 1 — Market discovery.

MarketScanner → MarketScorer → FeedRegistry.add_market
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core.scanner import FeedRegistry, MarketScanner, MarketScorer
from tests.mocks.fake_gamma import FakeGamma, make_market


@pytest.mark.asyncio
async def test_scanner_admits_top_n_markets(monkeypatch, patched_aiohttp_ws):
    gamma = FakeGamma()
    gamma.add(make_market("0x" + "11" * 32, "y1", "n1", volume_24h=100_000, category="crypto"))
    gamma.add(make_market("0x" + "22" * 32, "y2", "n2", volume_24h= 50_000, category="politics"))
    gamma.add(make_market("0x" + "33" * 32, "y3", "n3", volume_24h=  1_000, category="sports"))

    async def fake_fetch(self):
        return gamma.get_all()

    monkeypatch.setattr(
        "core.scanner.MarketScanner._fetch_all_active",
        fake_fetch,
        raising=True,
    )

    queue: asyncio.Queue = asyncio.Queue()
    registry = FeedRegistry(queue)
    on_added = AsyncMock(side_effect=registry.add_market)

    scanner = MarketScanner(on_market_added=on_added, scan_interval=0.01, max_feeds=2)
    await scanner._scan_once()

    # 2 (max_feeds) of 3 markets admitted — top scorers first.
    assert on_added.await_count == 2
    admitted_cids = {call.args[0] for call in on_added.await_args_list}
    assert "0x" + "33" * 32 not in admitted_cids       # lowest volume excluded
    assert registry.active_count == 2

    await registry.stop_all()


def test_market_scorer_v2_ranking():
    """V2: with equal inefficiency, more volume scores higher up to the penalty
    pivot; and a more inefficient market outranks a more efficient one."""
    scorer = MarketScorer()
    hi = make_market("0x" + "aa" * 32, "y", "n", volume_24h=100_000, days_until_close=10)
    lo = make_market("0x" + "bb" * 32, "y", "n", volume_24h=  1_000, days_until_close=10)
    assert scorer.score(hi) > scorer.score(lo)

    # Inefficiency dominates: a wider YES+NO dislocation outranks a tight one
    # even at the same volume.
    ineff = make_market("0x" + "cc" * 32, "y", "n", volume_24h=10_000,
                        outcome_prices='["0.56", "0.50"]')   # edge 0.06
    eff   = make_market("0x" + "dd" * 32, "y", "n", volume_24h=10_000,
                        outcome_prices='["0.50", "0.50"]')   # edge 0.00
    assert scorer.score(ineff) > scorer.score(eff)
