"""Step 2 — MarketFeed emits paired (yes_ask, no_ask) ticks via the WS."""

from __future__ import annotations

import asyncio

import pytest

from core.ws_feed import MarketFeed
from tests.mocks.fake_clob_ws import (
    FakeAiohttpSession,
    FakeWebSocket,
    make_book_message,
    make_price_change_message,
)


@pytest.mark.asyncio
async def test_feed_pushes_paired_tick_after_both_legs_seen(monkeypatch):
    yes_id, no_id = "YES_TOK", "NO_TOK"
    script = [
        make_book_message(yes_id, 0.47),       # only YES known so far → no tick
        make_book_message(no_id,  0.50),       # now both legs known → tick #1
        make_price_change_message(yes_id, 0.46),  # update YES → tick #2
    ]
    ws = FakeWebSocket(script)

    def _session_factory(*_a, **_kw):
        return FakeAiohttpSession(ws)

    monkeypatch.setattr("core.ws_feed.aiohttp.ClientSession", _session_factory, raising=True)

    queue: asyncio.Queue = asyncio.Queue()
    feed = MarketFeed(yes_id, no_id, "COND", queue, ping_interval=1.0)
    task = asyncio.create_task(feed.run())

    # Wait for both ticks to land.
    tick1 = await asyncio.wait_for(queue.get(), timeout=2.0)
    tick2 = await asyncio.wait_for(queue.get(), timeout=2.0)

    feed.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    for t in (tick1, tick2):
        assert t["type"] == "arb_tick"
        assert t["condition_id"] == "COND"
        assert 0.01 <= t["yes_ask"] <= 0.99
        assert 0.01 <= t["no_ask"]  <= 0.99

    assert tick1["yes_ask"] == 0.47
    assert tick1["no_ask"]  == 0.50
    assert tick2["yes_ask"] == 0.46          # price_change drove YES down
