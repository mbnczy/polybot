"""
tests/test_ws_multiplexing.py
─────────────────────────────
WS multiplexing: MarketShard carries MANY markets on ONE connection, routing
each event by asset_id; FeedRegistry packs markets into shards by capacity so
the connection count is ceil(N / capacity) instead of N.
"""

from __future__ import annotations

import asyncio

import pytest

import core.scanner as scanner_mod
from core.ws_feed import MarketShard
from tests.mocks.fake_clob_ws import (
    FakeAiohttpSession,
    FakeWebSocket,
    make_book_message,
    make_price_change_message,
)


@pytest.mark.asyncio
async def test_shard_routes_two_markets_on_one_connection(monkeypatch):
    """One shard, two markets → both legs of each routed by asset_id → 2 ticks."""
    script = [
        make_book_message("YA", 0.47),   # market A YES
        make_book_message("NA", 0.50),   # market A both legs → tick A
        make_book_message("YB", 0.30),   # market B YES
        make_book_message("NB", 0.68),   # market B both legs → tick B
        make_price_change_message("YB", 0.29),  # B YES drops → tick B#2
    ]
    ws = FakeWebSocket(script)
    monkeypatch.setattr("core.ws_feed.aiohttp.ClientSession",
                        lambda *a, **k: FakeAiohttpSession(ws), raising=True)

    queue: asyncio.Queue = asyncio.Queue()
    shard = MarketShard(queue, shard_id=0)
    shard.add("condA", "YA", "NA")
    shard.add("condB", "YB", "NB")
    assert shard.count == 2

    task = asyncio.create_task(shard.run())
    ticks = [await asyncio.wait_for(queue.get(), timeout=2.0) for _ in range(3)]
    shard.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    by_cond = {}
    for t in ticks:
        assert t["type"] == "arb_tick"
        by_cond.setdefault(t["condition_id"], []).append(t)

    # Both markets produced ticks on the SAME connection.
    assert set(by_cond) == {"condA", "condB"}
    assert by_cond["condA"][0]["yes_ask"] == 0.47
    assert by_cond["condA"][0]["no_ask"]  == 0.50
    assert by_cond["condB"][0]["yes_ask"] == 0.30
    assert by_cond["condB"][-1]["yes_ask"] == 0.29   # price_change routed to B


@pytest.mark.asyncio
async def test_shard_ignores_unknown_and_removed_markets(monkeypatch):
    ws = FakeWebSocket([
        make_book_message("YA", 0.47),
        make_book_message("NA", 0.50),
        make_book_message("ZZ", 0.99),   # unknown asset → ignored
    ])
    monkeypatch.setattr("core.ws_feed.aiohttp.ClientSession",
                        lambda *a, **k: FakeAiohttpSession(ws), raising=True)

    queue: asyncio.Queue = asyncio.Queue()
    shard = MarketShard(queue, shard_id=1)
    shard.add("condA", "YA", "NA")

    task = asyncio.create_task(shard.run())
    tick = await asyncio.wait_for(queue.get(), timeout=2.0)
    shard.stop(); task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert tick["condition_id"] == "condA"
    # removing a market drops its routing (idempotent, no crash)
    shard.remove("condA")
    assert shard.count == 0
    shard.remove("condA")   # no-op


# ── FeedRegistry sharding ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_packs_markets_into_shards_by_capacity(monkeypatch):
    # No real connections.
    async def _idle(self):
        await asyncio.Event().wait()
    monkeypatch.setattr("core.ws_feed.MarketShard.run", _idle, raising=True)
    # Capacity 2 → 5 markets should occupy ceil(5/2) = 3 shards.
    monkeypatch.setattr(scanner_mod, "_SHARD_CAPACITY", 2, raising=True)

    q: asyncio.Queue = asyncio.Queue()
    reg = scanner_mod.FeedRegistry(queue=q, max_feeds=0)
    for i in range(5):
        assert await reg.add_market(f"c{i}", f"y{i}", f"n{i}")

    assert reg.active_count == 5
    assert reg.shard_count == 3
    assert reg.condition_ids == frozenset(f"c{i}" for i in range(5))

    # Idempotent add.
    assert await reg.add_market("c0", "y0", "n0") is True
    assert reg.active_count == 5

    await reg.stop_all()
    assert reg.active_count == 0 and reg.shard_count == 0


@pytest.mark.asyncio
async def test_registry_stops_shard_when_it_empties(monkeypatch):
    async def _idle(self):
        await asyncio.Event().wait()
    monkeypatch.setattr("core.ws_feed.MarketShard.run", _idle, raising=True)
    monkeypatch.setattr(scanner_mod, "_SHARD_CAPACITY", 2, raising=True)

    q: asyncio.Queue = asyncio.Queue()
    reg = scanner_mod.FeedRegistry(queue=q, max_feeds=0)
    for i in range(3):                       # c0,c1 → shard0; c2 → shard1
        await reg.add_market(f"c{i}", f"y{i}", f"n{i}")
    assert reg.shard_count == 2

    await reg.remove_market("c2")            # empties shard1 → stopped
    assert reg.shard_count == 1
    assert reg.active_count == 2

    await reg.remove_market("c0")            # shard0 still has c1
    assert reg.shard_count == 1
    await reg.remove_market("c1")            # now shard0 empty → stopped
    assert reg.shard_count == 0
    await reg.stop_all()


@pytest.mark.asyncio
async def test_registry_respects_max_feeds_across_shards(monkeypatch):
    async def _idle(self):
        await asyncio.Event().wait()
    monkeypatch.setattr("core.ws_feed.MarketShard.run", _idle, raising=True)
    monkeypatch.setattr(scanner_mod, "_SHARD_CAPACITY", 2, raising=True)

    q: asyncio.Queue = asyncio.Queue()
    reg = scanner_mod.FeedRegistry(queue=q, max_feeds=3)
    results = [await reg.add_market(f"c{i}", f"y{i}", f"n{i}") for i in range(5)]

    assert results == [True, True, True, False, False]   # capped at 3 markets
    assert reg.active_count == 3
    await reg.stop_all()
