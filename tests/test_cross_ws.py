"""
Pushed books: a pair is priced the moment its asks add up, not at its turn.

On 2026-09-23 the sweep's stalest price was three minutes old, while the
arbitrages that were real opened and closed on a move of the underlying.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from execution.cross_ws import CrossBookWatch, WatchPair
from tests.test_cross_guard import ARB, _book, _guard, _imp


def _snap(token, asks, bids=()):
    return {"event_type": "book", "asset_id": token,
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids]}


def _change(token, side, price, size, best_bid, best_ask):
    return {"event_type": "price_change", "price_changes": [
        {"asset_id": token, "side": side, "price": str(price), "size": str(size),
         "best_bid": str(best_bid), "best_ask": str(best_ask)}]}


def _watch(min_edge=0.02, **kw):
    fired = []
    w = CrossBookWatch(lambda k, e: fired.append((k, round(e, 4))), min_edge=min_edge,
                       slack=0.0, **kw)
    w.watch([WatchPair(("n", "b"), "NO", "YES", 0.0, 0.0)])
    return w, fired


def test_asks_that_add_up_fire_the_callback():
    w, fired = _watch()
    w.handle(json.dumps([_snap("NO", [(0.50, 10)], [(0.48, 10)]),
                         _snap("YES", [(0.50, 10)], [(0.48, 10)])]))
    assert fired == []                                 # 0.50 + 0.50: no edge
    w.handle(json.dumps(_change("YES", "SELL", 0.45, 10, 0.44, 0.45)))
    assert fired == [(("n", "b"), 0.05)]


def test_the_fee_is_counted_before_calling():
    w, fired = _watch()
    w.watch([WatchPair(("n", "b"), "NO", "YES", 0.05, 0.05)])
    # 0.49 + 0.49 = 0.98 is a 0.02 edge before fees, 0.0050 after.
    w.handle(json.dumps([_snap("NO", [(0.49, 10)], [(0.47, 10)]),
                         _snap("YES", [(0.49, 10)], [(0.47, 10)])]))
    assert fired == []


def test_the_same_quotes_are_not_news_twice():
    w, fired = _watch()
    w.handle(json.dumps([_snap("NO", [(0.40, 10)], [(0.38, 10)]),
                         _snap("YES", [(0.45, 10)], [(0.43, 10)])]))
    w.handle(json.dumps(_change("NO", "BUY", 0.39, 5, 0.39, 0.40)))   # a bid moved only
    assert len(fired) == 1
    w.handle(json.dumps(_change("NO", "SELL", 0.39, 5, 0.38, 0.39)))  # a better ask
    assert len(fired) == 2


def test_a_crossed_book_is_stale_and_says_nothing():
    w, fired = _watch()
    w.handle(json.dumps([_snap("NO", [(0.40, 10)], [(0.45, 10)]),
                         _snap("YES", [(0.45, 10)], [(0.43, 10)])]))
    assert fired == []


def test_unwatched_tokens_are_ignored_and_dropped_on_change():
    w, fired = _watch()
    w.handle(json.dumps(_snap("OTHER", [(0.01, 10)])))
    assert "OTHER" not in w._tops
    w.handle(json.dumps(_snap("NO", [(0.40, 10)])))
    w.watch([WatchPair(("x", "y"), "A", "B")])
    assert w.tokens == {"A", "B"} and "NO" not in w._tops


def test_garbage_frames_do_not_raise():
    w, _ = _watch()
    w.handle("INVALID OPERATION")
    w.handle(json.dumps({"event_type": "price_change", "price_changes": [{"asset_id": "NO"}]}))


# ── the guard's side ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_pushed_pair_is_entered_from_fresh_books(tmp_path, monkeypatch):
    g, client, breaker, _ = _guard(tmp_path, monkeypatch, ARB, [_imp()], ws_enabled=True)
    imp = _imp()
    # The sweep deferred the pair ten minutes ago; the socket says it is there now.
    g._next_check[imp.key] = 1e12
    g.nudge(imp.key, 0.10)
    await g._urgent_pass()
    assert {t for t, _, _ in client.buys} == {"nn", "by"}
    assert breaker.status_dict()["cross_open"] == 1
    assert g.stats["ws_priced"] == 1


@pytest.mark.asyncio
async def test_a_pushed_pair_is_still_refused_when_the_books_do_not_pay(tmp_path, monkeypatch):
    books = {"nn": _book(asks=[(0.55, 50)], bids=[(0.53, 50)]),
             "by": _book(asks=[(0.50, 50)], bids=[(0.48, 50)])}
    g, client, breaker, _ = _guard(tmp_path, monkeypatch, books, [_imp()], ws_enabled=True)
    g.nudge(_imp().key, 0.05)                          # the socket was wrong
    await g._urgent_pass()
    assert client.buys == []
    assert breaker.status_dict()["cross_open"] == 0


@pytest.mark.asyncio
async def test_a_held_pair_is_not_priced_again_by_a_push(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, ARB, [_imp()], ws_enabled=True)
    g.nudge(_imp().key, 0.10)
    await g._urgent_pass()
    n = len(client.buys)
    g._urgent_at.clear()
    g.nudge(_imp().key, 0.10)
    await g._urgent_pass()
    assert len(client.buys) == n                       # "held" stops it


@pytest.mark.asyncio
async def test_the_nearest_pairs_are_the_ones_watched(tmp_path, monkeypatch):
    far = _imp(narrow="0xf", broad="0xg", narrow_no_token="fn", broad_yes_token="gy")
    near = _imp()
    g, *_ = _guard(tmp_path, monkeypatch, ARB, [far, near], ws_enabled=True)
    g._last_edge[far.key] = -0.30
    g._last_edge[near.key] = -0.01
    monkeypatch.setattr("execution.cross_guard.CROSS_WS_MAX_PAIRS", 1)
    import time as _t
    pairs = g._watch_pairs(_t.time())
    assert [p.key for p in pairs] == [near.key]
    assert (pairs[0].no_token, pairs[0].yes_token) == ("nn", "by")


@pytest.mark.asyncio
async def test_run_wakes_on_a_push_without_waiting_for_the_poll(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, ARB, [_imp()], poll_s=3600.0,
                             ws_enabled=False)
    polls = []

    async def _poll():
        polls.append(1)
    g.poll_once = _poll
    task = asyncio.create_task(g.run())
    await asyncio.sleep(0.05)
    assert polls == [1]
    g.nudge(_imp().key, 0.10)
    for _ in range(50):
        await asyncio.sleep(0.01)
        if client.buys:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert {t for t, _, _ in client.buys} == {"nn", "by"}
    assert polls == [1]                                # no second poll was needed


def test_new_tokens_wait_for_the_resubscription_interval(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("execution.cross_ws.time.monotonic", lambda: clock[0])
    w = CrossBookWatch(lambda k, e: None, min_edge=0.02, resub_s=300.0)
    w.watch([WatchPair(("a", "b"), "A1", "B1")])
    assert w._changed.is_set()                         # the first set: subscribe now
    w._changed.clear()
    w._subscribed, w._resub_at = {"A1", "B1"}, clock[0]
    w.watch([WatchPair(("c", "d"), "C1", "D1")])
    assert not w._changed.is_set()                     # 0 s later: wait
    clock[0] += 301.0
    w.watch([WatchPair(("c", "d"), "C1", "D1")])
    assert w._changed.is_set()


def test_a_subscribed_token_keeps_its_book_while_unwatched():
    w, fired = _watch()
    w._subscribed = {"NO", "YES"}
    w.handle(json.dumps(_snap("NO", [(0.40, 10)], [(0.38, 10)])))
    w.watch([])                                        # not watched for a pass
    w.handle(json.dumps(_change("NO", "SELL", 0.39, 5, 0.38, 0.39)))
    assert w._tops["NO"].ask == 0.39
    w.watch([WatchPair(("n", "b"), "NO", "YES", 0.0, 0.0)])
    w.handle(json.dumps(_snap("YES", [(0.45, 10)], [(0.43, 10)])))
    assert fired and fired[-1][0] == ("n", "b")        # priced at once on return
