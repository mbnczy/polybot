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


@pytest.mark.asyncio
async def test_the_first_set_is_subscribed_as_soon_as_it_arrives(monkeypatch):
    """2026-09-23 13:51: run() started before the first watch(), subscribed to
    nothing, and that empty subscription held the real one back five minutes."""
    w = CrossBookWatch(lambda k, e: None, min_edge=0.02, resub_s=300.0)
    opened = []

    async def _socket(tokens, sid):
        opened.append(sorted(tokens))
        await asyncio.sleep(3600)
    monkeypatch.setattr(w, "_socket", _socket)
    task = asyncio.create_task(w.run())
    await asyncio.sleep(0.05)
    w.watch([WatchPair(("a", "b"), "A1", "B1")])
    for _ in range(40):
        await asyncio.sleep(0.05)
        if opened:
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert opened == [["A1", "B1"]]


# ── price ladders: watched whatever their distance ───────────────────────────

def _rung(strike, narrow_strike, **kw):
    """NVDA's LOW ladder: hitting the lower strike implies hitting the higher."""
    t = "Will NVIDIA (NVDA) hit (LOW) ${} Week of September 21 2026?"
    return _imp(narrow_title=t.format(narrow_strike), broad_title=t.format(strike), **kw)


def _ladder_setup(tmp_path, monkeypatch, imps, **kw):
    g, *_ = _guard(tmp_path, monkeypatch, ARB, imps, ws_enabled=True, **kw)
    g._implications()                                  # classifies the file's pairs
    import time as _t
    return g, _t.time()


@pytest.mark.asyncio
async def test_a_far_ladder_pair_is_watched_though_the_nearest_are_full(tmp_path, monkeypatch):
    """On 2026-09-23 NVDA $196 ⊆ $204 paid +2.13%. Such a pair sits far below the
    threshold until the underlying moves, so nearness alone never watched it."""
    ladder = _rung(204, 196, narrow="0xl", broad="0xm",
                   narrow_no_token="ln", broad_yes_token="my")
    near = _imp()
    far = _imp(narrow="0xf", broad="0xg", narrow_no_token="fn", broad_yes_token="gy")
    g, now = _ladder_setup(tmp_path, monkeypatch, [ladder, near, far])
    g._last_edge.update({ladder.key: -0.30, near.key: -0.01, far.key: -0.25})
    monkeypatch.setattr("execution.cross_guard.CROSS_WS_MAX_PAIRS", 1)
    assert {p.key for p in g._watch_pairs(now)} == {ladder.key, near.key}
    assert g._ws_ladders == 1


@pytest.mark.asyncio
async def test_a_ladder_pair_is_watched_before_it_was_ever_priced(tmp_path, monkeypatch):
    ladder = _rung(204, 196)
    g, now = _ladder_setup(tmp_path, monkeypatch, [ladder])
    pairs = g._watch_pairs(now)
    assert [(p.key, p.no_token, p.yes_token) for p in pairs] == [(ladder.key, "nn", "by")]


@pytest.mark.asyncio
async def test_a_ladder_pair_the_guard_could_not_enter_is_not_watched(tmp_path, monkeypatch):
    held = _rung(204, 196)
    late = _rung(210, 200, narrow="0xp", broad="0xq", days=30)      # outside the window
    g, now = _ladder_setup(tmp_path, monkeypatch, [held, late])
    g._cooldown[held.key] = now + 600
    assert g._watch_pairs(now) == []


@pytest.mark.asyncio
async def test_over_the_ladder_cap_the_nearest_rungs_are_kept(tmp_path, monkeypatch):
    a = _rung(204, 196)
    b = _rung(210, 200, narrow="0xp", broad="0xq", narrow_no_token="pn", broad_yes_token="qy")
    c = _rung(220, 210, narrow="0xr", broad="0xs", narrow_no_token="rn", broad_yes_token="sy")
    g, now = _ladder_setup(tmp_path, monkeypatch, [a, b, c])
    g._last_edge.update({a.key: -0.20, b.key: -0.02})               # c never priced
    monkeypatch.setattr("execution.cross_guard.CROSS_WS_MAX_LADDER_PAIRS", 2)
    assert [p.key for p in g._watch_pairs(now)] == [b.key, a.key]


@pytest.mark.asyncio
async def test_ladders_can_be_switched_back_to_nearness_only(tmp_path, monkeypatch):
    ladder = _rung(204, 196)
    g, now = _ladder_setup(tmp_path, monkeypatch, [ladder])
    monkeypatch.setattr("execution.cross_guard.CROSS_WS_LADDERS", False)
    assert g._watch_pairs(now) == []                   # unpriced, so not near either


def test_only_two_rungs_of_one_kind_count_as_a_ladder():
    from execution.cross_guard import is_price_ladder
    assert is_price_ladder(_rung(204, 196))
    assert not is_price_ladder(_imp())                                  # over/unders
    mixed = _imp(narrow_title="Will NVIDIA (NVDA) hit (LOW) $196 Week of September 21 2026?",
                 broad_title="Will NVIDIA (NVDA) close above $190 on September 24?")
    assert not is_price_ladder(mixed)
