"""
Resting our own bids instead of crossing the spread.

The taker path pays the ask plus a fee on both legs, and in six days of live
evaluation its best edge was −1%. A maker fill pays no fee and buys a tick above
the bid — the same pair can be tradeable there. What it costs is queue risk: an
order can rest unfilled, or one leg can fill alone.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import execution.cross_guard as cg
from execution.cross_guard import evaluate_maker, maker_entry, maker_price
from tests.test_cross_guard import NOW, _book, _Client, _guard, _imp

# Our bids: NO 0.40 + tick, YES 0.50 + tick = 0.92 a pair, and both books are
# tight enough to rest in. Crossing would cost 0.45 + 0.55 = 1.00, so only the
# maker path can trade this.
MAKER = {"nn": _book(asks=[(0.45, 50)], bids=[(0.40, 50)]),
         "by": _book(asks=[(0.55, 50)], bids=[(0.50, 50)])}


class _MakerClient(_Client):
    def __init__(self, books):
        super().__init__(books)
        self.maker_orders: list[tuple[str, str, float, float]] = []
        self.cancelled: list[str] = []
        self.fills: dict[str, float] = {}
        self.fill_on_cancel: dict[str, float] = {}
        self.post_fail: set[str] = set()
        self.fill_queries: list[tuple] = []
        self.balances: dict[str, float] = {}       # what the wallet holds, by token
        self._n = 0

    async def post_maker_order(self, token_id, side, desired_price, size):
        if token_id in self.post_fail:
            return {"status": "error"}
        self._n += 1
        oid = f"m{self._n}"
        self.maker_orders.append((oid, token_id, desired_price, size))
        return {"status": "live", "order_id": oid}

    async def order_filled_size(self, order_id, token_id=None, market=None):
        self.fill_queries.append((order_id, token_id, market))
        return self.fills.get(order_id, 0.0)

    async def share_balance(self, token_id):
        return self.balances.get(token_id, 0.0)

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        if order_id in self.fill_on_cancel:
            self.fills[order_id] = self.fill_on_cancel.pop(order_id)
        return {"canceled": [order_id]}


def _maker_guard(tmp_path, monkeypatch, books=MAKER, imps=None, **kw):
    books = dict(books)                       # tests move the quotes; MAKER stays put
    client = _MakerClient(books)
    g, _, breaker, notifier = _guard(tmp_path, monkeypatch, books,
                                     imps if imps is not None else [_imp()], **kw)
    g._client = client
    return g, client, breaker, notifier


# ── pricing ───────────────────────────────────────────────────────────────────

def test_a_bid_is_posted_one_tick_above_the_best_one():
    assert maker_price(MAKER["nn"], 0.01) == 0.41


def test_a_one_tick_spread_joins_the_queue_instead_of_crossing():
    tight = _book(asks=[(0.41, 10)], bids=[(0.40, 10)])
    assert maker_price(tight, 0.01) == 0.40


def test_without_a_bid_there_is_nothing_to_rest_above():
    assert maker_price(_book(asks=[(0.41, 10)]), 0.01) is None


def test_the_markets_own_tick_is_used_when_the_reader_exports_it():
    fine = _book(asks=[(0.050, 10)], bids=[(0.040, 10)])
    assert maker_price(fine, 0.001) == 0.041


def test_the_maker_entry_pays_no_fee():
    quote = maker_entry(_imp(), MAKER["nn"], MAKER["by"])
    assert quote is not None
    assert quote[2] == pytest.approx(0.92)


def test_a_maker_opportunity_is_priced_at_our_own_bids():
    opp, why = evaluate_maker(_imp(), MAKER["nn"], MAKER["by"], now=NOW)
    assert why == "ok"
    assert opp.entry_per_pair == pytest.approx(0.92)
    assert opp.edge_per_pair == pytest.approx(0.08)
    assert opp.shares == pytest.approx(5.43)          # 5.00 USDC / 0.92


def test_a_maker_edge_below_the_minimum_is_refused():
    opp, why = evaluate_maker(_imp(), MAKER["nn"], MAKER["by"], now=NOW, min_edge=0.20)
    assert opp is None and "at our own bids" in why


# ── resting ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_it_says_what_it_would_rest_and_posts_nothing(tmp_path, monkeypatch):
    g, client, breaker, _ = _maker_guard(tmp_path, monkeypatch, enabled=False)
    await g.poll_once()
    assert client.maker_orders == [] and client.buys == []
    assert g.stats["would_rest"] == 1
    assert breaker.status_dict()["cross_open"] == 0


@pytest.mark.asyncio
async def test_enabled_it_rests_both_bids_and_remembers_them(tmp_path, monkeypatch):
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch)
    await g.poll_once()
    assert [(t, p) for _, t, p, _ in client.maker_orders] == [("nn", 0.41), ("by", 0.51)]
    assert breaker.status_dict()["cross_open"] == 1
    assert "RESTING" in notifier.messages[0]
    saved = json.loads((tmp_path / "positions_maker.json").read_text())
    assert len(saved) == 1 and saved[0]["no_order"] == "m1"
    # A pair with orders on the book is not evaluated again.
    await g.poll_once()
    assert len(client.maker_orders) == 2
    assert "resting 1" in g.stop_summary()


@pytest.mark.asyncio
async def test_both_legs_filling_opens_the_position(tmp_path, monkeypatch):
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch)
    await g.poll_once()
    client.fills = {"m1": 5.43, "m2": 5.43}
    await g.poll_once()
    (pos,) = g.open_positions()
    assert pos.size == pytest.approx(5.43)
    assert pos.entry_cost == pytest.approx(0.92)
    assert breaker.status_dict()["cross_open"] == 1
    assert json.loads((tmp_path / "positions_maker.json").read_text()) == []
    assert any("OPENED (maker)" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_one_leg_filled_is_completed_as_taker_when_that_still_pays(tmp_path, monkeypatch):
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fills = {"m1": 5.43}                        # only the NO leg filled
    client.books["by"] = _book(asks=[(0.50, 50)], bids=[(0.50, 50)])   # it came to us
    await g.poll_once()
    assert client.buys == [("by", 0.50, 5.43)]         # the missing leg, crossed
    (pos,) = g.open_positions()
    assert pos.entry_cost == pytest.approx(0.91)       # 0.41 rested + 0.50 crossed
    assert any("completion" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_one_leg_filled_is_sold_back_when_completing_would_not_pay(tmp_path, monkeypatch):
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fills = {"m1": 5.43}
    client.books["by"] = _book(asks=[(0.62, 50)], bids=[(0.50, 50)])   # it ran away
    await g.poll_once()
    assert client.buys == [] and client.sells == [("nn", 5.43)]
    assert g.open_positions() == []
    assert breaker.status_dict()["cross_open"] == 0
    # sold at the 0.40 bid against 0.41 paid
    # Sold at the 0.40 bid after paying 0.41, less the taker fee on the sale
    # (5.43 × 0.05 × 0.40 × 0.60): the wallet's number, not the book's.
    assert breaker.status_dict()["session_pnl"] == pytest.approx(
        5.43 * (0.40 - 0.41) - 5.43 * 0.05 * 0.40 * 0.60, abs=1e-4)
    assert any("HALF-FILL" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_nothing_filled_cancels_both_and_frees_the_ledger(tmp_path, monkeypatch):
    g, client, breaker, _ = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    await g.poll_once()
    assert client.cancelled == ["m1", "m2"]
    assert breaker.status_dict()["cross_open"] == 0
    assert g.open_positions() == []
    assert "cooldown 1" in g.stop_summary()


@pytest.mark.asyncio
async def test_a_fill_that_raced_the_cancel_is_not_lost(tmp_path, monkeypatch):
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fill_on_cancel = {"m1": 5.43, "m2": 5.43}   # both matched as the cancel landed
    await g.poll_once()
    (pos,) = g.open_positions()
    assert pos.size == pytest.approx(5.43)


@pytest.mark.asyncio
async def test_orders_left_resting_by_a_dead_process_are_cancelled(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, imps=[])
    (tmp_path / "positions_maker.json").write_text(json.dumps([{
        "narrow": "0xn", "broad": "0xb", "narrow_title": "n", "broad_title": "b",
        "no_token": "nn", "yes_token": "by", "no_price": 0.41, "yes_price": 0.51,
        "shares": 5.43, "no_order": "old-1", "yes_order": "old-2", "placed_at": NOW - 9_999,
        "resolves_ts": NOW + 2 * 86_400, "committed": 5.0, "no_filled": 0.0, "yes_filled": 0.0}]))
    g.restore()
    await g.poll_once()
    assert client.cancelled == ["old-1", "old-2"]


@pytest.mark.asyncio
async def test_with_the_maker_path_off_nothing_rests(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, maker_enabled=False)
    await g.poll_once()
    assert client.maker_orders == []
    assert "edge_below_min 1" in g.stop_summary()


def test_a_book_too_wide_to_rest_in_is_refused():
    """A bid a tick above an almost empty book is not an offer anyone will take:
    0.01/0.50 produced "+0.96 edge" pairs on 2026-09-17."""
    wide = {"nn": _book(asks=[(0.50, 50)], bids=[(0.01, 50)]),
            "by": _book(asks=[(0.50, 50)], bids=[(0.01, 50)])}
    opp, why = evaluate_maker(_imp(), wide["nn"], wide["by"], now=NOW)
    assert opp is None and "too wide a book" in why


def test_a_tight_book_still_rests():
    opp, why = evaluate_maker(_imp(), MAKER["nn"], MAKER["by"], now=NOW, max_spread=0.10)
    assert why == "ok" and opp is not None


def test_a_book_exactly_at_the_cap_still_rests():
    """0.55 - 0.50 is 0.05000000000000004 in float; the cap must not reject it."""
    at_cap = {"nn": _book(asks=[(0.45, 50)], bids=[(0.40, 50)]),
              "by": _book(asks=[(0.55, 50)], bids=[(0.50, 50)])}
    opp, why = evaluate_maker(_imp(), at_cap["nn"], at_cap["by"], now=NOW, max_spread=0.05)
    assert why == "ok" and opp is not None


@pytest.mark.asyncio
async def test_a_refused_book_does_not_get_to_claim_the_best_edge(tmp_path, monkeypatch):
    """The heartbeat read "best edge +0.9600" off a 0.01/0.99 book on 2026-09-17,
    on a pair the wide-book gate had already refused."""
    wide = {"nn": _book(asks=[(0.99, 50)], bids=[(0.01, 50)]),
            "by": _book(asks=[(0.99, 50)], bids=[(0.01, 50)])}
    g, _, _, _ = _maker_guard(tmp_path, monkeypatch, books=wide)
    await g.poll_once()
    line = g.stop_summary()
    assert "book_too_wide 1" in line
    assert "+0.9" not in line


@pytest.mark.asyncio
async def test_a_disabled_rest_is_written_down_for_the_paper_fill_replay(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, enabled=False)
    await g.poll_once()
    rows = [json.loads(l) for l in (tmp_path / "positions_paper_rests.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert (r["narrow"], r["broad"]) == ("0xn", "0xb")
    assert (r["no_price"], r["yes_price"], r["shares"]) == (0.41, 0.51, 5.43)
    assert r["no_bid"] == [0.40, 50.0] and r["yes_ask"] == [0.55, 50.0]
    assert client.maker_orders == []                        # nothing was posted


def test_a_leg_with_a_bid_and_no_offer_is_refused():
    """The +0.959 of 2026-09-18: NO on narrow bid 0.02, nothing offered."""
    one_sided = {"nn": _book(bids=[(0.02, 5)]), "by": MAKER["by"]}
    opp, why = evaluate_maker(_imp(), one_sided["nn"], one_sided["by"], now=NOW)
    assert opp is None and "one-sided" in why
    assert cg.stop_key(why) == "one_sided_book"


@pytest.mark.asyncio
async def test_the_paper_rest_carries_the_one_leg_edges(tmp_path, monkeypatch):
    """Rest NO at 0.41 and take YES at its 0.55 ask — the taker leg pays its fee:
    1 − 0.41 − (0.55 + 0.05 × 0.55 × 0.45) = +0.0276."""
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, enabled=False)
    await g.poll_once()
    (r,) = [json.loads(l) for l in (tmp_path / "positions_paper_rests.jsonl").read_text().splitlines()]
    fee = 0.05 * 0.55 * 0.45
    assert abs(r["one_leg_no_edge"] - (1 - 0.41 - 0.55 - fee)) < 1e-9
    assert abs(r["one_leg_yes_edge"] - (1 - 0.45 - fee - 0.51)) < 1e-9


@pytest.mark.asyncio
async def test_a_pair_twenty_minutes_from_kick_off_can_still_rest(tmp_path, monkeypatch):
    """The taker path stops thirty minutes out; the busiest hour for resting is the last."""
    imp = _imp(days=20 / 1440)                              # kick-off in 20 minutes
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, imps=[imp], enabled=False)
    await g.poll_once()
    assert g.stats["would_rest"] == 1


@pytest.mark.asyncio
async def test_a_pair_two_minutes_from_kick_off_does_not(tmp_path, monkeypatch):
    imp = _imp(days=2 / 1440)
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, imps=[imp], enabled=False)
    await g.poll_once()
    assert g.stats["would_rest"] == 0


# ── the slot kept for taker entries ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_maker_path_does_not_take_the_slot_kept_for_takers(tmp_path, monkeypatch):
    """
    2026-09-23: with one cross slot, resting pairs held it all day (62 rests,
    no fill) while 37,794 entries waited behind them. Two slots, one of them
    never the maker path's.
    """
    g, client, breaker, _ = _maker_guard(tmp_path, monkeypatch)
    breaker.on_cross_open(5.0)                 # one slot already taken
    await g.poll_once()
    assert client.maker_orders == []
    assert breaker.status_dict()["cross_open"] == 1
    assert "taker_reserve" in g.stop_summary()


@pytest.mark.asyncio
async def test_a_taker_arbitrage_still_enters_beside_a_resting_pair(tmp_path, monkeypatch):
    from tests.test_cross_guard import ARB
    books = {**MAKER, "nn2": ARB["nn"], "by2": ARB["by"]}
    arb = _imp(narrow="0xn2", broad="0xb2", narrow_no_token="nn2", broad_yes_token="by2",
               narrow_title="Other: O/U 1.5", broad_title="Other: O/U 0.5")
    g, client, breaker, _ = _maker_guard(tmp_path, monkeypatch, books=books,
                                         imps=[_imp(), arb])
    await g.poll_once()
    assert [t for _, t, _, _ in client.maker_orders] == ["nn", "by"]     # the maker pair
    assert {t for t, _, _ in client.buys} == {"nn2", "by2"}              # and the taker
    assert breaker.status_dict()["cross_open"] == 2


@pytest.mark.asyncio
async def test_no_reserve_lets_the_maker_path_use_every_slot(tmp_path, monkeypatch):
    g, client, breaker, _ = _maker_guard(tmp_path, monkeypatch, taker_reserve=0)
    breaker.on_cross_open(5.0)
    await g.poll_once()
    assert len(client.maker_orders) == 2


# ── PolyClient.post_maker_order: the price posted is never worse than priced ──

def _maker_client(monkeypatch, asks, bids=()):
    import core.clob_client as cc
    monkeypatch.setattr(cc, "_PAPER_TRADE", False)
    c = cc.PolyClient.__new__(cc.PolyClient)
    c.posted = []

    async def _get_orderbook(token_id):
        return {"bids": [{"price": p, "size": 100.0} for p in bids],
                "asks": [{"price": p, "size": 100.0} for p in asks]}
    c.get_orderbook = _get_orderbook
    c._touch = lambda *t: None
    c._remember_orders = lambda tokens, resp: None

    async def _run_with_retry(fn, *a, **kw):
        if "price" in kw:
            c.posted.append(kw["price"])
            return "signed"
        return {"status": "live", "order_id": "0xabc"}
    c._run_with_retry = _run_with_retry
    c._create_limit = lambda **kw: kw
    c._post = lambda signed: signed
    return c


@pytest.mark.asyncio
async def test_a_bid_below_one_cent_is_posted_where_it_was_priced(monkeypatch):
    """2026-09-23 15:10: a 0.003 bid on a 0.001-tick book went out at 0.01."""
    c = _maker_client(monkeypatch, asks=[0.009])
    await c.post_maker_order("T", "BUY", 0.003, 5.02)
    assert c.posted == [0.003]


@pytest.mark.asyncio
async def test_a_bid_above_ninety_nine_cents_is_posted_where_it_was_priced(monkeypatch):
    c = _maker_client(monkeypatch, asks=[0.995])
    await c.post_maker_order("T", "BUY", 0.992, 5.02)
    assert c.posted == [0.992]


@pytest.mark.asyncio
async def test_a_bid_at_the_ask_is_lowered_below_it_not_raised(monkeypatch):
    c = _maker_client(monkeypatch, asks=[0.50])
    await c.post_maker_order("T", "BUY", 0.50, 5.0)
    assert c.posted == [0.499]


@pytest.mark.asyncio
async def test_an_offer_under_the_bid_is_raised_above_it(monkeypatch):
    c = _maker_client(monkeypatch, asks=[], bids=[0.40])
    await c.post_maker_order("T", "SELL", 0.39, 5.0)
    assert c.posted == [0.401]


@pytest.mark.asyncio
async def test_a_price_outside_every_grid_is_refused_not_moved(monkeypatch):
    c = _maker_client(monkeypatch, asks=[])
    with pytest.raises(ValueError, match="worse price"):
        await c.post_maker_order("T", "BUY", 0.0005, 5.0)     # only 0.001 exists
    await c.post_maker_order("T", "BUY", 0.9995, 5.0)         # 0.999 is cheaper: fine
    assert c.posted == [0.999]


# ── the $1 marketable minimum and the maker's own cap ─────────────────────────

@pytest.mark.asyncio
async def test_a_completion_under_the_marketable_minimum_is_not_tried(tmp_path, monkeypatch):
    """5.43 shares at 0.10 is $0.54: the exchange would refuse it, so the filled
    leg is sold back straight away instead of after a refused order."""
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fills = {"m1": 5.43}                        # only the NO leg filled
    client.books["by"] = _book(asks=[(0.10, 50)], bids=[(0.09, 50)])
    await g.poll_once()
    assert client.buys == []
    assert [t for t, _ in client.sells] == ["nn"]


@pytest.mark.asyncio
async def test_a_resting_pair_keeps_its_own_cap_when_the_taker_cap_grows(tmp_path, monkeypatch):
    g, client, breaker, _ = _maker_guard(tmp_path, monkeypatch, max_position_usdc=10.0,
                                         maker_max_usdc=5.0, taker_reserve=0)
    await g.poll_once()
    sizes = {size for _, _, _, size in client.maker_orders}
    assert sizes == {5.43}                             # 5.00 / 0.92, not 10.00 / 0.92


@pytest.mark.asyncio
async def test_with_the_live_caps_a_rest_still_leaves_a_whole_taker_slot(tmp_path, monkeypatch):
    """Taker 10, maker 5, 15 in all: a 5-dollar rest leaves the 10 a taker needs.
    With 10 in all it would not, and the maker path would never rest again."""
    g, *_ = _maker_guard(tmp_path, monkeypatch, max_position_usdc=10.0, maker_max_usdc=5.0)
    g._breaker.cross_headroom = lambda: (2, 15.0)
    assert g._leaves_taker_room(5.0)
    g._breaker.cross_headroom = lambda: (2, 10.0)
    assert not g._leaves_taker_room(5.0)


@pytest.mark.asyncio
async def test_a_pair_too_small_to_cross_can_still_be_rested(tmp_path, monkeypatch):
    """A resting bid is not a marketable order; the $1 minimum is not its limit."""
    # Crossing: 0.80 + 0.14 plus fees, +0.046 — but $5 buys 5.24 pairs, $0.73 of
    # the YES leg. Resting: 0.79 + 0.13 = 0.92, and a resting bid has no minimum.
    books = {"nn": _book(asks=[(0.80, 50)], bids=[(0.78, 50)]),
             "by": _book(asks=[(0.14, 50)], bids=[(0.12, 50)])}
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, books=books, taker_reserve=0)
    await g._priced(_imp(), NOW)
    assert client.buys == []
    assert [(t, p) for _, t, p, _ in client.maker_orders] == [("nn", 0.79), ("by", 0.13)]


# ── fills the trade feed does not report ─────────────────────────────────────
# 2026-09-24: four cheap legs (Over 7.5 goals at 0.01-0.02) filled while the feed,
# filtered on our own token, said nothing had; the guard logged "expired unfilled
# — nothing held" and the shares sat in the wallet unwatched.

@pytest.mark.asyncio
async def test_fills_are_looked_up_by_market_not_by_token(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch)
    await g.poll_once()                                # rests m1 (NO) and m2 (YES)
    await g.poll_once()
    assert ("m1", "nn", "0xn") in client.fill_queries
    assert ("m2", "by", "0xb") in client.fill_queries


@pytest.mark.asyncio
async def test_a_fill_the_feed_missed_is_found_in_the_wallet_at_expiry(tmp_path, monkeypatch):
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.balances["by"] = 5.43                       # YES filled; the feed says 0
    await g.poll_once()
    assert g.stats["feed_missed"] == 1
    # Handled like any one-leg fill: here completing it still pays (0.51 + 0.45).
    assert client.buys == [("nn", 0.45, 5.43)]
    assert len(g.open_positions()) == 1


@pytest.mark.asyncio
async def test_a_fill_the_chain_settled_late_is_found_on_the_second_look(tmp_path, monkeypatch):
    g, client, _, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    await g.poll_once()                                # expired, nothing held yet
    assert client.buys == [] and client.sells == []
    client.balances["nn"] = 5.43                       # the NO fill lands after expiry
    client.books["by"] = _book(asks=[(0.62, 50)], bids=[(0.50, 50)])
    await g._watch_maker(NOW + cg.CROSS_FILL_AUDIT_S - 1)
    assert client.sells == []                          # not due yet
    await g._watch_maker(NOW + cg.CROSS_FILL_AUDIT_S + 1)
    assert g.stats["feed_missed"] == 1
    # Completing would cost 0.41 + 0.62: the NO leg is sold back instead.
    assert client.buys == [] and client.sells == [("nn", 5.43)]
    assert any("HALF-FILL" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_a_wallet_that_already_held_the_token_is_not_a_fill(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    client.balances["by"] = 20.0                       # held before the orders went out
    await g.poll_once()
    await g.poll_once()
    await g._watch_maker(NOW + cg.CROSS_FILL_AUDIT_S + 1)
    assert g.stats["feed_missed"] == 0 and client.buys == [] and client.sells == []


@pytest.mark.asyncio
async def test_without_a_balance_reader_nothing_is_audited(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    monkeypatch.setattr(type(client), "share_balance", None, raising=False)
    await g.poll_once()
    await g.poll_once()
    assert g._audits == []


# ── completing at once, and cancels the exchange refuses ─────────────────────
# 2026-09-27: every one-leg fill came 3-5 s after posting, while the completion
# only ran at the 180-second expiry — by then the other leg's ask had gone.

@pytest.mark.asyncio
async def test_a_one_leg_fill_is_completed_at_once_when_that_pays(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch)            # 180 s to expiry
    await g.poll_once()
    client.fills = {"m1": 5.43}                        # NO filled, seconds in
    client.books["by"] = _book(asks=[(0.50, 50)], bids=[(0.49, 50)])
    await g._watch_maker(cg.time.time() + 4)
    assert client.buys == [("by", 0.50, 5.43)]
    assert "m2" in client.cancelled
    assert len(g.open_positions()) == 1


@pytest.mark.asyncio
async def test_a_one_leg_fill_keeps_resting_while_completing_would_not_pay(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch)
    await g.poll_once()
    client.fills = {"m1": 5.43}
    client.books["by"] = _book(asks=[(0.62, 50)], bids=[(0.50, 50)])
    await g._watch_maker(cg.time.time() + 4)
    assert client.buys == [] and client.sells == [] and client.cancelled == []
    assert _imp().key in g._attempts                   # the YES bid may still fill


@pytest.mark.asyncio
async def test_a_resting_pair_is_watched_every_few_seconds_between_sweeps(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, poll_s=3600.0, ws_enabled=False)
    monkeypatch.setattr(cg, "CROSS_FILL_POLL_S", 0.02)
    await g.poll_once()                                # rests a pair
    polls, watched = [], []

    async def _poll():
        polls.append(1)

    async def _watch(now):
        watched.append(now)
    g.poll_once, g._watch_maker = _poll, _watch
    task = asyncio.create_task(g.run())
    await asyncio.sleep(0.2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert polls == [1] and len(watched) >= 3


@pytest.mark.asyncio
async def test_a_refused_cancel_keeps_the_pair_watched_and_is_retried(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    real_cancel = client.cancel_order

    async def refuse(order_id):
        raise RuntimeError("order manager not ready, please retry")
    client.cancel_order = refuse
    await g._watch_maker(cg.time.time() + 1)
    assert _imp().key in g._attempts                   # not settled over live orders
    client.cancel_order = real_cancel
    await g._watch_maker(cg.time.time() + 2)
    assert _imp().key not in g._attempts
    assert set(client.cancelled) == {"m1", "m2"}


@pytest.mark.asyncio
async def test_a_cancel_refused_past_the_retry_window_settles_with_a_warning(tmp_path, monkeypatch):
    g, client, _, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()

    async def refuse(order_id):
        raise RuntimeError("order manager not ready, please retry")
    client.cancel_order = refuse
    await g._watch_maker(cg.time.time() + cg.CROSS_CANCEL_RETRY_S + 1)
    assert _imp().key not in g._attempts
    assert any("would not cancel" in m for m in notifier.messages)


# ── where and when to rest ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_joining_the_queue_rests_at_the_best_bid(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, maker_join=True)
    await g.poll_once()
    assert [(t, p) for _, t, p, _ in client.maker_orders] == [("nn", 0.40), ("by", 0.50)]


def test_joining_never_crosses_and_needs_a_bid():
    assert maker_price(MAKER["nn"], 0.01, join=True) == 0.40
    assert maker_price(_book(asks=[(0.41, 10)]), 0.01, join=True) is None


@pytest.mark.asyncio
async def test_a_book_that_just_moved_gets_no_resting_bid(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch)

    class _Pushed:
        tokens: set = set()

        def since_move(self, token):
            return 5.0 if token == "by" else None
    g._ws = _Pushed()
    await g.poll_once()
    assert client.maker_orders == []
    assert "book_moving" in g.stop_summary()
    g._ws.since_move = lambda token: 45.0              # quiet for longer than the window
    await g.poll_once()
    assert len(client.maker_orders) == 2


@pytest.mark.asyncio
async def test_a_losing_one_leg_fill_blocks_the_pair_and_its_token(tmp_path, monkeypatch):
    """The 5-year Treasury pair was rested three times in 45 minutes, each one
    filled on NO alone and bought back from us a cent lower."""
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fills = {"m1": 5.43}                        # NO alone
    client.books["by"] = _book(asks=[(0.62, 50)], bids=[(0.50, 50)])
    await g.poll_once()                                # sold back at 0.40, bought at 0.41
    assert client.sells == [("nn", 5.43)]
    now = cg.time.time()
    assert g._maker_block[_imp().key] > now and g._maker_block["nn"] > now
    books = dict(MAKER)
    stop, _ = await g._maker_attempt(_imp(), books["nn"], books["by"], now + 3600,
                                     "edge_below_min", None)
    assert stop == "maker_cooldown"
    n = len(client.maker_orders)
    await g._maker_attempt(_imp(), books["nn"], books["by"],
                           now + cg.CROSS_HALF_FILL_COOLDOWN_S + 1, "edge_below_min", None)
    assert len(client.maker_orders) == n + 2           # rested again once the block ran out


def test_a_deep_pair_is_not_rested_in():
    """13 of 14 one-leg fills to 2026-09-28 were a leg bid at 0.01-0.02."""
    deep = {"nn": _book(asks=[(0.96, 50)], bids=[(0.94, 50)]),
            "by": _book(asks=[(0.03, 50)], bids=[(0.01, 50)])}
    opp, why = evaluate_maker(_imp(), deep["nn"], deep["by"], now=NOW, min_edge=0.01)
    assert opp is None and cg.stop_key(why) == "maker_deep"
    opp, why = evaluate_maker(_imp(), deep["nn"], deep["by"], now=NOW, min_edge=0.01,
                              min_price=0.0)
    assert why == "ok"                                 # 0 switches the band off
    opp, why = evaluate_maker(_imp(), MAKER["nn"], MAKER["by"], now=NOW)
    assert why == "ok"                                 # 0.41 and 0.51 are well inside


@pytest.mark.asyncio
async def test_a_one_leg_fill_sold_back_at_a_profit_blocks_nothing(tmp_path, monkeypatch):
    g, client, _, _ = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fills = {"m1": 5.43}
    client.books["nn"] = _book(asks=[(0.45, 50)], bids=[(0.44, 50)])  # NO rose under us
    client.books["by"] = _book(asks=[(0.62, 50)], bids=[(0.50, 50)])
    await g.poll_once()
    assert client.sells == [("nn", 5.43)] and g._maker_block == {}


# ── what a sell-back leaves behind ────────────────────────────────────────────
# 2026-09-28 14:17: the bid took 0.24 of 5.23 Over 7.5 shares; the other 4.99
# stayed in the wallet and nothing offered them again.

async def _half_filled_and_partly_sold(tmp_path, monkeypatch, cap):
    g, client, breaker, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fills = {"m1": 5.43}                        # NO alone
    client.books["by"] = _book(asks=[(0.62, 50)], bids=[(0.50, 50)])   # no completion
    client.sell_cap["nn"] = cap
    await g.poll_once()
    return g, client, breaker, notifier


@pytest.mark.asyncio
async def test_a_part_sold_back_leaves_the_rest_on_offer(tmp_path, monkeypatch):
    g, client, breaker, _ = await _half_filled_and_partly_sold(tmp_path, monkeypatch, 0.24)
    assert client.sells == [("nn", 5.43)]
    (lo,) = g._leftovers
    assert (lo.token, lo.shares) == ("nn", 5.19)
    saved = json.loads((tmp_path / "positions_leftovers.json").read_text())
    assert saved[0]["shares"] == 5.19
    now = cg.time.time()
    client.sell_cap.clear()
    client.balances["nn"] = 5.19
    await g._retry_leftovers(now + 1)                  # not due yet
    assert len(client.sells) == 1
    await g._retry_leftovers(now + cg.CROSS_LEFTOVER_RETRY_S + 1)
    assert client.sells[-1] == ("nn", 5.19)
    assert g._leftovers == []
    assert json.loads((tmp_path / "positions_leftovers.json").read_text()) == []


@pytest.mark.asyncio
async def test_a_leg_the_bid_would_not_take_at_all_is_offered_again(tmp_path, monkeypatch):
    """Malta vs. Liechtenstein, 2026-09-26: "No resting liquidity", then forgotten."""
    g, client, _, notifier = _maker_guard(tmp_path, monkeypatch, maker_ttl_s=0.0)
    await g.poll_once()
    client.fills = {"m1": 5.43}
    client.books["by"] = _book(asks=[(0.62, 50)], bids=[(0.50, 50)])
    client.sell_fail.add("nn")
    await g.poll_once()
    assert [(lo.token, lo.shares) for lo in g._leftovers] == [("nn", 5.43)]
    assert any("Offered again" in m for m in notifier.messages)
    client.sell_fail.clear()
    client.balances["nn"] = 5.43
    await g._retry_leftovers(cg.time.time() + cg.CROSS_LEFTOVER_RETRY_S + 1)
    assert g._leftovers == []


@pytest.mark.asyncio
async def test_a_remainder_still_held_after_ten_minutes_raises_one_alert(tmp_path, monkeypatch):
    g, client, _, notifier = await _half_filled_and_partly_sold(tmp_path, monkeypatch, 0.24)
    client.sell_cap["nn"] = 0.0                        # the bid is gone for good
    client.balances["nn"] = 5.19
    now = cg.time.time()
    for k in range(1, 30):
        await g._retry_leftovers(now + k * cg.CROSS_LEFTOVER_RETRY_S)
    alerts = [m for m in notifier.messages if "still held" in m]
    assert len(alerts) == 1 and g._leftovers[0].shares == 5.19


@pytest.mark.asyncio
async def test_a_remainder_the_wallet_no_longer_holds_is_closed(tmp_path, monkeypatch):
    g, client, _, _ = await _half_filled_and_partly_sold(tmp_path, monkeypatch, 0.24)
    client.balances["nn"] = 0.0                        # redeemed, or sold by hand
    await g._retry_leftovers(cg.time.time() + cg.CROSS_LEFTOVER_RETRY_S + 1)
    assert g._leftovers == [] and len(client.sells) == 1


@pytest.mark.asyncio
async def test_a_remainder_is_given_up_once_its_market_has_ended(tmp_path, monkeypatch):
    g, client, _, _ = await _half_filled_and_partly_sold(tmp_path, monkeypatch, 0.24)
    client.balances["nn"] = 5.19
    await g._retry_leftovers(_imp().resolves_ts + 1)
    assert g._leftovers == [] and len(client.sells) == 1


@pytest.mark.asyncio
async def test_remainders_survive_a_restart(tmp_path, monkeypatch):
    g, client, breaker, notifier = await _half_filled_and_partly_sold(tmp_path, monkeypatch, 0.24)
    g2 = cg.CrossGuard(client, breaker, notifier,
                       implications_path=str(tmp_path / "imps.json"),
                       positions_path=str(tmp_path / "positions.json"), enabled=True)
    g2.restore()
    assert [(lo.token, lo.shares) for lo in g2._leftovers] == [("nn", 5.19)]
