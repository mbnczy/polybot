"""
Resting our own bids instead of crossing the spread.

The taker path pays the ask plus a fee on both legs, and in six days of live
evaluation its best edge was −1%. A maker fill pays no fee and buys a tick above
the bid — the same pair can be tradeable there. What it costs is queue risk: an
order can rest unfilled, or one leg can fill alone.
"""

from __future__ import annotations

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
        self._n = 0

    async def post_maker_order(self, token_id, side, desired_price, size):
        if token_id in self.post_fail:
            return {"status": "error"}
        self._n += 1
        oid = f"m{self._n}"
        self.maker_orders.append((oid, token_id, desired_price, size))
        return {"status": "live", "order_id": oid}

    async def order_filled_size(self, order_id, token_id=None):
        return self.fills.get(order_id, 0.0)

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
    assert breaker.status_dict()["session_pnl"] == pytest.approx(5.43 * (0.40 - 0.41), abs=1e-6)
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
