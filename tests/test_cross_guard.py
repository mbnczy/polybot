"""
The cross-market executor: every gate before an order, both legs, the half-fill,
the early exit, resolution, and the bookkeeping that has to survive a restart.
"""

from __future__ import annotations

import json
import time

import pytest

import strategy.cross_exit as cross_exit
from execution.cross_guard import (
    CrossGuard,
    Implication,
    best_level,
    evaluate,
    load_implications,
    size_position,
    Level,
)
from risk.circuit_breaker import CircuitBreaker
from strategy.cross_exit import CrossPosition

NOW = 1_800_000_000.0
DAY = 86_400.0


@pytest.fixture(autouse=True)
def _no_fees(monkeypatch):
    """Fees off so the arithmetic in these tests is readable."""
    monkeypatch.setattr(cross_exit, "CROSS_TAKER_RATE", 0.0)


def _book(asks=(), bids=()):
    return {"asks": [{"price": p, "size": s} for p, s in asks],
            "bids": [{"price": p, "size": s} for p, s in bids]}


def _imp(days=2.0, **kw):
    base = dict(
        narrow="0xn", broad="0xb",
        narrow_title="Moik Baku vs. Simal: Moik Baku O/U 0.5",
        broad_title="Moik Baku vs. Simal: O/U 0.5",
        narrow_yes_token="ny", narrow_no_token="nn",
        broad_yes_token="by", broad_no_token="bn",
        narrow_end_ts=NOW + days * DAY, broad_end_ts=NOW + days * DAY,
        confidence=0.97,
    )
    base.update(kw)
    return Implication(**base)


class _Client:
    def __init__(self, books):
        self.books = books
        self.buys: list[tuple[str, float, float]] = []
        self.sells: list[tuple[str, float]] = []
        self.buy_fail: set[str] = set()
        self.sell_fail: set[str] = set()

    async def get_orderbook(self, token):
        return self.books.get(token, _book())

    async def post_order(self, token_id, side, price, size, _order_type="FOK"):
        self.buys.append((token_id, price, size))
        if token_id in self.buy_fail:
            return {"status": "unmatched"}
        return {"status": "matched", "making_amount": size * price, "taking_amount": size}

    async def unwind_leg(self, token_id, size, price=0.0):
        self.sells.append((token_id, size))
        if token_id in self.sell_fail:
            return {"status": "unmatched"}
        bid = best_level(self.books[token_id], "bids").price
        return {"status": "matched", "making_amount": size, "taking_amount": size * bid}


class _Notifier:
    def __init__(self):
        self.messages: list[str] = []

    async def notify(self, text):
        self.messages.append(text)


def _write_imps(tmp_path, imps):
    p = tmp_path / "imps.json"
    p.write_text(json.dumps({"generated_at": NOW, "implications": [
        {k: getattr(i, k) for k in (
            "narrow", "broad", "narrow_title", "broad_title", "narrow_yes_token",
            "narrow_no_token", "broad_yes_token", "broad_no_token",
            "narrow_end_ts", "broad_end_ts", "confidence")} for i in imps]}))
    return p


def _guard(tmp_path, monkeypatch, books, imps, *, enabled=True, **kw):
    monkeypatch.setenv("CROSS_MAX_POSITIONS", "2")
    monkeypatch.setenv("CROSS_MAX_COMMITTED_USDC", "10")
    monkeypatch.setattr(time, "time", lambda: NOW)
    client, notifier = _Client(books), _Notifier()
    breaker = CircuitBreaker(starting_balance=100.0)
    g = CrossGuard(client, breaker, notifier,
                   implications_path=str(_write_imps(tmp_path, imps)),
                   positions_path=str(tmp_path / "positions.json"),
                   enabled=enabled, **kw)
    return g, client, breaker, notifier


# Books with a real arbitrage: NO on narrow 0.40 + YES on broad 0.45 = 0.85.
ARB = {"nn": _book(asks=[(0.40, 50)]), "by": _book(asks=[(0.45, 20)])}


# ── the pure gates ────────────────────────────────────────────────────────────

def test_a_pair_locked_past_the_window_is_refused():
    opp, why = evaluate(_imp(days=8), ARB["nn"], ARB["by"], now=NOW)
    assert opp is None and "window" in why


def test_an_unknown_end_date_is_refused():
    opp, why = evaluate(_imp(narrow_end_ts=None), ARB["nn"], ARB["by"], now=NOW)
    assert opp is None and "unknown" in why


def test_a_pair_past_its_end_date_is_refused():
    opp, _ = evaluate(_imp(days=-0.1), ARB["nn"], ARB["by"], now=NOW)
    assert opp is None


def test_an_edge_that_does_not_survive_the_asks_is_refused():
    books = {"nn": _book(asks=[(0.83, 50)]), "by": _book(asks=[(0.96, 50)])}
    opp, why = evaluate(_imp(), books["nn"], books["by"], now=NOW)
    assert opp is None and "edge" in why


def test_a_leg_with_no_ask_is_refused():
    opp, _ = evaluate(_imp(), _book(), ARB["by"], now=NOW)
    assert opp is None


def test_size_is_bounded_by_the_thinner_touch_and_the_cap():
    shares = size_position(Level(0.40, 50), Level(0.45, 20), 0.85,
                           max_usdc=5.0, min_shares=5.0)
    assert shares == 5.88          # 5.00 / 0.85, under both touches


def test_a_touch_below_the_minimum_size_is_refused():
    opp, why = evaluate(_imp(), _book(asks=[(0.40, 3)]), ARB["by"], now=NOW)
    assert opp is None and "depth" in why


def test_the_best_ask_is_the_lowest_not_the_first():
    lvl = best_level(_book(asks=[(0.99, 10), (0.40, 7), (0.40, 3)]), "asks")
    assert lvl == Level(0.40, 10)


def test_a_good_pair_is_an_opportunity():
    opp, why = evaluate(_imp(), ARB["nn"], ARB["by"], now=NOW)
    assert why == "ok"
    assert opp.entry_per_pair == pytest.approx(0.85)
    assert opp.edge_per_pair == pytest.approx(0.15)


# ── entry ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_it_evaluates_but_never_orders(tmp_path, monkeypatch):
    g, client, breaker, _ = _guard(tmp_path, monkeypatch, ARB, [_imp()], enabled=False)
    await g.poll_once()
    assert client.buys == []
    assert g.stats["would_enter"] == 1
    assert breaker.status_dict()["cross_open"] == 0


@pytest.mark.asyncio
async def test_enabled_both_legs_are_bought_and_the_position_is_kept(tmp_path, monkeypatch):
    g, client, breaker, notifier = _guard(tmp_path, monkeypatch, ARB, [_imp()])
    await g.poll_once()
    assert len(client.buys) == 2
    # The thinner leg (YES on broad, 20 at the touch) went first.
    assert client.buys[0][0] == "by"
    st = breaker.status_dict()
    assert st["cross_open"] == 1
    assert st["cross_committed_usdc"] == pytest.approx(5.88 * 0.85, abs=1e-3)
    saved = json.loads((tmp_path / "positions.json").read_text())
    assert len(saved["open"]) == 1
    assert "OPENED" in notifier.messages[0]


@pytest.mark.asyncio
async def test_the_first_leg_failing_leaves_nothing_held(tmp_path, monkeypatch):
    g, client, breaker, _ = _guard(tmp_path, monkeypatch, ARB, [_imp()])
    client.buy_fail.add("by")
    await g.poll_once()
    assert len(client.buys) == 1 and client.sells == []
    assert breaker.status_dict()["cross_open"] == 0
    assert breaker.status_dict()["daily_pnl"] == 0.0


@pytest.mark.asyncio
async def test_a_half_fill_is_sold_back_and_booked(tmp_path, monkeypatch):
    books = {**ARB, "by": _book(asks=[(0.45, 20)], bids=[(0.43, 20)])}
    g, client, breaker, notifier = _guard(tmp_path, monkeypatch, books, [_imp()])
    client.buy_fail.add("nn")
    await g.poll_once()
    assert client.sells == [("by", 5.88)]
    st = breaker.status_dict()
    assert st["cross_open"] == 0
    assert st["daily_pnl"] == pytest.approx(5.88 * (0.43 - 0.45), abs=1e-6)
    assert "HALF-FILL" in notifier.messages[0]


@pytest.mark.asyncio
async def test_the_breaker_caps_entries(tmp_path, monkeypatch):
    imps = [_imp(narrow=f"0xn{i}", broad=f"0xb{i}") for i in range(3)]
    g, client, breaker, _ = _guard(tmp_path, monkeypatch, ARB, imps)
    await g.poll_once()
    assert breaker.status_dict()["cross_open"] == 2        # the cross position cap
    assert len(client.buys) == 4


@pytest.mark.asyncio
async def test_a_pair_is_not_entered_twice(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, ARB, [_imp()])
    await g.poll_once()
    await g.poll_once()
    assert len(client.buys) == 2


# ── afterwards ────────────────────────────────────────────────────────────────

def _open(g, **kw):
    pos = CrossPosition(**{**dict(
        narrow="0xn", broad="0xb", narrow_title="N", broad_title="B",
        narrow_yes_token="ny", narrow_no_token="nn",
        broad_yes_token="by", broad_no_token="bn", size=10.0,
        narrow_no_paid=0.40, broad_yes_paid=0.45, opened_ts=NOW - DAY,
        resolves_ts=NOW + DAY, paper=False), **kw})
    g._book.open(pos)
    g._breaker.on_cross_open(pos.entry_cost * pos.size)
    return pos


@pytest.mark.asyncio
async def test_it_exits_when_selling_captures_the_guaranteed_edge(tmp_path, monkeypatch):
    books = {"nn": _book(bids=[(0.55, 50)]), "by": _book(bids=[(0.50, 50)])}
    g, client, breaker, notifier = _guard(tmp_path, monkeypatch, books, [])
    _open(g)
    await g.poll_once()
    # Sale returns 1.05 against an entry of 0.85: +0.20/pair, over the +0.15 held.
    assert sorted(t for t, _ in client.sells) == ["by", "nn"]
    assert breaker.status_dict()["cross_open"] == 0
    assert breaker.status_dict()["daily_pnl"] == pytest.approx(2.0)
    assert "EXITED" in notifier.messages[0]


@pytest.mark.asyncio
async def test_it_holds_while_the_exit_is_worth_less_than_holding(tmp_path, monkeypatch):
    books = {"nn": _book(bids=[(0.45, 50)]), "by": _book(bids=[(0.45, 50)])}
    g, client, breaker, _ = _guard(tmp_path, monkeypatch, books, [])
    _open(g)
    await g.poll_once()
    assert client.sells == []
    assert breaker.status_dict()["cross_open"] == 1


@pytest.mark.asyncio
async def test_it_will_not_exit_into_a_touch_thinner_than_the_position(tmp_path, monkeypatch):
    books = {"nn": _book(bids=[(0.55, 4)]), "by": _book(bids=[(0.50, 50)])}
    g, client, _, _ = _guard(tmp_path, monkeypatch, books, [])
    _open(g)
    await g.poll_once()
    assert client.sells == []


@pytest.mark.asyncio
async def test_a_resolved_position_is_closed_at_its_payoff(tmp_path, monkeypatch):
    async def lookup(cid):
        return {"0xn": False, "0xb": True}[cid]      # the middle outcome: pays 2
    g, _, breaker, notifier = _guard(tmp_path, monkeypatch, {}, [],
                                     resolution_lookup=lookup)
    _open(g, resolves_ts=NOW - 60)
    await g.poll_once()
    assert breaker.status_dict()["cross_open"] == 0
    assert breaker.status_dict()["daily_pnl"] == pytest.approx(10 * (2.0 - 0.85))
    assert "RESOLVED" in notifier.messages[0]


@pytest.mark.asyncio
async def test_an_unsettled_market_is_waited_for(tmp_path, monkeypatch):
    async def lookup(cid):
        return None
    g, _, breaker, _ = _guard(tmp_path, monkeypatch, {}, [], resolution_lookup=lookup)
    _open(g, resolves_ts=NOW - 60)
    await g.poll_once()
    assert breaker.status_dict()["cross_open"] == 1


# ── bookkeeping that must survive a restart ───────────────────────────────────

def test_restore_puts_open_capital_back_in_the_ledger(tmp_path, monkeypatch):
    g, _, _, _ = _guard(tmp_path, monkeypatch, {}, [])
    _open(g)
    g._book.save()
    g2, _, breaker2, _ = _guard(tmp_path, monkeypatch, {}, [])
    assert g2.restore() == 1
    assert breaker2.status_dict()["cross_committed_usdc"] == pytest.approx(8.5)


def test_it_hands_its_markets_to_the_redeemer_and_the_reconciler(tmp_path, monkeypatch):
    g, _, _, _ = _guard(tmp_path, monkeypatch, {}, [])
    _open(g)
    assert g.condition_ids() == {"0xn", "0xb"}
    assert g.managed_titles() == {"N", "B"}


def test_the_reader_export_round_trips(tmp_path):
    p = _write_imps(tmp_path, [_imp()])
    loaded = load_implications(p)
    assert loaded == [_imp()]


def test_a_missing_export_means_nothing_to_trade(tmp_path):
    assert load_implications(tmp_path / "absent.json") == []
