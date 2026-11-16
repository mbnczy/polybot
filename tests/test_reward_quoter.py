"""
RewardQuoter: the bounded pilot of quoting for liquidity rewards.

The owner's terms (2026-11-15): about 25 USDC over 10-12 cheap markets for 7 days;
stop if no day pays 1 USDC of rewards after 3 days, or the loss reaches 8 USDC.
"""

from __future__ import annotations

import json

import pytest

import execution.reward_quoter as rq
from execution.reward_quoter import Quote, RewardQuoter
from risk.circuit_breaker import CircuitBreakerTripped
from strategy.reward_quotes import parse_reward_market

T0 = 1_794_000_000.0
DAY = 86_400.0


def _row(cid, yes, rate=5, price=0.03, event=None):
    return {
        "condition_id": cid, "question": f"Market {cid}", "rewards_config": [{"rate_per_day": rate}],
        "rewards_max_spread": 4.5, "rewards_min_size": 20, "volume_24hr": 10,
        "event_id": event or cid, "end_date": "2027-07-01 00:00:00+00",
        "tokens": [{"token_id": yes, "outcome": "Yes", "price": price},
                   {"token_id": yes + "-no", "outcome": "No", "price": 1 - price}],
    }


def _book(bids, asks):
    return {"bids": [{"price": p, "size": s} for p, s in bids],
            "asks": [{"price": p, "size": s} for p, s in asks]}


BOOK = _book([(0.02, 400), (0.01, 900)], [(0.03, 300), (0.05, 100)])


class _Client:
    def __init__(self, books=None):
        self.books = dict(books or {})
        self.buys, self.placed, self.cancelled = [], [], []
        self.fills: "list | None" = []
        self.balances: dict = {}
        self.scoring: "dict | None" = {}
        self.income: "list | None" = []
        self.percent = {}
        self.buy_status = "matched"
        self.refuse_side = None
        self._n = 0

    async def get_orderbooks(self, tokens):
        return {t: self.books[t] for t in tokens if t in self.books}

    async def get_orderbook(self, token):
        return self.books.get(token) or {"bids": [], "asks": []}

    async def post_order(self, token_id, side, price, size, _order_type="FOK"):
        self.buys.append((token_id, side, price, size, _order_type))
        if self.buy_status != "matched":
            return {"status": self.buy_status}
        self.balances[token_id] = self.balances.get(token_id, 0.0) + size
        return {"status": "matched", "order_id": "0xtaker", "making_amount": size * price,
                "taking_amount": size}

    async def post_maker_order(self, token_id, side, desired_price, size):
        if side == self.refuse_side:
            raise RuntimeError("not enough balance / allowance")
        self._n += 1
        oid = f"0xo{self._n}"
        self.placed.append((oid, token_id, side, desired_price, size))
        return {"status": "live", "order_id": oid}

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"canceled": [order_id], "not_canceled": {}}

    async def fills_since(self, since):
        return self.fills

    async def share_balance(self, token_id):
        return self.balances.get(token_id)

    async def orders_scoring(self, ids):
        return self.scoring if self.scoring is None else {i: self.scoring.get(i, True) for i in ids}

    async def reward_percentages(self):
        return self.percent

    async def recent_income(self, since):
        return self.income


class _Breaker:
    def __init__(self):
        self.booked = []
        self.trip = False

    def book_pnl(self, pnl):
        self.booked.append(pnl)
        if self.trip:
            raise CircuitBreakerTripped("daily loss limit")


class _Notifier:
    def __init__(self):
        self.sent = []

    async def notify(self, text, kind=None):
        self.sent.append((kind, text))


class _Fees:
    async def get_taker_rate(self, cid):
        return 0.05


def _rig(tmp_path, monkeypatch, rows=None, books=None, **kw):
    rows = [_row("0xa", "ta")] if rows is None else rows
    books = {r["tokens"][0]["token_id"]: BOOK for r in rows} if books is None else books
    client, breaker, notifier = _Client(books), _Breaker(), _Notifier()

    async def markets(keep):
        return [m for m in map(parse_reward_market, rows) if m and keep(m)]

    async def info(cid):
        return {"conditionId": cid, "closed": False, "acceptingOrders": True,
                "enableOrderBook": True}

    monkeypatch.setattr(rq, "REWARD_REQUOTE_MIN_S", 0.0)
    monkeypatch.setattr(rq, "REWARD_MAX_SPREAD_COST", 0.5)    # the test book is a cent wide
    kw.setdefault("enabled", True)
    kw.setdefault("dry_run", False)
    q = RewardQuoter(client, breaker, notifier, fee_engine=_Fees(),
                     state_path=str(tmp_path / "reward.json"),
                     fetch_markets=kw.pop("fetch_markets", markets),
                     fetch_info=kw.pop("fetch_info", info), **kw)
    return q, client, breaker, notifier


def _fill(order, side, shares, price, trade="t1", ts=T0 + 10):
    return {"trade": trade, "order": order, "token": "ta", "side": side, "shares": shares,
            "price": price, "ts": ts}


async def _opened(tmp_path, monkeypatch, **kw):
    q, c, b, n = _rig(tmp_path, monkeypatch, **kw)
    await q.poll_once(now=T0)
    return q, c, b, n


# ── off and dry run ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_off_sends_nothing(tmp_path, monkeypatch):
    q, c, _, n = _rig(tmp_path, monkeypatch, enabled=False)
    await q.poll_once(now=T0)
    assert c.buys == [] and c.placed == [] and n.sent == []
    assert q.summary() == "off"


@pytest.mark.asyncio
async def test_a_dry_run_chooses_and_sends_nothing(tmp_path, monkeypatch, caplog):
    q, c, _, n = _rig(tmp_path, monkeypatch, dry_run=True)
    with caplog.at_level("INFO"):
        await q.poll_once(now=T0)
    assert c.buys == [] and c.placed == [] and n.sent == []
    assert any("would quote" in r.message for r in caplog.records)
    assert not (tmp_path / "reward.json").exists()


# ── opening ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_it_buys_the_lot_and_rests_both_sides(tmp_path, monkeypatch):
    q, c, _, n = await _opened(tmp_path, monkeypatch)
    assert c.buys == [("ta", "BUY", 0.03, 34.0, "FAK")]
    sides = {(side, price, size) for _, _, side, price, size in c.placed}
    assert sides == {("SELL", 0.03, 34.0), ("BUY", 0.02, 34.0)}
    held = q._quotes["0xa"]
    assert held.shares == 34.0
    # 1.02 spent, and the taker fee: 0.05 x (1 - 0.03) of it.
    assert held.cost == pytest.approx(1.02 * (1 + 0.05 * 0.97))
    assert q.committed() == pytest.approx(held.cost + 34 * 0.02)
    assert n.sent[0][0] == "trade" and "REWARD QUOTES STARTED" in n.sent[0][1]
    assert q.managed_titles() == {"Market 0xa"}


@pytest.mark.asyncio
async def test_it_stays_inside_the_budget(tmp_path, monkeypatch):
    rows = [_row(f"0x{i}", f"t{i}", rate=5 + i) for i in range(8)]
    q, c, *_ = await _opened(tmp_path, monkeypatch, rows=rows, budget=5.5)
    assert len(q._quotes) == 3                       # 1.70 a market
    assert q.committed() <= 5.5 + 3 * 0.06           # the fees are the only overshoot


@pytest.mark.asyncio
async def test_it_takes_no_more_markets_than_allowed(tmp_path, monkeypatch):
    rows = [_row(f"0x{i}", f"t{i}") for i in range(8)]
    q, *_ = await _opened(tmp_path, monkeypatch, rows=rows, max_markets=2)
    assert len(q._quotes) == 2


@pytest.mark.asyncio
async def test_a_market_not_taking_orders_is_left(tmp_path, monkeypatch):
    async def info(cid):
        return {"conditionId": cid, "closed": False, "acceptingOrders": False}
    q, c, *_ = await _opened(tmp_path, monkeypatch, fetch_info=info)
    assert c.buys == [] and q._quotes == {}


@pytest.mark.asyncio
async def test_a_buy_that_did_not_fill_opens_nothing(tmp_path, monkeypatch):
    q, c, _, n = _rig(tmp_path, monkeypatch)
    c.buy_status = "unmatched"
    await q.poll_once(now=T0)
    assert q._quotes == {} and c.placed == [] and n.sent == []
    assert q._started_ts is None


@pytest.mark.asyncio
async def test_a_failed_listing_read_opens_nothing(tmp_path, monkeypatch):
    async def markets(keep):
        return None
    q, c, *_ = await _opened(tmp_path, monkeypatch, fetch_markets=markets)
    assert c.buys == []


# ── fills ─────────────────────────────────────────────────────────────────────

def _ids(c):
    return {side: oid for oid, _, side, _, _ in c.placed}


@pytest.mark.asyncio
async def test_a_bid_fill_adds_shares_at_its_price_and_they_are_offered(tmp_path, monkeypatch):
    q, c, b, n = await _opened(tmp_path, monkeypatch)
    ids = _ids(c)
    cost0 = q._quotes["0xa"].cost
    c.fills = [_fill(ids["BUY"], "BUY", 34.0, 0.02)]
    c.balances["ta"] = 68.0
    await q.poll_once(now=T0 + 60)
    held = q._quotes["0xa"]
    assert held.shares == 68.0 and held.cost == pytest.approx(cost0 + 0.68)
    assert b.booked == []                            # nothing is realised by buying
    assert ("SELL", 0.03, 68.0) in {(s, p, z) for _, _, s, p, z in c.placed}
    assert any("bought 34.00" in t for _, t in n.sent)


@pytest.mark.asyncio
async def test_an_ask_fill_books_the_result(tmp_path, monkeypatch):
    q, c, b, n = await _opened(tmp_path, monkeypatch)
    ids = _ids(c)
    avg = q._quotes["0xa"].avg_cost
    c.fills = [_fill(ids["SELL"], "SELL", 14.0, 0.03)]
    c.balances["ta"] = 20.0
    await q.poll_once(now=T0 + 60)
    held = q._quotes["0xa"]
    assert held.shares == pytest.approx(20.0)
    assert b.booked == [pytest.approx(14 * (0.03 - avg), abs=1e-6)]
    assert b.booked[0] < 0, "the taker fee on the way in is the cost of one round trip"
    assert any(k == "trade" and "sold 14.00" in t for k, t in n.sent)


@pytest.mark.asyncio
async def test_a_fill_is_applied_once(tmp_path, monkeypatch):
    q, c, b, _ = await _opened(tmp_path, monkeypatch)
    c.fills = [_fill(_ids(c)["SELL"], "SELL", 14.0, 0.03)]
    c.balances["ta"] = 20.0
    await q.poll_once(now=T0 + 60)
    await q.poll_once(now=T0 + 120)
    assert len(b.booked) == 1 and q._quotes["0xa"].shares == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_somebody_elses_order_in_the_feed_is_ignored(tmp_path, monkeypatch):
    q, c, b, _ = await _opened(tmp_path, monkeypatch)
    c.fills = [_fill("0xtaker", "BUY", 34.0, 0.03), _fill("0xsomeone", "SELL", 50.0, 0.03, "t2")]
    await q.poll_once(now=T0 + 60)
    assert q._quotes["0xa"].shares == 34.0 and b.booked == []


@pytest.mark.asyncio
async def test_a_fill_on_an_order_already_moved_still_counts(tmp_path, monkeypatch):
    q, c, b, _ = await _opened(tmp_path, monkeypatch)
    old_ask = _ids(c)["SELL"]
    c.books["ta"] = _book([(0.02, 400)], [(0.04, 300)])       # the ask moves up
    await q.poll_once(now=T0 + 60)
    assert old_ask in c.cancelled
    c.fills = [_fill(old_ask, "SELL", 34.0, 0.03)]            # it had filled as it was cancelled
    c.balances["ta"] = 0.0
    await q.poll_once(now=T0 + 120)
    assert len(b.booked) == 1


@pytest.mark.asyncio
async def test_a_market_sold_out_is_closed_and_its_bid_cancelled(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    ids = _ids(c)
    c.fills = [_fill(ids["SELL"], "SELL", 34.0, 0.03)]
    c.balances["ta"] = 0.0
    await q.poll_once(now=T0 + 60)
    assert q._quotes == {} and ids["BUY"] in c.cancelled
    assert q._closed[0]["title"] == "Market 0xa"


@pytest.mark.asyncio
async def test_a_tripped_day_propagates_with_the_state_saved(tmp_path, monkeypatch):
    q, c, b, _ = await _opened(tmp_path, monkeypatch)
    b.trip = True
    c.fills = [_fill(_ids(c)["SELL"], "SELL", 14.0, 0.03)]
    with pytest.raises(CircuitBreakerTripped):
        await q.poll_once(now=T0 + 60)
    saved = json.loads((tmp_path / "reward.json").read_text())
    assert saved["quotes"][0]["shares"] == pytest.approx(20.0)


# ── following the book ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orders_follow_the_book(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    ids = _ids(c)
    c.books["ta"] = _book([(0.01, 900)], [(0.02, 300)])
    await q.poll_once(now=T0 + 60)
    assert set(c.cancelled) == {ids["BUY"], ids["SELL"]}
    assert {(s, p) for _, _, s, p, _ in c.placed[2:]} == {("SELL", 0.02), ("BUY", 0.01)}


@pytest.mark.asyncio
async def test_orders_that_are_right_are_left_alone(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    # The book now shows our own orders beside the others'.
    c.books["ta"] = _book([(0.02, 434), (0.01, 900)], [(0.03, 334), (0.05, 100)])
    await q.poll_once(now=T0 + 60)
    assert c.cancelled == [] and len(c.placed) == 2


@pytest.mark.asyncio
async def test_alone_at_the_top_it_steps_back_to_the_others(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    c.books["ta"] = _book([(0.02, 34), (0.01, 900)], [(0.03, 334)])   # only we bid 0.02
    await q.poll_once(now=T0 + 60)
    assert ("BUY", 0.01) in {(s, p) for _, _, s, p, _ in c.placed[2:]}


@pytest.mark.asyncio
async def test_an_order_is_not_moved_before_it_has_rested(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    monkeypatch.setattr(rq, "REWARD_REQUOTE_MIN_S", 90.0)
    c.books["ta"] = _book([(0.01, 900)], [(0.02, 300)])
    await q.poll_once(now=T0 + 30)
    assert c.cancelled == []


@pytest.mark.asyncio
async def test_no_bid_where_the_touch_is_too_wide_to_score(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    ids = _ids(c)
    c.books["ta"] = _book([(0.01, 900)], [(0.11, 300)])
    await q.poll_once(now=T0 + 60)
    assert ids["BUY"] in c.cancelled
    assert not any(s == "BUY" for _, _, s, _, _ in c.placed[2:])
    assert ("SELL", 0.11) in {(s, p) for _, _, s, p, _ in c.placed[2:]}, "the shares stay offered"


@pytest.mark.asyncio
async def test_the_bid_stops_when_the_market_has_cost_its_cap(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch, max_cost=1.9)
    ids = _ids(c)
    c.fills = [_fill(ids["BUY"], "BUY", 34.0, 0.02)]
    c.balances["ta"] = 68.0
    await q.poll_once(now=T0 + 60)
    held = q._quotes["0xa"]
    assert held.cost > 1.7 and not held.bid_id, "another lot would pass the cap"


@pytest.mark.asyncio
async def test_the_ask_is_cut_to_what_the_wallet_holds(tmp_path, monkeypatch):
    q, c, *_ = _rig(tmp_path, monkeypatch)
    orig = c.post_order

    async def short(token_id, side, price, size, _order_type="FOK"):
        resp = await orig(token_id, side, price, size, _order_type)
        c.balances[token_id] = 33.98                 # the chain settles a hair under
        return resp

    c.post_order = short
    await q.poll_once(now=T0)
    assert ("SELL", 0.03, 33.98) in {(s, p, z) for _, _, s, p, z in c.placed}


@pytest.mark.asyncio
async def test_a_refused_order_is_tried_again_next_poll(tmp_path, monkeypatch):
    q, c, *_ = _rig(tmp_path, monkeypatch)
    c.refuse_side = "SELL"
    await q.poll_once(now=T0)
    assert not q._quotes["0xa"].ask_id
    c.refuse_side = None
    await q.poll_once(now=T0 + 60)
    assert q._quotes["0xa"].ask_id


# ── the wallet ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shares_gone_from_the_wallet_twice_running_are_booked_as_sold(tmp_path, monkeypatch):
    q, c, b, _ = await _opened(tmp_path, monkeypatch)
    c.balances["ta"] = 14.0
    monkeypatch.setattr(rq.time, "time", lambda: T0 + 600)
    q._next.pop("balances", None)
    await q.poll_once(now=T0 + 600)
    assert q._quotes["0xa"].shares == 34.0, "one read is not enough"
    q._next.pop("balances", None)
    await q.poll_once(now=T0 + 1200)
    assert q._quotes["0xa"].shares == pytest.approx(14.0) and len(b.booked) == 1


@pytest.mark.asyncio
async def test_extra_shares_in_the_wallet_are_never_offered(tmp_path, monkeypatch):
    """They belong to something else: a bid's fill is attributed by its order id."""
    q, c, b, _ = await _opened(tmp_path, monkeypatch)
    c.balances["ta"] = 90.0
    monkeypatch.setattr(rq.time, "time", lambda: T0 + 600)
    for i in (1, 2, 3):
        q._next.pop("balances", None)
        await q.poll_once(now=T0 + 600 * i)
    assert q._quotes["0xa"].shares == 34.0 and b.booked == []
    assert all(z <= 34.0 for _, _, s, _, z in c.placed if s == "SELL")


# ── rewards and scoring ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_income_is_reported_once_and_counted_by_day(tmp_path, monkeypatch):
    q, c, _, n = await _opened(tmp_path, monkeypatch)
    c.income = [{"type": "REWARD", "usdc": 1.37, "ts": int(T0 + DAY)},
                {"type": "MAKER_REBATE", "usdc": 0.05, "ts": int(T0 + DAY)}]
    q._next.pop("income", None)
    await q.poll_once(now=T0 + DAY + 60)
    q._next.pop("income", None)
    await q.poll_once(now=T0 + DAY + 120)
    paid = [t for k, t in n.sent if "REWARDS PAID" in t]
    assert len(paid) == 2 and all(k == "trade" for k, t in n.sent if "REWARDS PAID" in t)
    assert q.rewards() == pytest.approx(1.37)


@pytest.mark.asyncio
async def test_scoring_is_what_the_exchange_says_of_both_orders(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    assert q._quotes["0xa"].scoring is True
    c.scoring = {_ids(c)["BUY"]: False}
    q._next.pop("scoring", None)
    await q.poll_once(now=T0 + 600)
    assert q._quotes["0xa"].scoring is False
    assert "0 scoring" in q.summary()


# ── stopping ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_it_stops_at_the_loss_limit_and_keeps_the_asks(tmp_path, monkeypatch):
    q, c, _, n = await _opened(tmp_path, monkeypatch, max_loss=0.5)
    ids = _ids(c)
    c.books["ta"] = _book([(0.001, 900)], [(0.03, 300)])      # the bid side collapses
    await q.poll_once(now=T0 + 60)
    assert "limit" in q._stopped and ids["BUY"] in c.cancelled
    assert q._quotes["0xa"].ask_id and not q._quotes["0xa"].bid_id
    assert n.sent[-1][0] == "critical" and "REWARD QUOTES STOPPED" in n.sent[-1][1]
    placed = len(c.placed)
    c.books["ta"] = BOOK
    await q.poll_once(now=T0 + 120)
    assert not any(s == "BUY" for _, _, s, _, _ in c.placed[placed:]), "no bids once stopped"


@pytest.mark.asyncio
async def test_it_stops_when_three_days_paid_no_dollar(tmp_path, monkeypatch):
    q, c, _, n = await _opened(tmp_path, monkeypatch)
    c.income = [{"type": "REWARD", "usdc": 0.4, "ts": int(T0 + DAY)}]
    q._next.pop("income", None)
    await q.poll_once(now=T0 + 3 * DAY)
    assert not q._stopped, "the third day's payout has an hour to arrive"
    await q.poll_once(now=T0 + 3 * DAY + 3700)
    assert "no day paid" in q._stopped


@pytest.mark.asyncio
async def test_a_day_that_paid_a_dollar_keeps_it_going_until_the_pilot_ends(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    c.income = [{"type": "REWARD", "usdc": 1.2, "ts": int(T0 + DAY)}]
    q._next.pop("income", None)
    await q.poll_once(now=T0 + 3 * DAY + 3700)
    assert not q._stopped
    await q.poll_once(now=T0 + 7 * DAY + 1)
    assert "pilot is over" in q._stopped


@pytest.mark.asyncio
async def test_nothing_new_is_opened_once_stopped(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch, max_markets=2)
    await q._stop("test", T0)
    q._next.pop("select", None)
    buys = len(c.buys)
    await q.poll_once(now=T0 + 60)
    assert len(c.buys) == buys


# ── restarts ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_restart_keeps_the_shares_cancels_the_old_orders_and_requotes(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    old = set(_ids(c).values())
    q2, c2, _, n2 = _rig(tmp_path, monkeypatch)
    c2.balances["ta"] = 34.0
    assert q2.restore() == 1
    assert q2.managed_titles() == {"Market 0xa"}
    await q2.poll_once(now=T0 + 300)
    assert set(c2.cancelled) == old
    assert c2.buys == [], "the shares are already held"
    assert {(s, p, z) for _, _, s, p, z in c2.placed} == {("SELL", 0.03, 34.0), ("BUY", 0.02, 34.0)}
    assert q2._started_ts == T0


@pytest.mark.asyncio
async def test_switched_off_with_shares_held_it_still_owns_them_and_cancels_what_rested(tmp_path, monkeypatch):
    q, c, *_ = await _opened(tmp_path, monkeypatch)
    old = set(_ids(c).values())
    q2, c2, *_ = _rig(tmp_path, monkeypatch, enabled=False)
    q2.restore()
    await q2.poll_once(now=T0 + 300)
    assert set(c2.cancelled) == old and c2.placed == []
    assert q2.managed_titles() == {"Market 0xa"}
    assert "off" in q2.summary() and "1 market(s)" in q2.summary()


def test_an_unreadable_state_file_starts_empty(tmp_path, monkeypatch):
    (tmp_path / "reward.json").write_text("{not json")
    q, *_ = _rig(tmp_path, monkeypatch)
    assert q.restore() == 0


def test_a_quote_round_trips():
    q = Quote(condition_id="0xa", title="T", outcome="Yes", token_id="t", min_size=20,
              max_spread=4.5, rate=5, lot=34, shares=34, cost=1.07)
    assert Quote.from_dict({**q.as_dict(), "from_a_newer_version": 1}) == q
