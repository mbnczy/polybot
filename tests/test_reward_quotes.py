"""
What a minimum two-sided quote scores, costs and can lose (strategy/reward_quotes.py).

The scoring follows Polymarket's published liquidity-rewards methodology; the worked
example in the docs is reproduced below.
"""

from __future__ import annotations

import pytest

from strategy.reward_quotes import (
    RewardMarket,
    best_other,
    bid_shares_within,
    book_scores,
    depth_to,
    eligible,
    order_score,
    parse_reward_market,
    plan_quote,
    q_min,
    scoring_now,
    select,
    shares_to_buy,
)

NOW = 1_794_000_000.0
DAY = 86_400.0


def _row(**kw) -> dict:
    row = {
        "condition_id": "0xc1", "question": "Will Al-Hilal be relegated?",
        "rewards_config": [{"rate_per_day": 5}], "rewards_max_spread": 4.5,
        "rewards_min_size": 20, "volume_24hr": 0, "event_id": "e1",
        "end_date": "2027-07-01 00:00:00+00",
        "tokens": [{"token_id": "yes", "outcome": "Yes", "price": 0.03},
                   {"token_id": "no", "outcome": "No", "price": 0.97}],
    }
    row.update(kw)
    return row


def _market(**kw) -> RewardMarket:
    return parse_reward_market(_row(**kw))


def _book(bids, asks) -> dict:
    return {"bids": [{"price": p, "size": s} for p, s in bids],
            "asks": [{"price": p, "size": s} for p, s in asks]}


# ── the listing ───────────────────────────────────────────────────────────────

def test_a_row_is_read_with_its_cheaper_token():
    m = _market()
    assert m.rate == 5 and m.max_spread == 4.5 and m.min_size == 20
    assert m.cheaper() == ("yes", "Yes", 0.03)
    assert m.end_ts == pytest.approx(1814400000.0)


def test_the_cheaper_token_can_be_no():
    m = _market(tokens=[{"token_id": "yes", "outcome": "Yes", "price": 0.95},
                        {"token_id": "no", "outcome": "No", "price": 0.05}])
    assert m.cheaper()[:2] == ("no", "No")


@pytest.mark.parametrize("bad", [
    {"rewards_config": []},                       # pays nothing today
    {"rewards_config": [{"rate_per_day": 0}]},
    {"rewards_max_spread": 0},
    {"rewards_min_size": 0},
    {"tokens": [{"token_id": "yes", "outcome": "Yes", "price": 0.03}]},
    {"tokens": None},
])
def test_a_row_that_pays_nothing_or_is_malformed_is_no_market(bad):
    assert parse_reward_market(_row(**bad)) is None


def test_several_reward_configs_add_up():
    assert _market(rewards_config=[{"rate_per_day": 5}, {"rate_per_day": 2.5}]).rate == 7.5


# ── scoring: the documented example ───────────────────────────────────────────

def test_order_score_is_quadratic_and_zero_outside_the_band():
    assert order_score(3.0, 1.0) == pytest.approx((2 / 3) ** 2)
    assert order_score(3.0, 0.0) == 1.0
    assert order_score(3.0, 3.0) == 0.0
    assert order_score(3.0, 4.0) == 0.0
    assert order_score(0.0, 1.0) == 0.0


def test_the_worked_example_from_the_docs():
    """Midpoint 0.50, max spread 3c: 100 @ 0.49, 200 @ 0.48 bid; 100 @ 0.51 ask."""
    bids, asks = book_scores(_book([(0.49, 100), (0.48, 200)], [(0.51, 100)]), 0.50, 3.0)
    assert bids == pytest.approx((2 / 3) ** 2 * 100 + (1 / 3) ** 2 * 200)
    assert asks == pytest.approx((2 / 3) ** 2 * 100)


def test_one_sided_scores_a_third_in_the_middle_and_nothing_at_the_extremes():
    assert q_min(90.0, 0.0, 0.50) == pytest.approx(30.0)
    assert q_min(90.0, 60.0, 0.50) == pytest.approx(60.0)
    assert q_min(90.0, 0.0, 0.05) == 0.0
    assert q_min(90.0, 60.0, 0.95) == pytest.approx(60.0)


# ── reading the book ──────────────────────────────────────────────────────────

def test_best_other_looks_through_our_own_order():
    book = _book([(0.02, 20), (0.019, 500)], [(0.03, 300)])
    assert best_other(book, "bids") == (0.02, 20)
    assert best_other(book, "bids", own_price=0.02, own_size=20) == (0.019, 500)
    # Others rest beside us: the level is still theirs to join.
    assert best_other(_book([(0.02, 70)], []), "bids", own_price=0.02, own_size=20) == (0.02, 50)


def test_dust_is_not_a_level_to_join():
    """One share offered far inside the spread must not price a whole lot."""
    book = _book([(0.02, 400)], [(0.021, 1), (0.03, 300)])
    assert best_other(book, "asks", min_level=20) == (0.03, 300)
    assert best_other(book, "asks") == (0.021, 1)


def test_depth_counts_everything_at_the_price_or_better():
    book = _book([(0.02, 5), (0.019, 7)], [(0.021, 1), (0.03, 300), (0.04, 50)])
    assert depth_to(book, "asks", 0.03) == 301
    assert depth_to(book, "bids", 0.019) == 12


# ── the plan ──────────────────────────────────────────────────────────────────

def test_a_buy_under_a_dollar_is_sized_up_to_one():
    """The exchange refuses a marketable BUY under $1: 20 shares at 0.03 is $0.60."""
    assert shares_to_buy(0.03, 20, 1.02) == 34.0
    assert shares_to_buy(0.03, 20, 1.02) * 0.03 >= 1.02
    assert shares_to_buy(0.06, 20, 1.02) == 20
    assert shares_to_buy(0.0, 20, 1.02) == 0.0


def test_a_plan_on_a_cheap_token():
    m = _market()
    book = _book([(0.02, 400), (0.01, 900)], [(0.03, 300), (0.05, 100)])
    p = plan_quote(m, book, max_cost=2.5, max_spread_cost=0.5)
    assert p.token_id == "yes" and (p.bid, p.ask) == (0.02, 0.03)
    assert p.buy_shares == 34.0 and p.bid_shares == 34.0
    assert p.cost == pytest.approx(34 * 0.03 + 34 * 0.02)
    score = order_score(4.5, 0.5)                    # each side half a cent from 0.025
    assert p.ours == pytest.approx(34 * score)
    assert 0.0 < p.share < 1.0 and p.per_day == pytest.approx(5 * p.share)


def test_the_bid_is_cut_to_what_the_cost_allows():
    m = _market(tokens=[{"token_id": "yes", "outcome": "Yes", "price": 0.06},
                        {"token_id": "no", "outcome": "No", "price": 0.94}])
    book = _book([(0.05, 400)], [(0.06, 300)])
    p = plan_quote(m, book, max_cost=2.5)
    assert p.buy_shares == 20 and p.bid_shares == 20         # 1.20 + 1.00
    assert plan_quote(m, book, max_cost=2.0) is None         # only 16 fit: under the minimum
    assert p.cost <= 2.5


@pytest.mark.parametrize("book,why", [
    (_book([], [(0.03, 300)]), "no bid"),
    (_book([(0.02, 400)], []), "no ask"),
    (_book([(0.02, 5)], [(0.03, 300)]), "nobody bids a full lot"),
    (_book([(0.02, 400)], [(0.03, 30)]), "the asks cannot fill the buy"),
    (_book([(0.01, 400)], [(0.09, 300)]), "too wide to score"),
])
def test_no_plan_where_a_quote_cannot_be_placed(book, why):
    assert plan_quote(_market(), book, max_cost=2.5, min_score=0.2, max_spread_cost=5.0) is None, why


def test_no_plan_where_the_spread_on_the_shares_costs_too_much():
    """Bought at the ask, worth the bid: 34 shares across a 5-cent spread lose 1.70."""
    m = _market()
    wide = _book([(0.02, 400)], [(0.07, 300)])
    assert plan_quote(m, wide, max_cost=5.0, min_score=0.0, max_spread_cost=5.0) is not None
    assert plan_quote(m, wide, max_cost=5.0, min_score=0.0, max_spread_cost=0.25) is None
    fine = _book([(0.020, 400)], [(0.021, 300)])              # a 0.001 tick: 0.05 lost
    assert plan_quote(m, fine, max_cost=5.0) is not None


def test_no_plan_without_a_book():
    assert plan_quote(_market(), None, max_cost=2.5) is None


# ── which markets ─────────────────────────────────────────────────────────────

_KW = dict(max_price=0.08, min_days=15.0, max_volume=3000.0, min_rate=2.0, max_min_size=50.0)


def test_a_quiet_cheap_far_off_market_is_eligible():
    assert eligible(_market(), NOW, **_KW)


@pytest.mark.parametrize("change,why", [
    ({"end_date": "2026-11-20 00:00:00+00"}, "ends inside the cross path's window"),
    ({"end_date": None}, "no end date"),
    ({"volume_24hr": 5000}, "busy"),
    ({"rewards_config": [{"rate_per_day": 1}]}, "pool too small"),
    ({"rewards_min_size": 200}, "minimum size too large"),
    ({"tokens": [{"token_id": "yes", "outcome": "Yes", "price": 0.30},
                 {"token_id": "no", "outcome": "No", "price": 0.70}]}, "too dear"),
])
def test_what_makes_a_market_ineligible(change, why):
    assert not eligible(_market(**change), NOW, **_KW), why


def test_a_market_already_held_is_left_out():
    assert not eligible(_market(), NOW, excluded={"0xc1"}, **_KW)


def _plan(cid, event, rate, cost_ask=0.03, theirs_bids=400):
    m = _market(condition_id=cid, event_id=event, rewards_config=[{"rate_per_day": rate}])
    return plan_quote(m, _book([(round(cost_ask - 0.01, 3), theirs_bids)], [(cost_ask, 300)]),
                      max_cost=2.5, max_spread_cost=0.5)


def test_selection_takes_the_best_ratio_first_one_market_an_event_inside_the_budget():
    plans = [_plan("0xa", "e1", 5), _plan("0xb", "e1", 50), _plan("0xc", "e2", 10),
             _plan("0xd", "e3", 2)]
    chosen = select(plans, budget=3.5, max_markets=12)
    assert [p.market.condition_id for p in chosen] == ["0xb", "0xc"]
    assert sum(p.cost for p in chosen) <= 3.5


def test_selection_stops_at_the_market_count():
    plans = [_plan(f"0x{i}", f"e{i}", 5 + i) for i in range(6)]
    assert len(select(plans, budget=100.0, max_markets=3)) == 3


def test_an_empty_book_does_not_outrank_everything():
    """A thin moment promises the whole pool; the rank is capped at half of it."""
    thin = _plan("0xthin", "e1", 10, theirs_bids=20)
    full = _plan("0xfull", "e2", 55, theirs_bids=400)
    assert thin.share > 0.5
    assert thin.per_day > full.per_day, "uncapped, the thin book would rank first"
    assert [p.market.condition_id for p in select([thin, full], budget=10, max_markets=1)] == ["0xfull"]


# ── while a quote rests ───────────────────────────────────────────────────────

def test_the_bid_shrinks_as_shares_are_held_and_stops_under_the_minimum():
    assert bid_shares_within(2.5, 1.02, 0.02, 34, 20) == 34
    assert bid_shares_within(2.5, 1.90, 0.02, 34, 20) == 30
    assert bid_shares_within(2.5, 2.20, 0.02, 34, 20) == 0.0
    assert bid_shares_within(2.5, 1.0, 0.0, 34, 20) == 0.0


def test_what_counts_as_a_scoring_quote():
    assert scoring_now(0.02, 34, 0.03, 34, 20, 4.5)
    assert not scoring_now(None, 0, 0.03, 34, 20, 4.5)
    assert not scoring_now(0.02, 10, 0.03, 34, 20, 4.5)       # bid under the minimum size
    assert not scoring_now(0.01, 34, 0.11, 34, 20, 4.5)       # 5c each side of the midpoint
