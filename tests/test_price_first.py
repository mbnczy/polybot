"""
The model's budget goes where the prices would already pay.

Measured 2026-09-17: the budget followed the prefilter's shape ranking, and 94%
of what it judged were over/under families whose median edge is −38% — pairs
that cannot pay. In the same 12-hour window 14,728 pairs were priced as if an
implication were violated, and none of them had been judged at all.
"""

from __future__ import annotations

import json

import scripts.demo_cross_market as reader
from execution.cross_guard import load_implications, maker_price
from scripts.demo_cross_market import export_implications, price_first, screen_edge
from strategy.cross_market import Implication


def _mkt(cid, bid, ask, tick=0.01, q=None):
    return {"conditionId": cid, "question": q or f"market {cid}", "bestBid": bid,
            "bestAsk": ask, "orderPriceMinTickSize": tick,
            "endDate": "2099-01-01T00:00:00Z", "clobTokenIds": f'["{cid}y","{cid}n"]',
            "outcomes": '["Yes", "No"]'}


class _Cand:
    def __init__(self, a, b):
        self.a_id, self.b_id = a, b


# ── the screen ────────────────────────────────────────────────────────────────

def test_the_better_of_the_two_ways_in_is_reported():
    """Crossing: (1 − 0.90) + 0.85 = 0.95. Resting: (1 − 0.92 + 0.01) + (0.83 + 0.01) = 0.93."""
    edge = screen_edge(_mkt("n", 0.90, 0.92), _mkt("b", 0.83, 0.85))
    assert round(edge, 4) == 0.07


def test_crossing_alone_is_priced_from_the_narrow_bid_and_the_broad_ask():
    """A wide narrow book leaves nowhere to rest, so only crossing is priced."""
    edge = screen_edge(_mkt("n", 0.90, 0.99), {"conditionId": "b", "bestAsk": 0.85})
    assert round(edge, 4) == 0.05


def test_resting_our_own_bids_is_taken_when_it_is_the_better_way_in():
    # crossing: (1 − 0.40) + 0.60 = 1.20. resting: (1 − 0.50 + 0.01) + (0.50 + 0.01) = 1.02
    edge = screen_edge(_mkt("n", 0.40, 0.50), _mkt("b", 0.50, 0.60))
    assert round(edge, 4) == -0.02


def test_a_market_without_prices_screens_to_nothing():
    assert screen_edge(_mkt("n", None, None), _mkt("b", 0.5, 0.6)) is None


# ── the budget ────────────────────────────────────────────────────────────────

def _universe():
    markets = [
        _mkt("pay1", 0.95, 0.96), _mkt("pay2", 0.85, 0.86),      # pays if implied
        _mkt("pay3", 0.90, 0.91), _mkt("pay4", 0.84, 0.85),
        _mkt("flat1", 0.50, 0.51), _mkt("flat2", 0.50, 0.51),    # nothing there
        _mkt("wide1", 0.10, 0.90), _mkt("wide2", 0.10, 0.90),    # unreadable book
    ]
    cands = [_Cand("flat1", "flat2"), _Cand("wide1", "wide2"),
             _Cand("pay1", "pay2"), _Cand("pay3", "pay4")]
    return cands, markets


def test_the_pairs_the_prices_would_pay_for_come_first():
    cands, markets = _universe()
    chosen = price_first(cands, markets, share=0.5, budget=4)
    assert [(c.a_id, c.b_id) for c in chosen[:2]] == [("pay1", "pay2"), ("pay3", "pay4")]


def test_the_rest_of_the_budget_keeps_the_prefilter_order():
    cands, markets = _universe()
    chosen = price_first(cands, markets, share=0.5, budget=4)
    assert [(c.a_id, c.b_id) for c in chosen[2:]] == [("flat1", "flat2"), ("wide1", "wide2")]


def test_an_unreadable_book_never_wins_the_budget():
    """A 0.10/0.90 spread makes any pair look profitable; it is not tradeable."""
    cands, markets = _universe()
    chosen = price_first(cands, markets, share=1.0, budget=2)
    assert all(c.a_id != "wide1" for c in chosen[:1])


def test_the_budget_is_never_exceeded():
    cands, markets = _universe()
    assert len(price_first(cands, markets, share=0.75, budget=3)) == 3


# ── the tick reaches the bot ──────────────────────────────────────────────────

def test_the_export_carries_each_markets_tick(tmp_path):
    markets = [_mkt("0xn", 0.40, 0.50, tick=0.001), _mkt("0xb", 0.50, 0.60, tick=0.01)]
    path = tmp_path / "imps.json"
    export_implications([Implication("0xn", "0xb", 0.97, "e")], markets, str(path))
    (imp,) = load_implications(path)
    assert imp.narrow_tick == 0.001 and imp.broad_tick == 0.01
    # and the bot prices its bid with it
    book = {"bids": [{"price": 0.040, "size": 10}], "asks": [{"price": 0.050, "size": 10}]}
    assert maker_price(book, imp.narrow_tick) == 0.041


def test_a_market_without_a_tick_exports_none(tmp_path):
    markets = [{**_mkt("0xn", 0.4, 0.5), "orderPriceMinTickSize": None},
               _mkt("0xb", 0.5, 0.6)]
    path = tmp_path / "imps.json"
    export_implications([Implication("0xn", "0xb", 0.97, "e")], markets, str(path))
    (imp,) = load_implications(path)
    assert imp.narrow_tick is None and imp.broad_tick == 0.01


def test_a_gap_too_big_to_be_an_implication_is_dropped(caplog):
    """0.955 against 0.025 is not a dislocation: it is two unrelated markets.

    Of 2,143 confirmed implications, the 320 with a quotable book on both legs
    were violated zero times — best −0.009. Anything near +0.93 is noise.
    """
    markets = [_mkt("a", 0.95, 0.96), _mkt("b", 0.02, 0.03),
               _mkt("c", 0.95, 0.96), _mkt("d", 0.85, 0.86)]
    cands = [_Cand("a", "b"), _Cand("c", "d")]       # the big gap ranks first
    chosen = price_first(cands, markets, share=0.5, budget=2)
    assert (chosen[0].a_id, chosen[0].b_id) == ("c", "d")


def test_the_ceiling_can_be_lifted():
    markets = [_mkt("a", 0.95, 0.96), _mkt("b", 0.02, 0.03)]
    chosen = price_first([_Cand("a", "b")], markets, share=1.0, budget=1, max_edge=1.0)
    assert [(c.a_id, c.b_id) for c in chosen] == [("a", "b")]


# ── the direction the titles state ────────────────────────────────────────────

def test_a_ladder_priced_the_right_way_round_is_not_a_violation():
    """SPY above $780 at 0.005 against above $750 at 0.995, 2026-09-17.

    Scoring both ways read it as "above 750 implies above 780" and called it a
    +0.989 edge. The prices are right; the reading was backwards.
    """
    markets = [_mkt("hi", 0.001, 0.008, q="S&P 500 (SPY) closes above $780 on September 17"),
               _mkt("lo", 0.990, 0.999, q="S&P 500 (SPY) closes above $750 on September 17"),
               _mkt("c", 0.95, 0.96), _mkt("d", 0.85, 0.86)]
    cands = [_Cand("hi", "lo"), _Cand("c", "d")]     # the ladder ranks first
    chosen = price_first(cands, markets, share=0.5, budget=2, max_edge=1.0)
    assert (chosen[0].a_id, chosen[0].b_id) == ("c", "d")


def test_a_ladder_priced_the_wrong_way_round_still_counts():
    """The same two markets, with the higher threshold priced above the lower."""
    markets = [_mkt("hi", 0.60, 0.61, q="S&P 500 (SPY) closes above $780 on September 17"),
               _mkt("lo", 0.52, 0.53, q="S&P 500 (SPY) closes above $750 on September 17")]
    chosen = price_first([_Cand("hi", "lo")], markets, share=1.0, budget=1)
    assert [(c.a_id, c.b_id) for c in chosen] == [("hi", "lo")]


def test_titles_that_do_not_state_a_direction_are_left_to_the_model():
    markets = [_mkt("a", 0.95, 0.96, q="Team wins the match"),
               _mkt("b", 0.85, 0.86, q="Team wins the first half")]
    chosen = price_first([_Cand("a", "b")], markets, share=1.0, budget=1)
    assert [(c.a_id, c.b_id) for c in chosen] == [("a", "b")]
