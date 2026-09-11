"""
The quantity guard where it matters: before the model, and before an order.
"""

from __future__ import annotations

from execution.cross_guard import Implication, evaluate
from strategy.implication_mapper import build_candidates

M = "Al Faisaly Saudi Club vs. Al Ittihad Saudi Club"
NOW = 1_800_000_000.0


def _mkt(cid, q):
    return {"conditionId": cid, "question": q, "events": [{"id": "fx"}]}


def test_the_prefilter_never_sends_corners_against_goals_to_the_model():
    c = build_candidates([
        _mkt("c", f"{M}: Al Faisaly Saudi Club O/U 2.5 Corners"),
        _mkt("g", f"{M}: Al Faisaly Saudi Club O/U 2.5"),
    ], max_per_event=0)
    assert c == []


def test_the_prefilter_still_sends_a_real_nest():
    c = build_candidates([
        _mkt("t", f"{M}: Al Faisaly Saudi Club O/U 0.5"),
        _mkt("m", f"{M}: O/U 0.5"),
    ], max_per_event=0)
    assert len(c) == 1 and c[0].shape == "nest"


def test_the_guard_refuses_it_even_if_the_file_says_otherwise():
    imp = Implication(
        narrow="c", broad="g",
        narrow_title=f"{M}: Al Faisaly Saudi Club O/U 2.5 Corners",
        broad_title=f"{M}: Al Faisaly Saudi Club O/U 2.5",
        narrow_yes_token="ny", narrow_no_token="nn",
        broad_yes_token="by", broad_no_token="bn",
        narrow_end_ts=NOW + 3600, broad_end_ts=NOW + 3600,
    )
    # The live books that cleared every other gate: entry 0.7585.
    books = {"asks": [{"price": 0.59, "size": 5}]}, {"asks": [{"price": 0.15, "size": 32}]}
    opp, why = evaluate(imp, books[0], books[1], now=NOW)
    assert opp is None and "statistic" in why
