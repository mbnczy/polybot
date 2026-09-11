"""
Time priority in the implication prefilter.

A cross-market pair has no complete set to merge, so its arbitrage pays only when
the LATER leg resolves. The same 21% edge is 86% a year over 112 days and
several thousand percent over three, so the model budget should go to the pairs
that close soonest — but never at the expense of shape, which decides whether an
implication can exist at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import strategy.implication_mapper as im
from strategy.implication_mapper import _end_ts, build_candidates, time_score


def _mkt(cid, question, end_days=None, event=None):
    m = {"conditionId": cid, "question": question}
    if end_days is not None:
        end = datetime.now(timezone.utc) + timedelta(days=end_days)
        m["endDate"] = end.isoformat().replace("+00:00", "Z")
    if event:
        m["events"] = [{"id": event}]
    return m


# ── the score ─────────────────────────────────────────────────────────────────

def test_time_score_halves_every_half_life():
    assert time_score(0.0) == pytest.approx(1.0)
    assert time_score(30.0) == pytest.approx(0.5)
    assert time_score(112.0) == pytest.approx(0.211, abs=0.001)
    assert time_score(365.0) < 0.1


def test_an_unknown_lockup_earns_no_credit_for_speed():
    """Capital cost that cannot be priced gets no reward for speed it may lack."""
    assert time_score(None) == 0.0


def test_end_dates_parse_in_gamma_format():
    ts = _end_ts({"endDate": "2026-12-31T00:00:00Z"})
    assert ts == datetime(2026, 12, 31, tzinfo=timezone.utc).timestamp()
    assert _end_ts({}) is None
    assert _end_ts({"endDate": "not a date"}) is None


# ── the ordering ──────────────────────────────────────────────────────────────

def test_lockup_is_set_by_the_later_leg():
    """The money sits until BOTH have resolved."""
    c = build_candidates([
        _mkt("a", "Will BTC reach $80,000 by December 31?", 5, event="e"),
        _mkt("b", "Will BTC reach $70,000 by December 31?", 200, event="e"),
    ], max_per_event=0)
    assert len(c) == 1
    assert c[0].lockup_days == pytest.approx(200.0, abs=0.1)


def test_among_equal_shapes_the_sooner_pair_comes_first():
    c = build_candidates([
        _mkt("a1", "Will BTC reach $80,000 by December 31?", 200, event="btc"),
        _mkt("a2", "Will BTC reach $70,000 by December 31?", 200, event="btc"),
        _mkt("b1", "Will ETH reach $5,000 by September 30?", 5, event="eth"),
        _mkt("b2", "Will ETH reach $4,000 by September 30?", 5, event="eth"),
    ], max_per_event=0)
    ladders = [x for x in c if x.shape == "ladder"]
    assert len(ladders) == 2
    assert ladders[0].lockup_days < ladders[1].lockup_days


def test_speed_never_lifts_a_pair_past_a_better_shape():
    """
    A nest resolving in 300 days still outranks a ladder resolving tomorrow:
    shape decides whether there is an implication to find at all.
    """
    c = build_candidates([
        _mkt("n1", "Will the Lakers win by 5+ points?", 300, event="lal"),
        _mkt("n2", "Will the Lakers win?", 300, event="lal"),
        _mkt("l1", "Will BTC reach $80,000 by December 31?", 1, event="btc"),
        _mkt("l2", "Will BTC reach $70,000 by December 31?", 1, event="btc"),
    ], max_per_event=0)
    assert c[0].shape == "nest"


def test_oversized_weights_cannot_break_shape_dominance(monkeypatch):
    """A misconfiguration must not let speed outrank shape."""
    monkeypatch.setattr(im, "_TIME_WEIGHT", 5.0)
    monkeypatch.setattr(im, "_EVENT_WEIGHT", 5.0)
    c = build_candidates([
        _mkt("n1", "Will the Lakers win by 5+ points?", 300, event="lal"),
        _mkt("n2", "Will the Lakers win?", 300, event="lal"),
        _mkt("l1", "Will BTC reach $80,000 by December 31?", 1, event="btc"),
        _mkt("l2", "Will BTC reach $70,000 by December 31?", 1, event="btc"),
    ], max_per_event=0)
    assert c[0].shape == "nest"


def test_a_dated_pair_outranks_an_undated_one_of_the_same_shape():
    c = build_candidates([
        _mkt("a1", "Will BTC reach $80,000 by December 31?", 100, event="btc"),
        _mkt("a2", "Will BTC reach $70,000 by December 31?", 100, event="btc"),
        _mkt("b1", "Will ETH reach $5,000 by September 30?", None, event="eth"),
        _mkt("b2", "Will ETH reach $4,000 by September 30?", None, event="eth"),
    ], max_per_event=0)
    ladders = [x for x in c if x.shape == "ladder"]
    assert ladders[0].lockup_days is not None
    assert ladders[1].lockup_days is None


def test_the_optional_cap_drops_slow_pairs(monkeypatch):
    monkeypatch.setattr(im, "_MAX_LOCKUP_DAYS", 30.0)
    c = build_candidates([
        _mkt("a1", "Will BTC reach $80,000 by December 31?", 200, event="btc"),
        _mkt("a2", "Will BTC reach $70,000 by December 31?", 200, event="btc"),
        _mkt("b1", "Will ETH reach $5,000 by September 30?", 5, event="eth"),
        _mkt("b2", "Will ETH reach $4,000 by September 30?", 5, event="eth"),
    ], max_per_event=0)
    assert all(x.lockup_days is None or x.lockup_days <= 30.0 for x in c)
    assert any(x.lockup_days is not None and x.lockup_days < 10 for x in c)


def test_no_cap_by_default():
    assert im._MAX_LOCKUP_DAYS == 0.0


def test_under_a_cap_an_unknown_end_date_is_dropped(monkeypatch):
    """
    A pair whose lockup cannot be priced cannot be shown to fit the window, and
    execution will refuse it anyway — so it should not spend model budget.
    """
    monkeypatch.setattr(im, "_MAX_LOCKUP_DAYS", 7.0)
    c = build_candidates([
        _mkt("a1", "Will BTC reach $80,000 by December 31?", 3, event="btc"),
        _mkt("a2", "Will BTC reach $70,000 by December 31?", 3, event="btc"),
        _mkt("b1", "Will ETH reach $5,000 by September 30?", None, event="eth"),
        _mkt("b2", "Will ETH reach $4,000 by September 30?", None, event="eth"),
    ], max_per_event=0)
    assert c and all(x.lockup_days is not None and x.lockup_days <= 7.0 for x in c)
