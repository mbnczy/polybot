"""
LLM implication mapper — everything testable without spending money.

The API call itself is not mocked into a fake "it works" test; what is worth
guarding is the logic around it: which pairs reach the model, how a verdict
becomes a registry entry, and the caps that stop a confident-but-wrong model
from being trusted outright.
"""

from __future__ import annotations

import pytest

from strategy.implication_mapper import (
    Candidate,
    _MODEL_CONFIDENCE_CEILING,
    _to_implication,
    build_candidates,
    estimate_cost,
)


def _mkt(cid, q, event=None):
    d = {"conditionId": cid, "question": q}
    if event:
        d["eventId"] = event
    return d


# ── prefilter: keep the bill proportional to real candidates ──────────────────

def test_unrelated_markets_never_reach_the_model():
    markets = [
        _mkt("0xa", "Will it rain in Paris tomorrow?"),
        _mkt("0xb", "Will the Lakers win the championship?"),
    ]
    assert build_candidates(markets) == []


def test_related_markets_become_candidates():
    markets = [
        _mkt("0xa", "Will the Lakers win the game?"),
        _mkt("0xb", "Will the Lakers win the game by 5 points?"),
    ]
    cands = build_candidates(markets)
    assert len(cands) == 1
    assert cands[0].overlap > 0.5


def test_shared_event_lowers_the_overlap_bar():
    """An explicit event link is stronger evidence than wording alone."""
    markets = [
        _mkt("0xa", "Lakers victory margin above five", event="game-42"),
        _mkt("0xb", "Will the Lakers win?",             event="game-42"),
    ]
    assert len(build_candidates(markets)) == 1


def test_stopwords_alone_do_not_relate_two_markets():
    """'Will the ... in the ...' must not be enough to pair anything."""
    markets = [
        _mkt("0xa", "Will the price of tea in China rise?"),
        _mkt("0xb", "Will the winner of the race be disqualified?"),
    ]
    assert build_candidates(markets) == []


def test_candidate_cap_is_enforced():
    """Cost must stay bounded regardless of universe size."""
    markets = [_mkt(f"0x{i}", f"Will team alpha beta win match {i}?") for i in range(60)]
    cands = build_candidates(markets, max_pairs=25)
    assert len(cands) == 25


def test_candidates_are_ordered_best_first():
    """Under a cap, the strongest evidence must survive."""
    markets = [
        _mkt("0xa", "Will the Lakers win?",                  event="g1"),
        _mkt("0xb", "Will the Lakers win by 5 points?",      event="g1"),
        _mkt("0xc", "Will the Lakers win the championship?"),
    ]
    cands = build_candidates(markets, max_pairs=1)
    assert cands[0].same_event is True


# ── verdict -> registry entry ─────────────────────────────────────────────────

def _cand():
    return Candidate("0xnarrow", "0xbroad", "A wins by 5+", "A wins", 0.9, True)


def test_none_verdict_produces_no_relation():
    assert _to_implication(_cand(), "NONE", 0.99, "unrelated") is None


def test_a_implies_b_maps_the_right_way_round():
    rel = _to_implication(_cand(), "A_IMPLIES_B", 0.95, "margin is a subset")
    assert rel is not None
    assert rel.narrow == "0xnarrow" and rel.broad == "0xbroad"


def test_b_implies_a_reverses_the_direction():
    rel = _to_implication(_cand(), "B_IMPLIES_A", 0.95, "reversed")
    assert rel is not None
    assert rel.narrow == "0xbroad" and rel.broad == "0xnarrow"


def test_model_confidence_is_capped_below_certainty():
    """
    A model saying 1.0 must still rank below a hand-verified relation, so an
    inferred mapping can never be treated as ground truth.
    """
    rel = _to_implication(_cand(), "A_IMPLIES_B", 1.0, "certain")
    assert rel is not None
    assert rel.confidence == pytest.approx(_MODEL_CONFIDENCE_CEILING)
    assert rel.confidence < 1.0


def test_reasoning_is_retained_for_audit():
    """A bad mapping must be traceable to the reason the model gave."""
    rel = _to_implication(_cand(), "A_IMPLIES_B", 0.95, "winning by 5 implies winning")
    assert "winning by 5 implies winning" in rel.evidence
    assert "claude" in rel.evidence.lower()


def test_self_referential_verdict_is_rejected():
    bad = Candidate("0xsame", "0xsame", "A", "A", 1.0, True)
    assert _to_implication(bad, "A_IMPLIES_B", 0.95, "same market") is None


# ── cost ──────────────────────────────────────────────────────────────────────

def test_cost_estimate_scales_and_names_the_model():
    small = estimate_cost(10)
    large = estimate_cost(400)
    assert large["total_usd"] > small["total_usd"]
    assert small["model"] == "claude-opus-5"
    # a full pass over the live universe must stay under a dollar
    assert large["total_usd"] < 1.0
