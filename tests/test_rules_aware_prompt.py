"""
The model sees each market's resolution rules and must look for a counterexample
before answering.

Every implication it got wrong at whole-window scale was right about the titles
and wrong about the rules: a player "advances" on a walkover while the
completed-match market resolves No. Three such pairs resolved against the trade.
"""

from __future__ import annotations

import json

import scripts.demo_cross_market as demo
from strategy.implication_mapper import (
    Candidate, SYSTEM_PROMPT, _RESULT_SCHEMA, _user_prompt, classify_one, verdict_problem,
)
from strategy.market_rules import rules_digest
from strategy.verdict_cache import VerdictCache

TENNIS_RULES = (
    'This market refers to the tennis match between Julia Adams and Rositsa Dencheva. '
    'This market will resolve to "Yes" if all games and sets required to determine a '
    'match winner are played to completion through normal play. Otherwise, if the match '
    'is not completed for any reason, it will resolve to "No." If a forfeit of any kind '
    'occurs, including but not limited to a walkover or retirement, this market will '
    'resolve "No." The primary resolution source is the official statistics.'
)
PLAYERS = ("Julia Adams", "Rositsa Dencheva")


def _cand(a_out=PLAYERS, b_out=("Yes", "No")):
    return Candidate(a_id="0xa", b_id="0xb", a_title="W50 Pazardzhik: Julia Adams vs Rositsa Dencheva",
                     b_title="W50 Pazardzhik, Main Draw: Completed Match: Julia Adams vs Rositsa Dencheva",
                     overlap=0.8, same_event=True, event_id="e", a_outcomes=a_out, b_outcomes=b_out)


# ── the rules reach the model ─────────────────────────────────────────────────

def test_the_forfeit_clause_survives_the_digest():
    """It follows a closing quote — "No." If a forfeit… — which once glued it to the
    sentence before, too long to keep."""
    assert "walkover or retirement" in rules_digest(TENNIS_RULES)


def test_the_prompt_carries_both_markets_rules():
    prompt = _user_prompt(_cand(), {"0xb": rules_digest(TENNIS_RULES)})
    assert "Rules: " in prompt and "walkover" in prompt


def test_the_system_prompt_asks_for_a_counterexample_first():
    assert "COUNTEREXAMPLE" in SYSTEM_PROMPT and "The\nrules decide, not the titles" in SYSTEM_PROMPT
    assert _RESULT_SCHEMA["required"][:4] == ["yes_a", "yes_b", "counterexample_a_to_b",
                                              "counterexample_b_to_a"]


# ── a verdict that contradicts itself is refused ─────────────────────────────

def test_an_implication_with_a_counterexample_for_itself_is_refused():
    v = {"yes_a": "Julia Adams", "yes_b": "Yes", "relation": "A_IMPLIES_B", "confidence": 0.97,
         "counterexample_a_to_b": "Adams advances on a walkover; the match is not completed",
         "counterexample_b_to_a": "Dencheva wins a completed match"}
    assert "counterexample" in verdict_problem(_cand(), v)


def test_a_counterexample_for_the_other_direction_is_expected():
    """Measured: the model answered B_IMPLIES_A for 'total O/U 0.5' vs 'team O/U 0.5'
    and, correctly, gave 'the other team scores' — the counterexample to A_IMPLIES_B."""
    v = {"yes_a": "Over", "yes_b": "Over", "relation": "B_IMPLIES_A", "confidence": 1,
         "counterexample_a_to_b": "FC Lahti scores 1 goal and FF Jaro scores 0 goals",
         "counterexample_b_to_a": "none"}
    assert verdict_problem(_cand(("Over", "Under"), ("Over", "Under")), v) is None


def test_the_no_outcome_named_as_yes_is_refused():
    v = {"yes_a": "Rositsa Dencheva", "yes_b": "Yes", "relation": "A_IMPLIES_B",
         "confidence": 0.97, "counterexample_a_to_b": "none"}
    assert "NO outcome" in verdict_problem(_cand(), v)


def test_a_paraphrased_yes_condition_stands():
    """Measured: 'FF Jaro score 1 or more goals' for an O/U market whose YES is Over."""
    v = {"yes_a": "FF Jaro score 1 or more goals in regular play", "yes_b": "Over",
         "relation": "A_IMPLIES_B", "confidence": 1, "counterexample_a_to_b": "none"}
    assert verdict_problem(_cand(("Over", "Under"), ("Over", "Under")), v) is None


def test_a_refusal_is_never_second_guessed():
    assert verdict_problem(_cand(), {"relation": "NONE",
                                     "counterexample_a_to_b": "a walkover"}) is None


def test_a_provider_that_omits_the_new_fields_is_not_penalised():
    assert verdict_problem(_cand(), {"relation": "A_IMPLIES_B", "confidence": 0.9}) is None


def test_classify_one_refuses_a_self_contradicting_reply():
    class _Msg:
        content = json.dumps({"yes_a": "Julia Adams", "yes_b": "Yes",
                              "counterexample_a_to_b": "a walkover",
                              "counterexample_b_to_a": "none", "relation": "A_IMPLIES_B",
                              "confidence": 0.97, "reasoning": "advancing needs a finished match"})

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return type("R", (), {"choices": [type("C", (), {"message": _Msg})()]})()

    from strategy.implication_mapper import PROVIDERS
    assert classify_one(_Client, PROVIDERS["anthropic"], "m", _cand()) is None


# ── the reader fetches rules only for what it classifies ─────────────────────

def test_rules_are_fetched_in_batches_and_kept(monkeypatch):
    calls = []

    def fake_get(params):
        ids = [v for k, v in params if k == "condition_ids"]
        calls.append(ids)
        return [{"conditionId": c, "description": TENNIS_RULES} for c in ids]

    monkeypatch.setattr(demo, "_gamma_get", fake_get)
    monkeypatch.setattr(demo, "_RULES", {})
    ids = [f"0x{i:03x}" for i in range(120)]
    got = demo.fetch_rules(ids, concurrency=2)
    assert len(calls) == 3 and all(len(c) <= 50 for c in calls)
    assert all("walkover" in got[c] for c in ids)
    demo.fetch_rules(ids[:10])
    assert len(calls) == 3                                     # kept between passes


# ── old verdicts ──────────────────────────────────────────────────────────────

def test_version_two_refusals_are_kept_and_implications_rejudged(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"version": 2, "verdicts": {
        "0xa|0xb": {"narrow": "0xa", "broad": "0xb", "confidence": 0.97, "ts": 9e12},
        "0xc|0xd": {"narrow": None, "ts": 9e12},
    }}))
    cache = VerdictCache(path).load()

    class _C:
        def __init__(self, a, b): self.a_id, self.b_id = a, b

    known, unknown = cache.split([_C("0xa", "0xb"), _C("0xc", "0xd")])
    assert known == [] and [(u.a_id, u.b_id) for u in unknown] == [("0xa", "0xb")]
