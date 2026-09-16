"""
Which outcome is YES, outside Yes/No markets.

On 2026-09-13 the model read "Cesena FC vs. US Cremonese: O/U 5.5" as resolving
YES on Under ("a 0-0 means Under 0.5 goals, so it resolves YES"), while the
first token — the one the export calls YES and the bot buys — is Over. All 18
WOULD ENTERs in two days were that mistake: a bet against a goalless draw
dressed as an arbitrage.
"""

from __future__ import annotations

import json

from strategy.implication_mapper import (
    Candidate,
    SYSTEM_PROMPT,
    _to_implication,
    _user_prompt,
    build_candidates,
)
from strategy.outcomes import contradicts_yes_side, is_yes_no, market_outcomes
from strategy.verdict_cache import VerdictCache

OU = ("Over", "Under")
YN = ("Yes", "No")

CESENA_REASONING = ("If the exact score is 0-0, then the total number of goals is 0, "
                    "which means the Under 0.5 goals bet resolves YES.")


def _cand(a_title, b_title, a_out=YN, b_out=OU):
    return Candidate(a_id="0xa", b_id="0xb", a_title=a_title, b_title=b_title,
                     overlap=0.8, same_event=True, event_id="e1",
                     a_outcomes=a_out, b_outcomes=b_out)


# ── reading the labels ────────────────────────────────────────────────────────

def test_labels_are_read_in_token_order():
    assert market_outcomes({"outcomes": '["Over", "Under"]'}) == OU
    assert market_outcomes({"outcomes": ["Yes", "No"]}) == YN
    assert market_outcomes({"outcomes": "not json"}) is None
    assert market_outcomes({}) is None


def test_only_a_provable_yes_first_market_is_yes_no():
    assert is_yes_no(YN)
    assert not is_yes_no(OU)
    assert not is_yes_no(("No", "Yes"))          # first token would be NO
    assert not is_yes_no(None)                   # unknown is not proven


# ── the prompt says which side is YES ─────────────────────────────────────────

def test_the_prompt_names_the_yes_outcome_of_an_over_under_market():
    prompt = _user_prompt(_cand("Exact Score: Cesena FC 0 - 0 US Cremonese?",
                                "Cesena FC vs. US Cremonese: O/U 5.5"))
    assert 'Market B resolves YES only if the outcome is "Over"' in prompt
    assert "Market A resolves" not in prompt     # a Yes/No market needs no note


def test_the_system_prompt_carries_the_rule_and_the_counterexample():
    assert "Judge the implication ONLY for that YES outcome" in SYSTEM_PROMPT
    assert "a 0-0 is Under" in SYSTEM_PROMPT


# ── and a verdict about the other side is refused ────────────────────────────

def test_the_cesena_verdict_is_refused():
    c = _cand("Exact Score: Cesena FC 0 - 0 US Cremonese?", "Cesena FC vs. US Cremonese: O/U 0.5")
    assert _to_implication(c, "A_IMPLIES_B", 0.97, CESENA_REASONING, "m") is None


def test_a_verdict_about_the_yes_side_stands():
    c = _cand("X vs. Y: O/U 8.5 Total Corners", "X vs. Y: O/U 7.5 Total Corners", a_out=OU)
    rel = _to_implication(c, "A_IMPLIES_B", 0.97,
                          "If there are over 8.5 corners there are over 7.5.", "m")
    assert rel is not None and (rel.narrow, rel.broad) == ("0xa", "0xb")


def test_yes_no_markets_are_not_second_guessed():
    """'No' is ordinary prose; the check only applies where labels are not Yes/No."""
    c = _cand("A wins by 5+?", "A wins?", a_out=YN, b_out=YN)
    assert contradicts_yes_side("No team can win by 5 without winning", YN) is None
    assert _to_implication(c, "A_IMPLIES_B", 0.97,
                           "No way to win by 5+ without winning.", "m") is not None


def test_a_team_pick_reasoned_about_the_other_team_is_refused():
    picks = ("Arsenal", "Chelsea")
    assert contradicts_yes_side("If Chelsea wins by 2, Chelsea wins.", picks) == "Chelsea"
    assert contradicts_yes_side("If Arsenal wins by 2, Arsenal wins.", picks) is None


# ── the labels travel with the candidate ─────────────────────────────────────

def test_candidates_carry_their_outcome_labels():
    markets = [
        {"conditionId": "0x1", "question": "Cesena FC vs. US Cremonese: O/U 5.5",
         "outcomes": json.dumps(list(OU))},
        {"conditionId": "0x2", "question": "Cesena FC vs. US Cremonese: O/U 4.5",
         "outcomes": json.dumps(list(OU))},
    ]
    cands = build_candidates(markets, max_pairs=10)
    assert cands, "the ladder pair should survive the prefilter"
    assert cands[0].a_outcomes == OU and cands[0].b_outcomes == OU


# ── verdicts given under the old, ambiguous rules are not reused ─────────────

def test_version_one_verdicts_are_discarded(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"verdicts": {"0xa|0xb": {"narrow": "0xa", "broad": "0xb",
                                                         "confidence": 0.97, "ts": 9e12}}}))
    cache = VerdictCache(path).load()
    assert len(cache) == 0
    cache.save()
    assert json.loads(path.read_text())["version"] == 3
