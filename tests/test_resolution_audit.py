"""
Every exported implication is held to how its markets resolved.

On 2026-09-15 three "player advances ⊆ match completed" pairs resolved against
the trade within a day: each winner market split 50-50 (walkover), each
completed-match market resolved No. The trade paid 0.5 on a pair that cost more.
"""

from __future__ import annotations

import json

import scripts.demo_cross_market as reader
from strategy.cross_market import Implication
from strategy.resolution_audit import (
    ResolutionAudit, kind_of, resolved_prices, template_of, trade_payout,
)
from strategy.verdict_cache import VerdictCache

NOW = 1_800_000_000.0
TENNIS_N = "W50 Pazardzhik: Julia Adams vs Rositsa Dencheva"
TENNIS_B = "W50 Pazardzhik, Main Draw: Completed Match: Julia Adams vs Rositsa Dencheva"


def _closed(cid, prices):
    return {"conditionId": cid, "closed": True, "outcomePrices": json.dumps(prices)}


def _row(n, b, nt, bt, end=NOW - 10 * 3600):
    return {"narrow": n, "broad": b, "narrow_title": nt, "broad_title": bt,
            "narrow_end_ts": end, "broad_end_ts": end}


# ── reading a resolution ──────────────────────────────────────────────────────

def test_final_prices_are_read_and_splits_recognised():
    assert resolved_prices(_closed("x", ["1", "0"])) == (1.0, 0.0)
    assert resolved_prices(_closed("x", ["0.5", "0.5"])) == (0.5, 0.5)
    assert resolved_prices(_closed("x", ["0.9995", "0.0005"])) == (1.0, 0.0)
    assert resolved_prices(_closed("x", ["0.62", "0.38"])) is None       # closed, not settled
    assert resolved_prices({"closed": False, "outcomePrices": '["1","0"]'}) is None


def test_the_walkover_pair_paid_half():
    """NO on the winner market (split: 0.5) + YES on completed (No: 0)."""
    assert trade_payout((0.5, 0.5), (0.0, 1.0)) == 0.5


def test_templates_name_the_kinds_and_periods():
    assert template_of(TENNIS_N, TENNIS_B) == "head-to-head ⊆ completed-match"
    assert template_of("1st Half Spread: CA Osasuna (-2.5)", "Spread: CA Osasuna (-2.5)") \
        == "spread:half ⊆ spread"
    assert kind_of("FC Lahti vs. FF Jaro: FF Jaro O/U 0.5") == "over-under"


# ── the audit ─────────────────────────────────────────────────────────────────

def test_a_violation_is_found_and_its_template_blocked(tmp_path):
    audit = ResolutionAudit(tmp_path / "a.json")
    rows = [_row(f"0xn{i}", f"0xb{i}", TENNIS_N, TENNIS_B) for i in range(3)]
    audit.record(rows, now=NOW - 86_400)
    markets = []
    for i in range(3):
        markets += [_closed(f"0xn{i}", ["0.5", "0.5"]), _closed(f"0xb{i}", ["0", "1"])]
    summary = audit.run(lambda ids: markets, now=NOW)
    assert summary["checked"] == 3 and len(summary["violated"]) == 3
    assert summary["newly_blocked"] == ["head-to-head ⊆ completed-match"]
    assert audit.is_blocked("M15 Tsaghkadzor: A vs B", "M15 Tsaghkadzor, Main Draw: Completed Match: A vs B")


def test_a_rare_violation_among_many_held_blocks_nothing(tmp_path):
    """One exchange mis-resolution among a ladder's hundred held pairs."""
    audit = ResolutionAudit(tmp_path / "a.json")
    t_n, t_b = "X vs. Y: O/U 2.5", "X vs. Y: O/U 1.5"
    rows = [_row(f"0xn{i}", f"0xb{i}", t_n, t_b) for i in range(20)]
    audit.record(rows, now=NOW - 86_400)
    markets = []
    for i in range(20):
        bad = i < 1
        markets += [_closed(f"0xn{i}", ["1", "0"]), _closed(f"0xb{i}", ["0", "1"] if bad else ["1", "0"])]
    summary = audit.run(lambda ids: markets, now=NOW)
    assert len(summary["violated"]) == 1 and summary["held"] == 19
    assert audit.blocked_templates() == []


def test_an_unresolved_pair_waits_then_is_given_up(tmp_path):
    audit = ResolutionAudit(tmp_path / "a.json")
    audit.record([_row("0xn", "0xb", TENNIS_N, TENNIS_B)], now=NOW - 86_400)
    assert audit.run(lambda ids: [], now=NOW)["pending"] == 1
    assert audit.run(lambda ids: [], now=NOW + 20 * 86_400)["pending"] == 0


def test_a_pair_just_past_its_end_is_asked_about_but_not_judged_early(tmp_path):
    """Early checks look for a settled market; an unsettled one keeps waiting, never given up."""
    audit = ResolutionAudit(tmp_path / "a.json")
    audit.record([_row("0xn", "0xb", TENNIS_N, TENNIS_B, end=NOW - 600)], now=NOW - 3600)
    calls = []
    summary = audit.run(lambda ids: calls.append(ids) or [], now=NOW)
    assert calls == [["0xb", "0xn"]]
    assert summary["checked"] == 0 and summary["pending"] == 1


def test_it_survives_a_restart(tmp_path):
    path = tmp_path / "a.json"
    audit = ResolutionAudit(path)
    audit.record([_row(f"0xn{i}", f"0xb{i}", TENNIS_N, TENNIS_B) for i in range(2)], now=NOW - 86_400)
    audit.run(lambda ids: [_closed("0xn0", ["0.5", "0.5"]), _closed("0xb0", ["0", "1"]),
                           _closed("0xn1", ["0.5", "0.5"]), _closed("0xb1", ["0", "1"])], now=NOW)
    audit.save()
    again = ResolutionAudit(path).load()
    assert again.violated("0xn0", "0xb0") and again.is_blocked(TENNIS_N, TENNIS_B)


# ── what it changes ───────────────────────────────────────────────────────────

def test_a_violated_pair_is_forbidden_in_the_verdict_cache(tmp_path):
    cache = VerdictCache(tmp_path / "v.json")

    class _C:
        a_id, b_id = "0xn", "0xb"

    cache.remember([_C()], [Implication("0xn", "0xb", 0.97, "e")])
    cache.forbid("0xn", "0xb", "violated by its resolution")
    known, unknown = cache.split([_C()])
    assert known == [] and unknown == []


def test_the_export_leaves_out_blocked_templates(tmp_path):
    audit = ResolutionAudit(tmp_path / "a.json")
    audit.templates["head-to-head ⊆ completed-match"] = {"held": 0, "violated": 3, "blocked": True,
                                                         "examples": []}
    end = "2099-01-01T00:00:00Z"
    markets = [
        {"conditionId": "0xn", "question": TENNIS_N, "endDate": end, "clobTokenIds": '["1","2"]'},
        {"conditionId": "0xb", "question": TENNIS_B, "endDate": end, "clobTokenIds": '["3","4"]'},
        {"conditionId": "0xo", "question": "X vs. Y: O/U 2.5", "endDate": end, "clobTokenIds": '["5","6"]'},
        {"conditionId": "0xp", "question": "X vs. Y: O/U 1.5", "endDate": end, "clobTokenIds": '["7","8"]'},
    ]
    rels = [Implication("0xn", "0xb", 0.97, "e"), Implication("0xo", "0xp", 0.97, "e")]
    n = reader.export_implications(rels, markets, str(tmp_path / "imps.json"), audit)
    assert n == 1
    assert set(audit.pending) == {"0xo|0xp"}          # what was exported is recorded


def test_a_market_that_settles_before_its_end_date_is_audited(tmp_path):
    """Pazardzhik's walkover resolved on the 15th; the market's end date was the 22nd."""
    audit = ResolutionAudit(tmp_path / "a.json")
    rows = [_row(f"0xn{i}", f"0xb{i}", TENNIS_N, TENNIS_B, end=NOW + 6 * 86_400) for i in range(2)]
    audit.record(rows, now=NOW - 3600)
    markets = []
    for i in range(2):
        markets += [_closed(f"0xn{i}", ["0.5", "0.5"]), _closed(f"0xb{i}", ["0", "1"])]
    summary = audit.run(lambda ids: markets, now=NOW)
    assert len(summary["violated"]) == 2 and audit.is_blocked(TENNIS_N, TENNIS_B)


def test_an_open_market_before_its_end_date_just_waits(tmp_path):
    audit = ResolutionAudit(tmp_path / "a.json")
    audit.record([_row("0xn", "0xb", TENNIS_N, TENNIS_B, end=NOW + 6 * 86_400)], now=NOW - 3600)
    summary = audit.run(lambda ids: [], now=NOW)
    assert summary["checked"] == 0 and summary["pending"] == 1
