"""Implications between a match's over/unders, from the titles alone."""

from __future__ import annotations

import json

import strategy.structural_implications as si

EV = "Feyenoord Rotterdam vs. FC Utrecht"


def _m(cid, tail, outcomes=("Over", "Under")):
    return {"conditionId": cid, "question": f"{EV}: {tail}",
            "outcomes": json.dumps(list(outcomes))}


# ── reading a title ───────────────────────────────────────────────────────────

def test_every_goal_family_reads():
    cases = {
        "O/U 2.5": (si.TOTAL, si.FULL, 2.5),
        "1st Half O/U 0.5": (si.TOTAL, "1st Half", 0.5),
        "2nd Half O/U 1.5": (si.TOTAL, "2nd Half", 1.5),
        "Feyenoord Rotterdam O/U 1.5": ("Feyenoord Rotterdam", si.FULL, 1.5),
        "FC Utrecht 1st Half O/U 0.5": ("FC Utrecht", "1st Half", 0.5),
    }
    for tail, want in cases.items():
        ou = si.parse(_m("c", tail))
        assert ou is not None and (ou.scope, ou.period, ou.line) == want, tail


def test_a_half_is_not_read_as_a_team_called_1st_half():
    ou = si.parse(_m("c", "1st Half O/U 0.5"))
    assert ou.scope == si.TOTAL and ou.period == "1st Half"


def test_other_statistics_and_strangers_do_not_read():
    for tail in ("Corners O/U 9.5", "Cards O/U 4.5", "Santiago Giménez O/U 0.5",
                 "Games Total: O/U 2.5", "Both Teams to Score"):
        assert si.parse(_m("c", tail)) is None, tail


def test_an_integer_line_that_can_refund_does_not_read():
    assert si.parse(_m("c", "O/U 2")) is None


def test_a_market_whose_first_token_is_under_does_not_read():
    """The YES token is the first; the rule reasons about Over."""
    assert si.parse(_m("c", "O/U 2.5", outcomes=("Under", "Over"))) is None


# ── what implies what ─────────────────────────────────────────────────────────

def test_the_rule():
    half = si.parse(_m("h", "1st Half O/U 1.5"))
    full = si.parse(_m("f", "O/U 1.5"))
    team = si.parse(_m("t", "FC Utrecht O/U 1.5"))
    other = si.parse(_m("o", "Feyenoord Rotterdam O/U 1.5"))
    low = si.parse(_m("l", "O/U 0.5"))
    assert si.implies(half, full) and not si.implies(full, half)
    assert si.implies(team, full) and not si.implies(full, team)
    assert not si.implies(team, other) and not si.implies(other, team)
    assert si.implies(full, low) and not si.implies(low, full)
    assert si.implies(half, low)                    # a half over 1.5 ⇒ a goal at all


def test_two_matches_never_pair():
    a = si.parse(_m("a", "O/U 1.5"))
    b = si.parse({"conditionId": "b", "question": "Ajax vs. PSV: O/U 0.5",
                  "outcomes": '["Over", "Under"]'})
    assert not si.dominated(a, b) and not si.implies(a, b)


# ── the links of one match ────────────────────────────────────────────────────

def _match():
    tails = ["O/U 0.5", "O/U 1.5", "O/U 2.5", "1st Half O/U 0.5", "1st Half O/U 1.5",
             "FC Utrecht O/U 0.5", "FC Utrecht 1st Half O/U 0.5", "Corners O/U 9.5"]
    return [_m(f"c{i}", t) for i, t in enumerate(tails)]


def test_the_links_of_a_match():
    rels = {(r.narrow, r.broad) for r in si.links(_match())}
    assert rels == {
        ("c1", "c0"), ("c2", "c1"),          # match ladder
        ("c4", "c3"),                        # first-half ladder
        ("c3", "c0"), ("c4", "c1"),          # half inside match, same line
        ("c5", "c0"),                        # team inside match
        ("c6", "c5"), ("c6", "c3"), ("c6", "c0"),   # team half inside all three
    }


def test_every_link_is_an_implication_and_says_so():
    by_cid = {m["conditionId"]: si.parse(m) for m in _match()}
    for r in si.links(_match()):
        assert si.implies(by_cid[r.narrow], by_cid[r.broad])
        assert r.confidence == 1.0 and r.evidence.startswith("structural:")


def test_the_rule_decides_every_pair_of_one_matchs_over_unders():
    ms = _match()
    assert si.decides(ms[0], ms[5])                          # an implication
    assert si.decides(ms[5], _m("x", "Feyenoord Rotterdam O/U 0.5"))  # none, decided
    assert not si.decides(ms[0], ms[7])                      # corners: the model's


def test_the_reader_adds_the_rules_pairs_once():
    import scripts.demo_cross_market as reader
    from strategy.cross_market import Implication

    model = [Implication("c1", "c0", 0.97, "model")]      # the model found one too
    rels = reader.with_structural(model, _match())
    keys = [(r.narrow, r.broad) for r in rels]
    assert keys.count(("c1", "c0")) == 1 and rels[0].evidence == "model"
    assert len(rels) == 9
