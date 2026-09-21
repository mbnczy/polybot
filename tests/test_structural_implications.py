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


# ── spreads ───────────────────────────────────────────────────────────────────

def _sp(cid, team, line, first=None, ev="E1"):
    other = "FC Utrecht" if team != "FC Utrecht" else "Feyenoord Rotterdam"
    return {"conditionId": cid, "question": f"Spread: {team} (-{line})",
            "outcomes": json.dumps([first or team, other]), "events": [{"id": ev}]}


def _with_event(markets, ev="E1"):
    return [{**m, "events": [{"id": ev}]} for m in markets]


def test_a_spread_reads_only_with_its_team_first_and_an_event():
    assert si.parse_spread(_sp("s", "FC Utrecht", "1.5")).line == 1.5
    assert si.parse_spread(_sp("s", "FC Utrecht", "1.5", first="Feyenoord Rotterdam")) is None
    no_event = {k: v for k, v in _sp("s", "FC Utrecht", "1.5").items() if k != "events"}
    assert si.parse_spread(no_event) is None


def test_spread_links():
    ms = _with_event(_match()) + [
        _sp("s1", "FC Utrecht", "1.5"), _sp("s2", "FC Utrecht", "2.5"),
        _sp("t1", "Feyenoord Rotterdam", "1.5")]
    rels = {(r.narrow, r.broad) for r in si.links(ms) if "spread" in r.evidence}
    assert rels == {
        ("s2", "s1"),                # wins by 3 ⊆ wins by 2
        ("s1", "c1"),                # Utrecht by 2 ⇒ two goals at least; no team 1.5 line: total
        ("s2", "c2"),                # Utrecht by 3 ⇒ three goals
        ("t1", "c1"),                # the same for Feyenoord
    }


def test_a_spread_prefers_its_own_teams_line():
    ms = _with_event(_match() + [_m("u15", "FC Utrecht O/U 1.5")]) + [_sp("s1", "FC Utrecht", "1.5")]
    rels = {(r.narrow, r.broad) for r in si.links(ms) if "spread-goals" in r.evidence}
    assert rels == {("s1", "u15")}


def test_spreads_of_different_events_never_meet():
    ms = _with_event(_match(), ev="E1") + [_sp("s1", "FC Utrecht", "1.5", ev="E2")]
    assert not [r for r in si.links(ms) if "spread-goals" in r.evidence]


def test_the_rule_decides_spread_pairs_of_one_event():
    ms = _with_event(_match())
    a, b = _sp("s1", "FC Utrecht", "1.5"), _sp("t1", "Feyenoord Rotterdam", "1.5")
    assert si.decides(a, b) and si.decides(a, ms[0])
    assert not si.decides(a, _sp("x", "FC Utrecht", "1.5", ev="E2"))
    assert not si.decides(a, ms[7])                           # corners: the model's


# ── price ladders ─────────────────────────────────────────────────────────────

FRI = "2026-09-11T22:00:02Z"                 # the ladder is created the Friday before
MID = "2026-09-16T15:00:00Z"                 # Wednesday, in the middle of the session


def _p(cid, q, created=FRI, end="2026-09-18T20:00:00Z", first="Yes"):
    return {"conditionId": cid, "question": q, "createdAt": created, "endDate": end,
            "outcomes": json.dumps([first, "No" if first == "Yes" else "Yes"])}


def _hit(cid, d, x, created=FRI, tkr="AAPL"):
    return _p(cid, f"Will Apple ({tkr}) hit ({d}) ${x} Week of September 14 2026?", created)


def test_every_price_shape_reads():
    for q in ("Will Apple (AAPL) hit (HIGH) $360 Week of September 14 2026?",
              "Will Apple (AAPL) hit (LOW) $328 Week of September 14 2026?",
              "Will Apple (AAPL) finish week of September 14 above $300?",
              "Will NVIDIA (NVDA) close above $190 on September 18?",
              "S&P 500 (SPY) closes above $750 on September 18?",
              "Will the price of Bitcoin be above $82,000 on September 18?"):
        assert si.parse_strike(_p("c", q)) is not None, q


def test_a_strike_whose_first_outcome_is_not_yes_does_not_read():
    assert si.parse_strike(_p("c", "Will Apple (AAPL) finish week of September 14 above $300?",
                              first="No")) is None


def _rels(ms):
    return {(r.narrow, r.broad) for r in si.links(ms) if "price-ladder" in r.evidence}


def test_a_high_ladder_nests_upwards_and_a_low_one_downwards():
    ms = [_hit("h1", "HIGH", 350), _hit("h2", "HIGH", 360), _hit("l1", "LOW", 320), _hit("l2", "LOW", 310)]
    assert _rels(ms) == {("h2", "h1"), ("l2", "l1")}


def test_thousands_separators_read_as_one_number():
    ms = [_p(f"b{x}", f"Will the price of Bitcoin be above ${x:,} on September 18?")
          for x in (80000, 82000)]
    assert _rels(ms) == {("b82000", "b80000")}


def test_a_broader_strike_added_mid_week_is_not_linked():
    """$350 created on Wednesday cannot vouch for a $360 touch on Monday."""
    ms = [_hit("h1", "HIGH", 350, created=MID), _hit("h2", "HIGH", 360)]
    assert _rels(ms) == set()


def test_a_narrower_strike_added_mid_week_still_is():
    """$360 created on Wednesday: whenever it is hit, $350 already existed."""
    ms = [_hit("h1", "HIGH", 350), _hit("h2", "HIGH", 360, created=MID)]
    assert _rels(ms) == {("h2", "h1")}


def test_a_single_observation_ignores_creation():
    ms = [_p("f1", "Will Apple (AAPL) finish week of September 14 above $300?", created=MID),
          _p("f2", "Will Apple (AAPL) finish week of September 14 above $310?")]
    assert _rels(ms) == {("f2", "f1")}


def test_different_tickers_weeks_or_ends_never_pair():
    ms = [_hit("a", "HIGH", 350), _hit("b", "HIGH", 360, tkr="MSFT"),
          _p("c", "Will Apple (AAPL) hit (HIGH) $370 Week of September 21 2026?"),
          {**_hit("d", "HIGH", 380), "endDate": "2026-09-25T20:00:00Z"}]
    assert _rels(ms) == set()


def test_the_rule_decides_a_ladder_pair_but_not_an_unsafe_one():
    safe = (_hit("h1", "HIGH", 350), _hit("h2", "HIGH", 360))
    unsafe = (_hit("h1", "HIGH", 350, created=MID), _hit("h2", "HIGH", 360))
    assert si.decides(*safe) and si.decides(*reversed(safe))
    assert not si.decides(*unsafe)


# ── one match is one event ────────────────────────────────────────────────────

def test_two_games_of_a_series_never_link():
    """Twins–Angels on Friday and on Saturday: identical titles, different events."""
    title = "Minnesota Twins vs. Los Angeles Angels: O/U {}"
    fri = [{**_m(f"f{x}", f"O/U {x}"), "question": title.format(x), "events": [{"id": "FRI"}]}
           for x in ("7.5", "8.5")]
    sat = [{**_m(f"s{x}", f"O/U {x}"), "question": title.format(x), "events": [{"id": "SAT"}]}
           for x in ("7.5", "8.5")]
    rels = {(r.narrow, r.broad) for r in si.links(fri + sat)}
    assert rels == {("f8.5", "f7.5"), ("s8.5", "s7.5")}


def test_a_listed_twin_is_not_a_ladder_step():
    ms = [{**_m("a", "O/U 2.5"), "events": [{"id": "E"}]},
          {**_m("b", "O/U 2.5"), "events": [{"id": "E"}]}]
    assert si.links(ms) == []


def test_the_reader_drops_a_pair_from_two_different_games():
    import scripts.demo_cross_market as reader
    from strategy.cross_market import Implication
    fri = {"conditionId": "a", "question": "Spread: Chicago Cubs (-2.5)", "endDate": "2026-09-18T23:05:00Z"}
    sat = {"conditionId": "b", "question": "Spread: Chicago Cubs (-1.5)", "endDate": "2026-09-19T23:05:00Z"}
    same = {**sat, "conditionId": "c", "endDate": "2026-09-18T23:05:00Z"}
    rels = [Implication("a", "b", 0.97, "model"), Implication("a", "c", 0.97, "model")]
    kept = reader.drop_cross_fixture(rels, [fri, sat, same])
    assert [(r.narrow, r.broad) for r in kept] == [("a", "c")]


def test_the_fixture_guard_leaves_other_markets_alone():
    import scripts.demo_cross_market as reader
    a = {"question": "Will the price of XRP be above $1.90 on September 21?", "endDate": "2026-09-21T16:00:00Z"}
    b = {"question": "Will the price of XRP be above $1.80 on September 22?", "endDate": "2026-09-22T16:00:00Z"}
    assert reader.same_fixture(a, b)
