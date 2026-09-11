"""
A threshold implication read backwards is the costliest error this strategy can
make, because it turns a "guaranteed" floor into a bet that can pay nothing.

The live reader's ten largest reported arbitrages were one such pair — "hit 35%"
treated as narrower than "hit 25%" on a FALLING approval rating. These tests pin
that it is caught, and — just as important — that the check does not throw out
a correct implication on a family that is merely mispriced, which is what a
prices-only version did to "FDV above $X" ladders.
"""

from __future__ import annotations

import json

import pytest

from strategy.cross_market import Implication
from strategy.ladder_direction import (
    CONSISTENT,
    CONTRADICTS,
    FALLING,
    RISING,
    SOURCE_PRICES,
    SOURCE_STATED,
    UNKNOWN,
    check_direction,
    direction_of,
    family_trend,
    filter_implications,
    numbers,
    skeleton,
    stated_direction,
)


def _m(cid, question, yes):
    return {"conditionId": cid, "question": question,
            "outcomePrices": json.dumps([str(yes), str(round(1 - yes, 4))])}


# The live Trump family, as measured. "Hit" does not commit to a direction.
TRUMP = [
    _m("t20", "Will Trump's approval rating hit 20% in 2026?", 0.035),
    _m("t25", "Will Trump's approval rating hit 25% in 2026?", 0.045),
    _m("t30", "Will Trump's approval rating hit 30% in 2026?", 0.050),
    _m("t35", "Will Trump's approval rating hit 35% in 2026?", 0.320),
]

# A "reach" ladder — also ambiguous in principle — with ONE genuine local
# inversion: 80k priced above 70k.
BTC = [
    _m("b60", "Will Bitcoin reach $60,000 by December 31, 2026?", 0.40),
    _m("b70", "Will Bitcoin reach $70,000 by December 31, 2026?", 0.15),
    _m("b80", "Will Bitcoin reach $80,000 by December 31, 2026?", 0.20),
    _m("b90", "Will Bitcoin reach $90,000 by December 31, 2026?", 0.05),
]

# An "above" family priced BACKWARDS across the board — the shape the live
# Metamask and Consensys ladders showed. Higher thresholds cost more.
ABOVE_INVERTED = [
    _m("a1", "Consensys IPO closing market cap above $1B?", 0.050),
    _m("a2", "Consensys IPO closing market cap above $2B?", 0.070),
    _m("a3", "Consensys IPO closing market cap above $3B?", 0.090),
]


# ── parsing ───────────────────────────────────────────────────────────────────

def test_the_year_is_a_number_too():
    """Why the threshold cannot be "the number in the title"."""
    assert numbers("Will Trump's approval rating hit 35% in 2026?") == [35.0, 2026.0]


def test_magnitudes_and_separators():
    assert numbers("FDV above $8M one day after launch") == [8e6]
    assert numbers("Will Bitcoin reach $65,000?") == [65000.0]


def test_ladder_members_share_a_skeleton():
    assert skeleton(TRUMP[0]["question"]) == skeleton(TRUMP[3]["question"])
    assert skeleton(TRUMP[0]["question"]) != skeleton(BTC[0]["question"])


# ── what the title says ───────────────────────────────────────────────────────

@pytest.mark.parametrize("title,expected", [
    ("Consensys IPO closing market cap above $1B?", RISING),
    ("Will Metamask FDV be over $700M?", RISING),
    ("At least 50 seats?", RISING),
    ("Will the 10-year Treasury yield dip below 3.5% before 2027?", FALLING),
    ("Will unemployment fall to 3%?", FALLING),
    ("Will Trump's approval rating hit 35% in 2026?", None),
    ("Will Bitcoin reach $80,000?", None),
])
def test_stated_direction(title, expected):
    assert stated_direction(title) == expected


# ── the price trend, for verbs that do not commit ─────────────────────────────

def test_the_live_trump_family_is_a_falling_ladder():
    """All six pairs agree: price rises with the threshold."""
    assert family_trend([(20, .035), (25, .045), (30, .050), (35, .320)]) == +6


def test_one_local_inversion_does_not_flip_a_rising_ladder():
    assert family_trend([(60, .40), (70, .15), (80, .20), (90, .05)]) < 0


# ── the check ─────────────────────────────────────────────────────────────────

def test_the_reported_trump_arbitrage_is_backwards():
    """"Hit" is silent, so the family's prices decide — and they say 25% is narrow."""
    assert direction_of("t35", "t25", TRUMP)[:2] == (FALLING, SOURCE_PRICES)
    assert check_direction("t35", "t25", TRUMP) == CONTRADICTS
    assert check_direction("t35", "t30", TRUMP) == CONTRADICTS


def test_the_correct_trump_direction_passes():
    assert check_direction("t25", "t35", TRUMP) == CONSISTENT


def test_a_genuine_local_mispricing_survives():
    """80k above 70k in a rising ladder is a real arbitrage and must be kept."""
    assert check_direction("b80", "b70", BTC) == CONSISTENT


def test_an_above_ladder_priced_backwards_is_not_called_reversed():
    """
    The failure the prices-only version had. The whole family costs MORE at
    higher thresholds, so prices alone would call it falling and throw out the
    model's correct "above $3B implies above $1B". The title says "above"; the
    title wins; the implication is kept — and this is where the edge is.
    """
    assert direction_of("a3", "a1", ABOVE_INVERTED)[:2] == (RISING, SOURCE_STATED)
    assert check_direction("a3", "a1", ABOVE_INVERTED) == CONSISTENT


def test_a_stated_direction_catches_a_reversal_even_on_a_pair():
    """Words need no family: a two-market "above" pair read backwards is caught."""
    pair = ABOVE_INVERTED[:2]
    assert check_direction("a1", "a2", pair) == CONTRADICTS


def test_an_ambiguous_verb_on_a_pair_cannot_decide():
    """Two "hit" markets: a mispricing and a reversal look alike."""
    assert check_direction("t35", "t25", [TRUMP[1], TRUMP[3]]) == UNKNOWN


def test_different_skeletons_are_not_this_checks_business():
    assert check_direction("t35", "b70", TRUMP + BTC) == UNKNOWN


def test_missing_prices_shrink_the_family():
    thin = [dict(m) for m in TRUMP]
    for m in thin[:2]:
        m["outcomePrices"] = "[]"
    assert check_direction("t35", "t30", thin) == UNKNOWN


def test_unknown_markets_are_unknown():
    assert check_direction("nope", "t25", TRUMP) == UNKNOWN


# ── the filter ────────────────────────────────────────────────────────────────

def test_only_provably_backwards_relations_are_removed():
    rels = [
        Implication("t35", "t25", 0.97),   # backwards — the live error
        Implication("t25", "t35", 0.97),   # correct
        Implication("b80", "b70", 0.97),   # a real arbitrage
        Implication("a3", "a1", 0.97),     # correct, on a mispriced family
        Implication("x1", "x2", 0.97),     # unknown markets — no evidence
    ]
    kept, rejected = filter_implications(rels, TRUMP + BTC + ABOVE_INVERTED)
    assert [(r.narrow, r.broad) for r in rejected] == [("t35", "t25")]
    assert len(kept) == 4
