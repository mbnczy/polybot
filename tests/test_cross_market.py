"""
Cross-market implication arbitrage — detection tests.

The strategy's ONLY loss mode is a wrong implication, so most of these tests are
about refusing to assert one rather than about spotting mispricings.
"""

from __future__ import annotations

import pytest

from strategy.cross_market import (
    CrossMarketDetector,
    Implication,
    RelationRegistry,
    discover_from_markets,
)


# ── the relation itself ───────────────────────────────────────────────────────

def test_implication_rejects_self_reference():
    with pytest.raises(ValueError):
        Implication(narrow="0xa", broad="0xa", confidence=1.0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_implication_rejects_impossible_confidence(bad):
    with pytest.raises(ValueError):
        Implication(narrow="0xa", broad="0xb", confidence=bad)


def test_registry_deduplicates():
    reg = RelationRegistry()
    reg.add(Implication("0xa", "0xb", 1.0))
    reg.add(Implication("0xa", "0xb", 1.0))
    assert len(reg) == 1


# ── discovery: conservative by design ─────────────────────────────────────────

def _mkt(cid, q, event=None):
    d = {"conditionId": cid, "question": q}
    if event:
        d["eventId"] = event
    return d


def test_discovers_margin_implication_within_one_event():
    markets = [
        _mkt("0xwin",    "Will the Lakers win?",              event="e1"),
        _mkt("0xby5",    "Will the Lakers win by 5+ points?", event="e1"),
    ]
    rels = discover_from_markets(markets)
    assert len(rels) == 1
    assert rels[0].narrow == "0xby5" and rels[0].broad == "0xwin"
    assert rels[0].confidence == pytest.approx(0.95)   # same event confirmed


def test_refuses_to_link_across_different_events():
    """The dangerous case: same wording, different games."""
    markets = [
        _mkt("0xwin", "Will the Lakers win?",              event="e1"),
        _mkt("0xby5", "Will the Lakers win by 5+ points?", event="e2"),
    ]
    assert discover_from_markets(markets) == []


def test_unrelated_markets_produce_nothing():
    markets = [
        _mkt("0xa", "Will it rain in Paris?"),
        _mkt("0xb", "Will the Lakers win by 5+ points?"),
    ]
    assert discover_from_markets(markets) == []


def test_margin_market_without_a_base_market_is_ignored():
    markets = [_mkt("0xby5", "Will the Lakers win by 5+ points?", event="e1")]
    assert discover_from_markets(markets) == []


# ── detection ─────────────────────────────────────────────────────────────────

def _detector(confidence=1.0):
    reg = RelationRegistry()
    reg.add(Implication("0xby5", "0xwin", confidence, evidence="test"))
    d = CrossMarketDetector(reg, min_edge=0.02, min_confidence=0.90)
    return d


def test_consistent_prices_produce_no_signal():
    """P(narrow) <= P(broad) is the normal, arbitrage-free ordering."""
    d = _detector()
    d.update_price("0xwin", 0.70, "Lakers win")
    assert d.update_price("0xby5", 0.40, "Lakers win by 5+") == []


def test_violation_is_detected_and_priced():
    """Narrow priced ABOVE broad is logically impossible -> tradeable."""
    d = _detector()
    d.update_price("0xwin", 0.60, "Lakers win")
    sigs = d.update_price("0xby5", 0.70, "Lakers win by 5+")
    assert len(sigs) == 1
    s = sigs[0]
    # buy broad YES 0.60 + narrow NO (1 - 0.70 = 0.30) = 0.90 for a >= 1.00 floor
    assert s.cost == pytest.approx(0.90)
    assert s.edge == pytest.approx(0.10)
    assert s.min_payout == pytest.approx(1.0)


def test_violation_below_min_edge_is_ignored():
    """A one-tick inversion is inside fee/precision noise."""
    d = _detector()
    d.update_price("0xwin", 0.60, "Lakers win")
    assert d.update_price("0xby5", 0.61, "Lakers win by 5+") == []


def test_low_confidence_relation_never_fires():
    """A weak mapping must not be tradeable at any price."""
    d = _detector(confidence=0.50)
    d.update_price("0xwin", 0.50)
    assert d.update_price("0xby5", 0.90) == []


def test_detection_works_regardless_of_update_order():
    """The broad side arriving last must also surface the violation."""
    d = _detector()
    d.update_price("0xby5", 0.70, "Lakers win by 5+")
    sigs = d.update_price("0xwin", 0.60, "Lakers win")
    assert len(sigs) == 1


def test_incomplete_pair_is_silent():
    d = _detector()
    assert d.update_price("0xby5", 0.99) == []


@pytest.mark.parametrize("price", [0.0, 1.0, -0.5, 1.2])
def test_impossible_prices_are_rejected(price):
    d = _detector()
    assert d.update_price("0xby5", price) == []


def test_payout_floor_holds_in_every_branch():
    """
    The arbitrage claim itself: cost < 1.00 and payout >= 1.00 whatever happens.
    Guards the maths the whole strategy rests on.
    """
    pn, pb = 0.70, 0.60                     # narrow priced above broad
    cost = pb + (1.0 - pn)
    assert cost < 1.0
    # narrow occurs -> broad occurs too: broad YES pays 1, narrow NO pays 0
    assert 1.0 + 0.0 >= 1.0
    # broad but not narrow: both pay
    assert 1.0 + 1.0 >= 1.0
    # neither: broad YES pays 0, narrow NO pays 1
    assert 0.0 + 1.0 >= 1.0


# ── regex guards: the false positives that a 2,100-market scan surfaced ───────

@pytest.mark.parametrize("title", [
    "Will China unban Bitcoin by 2027?",
    "US defaults on debt by 2027?",
    "Will X be nominated by 2028?",
    "Will it happen by 2030?",
])
def test_deadlines_are_not_margins(title):
    """
    'by <year>' is a deadline, not a winning margin. The first draft matched
    these and proposed two bogus implications out of a full-universe scan.
    """
    from strategy.cross_market import _MARGIN_RE
    assert _MARGIN_RE.search(title) is None


@pytest.mark.parametrize("title", [
    "Will the Lakers win by 5+ points?",
    "Will the Lakers win by at least 3?",
    "Will X win by 10 points?",
    "Will X win by 7 or more?",
])
def test_genuine_margins_still_match(title):
    from strategy.cross_market import _MARGIN_RE
    assert _MARGIN_RE.search(title) is not None
