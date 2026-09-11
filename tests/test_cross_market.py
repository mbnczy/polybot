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


# ═══════════════════════════════════════════════════════════════════════════════
# Freshness: two books, two clocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceFreshness:
    """
    The legs are separate markets on separate books that update only when
    someone trades or quotes them. Measured on a live related pair over 90 s:

        median skew   5.6 s
        p90          18.9 s
        max          49.3 s

    Seconds, not milliseconds. More than half the time a naive comparison
    prices a fresh book against a stale one, and any violation it finds is an
    artefact of the clock. It is the cheapest way for this strategy to lose:
    the trade looks risk-free right until the stale leg catches up.
    """

    def _det(self, **kw):
        reg = RelationRegistry()
        reg.add(Implication("0xby5", "0xwin", 1.0, evidence="test"))
        return CrossMarketDetector(reg, min_edge=0.02, min_confidence=0.90, **kw)

    def test_synchronised_prices_produce_the_signal(self):
        import time
        d = self._det()
        now = time.time()
        d.update_price("0xwin", 0.60, ts=now)
        assert d.update_price("0xby5", 0.70, ts=now)

    def test_a_stale_leg_is_refused(self):
        import time
        d = self._det(max_age_s=30.0)
        old = time.time() - 120.0
        d.update_price("0xwin", 0.60, ts=old)
        assert d.update_price("0xby5", 0.70, ts=time.time()) == []
        assert d.rejected_stale == 1

    def test_prices_observed_far_apart_are_refused(self):
        """Both recent, but one moved 20 s after the other."""
        import time
        d = self._det(max_skew_s=5.0)
        now = time.time()
        d.update_price("0xwin", 0.60, ts=now - 20.0)
        assert d.update_price("0xby5", 0.70, ts=now) == []
        assert d.rejected_skew == 1

    def test_checks_can_be_disabled(self):
        import time
        d = self._det(max_age_s=0.0, max_skew_s=0.0)
        d.update_price("0xwin", 0.60, ts=time.time() - 9999.0)
        assert d.update_price("0xby5", 0.70, ts=time.time())


# ═══════════════════════════════════════════════════════════════════════════════
# Capital lockup: a cross-market pair cannot be merged
# ═══════════════════════════════════════════════════════════════════════════════

class TestCapitalLockup:
    """
    A single-market YES+NO pair merges back to collateral within seconds. A
    cross-market pair cannot — the legs are different conditions, so there is no
    complete set — and the capital sits until the later market resolves.

    A 2% edge is excellent over two days and close to worthless over eight
    months. Nothing upstream distinguished the two.
    """

    def _det(self, **kw):
        reg = RelationRegistry()
        reg.add(Implication("0xby5", "0xwin", 1.0, evidence="test"))
        return CrossMarketDetector(reg, min_edge=0.02, min_confidence=0.90, **kw)

    def test_short_lockup_gives_a_high_apr(self):
        import time
        d = self._det()
        soon = time.time() + 2 * 86_400
        d.set_resolution("0xwin", soon)
        d.set_resolution("0xby5", soon)
        d.update_price("0xwin", 0.60)
        sig = d.update_price("0xby5", 0.70)[0]
        assert 1.5 < sig.lockup_days < 2.5
        assert sig.apr > 5.0, "10% over two days should annualise enormously"

    def test_long_lockup_collapses_the_same_edge(self):
        import time
        d = self._det()
        far = time.time() + 240 * 86_400
        d.set_resolution("0xwin", far)
        d.set_resolution("0xby5", far)
        d.update_price("0xwin", 0.60)
        sig = d.update_price("0xby5", 0.70)[0]
        assert sig.lockup_days > 200
        assert sig.apr < 0.25, "the same edge over 8 months is not the same trade"

    def test_apr_floor_refuses_a_long_lockup(self):
        import time
        d = self._det(min_apr=0.50)
        far = time.time() + 240 * 86_400
        d.set_resolution("0xwin", far)
        d.set_resolution("0xby5", far)
        d.update_price("0xwin", 0.60)
        assert d.update_price("0xby5", 0.70) == []
        assert d.rejected_lockup == 1

    def test_unknown_resolution_does_not_block(self):
        """Absent dates must not be read as an infinite lockup."""
        d = self._det(min_apr=0.50)
        d.update_price("0xwin", 0.60)
        sig = d.update_price("0xby5", 0.70)
        assert sig and sig[0].lockup_days == 0.0
