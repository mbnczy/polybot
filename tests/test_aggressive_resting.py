"""
tests/test_aggressive_resting.py
────────────────────────────────
Standing where the selling flow arrives.

lead_book_bid returns the CHEAPEST price that leads the book, reasoning that
the extra ticks up to the ask "buy no position, only cost". That is true of
queue position and false of fill probability. On a 0.50/0.70 book a bid at 0.51
is hit only by a seller willing to take 0.51; a bid at 0.69 is hit by every
seller willing to take 0.69 or less. Same queue — alone at its own level either
way — and far more of the flow reaches it.

So a leg rests at the highest price where the bundle still clears: one that
lets us cross everything else at today's asks and keep the edge floor.

What this is NOT is a way around the queue. On a one-tick book the limit price
sits below the touch by construction — it has to be under the ask, and the only
price under the ask IS the best bid — so the quote would join a queue and is
refused. 79% of live markets are in that state and no amount of prediction
changes it. The tests pin both halves, because the second one is the reason
this is a smaller idea than it first looked.
"""

import pytest

from strategy.arbitrage import NegRiskArbDetector, lead_book_bid


def _det(**kw) -> NegRiskArbDetector:
    params = dict(desired_net_margin=0.0001, min_leg_shares=0.0,
                  min_relative_edge=0.005, max_legs=3, early_rest_p=0.99)
    params.update(kw)
    return NegRiskArbDetector(**params)


class TestItBidsHigherWhereThatStillPays:

    def test_a_wide_book_leg_is_quoted_far_above_the_cheapest_lead(self):
        d = _det()
        sig = d.evaluate_neg_risk(
            "g", ["A", "B", "C"], [0.66, 0.66, 0.66],
            maker_rebate=0.0, tick_size=0.01,
            no_best_bids=[0.50, 0.65, 0.65],
        )
        assert sig is not None
        wide = next(l for l in sig.legs if l.token_id == "A")
        cheapest = lead_book_bid(0.66, 0.50, 0.01)
        assert cheapest == pytest.approx(0.51)
        assert wide.no_bid > cheapest
        assert wide.no_bid < 0.66, "still a maker order, not a cross"

    def test_it_never_bids_below_what_it_would_have(self):
        """The raise is one-directional; it can only improve fill odds."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "g", ["A", "B", "C"], [0.66, 0.66, 0.66],
            maker_rebate=0.0, tick_size=0.01,
            no_best_bids=[0.50, 0.65, 0.65],
        )
        assert sig is not None
        for leg in sig.legs:
            assert leg.no_bid >= lead_book_bid(
                leg.no_ask, leg.book_bid, leg.tick or 0.01) - 1e-9


class TestTheTickGridStillWins:

    def test_a_one_tick_leg_is_left_alone(self):
        """
        The structural limit. Under the ask there is exactly one price — the
        best bid — so there is nowhere to rest that does not join a queue.

        An all-one-tick group is also refused outright by the completion filter,
        because crossing every leg costs more than the bundle pays. Both facts
        are the same fact: on these books the maker path does not exist.
        """
        d = _det()
        sig = d.evaluate_neg_risk(
            "g", ["A", "B", "C"], [0.66, 0.66, 0.66],
            maker_rebate=0.0, tick_size=0.01,
            no_best_bids=[0.65, 0.65, 0.65],
        )
        assert sig is None

    def test_a_one_tick_leg_beside_a_wide_one_is_not_raised(self):
        """The wide leg may be bid up; the one-tick legs cannot be."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "g", ["A", "B", "C"], [0.66, 0.66, 0.66],
            maker_rebate=0.0, tick_size=0.01,
            no_best_bids=[0.50, 0.65, 0.65],
        )
        assert sig is not None
        for leg in sig.legs:
            if leg.token_id != "A":
                assert leg.no_bid == pytest.approx(0.65)
                assert leg.leads is False

    def test_a_price_at_or_above_the_ask_is_never_used(self):
        """That is a cross, and crossing is the edge test's decision."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "g", ["A", "B", "C"], [0.60, 0.60, 0.60],
            maker_rebate=0.0, tick_size=0.01,
            no_best_bids=[0.30, 0.59, 0.59],
        )
        if sig is not None:
            for leg in sig.legs:
                assert leg.no_bid < leg.no_ask


class TestItStaysOffWithoutAMeasuredRate:

    def test_disabled_reproduces_the_cheapest_lead(self):
        d = _det(early_rest_p=None)
        sig = d.evaluate_neg_risk(
            "g", ["A", "B", "C"], [0.66, 0.66, 0.66],
            maker_rebate=0.0, tick_size=0.01,
            no_best_bids=[0.50, 0.65, 0.65],
        )
        assert sig is not None
        wide = next(l for l in sig.legs if l.token_id == "A")
        assert wide.no_bid == pytest.approx(lead_book_bid(0.66, 0.50, 0.01))

    def test_the_default_detector_has_it_off(self):
        """
        NEGRISK_EARLY_REST_P has no default, so a fresh detector must not be
        bidding more aggressively than it did yesterday.
        """
        d = NegRiskArbDetector(desired_net_margin=0.0001, min_leg_shares=0.0)
        assert d._early_rest is False
