"""
tests/test_negrisk_subset_selection.py
──────────────────────────────────────
Which k of the N outcomes to buy is a choice, and ranking them by implied
probability ignores what they cost to complete.

The taker fee is rate x p x (1-p) x size, so its burden per leg follows a bell
peaking at p=0.5 and vanishing at the extremes. Over a bundle that is
rate x SUM yes_i(1 - yes_i), which moves the break-even from SUM yes > 1.0071 on
an extreme-priced group to > 1.0394 on a flat one — a factor of five in how big
a Dutch book has to be before it pays.

The objective is a RATIO, and getting that wrong collapses the whole search.
Scoring absolute edge per bundle always picks the largest allowed bundle:
adding any leg raises the payout by exactly 1 and the cost by less than 1, so
"more legs" wins unconditionally and the choice is not a choice. Capital is what
binds — n_bundles is max_position_usdc over the combined bid — so the score is
edge per USDC committed.
"""

import itertools

import pytest

from strategy.arbitrage import NegRiskArbDetector, effective_taker_fee


def _det(**kw) -> NegRiskArbDetector:
    params = dict(desired_net_margin=0.0001, min_leg_shares=0.0,
                  min_relative_edge=0.0, max_legs=4)
    params.update(kw)
    return NegRiskArbDetector(**params)


def _cand(tid, ask, bid=None, tick=0.01):
    """A candidate tuple in the detector's internal shape."""
    return (tid, ask, 1.0 - ask, float("inf"), bid, tick, None)


class TestObjectiveIsReturnNotEdge:

    def test_a_cheaper_subset_can_beat_a_bigger_one(self):
        """
        Live-shaped group at a 5% taker rate. Four legs earn more per bundle;
        two legs earn more per dollar, and the wallet is what is scarce.
        """
        d = _det()
        big = [_cand("A", 0.06, 0.05), _cand("B", 0.50, 0.49),
               _cand("C", 0.52, 0.51), _cand("D", 0.90, 0.89)]
        small = big[:2]
        assert d._subset_return(small, 0.01) > d._subset_return(big, 0.01)

    def test_absolute_edge_would_have_picked_the_bigger_one(self):
        """
        The trap this replaced. Guards the reasoning, not just the result: if
        someone switches the score back to a difference, this fails.
        """
        def abs_edge(combo):
            payout = len(combo) - 1
            return payout - sum(
                a * (1.0 + effective_taker_fee(a)) for _, a, *_ in combo)

        big = [_cand("A", 0.06, 0.05), _cand("B", 0.50, 0.49),
               _cand("C", 0.52, 0.51), _cand("D", 0.90, 0.89)]
        assert abs_edge(big) > abs_edge(big[:2])

    def test_adding_any_leg_always_raises_absolute_edge(self):
        """
        Why the ratio is necessary rather than merely nicer: a leg costs less
        than 1 and pays 1, at every price the exchange allows.
        """
        for ask in (0.02, 0.25, 0.50, 0.75, 0.98):
            assert ask * (1.0 + effective_taker_fee(ask)) < 1.0


class TestSelectionEndToEnd:

    def test_the_search_beats_the_probability_ranking_on_return(self):
        asks = [0.06, 0.50, 0.52, 0.90, 0.93]
        bids = [round(a - 0.01, 2) for a in asks]
        ids = [f"L{i}" for i in range(len(asks))]

        chosen = _det().evaluate_neg_risk(
            "g", ids, asks, maker_rebate=0.0, tick_size=0.01,
            no_best_bids=bids,
        )
        ranked = _det(subset_search=False).evaluate_neg_risk(
            "g", ids, asks, maker_rebate=0.0, tick_size=0.01,
            no_best_bids=bids,
        )
        assert chosen is not None and ranked is not None

        def ret(sig):
            cost = sum(l.no_ask * (1.0 + effective_taker_fee(l.no_ask))
                       for l in sig.legs)
            spend = sum(l.no_bid for l in sig.legs)
            return (len(sig.legs) - 1 - cost) / spend

        assert ret(chosen) > ret(ranked)
        assert len(chosen.legs) < len(ranked.legs), "it wins by buying less"

    def test_the_search_never_exceeds_the_leg_cap(self):
        d = _det(max_legs=3)
        sig = d.evaluate_neg_risk(
            "g", [f"L{i}" for i in range(6)],
            [0.60, 0.62, 0.64, 0.66, 0.68, 0.70],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is not None
        assert len(sig.legs) <= 3

    def test_a_bundle_always_has_at_least_two_legs(self):
        """One leg guarantees nothing — there is no k-1 payout to collect."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "g", ["A", "B", "C"], [0.30, 0.33, 0.36],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is None or len(sig.legs) >= 2

    def test_it_can_be_switched_off_for_comparison(self):
        """
        The old ranking has to stay reachable: a live A/B is the only thing that
        settles whether the search is actually better, and an argument is not.
        """
        d = _det(subset_search=False)
        assert d._subset_search is False
        sig = d.evaluate_neg_risk(
            "g", ["p10", "p32", "p05", "p22", "p26", "p40"],
            [0.90, 0.68, 0.95, 0.78, 0.74, 0.60],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is not None
        assert [l.token_id for l in sig.legs] == ["p40", "p32", "p26", "p22"]
