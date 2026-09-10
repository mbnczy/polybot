"""
tests/test_bundle_execution_split.py
────────────────────────────────────
Which legs to rest and which to buy outright.

The taker fee is rate x p x (1-p) x size, largest at p=0.5 and vanishing at the
extremes, so the value of resting instead of crossing is not the same on every
leg. And ticks do not compare: half the live universe trades on a 0.001 grid
where one tick is 0.1%, the other half on 0.01 where it is ten times that, so
anything that reasons about "wide spread" has to say wide in price.

The dangerous part is not the split, it is what the split does to risk.
Crossing a leg early does not remove execution risk, it MOVES it: that leg is
then certainly owned, so if the resting legs fail, the bundle is naked on what
we bought — the -0.20 USDC unwind we already pay for. It is only an improvement
when the legs left resting are ones we expect to fill, and the tests below pin
that condition rather than the happy path.
"""

import pytest

from strategy.arbitrage import (
    ArbLeg,
    maker_saving,
    plan_bundle_execution,
    spread_fraction,
)


def _leg(tid, ask, bid, reachable=True, size=10.0):
    return ArbLeg(token_id=tid, no_ask=ask, no_bid=bid, size=size,
                  reachable=reachable)


class TestSpreadIsMeasuredInPriceNotTicks:

    def test_one_tick_means_different_things_on_the_two_grids(self):
        fine = spread_fraction(0.500, 0.499)      # 0.001 grid
        coarse = spread_fraction(0.50, 0.49)      # 0.01 grid
        assert round(fine * 100, 2) == 0.20
        assert round(coarse * 100, 2) == 2.00
        assert coarse > fine * 9

    def test_a_wide_tick_count_can_still_be_a_thin_spread(self):
        """Ten ticks on the fine grid is 1%, less than one tick on the coarse."""
        assert spread_fraction(0.968, 0.958) < spread_fraction(0.50, 0.49)

    def test_a_one_sided_book_has_no_spread(self):
        assert spread_fraction(0.50, None) is None


class TestMakerSavingIsMostlyTheFee:

    def test_the_fee_dominates_the_spread_on_a_fine_grid(self):
        """
        One tick of spread at 0.001 is worth 0.001; not paying the taker fee at
        p=0.5 on a 5% market is worth ten times that. Measuring the maker path
        in ticks would miss almost all of its value.
        """
        saving = maker_saving(0.500, 0.499)
        assert saving > 0.010
        assert saving - 0.001 > 0.009          # the fee, not the spread

    def test_saving_collapses_at_the_extremes(self):
        """p(1-p) vanishes, so there is little fee left to save."""
        assert maker_saving(0.950, 0.949) < maker_saving(0.500, 0.499)


class TestTheSplit:

    def test_an_unreachable_leg_is_crossed(self):
        legs = [_leg("A", 0.50, 0.49, reachable=False),
                _leg("B", 0.30, 0.29, reachable=True)]
        rest, cross = plan_bundle_execution(legs, early_cross=True)
        assert [l.token_id for l in rest] == ["B"]
        assert [l.token_id for l in cross] == ["A"]

    def test_nothing_is_crossed_when_a_resting_leg_is_also_unreachable(self):
        """
        The safety rule. Crossing A while B cannot fill either buys a naked
        position more cheaply — the bundle rests intact instead, and the guard's
        completion path resolves it as it always has.
        """
        legs = [_leg("A", 0.50, 0.49, reachable=False),
                _leg("B", 0.30, 0.29, reachable=False)]
        rest, cross = plan_bundle_execution(legs, early_cross=True)
        assert len(rest) == 2 and cross == []

    def test_a_leg_saving_too_little_is_crossed(self):
        """
        Resting has to be worth its non-fill risk. A leg near the extreme on a
        fine grid saves ~0.3% by resting, which does not buy much.
        """
        legs = [_leg("A", 0.950, 0.949, reachable=True),
                _leg("B", 0.300, 0.290, reachable=True)]
        rest, cross = plan_bundle_execution(
            legs, early_cross=True, min_saving_frac=0.004)
        assert [l.token_id for l in cross] == ["A"]

    def test_a_bundle_that_can_rest_entirely_is_left_alone(self):
        legs = [_leg("A", 0.50, 0.49), _leg("B", 0.30, 0.29)]
        rest, cross = plan_bundle_execution(legs, early_cross=True)
        assert len(rest) == 2 and cross == []

    def test_a_bundle_with_nothing_worth_resting_is_left_alone(self):
        """
        Every leg crossed is a full taker play, and whether that is worth doing
        is the edge test's decision, not this split's.
        """
        legs = [_leg("A", 0.50, 0.49, reachable=False),
                _leg("B", 0.30, 0.29, reachable=False)]
        rest, cross = plan_bundle_execution(legs, early_cross=True)
        assert cross == []

    def test_it_is_off_by_default(self):
        """
        Production behaviour must not change on the strength of an argument.
        The fill log has to show the reachability call is accurate first.
        """
        legs = [_leg("A", 0.50, 0.49, reachable=False),
                _leg("B", 0.30, 0.29, reachable=True)]
        rest, cross = plan_bundle_execution(legs)
        assert len(rest) == 2 and cross == []

    def test_an_empty_bundle_is_handled(self):
        assert plan_bundle_execution([], early_cross=True) == ([], [])
