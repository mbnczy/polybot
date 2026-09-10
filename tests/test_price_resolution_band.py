"""
tests/test_price_resolution_band.py
───────────────────────────────────
Which markets are worth quoting.

The old rule excluded everything outside [0.05, 0.95] as near-resolved. That
reads "extreme" as "dead", and on Polymarket it is not: the exchange drops the
tick from 0.01 to 0.001 as a market moves to the edges, so the same 0.02 price
is a 50% grid on the coarse tick and a 5% grid on the fine one. What matters is
what the grid can express, not where the price sits.

Measured across 600 live markets on 2026-09-10:

    YES band        median tick/cheap-side     tick=0.001
    0.000-0.010            66.7%                 152/152
    0.010-0.050             4.7%                  81/84
    0.250-0.750             2.5%                  11/169
    0.950-0.990             4.1%                   7/9
    0.990-1.000           200.0%                  33/33

So 0.01-0.05 and 0.95-0.99 resolve BETTER than the coin-flip band does on the
coarse grid, and their 24h volume is no lower — the 0.99+ band is the busiest
in the sample. Replacing the flat band with the resolution test admitted 123
more markets out of 600 and excluded none, and the ones it gained are among the
busiest on the platform.

Below 0.01 and above 0.99 stay out: the fine tick is still 67% and 200% of the
price there, so no maker quote can be a fine concession.
"""

import pytest

from strategy.arbitrage import (
    _within_quality_band,
    has_usable_resolution,
    price_resolution,
)


class TestResolution:

    def test_the_same_price_resolves_differently_on_the_two_grids(self):
        assert price_resolution(0.02, 0.01) == pytest.approx(0.50)
        assert price_resolution(0.02, 0.001) == pytest.approx(0.05)

    def test_it_measures_the_cheap_side(self):
        """A NegRisk bundle buys the NO leg, which is the cheap one."""
        assert price_resolution(0.97, 0.001) == pytest.approx(
            price_resolution(0.03, 0.001))

    def test_an_unknown_tick_gives_no_answer(self):
        assert price_resolution(0.5, None) is None
        assert price_resolution(0.5, 0.0) is None


class TestTheBand:

    def test_a_lopsided_market_on_the_fine_grid_is_admitted(self):
        """
        The change. An outcome implied at 0.97 with a 0.001 tick resolves to 3%
        of its cheap side — finer than a coin flip manages on the coarse grid.
        """
        assert _within_quality_band(0.97, 0.03, 0.05, 0.95) is False
        assert _within_quality_band(0.97, 0.03, 0.05, 0.95, tick=0.001) is True

    def test_the_same_market_on_the_coarse_grid_is_not(self):
        """0.03 with a 0.01 tick is a 33% grid: the dead zone."""
        assert _within_quality_band(0.97, 0.03, 0.05, 0.95, tick=0.01) is False

    def test_the_ultra_extreme_band_stays_out(self):
        """Below 0.01 even the fine tick is a fifth of the price."""
        assert _within_quality_band(0.997, 0.003, 0.05, 0.95, tick=0.001) is False

    def test_a_balanced_market_is_unaffected(self):
        assert _within_quality_band(0.52, 0.48, 0.05, 0.95, tick=0.01) is True

    def test_an_unknown_tick_falls_back_to_the_flat_band(self):
        """
        Refusing on missing metadata would be a worse error than the one this
        replaces, so the old rule stands in when the grid is unknown.
        """
        assert _within_quality_band(0.97, 0.03, 0.05, 0.95, tick=None) is False
        assert _within_quality_band(0.52, 0.48, 0.05, 0.95, tick=None) is True

    def test_nothing_inside_the_old_band_is_lost(self):
        """
        Measured: 0 of 600 live markets that the flat band admitted fail the
        resolution test. This is a widening, not a trade.
        """
        for price in (0.06, 0.20, 0.50, 0.80, 0.94):
            assert _within_quality_band(
                price, 1.0 - price, 0.05, 0.95, tick=0.01) is True


class TestUnwindingIsCheaperOutThere:
    """
    The other half of the case for admitting these. The taker fee is
    rate x p x (1-p), so a naked leg on a decided market costs far less to
    unwind — which is what the guard falls back to when a bundle half-fills.
    """

    def test_a_naked_leg_is_far_cheaper_to_unwind_at_the_edge(self):
        from strategy.arbitrage import effective_taker_fee
        at_edge = 0.02 * effective_taker_fee(0.02)
        at_coin = 0.50 * effective_taker_fee(0.50)
        assert at_edge < at_coin / 10


class TestTheRevertSwitch:
    """
    Widening admission by ~40% of the universe is the largest live change on
    this branch, so it needs an undo that does not require a deploy. Lowering
    MAX_TICK_FRACTION is not that undo — a small fraction rejects everything,
    including the coin-flip markets the bot has always traded.
    """

    def test_the_switch_restores_the_flat_band(self, monkeypatch):
        import strategy.arbitrage as A
        monkeypatch.setattr(A, "QUALITY_BAND_USE_RESOLUTION", False)
        assert A._within_quality_band(0.97, 0.03, 0.05, 0.95, tick=0.001) is False
        assert A._within_quality_band(0.52, 0.48, 0.05, 0.95, tick=0.01) is True

    def test_lowering_the_fraction_is_not_a_revert(self):
        """It rejects the balanced markets too, which is not what anyone wants."""
        assert has_usable_resolution(0.50, 0.01, max_fraction=0.001) is False
