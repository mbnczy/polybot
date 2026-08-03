"""
tests/test_tick_size.py
───────────────────────
Per-market tick-size handling for maker bids.

Regression for the live "ARB EXECUTION ERROR: price must conform to tick size
0.01 with at most 2 decimal places" — a genuine 0.49/0.50 arb (7-minute window,
218 bps edge) was lost because the hardcoded 0.001 tick produced off-grid
3-decimal bids (0.489/0.499) on a 0.01-tick market.
"""

from __future__ import annotations

import pytest

from strategy.arbitrage import (
    DEFAULT_TICK_SIZE,
    DutchBookPricer,
    NegRiskArbDetector,
    snap_post_only_bid,
)


class TestSnapPostOnlyBid:
    def test_penny_tick_market_two_decimals(self):
        # 0.01-tick market: bid one cent below, exactly 2 decimals.
        assert snap_post_only_bid(0.49, 0.01) == 0.48
        assert snap_post_only_bid(0.50, 0.01) == 0.49

    def test_milli_tick_market_three_decimals(self):
        # 0.001-tick market: bid one tenth-cent below, 3 decimals.
        assert snap_post_only_bid(0.49, 0.001) == 0.489
        assert snap_post_only_bid(0.50, 0.001) == 0.499

    def test_unknown_tick_defaults_to_penny_grid(self):
        # None → coarse 0.01 grid (valid on every market).
        b = snap_post_only_bid(0.49, None)
        assert b == 0.48
        assert round(b, 2) == b   # never more than 2 decimals

    def test_result_always_on_grid(self):
        for ask in (0.11, 0.235, 0.49, 0.755, 0.89):
            for tick in (0.01, 0.001, None):
                bid = snap_post_only_bid(ask, tick)
                dec = 2 if (tick or DEFAULT_TICK_SIZE) >= 0.01 else 3
                assert round(bid, dec) == bid, (ask, tick, bid)
                assert bid < ask            # strictly below (post-only)
                assert bid > 0.0

    def test_clamped_to_valid_range(self):
        # Near the edges the bid must not fall to 0 or cross 1.
        assert snap_post_only_bid(0.01, 0.01) >= 0.01
        assert snap_post_only_bid(0.99, 0.01) <= 0.99


class TestEvaluateMakerTickAware:
    def test_penny_market_produces_valid_2decimal_bids(self):
        """The exact live case: 0.49/0.50 on a 0.01-tick market."""
        p = DutchBookPricer(desired_net_margin=0.005,
                            extreme_lo=0.10, extreme_hi=0.90, min_real_edge=0.005)
        sig = p.evaluate_maker("0x32b0", "y", "n", yes_ask=0.49, no_ask=0.50,
                               maker_rebate=0.01, tick_size=0.01)
        assert sig is not None
        # Bids must be valid 2-decimal prices (would have been rejected before).
        assert sig.yes_bid == 0.48 and sig.no_bid == 0.49
        assert round(sig.yes_bid, 2) == sig.yes_bid
        assert round(sig.no_bid, 2) == sig.no_bid

    def test_milli_market_keeps_fine_bids(self):
        p = DutchBookPricer(desired_net_margin=0.005,
                            extreme_lo=0.10, extreme_hi=0.90, min_real_edge=0.005)
        sig = p.evaluate_maker("c", "y", "n", yes_ask=0.49, no_ask=0.50,
                               maker_rebate=0.01, tick_size=0.001)
        assert sig is not None
        assert sig.yes_bid == 0.489 and sig.no_bid == 0.499

    def test_negrisk_uses_tick_grid(self):
        d = NegRiskArbDetector(desired_net_margin=0.005)
        sig = d.evaluate_neg_risk("c", ["a", "b", "c"], [0.30, 0.30, 0.30],
                                  maker_rebate=0.01, tick_size=0.01)
        assert sig is not None
        for leg in sig.legs:
            assert round(leg.no_bid, 2) == leg.no_bid   # on the 0.01 grid
