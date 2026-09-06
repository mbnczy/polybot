"""
The eight fixes of 2026-09-06, each pinned to the evidence that motivated it.

These all come from one night of live trading in which the bot completed zero
bundles, unwound every partial at a loss, and leaked 29 pUSD into an unwatched
directional position. The common thread is that the bot's own account of events
was internally consistent and wrong, so each test here asserts against something
external: a measured fee, a real book, an on-chain balance.
"""

from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fee calibrated from settled trades, not assumed
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeeCalibration:
    def test_calibrate_replaces_the_assumed_default(self):
        from strategy.arbitrage import FeeEngine
        fe = FeeEngine(default_fee=0.02)
        assert not fe.calibrated
        fe.calibrate(0.0)
        assert fe.calibrated
        assert fe._default == pytest.approx(0.0)

    def test_calibration_is_clamped_to_the_sane_band(self):
        from strategy.arbitrage import FeeEngine, MAX_TAKER_FEE
        fe = FeeEngine()
        fe.calibrate(99.0)
        assert fe._default == pytest.approx(MAX_TAKER_FEE)
        fe.calibrate(-1.0)
        assert fe._default == pytest.approx(0.0)

    def test_calibrating_drops_cached_assumptions(self):
        """Cached fees priced with the old assumption must not survive."""
        from strategy.arbitrage import FeeEngine
        fe = FeeEngine(default_fee=0.02)
        fe.prime_cache("0xabc", 0.02)
        fe.calibrate(0.0)
        assert fe.peek_taker_fee("0xabc") is None

    def test_gamma_base_fee_is_still_rejected_as_implausible(self):
        """
        Gamma returns takerBaseFee=1000, which normalises to 10%. That is not a
        real rate and must not become one — it is why the lookup fell through
        in the first place.
        """
        from strategy.arbitrage import _normalise_fee
        assert _normalise_fee(1000) is None
        assert _normalise_fee(100) == pytest.approx(0.01)
        assert _normalise_fee(0) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Bids that buy queue position, not just edge
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerLegTickSnapping:
    """
    3. Each leg is quoted on its OWN tick grid.

    The group published only the coarsest tick (max over members), so a leg
    whose real grid is 0.001 was snapped to 0.01 and quoted a full cent below
    the touch — invisible on the book while its coarse siblings quoted normally.

    What is deliberately NOT here is a way to out-bid the touch. naive =
    ask - tick and best_bid = ask - spread, so the quote only lands at or below
    the best bid when the spread IS one tick, and the next price up is then the
    ask itself. Post-only cannot improve a one-tick book; crossing is the only
    lever, and that is _try_complete's job.
    """

    def test_fine_tick_leg_is_quoted_on_its_own_grid(self):
        from strategy.arbitrage import snap_post_only_bid
        assert snap_post_only_bid(0.960, 0.001) == pytest.approx(0.959)
        assert snap_post_only_bid(0.960, 0.010) == pytest.approx(0.95)

    def test_coarse_grid_costs_a_full_cent_of_queue_position(self):
        from strategy.arbitrage import snap_post_only_bid
        own    = snap_post_only_bid(0.960, 0.001)
        coarse = snap_post_only_bid(0.960, 0.010)
        assert own - coarse == pytest.approx(0.009, abs=1e-9)

    def test_one_tick_spread_leaves_the_quote_on_the_touch(self):
        """The structural limit that makes taker completion necessary."""
        from strategy.arbitrage import snap_post_only_bid
        ask, tick, best_bid = 0.85, 0.01, 0.84
        assert snap_post_only_bid(ask, tick) == pytest.approx(best_bid)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. The unwind depth probe must ignore dust
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnwindDepthBand:
    def test_dust_bids_are_not_counted_as_depth(self):
        """
        The live book carried thousands of shares at 0.001-0.003 under a 0.96
        touch. Summing the whole side reported effectively infinite depth, so
        the probe never sliced and the FOK was priced against liquidity that
        would never fill it.
        """
        from core.clob_client import _UNWIND_DEPTH_BAND
        levels = [(0.957, 40.0), (0.956, 552.0), (0.003, 11385.0), (0.001, 4155.0)]
        best  = max(p for p, _ in levels)
        floor = best * (1.0 - _UNWIND_DEPTH_BAND)
        usable = sum(sz for p, sz in levels if p >= floor)
        assert usable == pytest.approx(592.0)
        assert usable < 1000, "dust still counted as real depth"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. The tuner must react to fills, not to signals
# ═══════════════════════════════════════════════════════════════════════════════

class TestBreakerFillTracking:
    def test_fills_are_timestamped_and_queryable(self):
        import time
        from risk.circuit_breaker import CircuitBreaker
        b = CircuitBreaker(starting_balance=100.0)
        assert b.fills_since(0) == 0
        b.on_arb_open(); b.on_fill(pnl=0.1)
        b.on_arb_open(); b.on_fill(pnl=0.2)
        assert b.fills_since(0) == 2
        assert b.fills_since(time.time() + 60) == 0

    def test_fill_history_is_bounded(self):
        from risk.circuit_breaker import CircuitBreaker
        b = CircuitBreaker(starting_balance=1000.0)
        for _ in range(700):
            b.on_arb_open(); b.on_fill(pnl=0.0)
        assert len(b._fill_times) <= 512
