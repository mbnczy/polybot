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


# ═══════════════════════════════════════════════════════════════════════════════
# Realistic-completion filter — score the bundle at the price we will pay
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealisticCompletionFilter:
    """
    An all-maker price is an assumption, not a plan.

    A leg whose spread IS one tick cannot be quoted above the touch post-only,
    so our order joins a queue and usually does not fill; the bundle then either
    finishes as a taker on that leg or not at all. Scoring at the all-maker
    price prices an outcome that does not happen, which is how a full night of
    bundles "cleared" on entry and lost money on resolution.

    Measured over 43 live NegRisk groups: 7 cleared all-maker, 1 still cleared
    after crossing a leg, and 0 became an arb only by crossing. The filter
    removes signals; that is the point, and the six it removes are the ones
    that half-fill.
    """

    def _det(self, **kw):
        from strategy.arbitrage import NegRiskArbDetector
        return NegRiskArbDetector(
            desired_net_margin=0.001, min_outcome_prob=0.02, max_legs=4,
            default_rebate_rate=0.0, min_relative_edge=0.001, **kw
        )

    def _call(self, det, asks, bids, ticks):
        return det.evaluate_neg_risk(
            condition_id="0xg",
            outcome_token_ids=[f"T{i}" for i in range(len(asks))],
            no_asks=asks,
            max_position_usdc=100.0,
            tick_size=max(ticks),
            no_ask_sizes=[500.0] * len(asks),
            no_best_bids=bids,
            leg_tick_sizes=ticks,
        )

    def test_bundle_that_only_clears_all_maker_is_rejected(self):
        """
        Three one-tick legs: each quote lands ON the touch, so realistically we
        pay the ask on all three. All-maker says +0.03; completion says -0.00.
        """
        det = self._det()
        asks  = [0.85, 0.60, 0.56]
        bids  = [0.84, 0.59, 0.55]     # spread == tick on every leg
        ticks = [0.01, 0.01, 0.01]
        # all-maker cost 0.84+0.59+0.55 = 1.98 vs floor 2.00 -> +0.02
        # realistic  cost 0.85+0.60+0.56 = 2.01 vs floor 2.00 -> -0.01
        assert self._call(det, asks, bids, ticks) is None

    def test_bundle_that_survives_completion_is_kept(self):
        """Wide spreads: our quotes lead the book, so maker prices are real."""
        det = self._det()
        asks  = [0.85, 0.60, 0.56]
        bids  = [0.70, 0.45, 0.40]     # our quotes lead comfortably
        ticks = [0.01, 0.01, 0.01]
        sig = self._call(det, asks, bids, ticks)
        assert sig is not None
        assert sig.net_edge > 0

    def test_filter_can_be_disabled(self):
        """The old all-maker scoring stays reachable for comparison."""
        det = self._det()
        det._require_completable = False
        asks  = [0.85, 0.60, 0.56]
        bids  = [0.84, 0.59, 0.55]
        assert self._call(det, asks, bids, [0.01, 0.01, 0.01]) is not None

    def test_absent_bid_data_does_not_block_a_signal(self):
        """Unknown bids must not be treated as 'we are behind the touch'."""
        det = self._det()
        sig = self._call(det, [0.85, 0.60, 0.56], [None, None, None],
                         [0.01, 0.01, 0.01])
        assert sig is not None

    def test_a_single_hard_leg_is_priced_at_its_ask(self):
        """
        Mixed case: two legs lead their books, one is stuck on a one-tick
        spread. Only that leg should be charged at the ask.
        """
        det = self._det()
        asks  = [0.85, 0.60, 0.55]
        bids  = [0.70, 0.45, 0.54]     # leg 2 is the stuck one
        ticks = [0.01, 0.01, 0.01]
        # realistic = 0.84 + 0.59 + 0.55*(1+fee) = 1.9899 -> +0.0101, clears.
        # Note the fee IS charged on the crossed leg: without it this fixture
        # reads 1.98, and entry would be more optimistic than completion.
        sig = self._call(det, asks, bids, ticks)
        assert sig is not None


# ═══════════════════════════════════════════════════════════════════════════════
# The taker fee, solved rather than assumed
# ═══════════════════════════════════════════════════════════════════════════════

class TestTakerFeeModel:
    """
    Solved from 68 settled taker fills against data-api `usdcSize` (net of
    fees). The answer is exactly bimodal — 0.04 or 0.05, no scatter — and the
    two sets are disjoint in time:

        0.05   51 fills   2026-07-19 .. 2026-09-04
        0.04   17 fills   2026-09-05 .. 2026-09-06

    So the rate was cut; it does not vary by market. It only looked per-market
    because each market happened to be traded on one side of the cut.
    """

    def test_formula_reproduces_real_charges(self):
        from strategy.arbitrage import effective_taker_fee as f
        # (size, price, gross, net) straight from settled fills at rate 0.04
        for size, px, gross, net in ((5.07, 0.9580, 4.85706, 4.84891),
                                     (5.07, 0.1800, 0.91260, 0.88267),
                                     (4.00, 0.8300, 3.32000, 3.29743),
                                     (65.91, 0.9566, 63.04996, 62.94053)):
            charged = (gross - net) / gross
            assert f(px, rate=0.04) == pytest.approx(charged, abs=2e-5)

    def test_fee_rises_as_price_falls(self):
        from strategy.arbitrage import effective_taker_fee as f
        pcts = [f(p, rate=0.04) for p in (0.96, 0.83, 0.50, 0.18, 0.04)]
        assert pcts == sorted(pcts), "fee must increase as the asset gets cheaper"

    def test_dollar_fee_peaks_in_the_middle(self):
        """
        The shape that made this confusing: as a FRACTION the fee grows as price
        falls, but in DOLLARS it is symmetric and peaks at p = 0.50.
        """
        from strategy.arbitrage import effective_taker_fee as f
        dollars = {p: f(p, rate=0.04) * (100 * p) for p in (0.04, 0.5, 0.96)}
        assert dollars[0.5] > dollars[0.04]
        assert dollars[0.04] == pytest.approx(dollars[0.96], abs=1e-9)

    def test_makers_pay_nothing(self):
        from strategy.arbitrage import maker_fee
        assert all(maker_fee(p) == 0.0 for p in (0.02, 0.5, 0.98))

    def test_rate_can_be_recalibrated_at_runtime(self):
        """
        The constant is a seed, not a truth. The exchange cut the rate once
        already; hard-coding either value would be wrong the next time.
        """
        import strategy.arbitrage as a
        prev = a._LIVE_TAKER_RATE
        try:
            a.set_taker_rate(0.05)
            assert a.effective_taker_fee(0.50) == pytest.approx(0.025)
            a.set_taker_rate(0.04)
            assert a.effective_taker_fee(0.50) == pytest.approx(0.020)
        finally:
            a.set_taker_rate(prev)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_impossible_prices_carry_no_fee(self, bad):
        from strategy.arbitrage import effective_taker_fee
        assert effective_taker_fee(bad) == 0.0

    def test_entry_charges_the_same_fee_completion_will(self):
        """
        Entry and completion must price crossing identically. Omitting the fee
        at entry made it the more optimistic of the two, so bundles were
        admitted that _try_complete then refused — leaving precisely the
        half-filled position both checks exist to prevent.
        """
        from strategy.arbitrage import (
            NegRiskArbDetector, effective_taker_fee, snap_post_only_bid,
        )
        det = NegRiskArbDetector(
            desired_net_margin=0.001, min_outcome_prob=0.02, max_legs=4,
            default_rebate_rate=0.0, min_relative_edge=0.001,
        )
        asks  = [0.85, 0.60, 0.56]
        bids  = [0.84, 0.45, 0.40]          # leg 0 is stuck on a one-tick spread
        ticks = [0.01, 0.01, 0.01]
        sig = det.evaluate_neg_risk(
            condition_id="0xg",
            outcome_token_ids=["T0", "T1", "T2"],
            no_asks=asks, max_position_usdc=100.0, tick_size=0.01,
            no_ask_sizes=[500.0] * 3, no_best_bids=bids, leg_tick_sizes=ticks,
        )
        # realistic cost = 0.85*(1+fee(0.85)) + 0.59 + 0.55 = 1.9851 -> +0.0149
        crossed = 0.85 * (1.0 + effective_taker_fee(0.85))
        realistic = crossed + snap_post_only_bid(0.60, 0.01) + snap_post_only_bid(0.56, 0.01)
        assert 2.0 - realistic > 0.002, "fixture should still clear"
        assert sig is not None
        # and the fee must actually be charged: without it the cost would be
        # 0.85 flat, a materially different number
        assert crossed > 0.85


class TestLeadBookBid:
    """
    Quote the cheapest price that still leads the book, not `ask - tick`.

    Queue priority is price-then-time, so any price above the resting best bid
    is first in line. On a five-tick spread `ask - tick` pays three ticks for a
    position `best_bid + tick` already has. Measured across 102 live NegRisk
    legs: 25 over-quoted, 0.408 in total, median 0.004 per bundle against a
    typical bundle edge of 0.02-0.03.
    """

    def test_wide_spread_quotes_just_above_the_touch(self):
        from strategy.arbitrage import lead_book_bid
        assert lead_book_bid(0.90, 0.85, 0.01) == pytest.approx(0.86)

    def test_one_tick_spread_is_unchanged(self):
        """Nothing to save when ask - tick already IS the best bid."""
        from strategy.arbitrage import lead_book_bid, snap_post_only_bid
        assert lead_book_bid(0.85, 0.84, 0.01) == pytest.approx(
            snap_post_only_bid(0.85, 0.01)
        )

    def test_never_crosses_the_ask(self):
        from strategy.arbitrage import lead_book_bid
        # a bid already at the ask must still produce a non-crossing quote
        assert lead_book_bid(0.50, 0.60, 0.01) < 0.50

    def test_fine_tick_grid_is_respected(self):
        from strategy.arbitrage import lead_book_bid
        q = lead_book_bid(0.982, 0.966, 0.001)
        assert q == pytest.approx(0.967)
        assert round(q, 3) == q, "must sit on the market's grid"

    def test_unknown_bid_falls_back_to_ask_minus_tick(self):
        from strategy.arbitrage import lead_book_bid, snap_post_only_bid
        assert lead_book_bid(0.90, None, 0.01) == pytest.approx(
            snap_post_only_bid(0.90, 0.01)
        )

    def test_quote_still_leads_the_book(self):
        """The saving must not cost queue position — that is the whole point."""
        from strategy.arbitrage import lead_book_bid
        for ask, bid, tick in ((0.90, 0.85, 0.01), (0.58, 0.53, 0.01),
                               (0.982, 0.966, 0.001)):
            assert lead_book_bid(ask, bid, tick) > bid


class TestMakerRebateIsNotEntryIncome:
    """
    The maker rebate is real but it is neither per-fill nor 1%.

    Measured 2026-09-06: 175 maker fills moved exactly size x price (max
    deviation 7e-10), and the account has received one MAKER_REBATE activity
    entry of 2.4467 USDC — 0.2157% against 1134.33 USDC of maker volume, versus
    the 1% the code assumed for politics.

    Discounting entry cost by it books unearned, conditional, separately-paid
    income at the moment of purchase. On a bundle whose real edge is +0.0300 the
    assumed 1% added +0.0197: 40% of the perceived edge, invented.
    """

    def test_default_rebate_is_zero(self):
        from strategy.arbitrage import DEFAULT_MAKER_REBATE
        assert DEFAULT_MAKER_REBATE == 0.0

    def test_engine_defaults_to_no_rebate(self):
        from strategy.arbitrage import MakerRebateEngine
        assert MakerRebateEngine()._default == 0.0

    def test_bundle_edge_is_not_inflated_by_a_phantom_rebate(self):
        """The arb must stand on its own; the rebate is upside on top."""
        combined_bid, payout = 1.97, 2.0
        assert payout - combined_bid * (1 - 0.0) == pytest.approx(0.03)
        # what the old default did
        assert payout - combined_bid * (1 - 0.01) == pytest.approx(0.0497, abs=1e-4)

    def test_an_explicit_rebate_is_still_honoured(self):
        """Opt-in remains possible for anyone who has measured the credit."""
        from strategy.arbitrage import MakerRebateEngine
        eng = MakerRebateEngine(default_rebate=0.002)
        assert eng._default == pytest.approx(0.002)


class TestNegRiskGroupVolumeFloor:
    """
    A NegRisk group needs a live book, not just a wide spread.

    The binary-pair path has had a volume floor since forever; the group path
    had none, so a group was registered on the negRisk flag alone regardless of
    whether anything traded there. The bot ended up placing 114 bundles an hour
    into "Will Jack Doherty be sentenced to at least 5 years in prison" — 49 USD
    of 24h volume, 22-tick spread, zero fills in 33 hours.

    The failure is subtle because a dead market looks BEST to a price-only
    detector: the wide spread lets a quote sit far below the ask and still lead
    the book, so the entry filter sees both legs leading and passes it.
    """

    def _scanner(self, floor=5000.0):
        from core.scanner import MarketScanner
        sc = MarketScanner.__new__(MarketScanner)
        sc._negrisk_min_group_volume = floor
        return sc

    def test_dead_group_is_below_the_floor(self):
        from core.scanner import _market_volume
        doherty = [{"volume24hr": "49.25"}, {"volume24hr": "0"}]
        assert sum(_market_volume(m) for m in doherty) < 5000.0

    def test_a_real_group_clears_it(self):
        """The smallest live NegRisk group measured turns over 53k in 24h."""
        from core.scanner import _market_volume
        real = [{"volume24hr": "30000"}, {"volume24hr": "23011"}]
        assert sum(_market_volume(m) for m in real) >= 5000.0

    def test_floor_is_configurable(self, monkeypatch):
        import importlib
        monkeypatch.setenv("NEGRISK_MIN_GROUP_VOLUME_24H", "123")
        import core.scanner as sc
        importlib.reload(sc)
        try:
            assert sc.NEGRISK_MIN_GROUP_VOLUME_24H == pytest.approx(123.0)
        finally:
            monkeypatch.delenv("NEGRISK_MIN_GROUP_VOLUME_24H", raising=False)
            importlib.reload(sc)

    def test_missing_volume_field_reads_as_zero(self):
        """An absent figure must not be treated as unlimited liquidity."""
        from core.scanner import _market_volume
        assert _market_volume({}) == 0.0


class TestFeeRateTracksChanges:
    """
    The taker rate is not a constant to be pinned. The exchange has cut it twice
    inside one sample:

        0.05   2026-07-19 .. 2026-09-04
        0.04   2026-09-05 .. 2026-09-06
        0.03   2026-09-07

    Calibrating only at startup left the bot quoting 0.04 for hours after the
    cut to 0.03, over-charging every completion it evaluated and refusing trades
    that were profitable.
    """

    def test_set_taker_rate_moves_the_estimate(self):
        import strategy.arbitrage as a
        prev = a._LIVE_TAKER_RATE
        try:
            a.set_taker_rate(0.03)
            assert a.effective_taker_fee(0.85) == pytest.approx(0.03 * 0.15)
            a.set_taker_rate(0.05)
            assert a.effective_taker_fee(0.85) == pytest.approx(0.05 * 0.15)
        finally:
            a.set_taker_rate(prev)

    def test_todays_observed_rate_reproduces_todays_charges(self):
        """Both fills of 2026-09-07 solve to exactly 0.0300."""
        import strategy.arbitrage as a
        prev = a._LIVE_TAKER_RATE
        try:
            a.set_taker_rate(0.03)
            for size, px, gross, net in ((5.13, 0.83, 4.25790, 4.23619),
                                         (5.13, 0.85, 4.36050, 4.38012)):
                charged = abs(gross - net) / gross
                assert a.effective_taker_fee(px) == pytest.approx(charged, abs=2e-4)
        finally:
            a.set_taker_rate(prev)

    def test_rate_is_clamped_to_a_sane_band(self):
        import strategy.arbitrage as a
        prev = a._LIVE_TAKER_RATE
        try:
            a.set_taker_rate(-1.0)
            assert a._LIVE_TAKER_RATE == 0.0
            a.set_taker_rate(99.0)
            assert a._LIVE_TAKER_RATE <= 0.5
        finally:
            a.set_taker_rate(prev)


class TestVolumeFloorUsesRealTwentyFourHourVolume:
    """
    A 24h liquidity floor must not be satisfied by lifetime volume.

    _market_volume falls through to lifetime `volume` when the 24h figure is
    absent or None. That is reasonable for RANKING and wrong for a FLOOR: a
    long-dead market with a large lifetime total then reads as highly liquid.
    The Jack Doherty group — 49 USD of 24h volume — passed a 5000 floor on
    exactly this and kept being quoted 63 times an hour.
    """

    def test_missing_24h_field_does_not_borrow_lifetime_volume(self):
        from core.scanner import _market_volume, _market_volume_24h_strict
        dead = {"volume": "250000"}          # no 24h figure at all
        assert _market_volume(dead) == pytest.approx(250000.0)   # ranking helper
        assert _market_volume_24h_strict(dead) == 0.0            # the floor

    def test_null_24h_field_is_also_zero(self):
        from core.scanner import _market_volume_24h_strict
        assert _market_volume_24h_strict({"volume24hr": None, "volume": "250000"}) == 0.0

    def test_a_real_24h_figure_is_used_as_is(self):
        from core.scanner import _market_volume_24h_strict
        assert _market_volume_24h_strict(
            {"volume24hr": "49.25", "volume": "250000"}
        ) == pytest.approx(49.25)

    def test_garbage_reads_as_zero_not_as_liquid(self):
        from core.scanner import _market_volume_24h_strict
        for bad in ({"volume24hr": "abc"}, {"volume24hr": float("nan")}, {}):
            assert _market_volume_24h_strict(bad) == 0.0


class TestRebateTableHonoursTheMeasurement:
    """
    Zeroing DEFAULT_MAKER_REBATE only changed the FALLBACK. The category table
    was still consulted, so politics markets kept being discounted 1% and every
    NegRisk signal went on logging "rebate=1.00%" hours after the fix landed.
    """

    def test_politics_lookup_returns_zero_while_the_default_is_zero(self):
        from strategy.arbitrage import _resolve_rebate
        assert _resolve_rebate("politics") == 0.0
        assert _resolve_rebate("crypto") == 0.0

    def test_the_published_table_is_still_intact_underneath(self):
        """Kept for reference and for anyone who re-enables the credit."""
        from strategy.arbitrage import _resolve_rebate_from_table
        assert _resolve_rebate_from_table("politics") == pytest.approx(0.01)
