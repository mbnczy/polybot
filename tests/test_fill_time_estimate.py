"""
tests/test_fill_time_estimate.py
────────────────────────────────
How long a resting quote would wait.

Whether a post-only order fills is queue_ahead divided by the rate the queue
drains, and the bot only ever had the numerator. The WS feed carries `book` and
`price_change` and no trade tape, and a level shrinking in a price_change is a
cancel and a fill alike — so the flow number comes from the 24h volume the
scanner already computes for its liquidity floor and used to discard.

Why this matters more than a queue multiple: a queue is counted in SHARES, and
at an extreme price shares are cheap. The Fed market holding 93,756 shares at
0.007 is 656 dollars of book against a two-million-dollar day. The multiple
calls that 9,376x our order and hopeless; the flow says it clears in about a
minute. Where the two disagree the flow is right, because it is measuring the
thing we actually care about.

It is an estimate and the code says so. A day of volume says nothing about the
next five minutes.
"""

import math

import pytest

from strategy.arbitrage import (
    expected_fill_seconds,
    maker_quote_is_hopeless,
)


class TestTheEstimate:

    def test_a_thin_queue_on_a_busy_market_clears_in_seconds(self):
        s = expected_fill_seconds(90, 500_000, 0.50)
        assert s is not None and s < 60

    def test_the_median_one_tick_book_takes_half_an_hour(self):
        s = expected_fill_seconds(10_558, 500_000, 0.50)
        assert s is not None and 1_200 < s < 2_400

    def test_shares_are_cheap_at_an_extreme_price(self):
        """
        93,756 shares at 0.007 is 656 dollars of book. Counted in shares it
        looks impassable; counted in money it is a minute of flow.
        """
        s = expected_fill_seconds(93_756, 2_000_000, 0.007)
        assert s is not None and s < 120

    def test_our_own_price_level_has_no_queue(self):
        assert expected_fill_seconds(0, 500_000, 0.50) == 0.0

    def test_a_market_that_never_trades_never_fills(self):
        assert expected_fill_seconds(100, 0.0, 0.50) == math.inf

    def test_a_missing_input_gives_no_answer_rather_than_a_fast_one(self):
        """
        None must never read as "fills instantly" — that is the failure mode
        that turns a missing measurement into a green light.
        """
        assert expected_fill_seconds(None, 500_000, 0.50) is None
        assert expected_fill_seconds(1_000, None, 0.50) is None


class TestItOverridesTheQueueMultiple:

    def test_the_estimate_rescues_a_leg_the_multiple_would_discard(self):
        """
        50,000 shares at 0.01 on a 3M market: 5,000x our order by the multiple,
        29 seconds by the flow. Inside a 45-second TTL, so it is worth waiting
        for — and the multiple alone would have thrown it away.
        """
        est = expected_fill_seconds(50_000, 3_000_000, 0.01)
        assert maker_quote_is_hopeless(50_000, 10) is True
        assert maker_quote_is_hopeless(
            50_000, 10, expected_s=est, ttl_s=45.0) is False

    def test_the_estimate_still_condemns_a_genuinely_stuck_leg(self):
        est = expected_fill_seconds(10_558, 500_000, 0.50)
        assert maker_quote_is_hopeless(
            10_558, 10, expected_s=est, ttl_s=45.0) is True

    def test_without_an_estimate_the_multiple_still_applies(self):
        """The estimate is an improvement, not a dependency."""
        assert maker_quote_is_hopeless(10_558, 10, expected_s=None) is True

    def test_a_leg_leading_the_book_is_never_hopeless(self):
        assert maker_quote_is_hopeless(
            999_999, 10, leads=True, expected_s=1e9, ttl_s=45.0) is False


class TestTheFlowStore:

    def test_a_scanned_group_is_remembered(self):
        from core import market_flow
        market_flow.clear()
        market_flow.remember("0xgrp", 53_000.0)
        assert market_flow.get("0xgrp") == 53_000.0

    def test_an_unscanned_group_is_unknown_not_zero(self):
        """
        Zero would mean "never trades", which the estimator reads as never
        fills. Not knowing has to stay distinct from knowing it is dead.
        """
        from core import market_flow
        market_flow.clear()
        assert market_flow.get("0xnever-seen") is None

    def test_junk_is_ignored_rather_than_stored(self):
        from core import market_flow
        market_flow.clear()
        market_flow.remember("0xgrp", "not a number")
        market_flow.remember("", 100.0)
        assert market_flow.get("0xgrp") is None
        assert market_flow.tracked() == 0
