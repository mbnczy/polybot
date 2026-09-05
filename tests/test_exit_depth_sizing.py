"""
Never post a maker pair larger than we could sell back.

A maker pair is not atomic: either leg can fill alone, leaving a naked position
the guard must unwind into the bid book. On 2026-09-05 a naked leg exceeded what
that book could absorb, the FOK unwind was killed repeatedly, and ~35 pUSD (half
the account) sat stranded in an unhedged position.

Sizing therefore caps to MAKER_MAX_EXIT_FRACTION of the THINNER leg's visible
bid depth — the side that would be hardest to exit.
"""

from __future__ import annotations

import pytest

from strategy.arbitrage import DutchBookPricer


def _pricer():
    return DutchBookPricer(desired_net_margin=0.015)


BASE = dict(
    condition_id="0xcond", yes_token_id="Y", no_token_id="N",
    yes_ask=0.66, no_ask=0.33, max_position_usdc=10.0,
    maker_rebate=0.01, tick_size=0.01,
)


def test_no_depth_data_leaves_sizing_untouched():
    """Backward compatible: absent depth, the capital cap alone applies."""
    sig = _pricer().evaluate_maker(**BASE, exit_depth=None)
    assert sig is not None
    assert sig.yes_size == pytest.approx(10.3)


def test_deep_book_does_not_cap():
    sig = _pricer().evaluate_maker(**BASE, exit_depth=10_000.0)
    assert sig is not None
    assert sig.yes_size == pytest.approx(10.3)


def test_thin_book_caps_to_a_sellable_size():
    """20 shares of exit liquidity, 25% cap -> 5 shares, not 10.3."""
    sig = _pricer().evaluate_maker(**BASE, exit_depth=20.0)
    assert sig is not None
    assert sig.yes_size == pytest.approx(5.0)
    assert sig.no_size == pytest.approx(5.0)


def test_both_legs_capped_equally():
    """A pair must stay balanced — an uneven pair is not an arb."""
    sig = _pricer().evaluate_maker(**BASE, exit_depth=20.0)
    assert sig is not None
    assert sig.yes_size == pytest.approx(sig.no_size)


def test_book_too_thin_to_trade_is_rejected():
    """Below the minimum tradeable size, produce no signal at all."""
    sig = _pricer().evaluate_maker(**BASE, exit_depth=0.02)
    assert sig is None
