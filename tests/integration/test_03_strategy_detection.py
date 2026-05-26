"""Step 3 — ArbDetector + FeeEngine produce an ArbSignal on crafted prices."""

from __future__ import annotations

import pytest

from strategy.arbitrage import ArbDetector, FeeEngine


def test_detector_emits_signal_when_combined_with_fee_below_threshold():
    detector = ArbDetector(desired_net_margin=0.005, default_fee_rate=0.0)
    sig = detector.evaluate(
        condition_id="0x" + "ab" * 32,
        yes_token_id="Y",
        no_token_id="N",
        yes_ask=0.47,
        no_ask=0.50,
        max_position_usdc=50.0,
        fee_rate=0.0,
    )
    assert sig is not None
    assert sig.combined_cost == pytest.approx(0.97, rel=1e-6)
    assert sig.net_edge == pytest.approx(0.03, rel=1e-6)
    assert sig.yes_size > 0 and sig.yes_size == sig.no_size


def test_detector_returns_none_when_no_edge():
    detector = ArbDetector(desired_net_margin=0.005, default_fee_rate=0.02)
    sig = detector.evaluate(
        condition_id="0x" + "cd" * 32,
        yes_token_id="Y",
        no_token_id="N",
        yes_ask=0.55,
        no_ask=0.50,        # combined 1.05 — no arb
        fee_rate=0.02,
    )
    assert sig is None


@pytest.mark.asyncio
async def test_fee_engine_cache_priming():
    fee = FeeEngine(default_fee=0.02)
    fee.prime_cache("0xCOND", 0.01)
    assert await fee.get_taker_fee("0xCOND") == 0.01
