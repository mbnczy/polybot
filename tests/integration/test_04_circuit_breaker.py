"""Step 4 — CircuitBreaker sizing & blocking rules."""

from __future__ import annotations

import pytest

import risk.circuit_breaker as cb_mod
from risk.circuit_breaker import (
    ArbOrderIntent,
    CircuitBreaker,
    CircuitBreakerTripped,
    MAX_ARB_PAIR_USDC,
)


@pytest.fixture(autouse=True)
def _isolate_daily_state(tmp_path, monkeypatch):
    monkeypatch.setattr(cb_mod, "_DAILY_STATE_PATH", str(tmp_path / "daily.json"))
    yield


def _intent(cost: float) -> ArbOrderIntent:
    return ArbOrderIntent(
        condition_id="0x" + "ab" * 32,
        yes_token_id="Y", no_token_id="N",
        yes_price=0.47, no_price=0.50,
        n_shares=cost / 0.97,
        combined_cost_usdc=cost,
    )


def test_calculate_position_size_respects_50_usdc_cap():
    breaker = CircuitBreaker(starting_balance=500.0)
    n = breaker.calculate_position_size(yes_ask=0.47, no_ask=0.50)
    assert n * 0.97 <= MAX_ARB_PAIR_USDC + 1e-9


def test_pair_cost_above_cap_is_blocked():
    breaker = CircuitBreaker(starting_balance=500.0)
    assert breaker.check_arb(_intent(cost=MAX_ARB_PAIR_USDC + 1.0)) is False


def test_daily_loss_limit_raises():
    breaker = CircuitBreaker(starting_balance=500.0)
    breaker._daily.daily_pnl = cb_mod.DAILY_LOSS_LIMIT - 0.01
    with pytest.raises(CircuitBreakerTripped):
        breaker.check_arb(_intent(cost=10.0))


def test_position_cap_blocks_after_max(monkeypatch):
    monkeypatch.setenv("MAX_POSITIONS", "2")
    breaker = CircuitBreaker(starting_balance=500.0)
    breaker._state.open_positions = 2
    assert breaker.check_arb(_intent(cost=10.0)) is False


def test_drawdown_block(monkeypatch):
    monkeypatch.setenv("MAX_SESSION_DRAWDOWN_PCT", "10.0")
    breaker = CircuitBreaker(starting_balance=500.0)
    breaker._state.session_pnl = -75.0   # 15% drawdown > 10% cap
    assert breaker.check_arb(_intent(cost=10.0)) is False
