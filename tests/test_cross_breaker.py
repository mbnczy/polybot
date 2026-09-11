"""
The breaker's cross ledger.

A cross position is held until its markets resolve — up to a week — so it gets
its own slots and capital ceiling rather than one of the bundle path's three
slots, which it would otherwise hold for days. Its realised result, though, lands
in the same session and daily P&L as everything else: the daily loss limit and
the drawdown guard cover both strategies.
"""

from __future__ import annotations

import pytest

import risk.circuit_breaker as cb
from risk.circuit_breaker import ArbOrderIntent, CircuitBreaker, CircuitBreakerTripped


def _breaker(monkeypatch, **env) -> CircuitBreaker:
    monkeypatch.setenv("CROSS_MAX_POSITIONS", str(env.get("positions", 2)))
    monkeypatch.setenv("CROSS_MAX_COMMITTED_USDC", str(env.get("committed", 10.0)))
    monkeypatch.setenv("MAX_POSITIONS", "3")
    return CircuitBreaker(starting_balance=100.0)


def _intent(cost=1.0):
    return ArbOrderIntent("0xc", "y", "n", 0.49, 0.49, 1.0, cost)


def test_a_cross_position_does_not_take_a_bundle_slot(monkeypatch):
    b = _breaker(monkeypatch)
    b.on_cross_open(4.0)
    b.on_cross_open(4.0)
    st = b.status_dict()
    assert st["open_positions"] == 0
    assert st["cross_open"] == 2
    # The bundle path still has all three of its slots.
    assert b.check_arb(_intent()) is True


def test_the_cross_position_cap(monkeypatch):
    b = _breaker(monkeypatch, positions=2)
    b.on_cross_open(1.0)
    b.on_cross_open(1.0)
    assert b.check_cross(1.0) is False


def test_the_cross_capital_cap(monkeypatch):
    b = _breaker(monkeypatch, committed=10.0)
    b.on_cross_open(8.0)
    assert b.check_cross(3.0) is False
    assert b.check_cross(2.0) is True


def test_nothing_to_commit_is_refused(monkeypatch):
    assert _breaker(monkeypatch).check_cross(0.0) is False


def test_the_result_lands_in_the_shared_daily_pnl(monkeypatch):
    b = _breaker(monkeypatch)
    b.on_cross_open(4.0)
    b.on_cross_close(-1.25, 4.0)
    st = b.status_dict()
    assert st["daily_pnl"] == pytest.approx(-1.25)
    assert st["session_pnl"] == pytest.approx(-1.25)
    assert st["cross_open"] == 0
    assert st["cross_committed_usdc"] == 0.0


def test_a_cross_loss_can_trip_the_daily_limit(monkeypatch):
    b = _breaker(monkeypatch)
    monkeypatch.setattr(cb, "DAILY_LOSS_LIMIT", -1.0)
    b.on_cross_open(4.0)
    with pytest.raises(CircuitBreakerTripped):
        b.on_cross_close(-2.0, 4.0)


def test_no_cross_entry_once_the_day_is_through_its_limit(monkeypatch):
    b = _breaker(monkeypatch)
    monkeypatch.setattr(cb, "DAILY_LOSS_LIMIT", -1.0)
    b.on_cross_open(4.0)
    with pytest.raises(CircuitBreakerTripped):
        b.on_cross_close(-2.0, 4.0)
    with pytest.raises(CircuitBreakerTripped):
        b.check_cross(1.0)


def test_a_release_books_nothing(monkeypatch):
    b = _breaker(monkeypatch)
    b.on_cross_open(4.0)
    b.release_cross(4.0)
    st = b.status_dict()
    assert st["cross_open"] == 0 and st["daily_pnl"] == 0.0


def test_drawdown_blocks_cross_entries_too(monkeypatch):
    b = _breaker(monkeypatch)
    b._state.session_pnl = -20.0            # 20% of a 100 balance, past the 15% guard
    assert b.check_cross(1.0) is False


def test_a_cross_close_does_not_feed_the_tuner(monkeypatch):
    """Fill timestamps measure BUNDLE conversion; a cross exit is not one."""
    b = _breaker(monkeypatch)
    b.on_cross_open(4.0)
    b.on_cross_close(0.5, 4.0)
    assert b.fills_since(0.0) == 0
