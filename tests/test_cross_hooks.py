"""
The two hooks that let cross positions live alongside the rest of the bot:
the reconciler leaves their markets out, and the redeemer takes them in.
"""

from __future__ import annotations

import asyncio

import pytest

from execution.auto_redeem import AutoRedeemer
from execution.reconciler import WalletReconciler


class _Client:
    def __init__(self, rows, cash=100.0):
        self.rows, self.cash = rows, cash

    async def open_positions_detail(self):
        return self.rows

    async def collateral_balance(self):
        return self.cash


class _Breaker:
    def status_dict(self):
        return {"open_positions": 0}


class _Notifier:
    def __init__(self):
        self.messages = []

    async def notify(self, text):
        self.messages.append(text)


CROSS = "Moik Baku vs. Simal: Moik Baku O/U 0.5"


def _row(title, outcome, size):
    return {"title": title, "outcome": outcome, "size": size}


def test_cross_inventory_is_not_reported_unmanaged_at_startup():
    client = _Client([_row(CROSS, "No", 10.0), _row("Some other market", "Yes", 3.0)])
    n = _Notifier()
    r = WalletReconciler(client, _Breaker(), n, managed=lambda: {CROSS})
    asyncio.run(r.check_once())
    assert len(n.messages) == 1
    assert "Some other market" in n.messages[0]
    assert "Moik" not in n.messages[0]


def test_a_cross_entry_or_exit_is_not_an_escape():
    client = _Client([])
    r = WalletReconciler(client, _Breaker(), _Notifier(), managed=lambda: {CROSS})
    asyncio.run(r.check_once())                        # baseline: nothing held
    client.rows = [_row(CROSS, "No", 10.0)]
    assert asyncio.run(r.check_once()) is None


def test_an_unmanaged_move_still_alerts():
    """The exclusion must not switch off the leak check for everything else."""
    client = _Client([])
    r = WalletReconciler(client, _Breaker(), _Notifier(), managed=lambda: {CROSS})
    asyncio.run(r.check_once())
    client.rows = [_row("Some other market", "Yes", 3.0)]
    assert asyncio.run(r.check_once()) is not None


def test_without_a_hook_nothing_changes():
    client = _Client([])
    r = WalletReconciler(client, _Breaker(), _Notifier())
    asyncio.run(r.check_once())
    client.rows = [_row(CROSS, "No", 10.0)]
    assert asyncio.run(r.check_once()) is not None


def _redeemer(registry_ids, extra, redeemed=()):
    r = AutoRedeemer.__new__(AutoRedeemer)
    r._registry = type("R", (), {"condition_ids": set(registry_ids)})()
    r._extra_ids = extra
    r._redeemed = set(redeemed)
    return r


def test_the_redeemer_takes_the_cross_markets_in():
    r = _redeemer({"0xfeed"}, lambda: {"0xn", "0xb"})
    assert r._candidate_ids() == {"0xfeed", "0xn", "0xb"}


def test_already_redeemed_markets_stay_out():
    r = _redeemer({"0xfeed"}, lambda: {"0xn"}, redeemed={"0xn"})
    assert r._candidate_ids() == {"0xfeed"}


def test_a_failing_hook_does_not_stop_redemption():
    def boom():
        raise RuntimeError("book unreadable")
    assert _redeemer({"0xfeed"}, boom)._candidate_ids() == {"0xfeed"}
