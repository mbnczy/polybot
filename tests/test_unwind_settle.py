"""
A half-fill's unwind must wait for the fill to reach the chain, and a killed FOK
must not be re-sent. Both failed together on 2026-09-11 and stranded 10.7 shares.
"""

from __future__ import annotations

import asyncio

import core.clob_client as cc
from core.clob_client import PolyClient, _terminal_order_error


def _client(balances):
    c = PolyClient.__new__(PolyClient)
    seq = iter(balances)
    last = [None]

    async def share_balance(_token):
        last[0] = next(seq, last[0])
        return last[0]

    c.share_balance = share_balance
    return c


def test_the_unwind_waits_for_the_fill_to_settle(monkeypatch):
    monkeypatch.setattr(cc, "_UNWIND_SETTLE_POLL_S", 0.0)
    c = _client([0.0, 0.0, 10.7])
    assert asyncio.run(c._settled_balance("tok", 10.7)) == 10.7


def test_a_short_wallet_still_caps_after_the_window(monkeypatch):
    """The 2026-09-08 case: the guard says 10.30, the chain holds 10.00."""
    monkeypatch.setattr(cc, "_UNWIND_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(cc, "_UNWIND_SETTLE_S", 0.01)
    c = _client([10.0])
    assert asyncio.run(c._settled_balance("tok", 10.3)) == 10.0


def test_a_failed_balance_read_is_not_waited_on():
    c = _client([None])
    assert asyncio.run(c._settled_balance("tok", 10.7)) is None


def test_a_killed_fok_is_terminal():
    exc = Exception("order couldn't be fully filled. FOK orders are fully filled or killed.")
    assert _terminal_order_error(exc) == "FOK not filled"


def test_a_transport_fault_is_still_retried():
    assert _terminal_order_error(Exception("Connection reset by peer")) is None


def test_a_missing_orderbook_is_terminal():
    """136 of these burned five retries each on 2026-09-12."""
    exc = Exception("No orderbook exists for the requested token id")
    assert _terminal_order_error(exc) == "no orderbook for token"
