"""
A violation must survive the asks actually payable before it is alerted.

The first live pass after the exit-tracking branch went up sent "+200 bps" on
"Moik Baku O/U 0.5" under "O/U 0.5" to Telegram — narrow YES at 0.98, so the
detector priced NO at 0.02 — while the only NO ask on the book was 0.83 and the
real entry was 1.79. The tracker refused it in the same pass; the alert had
already gone. These tests pin that the check now runs first.
"""

from __future__ import annotations

import json

import pytest

import scripts.demo_cross_market as reader
from strategy.cross_market import CrossMarketSignal


def _mkt(cid, yes_tok, no_tok):
    return {"conditionId": cid, "clobTokenIds": json.dumps([yes_tok, no_tok])}


MARKETS = [_mkt("0xnarrow", "ny", "nn"), _mkt("0xbroad", "by", "bn")]


def _sig(edge=0.02):
    return CrossMarketSignal(
        narrow="0xnarrow", broad="0xbroad",
        narrow_title="Moik Baku vs. Simal: Moik Baku O/U 0.5",
        broad_title="Moik Baku vs. Simal: O/U 0.5",
        narrow_price=0.98, broad_price=0.96, violation=0.02, cost=0.98,
        min_payout=1.0, edge=edge, confidence=0.97, evidence="test",
    )


def _books(monkeypatch, asks):
    """asks: token -> ask price (None = empty side)."""
    monkeypatch.setattr(reader, "book_top", lambda tok: (None, asks.get(tok)))


def test_the_live_moik_baku_alert_is_suppressed(monkeypatch):
    """NO ask 0.83 against YES 0.96: the entry is 1.79, not 0.98."""
    _books(monkeypatch, {"nn": 0.83, "by": 0.96})
    real, illusory = reader.confirm_at_asks([_sig()], MARKETS)
    assert real == [] and len(illusory) == 1


def test_a_violation_that_survives_the_asks_is_alerted(monkeypatch):
    _books(monkeypatch, {"nn": 0.30, "by": 0.40})
    real, illusory = reader.confirm_at_asks([_sig()], MARKETS)
    assert len(real) == 1 and illusory == []


def test_fees_are_part_of_the_test(monkeypatch):
    """0.50 + 0.495 is under 1.00 on the ask alone and over it with fees."""
    _books(monkeypatch, {"nn": 0.50, "by": 0.495})
    real, illusory = reader.confirm_at_asks([_sig()], MARKETS)
    assert real == []


def test_a_leg_with_no_ask_is_not_an_opportunity(monkeypatch):
    """Nobody to buy it from."""
    _books(monkeypatch, {"nn": None, "by": 0.40})
    real, illusory = reader.confirm_at_asks([_sig()], MARKETS)
    assert real == [] and len(illusory) == 1


def test_unknown_markets_keep_the_old_behaviour(monkeypatch):
    """Suppressing on missing data would hide more than it saves."""
    _books(monkeypatch, {})
    real, illusory = reader.confirm_at_asks([_sig()], [])
    assert len(real) == 1
