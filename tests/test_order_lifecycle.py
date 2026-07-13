"""
tests/test_order_lifecycle.py
─────────────────────────────
Orphan-prevention invariants around order submission and tracking:

  - PolyClient._submit_pair keeps the surviving leg when one submission
    raises (a plain gather would discard it → orphaned live order)
  - InventoryManager.is_tracking exposes the settle window so the strategy
    loop can refuse re-entry on a busy market
"""

from __future__ import annotations

import asyncio

import pytest

import core.clob_client as cc


async def _ok(resp: dict) -> dict:
    return resp


async def _boom() -> dict:
    raise cc.ClobApiError(500, "submit failed")


def _bare_client() -> cc.PolyClient:
    return cc.PolyClient.__new__(cc.PolyClient)   # skip network __init__


@pytest.mark.asyncio
async def test_submit_pair_keeps_surviving_leg_on_single_failure():
    poly = _bare_client()
    yes, no = await poly._submit_pair(
        _ok({"status": "live", "order_id": "oy"}),
        _boom(),
    )
    assert yes == {"status": "live", "order_id": "oy"}
    assert no["status"] == "error"
    # classify_fills must treat the error leg as unfilled, not crash.
    assert cc.classify_fills(yes, no) == "none"


@pytest.mark.asyncio
async def test_submit_pair_filled_survivor_classifies_half_fill():
    poly = _bare_client()
    yes, no = await poly._submit_pair(
        _ok({"status": "matched", "order_id": "oy"}),
        _boom(),
    )
    assert cc.classify_fills(yes, no) == "yes_only"   # → unwind/guard path


@pytest.mark.asyncio
async def test_submit_pair_raises_only_when_both_legs_fail():
    poly = _bare_client()
    with pytest.raises(cc.ClobApiError):
        await poly._submit_pair(_boom(), _boom())


def test_inventory_is_tracking_lifecycle(patched_web3, fake_telegram):
    from execution.inventory_manager import InventoryManager, Position, LegFill
    import time

    inv = InventoryManager.__new__(InventoryManager)   # skip web3 __init__
    inv._positions = {}
    assert inv.is_tracking("0xabc") is False

    pos = Position(
        condition_id="0xabc", yes_token_id="y", no_token_id="n",
        yes_order_id="1", no_order_id="2", n_shares=1.0,
        submitted_at=time.monotonic(),
    )
    inv._positions["0xabc"] = pos
    assert inv.is_tracking("0xabc") is True      # PENDING
    pos.status = "SETTLING"
    assert inv.is_tracking("0xabc") is True
    pos.status = "SETTLED"
    assert inv.is_tracking("0xabc") is False
