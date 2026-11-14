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


# ── order_filled_size by market ───────────────────────────────────────────────

def test_a_market_lookup_finds_a_maker_fill_filed_under_the_other_token(monkeypatch):
    """Our resting BUY of YES matched a taker buying NO: the exchange minted the
    pair and filed the trade under the taker's token. A token filter on YES
    cannot see it; the market filter holds both."""
    import asyncio
    from types import SimpleNamespace as NS
    c = _bare_client()
    c._touch_order = lambda *a, **k: None
    asked = []
    trade = NS(status="CONFIRMED", taker_order_id="0xtaker", size="5.25", asset_id="NO",
               maker_orders=[NS(order_id="0xOURS", matched_amount="4.99"),
                             NS(order_id="0xother", matched_amount="1.00")])

    def list_account_trades(**kw):
        asked.append(kw)
        items = [trade] if kw.get("market") == "0xcond" or kw.get("token_id") == "NO" else []
        return [NS(items=items)]
    c._client = NS(list_account_trades=list_account_trades)
    monkeypatch.setattr(cc, "_PAPER_TRADE", False)
    by_token = asyncio.run(c.order_filled_size("0xours", "YES"))
    by_market = asyncio.run(c.order_filled_size("0xours", "YES", market="0xcond"))
    assert by_token == 0.0 and by_market == 4.99
    assert asked == [{"token_id": "YES"}, {"market": "0xcond"}]
