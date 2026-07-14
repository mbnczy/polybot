"""Step 6 — InventoryManager.register_matched_pair triggers paper settlement."""

from __future__ import annotations

import asyncio

import pytest

from risk.circuit_breaker import CircuitBreaker
from strategy.arbitrage import ArbSignal


@pytest.mark.asyncio
async def test_register_matched_pair_settles_in_paper_mode(
    patched_web3, patched_clob, fake_telegram, tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "risk.circuit_breaker._DAILY_STATE_PATH",
        str(tmp_path / "daily.json"),
    )

    from core.clob_client          import PolyClient
    from execution.inventory_manager import InventoryManager

    client  = PolyClient()
    breaker = CircuitBreaker(starting_balance=500.0)
    invmgr  = InventoryManager(client, breaker, fake_telegram)

    sig = ArbSignal(
        condition_id="0x" + "ab" * 32,
        yes_token_id="Y", no_token_id="N",
        yes_ask=0.47, no_ask=0.50,
        combined_cost=0.97, fee_rate=0.0, fee_cost=0.0, net_edge=0.03,
        yes_size=10.0, no_size=10.0,
    )

    invmgr.register_matched_pair(sig, tx_hash="0x" + "de" * 32)
    # _settle is fire-and-forget — yield to let it run in paper mode.
    await asyncio.sleep(0.05)

    pos = invmgr._positions.get(sig.condition_id)
    # In paper mode `_on_settled` pops the position entry on success.
    assert pos is None
    # Booked profit ≈ shares × (1 − combined) = 10 × 0.03 = 0.3 USDC
    assert breaker._state.session_pnl == pytest.approx(0.3, rel=1e-6)


@pytest.mark.asyncio
async def test_merge_complementary_set_live_routes_through_sdk(
    monkeypatch, patched_web3, patched_clob, fake_telegram, fake_w3, tmp_path,
):
    """When paper mode is disabled the merge is routed through the V2 SDK
    (post-pUSD-migration contracts), not the legacy raw CTF call."""
    monkeypatch.setattr(
        "risk.circuit_breaker._DAILY_STATE_PATH",
        str(tmp_path / "daily.json"),
    )
    monkeypatch.setattr("execution.inventory_manager._PAPER_TRADE", False)
    monkeypatch.setattr("core.clob_client._PAPER_TRADE", False)

    from core.clob_client          import PolyClient
    from execution.inventory_manager import InventoryManager

    invmgr = InventoryManager(PolyClient(), CircuitBreaker(starting_balance=500.0), fake_telegram)
    cid = "0x" + "ab" * 32
    tx = await invmgr.merge_complementary_set(cid, n_shares=5.0)

    assert tx is not None
    # 5.0 shares × 1e6 base units, routed to SecureClient.merge_positions
    assert patched_clob.merges == [(cid, 5_000_000)]


@pytest.mark.asyncio
async def test_register_paired_fill_recycles_without_double_booking(
    patched_web3, patched_clob, fake_telegram, tmp_path, monkeypatch,
):
    """Binary FOK/maker path: settlement recycles capital but does NOT book P&L
    (the strategy loop already booked it at fill time)."""
    monkeypatch.setattr(
        "risk.circuit_breaker._DAILY_STATE_PATH", str(tmp_path / "daily.json"),
    )
    from prometheus_client import REGISTRY
    from core.clob_client            import PolyClient
    from execution.inventory_manager import InventoryManager

    client  = PolyClient()
    breaker = CircuitBreaker(starting_balance=500.0)
    invmgr  = InventoryManager(client, breaker, fake_telegram)

    sig = ArbSignal(
        condition_id="0x" + "cd" * 32,
        yes_token_id="Y", no_token_id="N",
        yes_ask=0.47, no_ask=0.50,
        combined_cost=0.97, fee_rate=0.0, fee_cost=0.0, net_edge=0.03,
        yes_size=10.0, no_size=10.0,
    )

    recycled_before = REGISTRY.get_sample_value("polly_capital_recycled_total") or 0.0
    invmgr.register_paired_fill(sig)
    await asyncio.sleep(0.05)   # let fire-and-forget _settle run (paper)

    # Position settled + popped …
    assert invmgr._positions.get(sig.condition_id) is None
    # … capital recycled metric incremented …
    recycled_after = REGISTRY.get_sample_value("polly_capital_recycled_total") or 0.0
    assert recycled_after - recycled_before == pytest.approx(1.0)
    # … but P&L was NOT booked here (no double-count).
    assert breaker._state.session_pnl == pytest.approx(0.0, abs=1e-9)
