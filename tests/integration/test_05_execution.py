"""Step 5 — PolyClient.execute_arb_pair fires both legs concurrently (paper mode)."""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_execute_arb_pair_returns_two_paper_fills(patched_clob):
    # Import after env vars are set & ClobClient is patched (fixture autouse).
    from core.clob_client import PolyClient

    client = PolyClient()
    t0 = time.monotonic()
    yes, no = await client.execute_arb_pair(
        yes_token_id="YES_TOK", yes_price=0.47, yes_size=10.0,
        no_token_id="NO_TOK",   no_price=0.50, no_size=10.0,
    )
    elapsed = time.monotonic() - t0

    assert yes["status"] == "paper"
    assert no["status"]  == "paper"
    assert yes["token_id"] == "YES_TOK"
    assert no["token_id"]  == "NO_TOK"
    assert yes["price"] == 0.47 and no["price"] == 0.50
    # Paper-mode "execution" must be near-instant (no thread-pool / network).
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_execute_arb_pair_paper_profit_is_combined_below_one(patched_clob):
    from core.clob_client import PolyClient

    client = PolyClient()
    yes, no = await client.execute_arb_pair(
        yes_token_id="Y", yes_price=0.40, yes_size=5.0,
        no_token_id="N",  no_price=0.45, no_size=5.0,
    )
    combined_cost = yes["cost_usdc"] + no["cost_usdc"]
    assert combined_cost == pytest.approx((0.40 + 0.45) * 5.0, rel=1e-6)
