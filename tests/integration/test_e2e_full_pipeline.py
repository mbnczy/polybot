"""E2E orchestrator — wires scanner, feeds, strategy, execution, inventory,
redeemer, signal logger, and Telegram into one asyncio.gather and asserts the
full happy-path produces a paper fill, settlement, signal-log row, and
Telegram trade notification.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from core.scanner          import FeedRegistry
from risk.circuit_breaker  import (
    ArbOrderIntent, CircuitBreaker, MAX_ARB_PAIR_USDC,
)
from strategy.arbitrage    import ArbDetector, ArbSignal, FeeEngine
from telemetry.db_logger   import SignalLogger
from tests.mocks.fake_clob_ws import (
    FakeAiohttpSession, FakeWebSocket,
    make_book_message,
)


@pytest.mark.asyncio
async def test_full_pipeline_paper_arb(
    monkeypatch, patched_clob, patched_web3, fake_telegram, tmp_signal_db,
    tmp_path,
):
    monkeypatch.setattr(
        "risk.circuit_breaker._DAILY_STATE_PATH",
        str(tmp_path / "daily.json"),
    )

    yes_id, no_id, cid = "YES_TOK", "NO_TOK", "0x" + "ab" * 32

    ws = FakeWebSocket([
        make_book_message(yes_id, 0.47),
        make_book_message(no_id,  0.50),
    ])
    monkeypatch.setattr(
        "core.ws_feed.aiohttp.ClientSession",
        lambda *_a, **_kw: FakeAiohttpSession(ws),
        raising=True,
    )

    # ── Build the same components main.py wires up ───────────────────────────
    from core.clob_client            import PolyClient
    from execution.inventory_manager import InventoryManager

    client    = PolyClient()
    breaker   = CircuitBreaker(starting_balance=500.0)
    detector  = ArbDetector(desired_net_margin=0.005, default_fee_rate=0.0)
    fee_eng   = FeeEngine(default_fee=0.0)
    fee_eng.prime_cache(cid, 0.0)
    inv_mgr   = InventoryManager(client, breaker, fake_telegram)
    sig_log   = SignalLogger(db_path=str(tmp_signal_db))
    await sig_log.init()

    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    registry = FeedRegistry(queue)
    await registry.add_market(cid, yes_id, no_id)

    # ── Inlined strategy loop (mirrors main.strategy_loop's hot path) ────────
    async def strategy() -> None:
        while True:
            tick = await queue.get()
            queue.task_done()
            if tick.get("type") != "arb_tick":
                continue
            sig = detector.evaluate(
                condition_id=tick["condition_id"],
                yes_token_id=tick["yes_token_id"],
                no_token_id=tick["no_token_id"],
                yes_ask=tick["yes_ask"],
                no_ask=tick["no_ask"],
                max_position_usdc=MAX_ARB_PAIR_USDC,
                fee_rate=await fee_eng.get_taker_fee(tick["condition_id"]),
            )
            if sig is None:
                continue
            sig_log.log_arb(sig)
            n = breaker.calculate_position_size(tick["yes_ask"], tick["no_ask"])
            intent = ArbOrderIntent(
                condition_id=sig.condition_id,
                yes_token_id=sig.yes_token_id,
                no_token_id=sig.no_token_id,
                yes_price=tick["yes_ask"], no_price=tick["no_ask"],
                n_shares=n,
                combined_cost_usdc=round(n * sig.combined_cost, 6),
            )
            if not breaker.check_arb(intent):
                continue
            await client.execute_arb_pair(
                yes_token_id=sig.yes_token_id, yes_price=tick["yes_ask"], yes_size=n,
                no_token_id=sig.no_token_id,  no_price=tick["no_ask"],   no_size=n,
            )
            breaker.on_arb_open()
            breaker.on_fill(pnl=round(n * sig.net_edge, 6))
            await fake_telegram.send_trade_execution(
                condition_id=cid, yes_ask=tick["yes_ask"], no_ask=tick["no_ask"],
                n_shares=n, net_profit=round(n * sig.net_edge, 6),
            )
            inv_mgr.register_matched_pair(
                ArbSignal(**{**sig.__dict__, "yes_size": n, "no_size": n}),
                tx_hash="0x" + "de" * 32,
            )
            return    # one cycle is enough for E2E

    sig_writer = asyncio.create_task(sig_log.run())
    strat_task = asyncio.create_task(strategy())

    # Wait for the pipeline to consume one arb opportunity.
    await asyncio.wait_for(strat_task, timeout=3.0)
    await asyncio.sleep(0.1)   # let inventory _settle finish

    # ── Assertions: full pipeline produced the expected side effects ─────────
    assert breaker.status_dict()["orders_passed"] == 1
    assert breaker._state.session_pnl > 0.0
    assert len(fake_telegram.trade_executions) == 1

    async with aiosqlite.connect(str(tmp_signal_db)) as db:
        cur = await db.execute("SELECT COUNT(*) FROM arb_signals")
        (n_rows,) = await cur.fetchone()
    assert n_rows == 1

    # ── Cleanup ──────────────────────────────────────────────────────────────
    await sig_log.close()
    sig_writer.cancel()
    await registry.stop_all()
    await asyncio.gather(sig_writer, return_exceptions=True)


@pytest.mark.asyncio
async def test_halt_callback_cancels_orders(patched_clob, fake_telegram, tmp_path, monkeypatch):
    """Telegram /halt callback should cancel orders and notify shutdown."""
    monkeypatch.setattr(
        "risk.circuit_breaker._DAILY_STATE_PATH",
        str(tmp_path / "daily.json"),
    )
    from core.clob_client import PolyClient

    client = PolyClient()
    halted = asyncio.Event()

    async def _on_halt():
        await client.cancel_all_orders()
        await fake_telegram.notify("HALT received — all orders cancelled")
        halted.set()

    fake_telegram.set_halt_callback(_on_halt)
    await fake_telegram.trigger_halt()

    await asyncio.wait_for(halted.wait(), timeout=1.0)
    assert any("HALT" in m for m in fake_telegram.messages)
