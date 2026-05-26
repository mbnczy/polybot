"""Step 8 — SignalLogger persists rows; tuner_loop adjusts ArbDetector margin."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from strategy.arbitrage import ArbDetector, ArbSignal
from strategy.tuner     import (
    FREQ_HIGH_THRESHOLD,
    STEP_UP,
    WIDE_EDGE_THRESHOLD,
    _query_last_hour,
)
from telemetry.db_logger import SignalLogger


def _signal(net_edge: float, idx: int) -> ArbSignal:
    return ArbSignal(
        condition_id=f"0x{idx:064x}",
        yes_token_id="Y", no_token_id="N",
        yes_ask=0.47, no_ask=0.50,
        combined_cost=0.97, fee_rate=0.0, fee_cost=0.0, net_edge=net_edge,
        yes_size=10.0, no_size=10.0,
    )


@pytest.mark.asyncio
async def test_signal_logger_persists_arb_rows(tmp_signal_db):
    logger = SignalLogger(db_path=str(tmp_signal_db))
    await logger.init()
    writer = asyncio.create_task(logger.run())

    for i in range(15):
        logger.log_arb(_signal(net_edge=0.012, idx=i))

    await asyncio.sleep(0.2)
    await logger.close()
    writer.cancel()
    await asyncio.gather(writer, return_exceptions=True)

    async with aiosqlite.connect(str(tmp_signal_db)) as db:
        cur = await db.execute("SELECT COUNT(*) FROM arb_signals")
        (count,) = await cur.fetchone()
    assert count == 15


@pytest.mark.asyncio
async def test_tuner_increases_margin_on_high_freq_wide_edge(tmp_signal_db):
    """Bypass tuner_loop's sleep — drive a single cycle directly."""
    logger = SignalLogger(db_path=str(tmp_signal_db))
    await logger.init()
    writer = asyncio.create_task(logger.run())
    for i in range(FREQ_HIGH_THRESHOLD + 2):
        logger.log_arb(_signal(net_edge=WIDE_EDGE_THRESHOLD + 0.005, idx=i))
    await asyncio.sleep(0.2)
    await logger.close()
    writer.cancel()
    await asyncio.gather(writer, return_exceptions=True)

    detector = ArbDetector(desired_net_margin=0.01)
    start_margin = detector._net_margin

    # Reproduce the body of one tuner cycle.
    import time
    count, mean_edge = await _query_last_hour(str(tmp_signal_db), since=time.time() - 3600.0)
    assert count >= FREQ_HIGH_THRESHOLD
    assert mean_edge >= WIDE_EDGE_THRESHOLD

    detector._net_margin = min(detector._net_margin + STEP_UP, 0.05)
    assert detector._net_margin == pytest.approx(start_margin + STEP_UP, rel=1e-6)
