"""
strategy/tuner.py
──────────────────
Dynamic Margin Optimizer — background asyncio task.

Runs every TUNE_INTERVAL seconds (default 3600 = 60 min).  On each wake-up it
opens a non-blocking aiosqlite read connection to the WAL-mode signals database
and counts the ArbSignal + NegRiskSignal rows logged in the preceding hour.
It then adjusts `ArbDetector._net_margin` in-place according to the following
rules:

  High frequency (≥ FREQ_HIGH_THRESHOLD signals in the hour) AND
  wide mean spread (mean net_edge ≥ WIDE_EDGE_THRESHOLD):
      margin ← min(margin + STEP_UP, MAX_MARGIN)
      Rationale: profitable opportunities are abundant; demand more value per
                 trade to improve quality over quantity.

  Zero frequency (0 signals in the entire hour):
      margin ← max(margin − STEP_DOWN, MIN_ABSOLUTE_MARGIN)
      Rationale: the bot is being out-competed; loosen the filter to regain
                 order flow.

  Else (moderate activity or narrow spreads):
      No adjustment.

SQLite access
─────────────
A fresh aiosqlite connection is opened and closed on every tuning cycle.
WAL mode allows concurrent readers alongside the SignalLogger writer; reads
are non-blocking and never stall the hot path.

Constants
─────────
  MIN_ABSOLUTE_MARGIN   0.005  (0.50 %) — hard floor; margin never falls below
  MAX_MARGIN            0.050  (5.00 %) — hard ceiling
  STEP_UP               0.001  (0.10 %) — per-cycle increase
  STEP_DOWN             0.001  (0.10 %) — per-cycle decrease
  FREQ_HIGH_THRESHOLD   10     signals / hour → "high frequency"
  WIDE_EDGE_THRESHOLD   0.010  (1.00 %) mean net_edge → "wide spread"
  TUNE_INTERVAL         3600.0 seconds between tuning runs (60 min)

Environment
───────────
  SIGNAL_DB_PATH   path to the SQLite signals file (default: "signals.db")
                   Must match the value used by telemetry/db_logger.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from strategy.arbitrage import ArbDetector

logger = logging.getLogger(__name__)

# ── Tuning constants (public — importable for tests / overrides) ──────────────
MIN_ABSOLUTE_MARGIN: float = 0.005   # 0.50 % — hard floor; never lowered past this
MAX_MARGIN:          float = 0.050   # 5.00 % — hard ceiling
STEP_UP:             float = 0.001   # 0.10 % per cycle
STEP_DOWN:           float = 0.001   # 0.10 % per cycle
FREQ_HIGH_THRESHOLD: int   = 10      # signals/hour → "high frequency"
WIDE_EDGE_THRESHOLD: float = 0.010   # mean net_edge ≥ this → "wide spread"
TUNE_INTERVAL:       float = 3600.0  # seconds (60 min)

_DB_PATH: str = os.environ.get("SIGNAL_DB_PATH", "signals.db")


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _query_last_hour(db_path: str, since: float) -> tuple[int, float]:
    """
    Return (signal_count, mean_net_edge) for all signals with created_at ≥ since.

    Opens a fresh read connection (WAL-compatible; never blocks the writer).
    Returns (0, 0.0) when the database does not yet exist or contains no rows.
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            arb_cursor = await db.execute(
                "SELECT COUNT(*), AVG(net_edge) FROM arb_signals "
                "WHERE created_at >= ?",
                (since,),
            )
            arb_row = await arb_cursor.fetchone()

            neg_cursor = await db.execute(
                "SELECT COUNT(*), AVG(net_edge) FROM neg_risk_signals "
                "WHERE created_at >= ?",
                (since,),
            )
            neg_row = await neg_cursor.fetchone()

    except Exception as exc:  # noqa: BLE001
        logger.warning("tuner: db query failed — %s", exc)
        return 0, 0.0

    arb_count  = arb_row[0] if arb_row and arb_row[0] else 0
    neg_count  = neg_row[0] if neg_row and neg_row[0] else 0
    total      = arb_count + neg_count

    if total == 0:
        return 0, 0.0

    # Weighted mean net_edge across both signal types
    arb_sum = (arb_row[1] or 0.0) * arb_count
    neg_sum = (neg_row[1] or 0.0) * neg_count
    mean_edge = (arb_sum + neg_sum) / total
    return total, mean_edge


# ── Background coroutine ──────────────────────────────────────────────────────

async def tuner_loop(
    detector:  "ArbDetector",
    db_path:   str = _DB_PATH,
    interval:  float = TUNE_INTERVAL,
) -> None:
    """
    Long-running asyncio coroutine — start as a named Task.

    Adjusts `detector._net_margin` in-place every `interval` seconds based on
    the signal history in the WAL-mode SQLite database at `db_path`.

    Parameters
    ----------
    detector : ArbDetector
        Live detector instance whose _net_margin will be tuned.
    db_path : str
        Path to the aiosqlite signals database (default: SIGNAL_DB_PATH env var).
    interval : float
        Seconds between tuning cycles (default: TUNE_INTERVAL = 3600).
    """
    logger.info("tuner_loop started | interval=%.0fs db=%s", interval, db_path)
    try:
        while True:
            await asyncio.sleep(interval)

            since             = time.time() - interval
            count, mean_edge  = await _query_last_hour(db_path, since)
            current           = detector._net_margin

            if count == 0:
                # Zero flow — we are being out-competed; relax the filter
                new_margin = max(current - STEP_DOWN, MIN_ABSOLUTE_MARGIN)
                action     = "decrease"
            elif count >= FREQ_HIGH_THRESHOLD and mean_edge >= WIDE_EDGE_THRESHOLD:
                # Abundant, high-value opportunities — demand more per trade
                new_margin = min(current + STEP_UP, MAX_MARGIN)
                action     = "increase"
            else:
                new_margin = current
                action     = "hold"

            if new_margin != current:
                detector._net_margin = new_margin
                logger.info(
                    "tuner | %s margin %.4f → %.4f "
                    "(signals=%d mean_edge=%.4f)",
                    action, current, new_margin, count, mean_edge,
                )
            else:
                logger.debug(
                    "tuner | %s margin=%.4f "
                    "(signals=%d mean_edge=%.4f)",
                    action, current, count, mean_edge,
                )

    except asyncio.CancelledError:
        logger.info("tuner_loop stopped")
        raise
