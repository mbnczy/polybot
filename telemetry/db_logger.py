"""
telemetry/db_logger.py
──────────────────────
Async, non-blocking SQLite signal logger — zero hot-path latency edition.

Design
──────
• SQLite in WAL (Write-Ahead Logging) mode: readers never block writers and
  vice-versa — critical so the 500 ms strategy_loop is never stalled by a
  disk flush.
• Every `log_arb()` / `log_neg_risk()` call schedules an asyncio background
  Task that does the actual INSERT.  The caller returns immediately (fire and
  forget), adding zero latency to the hot path.
• A bounded asyncio.Queue feeds a single writer coroutine, preventing
  unbounded memory growth if disk I/O falls behind.
• The writer coroutine runs inside the event loop and uses `aiosqlite` — no
  thread-pool dispatch needed for WAL-mode writes.

Tables
──────
arb_signals
    id, ts_iso, condition_id, yes_token_id, no_token_id,
    yes_ask, no_ask, combined_cost, fee_rate, fee_cost, net_edge, created_at

neg_risk_signals
    id, ts_iso, condition_id, n_outcomes, combined_bid, payout,
    maker_rebate, effective_cost, net_edge, relative_edge, n_bundles, created_at

Usage
─────
    logger = SignalLogger("signals.db")
    await logger.init()                  # call once at startup
    asyncio.create_task(logger.run())    # long-running writer coroutine

    # From strategy_loop hot path (non-blocking):
    logger.log_arb(arb_signal)
    logger.log_neg_risk(neg_risk_signal)

    await logger.close()                 # drain queue, close DB

Environment
───────────
    SIGNAL_DB_PATH   path to the SQLite file (default: "signals.db")
    SIGNAL_DB_MAXQ   max pending writes before dropping (default: 4096)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from strategy.arbitrage import ArbSignal, NegRiskSignal

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_DB_PATH: str = os.environ.get("SIGNAL_DB_PATH", "signals.db")
_MAXQ:    int = int(os.environ.get("SIGNAL_DB_MAXQ", 4096))

# Sentinel to drain the queue on shutdown
_STOP = object()


# ═════════════════════════════════════════════════════════════════════════════
# SignalLogger
# ═════════════════════════════════════════════════════════════════════════════

class SignalLogger:
    """
    Fire-and-forget signal persistence.

    Call `log_arb()` or `log_neg_risk()` from the hot path — both return
    immediately.  The actual INSERT runs in the background writer coroutine
    (`run()`), which must be started as an asyncio Task before any logging.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file.  Created if it doesn't exist.
    maxq : int
        Maximum pending write records.  Entries are dropped (with a warning)
        when the queue is full to protect the event loop.
    """

    def __init__(
        self,
        db_path: str = _DB_PATH,
        maxq:    int = _MAXQ,
    ) -> None:
        self._db_path = db_path
        self._queue:  asyncio.Queue = asyncio.Queue(maxsize=maxq)
        self._db:     aiosqlite.Connection | None = None
        self._closed: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Startup / teardown
    # ──────────────────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """
        Open the database and create tables + WAL pragma.

        Must be awaited once before `run()` is started.
        """
        self._db = await aiosqlite.connect(self._db_path)

        # WAL mode: writers don't block readers; readers don't block writers.
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")   # safe with WAL

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS arb_signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_iso        TEXT    NOT NULL,
                condition_id  TEXT    NOT NULL,
                yes_token_id  TEXT    NOT NULL,
                no_token_id   TEXT    NOT NULL,
                yes_ask       REAL    NOT NULL,
                no_ask        REAL    NOT NULL,
                combined_cost REAL    NOT NULL,
                fee_rate      REAL    NOT NULL,
                fee_cost      REAL    NOT NULL,
                net_edge      REAL    NOT NULL,
                created_at    REAL    NOT NULL   -- time.time() float
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS neg_risk_signals (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_iso         TEXT    NOT NULL,
                condition_id   TEXT    NOT NULL,
                n_outcomes     INTEGER NOT NULL,
                combined_bid   REAL    NOT NULL,
                payout         REAL    NOT NULL,
                maker_rebate   REAL    NOT NULL,
                effective_cost REAL    NOT NULL,
                net_edge       REAL    NOT NULL,
                relative_edge  REAL    NOT NULL,
                n_bundles      INTEGER NOT NULL,
                created_at     REAL    NOT NULL
            )
        """)

        await self._db.commit()
        logger.info(
            "SignalLogger initialised | db=%s WAL=on maxq=%d",
            self._db_path, _MAXQ,
        )

    async def close(self) -> None:
        """
        Drain the queue and close the database connection.

        Sends a stop sentinel and waits for the writer to finish all pending
        records before closing the connection.
        """
        if self._closed:
            return
        self._closed = True
        await self._queue.put(_STOP)
        # Give the writer time to drain
        await asyncio.sleep(0.1)
        if self._db is not None:
            await self._db.close()
            logger.info("SignalLogger closed — db=%s", self._db_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Hot-path entry points (non-blocking, fire-and-forget)
    # ──────────────────────────────────────────────────────────────────────────

    def log_arb(self, signal: "ArbSignal") -> None:
        """
        Enqueue an ArbSignal for background persistence.

        Returns immediately — never blocks the caller.
        Drops silently (with a warning) when the queue is full.
        """
        now = time.time()
        row = (
            "arb",
            datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            signal.condition_id,
            signal.yes_token_id,
            signal.no_token_id,
            signal.yes_ask,
            signal.no_ask,
            signal.combined_cost,
            signal.fee_rate,
            signal.fee_cost,
            signal.net_edge,
            now,
        )
        self._enqueue(row)

    def log_neg_risk(self, signal: "NegRiskSignal") -> None:
        """
        Enqueue a NegRiskSignal for background persistence.

        Returns immediately — never blocks the caller.
        Drops silently (with a warning) when the queue is full.
        """
        now = time.time()
        row = (
            "neg_risk",
            datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            signal.condition_id,
            signal.n_outcomes,
            signal.combined_bid,
            signal.payout,
            signal.maker_rebate,
            signal.effective_cost,
            signal.net_edge,
            signal.relative_edge,
            signal.n_bundles,
            now,
        )
        self._enqueue(row)

    def _enqueue(self, row: tuple) -> None:
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            logger.warning(
                "SignalLogger queue full (%d) — dropping signal record",
                _MAXQ,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Background writer coroutine
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Long-running writer coroutine.  Start as an asyncio.Task.

        Consumes rows from the queue and INSERTs them into SQLite using
        WAL-mode non-blocking writes.  Commits in micro-batches (up to 32
        rows per commit) to amortise fsync overhead without losing data.
        """
        logger.info("SignalLogger writer started")
        _BATCH = 32
        pending: list[tuple] = []

        async def _flush() -> None:
            if not pending or self._db is None:
                return
            arb_rows      = [r for r in pending if r[0] == "arb"]
            neg_risk_rows = [r for r in pending if r[0] == "neg_risk"]

            if arb_rows:
                await self._db.executemany(
                    """INSERT INTO arb_signals
                       (ts_iso, condition_id, yes_token_id, no_token_id,
                        yes_ask, no_ask, combined_cost, fee_rate, fee_cost,
                        net_edge, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    [r[1:] for r in arb_rows],
                )
            if neg_risk_rows:
                await self._db.executemany(
                    """INSERT INTO neg_risk_signals
                       (ts_iso, condition_id, n_outcomes, combined_bid, payout,
                        maker_rebate, effective_cost, net_edge, relative_edge,
                        n_bundles, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    [r[1:] for r in neg_risk_rows],
                )
            await self._db.commit()
            pending.clear()

        try:
            while True:
                row = await self._queue.get()

                if row is _STOP:
                    await _flush()
                    logger.info("SignalLogger writer drained — stopping")
                    return

                pending.append(row)

                # Drain any immediately available rows (up to batch limit)
                while len(pending) < _BATCH:
                    try:
                        nxt = self._queue.get_nowait()
                        if nxt is _STOP:
                            await _flush()
                            return
                        pending.append(nxt)
                    except asyncio.QueueEmpty:
                        break

                await _flush()

        except asyncio.CancelledError:
            # Best-effort flush on cancellation
            try:
                await _flush()
            except Exception:  # noqa: BLE001
                pass
            logger.info("SignalLogger writer cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("SignalLogger writer error: %s", exc)
            raise
