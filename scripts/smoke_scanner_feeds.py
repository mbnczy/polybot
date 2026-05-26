"""
scripts/smoke_scanner_feeds.py
──────────────────────────────
Smoke-test: env → config → MarketScanner wiring for MAX_FEEDS.

Verifies that the MAX_FEEDS env var is read correctly and that
MarketScanner.__init__ and run() log the expected lines.

Run:
    MAX_FEEDS=100 python scripts/smoke_scanner_feeds.py   # limited path
    MAX_FEEDS=0   python scripts/smoke_scanner_feeds.py   # unlimited path

The script performs ONE real Gamma API scan cycle then exits cleanly.
No CLOB credentials or Telegram token required.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("smoke")

# Import after env is loaded so strategy module reads correct defaults.
from core.scanner import FeedRegistry, MarketScanner  # noqa: E402


async def _noop_callback(condition_id: str, yes_id: str, no_id: str) -> None:
    """Drop feeds — we only want to observe the scanner logging."""
    pass


async def run_smoke() -> None:
    max_feeds = int(os.environ.get("MAX_FEEDS", 50))
    logger.info("=== smoke_scanner_feeds | MAX_FEEDS=%d ===", max_feeds)

    queue = asyncio.Queue(maxsize=2048)
    # FeedRegistry not used; we pass a no-op callback directly.
    scanner = MarketScanner(
        on_market_added=_noop_callback,
        scan_interval=9999,      # no second scan during the test
        max_feeds=max_feeds,
    )

    # Run exactly one scan cycle, then cancel.
    task = asyncio.create_task(scanner.run())
    # Allow enough time for the scan cycle to complete (Gamma API latency ≤ 30 s).
    await asyncio.sleep(35)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    logger.info("=== smoke complete ===")


if __name__ == "__main__":
    asyncio.run(run_smoke())
