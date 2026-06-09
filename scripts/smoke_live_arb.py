"""
scripts/smoke_live_arb.py
─────────────────────────
LIVE smoke test: does the bot actually connect to Polymarket and actively
search for arbitrage on real markets?

Exercises the exact live data + detection path:

    MarketScanner (Gamma API)
        → FeedRegistry.add_market
        → MarketFeed  (real wss://ws-subscriptions-clob.polymarket.com/ws/market)
        → arb_tick queue
        → ArbDetector.evaluate / DutchBookPricer.evaluate_maker   ← identical to main.py

It deliberately omits the execution layer (PolyClient) and Telegram, so NO
credentials are required — only public endpoints are touched. Nothing is traded.

Reports, after the run window:
  • markets discovered & feeds connected
  • live arb_ticks received (and how many distinct markets quoted)
  • signals detected (taker + maker paths)
  • tightest combined cost (yes_ask+no_ask) observed — the real-market spread

Usage:
    python scripts/smoke_live_arb.py                 # 90 s, 60 feeds
    DURATION=120 MAX_FEEDS=80 python scripts/smoke_live_arb.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("smoke_live")

# Import after env load (modules read env defaults at import).
from core.scanner import FeedRegistry, MarketScanner          # noqa: E402
from strategy.arbitrage import (                              # noqa: E402
    ArbDetector,
    DutchBookPricer,
    DESIRED_NET_MARGIN,
    DEFAULT_MAKER_REBATE,
)

_DURATION    = float(os.environ.get("DURATION", "90"))
_MAX_FEEDS   = int(os.environ.get("MAX_FEEDS", "60"))
_DEFAULT_FEE = float(os.environ.get("DEFAULT_TAKER_FEE", "0.02"))
_PAIR_CAP    = 50.0


async def run() -> None:
    logger.info(
        "=== LIVE arb smoke | duration=%.0fs max_feeds=%d net_margin=%.4f ===",
        _DURATION, _MAX_FEEDS, DESIRED_NET_MARGIN,
    )

    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=4096)
    registry = FeedRegistry(queue=queue)
    scanner = MarketScanner(
        on_market_added=registry.add_market,
        scan_interval=9999,            # single scan during the window
        max_feeds=_MAX_FEEDS,
    )
    detector = ArbDetector(
        desired_net_margin=DESIRED_NET_MARGIN,
        default_fee_rate=_DEFAULT_FEE,
    )
    dutch = DutchBookPricer(desired_net_margin=DESIRED_NET_MARGIN)

    # ── Counters ──────────────────────────────────────────────────────────────
    stats = {
        "ticks": 0,
        "two_sided": 0,
        "taker_signals": 0,
        "maker_signals": 0,
        "min_combined": 99.0,
        "min_combined_market": "",
    }
    quoting_markets: set[str] = set()
    samples: list[tuple[str, float, float]] = []

    scan_task = asyncio.create_task(scanner.run(), name="scanner")

    async def consume() -> None:
        while True:
            tick = await queue.get()
            queue.task_done()
            if not isinstance(tick, dict):
                continue
            stats["ticks"] += 1

            cid     = tick.get("condition_id", "")
            yes_id  = tick.get("yes_token_id", "")
            no_id   = tick.get("no_token_id", "")
            yes_ask = tick.get("yes_ask")
            no_ask  = tick.get("no_ask")
            quoting_markets.add(cid)

            if yes_ask is None or no_ask is None:
                continue
            stats["two_sided"] += 1

            combined = yes_ask + no_ask
            if combined < stats["min_combined"]:
                stats["min_combined"] = combined
                stats["min_combined_market"] = cid
            if len(samples) < 8:
                samples.append((cid, yes_ask, no_ask))

            # ── identical detection logic to main.py strategy_loop ──
            sig = detector.evaluate(
                condition_id=cid,
                yes_token_id=yes_id,
                no_token_id=no_id,
                yes_ask=yes_ask,
                no_ask=no_ask,
                max_position_usdc=_PAIR_CAP,
                fee_rate=_DEFAULT_FEE,
            )
            if sig is not None:
                stats["taker_signals"] += 1
                logger.info(
                    "TAKER ARB | %s yes=%.4f no=%.4f edge=%.1f bps",
                    cid[:18], yes_ask, no_ask, sig.net_edge * 10_000,
                )
                continue

            msig = dutch.evaluate_maker(
                condition_id=cid,
                yes_token_id=yes_id,
                no_token_id=no_id,
                yes_ask=yes_ask,
                no_ask=no_ask,
                max_position_usdc=_PAIR_CAP,
                maker_rebate=DEFAULT_MAKER_REBATE,
            )
            if msig is not None:
                stats["maker_signals"] += 1

    consume_task = asyncio.create_task(consume(), name="consumer")

    # ── Run window ──────────────────────────────────────────────────────────
    t0 = time.monotonic()
    last_log = 0.0
    while time.monotonic() - t0 < _DURATION:
        await asyncio.sleep(2)
        elapsed = time.monotonic() - t0
        if elapsed - last_log >= 15:
            last_log = elapsed
            logger.info(
                "  [%3.0fs] feeds=%d ticks=%d two_sided=%d quoting=%d "
                "taker=%d maker=%d min_combined=%.4f",
                elapsed, registry.active_count, stats["ticks"],
                stats["two_sided"], len(quoting_markets),
                stats["taker_signals"], stats["maker_signals"],
                stats["min_combined"],
            )

    # ── Teardown ──────────────────────────────────────────────────────────────
    scan_task.cancel()
    consume_task.cancel()
    for t in (scan_task, consume_task):
        try:
            await t
        except asyncio.CancelledError:
            pass
    await registry.stop_all()

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  LIVE ARB SMOKE — RESULT")
    print("═" * 64)
    print(f"  Run window            : {_DURATION:.0f} s")
    print(f"  Feeds connected       : {registry.active_count}")
    print(f"  Markets that quoted   : {len(quoting_markets)}")
    print(f"  arb_ticks received    : {stats['ticks']}")
    print(f"  two-sided ticks       : {stats['two_sided']}")
    print(f"  TAKER signals         : {stats['taker_signals']}")
    print(f"  MAKER signals         : {stats['maker_signals']}")
    print(f"  tightest combined cost: {stats['min_combined']:.4f} "
          f"({stats['min_combined_market'][:18]})")
    print("─" * 64)
    if samples:
        print("  sample live quotes (yes_ask / no_ask / sum):")
        for cid, ya, na in samples:
            print(f"    {cid[:20]:<22} {ya:.4f} / {na:.4f} / {ya+na:.4f}")
    print("─" * 64)

    # ── Verdict ────────────────────────────────────────────────────────────────
    if stats["ticks"] == 0:
        print("  VERDICT: NO LIVE DATA — WS feeds produced no ticks (check network)")
    elif stats["two_sided"] == 0:
        print("  VERDICT: CONNECTED but no two-sided quotes in window")
    else:
        print("  VERDICT: WORKING — bot connected to live markets and actively")
        print("           evaluated real two-sided quotes for arbitrage.")
        if stats["taker_signals"] == 0 and stats["maker_signals"] == 0:
            print("           No arb signals — expected: live asks sum > 1.0"
                  " (real spread).")
    print("═" * 64 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\ninterrupted")
