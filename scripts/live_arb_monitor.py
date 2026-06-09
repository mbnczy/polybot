"""
scripts/live_arb_monitor.py
───────────────────────────
Persistent LIVE arbitrage monitor — watches Polymarket in real time and sends a
Telegram alert whenever an arbitrage is detected under the current strategy
settings.  NO trading: the execution client (PolyClient) is never constructed,
so no wallet key is touched and nothing is ever ordered.

Path (identical detection logic to main.py, execution layer removed):

    MarketScanner (Gamma API, re-scans every SCAN_INTERVAL)
        → FeedRegistry.add_market
        → MarketFeed  (real wss://ws-subscriptions-clob.polymarket.com/ws/market)
        → arb_tick queue
        → ArbDetector.evaluate  (taker path)
          └ DutchBookPricer.evaluate_maker  (maker path, per-market rebate)
        → TelegramNotifier.send_arb_detected   (throttled: per-market cooldown
                                                 + ARB_ALERT_MIN_BPS floor)

Settings honoured (from .env, same as the live bot):
    DESIRED_NET_MARGIN   minimum edge to fire           (default 0.005)
    DEFAULT_TAKER_FEE    taker-path fee assumption       (default 0.02)
    SCAN_INTERVAL        seconds between Gamma re-scans   (default 300)
    MAX_FEEDS            markets watched concurrently     (default 50)
    MAX_ARB_PAIR_USDC    per-pair sizing cap             (default 50)
    ARB_ALERT_COOLDOWN_S per-market alert cooldown        (default 300)
    ARB_ALERT_MIN_BPS    suppress alerts below this edge  (default 0)
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID                 (required for alerts)

Run:
    python scripts/live_arb_monitor.py
    nohup python scripts/live_arb_monitor.py > /tmp/live_monitor.log 2>&1 &
Stop:  Ctrl-C  (or kill the PID) — sends an "offline" Telegram notice.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
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
logger = logging.getLogger("live_monitor")

# Import after env load.
from core.scanner import FeedRegistry, MarketScanner            # noqa: E402
from strategy.arbitrage import (                                # noqa: E402
    ArbDetector,
    DutchBookPricer,
    MakerRebateEngine,
    DESIRED_NET_MARGIN,
    DEFAULT_MAKER_REBATE,
    EXTREME_PRICE_LO,
    EXTREME_PRICE_HI,
    MIN_REAL_EDGE,
)
from telemetry.telegram import TelegramNotifier                 # noqa: E402

# ── Settings (same env vars as the live bot) ──────────────────────────────────
_MARGIN       = float(os.environ.get("DESIRED_NET_MARGIN", DESIRED_NET_MARGIN))
_DEFAULT_FEE  = float(os.environ.get("DEFAULT_TAKER_FEE", "0.02"))
_SCAN_INTERVAL = float(os.environ.get("SCAN_INTERVAL", "300"))
_MAX_FEEDS    = int(os.environ.get("MAX_FEEDS", "50"))
_PAIR_CAP     = float(os.environ.get("MAX_ARB_PAIR_USDC", "50"))
_HEARTBEAT_S  = float(os.environ.get("MONITOR_HEARTBEAT_S", "3600"))   # local + TG ping
_PRUNE_IDLE_S  = float(os.environ.get("FEED_PRUNE_IDLE_S", "600"))     # drop dead feeds
_MIN_VOLUME_24H = float(os.environ.get("MIN_VOLUME_24H", "0"))         # liquidity floor
_EXTREME_LO    = float(os.environ.get("EXTREME_PRICE_LO", EXTREME_PRICE_LO))
_EXTREME_HI    = float(os.environ.get("EXTREME_PRICE_HI", EXTREME_PRICE_HI))
_MIN_REAL_EDGE = float(os.environ.get("MIN_REAL_EDGE", MIN_REAL_EDGE))

_shutdown = asyncio.Event()


async def _consume(
    queue:    "asyncio.Queue[dict]",
    detector: ArbDetector,
    dutch:    DutchBookPricer,
    rebates:  MakerRebateEngine,
    notifier: TelegramNotifier,
    stats:    dict,
) -> None:
    """Evaluate every live tick; alert on each detected arb (throttled)."""
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
        if yes_ask is None or no_ask is None:
            continue
        stats["two_sided"] += 1

        # ── Taker path ────────────────────────────────────────────────────────
        sig = detector.evaluate(
            condition_id=cid, yes_token_id=yes_id, no_token_id=no_id,
            yes_ask=yes_ask, no_ask=no_ask,
            max_position_usdc=_PAIR_CAP, fee_rate=_DEFAULT_FEE,
        )
        display_edge = None
        is_maker = False
        if sig is not None:
            display_edge = sig.net_edge
        else:
            # ── Maker path (per-market rebate, like main.py) ──────────────────
            try:
                rebate = await rebates.get_maker_rebate(cid)
            except Exception:                       # network hiccup → conservative
                rebate = DEFAULT_MAKER_REBATE
            sig = dutch.evaluate_maker(
                condition_id=cid, yes_token_id=yes_id, no_token_id=no_id,
                yes_ask=yes_ask, no_ask=no_ask,
                max_position_usdc=_PAIR_CAP, maker_rebate=rebate,
            )
            if sig is not None:
                display_edge = sig.maker_net_edge
                is_maker = True

        if sig is None:
            continue

        stats["signals"] += 1
        logger.info(
            "ARB DETECTED (%s) | %s yes=%.4f no=%.4f edge=%.1f bps",
            "MAKER" if is_maker else "TAKER", cid[:18],
            yes_ask, no_ask, (display_edge or 0.0) * 10_000,
        )
        # Fire-and-forget Telegram alert (notifier self-throttles per market).
        notifier.send_arb_detected(
            condition_id=cid,
            combined_cost=sig.combined_cost,
            net_edge=display_edge or 0.0,
            is_maker=is_maker,
            yes_price=yes_ask,
            no_price=no_ask,
        )


async def _heartbeat(registry: FeedRegistry, notifier: TelegramNotifier, stats: dict) -> None:
    """Periodic liveness ping — local log always, Telegram every _HEARTBEAT_S."""
    t0 = time.monotonic()
    last_tg = 0.0
    while not _shutdown.is_set():
        await asyncio.sleep(30)
        elapsed = time.monotonic() - t0
        logger.info(
            "  [alive %.0fs] feeds=%d ticks=%d two_sided=%d signals=%d",
            elapsed, registry.active_count, stats["ticks"],
            stats["two_sided"], stats["signals"],
        )
        if elapsed - last_tg >= _HEARTBEAT_S:
            last_tg = elapsed
            await notifier.notify(
                f"💓 Arb monitor alive — feeds={registry.active_count} "
                f"ticks={stats['ticks']} signals={stats['signals']}"
            )


async def main() -> None:
    logger.info(
        "=== LIVE ARB MONITOR (no trading) | margin=%.4f fee=%.3f "
        "max_feeds=%d scan=%.0fs pair_cap=%.0f ===",
        _MARGIN, _DEFAULT_FEE, _MAX_FEEDS, _SCAN_INTERVAL, _PAIR_CAP,
    )

    notifier = TelegramNotifier()
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=4096)
    registry = FeedRegistry(queue=queue, max_feeds=_MAX_FEEDS)
    scanner = MarketScanner(
        on_market_added=registry.add_market,
        scan_interval=_SCAN_INTERVAL,
        max_feeds=_MAX_FEEDS,
        feed_registry=registry,
        prune_idle_s=_PRUNE_IDLE_S,
        min_volume_24h=_MIN_VOLUME_24H,
    )
    detector = ArbDetector(
        desired_net_margin=_MARGIN, default_fee_rate=_DEFAULT_FEE,
        extreme_lo=_EXTREME_LO, extreme_hi=_EXTREME_HI,
    )
    dutch    = DutchBookPricer(
        desired_net_margin=_MARGIN,
        extreme_lo=_EXTREME_LO, extreme_hi=_EXTREME_HI, min_real_edge=_MIN_REAL_EDGE,
    )
    rebates  = MakerRebateEngine()
    stats    = {"ticks": 0, "two_sided": 0, "signals": 0}

    # Startup notice (also validates the Telegram token end-to-end).
    ok = await notifier.notify(
        "🟢 <b>Live arb monitor online</b>\n"
        f"Watching up to {_MAX_FEEDS} markets in real time — "
        f"alerting on arbs ≥ {_MARGIN*10_000:.0f} bps. No trading.",
        parse_mode="HTML",
    )
    logger.info("Telegram startup notice sent: %s", ok)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()

    def _stop(signame: str) -> None:
        logger.info("Received %s — shutting down monitor", signame)
        _shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop, sig.name)
        except NotImplementedError:
            pass

    scan_task = asyncio.create_task(scanner.run(), name="scanner")
    cons_task = asyncio.create_task(
        _consume(queue, detector, dutch, rebates, notifier, stats), name="consumer"
    )
    hb_task   = asyncio.create_task(_heartbeat(registry, notifier, stats), name="heartbeat")

    await _shutdown.wait()

    # ── Teardown ──────────────────────────────────────────────────────────────
    for t in (scan_task, cons_task, hb_task):
        t.cancel()
    for t in (scan_task, cons_task, hb_task):
        try:
            await t
        except asyncio.CancelledError:
            pass
    await registry.stop_all()
    await notifier.notify(
        f"🔴 Live arb monitor offline — ran with {stats['signals']} arb alert(s) "
        f"over {stats['ticks']} ticks."
    )
    await notifier.close()
    logger.info(
        "Monitor stopped — ticks=%d two_sided=%d signals=%d",
        stats["ticks"], stats["two_sided"], stats["signals"],
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
