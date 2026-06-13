"""
scripts/compare_monitor.py
──────────────────────────
Branch-labelled, detection-only arbitrage monitor for A/B comparison.

Run two instances pointed at two different checkouts (e.g. a `main` worktree and
the refactored trunk) with the SAME Telegram bot token but different BOT_LABEL.
Every message is prefixed with `[<BOT_LABEL>]` so you can tell which pipeline
produced it.

Design notes
────────────
- **Send-only**: messages go out via TelegramNotifier.notify(). It does NOT call
  run_listener(), so it never opens a getUpdates long-poll — multiple instances
  can share one bot token without a 409 Conflict.
- **No trading / no keys**: no PolyClient, no execution. Pure detection.
- **Common API only**: uses scanner / arbitrage / telegram surfaces present in
  BOTH the original (`main`) pipeline and the refactored trunk, so the SAME
  script runs against either checkout and the differences you observe come from
  the pipeline code itself (signal-quality guards, orjson, dedup, …).

Env:
    BOT_LABEL=main|refactored   label prefix (required for a meaningful compare)
    MAX_FEEDS, SCAN_INTERVAL, DEFAULT_TAKER_FEE, DESIRED_NET_MARGIN
    ARB_ALERT_COOLDOWN_S (per-market alert cooldown, default 300)
    ARB_ALERT_MIN_BPS    (suppress alerts below this edge, default 0)
    MONITOR_HEARTBEAT_S  (Telegram heartbeat cadence, default 1800)
    DURATION             (seconds; 0 = run forever, default 0)
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
logger = logging.getLogger("compare_monitor")

from core.scanner import FeedRegistry, MarketScanner               # noqa: E402
from strategy.arbitrage import (                                   # noqa: E402
    ArbDetector,
    DutchBookPricer,
    MakerRebateEngine,
    DESIRED_NET_MARGIN,
    DEFAULT_MAKER_REBATE,
)
from telemetry.telegram import TelegramNotifier                    # noqa: E402

LABEL         = os.environ.get("BOT_LABEL", "?")
MAX_FEEDS     = int(os.environ.get("MAX_FEEDS", "50"))
SCAN_INTERVAL = float(os.environ.get("SCAN_INTERVAL", "300"))
DEFAULT_FEE   = float(os.environ.get("DEFAULT_TAKER_FEE", "0.02"))
MARGIN        = float(os.environ.get("DESIRED_NET_MARGIN", DESIRED_NET_MARGIN))
PAIR_CAP      = float(os.environ.get("MAX_ARB_PAIR_USDC", "50"))
COOLDOWN_S    = float(os.environ.get("ARB_ALERT_COOLDOWN_S", "300"))
MIN_BPS       = float(os.environ.get("ARB_ALERT_MIN_BPS", "0"))
HEARTBEAT_S   = float(os.environ.get("MONITOR_HEARTBEAT_S", "1800"))
DURATION      = float(os.environ.get("DURATION", "0"))   # 0 = forever

_shutdown = asyncio.Event()


def _tag(text: str) -> str:
    return f"[{LABEL}] {text}"


async def run() -> None:
    logger.info("=== compare_monitor [%s] | max_feeds=%d margin=%.4f ===",
                LABEL, MAX_FEEDS, MARGIN)

    notifier = TelegramNotifier()
    queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
    registry = FeedRegistry(queue=queue)
    scanner = MarketScanner(
        on_market_added=registry.add_market,
        scan_interval=SCAN_INTERVAL,
        max_feeds=MAX_FEEDS,
    )
    detector = ArbDetector(desired_net_margin=MARGIN, default_fee_rate=DEFAULT_FEE)
    dutch    = DutchBookPricer(desired_net_margin=MARGIN)
    rebates  = MakerRebateEngine()

    last_alert: dict[str, float] = {}
    stats = {"ticks": 0, "two_sided": 0, "signals": 0}

    ok = await notifier.notify(_tag(
        f"🟢 <b>{LABEL} monitor online</b> — detection-only, no trading. "
        f"Watching up to {MAX_FEEDS} markets; alerting on arbs ≥ {MARGIN*1e4:.0f} bps."
    ), parse_mode="HTML")
    logger.info("[%s] startup notice sent: %s", LABEL, ok)

    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _shutdown.set)
        except NotImplementedError:
            pass

    scan_task = asyncio.create_task(scanner.run(), name="scanner")

    async def _consume() -> None:
        while True:
            tick = await queue.get()
            if not isinstance(tick, dict) or tick.get("type") != "arb_tick":
                continue
            stats["ticks"] += 1
            cid = tick.get("condition_id", "")
            yid = tick.get("yes_token_id", "")
            nid = tick.get("no_token_id", "")
            ya  = tick.get("yes_ask")
            na  = tick.get("no_ask")
            if ya is None or na is None:
                continue
            stats["two_sided"] += 1

            sig = detector.evaluate(
                condition_id=cid, yes_token_id=yid, no_token_id=nid,
                yes_ask=ya, no_ask=na, max_position_usdc=PAIR_CAP, fee_rate=DEFAULT_FEE,
            )
            is_maker = False
            edge = sig.net_edge if sig is not None else None
            if sig is None:
                try:
                    rb = await rebates.get_maker_rebate(cid)
                except Exception:
                    rb = DEFAULT_MAKER_REBATE
                sig = dutch.evaluate_maker(
                    condition_id=cid, yes_token_id=yid, no_token_id=nid,
                    yes_ask=ya, no_ask=na, max_position_usdc=PAIR_CAP, maker_rebate=rb,
                )
                if sig is not None:
                    edge = sig.maker_net_edge
                    is_maker = True
            if sig is None:
                continue

            edge_bps = round((edge or 0.0) * 10_000, 1)
            if edge_bps < MIN_BPS:
                continue
            now = time.monotonic()
            if cid in last_alert and (now - last_alert[cid]) < COOLDOWN_S:
                continue
            last_alert[cid] = now
            stats["signals"] += 1

            path = "MAKER" if is_maker else "TAKER"
            logger.info("[%s] ARB %s %s yes=%.4f no=%.4f edge=%.1fbps",
                        LABEL, path, cid[:18], ya, na, edge_bps)
            await notifier.notify(_tag(
                f"<b>ARB DETECTED</b> ({path})\n"
                f"market <code>{cid[:24]}…</code>\n"
                f"YES {ya:.4f} / NO {na:.4f} / combined {sig.combined_cost:.4f}\n"
                f"edge {edge_bps:.1f} bps"
            ), parse_mode="HTML")

    cons_task = asyncio.create_task(_consume(), name="consumer")

    async def _heartbeat() -> None:
        t0 = time.monotonic()
        last_tg = 0.0
        while not _shutdown.is_set():
            await asyncio.sleep(30)
            el = time.monotonic() - t0
            logger.info("[%s alive %.0fs] feeds=%d ticks=%d two_sided=%d signals=%d",
                        LABEL, el, registry.active_count, stats["ticks"],
                        stats["two_sided"], stats["signals"])
            if el - last_tg >= HEARTBEAT_S:
                last_tg = el
                await notifier.notify(_tag(
                    f"💓 alive — feeds={registry.active_count} "
                    f"ticks={stats['ticks']} signals={stats['signals']}"
                ))
            if DURATION and el >= DURATION:
                _shutdown.set()

    hb_task = asyncio.create_task(_heartbeat(), name="heartbeat")

    await _shutdown.wait()
    for t in (scan_task, cons_task, hb_task):
        t.cancel()
    for t in (scan_task, cons_task, hb_task):
        try:
            await t
        except asyncio.CancelledError:
            pass
    await registry.stop_all()
    await notifier.notify(_tag(
        f"🔴 {LABEL} monitor offline — {stats['signals']} alert(s) over "
        f"{stats['ticks']} ticks."
    ))
    await notifier.close()
    logger.info("[%s] stopped — ticks=%d signals=%d", LABEL, stats["ticks"], stats["signals"])


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\ninterrupted")
