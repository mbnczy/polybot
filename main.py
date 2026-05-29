"""
main.py
───────
Polymarket CLOB Bot — Phase 7: Autonomous, Scalable, Fee-Aware Edition.

Architecture
────────────
Six long-running coroutines run concurrently under asyncio.gather:

  scanner_loop      ←  MarketScanner polls Gamma API → FeedRegistry
  strategy_loop     ←  Queue consumer → FeeEngine → ArbDetector →
                        CircuitBreaker → dual-FOK → TelegramNotifier
  auto_redeem_loop  ←  AutoRedeemer polls resolved positions → CTF redeem
  heartbeat_loop    ←  60-second Telegram health ping
  telegram_loop     ←  python-telegram-bot polling loop; handles /status and
                        /halt commands from the authorised TELEGRAM_CHAT_ID
  [feed tasks]      ←  One MarketFeed per discovered market (managed by
                        FeedRegistry; each pushes arb_tick dicts to the queue)

Startup sequence
────────────────
  1.  Load .env (python-dotenv)
  2.  Configure logging
  3.  Initialise PolyClient (derives L2 CLOB API creds)
  4.  Initialise CircuitBreaker with STARTING_BALANCE
  5.  Initialise TelegramNotifier
  6.  Initialise FeeEngine (async fee cache; no network call until first use)
  7.  Initialise ArbDetector(desired_net_margin, default_fee_rate)
  8.  Initialise FeedRegistry (shared arb_tick queue)
  9.  Seed FeedRegistry with ENV-configured YES/NO pair (if set)
  10. Initialise MarketScanner → FeedRegistry.add_market callback
  11. Initialise AutoRedeemer(FeedRegistry, TelegramNotifier)
  12. asyncio.gather all tasks

strategy_loop hot path
───────────────────────
  arb_tick arrives → FeeEngine.get_taker_fee() → ArbDetector.evaluate()
  → CircuitBreaker.calculate_position_size() → ArbOrderIntent
  → CircuitBreaker.check_arb()
  → PolyClient.execute_arb_pair()  (asyncio.gather dual-FOK)
  → breaker.on_arb_open() + on_fill(guaranteed_profit)
  → TelegramNotifier.send_trade_execution()  (fire-and-forget)

Signal handling
───────────────
SIGINT / SIGTERM cancel all tasks and call FeedRegistry.stop_all() for clean
teardown of dynamically-created per-market feed tasks.

Environment variables
─────────────────────
  POLY_PRIVATE_KEY, POLY_FUNDER_ADDRESS, POLYGON_RPC_URL
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  STARTING_BALANCE        float USDC (default 500.0)
  DESIRED_NET_MARGIN      float 0–1  (default 0.005 = 0.5% after fees)
  DEFAULT_TAKER_FEE       float 0–1  (default 0.02 = 2.0%)
  LOG_LEVEL               DEBUG | INFO | WARNING (default INFO)
  PAPER_TRADE_MODE        "true" → simulate orders locally
  REDEEM_POLL_INTERVAL    seconds between redemption scans (default 300)
  SCAN_INTERVAL           seconds between Gamma API market scans (default 300)

  Optional single-market seed (backward-compatible):
  YES_TOKEN_ID, NO_TOKEN_ID, CONDITION_ID
"""

from __future__ import annotations

# ── uvloop: must be installed before the event loop is created ─────────────
# Replaces CPython's default selector-based loop with libuv (C extension),
# giving ~2–4× higher throughput for I/O-heavy async workloads.
# uvloop is Linux/macOS only; falls back silently on Windows dev machines.
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass  # Windows: standard asyncio event loop used

import asyncio
import logging
import os
import signal
import sys
from typing import Optional

from dotenv import load_dotenv

# ── load .env before any module that reads os.environ ──────────────────────
load_dotenv()

from core.clob_client import PolyClient                            # noqa: E402
from core.scanner import FeedRegistry, MarketScanner               # noqa: E402
from execution.auto_redeem import AutoRedeemer                     # noqa: E402
from risk.circuit_breaker import (                                 # noqa: E402
    ArbOrderIntent,
    CircuitBreaker,
    CircuitBreakerTripped,
    MAX_ARB_PAIR_USDC,
)
from core.clob_client import BundleLeg                             # noqa: E402
from strategy.arbitrage import (                                   # noqa: E402
    ArbDetector,
    ArbSignal,
    DEFAULT_TAKER_FEE,
    DESIRED_NET_MARGIN,
    DutchBookPricer,
    FeeEngine,
    MakerRebateEngine,
    NegRiskArbDetector,
    NegRiskSignal,
)
from strategy.tuner import tuner_loop                               # noqa: E402
from telemetry.db_logger import SignalLogger                        # noqa: E402
from telemetry.metrics import (                                    # noqa: E402
    ACTIVE_MARKETS,
    CUMULATIVE_PNL,
    USDC_BALANCE,
    WS_LATENCY_MS,
    metrics_server,
)
from telemetry.telegram import TelegramNotifier                    # noqa: E402

# ── logging ────────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s — %(message)s"
_LOG_DATE   = "%H:%M:%S"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format=_LOG_FORMAT,
    datefmt=_LOG_DATE,
    stream=sys.stdout,
)
logger = logging.getLogger("main")

# ── configuration ──────────────────────────────────────────────────────────
STARTING_BALANCE:    float = float(os.environ.get("STARTING_BALANCE",    500.0))
HEARTBEAT_INTERVAL:  float = 60.0
QUEUE_MAXSIZE:       int   = 2048

# Fee / margin — override defaults from strategy/arbitrage.py via ENV
_desired_margin = float(os.environ.get("DESIRED_NET_MARGIN", DESIRED_NET_MARGIN))
_default_fee    = float(os.environ.get("DEFAULT_TAKER_FEE",  DEFAULT_TAKER_FEE))
_scan_interval  = float(os.environ.get("SCAN_INTERVAL",      300.0))
_max_feeds      = int(os.environ.get("MAX_FEEDS",            50))

# Optional single-market seed (backward-compatible with Phase ≤ 6 .env files)
YES_TOKEN_ID: str = os.environ.get("YES_TOKEN_ID", "").strip()
NO_TOKEN_ID:  str = os.environ.get("NO_TOKEN_ID",  "").strip()
CONDITION_ID: str = os.environ.get("CONDITION_ID", "").strip()

# ── global shutdown event ──────────────────────────────────────────────────
_shutdown_event: asyncio.Event = asyncio.Event()


# ═══════════════════════════════════════════════════════════════════════════
# Coroutines
# ═══════════════════════════════════════════════════════════════════════════

async def scanner_loop(scanner: MarketScanner) -> None:
    """Drive the Gamma API market scanner."""
    logger.info("scanner_loop started")
    try:
        await scanner.run()
    except asyncio.CancelledError:
        scanner.stop()
        logger.info("scanner_loop stopped")
        raise


async def strategy_loop(
    queue:          asyncio.Queue,
    detector:       ArbDetector,
    dutch_pricer:   DutchBookPricer,
    neg_risk_det:   NegRiskArbDetector,
    rebate_engine:  MakerRebateEngine,
    fee_engine:     FeeEngine,
    breaker:        CircuitBreaker,
    client:         PolyClient,
    notifier:       TelegramNotifier,
    sig_logger:     SignalLogger,
) -> None:
    """
    Consumes arb_tick / neg_risk_tick snapshots from ALL active market feeds.

    arb_tick path (binary market):
      1. Taker arb — FeeEngine → ArbDetector → FOK execution
      2. Maker arb fallback — MakerRebateEngine → DutchBookPricer → GTC execution

    neg_risk_tick path (multi-outcome NegRisk market):
      MakerRebateEngine → NegRiskArbDetector → matchOrders bundle execution
    """
    logger.info("strategy_loop started")
    try:
        while True:
            tick: dict = await queue.get()
            queue.task_done()

            tick_type = tick.get("type")

            # ── NegRisk multi-outcome path ────────────────────────────────────
            if tick_type == "neg_risk_tick":
                condition_id      = tick["condition_id"]
                outcome_token_ids: list[str]   = tick["outcome_token_ids"]
                no_asks:           list[float]  = tick["no_asks"]

                WS_LATENCY_MS.set(
                    (asyncio.get_event_loop().time() - tick.get("ts", 0)) * 1000
                )

                rebate = await rebate_engine.get_maker_rebate(condition_id)
                nr_signal: Optional[NegRiskSignal] = neg_risk_det.evaluate_neg_risk(
                    condition_id=condition_id,
                    outcome_token_ids=outcome_token_ids,
                    no_asks=no_asks,
                    max_position_usdc=MAX_ARB_PAIR_USDC,
                    maker_rebate=rebate,
                )
                if nr_signal is None:
                    continue

                bundle_cost = round(nr_signal.combined_bid * nr_signal.n_bundles, 6)
                nr_intent = ArbOrderIntent(
                    condition_id=condition_id,
                    yes_token_id=nr_signal.legs[0].token_id,
                    no_token_id=(
                        nr_signal.legs[1].token_id if len(nr_signal.legs) > 1 else ""
                    ),
                    yes_price=nr_signal.legs[0].no_bid,
                    no_price=(
                        nr_signal.legs[1].no_bid if len(nr_signal.legs) > 1 else 0.0
                    ),
                    n_shares=nr_signal.n_bundles,
                    combined_cost_usdc=bundle_cost,
                )
                if not breaker.check_arb(nr_intent):
                    continue

                bundle_legs = [
                    BundleLeg(token_id=leg.token_id, bid=leg.no_bid, size=leg.size)
                    for leg in nr_signal.legs
                ]
                try:
                    await client.execute_arb_maker_bundle(bundle_legs)

                    net_profit = round(nr_signal.n_bundles * nr_signal.net_edge, 6)
                    breaker.on_arb_open()
                    breaker.on_fill(pnl=net_profit)

                    status = breaker.status_dict()
                    CUMULATIVE_PNL.set(status["session_pnl"])
                    USDC_BALANCE.set(
                        status["starting_balance"] + status["session_pnl"]
                    )

                    await notifier.notify(
                        f"NEG RISK ARB | condition={condition_id[:20]}\n"
                        f"outcomes={nr_signal.n_outcomes} "
                        f"bundles={nr_signal.n_bundles:.2f} "
                        f"net_edge={nr_signal.net_edge:.4f} "
                        f"profit={net_profit:.4f} USDC"
                    )

                except CircuitBreakerTripped:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("NegRisk bundle execution failed: %s", exc)
                    await notifier.notify(f"NEG RISK EXECUTION ERROR: {exc}")

                continue

            # ── Binary market path (arb_tick) ─────────────────────────────────
            if tick_type != "arb_tick":
                continue

            condition_id: str   = tick["condition_id"]
            yes_token_id: str   = tick["yes_token_id"]
            no_token_id:  str   = tick["no_token_id"]
            yes_ask:      float = tick["yes_ask"]
            no_ask:       float = tick["no_ask"]

            # ── Metrics: WebSocket delivery latency ───────────────────────────
            WS_LATENCY_MS.set(
                (asyncio.get_event_loop().time() - tick["ts"]) * 1000
            )

            # ── 1. Fetch real-time taker fee (cached, ~0 overhead) ────────────
            fee_rate: float = await fee_engine.get_taker_fee(condition_id)

            # ── 2a. Taker arb evaluation ──────────────────────────────────────
            arb_signal: Optional[ArbSignal] = detector.evaluate(
                condition_id=condition_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_ask=yes_ask,
                no_ask=no_ask,
                max_position_usdc=MAX_ARB_PAIR_USDC,
                fee_rate=fee_rate,
            )

            # ── 2b. Maker arb fallback (DutchBookPricer) ──────────────────────
            if arb_signal is None:
                rebate = await rebate_engine.get_maker_rebate(condition_id)
                arb_signal = dutch_pricer.evaluate_maker(
                    condition_id=condition_id,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                    max_position_usdc=MAX_ARB_PAIR_USDC,
                    maker_rebate=rebate,
                )

            if arb_signal is None:
                continue

            # ── 3. Persist signal ─────────────────────────────────────────────
            sig_logger.log_arb(arb_signal)

            # ── 4. Size the position ──────────────────────────────────────────
            if arb_signal.is_maker_signal:
                n_shares = arb_signal.yes_size
            else:
                n_shares = breaker.calculate_position_size(yes_ask, no_ask)
            if n_shares < 0.01:
                logger.warning(
                    "Position size %.4f below minimum — skipping "
                    "(yes=%.4f no=%.4f)",
                    n_shares, yes_ask, no_ask,
                )
                continue

            combined_cost_usdc: float = round(n_shares * arb_signal.combined_cost, 6)

            # ── 5. Circuit-breaker gate ───────────────────────────────────────
            intent = ArbOrderIntent(
                condition_id=condition_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_price=yes_ask if not arb_signal.is_maker_signal else arb_signal.yes_bid,
                no_price=no_ask  if not arb_signal.is_maker_signal else arb_signal.no_bid,
                n_shares=n_shares,
                combined_cost_usdc=combined_cost_usdc,
            )
            if not breaker.check_arb(intent):
                continue

            # ── 6. Execute ────────────────────────────────────────────────────
            try:
                if arb_signal.is_maker_signal:
                    await client.execute_arb_maker_pair(
                        yes_token_id=yes_token_id,
                        yes_bid=arb_signal.yes_bid,
                        yes_size=n_shares,
                        no_token_id=no_token_id,
                        no_bid=arb_signal.no_bid,
                        no_size=n_shares,
                    )
                else:
                    await client.execute_arb_pair(
                        yes_token_id=yes_token_id,
                        yes_price=yes_ask,
                        yes_size=n_shares,
                        no_token_id=no_token_id,
                        no_price=no_ask,
                        no_size=n_shares,
                    )

                net_profit: float = round(n_shares * arb_signal.net_edge, 6)
                breaker.on_arb_open()
                breaker.on_fill(pnl=net_profit)

                status = breaker.status_dict()
                CUMULATIVE_PNL.set(status["session_pnl"])
                USDC_BALANCE.set(
                    status["starting_balance"] + status["session_pnl"]
                )

                await notifier.send_trade_execution(
                    condition_id=condition_id,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                    n_shares=n_shares,
                    combined_cost=arb_signal.combined_cost,
                    guaranteed_profit=net_profit,
                )

            except CircuitBreakerTripped:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("Arb pair execution failed: %s", exc)
                await notifier.notify(f"ARB EXECUTION ERROR: {exc}")

    except asyncio.CancelledError:
        logger.info("strategy_loop stopped")
        raise


async def auto_redeem_loop(redeemer: AutoRedeemer) -> None:
    """Drive the CTF position redeemer."""
    logger.info("auto_redeem_loop started")
    try:
        await redeemer.run()
    except asyncio.CancelledError:
        logger.info("auto_redeem_loop stopped")
        raise


async def heartbeat_loop(
    breaker:       CircuitBreaker,
    notifier:      TelegramNotifier,
    feed_registry: FeedRegistry,
) -> None:
    """Send a periodic system health update to Telegram and refresh metrics."""
    logger.info("heartbeat_loop started")
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            status = breaker.status_dict()
            logger.info("Heartbeat | %s", status)
            await notifier.heartbeat(status)

            # ── Metrics: refresh balance + market count (60-s cadence) ────
            USDC_BALANCE.set(
                status["starting_balance"] + status["session_pnl"]
            )
            CUMULATIVE_PNL.set(status["session_pnl"])
            ACTIVE_MARKETS.set(feed_registry.active_count)
    except asyncio.CancelledError:
        logger.info("heartbeat_loop stopped")
        raise


async def telegram_loop(notifier: TelegramNotifier) -> None:
    """
    Drive the Telegram command listener (python-telegram-bot long-polling).

    Runs concurrently via asyncio.gather — does not block the hot path.
    Handles /status and /halt commands from the authorised TELEGRAM_CHAT_ID.
    """
    logger.info("telegram_loop started")
    try:
        await notifier.run_listener()
    except asyncio.CancelledError:
        logger.info("telegram_loop stopped")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _register_shutdown_signals(
    tasks:         list[asyncio.Task],
    feed_registry: FeedRegistry,
) -> None:
    """Register SIGINT / SIGTERM to cancel all tasks and stop feed tasks."""
    loop = asyncio.get_running_loop()

    def _cancel_all(sig_name: str) -> None:
        logger.info("Signal %s received — shutting down …", sig_name)
        _shutdown_event.set()
        for t in tasks:
            t.cancel()
        # Feed tasks are not in `tasks` (created dynamically by FeedRegistry);
        # schedule stop_all() as a fire-and-forget coroutine.
        asyncio.ensure_future(feed_registry.stop_all())

    for sig, name in [(signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")]:
        try:
            loop.add_signal_handler(sig, _cancel_all, name)
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals.
            signal.signal(sig, lambda *_: _cancel_all(name))


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    logger.info("=== Polymarket ARB Bot — Phase 8 starting ===")
    logger.info(
        "Config | balance=%.2f net_margin=%.3f default_fee=%.3f pair_cap=%.2f USDC",
        STARTING_BALANCE, _desired_margin, _default_fee, MAX_ARB_PAIR_USDC,
    )

    # ── initialise components ─────────────────────────────────────────────
    client         = PolyClient()
    breaker        = CircuitBreaker(starting_balance=STARTING_BALANCE)
    notifier       = TelegramNotifier(on_status=breaker.status_dict)
    fee_engine     = FeeEngine(default_fee=_default_fee)
    rebate_engine  = MakerRebateEngine()
    detector       = ArbDetector(
                         desired_net_margin=_desired_margin,
                         default_fee_rate=_default_fee,
                     )
    dutch_pricer   = DutchBookPricer(desired_net_margin=_desired_margin)
    neg_risk_det   = NegRiskArbDetector(desired_net_margin=_desired_margin)

    # ── signal logger (async WAL-mode SQLite, zero hot-path latency) ──────
    sig_logger = SignalLogger()
    await sig_logger.init()

    market_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    feed_registry = FeedRegistry(queue=market_queue)

    # ── seed with ENV-configured market (backward-compatible) ────────────
    if YES_TOKEN_ID and NO_TOKEN_ID:
        seed_cond = CONDITION_ID or f"{YES_TOKEN_ID[:8]}/{NO_TOKEN_ID[:8]}"
        logger.info(
            "Seeding feed from ENV | condition=%s yes=%s no=%s",
            seed_cond[:20], YES_TOKEN_ID[:20], NO_TOKEN_ID[:20],
        )
        await feed_registry.add_market(seed_cond, YES_TOKEN_ID, NO_TOKEN_ID)
        # Pre-warm the fee cache with the seeded market
        seed_fee = await fee_engine.get_taker_fee(seed_cond)
        logger.info(
            "Fee for seeded market: %.4f (%.2f%%)", seed_fee, seed_fee * 100
        )
    else:
        logger.info(
            "No ENV market seed — waiting for scanner to discover markets"
        )
        seed_cond = None

    # ── scanner: pre-populate known set with any ENV-seeded condition ─────
    seed_ids: set[str] = {seed_cond} if seed_cond and CONDITION_ID else set()
    scanner = MarketScanner(
        on_market_added=feed_registry.add_market,
        scan_interval=_scan_interval,
        seed_condition_ids=seed_ids,
        max_feeds=_max_feeds,
    )

    # ── auto redeemer ─────────────────────────────────────────────────────
    redeemer = AutoRedeemer(
        feed_registry=feed_registry,
        notifier=notifier,
    )

    await notifier.notify(
        f"Polymarket ARB Bot v7 online\n"
        f"balance={STARTING_BALANCE} USDC | margin={_desired_margin:.3f} "
        f"| default_fee={_default_fee:.3f} | cap={MAX_ARB_PAIR_USDC} USDC\n"
        f"scan_interval={_scan_interval:.0f}s | "
        f"feeds_seeded={feed_registry.active_count}"
    )

    # ── build halt callback (closes over all live components) ────────────────
    # _halt_tasks is populated after task creation and captured by reference,
    # so /halt always cancels exactly the tasks that are currently running.
    _halt_tasks: list[asyncio.Task] = []

    async def _do_halt() -> None:
        """
        Manual kill switch triggered by the Telegram /halt command.

        Mirrors the CircuitBreakerTripped shutdown path:
          1. Set the global shutdown event.
          2. Cancel all open orders via the CLOB REST client.
          3. Cancel every asyncio task in the main gather group.
          4. Stop all per-market feed tasks.
        """
        logger.critical("MANUAL HALT — /halt command received via Telegram")
        _shutdown_event.set()
        await notifier.send_critical_error(
            "MANUAL HALT — /halt command received via Telegram\n"
            "Cancelling all open orders..."
        )
        try:
            await client.cancel_all_orders()
            await notifier.notify("All orders cancelled. Bot halted.")
        except Exception as cancel_exc:  # noqa: BLE001
            logger.error("cancel_all_orders failed during manual halt: %s", cancel_exc)
            await notifier.notify(f"WARNING: cancel_all_orders failed: {cancel_exc}")
        for t in _halt_tasks:
            t.cancel()
        await feed_registry.stop_all()

    notifier.set_halt_callback(_do_halt)

    # ── launch coroutines ─────────────────────────────────────────────────
    tasks = [
        asyncio.create_task(scanner_loop(scanner),                       name="scanner"),
        asyncio.create_task(
            strategy_loop(
                market_queue, detector, dutch_pricer, neg_risk_det, rebate_engine,
                fee_engine, breaker, client, notifier, sig_logger,
            ),
            name="strategy",
        ),
        asyncio.create_task(auto_redeem_loop(redeemer),                        name="auto_redeem"),
        asyncio.create_task(heartbeat_loop(breaker, notifier, feed_registry),  name="heartbeat"),
        asyncio.create_task(telegram_loop(notifier),                           name="telegram"),
        asyncio.create_task(sig_logger.run(),                                  name="sig_logger"),
        asyncio.create_task(metrics_server(),                                  name="metrics"),
        asyncio.create_task(tuner_loop(detector),                              name="tuner"),
    ]
    # Populate halt-tasks so /halt can cancel this exact gather group.
    _halt_tasks.extend(tasks)

    _register_shutdown_signals(tasks, feed_registry)

    try:
        await asyncio.gather(*tasks)
    except CircuitBreakerTripped as exc:
        logger.critical("DAILY LOSS LIMIT BREACHED — halting bot | %s", exc)
        await notifier.send_critical_error(
            f"EMERGENCY HALT — daily loss limit breached\n{exc}\n"
            f"Cancelling all open orders..."
        )
        try:
            await client.cancel_all_orders()
            await notifier.notify("All orders cancelled. Bot halted.")
        except Exception as cancel_exc:  # noqa: BLE001
            logger.error("cancel_all_orders failed during emergency halt: %s", cancel_exc)
            await notifier.notify(f"WARNING: cancel_all_orders failed: {cancel_exc}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        await feed_registry.stop_all()
        logger.info("Flushing signal logger …")
        await sig_logger.close()
        logger.info("Flushing Telegram session …")
        await notifier.close()
        logger.info("=== Bot shutdown complete ===")


if __name__ == "__main__":
    asyncio.run(main())
