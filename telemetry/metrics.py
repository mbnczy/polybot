"""
telemetry/metrics.py
─────────────────────
Prometheus metrics exporter for the Polymarket ARB Bot.

Exposes a /metrics endpoint on METRICS_PORT (default 8000) via an aiohttp
async HTTP server task.  All metric objects are module-level singletons so
any module can import and update them without passing references around —
this is the standard prometheus_client pattern.

Prometheus scrape endpoint:
    http://localhost:8000/metrics

Metrics exposed
───────────────
  polly_usdc_balance       Gauge   Real-time USDC wallet balance (starting_balance + session_pnl)
  polly_cumulative_pnl     Gauge   Cumulative session PnL in USDC
  polly_ws_latency_ms      Gauge   Latest tick-to-strategy-loop WebSocket latency (ms)
  polly_active_markets     Gauge   Number of live WebSocket market feed tasks
  polly_match_orders_total Counter Total atomic matchOrders / execute_arb_pair calls executed

Environment
───────────
  METRICS_PORT   TCP port for the Prometheus scrape endpoint (default: 8000)
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp.web
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

METRICS_PORT: int = int(os.environ.get("METRICS_PORT", 8000))

# ── Metric definitions ────────────────────────────────────────────────────────
# Module-level singletons: import from anywhere, update in-place.

USDC_BALANCE = Gauge(
    "polly_usdc_balance",
    "Real-time USDC wallet balance (starting_balance + session_pnl)",
)
CUMULATIVE_PNL = Gauge(
    "polly_cumulative_pnl",
    "Cumulative session PnL in USDC since bot start",
)
WS_LATENCY_MS = Gauge(
    "polly_ws_latency_ms",
    "Latest WebSocket tick-to-strategy-loop delivery latency in milliseconds",
)
ACTIVE_MARKETS = Gauge(
    "polly_active_markets",
    "Number of live WebSocket market feed tasks",
)
MATCH_ORDERS_TOTAL = Counter(
    "polly_match_orders_total",
    "Total atomic matchOrders / execute_arb_pair executions",
)

# ── Scanner / scorer metrics ───────────────────────────────────────────────────

SCANNER_FEEDS_FETCHED = Counter(
    "polly_scanner_feeds_fetched_total",
    "Total binary markets fetched from the Gamma API across all scans",
)
SCANNER_SCAN_DURATION = Histogram(
    "polly_scanner_scan_duration_seconds",
    "Wall-clock time spent inside MarketScanner._scan_once() per invocation",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
SCANNER_SCORE_MIN = Gauge(
    "polly_scanner_score_min",
    "Minimum MarketScorer score observed in the last scan (scoring path only)",
)
SCANNER_SCORE_AVG = Gauge(
    "polly_scanner_score_avg",
    "Mean MarketScorer score observed in the last scan (scoring path only)",
)
SCANNER_SCORE_MAX = Gauge(
    "polly_scanner_score_max",
    "Maximum MarketScorer score observed in the last scan (scoring path only)",
)
SCANNER_CANDIDATES = Gauge(
    "polly_scanner_candidates",
    "Number of unseen binary markets found in the last scan",
)
SCANNER_ADMITTED = Gauge(
    "polly_scanner_admitted",
    "Number of markets whose on_market_added callback was fired in the last scan",
)

# ── Reliability / efficiency metrics ──────────────────────────────────────────
# Surface the new safety/efficiency paths so half-fills, unwinds, pruning, stale
# ticks and signal quality are measurable (not just logged).

ARB_DETECTED_TOTAL = Counter(
    "polly_arb_detected_total",
    "Arbitrage signals detected (taker + maker) before execution/risk gating",
)
ARB_HALF_FILLS = Counter(
    "polly_arb_half_fills_total",
    "Two-leg executions where only one leg filled (naked exposure → unwind)",
)
ARB_UNWIND_FAILURES = Counter(
    "polly_arb_unwind_failures_total",
    "Half-fill unwind attempts that failed (manual intervention required)",
)
FEEDS_PRUNED = Counter(
    "polly_feeds_pruned_total",
    "Market feeds pruned for staleness (no two-sided quote within window)",
)
STALE_TICKS_SKIPPED = Counter(
    "polly_stale_ticks_skipped_total",
    "arb_ticks skipped because their age exceeded MAX_TICK_AGE_S",
)
REAL_EDGE_BPS = Gauge(
    "polly_real_edge_bps",
    "Real (pre-rebate) Dutch-book edge of the latest detected signal, in bps "
    "[= (1 - (yes_ask + no_ask)) * 10000]",
)
EVAL_LATENCY = Histogram(
    "polly_eval_latency_seconds",
    "Hot-path latency from arb_tick dequeue to signal decision (lower = faster "
    "reaction to fleeting arbs)",
    buckets=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
)


# ── Async HTTP server ─────────────────────────────────────────────────────────

async def _metrics_handler(request: aiohttp.web.Request) -> aiohttp.web.Response:
    output = generate_latest()
    return aiohttp.web.Response(
        body=output,
        content_type=CONTENT_TYPE_LATEST,
    )


async def metrics_server(port: int = METRICS_PORT) -> None:
    """
    Long-running asyncio coroutine — start as a named Task.

    Serves the Prometheus /metrics scrape endpoint on `port` using aiohttp so
    it runs entirely within the existing event loop without spawning threads.

    Cancelling this task triggers graceful AppRunner cleanup.
    """
    app = aiohttp.web.Application()
    app.router.add_get("/metrics", _metrics_handler)

    runner = aiohttp.web.AppRunner(app, access_log=None)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Prometheus metrics server listening on :%d/metrics", port)

    try:
        # Park here until the task is cancelled by shutdown or /halt.
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info("Metrics server shutting down …")
    finally:
        await runner.cleanup()
        logger.info("Metrics server stopped")
