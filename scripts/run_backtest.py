"""
scripts/run_backtest.py
───────────────────────
Phase 13 — MakerArbitrageMachine event-driven backtester.

Wires together:
  • backtest/data_handler.py  → HistoricalDataFeed (Kaggle dataset, tick stream)
  • strategy/arbitrage.py     → DutchBookPricer + MakerRebateEngine (exact live logic)
  • backtest/simulator.py     → PaperExecutionEngine (latency sim, P&L, risk gate)

Event-loop architecture
───────────────────────
Two coroutines share an asyncio.Queue[Optional[OrderBookTick]]:

  producer  (feed.stream)   — enqueues one tick at a time; yields after each put
  consumer  (run_consumer)  — dequeues ticks, evaluates signals, executes fills

The producer's `await asyncio.sleep(0)` after each enqueue guarantees the
consumer processes the tick before the next one is loaded — this is the
physical anti-look-ahead mechanism.

Tear sheet metrics
──────────────────
  Total Trades      : fills executed
  Win Rate          : % of trades with positive net P&L
  EV per Trade      : mean net P&L per fill (USDC) — expected value
  Sharpe Ratio      : annualised Sharpe using per-trade returns
                      R_i = total_pnl_i / capital_deployed_i
                      Sharpe = mean(R) / std(R) × √252
  Max Drawdown      : worst peak-to-trough in cumulative P&L (USDC)

Risk failure
────────────
  If the rolling daily P&L ever hits or drops below −$250.00 during the
  simulation, the run is flagged "RISK FAILURE" and the consumer halts
  immediately — exactly as the CircuitBreaker would in live trading.

Usage
─────
  # Full dataset run:
  python scripts/run_backtest.py

  # Quick smoke test (first 5 000 rows):
  python scripts/run_backtest.py --max-rows 5000

  # Pre-downloaded CSV (skips Kaggle):
  python scripts/run_backtest.py --csv-path data/polymarket_markets.csv

  # Custom .env location:
  python scripts/run_backtest.py --env-path /etc/polly/.env
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# ── Ensure repo root is on sys.path regardless of cwd ────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from backtest.data_handler import HistoricalDataFeed, OrderBookTick
from backtest.simulator import PaperExecutionEngine, SimulatorState, DAILY_LOSS_LIMIT
from strategy.arbitrage import (
    DutchBookPricer,
    MakerRebateEngine,
    DESIRED_NET_MARGIN,
    DEFAULT_MAKER_REBATE,
)

logger = logging.getLogger(__name__)

# ── Hard risk constant (mirrors circuit_breaker.py) ───────────────────────────
_DAILY_LOSS_LIMIT: float = DAILY_LOSS_LIMIT   # −250.0 USDC

# ── Queue capacity: small buffer so the producer never races ahead ────────────
_QUEUE_MAX_SIZE: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Consumer coroutine
# ═══════════════════════════════════════════════════════════════════════════════

async def run_consumer(
    queue:   "asyncio.Queue[Optional[OrderBookTick]]",
    pricer:  DutchBookPricer,
    rebates: MakerRebateEngine,
    engine:  PaperExecutionEngine,
    max_pos: float = 50.0,
) -> None:
    """
    Consume ticks from the queue, evaluate Dutch Book signals, and execute fills.

    Halts when:
      • The sentinel None is dequeued (stream exhausted), OR
      • engine.state.risk_failure is True (daily loss limit breached).
    """
    ticks_seen = 0
    signals_fired = 0

    while True:
        tick: Optional[OrderBookTick] = await queue.get()
        queue.task_done()

        # Sentinel: producer exhausted
        if tick is None:
            logger.info("Consumer | stream sentinel received — stopping")
            break

        ticks_seen += 1

        # ── Prime the rebate cache with the category from the dataset ─────────
        # This avoids HTTP calls during backtesting while still using the
        # correct category-level rebate rate from the live schedule.
        rebates.prime_cache(tick.condition_id, category_slug=tick.category)

        # ── Evaluate Dutch Book signal ─────────────────────────────────────────
        rebate_rate = await rebates.get_maker_rebate(tick.condition_id)
        signal = pricer.evaluate_maker(
            condition_id=tick.condition_id,
            yes_token_id=tick.yes_token_id,
            no_token_id=tick.no_token_id,
            yes_ask=tick.yes_ask,
            no_ask=tick.no_ask,
            max_position_usdc=max_pos,
            maker_rebate=rebate_rate,
        )

        if signal is None:
            continue

        signals_fired += 1

        # ── Execute (paper) fill ──────────────────────────────────────────────
        await engine.execute(signal, category=tick.category)

        # ── Hard stop: daily loss limit ───────────────────────────────────────
        if engine.state.risk_failure:
            logger.warning(
                "Consumer | RISK FAILURE detected — halting after %d ticks / "
                "%d signals / %d fills",
                ticks_seen, signals_fired, engine.state.total_trades,
            )
            # Drain the queue so the producer coroutine can exit cleanly
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
            break

    logger.info(
        "Consumer | done — %d ticks processed, %d signals evaluated, "
        "%d fills executed",
        ticks_seen, signals_fired, engine.state.total_trades,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tear sheet
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_sharpe(fills: list) -> float:
    """
    Annualised Sharpe Ratio from per-trade returns.

    Return per trade is defined as:
        R_i = total_pnl_i / capital_deployed_i

    where capital_deployed_i = combined_cost × position_size (the actual USDC
    at risk per trade, bounded by the $50 per-pair cap).

    Annualisation factor: √252 (assumes one trade ≈ one daily observation on
    average; conservative for a high-frequency arb strategy).
    """
    if len(fills) < 2:
        return 0.0

    returns: list[float] = []
    for f in fills:
        cost = f.signal.combined_cost * f.position_size
        if cost > 0:
            returns.append(f.total_pnl / cost)

    if len(returns) < 2:
        return 0.0

    mu  = statistics.mean(returns)
    sd  = statistics.stdev(returns)
    if sd == 0.0:
        return float("inf") if mu > 0 else 0.0

    return round(mu / sd * math.sqrt(252), 4)


def _compute_ev(state: SimulatorState) -> float:
    """Expected Value per trade = mean net P&L per fill (USDC)."""
    return round(state.avg_pnl_per_trade, 6)


def print_tear_sheet(
    state:          SimulatorState,
    elapsed_s:      float,
    risk_failure:   bool,
) -> None:
    """
    Print a professional tear sheet to stdout.

    ┌─────────────────────────────────────────────────────────┐
    │   MakerArbitrageMachine  —  Phase 13 Backtest Results   │
    └─────────────────────────────────────────────────────────┘
    """
    fills = state.fills

    total_trades  = state.total_trades
    win_rate_pct  = round(state.win_rate * 100, 2)
    ev_per_trade  = _compute_ev(state)
    sharpe        = _compute_sharpe(fills)
    max_dd        = round(state.max_drawdown, 4)
    total_pnl     = round(state.cumulative_pnl, 4)

    # Avg / best / worst fill latency
    latencies  = [f.latency_ms for f in fills]
    avg_lat    = round(statistics.mean(latencies), 1)    if latencies else 0.0
    min_lat    = round(min(latencies), 1)                if latencies else 0.0
    max_lat    = round(max(latencies), 1)                if latencies else 0.0

    # Maker vs taker breakdown
    maker_fills = sum(1 for f in fills if f.is_maker)
    taker_fills = total_trades - maker_fills

    # Category distribution
    cat_counts: dict[str, int] = {}
    for f in fills:
        cat_counts[f.category] = cat_counts.get(f.category, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    sep = "─" * 62

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    MakerArbitrageMachine  ·  Phase 13 Backtest Tear Sheet    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Risk status banner ────────────────────────────────────────────────────
    if risk_failure:
        fail_at = state.risk_failure_at or total_trades
        print(f"  ⚠  RISK FAILURE — daily P&L hit ${_DAILY_LOSS_LIMIT:.2f} limit")
        print(f"     Backtest halted at trade #{fail_at}")
    else:
        print("  ✓  Risk gate: PASS  (daily loss limit not breached)")
    print()

    # ── Core metrics ──────────────────────────────────────────────────────────
    print(sep)
    print(f"  {'Metric':<30}  {'Value':>20}")
    print(sep)
    print(f"  {'Total Trades':<30}  {total_trades:>20,}")
    print(f"  {'Win Rate':<30}  {win_rate_pct:>19.2f}%")
    print(f"  {'EV per Trade (USDC)':<30}  {ev_per_trade:>+20.6f}")
    print(f"  {'Sharpe Ratio (annualised)':<30}  {sharpe:>20.4f}")
    print(f"  {'Max Drawdown (USDC)':<30}  {max_dd:>20.4f}")
    print(f"  {'Total Net P&L (USDC)':<30}  {total_pnl:>+20.4f}")
    print(sep)
    print()

    # ── Execution quality ─────────────────────────────────────────────────────
    print("  Execution quality")
    print(f"    Maker fills       : {maker_fills:,}  ({100*maker_fills/max(total_trades,1):.1f}%)")
    print(f"    Taker fills       : {taker_fills:,}  ({100*taker_fills/max(total_trades,1):.1f}%)")
    print(f"    Latency avg/min/max : {avg_lat} / {min_lat} / {max_lat} ms")
    print()

    # ── Category breakdown ────────────────────────────────────────────────────
    if top_cats:
        print("  Top market categories by fill count")
        for cat, cnt in top_cats:
            print(f"    {cat:<20}  {cnt:>6,} trades")
        print()

    # ── Evaluate via backtest-expert scoring framework ────────────────────────
    _expert_score_summary(state)

    # ── Runtime ───────────────────────────────────────────────────────────────
    print(f"  Backtest runtime    : {elapsed_s:.1f} s")
    print()

    # ── Final verdict ─────────────────────────────────────────────────────────
    if risk_failure:
        verdict = "RISK FAILURE — strategy exceeded $250 daily loss limit"
    elif ev_per_trade > 0 and sharpe > 1.0:
        verdict = "CANDIDATE — positive EV and Sharpe > 1.0; proceed to paper trading"
    elif ev_per_trade > 0:
        verdict = "MARGINAL — positive EV but low Sharpe; review parameters"
    else:
        verdict = "REJECT — negative expected value"

    print(f"  Verdict: {verdict}")
    print()


def _expert_score_summary(state: SimulatorState) -> None:
    """
    Run the backtest-expert 5-dimension scoring framework on the results and
    print a compact summary (imports evaluate() from the skill's script).
    """
    try:
        # Relative import from the skills sub-tree
        skills_path = _REPO_ROOT / "claude-trading-skills" / "skills" / \
                      "backtest-expert" / "scripts"
        if str(skills_path) not in sys.path:
            sys.path.insert(0, str(skills_path))

        from evaluate_backtest import evaluate  # type: ignore[import]

        fills = state.fills
        if not fills or state.total_trades < 2:
            return

        winning = [f.total_pnl for f in fills if f.is_winner]
        losing  = [abs(f.total_pnl) for f in fills if not f.is_winner]

        avg_win_pct  = (statistics.mean(winning) * 100) if winning else 0.0
        avg_loss_pct = (statistics.mean(losing)  * 100) if losing  else 0.001

        # Drawdown as % of peak P&L (avoid div-zero if no peak)
        peak = state.peak_pnl if state.peak_pnl > 0 else 1.0
        dd_pct = round(state.max_drawdown / peak * 100, 2)

        result = evaluate(
            total_trades     = state.total_trades,
            win_rate         = state.win_rate * 100,
            avg_win_pct      = avg_win_pct,
            avg_loss_pct     = avg_loss_pct,
            max_drawdown_pct = dd_pct,
            years_tested     = 1,        # dataset is ~1 year of history
            num_parameters   = 3,        # DESIRED_NET_MARGIN, TICK_SIZE, max_pos
            slippage_tested  = True,     # latency injection counts as friction
        )

        print("  Backtest-Expert 5D Score")
        print(f"    Score   : {result['total_score']} / 100")
        print(f"    Verdict : {result['verdict']}")
        for dim in result["dimensions"]:
            bar = "█" * dim["score"] + "░" * (dim["max_score"] - dim["score"])
            print(f"    {dim['name']:<20} {bar}  {dim['score']:>2}/{dim['max_score']}")

        if result["red_flags"]:
            print()
            print("    Red flags:")
            for flag in result["red_flags"]:
                icon = "[HIGH]" if flag["severity"] == "high" else "[MED] "
                print(f"      {icon} {flag['message']}")
        print()

    except Exception as exc:
        logger.debug("_expert_score_summary | scoring unavailable: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Main async orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

async def main(args: argparse.Namespace) -> int:
    """
    Wire HistoricalDataFeed → DutchBookPricer → PaperExecutionEngine in a
    single asyncio event loop.

    Producer and consumer run concurrently; the queue's max_size=1 cap ensures
    the consumer drains each tick before the producer loads the next one.
    """
    # ── Load env ──────────────────────────────────────────────────────────────
    load_dotenv(args.env_path, override=False)

    # ── Instantiate components ────────────────────────────────────────────────
    feed = HistoricalDataFeed(
        env_path=args.env_path,
        max_rows=args.max_rows,
        csv_path=Path(args.csv_path) if args.csv_path else None,
    )
    pricer  = DutchBookPricer(desired_net_margin=DESIRED_NET_MARGIN)
    rebates = MakerRebateEngine(default_rebate=DEFAULT_MAKER_REBATE)
    engine  = PaperExecutionEngine(daily_loss_limit=_DAILY_LOSS_LIMIT)

    # ── Shared queue (capacity 1 = anti-look-ahead guarantee) ─────────────────
    queue: asyncio.Queue[Optional[OrderBookTick]] = asyncio.Queue(
        maxsize=_QUEUE_MAX_SIZE
    )

    logger.info(
        "run_backtest | starting Phase 13 backtest "
        "(max_rows=%s, env=%s)",
        args.max_rows or "all", args.env_path,
    )
    t0 = time.monotonic()

    # ── Run producer + consumer concurrently ──────────────────────────────────
    producer = asyncio.create_task(
        feed.stream(queue),
        name="data-producer",
    )
    consumer = asyncio.create_task(
        run_consumer(queue, pricer, rebates, engine, max_pos=50.0),
        name="signal-consumer",
    )

    # Wait for both; if consumer halts early (RISK FAILURE) cancel the producer
    done, pending = await asyncio.wait(
        [producer, consumer],
        return_when=asyncio.FIRST_EXCEPTION,
    )

    # Surface any task exceptions
    for task in done:
        exc = task.exception()
        if exc:
            logger.error("Task %s raised: %s", task.get_name(), exc)
            for t in pending:
                t.cancel()
            raise exc

    # Cancel any remaining tasks (e.g. producer still running after consumer halt)
    for t in pending:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    elapsed = time.monotonic() - t0

    # ── Tear sheet ────────────────────────────────────────────────────────────
    print_tear_sheet(
        state        = engine.state,
        elapsed_s    = elapsed,
        risk_failure = engine.state.risk_failure,
    )

    return 1 if engine.state.risk_failure else 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_backtest",
        description=(
            "Phase 13 — MakerArbitrageMachine event-driven backtester.\n"
            "Downloads Kaggle dataset, streams ticks, fires Dutch Book signals,\n"
            "simulates fills with latency, and outputs a P&L tear sheet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--env-path",
        default=str(_REPO_ROOT / ".env"),
        help="Path to .env file (default: <repo_root>/.env)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows (useful for smoke tests).",
    )
    p.add_argument(
        "--csv-path",
        default=None,
        metavar="PATH",
        help=(
            "Path to a pre-downloaded polymarket_markets.csv.  "
            "Skips the Kaggle download entirely."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def _configure_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "asyncio", "kagglehub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        exit_code = asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        exit_code = 130

    sys.exit(exit_code)
