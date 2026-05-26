"""
backtest/simulator.py
─────────────────────
PaperExecutionEngine — mocks core/clob_client.py for event-driven backtesting.

Design contract
───────────────
  • Accepts an ArbSignal (from DutchBookPricer or ArbDetector).
  • Delays execution by a randomised 50–150 ms to simulate real-world network
    latency (TCP round-trip + Polymarket order-book processing time).
  • Computes net P&L using the April 2026 Polymarket Maker Rebate schedule
    imported directly from strategy/arbitrage.py — the exact rates that will
    be applied in live trading.
  • Maintains a running SimulatorState ledger (trades, P&L, drawdown).
  • Checks the $250 DAILY_LOSS_LIMIT after every fill and sets
    state.risk_failure = True when breached.

P&L model (maker path — Dutch Book)
─────────────────────────────────────
    gross_pnl_per_pair   = 1.0 − combined_bid         (guaranteed Dutch Book profit)
    rebate_credit_per_pair = combined_bid × rebate_rate  (maker rebate subsidy)
    net_pnl_per_pair     = gross_pnl_per_pair + rebate_credit_per_pair
    total_pnl            = net_pnl_per_pair × position_size   (USDC)

P&L model (taker path — FOK)
──────────────────────────────
    gross_pnl_per_pair   = 1.0 − combined_ask
    fee_cost_per_pair    = combined_ask × fee_rate       (taker fee deduction)
    net_pnl_per_pair     = gross_pnl_per_pair − fee_cost_per_pair
                         = signal.net_edge               (already computed)
    total_pnl            = net_pnl_per_pair × position_size

Rebate schedule (April 2026, imported from strategy/arbitrage.py)
──────────────────────────────────────────────────────────────────
  politics       1.00 %
  crypto         1.44 %
  sports         0.75 %
  entertainment  0.75 %
  economy        0.80 %
  science        0.60 %
  other/unknown  0.50 %   (DEFAULT_MAKER_REBATE conservative default)

Risk gate
─────────────────────────────────────────────────────────────────
  DAILY_LOSS_LIMIT = −$250.00 USDC
  Checked after every fill.  Once breached, state.risk_failure is set and
  the orchestrator is expected to halt immediately.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from strategy.arbitrage import (
    ArbSignal,
    DEFAULT_MAKER_REBATE,
    MAKER_REBATES,
    MAX_MAKER_REBATE,
)

logger = logging.getLogger(__name__)

# ── Risk constants — mirrors circuit_breaker.py ───────────────────────────────
DAILY_LOSS_LIMIT: float = -250.0   # USDC; triggers RISK FAILURE flag

# ── Simulated network latency window ─────────────────────────────────────────
_LATENCY_MIN_MS: float = 50.0
_LATENCY_MAX_MS: float = 150.0


# ═══════════════════════════════════════════════════════════════════════════════
# FillRecord — immutable result of one simulated execution
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FillRecord:
    """
    Result of one simulated fill.  Stored in SimulatorState.fills for later
    tear-sheet analysis.

    All monetary fields are in USDC.
    """
    signal:           ArbSignal
    fill_monotonic:   float   # time.monotonic() at fill — for latency logging
    latency_ms:       float   # sampled network latency (50–150 ms)
    is_maker:         bool    # True if signal came from DutchBookPricer

    # Rebate / fee accounting
    maker_rebate:     float   # rebate rate applied (fraction, e.g. 0.0100)
    taker_fee:        float   # taker fee applied (0.0 for maker fills)

    # P&L breakdown
    gross_pnl:        float   # 1.0 − combined_cost (pre-rebate/fee, per pair)
    rebate_credit:    float   # combined_cost × rebate_rate × size (total USDC)
    fee_debit:        float   # combined_cost × fee_rate  × size  (total USDC)
    net_pnl_per_pair: float   # per-pair net profit (USDC)
    position_size:    float   # shares (yes_size == no_size)
    total_pnl:        float   # net_pnl_per_pair × position_size (USDC)

    is_winner:        bool    # total_pnl > 0
    category:         str     # market category string from the tick


# ═══════════════════════════════════════════════════════════════════════════════
# SimulatorState — mutable ledger maintained by PaperExecutionEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulatorState:
    """
    Running ledger updated after every fill.  Read by run_backtest.py to
    produce the final tear sheet.
    """
    total_trades:    int         = 0
    winning_trades:  int         = 0
    cumulative_pnl:  float       = 0.0
    peak_pnl:        float       = 0.0      # high-water mark for drawdown calc
    max_drawdown:    float       = 0.0      # worst trough from peak (positive USDC)
    daily_pnl:       float       = 0.0      # rolling session P&L for risk gate
    risk_failure:    bool        = False     # True once DAILY_LOSS_LIMIT breached
    risk_failure_at: Optional[int] = None   # trade index when limit was hit
    fills:           list[FillRecord] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Fraction of winning trades (0.0 – 1.0)."""
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    @property
    def avg_pnl_per_trade(self) -> float:
        """Mean P&L per trade in USDC (= EV)."""
        return self.cumulative_pnl / self.total_trades if self.total_trades else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PaperExecutionEngine
# ═══════════════════════════════════════════════════════════════════════════════

class PaperExecutionEngine:
    """
    Drop-in mock for core/clob_client.py, designed for backtesting.

    Usage::

        engine = PaperExecutionEngine()

        async for tick in feed:
            signal = pricer.evaluate_maker(...)
            if signal:
                fill = await engine.execute(signal, category=tick.category)
                if engine.state.risk_failure:
                    break   # halt: daily loss limit exceeded

        state = engine.state   # read for tear sheet
    """

    def __init__(self, daily_loss_limit: float = DAILY_LOSS_LIMIT) -> None:
        if daily_loss_limit >= 0:
            raise ValueError(
                f"daily_loss_limit must be negative (got {daily_loss_limit})"
            )
        self._daily_loss_limit = daily_loss_limit
        self.state = SimulatorState()

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute(
        self,
        signal: ArbSignal,
        category: str = "other",
    ) -> FillRecord:
        """
        Simulate execution of an ArbSignal.

        Parameters
        ----------
        signal   : ArbSignal produced by DutchBookPricer.evaluate_maker() or
                   ArbDetector.evaluate().
        category : Market category string (from OrderBookTick.category) used to
                   look up the April 2026 maker rebate rate.

        Returns
        -------
        FillRecord with full P&L breakdown.  Also updates self.state.
        """
        # ── 1. Simulate network latency ───────────────────────────────────────
        latency_ms = random.uniform(_LATENCY_MIN_MS, _LATENCY_MAX_MS)
        await asyncio.sleep(latency_ms / 1_000.0)
        fill_ts = time.monotonic()

        # ── 2. Classify signal path ───────────────────────────────────────────
        is_maker = signal.is_maker_signal

        # ── 3. Resolve rebate / fee ───────────────────────────────────────────
        maker_rebate = self._resolve_maker_rebate(signal, category)
        taker_fee    = signal.fee_rate if not is_maker else 0.0

        # ── 4. P&L calculation ────────────────────────────────────────────────
        position_size = signal.yes_size   # yes_size == no_size by construction

        if is_maker:
            # Maker path: Dutch Book spread + rebate subsidy
            combined_cost = signal.combined_cost   # yes_bid + no_bid
            gross_pnl     = round(1.0 - combined_cost, 6)
            rebate_credit = round(combined_cost * maker_rebate * position_size, 6)
            fee_debit     = 0.0
            net_pnl_per_pair = round(gross_pnl + combined_cost * maker_rebate, 6)
        else:
            # Taker path: Dutch Book spread minus taker fee (pre-computed in signal)
            combined_cost    = signal.combined_cost   # yes_ask + no_ask
            gross_pnl        = round(1.0 - combined_cost, 6)
            fee_debit        = round(signal.fee_cost * position_size, 6)
            rebate_credit    = 0.0
            net_pnl_per_pair = signal.net_edge   # already net of fee

        total_pnl = round(net_pnl_per_pair * position_size, 6)

        # ── 5. Build the fill record ──────────────────────────────────────────
        fill = FillRecord(
            signal=signal,
            fill_monotonic=fill_ts,
            latency_ms=round(latency_ms, 2),
            is_maker=is_maker,
            maker_rebate=maker_rebate,
            taker_fee=taker_fee,
            gross_pnl=gross_pnl,
            rebate_credit=rebate_credit,
            fee_debit=fee_debit,
            net_pnl_per_pair=net_pnl_per_pair,
            position_size=position_size,
            total_pnl=total_pnl,
            is_winner=(total_pnl > 0.0),
            category=category,
        )

        # ── 6. Update ledger + risk gate ──────────────────────────────────────
        self._update_ledger(fill)

        logger.debug(
            "PAPER FILL | latency=%.1f ms  maker=%s  rebate=%.2f%%  "
            "gross=%.4f  credit=%.4f  net_pnl=%.6f  total=%.4f USDC  "
            "cumulative=%.4f USDC",
            latency_ms, is_maker, maker_rebate * 100,
            gross_pnl, rebate_credit, net_pnl_per_pair,
            total_pnl, self.state.cumulative_pnl,
        )

        return fill

    def reset_daily_pnl(self) -> None:
        """Reset the daily P&L accumulator (call at start of each simulated day)."""
        self.state.daily_pnl = 0.0

    # ── Rebate resolution ─────────────────────────────────────────────────────

    def _resolve_maker_rebate(self, signal: ArbSignal, category: str) -> float:
        """
        Resolve the April 2026 maker rebate rate for a fill.

        Priority:
          1. Rebate embedded in the ArbSignal (set by MakerRebateEngine at
             signal creation — most accurate).
          2. Category slug lookup against MAKER_REBATES schedule.
          3. DEFAULT_MAKER_REBATE (conservative 0.50 %).
        """
        # Pre-fetched rebate takes priority
        if signal.maker_rebate > 0.0:
            return min(signal.maker_rebate, MAX_MAKER_REBATE)

        # Category-slug lookup
        slug = category.strip().lower()
        if slug in MAKER_REBATES:
            return MAKER_REBATES[slug]

        # Partial match (e.g. "us-politics" → "politics")
        for key, rate in MAKER_REBATES.items():
            if key in slug or slug in key:
                return rate

        return DEFAULT_MAKER_REBATE

    # ── Ledger update ─────────────────────────────────────────────────────────

    def _update_ledger(self, fill: FillRecord) -> None:
        """Update SimulatorState after a fill and check the daily loss limit."""
        st = self.state

        st.total_trades   += 1
        st.cumulative_pnl  = round(st.cumulative_pnl + fill.total_pnl, 6)
        st.daily_pnl       = round(st.daily_pnl      + fill.total_pnl, 6)

        if fill.is_winner:
            st.winning_trades += 1

        st.fills.append(fill)

        # ── Update high-water mark and max drawdown ───────────────────────────
        if st.cumulative_pnl > st.peak_pnl:
            st.peak_pnl = st.cumulative_pnl

        current_drawdown = st.peak_pnl - st.cumulative_pnl
        if current_drawdown > st.max_drawdown:
            st.max_drawdown = round(current_drawdown, 6)

        # ── Daily loss limit gate ─────────────────────────────────────────────
        if st.daily_pnl <= self._daily_loss_limit and not st.risk_failure:
            st.risk_failure    = True
            st.risk_failure_at = st.total_trades
            logger.warning(
                "*** RISK FAILURE *** daily_pnl=%.4f USDC ≤ limit=%.2f USDC "
                "after trade #%d — backtest halted",
                st.daily_pnl, self._daily_loss_limit, st.total_trades,
            )
