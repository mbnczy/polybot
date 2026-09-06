"""
risk/circuit_breaker.py
───────────────────────
In-process circuit breaker and position guard — Arbitrage Edition.

Runs entirely in-memory with zero I/O overhead on the hot check path.

Rules enforced
──────────────
1. Hard daily loss limit  (raises CircuitBreakerTripped)
   Trips when cumulative realised PnL over the rolling 24-hour window drops
   to or below DAILY_LOSS_LIMIT (-250.0 USDC).  State is persisted to
   daily_state.json so the limit survives VPS restarts.

2. Session drawdown guard  (returns False)
   Block new orders when cumulative session PnL falls below
   -(MAX_SESSION_DRAWDOWN_PCT / 100) × starting_balance.  Default: 15 %.

3. Position count cap  (returns False)
   Block new orders when the number of open arb positions reaches MAX_POSITIONS.
   Each arb pair counts as ONE position.  Default: 5.

4. Per-trade combined cost cap  (returns False)
   Hard limit: the combined cost of any single arb pair (YES cost + NO cost)
   must not exceed MAX_ARB_PAIR_USDC (default 50.0 USDC).

5. Arbitrage position sizing via calculate_position_size()
   Computes the maximum number of paired shares N such that:

       N × (yes_ask + no_ask) ≤ MAX_ARB_PAIR_USDC   ($50.00 hard ceiling)

   Returns N rounded DOWN to 2 decimal places (Polymarket minimum increment).
   Always returns ≤ 50.0 USDC combined cost regardless of inputs.

Configuration (environment variables)
──────────────────────────────────────
  MAX_POSITIONS              (int,   default 5)
  MAX_SESSION_DRAWDOWN_PCT   (float, default 15.0)

Session reset
─────────────
Call `.reset_session(new_balance)` at the start of each trading day to set a
fresh baseline.  The 24-hour daily window is managed separately and unaffected.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hard limits ────────────────────────────────────────────────────────────────
def _daily_loss_limit_from_env() -> float:
    """
    DAILY_LOSS_LIMIT — hard daily realised-loss floor (negative USDC). Breaching
    it raises CircuitBreakerTripped → cancel-all + halt. Env-configurable so it
    can be scaled to the actual account size (a −250 default is meaningless on a
    small account). Must be negative.
    """
    raw = os.environ.get("DAILY_LOSS_LIMIT", "").strip()
    if not raw:
        return -250.0
    try:
        lim = float(raw)
    except ValueError as exc:
        raise ValueError(f"DAILY_LOSS_LIMIT={raw!r} is not a valid float") from exc
    if lim >= 0.0:
        raise ValueError(f"DAILY_LOSS_LIMIT must be negative; got {lim}")
    return lim


DAILY_LOSS_LIMIT:   float = _daily_loss_limit_from_env()  # USDC; triggers CircuitBreakerTripped


def _pair_cap_from_env() -> float:
    """
    MAX_ARB_PAIR_USDC — hard ceiling on the combined YES+NO cost of one arb
    pair.  Env-configurable (same variable the monitor scripts read); fails
    fast at import on a non-positive or non-numeric value.
    """
    raw = os.environ.get("MAX_ARB_PAIR_USDC", "").strip()
    if not raw:
        return 50.0
    try:
        cap = float(raw)
    except ValueError as exc:
        raise ValueError(f"MAX_ARB_PAIR_USDC={raw!r} is not a valid float") from exc
    if cap <= 0.0:
        raise ValueError(f"MAX_ARB_PAIR_USDC must be > 0; got {cap}")
    return cap


MAX_ARB_PAIR_USDC:  float = _pair_cap_from_env()

# Path to persisted daily state — resolves to <project_root>/daily_state.json
_DAILY_STATE_PATH: str = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "daily_state.json")
)


class CircuitBreakerTripped(Exception):
    """
    Raised (not just returned) when the hard daily loss limit is breached.

    Propagates out of strategy_loop uncaught and surfaces in main() so the
    event loop can call cancel_all_orders() and halt the bot.
    """


@dataclass
class OrderIntent:
    """Single-leg order intent (retained for API compatibility and testing)."""
    token_id:  str
    side:      str    # "BUY" | "SELL"
    price:     float
    size_usdc: float  # USDC notional of this leg


@dataclass
class ArbOrderIntent:
    """
    Describes a pending dual-leg arbitrage trade.

    combined_cost_usdc = yes_size × yes_price + no_size × no_price
    This is the value checked against the $50 hard ceiling.
    """
    condition_id:       str
    yes_token_id:       str
    no_token_id:        str
    yes_price:          float
    no_price:           float
    n_shares:           float    # shares per leg (YES == NO for symmetric arb)
    combined_cost_usdc: float    # total USDC spent on both legs


@dataclass
class _State:
    starting_balance: float
    session_pnl:      float = 0.0
    open_positions:   int   = 0
    orders_blocked:   int   = 0
    orders_passed:    int   = 0


@dataclass
class _DailyState:
    window_start: datetime.datetime   # UTC-aware; anchor of the rolling 24-h window
    daily_pnl:    float = 0.0


class CircuitBreaker:
    """
    Synchronous guard evaluated before every arb pair submission.

    Example::

        cb = CircuitBreaker(starting_balance=500.0)
        n_shares = cb.calculate_position_size(yes_ask=0.47, no_ask=0.50)
        intent = ArbOrderIntent(
            condition_id="0x…",
            yes_token_id="0xYES…",
            no_token_id="0xNO…",
            yes_price=0.47,
            no_price=0.50,
            n_shares=n_shares,
            combined_cost_usdc=n_shares * (0.47 + 0.50),
        )
        if cb.check_arb(intent):          # raises CircuitBreakerTripped if daily limit hit
            await client.execute_arb_pair(...)
            guaranteed_profit = n_shares * (1.0 - (0.47 + 0.50))
            cb.on_arb_open()
            cb.on_fill(pnl=guaranteed_profit)
    """

    def __init__(self, starting_balance: float) -> None:
        self._max_positions:   int   = int(os.environ.get("MAX_POSITIONS", 5))
        self._max_drawdown_pct: float = float(
            os.environ.get("MAX_SESSION_DRAWDOWN_PCT", 15.0)
        )
        self._state  = _State(starting_balance=starting_balance)
        self._fill_times: list[float] = []
        self._lock   = threading.Lock()
        self._daily  = self._load_daily_state()
        logger.info(
            "CircuitBreaker init | balance=%.2f max_pos=%d drawdown_pct=%.1f "
            "pair_cap=%.2f USDC daily_pnl=%.2f",
            starting_balance,
            self._max_positions,
            self._max_drawdown_pct,
            MAX_ARB_PAIR_USDC,
            self._daily.daily_pnl,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Daily state persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _load_daily_state(self) -> _DailyState:
        try:
            with open(_DAILY_STATE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            window_start = datetime.datetime.fromisoformat(raw["window_start"])
            if window_start.tzinfo is None:
                window_start = window_start.replace(tzinfo=datetime.timezone.utc)
            return _DailyState(
                window_start=window_start,
                daily_pnl=float(raw["daily_pnl"]),
            )
        except FileNotFoundError:
            logger.info("daily_state.json not found — starting fresh daily window")
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("daily_state.json corrupt (%s) — resetting daily window", exc)
        return _DailyState(
            window_start=datetime.datetime.now(datetime.timezone.utc),
            daily_pnl=0.0,
        )

    def _save_daily_state(self) -> None:
        """Atomically persist daily state. Must be called with self._lock held."""
        payload = {
            "window_start": self._daily.window_start.isoformat(),
            "daily_pnl":    round(self._daily.daily_pnl, 6),
        }
        tmp_path = _DAILY_STATE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, _DAILY_STATE_PATH)

    def _maybe_roll_window(self) -> None:
        """Roll the 24-hour window if expired. Must be called with self._lock held."""
        now = datetime.datetime.now(datetime.timezone.utc)
        age = now - self._daily.window_start
        if age >= datetime.timedelta(hours=24):
            logger.info(
                "Daily window rolled | old_pnl=%.2f window_age_h=%.1f",
                self._daily.daily_pnl,
                age.total_seconds() / 3600,
            )
            self._daily = _DailyState(window_start=now, daily_pnl=0.0)
            self._save_daily_state()

    # ──────────────────────────────────────────────────────────────────────────
    # Arbitrage position sizing
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_position_size(self, yes_ask: float, no_ask: float) -> float:
        """
        Compute the maximum number of PAIRED shares within the $50 hard ceiling.

        For each share-pair we pay (yes_ask + no_ask) USDC.
        Maximum shares:
            N = floor₂( MAX_ARB_PAIR_USDC / (yes_ask + no_ask) )

        where floor₂ rounds down to 2 decimal places (Polymarket's minimum
        order increment is 0.01 shares).

        Returns the number of shares to purchase on EACH leg (yes_size == no_size).
        Always returns a value such that N × (yes_ask + no_ask) ≤ 50.0 USDC.
        Returns 0.0 when the prices are invalid or the position is too small.
        """
        combined_cost_per_share = yes_ask + no_ask

        if combined_cost_per_share <= 0.0:
            logger.warning(
                "calculate_position_size | non-positive combined cost: %.4f",
                combined_cost_per_share,
            )
            return 0.0

        raw_shares = MAX_ARB_PAIR_USDC / combined_cost_per_share
        # math.floor to 2 d.p. ensures N × cost ≤ 50.0 exactly.
        n_shares = math.floor(raw_shares * 100) / 100.0

        if n_shares < 0.01:
            logger.warning(
                "calculate_position_size | n_shares=%.4f below minimum "
                "(yes=%.4f no=%.4f combined=%.4f cap=%.2f)",
                n_shares, yes_ask, no_ask, combined_cost_per_share, MAX_ARB_PAIR_USDC,
            )
            return 0.0

        actual_cost = round(n_shares * combined_cost_per_share, 6)
        logger.debug(
            "Position size | yes=%.4f no=%.4f combined=%.4f "
            "→ %.2f shares × %.4f = %.4f USDC (cap=%.2f)",
            yes_ask, no_ask, combined_cost_per_share,
            n_shares, combined_cost_per_share, actual_cost, MAX_ARB_PAIR_USDC,
        )
        return n_shares

    # ──────────────────────────────────────────────────────────────────────────
    # Arb-specific guard check
    # ──────────────────────────────────────────────────────────────────────────

    def check_arb(self, intent: ArbOrderIntent) -> bool:
        """
        Return True if the arb pair is safe to submit, False to block it.

        Raises CircuitBreakerTripped (does NOT return False) when the hard
        daily loss limit has been reached — this must propagate to main().

        Checks applied in order:
          1. Daily loss limit (raises on breach)
          2. Session drawdown guard
          3. Position count cap
          4. Combined pair cost vs. MAX_ARB_PAIR_USDC
        """
        with self._lock:
            self._maybe_roll_window()
            if self._daily.daily_pnl <= DAILY_LOSS_LIMIT:
                raise CircuitBreakerTripped(
                    f"Daily loss limit breached: daily_pnl={self._daily.daily_pnl:.2f} "
                    f"limit={DAILY_LOSS_LIMIT:.2f}"
                )

        reason = self._arb_blocking_reason(intent)
        if reason:
            self._state.orders_blocked += 1
            logger.warning("Arb pair BLOCKED [%s] | %s", reason, intent)
            return False

        self._state.orders_passed += 1
        return True

    def _arb_blocking_reason(self, intent: ArbOrderIntent) -> Optional[str]:
        s = self._state

        # 1. Session drawdown guard
        drawdown_limit = (self._max_drawdown_pct / 100.0) * s.starting_balance
        if s.session_pnl <= -drawdown_limit:
            return (
                f"drawdown limit hit: pnl={s.session_pnl:.2f} "
                f"limit=-{drawdown_limit:.2f}"
            )

        # 2. Position count cap (each arb pair = 1 open position)
        if s.open_positions >= self._max_positions:
            return (
                f"position cap: open={s.open_positions} max={self._max_positions}"
            )

        # 3. Combined pair cost ceiling
        if intent.combined_cost_usdc > MAX_ARB_PAIR_USDC:
            return (
                f"pair cost {intent.combined_cost_usdc:.4f} USDC exceeds "
                f"cap {MAX_ARB_PAIR_USDC:.2f} USDC"
            )

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy single-leg check (kept for API compatibility / unit tests)
    # ──────────────────────────────────────────────────────────────────────────

    def check(self, intent: OrderIntent) -> bool:
        """Single-leg order guard — delegates to check_arb internally."""
        arb_intent = ArbOrderIntent(
            condition_id="legacy",
            yes_token_id=intent.token_id,
            no_token_id="",
            yes_price=intent.price,
            no_price=0.0,
            n_shares=intent.size_usdc / max(intent.price, 1e-9),
            combined_cost_usdc=intent.size_usdc,
        )
        return self.check_arb(arb_intent)

    # ──────────────────────────────────────────────────────────────────────────
    # State mutators  (called after confirmed dual-leg fills)
    # ──────────────────────────────────────────────────────────────────────────

    def on_arb_open(self) -> None:
        """Call when an arb pair has been submitted (both legs sent)."""
        self._state.open_positions += 1

    def release_open(self) -> None:
        """
        Release an on_arb_open reservation WITHOUT booking P&L.

        Used when an arb that was reserved before dispatch (gate + on_arb_open)
        does not result in a booked fill — a half-fill that was unwound, a
        no-fill (resting maker order), or an execution error — so open_positions
        accounting stays balanced and the slot is freed.
        """
        self._state.open_positions = max(0, self._state.open_positions - 1)

    def on_open(self) -> None:
        """Alias for on_arb_open() (API compatibility)."""
        self.on_arb_open()

    def on_fill(self, pnl: float) -> None:
        """
        Call when an arb position is realised (market resolves or legs close).

        For arb trades the `pnl` passed should be the guaranteed profit:
            pnl = n_shares × (1.0 - combined_cost_per_share)

        Updates both session PnL (in-memory) and daily PnL (persisted to JSON).
        Raises CircuitBreakerTripped if the daily loss limit is now breached.
        """
        self._state.session_pnl  += pnl
        self._state.open_positions = max(0, self._state.open_positions - 1)
        # Timestamps of realised fills. The auto-tuner needs to know whether
        # signals are being CONVERTED, not merely counted: a margin loosened on
        # the strength of signals that never fill just lowers the quality bar
        # on trades that were already failing to execute.
        self._fill_times.append(time.time())
        if len(self._fill_times) > 512:
            del self._fill_times[:-512]

        with self._lock:
            self._maybe_roll_window()
            self._daily.daily_pnl += pnl
            self._save_daily_state()
            daily_pnl_snapshot = self._daily.daily_pnl

        logger.info(
            "Position closed | pnl=%.6f session_pnl=%.4f "
            "daily_pnl=%.4f open=%d",
            pnl,
            self._state.session_pnl,
            daily_pnl_snapshot,
            self._state.open_positions,
        )

        if daily_pnl_snapshot <= DAILY_LOSS_LIMIT:
            raise CircuitBreakerTripped(
                f"Daily loss limit hit on fill: daily_pnl={daily_pnl_snapshot:.2f} "
                f"limit={DAILY_LOSS_LIMIT:.2f}"
            )

    def reset_session(self, new_balance: float) -> None:
        """Reset session counters for a new trading session. Daily window unaffected."""
        self._state = _State(starting_balance=new_balance)
        logger.info("CircuitBreaker session reset | new_balance=%.2f", new_balance)

    # ──────────────────────────────────────────────────────────────────────────
    # Observability
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_tripped(self) -> bool:
        """True when the session drawdown limit has been reached."""
        s = self._state
        return s.session_pnl <= -(self._max_drawdown_pct / 100.0) * s.starting_balance

    def fills_since(self, ts: float) -> int:
        """Realised fills at or after `ts` (unix seconds)."""
        return sum(1 for t in self._fill_times if t >= ts)

    def status_dict(self) -> dict:
        with self._lock:
            daily_pnl          = self._daily.daily_pnl
            daily_window_start = self._daily.window_start.isoformat()
        s = self._state
        return {
            "starting_balance":  s.starting_balance,
            "session_pnl":       round(s.session_pnl, 4),
            "daily_pnl":         round(daily_pnl, 4),
            "daily_loss_limit":  DAILY_LOSS_LIMIT,
            "daily_window_start": daily_window_start,
            "open_positions":    s.open_positions,
            "pair_cap_usdc":     MAX_ARB_PAIR_USDC,
            "orders_passed":     s.orders_passed,
            "orders_blocked":    s.orders_blocked,
            "tripped":           self.is_tripped,
        }
