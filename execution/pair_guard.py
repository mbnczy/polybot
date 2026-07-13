"""
execution/pair_guard.py
───────────────────────
MakerPairGuard — one-leg (naked-leg) protection for the GTC maker arb path.

Why this exists
───────────────
`PolyClient.execute_arb_maker_pair()` places two INDEPENDENT GTC limit orders
(synthetic post-only bids one tick below each best ask).  Unlike the FOK taker
path and the on-chain matchOrders bundle, nothing guarantees both legs fill:
if the market moves away from one bid, only the other leg fills and the bot is
left holding a naked directional position while the stale order keeps resting
on the book.

Lifecycle of a watched pair
───────────────────────────
  watch_pair()          ← called from strategy_loop after a maker submission
                          that did not confirm both legs at ack time
  poll (PAIR_GUARD_POLL_S, default 1 s)
    │
    ├─ both legs fully matched            → book P&L, hand off to
    │                                       InventoryManager for mergePositions
    │
    ├─ legs imbalanced (one filled more than the other) for longer than
    │  HEDGE_TIMEOUT_S (default 8 s)      → cancel the lagging order, then:
    │      • COMPLETE the pair as a taker when the missing leg's current ask
    │        still yields combined cost < $1 (locks the arb, small profit), or
    │      • UNWIND the naked shares at market (bounded loss) otherwise
    │
    └─ neither leg matched after MAKER_ORDER_TTL_S (default 45 s)
                                          → cancel both orders, release the
                                            circuit-breaker reservation
                                            (signal went stale; no exposure)

Circuit-breaker accounting contract
───────────────────────────────────
The strategy loop reserves one position slot (`breaker.on_arb_open()`) before
dispatching an execution.  When a maker pair is handed to this guard, the
reservation is KEPT (resting orders are live exposure) and this guard owns its
release: `on_fill()` when any paired shares are booked, `release_open()` when
the pair dissolves with no position.

Environment variables
─────────────────────
  PAIR_GUARD_POLL_S    poll cadence in seconds            (default 1.0)
  HEDGE_TIMEOUT_S      max seconds a one-leg imbalance may persist before the
                       guard hedges/unwinds                (default 8.0)
  MAKER_ORDER_TTL_S    max seconds an entirely-unfilled pair may rest before
                       both orders are cancelled           (default 45.0)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from risk.circuit_breaker import CircuitBreakerTripped
from strategy.arbitrage import DEFAULT_TAKER_FEE
from telemetry.metrics import ARB_HALF_FILLS, ARB_UNWIND_FAILURES

if TYPE_CHECKING:
    from core.clob_client import PolyClient
    from execution.inventory_manager import InventoryManager
    from risk.circuit_breaker import CircuitBreaker
    from strategy.arbitrage import ArbSignal
    from telemetry.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PAIR_GUARD_POLL_S: float = float(os.environ.get("PAIR_GUARD_POLL_S", 1.0))
HEDGE_TIMEOUT_S:   float = float(os.environ.get("HEDGE_TIMEOUT_S", 8.0))
MAKER_ORDER_TTL_S: float = float(os.environ.get("MAKER_ORDER_TTL_S", 45.0))

# Shares are quoted to 2 d.p.; anything below half an increment is noise.
_SHARE_EPS: float = 0.005

# Order states considered still-resting on the CLOB.
_OPEN_STATUSES:   frozenset[str] = frozenset({"live", "open", "pending", "delayed", "new"})
_FILLED_STATUSES: frozenset[str] = frozenset({"matched", "filled", "paper"})


@dataclass
class _Leg:
    """Mutable per-leg fill state for one side of a watched maker pair."""
    label:            str     # "YES" | "NO"
    token_id:         str
    order_id:         str
    bid:              float   # limit price the GTC order rests at
    size:             float   # shares requested
    matched:          float = 0.0
    open:             bool  = True
    cancel_requested: bool  = False

    @property
    def fully_matched(self) -> bool:
        return self.matched >= self.size - _SHARE_EPS


@dataclass
class _WatchedPair:
    pair_id:         str                   # unique key — one condition may be
                                           # watched more than once over time
    condition_id:    str
    signal:          "ArbSignal"
    yes:             _Leg
    no:              _Leg
    created_at:      float                 # time.monotonic() at registration
    imbalance_since: float | None = None   # monotonic ts when legs first diverged
    finalizing:      bool = field(default=False)

    @property
    def paired(self) -> float:
        return min(self.yes.matched, self.no.matched)

    @property
    def legs(self) -> tuple[_Leg, _Leg]:
        return (self.yes, self.no)


class MakerPairGuard:
    """
    Watches resting GTC maker pairs and guarantees no leg is ever left naked
    or resting unattended.

    Usage (in main.py)::

        guard = MakerPairGuard(client, breaker, notifier, inventory=inventory)
        asyncio.create_task(guard.run(), name="pair_guard")

        # strategy_loop, after execute_arb_maker_pair() when fill_state != "both":
        guard.watch_pair(arb_signal, n_shares, yes_resp, no_resp)
    """

    def __init__(
        self,
        clob_client:     "PolyClient",
        circuit_breaker: "CircuitBreaker",
        notifier:        "TelegramNotifier",
        inventory:       "InventoryManager | None" = None,
        poll_interval_s: float = PAIR_GUARD_POLL_S,
        hedge_timeout_s: float = HEDGE_TIMEOUT_S,
        order_ttl_s:     float = MAKER_ORDER_TTL_S,
        taker_fee_est:   float = DEFAULT_TAKER_FEE,
    ) -> None:
        self._client    = clob_client
        self._breaker   = circuit_breaker
        self._notifier  = notifier
        self._inventory = inventory
        self._poll_s    = max(0.05, poll_interval_s)
        self._hedge_s   = hedge_timeout_s
        self._ttl_s     = order_ttl_s
        self._fee_est   = taker_fee_est
        self._pairs: dict[str, _WatchedPair] = {}
        logger.info(
            "MakerPairGuard init | poll=%.2fs hedge_timeout=%.1fs order_ttl=%.1fs",
            self._poll_s, self._hedge_s, self._ttl_s,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────────────────────────

    def watch_pair(
        self,
        signal:   "ArbSignal",
        n_shares: float,
        yes_resp: dict | None,
        no_resp:  dict | None,
    ) -> None:
        """
        Start watching a maker pair whose legs were NOT both confirmed filled
        at submission-ack time.

        The caller must NOT release the circuit-breaker reservation — the guard
        owns it from this point (on_fill / release_open at finalisation).
        """
        yes = self._leg_from_resp("YES", signal.yes_token_id, signal.yes_bid,
                                  n_shares, yes_resp)
        no  = self._leg_from_resp("NO",  signal.no_token_id,  signal.no_bid,
                                  n_shares, no_resp)
        # Unique key: NEVER key by condition_id alone — a second watch on the
        # same market would silently overwrite (and orphan) the first pair.
        pair_id = f"{signal.condition_id[:16]}-{time.monotonic_ns()}"
        pair = _WatchedPair(
            pair_id=pair_id,
            condition_id=signal.condition_id,
            signal=signal,
            yes=yes,
            no=no,
            created_at=time.monotonic(),
        )
        self._pairs[pair_id] = pair
        logger.info(
            "PairGuard watching | condition=%s shares=%.2f "
            "yes[%s open=%s matched=%.2f] no[%s open=%s matched=%.2f]",
            signal.condition_id[:16], n_shares,
            yes.order_id[:12], yes.open, yes.matched,
            no.order_id[:12], no.open, no.matched,
        )

    @staticmethod
    def _leg_from_resp(
        label: str, token_id: str, bid: float, size: float, resp: dict | None
    ) -> _Leg:
        """Seed a leg's state from the submission-ack response."""
        order_id = str((resp or {}).get("order_id") or (resp or {}).get("orderID") or "")
        status   = str((resp or {}).get("status", "")).strip().lower()
        if status in _FILLED_STATUSES:
            # Leg crossed at submission (rare for post-only bids, but possible).
            return _Leg(label, token_id, order_id, bid, size,
                        matched=size, open=False)
        return _Leg(label, token_id, order_id, bid, size)

    @property
    def watched_count(self) -> int:
        return len(self._pairs)

    def is_watching(self, condition_id: str) -> bool:
        """True while any pair on this market is still being resolved.

        The strategy loop uses this to refuse a new entry on a market whose
        previous maker pair is unresolved (prevents pyramiding exposure and
        double-booking against the same complementary set)."""
        return any(p.condition_id == condition_id for p in self._pairs.values())

    # ──────────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Long-running poll loop.  Run as an asyncio.Task."""
        logger.info("MakerPairGuard started")
        try:
            while True:
                await asyncio.sleep(self._poll_s)
                try:
                    await self.poll_once()
                except (asyncio.CancelledError, CircuitBreakerTripped):
                    raise   # kill switch must propagate to main()'s gather
                except Exception as exc:  # noqa: BLE001
                    logger.error("MakerPairGuard poll error: %s", exc)
        except asyncio.CancelledError:
            await self._cancel_all_resting()
            logger.info("MakerPairGuard stopped")
            raise

    async def poll_once(self) -> None:
        """One reconciliation pass over all watched pairs (public for tests)."""
        for pair in list(self._pairs.values()):
            if pair.finalizing:
                continue
            await self._update_pair(pair)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-pair reconciliation
    # ──────────────────────────────────────────────────────────────────────────

    async def _update_pair(self, pair: _WatchedPair) -> None:
        for leg in pair.legs:
            if leg.open:
                await self._refresh_leg(leg)

        now = time.monotonic()
        yes, no = pair.yes, pair.no

        # Case 1 — both legs fully matched → hedged Dutch Book, settle it.
        if yes.fully_matched and no.fully_matched:
            await self._finalize(pair)
            return

        # Case 2 — both legs closed (filled/cancelled) but not fully paired.
        if not yes.open and not no.open:
            await self._finalize(pair)
            return

        # Case 3 — legs imbalanced: one side holds more shares than the other.
        imbalance = abs(yes.matched - no.matched)
        if imbalance > _SHARE_EPS:
            if pair.imbalance_since is None:
                pair.imbalance_since = now
                logger.warning(
                    "PairGuard | one-leg imbalance on %s: yes=%.2f no=%.2f — "
                    "hedge window %.1fs started",
                    pair.condition_id[:16], yes.matched, no.matched, self._hedge_s,
                )
            elif now - pair.imbalance_since >= self._hedge_s:
                # Grace period over: pull the lagging order and resolve exposure.
                lagging = yes if yes.matched < no.matched else no
                if lagging.open:
                    await self._cancel_leg(pair, lagging)
                # Re-check once — the cancel may have raced a fill.
                if lagging.open:
                    await self._refresh_leg(lagging)
                if not lagging.open:
                    await self._finalize(pair)
            return

        # Legs are balanced again (e.g. the second leg caught up) — reset timer.
        pair.imbalance_since = None

        # Case 4 — nothing filled at all and the signal has gone stale.
        if (
            yes.matched <= _SHARE_EPS
            and no.matched <= _SHARE_EPS
            and now - pair.created_at >= self._ttl_s
        ):
            logger.info(
                "PairGuard | TTL expired with no fills on %s — cancelling both legs",
                pair.condition_id[:16],
            )
            for leg in pair.legs:
                if leg.open:
                    await self._cancel_leg(pair, leg)
            for leg in pair.legs:
                if leg.open:
                    await self._refresh_leg(leg)
            if not yes.open and not no.open:
                await self._finalize(pair)

    async def _refresh_leg(self, leg: _Leg) -> None:
        """Pull current order state from the CLOB and update the leg."""
        if not leg.order_id:
            # No order id in the ack — we cannot poll it; treat as closed with
            # whatever we knew at registration so the pair can still finalize.
            leg.open = False
            return
        try:
            order = await self._client.get_order_status(leg.order_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PairGuard | get_order_status(%s) failed: %s", leg.order_id[:12], exc
            )
            return

        if order is None:
            # Order no longer tracked by the CLOB: it either fully filled or
            # was cancelled.  If we did not request the cancel, assume FILLED —
            # over-estimating exposure is safe (we hedge), under-estimating is not.
            leg.open = False
            if not leg.cancel_requested:
                if leg.matched < leg.size - _SHARE_EPS:
                    logger.warning(
                        "PairGuard | order %s vanished without cancel — "
                        "assuming fully filled (%.2f shares)",
                        leg.order_id[:12], leg.size,
                    )
                leg.matched = leg.size
            return

        status  = str(order.get("status", "")).strip().lower()
        matched = order.get("size_matched", order.get("sizeMatched", None))
        try:
            if matched is not None:
                leg.matched = max(leg.matched, float(matched))
        except (TypeError, ValueError):
            pass

        if status in _FILLED_STATUSES:
            leg.matched = max(leg.matched, leg.size)
            leg.open = False
        elif status in _OPEN_STATUSES:
            leg.open = True
        else:  # cancelled / unknown terminal state — keep matched as reported
            leg.open = False

    async def _cancel_leg(self, pair: _WatchedPair, leg: _Leg) -> None:
        leg.cancel_requested = True
        try:
            await self._client.cancel_order(leg.order_id)
            logger.info(
                "PairGuard | cancelled %s leg %s on %s",
                leg.label, leg.order_id[:12], pair.condition_id[:16],
            )
            leg.open = False
        except Exception as exc:  # noqa: BLE001
            # Cancel can fail because the order just filled — next refresh
            # resolves it either way.
            logger.warning(
                "PairGuard | cancel %s leg %s failed (may have filled): %s",
                leg.label, leg.order_id[:12], exc,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Finalisation — book pairs, hedge or unwind any naked excess
    # ──────────────────────────────────────────────────────────────────────────

    async def _finalize(self, pair: _WatchedPair) -> None:
        """
        Terminal reconciliation once no leg is resting any more (or both are
        fully matched).  Exactly one of on_fill / release_open is called so the
        circuit-breaker reservation taken by the strategy loop is balanced.
        """
        pair.finalizing = True
        self._pairs.pop(pair.pair_id, None)

        yes, no = pair.yes, pair.no
        # Make sure nothing is left resting on the book.
        for leg in pair.legs:
            if leg.open:
                await self._cancel_leg(pair, leg)

        paired  = pair.paired
        rich    = yes if yes.matched > no.matched else no
        deficit = no  if rich is yes else yes
        excess  = rich.matched - deficit.matched
        pnl     = paired * (1.0 - (yes.bid + no.bid))

        hedged_note = ""
        if excess > _SHARE_EPS:
            ARB_HALF_FILLS.inc()
            completed_pnl = await self._complete_or_unwind(pair, rich, deficit, excess)
            if completed_pnl is not None:
                pnl += completed_pnl
                paired += excess
                hedged_note = f" ({excess:.2f} shares completed as taker)"
            else:
                hedged_note = f" ({excess:.2f} naked shares unwound)"

        if paired > _SHARE_EPS:
            # Books P&L and releases the position reservation.
            self._breaker.on_fill(pnl=round(pnl, 6))
            await self._notifier.notify(
                f"✅ Maker pair settled on {pair.condition_id[:16]} — "
                f"{paired:.2f} paired shares, pnl={pnl:+.4f} USDC{hedged_note}"
            )
            self._register_settlement(pair, paired)
        else:
            self._breaker.release_open()
            logger.info(
                "PairGuard | pair on %s dissolved with no position%s",
                pair.condition_id[:16], hedged_note,
            )
            if hedged_note:
                await self._notifier.notify(
                    f"⚠️ Maker pair on {pair.condition_id[:16]} dissolved — "
                    f"no profit booked{hedged_note}"
                )

    async def _complete_or_unwind(
        self,
        pair:    _WatchedPair,
        rich:    _Leg,
        deficit: _Leg,
        excess:  float,
    ) -> float | None:
        """
        Resolve `excess` naked shares held on the `rich` leg.

        Preferred: BUY the missing `deficit` shares as an FOK taker when the
        current ask still keeps the completed pairs at combined cost < $1
        (locks a guaranteed non-negative payoff).  Fallback: market-sell the
        naked shares (`unwind_leg`) to flatten directional risk.

        Returns the P&L of the completed pairs, or None when unwound instead.
        """
        excess = round(excess, 2)
        ask = await self._best_ask(deficit.token_id)
        if ask is not None:
            completion_cost = rich.bid + ask * (1.0 + self._fee_est)
            if completion_cost < 1.0:
                try:
                    resp = await self._client.post_order(
                        token_id=deficit.token_id,
                        side="BUY",
                        price=ask,
                        size=excess,
                    )
                    status = str(resp.get("status", "")).strip().lower()
                    if status in _FILLED_STATUSES:
                        logger.info(
                            "PairGuard | HEDGE COMPLETED %s: bought %.2f %s @ %.4f "
                            "(pair cost %.4f < 1.0)",
                            pair.condition_id[:16], excess, deficit.label,
                            ask, completion_cost,
                        )
                        return round(excess * (1.0 - completion_cost), 6)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "PairGuard | taker completion failed on %s: %s — unwinding",
                        pair.condition_id[:16], exc,
                    )

        # Completion not possible/profitable — flatten the naked shares.
        try:
            await self._client.unwind_leg(rich.token_id, excess)
            logger.warning(
                "PairGuard | UNWOUND %.2f naked %s shares on %s",
                excess, rich.label, pair.condition_id[:16],
            )
        except Exception as exc:  # noqa: BLE001
            ARB_UNWIND_FAILURES.inc()
            logger.error(
                "PairGuard | unwind failed on %s: %s", pair.condition_id[:16], exc
            )
            await self._notifier.send_critical_error(
                f"PAIR GUARD UNWIND FAILED {pair.condition_id[:16]} — "
                f"{excess:.2f} naked {rich.label} shares — MANUAL INTERVENTION REQUIRED"
            )
        return None

    async def _best_ask(self, token_id: str) -> float | None:
        try:
            book = await self._client.get_orderbook(token_id)
            asks = book.get("asks", [])
            if not asks:
                return None
            return min(
                float(a["price"] if isinstance(a, dict) else a) for a in asks
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PairGuard | orderbook fetch failed for %s: %s",
                           token_id[:12], exc)
            return None

    def _register_settlement(self, pair: _WatchedPair, paired: float) -> None:
        """Hand the complementary set to the InventoryManager for merging."""
        if self._inventory is None:
            return
        if paired >= pair.signal.yes_size - _SHARE_EPS:
            self._inventory.register_paired_fill(pair.signal)
        else:
            # Partial pair — merge exactly the paired share count.  Run in a
            # wrapper so a merge failure is loudly reported instead of dying
            # as an unretrieved task exception.
            async def _merge_partial(cid: str = pair.condition_id,
                                     shares: float = paired) -> None:
                try:
                    await self._inventory.merge_complementary_set(cid, shares)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "PairGuard | partial merge FAILED condition=%s "
                        "shares=%.2f: %s", cid[:16], shares, exc,
                    )
                    await self._notifier.send_critical_error(
                        f"PARTIAL MERGE FAILED {cid[:16]} — {shares:.2f} shares "
                        f"stuck as CTF tokens (redeemable at resolution): {exc}"
                    )
            asyncio.ensure_future(_merge_partial())

    async def _cancel_all_resting(self) -> None:
        """Shutdown hygiene: never leave watched GTC orders on the book."""
        for pair in list(self._pairs.values()):
            for leg in pair.legs:
                if leg.open and leg.order_id:
                    try:
                        await self._client.cancel_order(leg.order_id)
                        logger.info(
                            "PairGuard shutdown | cancelled %s leg %s",
                            leg.label, leg.order_id[:12],
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "PairGuard shutdown | cancel %s failed: %s",
                            leg.order_id[:12], exc,
                        )
