"""
execution/negrisk_guard.py
──────────────────────────
NegRiskBundleGuard — partial-fill protection for the N-leg NegRisk CLOB path.

Why this exists
───────────────
`PolyClient.execute_negrisk_clob_bundle()` places N INDEPENDENT GTC limit orders
(synthetic post-only bids, one per selected outcome).  The on-chain matchOrders
bundle it replaces was atomic; N limit orders are not.  arXiv:2508.03474 §6
states the risk plainly: "since placing multiple orders in an order book is
non-atomic (only a subset of the attempts may succeed), there is some inherent
risk to attempting arbitrage."

Why a partial bundle is not a smaller arbitrage
───────────────────────────────────────────────
Holding one NO share on each of M mutually exclusive outcomes guarantees `M−1`
at expiry (the winner's NO pays nothing, every other NO pays $1).  Fill only
M' < M of them and the guarantee drops to `M'−1` while the cost stays whatever
those M' legs cost — the edge is `Σ implied_yes` over the filled legs minus 1,
which turns negative as legs drop out.  Two legs of a four-leg bundle is a
directional bet, not an arb.  So unlike MakerPairGuard, which can often complete
a half-filled pair as a taker, this guard's default resolution for an incomplete
bundle is to flatten everything that did fill.

Lifecycle of a watched bundle
─────────────────────────────
  watch_bundle()        ← called from strategy_loop after submission
  poll (NEGRISK_GUARD_POLL_S, default 1 s)
    │
    ├─ every leg fully matched                → book the guaranteed profit,
    │                                           register for settlement
    │
    ├─ some legs matched, others not, for longer than
    │  NEGRISK_BUNDLE_TIMEOUT_S (default 10 s)
    │                                         → cancel the unfilled legs, then
    │                                           unwind every filled leg at market
    │                                           (bounded loss, no naked exposure)
    │
    └─ nothing matched after NEGRISK_ORDER_TTL_S (default 45 s)
                                              → cancel all orders, release the
                                                circuit-breaker reservation

Circuit-breaker accounting contract
───────────────────────────────────
Identical to MakerPairGuard: the strategy loop reserves one slot via
`breaker.on_arb_open()` before dispatch; this guard then owns the release —
`on_fill()` when anything is realised, `release_open()` when the bundle
dissolves without exposure.  Exactly one of the two is called per bundle.

Environment variables
─────────────────────
  NEGRISK_GUARD_POLL_S       poll cadence in seconds                (default 1.0)
  NEGRISK_BUNDLE_TIMEOUT_S   max seconds an incomplete bundle may persist before
                             the guard cancels and unwinds           (default 10.0)
  NEGRISK_ORDER_TTL_S        max seconds a fully-unfilled bundle may rest before
                             every order is cancelled                (default 45.0)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from telemetry.metrics import ARB_HALF_FILLS, ARB_UNWIND_FAILURES

if TYPE_CHECKING:
    from core.clob_client import PolyClient
    from risk.circuit_breaker import CircuitBreaker
    from strategy.arbitrage import NegRiskSignal
    from telemetry.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
NEGRISK_GUARD_POLL_S:     float = float(os.environ.get("NEGRISK_GUARD_POLL_S", 1.0))
NEGRISK_BUNDLE_TIMEOUT_S: float = float(os.environ.get("NEGRISK_BUNDLE_TIMEOUT_S", 10.0))
NEGRISK_ORDER_TTL_S:      float = float(os.environ.get("NEGRISK_ORDER_TTL_S", 45.0))

# Shares are quoted to 2 d.p.; anything below half an increment is noise.
_SHARE_EPS: float = 0.005

_OPEN_STATUSES:   frozenset[str] = frozenset({"live", "open", "pending", "delayed", "new"})
_FILLED_STATUSES: frozenset[str] = frozenset({"matched", "filled", "paper"})


@dataclass
class _BundleLegState:
    """Mutable per-leg fill state for one outcome of a watched bundle."""
    idx:              int
    token_id:         str
    order_id:         str
    bid:              float
    size:             float
    matched:          float = 0.0
    open:             bool  = True
    cancel_requested: bool  = False

    @property
    def fully_matched(self) -> bool:
        return self.matched >= self.size - _SHARE_EPS


@dataclass
class _WatchedBundle:
    bundle_id:    str
    condition_id: str            # negRiskMarketID of the group
    signal:       "NegRiskSignal"
    legs:         list[_BundleLegState]
    created_at:   float
    imbalance_since: float | None = None
    finalizing:      bool = field(default=False)

    @property
    def all_matched(self) -> bool:
        return all(leg.fully_matched for leg in self.legs)

    @property
    def any_matched(self) -> bool:
        return any(leg.matched > _SHARE_EPS for leg in self.legs)

    @property
    def any_open(self) -> bool:
        return any(leg.open for leg in self.legs)


class NegRiskBundleGuard:
    """
    Polls submitted NegRisk bundles and resolves anything that does not fill
    completely.

    Usage::

        guard = NegRiskBundleGuard(client, breaker, notifier)
        asyncio.create_task(guard.run(), name="negrisk-guard")
        ...
        guard.watch_bundle(signal, responses)
    """

    def __init__(
        self,
        client:   "PolyClient",
        breaker:  "CircuitBreaker",
        notifier: "TelegramNotifier",
        *,
        poll_interval:  float = NEGRISK_GUARD_POLL_S,
        bundle_timeout: float = NEGRISK_BUNDLE_TIMEOUT_S,
        order_ttl:      float = NEGRISK_ORDER_TTL_S,
    ) -> None:
        self._client   = client
        self._breaker  = breaker
        self._notifier = notifier
        self._poll     = max(0.1, poll_interval)
        self._timeout  = max(0.0, bundle_timeout)
        self._ttl      = max(0.0, order_ttl)
        self._bundles: dict[str, _WatchedBundle] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────────────────────────

    def watch_bundle(
        self,
        signal:    "NegRiskSignal",
        responses: list[dict],
    ) -> None:
        """
        Start watching a submitted bundle.

        The caller must NOT release the circuit-breaker reservation — the guard
        owns it from here (on_fill / release_open at finalisation).
        """
        legs: list[_BundleLegState] = []
        for i, (leg, resp) in enumerate(zip(signal.legs, responses)):
            resp = resp or {}
            order_id = str(resp.get("order_id") or resp.get("orderID") or "")
            status   = str(resp.get("status", "")).strip().lower()
            if status == "error":
                # Never reached the book — nothing to poll, nothing to cancel.
                legs.append(_BundleLegState(
                    i, leg.token_id, order_id, leg.no_bid, leg.size, open=False,
                ))
                continue
            matched = leg.size if status in _FILLED_STATUSES else 0.0
            legs.append(_BundleLegState(
                i, leg.token_id, order_id, leg.no_bid, leg.size,
                matched=matched, open=status not in _FILLED_STATUSES,
            ))

        bundle_id = f"{signal.condition_id[:16]}-{time.monotonic_ns()}"
        self._bundles[bundle_id] = _WatchedBundle(
            bundle_id=bundle_id,
            condition_id=signal.condition_id,
            signal=signal,
            legs=legs,
            created_at=time.monotonic(),
        )
        logger.info(
            "NegRiskGuard watching | group=%s legs=%d bundles=%.2f "
            "accepted=%d filled_at_ack=%d",
            signal.condition_id[:16], len(legs), signal.n_bundles,
            sum(1 for l in legs if l.order_id),
            sum(1 for l in legs if l.fully_matched),
        )

    @property
    def watched_count(self) -> int:
        return len(self._bundles)

    def is_watching(self, condition_id: str) -> bool:
        """True while any bundle on this group is still unresolved."""
        return any(b.condition_id == condition_id for b in self._bundles.values())

    # ──────────────────────────────────────────────────────────────────────────
    # Poll loop
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info(
            "NegRiskBundleGuard started | poll=%.1fs timeout=%.1fs ttl=%.1fs",
            self._poll, self._timeout, self._ttl,
        )
        try:
            while True:
                await asyncio.sleep(self._poll)
                try:
                    await self.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("NegRiskBundleGuard poll error: %s", exc)
        except asyncio.CancelledError:
            logger.info("NegRiskBundleGuard stopped")
            raise

    async def poll_once(self) -> None:
        for bundle in list(self._bundles.values()):
            if bundle.finalizing:
                continue
            await self._update_bundle(bundle)

    async def _update_bundle(self, bundle: _WatchedBundle) -> None:
        await asyncio.gather(*(
            self._refresh_leg(leg) for leg in bundle.legs if leg.open
        ))

        now = time.monotonic()
        age = now - bundle.created_at

        if bundle.all_matched:
            await self._finalize(bundle, complete=True)
            return

        if bundle.any_matched:
            # Incomplete bundle with real exposure — start (or continue) the
            # imbalance clock.
            if bundle.imbalance_since is None:
                bundle.imbalance_since = now
            # Checked in the same pass that arms the clock, so a timeout of 0
            # means "tolerate no imbalance at all" rather than "one free poll".
            if now - bundle.imbalance_since >= self._timeout:
                await self._finalize(bundle, complete=False)
            return

        # Nothing filled anywhere.
        if not bundle.any_open or age >= self._ttl:
            await self._finalize(bundle, complete=False)

    async def _refresh_leg(self, leg: _BundleLegState) -> None:
        """Pull current order state from the CLOB and update the leg."""
        if not leg.order_id:
            leg.open = False
            return
        try:
            order = await self._client.get_order_status(leg.order_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "NegRiskGuard | get_order_status(%s) failed: %s",
                leg.order_id[:12], exc,
            )
            return

        if order is None:
            # Untracked by the CLOB: filled or cancelled.  Absent an explicit
            # cancel from us, assume FILLED — over-estimating exposure makes us
            # unwind unnecessarily, under-estimating leaves naked shares.
            leg.open = False
            if not leg.cancel_requested:
                if leg.matched < leg.size - _SHARE_EPS:
                    logger.warning(
                        "NegRiskGuard | order %s vanished without cancel — "
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
        else:
            leg.open = False

    async def _cancel_leg(self, bundle: _WatchedBundle, leg: _BundleLegState) -> None:
        if not leg.order_id:
            leg.open = False
            return
        leg.cancel_requested = True
        try:
            await self._client.cancel_order(leg.order_id)
            logger.info(
                "NegRiskGuard | cancelled leg[%d] %s on %s",
                leg.idx, leg.order_id[:12], bundle.condition_id[:16],
            )
            leg.open = False
        except Exception as exc:  # noqa: BLE001
            # Cancel can fail because the order just filled — the next refresh
            # resolves it either way.
            logger.warning(
                "NegRiskGuard | cancel leg[%d] %s failed (may have filled): %s",
                leg.idx, leg.order_id[:12], exc,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Finalisation
    # ──────────────────────────────────────────────────────────────────────────

    async def _finalize(self, bundle: _WatchedBundle, *, complete: bool) -> None:
        """
        Terminal reconciliation.  Exactly one of on_fill / release_open runs so
        the circuit-breaker reservation taken by the strategy loop is balanced.
        """
        bundle.finalizing = True
        self._bundles.pop(bundle.bundle_id, None)

        # Nothing may be left resting on the book.
        for leg in bundle.legs:
            if leg.open:
                await self._cancel_leg(bundle, leg)

        sig = bundle.signal

        if complete:
            # Every leg filled: the bundle is the guaranteed arb the detector
            # priced. net_edge is per bundle and already net of the rebate.
            filled  = min(leg.matched for leg in bundle.legs)
            pnl     = round(filled * sig.net_edge, 6)
            self._breaker.on_fill(pnl=pnl)
            logger.info(
                "NegRiskGuard | BUNDLE COMPLETE %s | legs=%d bundles=%.2f "
                "pnl=%+.4f USDC",
                bundle.condition_id[:16], len(bundle.legs), filled, pnl,
            )
            await self._notifier.notify(
                f"✅ NegRisk bundle on {bundle.condition_id[:16]} — "
                f"{len(bundle.legs)} legs × {filled:.2f}, pnl={pnl:+.4f} USDC"
            )
            return

        naked = [leg for leg in bundle.legs if leg.matched > _SHARE_EPS]
        if not naked:
            # Bundle dissolved with no exposure — pure release, nothing booked.
            logger.info(
                "NegRiskGuard | bundle on %s expired unfilled — releasing slot",
                bundle.condition_id[:16],
            )
            self._breaker.release_open()
            return

        # Incomplete with exposure: flatten every filled leg.  See the module
        # docstring for why a partial bundle cannot simply be held.
        ARB_HALF_FILLS.inc()
        realised, failures = await self._unwind_all(bundle, naked)

        self._breaker.on_fill(pnl=round(realised, 6))
        note = f" ({failures} leg(s) STUCK — manual)" if failures else ""
        logger.warning(
            "NegRiskGuard | BUNDLE INCOMPLETE %s | %d/%d legs filled — "
            "unwound, realised=%+.4f USDC%s",
            bundle.condition_id[:16], len(naked), len(bundle.legs),
            realised, note,
        )
        await self._notifier.notify(
            f"🔻 NegRisk bundle on {bundle.condition_id[:16]} incomplete — "
            f"{len(naked)}/{len(bundle.legs)} legs filled, unwound, "
            f"pnl={realised:+.4f} USDC{note}"
        )

    async def _unwind_all(
        self,
        bundle: _WatchedBundle,
        naked:  list[_BundleLegState],
    ) -> tuple[float, int]:
        """
        Market-sell every filled leg.  Returns (realised_pnl, failure_count).

        Legs are unwound concurrently: each extra second of a partly-filled
        NegRisk bundle is unhedged directional exposure across several outcomes
        at once, so serialising the sells would compound the drift.
        """
        async def _one(leg: _BundleLegState) -> tuple[float, int]:
            size = math.floor(leg.matched * 100) / 100.0
            if size < 0.01:
                return 0.0, 0
            cost_basis = size * leg.bid
            try:
                resp   = await self._client.unwind_leg(leg.token_id, size)
                status = str(resp.get("status", "")).strip().lower()
                if status not in _FILLED_STATUSES:
                    raise RuntimeError(f"unwind not filled: {resp}")
                proceeds = float(resp.get("taking_amount") or 0.0)
                sold     = float(resp.get("making_amount") or size)
                realised = proceeds - sold * leg.bid
                logger.warning(
                    "NegRiskGuard | UNWOUND leg[%d] %.2f shares on %s — "
                    "proceeds=%.4f cost=%.4f realised=%+.4f",
                    leg.idx, sold, bundle.condition_id[:16],
                    proceeds, sold * leg.bid, realised,
                )
                return realised, 0
            except Exception as exc:  # noqa: BLE001
                ARB_UNWIND_FAILURES.inc()
                logger.error(
                    "NegRiskGuard | unwind leg[%d] failed on %s: %s",
                    leg.idx, bundle.condition_id[:16], exc,
                )
                await self._notifier.send_critical_error(
                    f"NEGRISK UNWIND FAILED {bundle.condition_id[:16]} "
                    f"leg[{leg.idx}] — {size:.2f} naked shares "
                    f"(cost≈{cost_basis:.2f} pUSD) — MANUAL INTERVENTION REQUIRED"
                )
                return 0.0, 1

        results = await asyncio.gather(*(_one(leg) for leg in naked))
        realised = sum(r for r, _ in results)
        failures = sum(f for _, f in results)
        return realised, failures
