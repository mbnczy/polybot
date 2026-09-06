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

from core import market_titles

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
NEGRISK_GUARD_POLL_S:     float = float(os.environ.get("NEGRISK_GUARD_POLL_S", 1.0))
NEGRISK_BUNDLE_TIMEOUT_S: float = float(os.environ.get("NEGRISK_BUNDLE_TIMEOUT_S", 10.0))
NEGRISK_ORDER_TTL_S:      float = float(os.environ.get("NEGRISK_ORDER_TTL_S", 45.0))
# After a bundle has to be flattened, how long to leave that NegRisk group alone.
# Without this the loop re-entered the SAME group within 200 ms of unwinding it:
# 9 bundles on one group in 4 minutes on 2026-09-05, each paying the spread
# twice. A group that just failed to fill cleanly is evidence about the book,
# not an invitation to retry immediately.
NEGRISK_GROUP_COOLDOWN_S: float = float(
    os.environ.get("NEGRISK_GROUP_COOLDOWN_S", 60.0)
)

# Each consecutive failure on the same group doubles its cooldown, up to this
# ceiling. A group that repeatedly refuses to complete is telling us something
# about its book; a fixed 60 s just meant we relearned it every minute.
NEGRISK_COOLDOWN_MAX_S: float = float(
    os.environ.get("NEGRISK_COOLDOWN_MAX_S", 900.0)
)

# Complete a partially-filled bundle as taker when the missing legs are cheap
# enough that the finished bundle still clears its payout floor. Set 0 to
# disable and always flatten (the behaviour before 2026-09-06).
NEGRISK_COMPLETE_PARTIAL: bool = os.environ.get(
    "NEGRISK_COMPLETE_PARTIAL", "true"
).strip().lower() in ("1", "true", "yes", "on")

# Minimum USDC profit for completion to be worth crossing the spread for.
NEGRISK_MIN_COMPLETE_PROFIT: float = float(
    os.environ.get("NEGRISK_MIN_COMPLETE_PROFIT", 0.01)
)

# Assumed taker fee when completing. Overridden at startup by the rate measured
# from settled trades — see PolyClient.observed_fee_rates().
NEGRISK_TAKER_FEE: float = float(os.environ.get("DEFAULT_TAKER_FEE", 0.02))

# Grace period before an order the CLOB does not know about is judged.
#
# The CLOB returns a null body for an order it "no longer tracks" — but it
# returns the same null body for an order it does NOT YET track. A limit order
# is not queryable for a moment after it is accepted. Judging it inside that
# window declares a live resting order dead: on 2026-09-05 the guard released
# eight bundles ~200 ms after placing them, every leg logged "cancelled, not
# filled", and 6 of those 24 abandoned legs went on to fill unmanaged — 30.42
# shares and 29 pUSD of naked directional position that no guard was watching.
NEGRISK_ORDER_INDEX_GRACE_S: float = float(
    os.environ.get("NEGRISK_ORDER_INDEX_GRACE_S", 5.0)
)

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
    # monotonic clock at submission — an order cannot be judged before the
    # exchange has had time to index it.
    placed_at:        float = field(default_factory=time.monotonic)

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
        group_cooldown: float = NEGRISK_GROUP_COOLDOWN_S,
        cooldown_max:   float = NEGRISK_COOLDOWN_MAX_S,
        index_grace:    float = NEGRISK_ORDER_INDEX_GRACE_S,
        complete_partial: bool  = NEGRISK_COMPLETE_PARTIAL,
        taker_fee:        float = NEGRISK_TAKER_FEE,
        min_complete_profit: float = NEGRISK_MIN_COMPLETE_PROFIT,
    ) -> None:
        self._client   = client
        self._breaker  = breaker
        self._notifier = notifier
        self._poll     = max(0.1, poll_interval)
        self._timeout  = max(0.0, bundle_timeout)
        self._ttl      = max(0.0, order_ttl)
        self._cooldown = max(0.0, group_cooldown)
        self._cool_max = max(self._cooldown, cooldown_max)
        self._grace    = max(0.0, index_grace)
        self._complete = bool(complete_partial)
        self._fee      = max(0.0, taker_fee)
        self._min_cp   = max(0.0, min_complete_profit)
        # condition_id -> consecutive failures, for the escalating backoff
        self._strikes: dict[str, int] = {}
        self._bundles: dict[str, _WatchedBundle] = {}
        # condition_id -> monotonic deadline before which the group is off limits
        self._cooling: dict[str, float] = {}

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
            "NegRiskGuard watching | %s legs=%d bundles=%.2f "
            "accepted=%d filled_at_ack=%d",
            market_titles.label(signal.condition_id), len(legs), signal.n_bundles,
            sum(1 for l in legs if l.order_id),
            sum(1 for l in legs if l.fully_matched),
        )

    @property
    def watched_count(self) -> int:
        return len(self._bundles)

    def is_watching(self, condition_id: str) -> bool:
        """True while any bundle on this group is still unresolved."""
        return any(b.condition_id == condition_id for b in self._bundles.values())

    def _arm_cooldown(self, condition_id: str) -> float:
        """Escalate this group's backoff and return the window applied."""
        if self._cooldown <= 0.0:
            return 0.0
        n = self._strikes.get(condition_id, 0) + 1
        self._strikes[condition_id] = n
        window = min(self._cooldown * (2 ** (n - 1)), self._cool_max)
        self._cooling[condition_id] = time.monotonic() + window
        logger.info(
            "NegRiskGuard | group %s cooling %.0fs (strike %d)",
            condition_id[:16], window, n,
        )
        return window

    def set_taker_fee(self, rate: float) -> None:
        """Adopt the fee measured from settled trades. The completion test is
        entirely decided by this number, so an assumed one is not good enough:
        at the hard-coded 2% every completion tonight scored as a loss, and at
        the real 0% every one of them was a profit."""
        self._fee = max(0.0, float(rate))
        logger.info("NegRiskGuard | taker fee set to %.4f", self._fee)

    def clear_strikes(self, condition_id: str) -> None:
        """A group that completed cleanly starts from zero again."""
        self._strikes.pop(condition_id, None)
        self._cooling.pop(condition_id, None)

    def cool_down(self, condition_id: str) -> None:
        """
        Put a group off limits for the cooldown window without a bundle having
        been watched — used when submission itself failed. A leg the exchange
        rejected as a duplicate will be rejected again for as long as it holds
        that order hash, so re-signalling the same group 50 seconds later just
        reproduces the rejection.
        """
        self._arm_cooldown(condition_id)

    def is_busy(self, condition_id: str) -> bool:
        """
        True when a new bundle on this group must NOT be submitted — either one
        is already in flight, or the group is cooling off after a flatten.
        """
        if self.is_watching(condition_id):
            return True
        until = self._cooling.get(condition_id)
        if until is None:
            return False
        if time.monotonic() >= until:
            del self._cooling[condition_id]
            return False
        return True

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
            self._refresh_leg(bundle, leg) for leg in bundle.legs if leg.open
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

    async def _refresh_leg(
        self, bundle: _WatchedBundle, leg: _BundleLegState
    ) -> None:
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
            # A null body means the CLOB is not tracking this order — which
            # covers "filled", "cancelled" AND "accepted a moment ago and not
            # indexed yet". Judging the third case as one of the first two
            # abandons a live resting order, so wait out the grace period and
            # let the next poll decide.
            if (
                not leg.cancel_requested
                and time.monotonic() - leg.placed_at < self._grace
            ):
                logger.debug(
                    "NegRiskGuard | order %s not indexed yet — still watching",
                    leg.order_id[:12],
                )
                return

            # Past the grace window it really is filled or cancelled, and the
            # response cannot distinguish them, so ask the trade feed rather
            # than guess. Guessing "filled" fabricates P&L and
            # triggers unwinds of shares that do not exist; guessing
            # "cancelled" leaves a filled leg naked. Both were observed live
            # on 2026-09-04.
            leg.open = False
            if not leg.cancel_requested:
                filled = await self._client.order_filled_size(
                    leg.order_id, leg.token_id
                )
                if filled is None:
                    # Trade feed unreachable. Assume filled so we flatten rather
                    # than sit on a possibly-naked leg, and make it loud — this
                    # is the one branch that can still be wrong.
                    logger.error(
                        "NegRiskGuard | order %s untracked and the trade feed is "
                        "unavailable — assuming filled (%.2f shares); verify "
                        "the wallet manually",
                        leg.order_id[:12], leg.size,
                    )
                    leg.matched = leg.size
                else:
                    verified = min(leg.size, filled)
                    if verified > leg.matched + _SHARE_EPS:
                        logger.warning(
                            "NegRiskGuard | order %s untracked — trade feed attributes "
                            "%.2f share(s) to it, treating as filled",
                            leg.order_id[:12], verified,
                        )
                    elif verified < _SHARE_EPS:
                        # NOT proof the order is gone. "Untracked" also covers
                        # an order the CLOB has simply not surfaced yet, and a
                        # resting order that has not traded looks exactly like
                        # this. Marking it closed here made _finalize skip it in
                        # the cancel loop, so it stayed live on the book and
                        # filled later with nobody watching: 35 bundles logged
                        # "expired unfilled" between 23:49 and 00:31 while the
                        # wallet quietly bought 27 pUSD of NH-01 legs.
                        #
                        # Cancel it for real. Cancelling an order that genuinely
                        # is gone is harmless; assuming it is gone is not.
                        logger.info(
                            "NegRiskGuard | order %s untracked with no attributed "
                            "fills — cancelling to be certain",
                            leg.order_id[:12],
                        )
                        await self._cancel_leg(bundle, leg)
                    leg.matched = max(leg.matched, verified)
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

    async def _best_ask(self, token_id: str) -> "float | None":
        try:
            book = await self._client.get_orderbook(token_id)
            asks = book.get("asks", [])
            if not asks:
                return None
            # The exchange returns the book WORST-first, so scan for the
            # minimum rather than trusting index 0.
            return min(
                float(a["price"] if isinstance(a, dict) else a) for a in asks
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "NegRiskGuard | orderbook fetch failed for %s: %s",
                str(token_id)[:12], exc,
            )
            return None

    async def _try_complete(
        self, bundle: _WatchedBundle
    ) -> "float | None":
        """
        Buy the missing legs as taker when the finished bundle still clears its
        payout floor. Returns realised profit if completed, else None.

        Why this exists
        ---------------
        The module docstring argued that a partial bundle cannot be held, and
        that is right — but it concluded the only resolution is to flatten, and
        that is wrong. Flattening pays the spread on every leg that filled and
        books a certain loss. Completing pays the spread once, on the legs that
        did NOT fill, and buys the guarantee outright.

        Measured on the three flattens of 2026-09-05/06, at the real 0% fee:

            bundle        complete    flattened
            2/3 filled     +0.0100      -0.0659
            1/3 filled     +0.0100      -0.0102
            1/3 filled     +0.0100      -0.0101

        Every one of them was a profit available and a loss taken instead.

        The test is the entry test again: holding n bundles of k legs pays
        n*(k-1) at expiry, so completion is worth it when everything already
        spent plus the cost of the missing legs stays under that.
        """
        if not self._complete:
            return None

        legs = bundle.legs
        k    = len(legs)
        if k < 2:
            return None

        n = max(leg.matched for leg in legs)
        if n <= _SHARE_EPS:
            return None
        n = math.floor(n * 100) / 100.0

        spent  = sum(leg.matched * leg.bid for leg in legs)
        wanted: list[tuple[_BundleLegState, float, float]] = []
        cost   = 0.0
        for leg in legs:
            qty = math.floor(max(0.0, n - leg.matched) * 100) / 100.0
            if qty <= _SHARE_EPS:
                continue
            ask = await self._best_ask(leg.token_id)
            if ask is None or not (0.0 < ask < 1.0):
                logger.debug(
                    "NegRiskGuard | no usable ask on leg[%d] — cannot complete",
                    leg.idx,
                )
                return None
            wanted.append((leg, qty, ask))
            cost += qty * ask * (1.0 + self._fee)

        if not wanted:
            return None

        payout = n * (k - 1)
        profit = payout - (spent + cost)
        if profit < self._min_cp:
            logger.info(
                "NegRiskGuard | completion on %s would net %+.4f (< %.4f) — "
                "flattening instead",
                bundle.condition_id[:16], profit, self._min_cp,
            )
            return None

        # Commit. A leg that fails to fill leaves us no worse off than the
        # flatten path we would otherwise have taken, and the shares bought so
        # far are still counted, so the caller can fall through to unwinding.
        bought = 0
        for leg, qty, ask in wanted:
            try:
                resp = await self._client.post_order(
                    token_id=leg.token_id, side="BUY", price=ask, size=qty,
                )
                status = str(resp.get("status", "")).strip().lower()
                if status not in _FILLED_STATUSES:
                    raise RuntimeError(f"completion leg not filled: {resp}")
                leg.matched += qty
                bought += 1
                logger.info(
                    "NegRiskGuard | completion bought %.2f of leg[%d] @ %.4f",
                    qty, leg.idx, ask,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NegRiskGuard | completion leg[%d] failed on %s: %s — "
                    "falling back to flatten",
                    leg.idx, bundle.condition_id[:16], exc,
                )
                return None

        logger.warning(
            "NegRiskGuard | BUNDLE COMPLETED AS TAKER %s | %d leg(s) bought, "
            "%.2f bundles, profit=%+.4f USDC",
            bundle.condition_id[:16], bought, n, profit,
        )
        await self._notifier.notify(
            f"\u2705 NegRisk bundle on {market_titles.label(bundle.condition_id)} "
            f"completed as taker \u2014 {n:.2f} bundles, "
            f"profit={profit:+.4f} USDC"
        )
        return round(profit, 6)

    async def _recheck_fills(
        self, bundle: _WatchedBundle
    ) -> list[_BundleLegState]:
        """
        Final authoritative check for fills the status polls may have missed.

        Runs only on the path where the guard believes nothing filled, so it
        costs one trade-feed lookup per leg per dissolved bundle and nothing at
        all on the common paths. Updates `matched` in place and returns the legs
        that carry exposure.
        """
        exposed: list[_BundleLegState] = []
        for leg in bundle.legs:
            if not leg.order_id:
                continue
            filled = await self._client.order_filled_size(
                leg.order_id, leg.token_id
            )
            if filled is None or filled <= _SHARE_EPS:
                continue
            leg.matched = max(leg.matched, min(leg.size, filled))
            if leg.matched > _SHARE_EPS:
                logger.warning(
                    "NegRiskGuard | leg %s filled %.2f share(s) that polling "
                    "never saw — flattening instead of releasing",
                    leg.order_id[:12], leg.matched,
                )
                exposed.append(leg)
        return exposed

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
            self.clear_strikes(bundle.condition_id)
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
            # `matched` is only ever as good as the status polls that produced
            # it, and those can fail for a leg's whole life — Cloudflare began
            # blocking /data/order with a 400 on 2026-09-06. A blind guard sees
            # matched == 0 and reads it as "nothing filled", which releases the
            # slot and walks away from a leg that DID fill: the same naked
            # position this guard exists to prevent, reached by a different
            # route. Ask the trade feed once before believing the quiet.
            naked = await self._recheck_fills(bundle)

        if not naked:
            # Bundle dissolved with no exposure — pure release, nothing booked.
            logger.info(
                "NegRiskGuard | bundle on %s expired unfilled — releasing slot",
                bundle.condition_id[:16],
            )
            self._breaker.release_open()
            return

        # Incomplete with exposure. A partial bundle cannot be HELD (see the
        # module docstring), but flattening is not the only way to resolve it:
        # buying the missing legs converts it into the guaranteed bundle it was
        # meant to be, and that is usually cheaper than paying the spread to
        # undo every leg that filled. Try that first.
        ARB_HALF_FILLS.inc()
        completed = await self._try_complete(bundle)
        if completed is not None:
            self._breaker.on_fill(pnl=completed)
            self.clear_strikes(bundle.condition_id)
            return

        self._arm_cooldown(bundle.condition_id)
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
