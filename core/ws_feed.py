"""
core/ws_feed.py
───────────────
Polymarket CLOB WebSocket ingestor — Arbitrage Edition.

Subscribes to BOTH legs (YES token and NO token) of a single binary market
and maintains a real-time best-ask (lowest available ask) price for each leg.

When an incoming message updates EITHER leg's best-ask, a paired snapshot is
immediately pushed to the asyncio.Queue — but ONLY after BOTH prices are
available.  This gives the strategy loop an atomic (yes_ask, no_ask) pair
to evaluate the arbitrage condition with no stale-leg risk.

Orderbook parsing
─────────────────
Polymarket's market WebSocket sends two primary event shapes:

  "book" event (full snapshot on connect, or after reconnect):
    { "event_type": "book",
      "asset_id": "<token_id>",
      "asks": [{"price": "0.52", "size": "300"}, ...],   ← sorted ascending
      "bids": [...] }

  "price_change" event (delta on every fill or quote change):
    { "event_type": "price_change",
      "asset_id": "<token_id>",
      "price": "0.51",
      "side": "SELL",   ← "SELL" == ask side
      "size": "50" }

The feed updates the best-ask from:
  1. Full "book" events → asks[0].price  (lowest ask, most authoritative)
  2. "price_change" with side="SELL" →
       if price < current_best_ask  : update (new cheaper ask appeared)
       if size == "0"               : invalidate (level removed, trigger REST
                                      refresh on next "book")

Messages may arrive as a JSON array of event dicts or as a single dict;
both forms are handled.

Reconnection
────────────
Identical to the original feed: truncated exponential back-off
1 s → 2 s → 4 s → … → 30 s (cap), reset to 1 s after STABLE_AFTER seconds.

Paired tick pushed to queue
───────────────────────────
{
    "type":          "arb_tick",
    "condition_id":  str,
    "yes_token_id":  str,
    "no_token_id":   str,
    "yes_ask":       float,   # best ask for YES leg
    "no_ask":        float,   # best ask for NO leg
    "ts":            float,   # event_loop.time() at push time
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_INITIAL_BACKOFF = 1.0    # seconds
_MAX_BACKOFF     = 30.0   # seconds
_STABLE_AFTER    = 10.0   # reset back-off after this many seconds connected


class MarketFeed:
    """
    Async WebSocket feed tracking the best-ask for both legs of one market.

    Usage::

        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2048)
        feed = MarketFeed(
            yes_token_id="0xabc...",
            no_token_id="0xdef...",
            condition_id="0x123...",
            queue=queue,
        )
        asyncio.create_task(feed.run())
    """

    def __init__(
        self,
        yes_token_id: str,
        no_token_id: str,
        condition_id: str,
        queue: asyncio.Queue,
        *,
        ping_interval: float = 20.0,
    ) -> None:
        self._yes_token_id = yes_token_id
        self._no_token_id  = no_token_id
        self._condition_id = condition_id
        self._token_ids    = [yes_token_id, no_token_id]
        self._queue        = queue
        self._ping_interval = ping_interval
        self._running      = False

        # Liveness tracking — used by FeedRegistry.prune_stale to drop feeds on
        # dead/illiquid markets that never produce a two-sided quote.
        self._created_monotonic   = time.monotonic()
        self._last_tick_monotonic = self._created_monotonic

        # Live best-ask state: None until we receive a confirmed ask price
        self._best_ask: dict[str, Optional[float]] = {
            yes_token_id: None,
            no_token_id:  None,
        }

        # Dedup: the WS emits an event on every book/price_change even when the
        # best ask is unchanged.  Remember the last pushed pair so we never
        # enqueue an identical tick (saves downstream re-evaluation + alerts).
        self._last_pushed: tuple[Optional[float], Optional[float]] = (None, None)

    def idle_seconds(self, now: Optional[float] = None) -> float:
        """Seconds since this feed last pushed a two-sided arb_tick.

        For a feed that has never produced a tick this measures time since
        construction, so persistently one-sided / dead markets become prunable.
        """
        ref = now if now is not None else time.monotonic()
        return ref - self._last_tick_monotonic

    # ──────────────────────────────────────────────────────────────────────────
    # Public control
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Long-running coroutine.  Call as an asyncio Task; cancel to stop.
        Reconnects automatically on any error with exponential back-off.
        """
        self._running = True
        backoff = _INITIAL_BACKOFF

        while self._running:
            connected_at: Optional[float] = None
            try:
                # trust_env=False: explicitly bypass any HTTP_PROXY / HTTPS_PROXY
                # env vars set by the REST ProxyRotator.  The WebSocket must
                # connect directly to Polymarket for sub-millisecond ingestion.
                async with aiohttp.ClientSession(trust_env=False) as session:
                    async with session.ws_connect(
                        _WS_URL,
                        heartbeat=self._ping_interval,
                        receive_timeout=60.0,
                    ) as ws:
                        # Fresh connection → invalidate stale prices so we
                        # wait for the initial "book" snapshot before firing.
                        self._reset_state()
                        await self._subscribe(ws)
                        connected_at = asyncio.get_event_loop().time()
                        logger.info(
                            "WS connected | yes=%s no=%s",
                            self._yes_token_id[:12], self._no_token_id[:12],
                        )

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._dispatch(msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.warning("WS error frame: %s", msg.data)
                                break
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.CLOSED,
                            ):
                                logger.info("WS closed by server")
                                break

            except asyncio.CancelledError:
                self._running = False
                logger.info("MarketFeed cancelled — shutting down")
                return

            except Exception as exc:  # noqa: BLE001
                logger.error("WS error: %s", exc)

            # Reset back-off if connection was stable long enough
            if connected_at is not None:
                uptime = asyncio.get_event_loop().time() - connected_at
                if uptime >= _STABLE_AFTER:
                    backoff = _INITIAL_BACKOFF

            logger.info("WS reconnecting in %.1f s …", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)

    def stop(self) -> None:
        """Signal the feed to stop after the current connection closes."""
        self._running = False

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Invalidate both leg prices so we wait for a fresh book snapshot."""
        self._best_ask = {
            self._yes_token_id: None,
            self._no_token_id:  None,
        }

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        payload = json.dumps(
            {"assets_ids": self._token_ids, "type": "market"}
        )
        await ws.send_str(payload)
        logger.debug("Subscription sent: %s", payload)

    async def _dispatch(self, raw: str) -> None:
        """Parse one raw WS text frame and update best-ask state."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON WS message dropped: %s", raw[:120])
            return

        # Polymarket may send a single dict or an array of event dicts.
        events: list[dict] = data if isinstance(data, list) else [data]

        updated = False
        for event in events:
            if isinstance(event, dict):
                updated |= self._update_leg(event)

        if updated:
            self._maybe_push_tick()

    def _update_leg(self, event: dict) -> bool:
        """
        Extract best-ask from a single event dict and update internal state.

        Returns True if any leg's best-ask changed.
        """
        asset_id = (
            event.get("asset_id")
            or event.get("token_id")
            or ""
        )
        if asset_id not in self._best_ask:
            return False  # event for a token we don't track

        event_type = event.get("event_type", "").lower()

        if event_type == "book":
            return self._handle_book(asset_id, event)
        elif event_type == "price_change":
            return self._handle_price_change(asset_id, event)
        else:
            # Unknown event type — attempt generic best-ask extraction
            return self._handle_generic(asset_id, event)

    def _handle_book(self, asset_id: str, event: dict) -> bool:
        """
        Full orderbook snapshot → take asks[0].price as the definitive best ask.
        Asks are sorted ascending (lowest first) per Polymarket CLOB convention.
        """
        asks = event.get("asks", [])
        if not asks:
            return False
        try:
            entry = asks[0]
            price = float(entry["price"] if isinstance(entry, dict) else entry)
            if price <= 0.0:
                return False
            changed = self._best_ask[asset_id] != price
            self._best_ask[asset_id] = price
            if changed:
                logger.debug(
                    "BOOK  | %s ask=%.6f", asset_id[:12], price
                )
            return changed
        except (KeyError, IndexError, TypeError, ValueError):
            return False

    def _handle_price_change(self, asset_id: str, event: dict) -> bool:
        """
        Price-level delta event.

        side="SELL" means an ask-side level changed.
        If the new price is lower than the current best ask, it is the new best.
        If size="0", the level was removed; we invalidate our cached value to
        await the next authoritative "book" snapshot.
        """
        side = event.get("side", "").upper()
        if side not in ("SELL", "ASK"):
            return False  # bid-side change; irrelevant for arb ask tracking

        try:
            price = float(event["price"])
        except (KeyError, TypeError, ValueError):
            return False

        # Level removed — invalidate to force a conservative re-evaluation.
        size_raw = event.get("size", "1")
        try:
            size = float(size_raw)
        except (TypeError, ValueError):
            size = 1.0

        if size <= 0.0:
            if self._best_ask[asset_id] is not None:
                logger.debug(
                    "PRICE_CHANGE | %s ask level %.6f removed — invalidating",
                    asset_id[:12], self._best_ask[asset_id],
                )
                self._best_ask[asset_id] = None
                return True
            return False

        # New ask level; update if it beats the current best.
        current = self._best_ask[asset_id]
        if current is None or price < current:
            self._best_ask[asset_id] = price
            logger.debug(
                "PRICE_CHANGE | %s ask=%.6f (prev=%s)",
                asset_id[:12], price,
                f"{current:.6f}" if current is not None else "none",
            )
            return True

        return False

    def _handle_generic(self, asset_id: str, event: dict) -> bool:
        """
        Fallback parser for events without a recognised event_type.

        Tries asks[] first, then falls back to a top-level "price" field on
        the ask side — mirrors the legacy midpoint-extraction heuristic.
        """
        asks = event.get("asks", [])
        if asks:
            try:
                entry = asks[0]
                price = float(entry["price"] if isinstance(entry, dict) else entry)
                if price > 0.0:
                    changed = self._best_ask[asset_id] != price
                    self._best_ask[asset_id] = price
                    return changed
            except (KeyError, IndexError, TypeError, ValueError):
                pass

        side = event.get("side", "").upper()
        if side in ("SELL", "ASK"):
            try:
                price = float(event["price"])
                current = self._best_ask[asset_id]
                if price > 0.0 and (current is None or price < current):
                    self._best_ask[asset_id] = price
                    return True
            except (KeyError, TypeError, ValueError):
                pass

        return False

    def _maybe_push_tick(self) -> None:
        """
        If BOTH legs have a valid best-ask price, push a paired tick dict.

        Drops the oldest entry when the queue is full (strategy-lag protection).
        """
        yes_ask = self._best_ask.get(self._yes_token_id)
        no_ask  = self._best_ask.get(self._no_token_id)

        if yes_ask is None or no_ask is None:
            return  # still waiting for one or both legs

        # Dedup — skip if neither leg's best ask changed since the last push.
        if (yes_ask, no_ask) == self._last_pushed:
            return
        self._last_pushed = (yes_ask, no_ask)

        now = time.monotonic()
        self._last_tick_monotonic = now   # liveness marker for stale-feed pruning
        tick: dict = {
            "type":         "arb_tick",
            "condition_id": self._condition_id,
            "yes_token_id": self._yes_token_id,
            "no_token_id":  self._no_token_id,
            "yes_ask":      yes_ask,
            "no_ask":       no_ask,
            "ts":           now,
        }

        try:
            self._queue.put_nowait(tick)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(tick)
            logger.debug("Queue full — oldest arb_tick evicted")
