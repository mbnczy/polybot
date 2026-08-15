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
import contextlib
import json
import logging
import time
from typing import Optional

import aiohttp

from telemetry.metrics import WS_PARSE_SECONDS

logger = logging.getLogger(__name__)

# ── Fast JSON parse ───────────────────────────────────────────────────────────
# orjson is ~2–5× faster than stdlib json on the per-message WS hot path. It is
# optional: when absent we fall back to stdlib json with zero behavioural change
# (add `orjson` to requirements.txt to activate the speedup in production).
try:
    import orjson

    def _loads(raw: "str | bytes"):
        return orjson.loads(raw)
except ImportError:  # pragma: no cover - fallback path
    def _loads(raw: "str | bytes"):
        return json.loads(raw)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_INITIAL_BACKOFF = 1.0    # seconds
_MAX_BACKOFF     = 30.0   # seconds
_STABLE_AFTER    = 10.0   # reset back-off after this many seconds connected


class _MarketState:
    """
    Per-market best-ask state and event handling for ONE binary market.

    Shared by MarketFeed (single-market connection) and MarketShard (many
    markets on one multiplexed connection).  Encapsulates the book /
    price_change parsing, tick-size capture, dedup and liveness tracking that
    used to live directly on MarketFeed, so both feed types behave identically.
    """

    def __init__(self, condition_id: str, yes_token_id: str, no_token_id: str) -> None:
        self.condition_id = condition_id
        self.yes_token_id = yes_token_id
        self.no_token_id  = no_token_id
        self.token_ids    = [yes_token_id, no_token_id]

        now = time.monotonic()
        self.created_monotonic   = now
        self.last_tick_monotonic = now

        self._best_ask: dict[str, Optional[float]] = {
            yes_token_id: None, no_token_id: None,
        }
        self._tick_size: dict[str, Optional[float]] = {
            yes_token_id: None, no_token_id: None,
        }
        self._last_pushed: tuple[Optional[float], Optional[float]] = (None, None)

    def owns(self, asset_id: str) -> bool:
        return asset_id in self._best_ask

    def reset(self) -> None:
        """Invalidate best-asks (called on reconnect → await fresh snapshot)."""
        self._best_ask = {self.yes_token_id: None, self.no_token_id: None}
        self._last_pushed = (None, None)

    def idle_seconds(self, now: Optional[float] = None) -> float:
        ref = now if now is not None else time.monotonic()
        return ref - self.last_tick_monotonic

    # ── Event handling ─────────────────────────────────────────────────────────

    def update_leg(self, asset_id: str, event: dict) -> bool:
        """Update one leg's best-ask from an event.  Returns True if it changed."""
        if asset_id not in self._best_ask:
            return False
        event_type = event.get("event_type", "").lower()
        if event_type == "book":
            return self._handle_book(asset_id, event)
        elif event_type == "price_change":
            return self._handle_price_change(asset_id, event)
        return self._handle_generic(asset_id, event)

    def _handle_book(self, asset_id: str, event: dict) -> bool:
        # Capture tick size from the snapshot (Polymarket book events carry it).
        _ts = event.get("tick_size", event.get("tickSize"))
        if _ts is not None:
            try:
                self._tick_size[asset_id] = float(_ts)
            except (TypeError, ValueError):
                pass

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
            return changed
        except (KeyError, IndexError, TypeError, ValueError):
            return False

    def _handle_price_change(self, asset_id: str, event: dict) -> bool:
        side = event.get("side", "").upper()
        if side not in ("SELL", "ASK"):
            return False
        try:
            price = float(event["price"])
        except (KeyError, TypeError, ValueError):
            return False
        size_raw = event.get("size", "1")
        try:
            size = float(size_raw)
        except (TypeError, ValueError):
            size = 1.0
        if size <= 0.0:
            if self._best_ask[asset_id] is not None:
                self._best_ask[asset_id] = None
                return True
            return False
        current = self._best_ask[asset_id]
        if current is None or price < current:
            self._best_ask[asset_id] = price
            return True
        return False

    def _handle_generic(self, asset_id: str, event: dict) -> bool:
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

    def build_tick(self) -> Optional[dict]:
        """
        Return a paired arb_tick dict when BOTH legs have a fresh best-ask and
        the pair changed since the last push, else None.  Updates dedup and
        liveness state as a side effect (mirrors the old _maybe_push_tick).
        """
        yes_ask = self._best_ask.get(self.yes_token_id)
        no_ask  = self._best_ask.get(self.no_token_id)
        if yes_ask is None or no_ask is None:
            return None
        if (yes_ask, no_ask) == self._last_pushed:
            return None
        self._last_pushed = (yes_ask, no_ask)

        now = time.monotonic()
        self.last_tick_monotonic = now
        _yt = self._tick_size.get(self.yes_token_id)
        _nt = self._tick_size.get(self.no_token_id)
        _ticks = [t for t in (_yt, _nt) if t is not None]
        return {
            "type":         "arb_tick",
            "condition_id": self.condition_id,
            "yes_token_id": self.yes_token_id,
            "no_token_id":  self.no_token_id,
            "yes_ask":      yes_ask,
            "no_ask":       no_ask,
            "tick_size":    max(_ticks) if _ticks else None,
            "ts":           now,
        }


def _queue_push(queue: asyncio.Queue, tick: dict) -> None:
    """Enqueue a tick, evicting the oldest entry when the queue is full."""
    try:
        queue.put_nowait(tick)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        queue.put_nowait(tick)
        logger.debug("Queue full — oldest arb_tick evicted")


def _parse_events(raw: str) -> list[dict]:
    """Parse a raw WS text frame into a list of event dicts (or [] on error)."""
    _parse_t0 = time.monotonic()
    try:
        data = _loads(raw)
    except json.JSONDecodeError:
        logger.warning("Non-JSON WS message dropped: %s", raw[:120])
        return []
    WS_PARSE_SECONDS.observe(time.monotonic() - _parse_t0)
    events = data if isinstance(data, list) else [data]
    return [e for e in events if isinstance(e, dict)]


class MarketFeed:
    """
    Async WebSocket feed tracking the best-ask for both legs of ONE market.

    Retained for the single-market path and direct testing.  Delegates all
    event handling to a `_MarketState`; the multiplexed path uses MarketShard.

    Usage::

        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2048)
        feed = MarketFeed("0xabc...", "0xdef...", "0x123...", queue)
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
        self._queue        = queue
        self._ping_interval = ping_interval
        self._running      = False
        self._state = _MarketState(condition_id, yes_token_id, no_token_id)

    def idle_seconds(self, now: Optional[float] = None) -> float:
        """Seconds since this feed last pushed a two-sided arb_tick."""
        return self._state.idle_seconds(now)

    # ── Compat accessors (delegate to the underlying _MarketState) ─────────────
    @property
    def _best_ask(self) -> dict:
        return self._state._best_ask

    @property
    def _last_tick_monotonic(self) -> float:
        return self._state.last_tick_monotonic

    def _maybe_push_tick(self) -> None:
        """Build and enqueue a paired tick if both legs are known (compat)."""
        tick = self._state.build_tick()
        if tick is not None:
            _queue_push(self._queue, tick)

    async def run(self) -> None:
        """Long-running coroutine.  Cancel to stop.  Auto-reconnects."""
        self._running = True
        backoff = _INITIAL_BACKOFF
        while self._running:
            connected_at: Optional[float] = None
            try:
                async with aiohttp.ClientSession(trust_env=False) as session:
                    async with session.ws_connect(
                        _WS_URL,
                        heartbeat=self._ping_interval,
                        receive_timeout=60.0,
                    ) as ws:
                        self._state.reset()
                        await ws.send_str(json.dumps(
                            {"assets_ids": self._state.token_ids, "type": "market"}
                        ))
                        connected_at = asyncio.get_event_loop().time()
                        logger.info(
                            "WS connected | yes=%s no=%s",
                            self._yes_token_id[:12], self._no_token_id[:12],
                        )
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._dispatch(msg.data)
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

            if connected_at is not None:
                uptime = asyncio.get_event_loop().time() - connected_at
                if uptime >= _STABLE_AFTER:
                    backoff = _INITIAL_BACKOFF
            logger.info("WS reconnecting in %.1f s …", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)

    def stop(self) -> None:
        self._running = False

    def _dispatch(self, raw: str) -> None:
        st = self._state
        changed = False
        for event in _parse_events(raw):
            asset_id = event.get("asset_id") or event.get("token_id") or ""
            changed |= st.update_leg(asset_id, event)
        if changed:
            tick = st.build_tick()
            if tick is not None:
                _queue_push(self._queue, tick)


class MarketShard:
    """
    ONE multiplexed WebSocket connection carrying MANY markets.

    Instead of one connection per market (which explodes the per-IP connection
    count at scale), a shard subscribes to every member market's asset IDs on a
    single connection and routes each incoming event by asset_id to the owning
    `_MarketState`.  This keeps the connection count at ceil(N / capacity)
    instead of N.

    Membership is dynamic:
      • add()    — subscribes the new market incrementally (one extra frame on
                   the live socket; no reconnect, so peers are undisturbed).
      • remove() — stops routing the market; its lingering subscription is
                   dropped on the next reconnect, which re-subscribes to exactly
                   the current member set.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        *,
        shard_id: int = 0,
        ping_interval: float = 20.0,
    ) -> None:
        self._queue         = queue
        self._shard_id      = shard_id
        self._ping_interval = ping_interval
        self._running       = False
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

        self._states:  dict[str, _MarketState] = {}   # condition_id → state
        self._routing: dict[str, _MarketState] = {}   # asset_id     → state

    # ── Membership ─────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._states)

    def condition_ids(self) -> list[str]:
        return list(self._states.keys())

    def contains(self, condition_id: str) -> bool:
        return condition_id in self._states

    def add(self, condition_id: str, yes_token_id: str, no_token_id: str) -> None:
        """Add a market to this shard and (if connected) subscribe incrementally."""
        if condition_id in self._states:
            return
        st = _MarketState(condition_id, yes_token_id, no_token_id)
        self._states[condition_id] = st
        self._routing[yes_token_id] = st
        self._routing[no_token_id]  = st
        if self._ws is not None and not self._ws.closed:
            # Incremental subscribe on the live socket — no reconnect churn.
            asyncio.ensure_future(self._subscribe([yes_token_id, no_token_id]))

    def remove(self, condition_id: str) -> None:
        """Stop tracking a market (its subscription is cleaned on next reconnect)."""
        st = self._states.pop(condition_id, None)
        if st is None:
            return
        self._routing.pop(st.yes_token_id, None)
        self._routing.pop(st.no_token_id, None)

    def idle_seconds(self, condition_id: str, now: Optional[float] = None) -> Optional[float]:
        st = self._states.get(condition_id)
        return None if st is None else st.idle_seconds(now)

    # ── Connection lifecycle ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Long-running coroutine.  Cancel to stop.  Auto-reconnects."""
        self._running = True
        backoff = _INITIAL_BACKOFF
        while self._running:
            connected_at: Optional[float] = None
            try:
                async with aiohttp.ClientSession(trust_env=False) as session:
                    async with session.ws_connect(
                        _WS_URL,
                        heartbeat=self._ping_interval,
                        receive_timeout=60.0,
                    ) as ws:
                        self._ws = ws
                        for st in self._states.values():
                            st.reset()
                        all_ids = [tid for st in self._states.values()
                                   for tid in st.token_ids]
                        if all_ids:
                            await self._subscribe(all_ids)
                        connected_at = asyncio.get_event_loop().time()
                        logger.info(
                            "WS shard[%d] connected | markets=%d assets=%d",
                            self._shard_id, len(self._states), len(all_ids),
                        )
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._dispatch(msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.warning("WS shard[%d] error frame: %s",
                                               self._shard_id, msg.data)
                                break
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.CLOSED,
                            ):
                                logger.info("WS shard[%d] closed by server",
                                            self._shard_id)
                                break
            except asyncio.CancelledError:
                self._running = False
                self._ws = None
                logger.info("WS shard[%d] cancelled — shutting down", self._shard_id)
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("WS shard[%d] error: %s", self._shard_id, exc)
            finally:
                self._ws = None

            if connected_at is not None:
                uptime = asyncio.get_event_loop().time() - connected_at
                if uptime >= _STABLE_AFTER:
                    backoff = _INITIAL_BACKOFF
            logger.info("WS shard[%d] reconnecting in %.1f s …",
                        self._shard_id, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)

    def stop(self) -> None:
        self._running = False

    async def _subscribe(self, asset_ids: list[str]) -> None:
        ws = self._ws
        if ws is None or ws.closed or not asset_ids:
            return
        with contextlib.suppress(Exception):
            await ws.send_str(json.dumps(
                {"assets_ids": asset_ids, "type": "market"}
            ))

    def _dispatch(self, raw: str) -> None:
        changed: set[_MarketState] = set()
        for event in _parse_events(raw):
            asset_id = event.get("asset_id") or event.get("token_id") or ""
            st = self._routing.get(asset_id)
            if st is None:
                continue   # event for a market we no longer track
            if st.update_leg(asset_id, event):
                changed.add(st)
        for st in changed:
            tick = st.build_tick()
            if tick is not None:
                _queue_push(self._queue, tick)
