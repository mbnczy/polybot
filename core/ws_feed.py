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
  1. Full "book" events → min(asks).price (most authoritative). NOT asks[0]:
     the exchange returns both sides WORST-first, so asks[0] is the dearest
     offer on the book. See _best_ask_level().
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


def _best_ask_level(asks: "list | None") -> "tuple[float, float] | None":
    """
    Lowest ask price and the size resting at it, or None if there is no usable
    offer.

    The CLOB returns BOTH book sides worst-first: asks descend from 0.999 and
    bids ascend from 0.001, so `asks[0]` is the WORST offer on the book rather
    than the best. Verified live on 2026-09-05 against the REST book and the
    websocket `book` event for the same token in the same second:

        asks[0] = 0.999      true best ask = 0.969
        bids[0] = 0.001      true best bid = 0.960

    Reading asks[0] therefore priced every market ~3 cents too high, which both
    hides real arbitrage and produces synthetic maker bids that rest far from
    the touch and never fill. Scan for the minimum instead, and never assume an
    ordering the exchange has not promised.
    """
    best_p: float | None = None
    best_s: float = 0.0
    for lvl in asks or []:
        try:
            if isinstance(lvl, dict):
                price = float(lvl["price"])
                size  = float(lvl.get("size", 0.0) or 0.0)
            else:
                price, size = float(lvl), 0.0
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0.0:
            continue
        if best_p is None or price < best_p:
            best_p, best_s = price, size
    return None if best_p is None else (best_p, best_s)


def _best_bid_price(bids: "list | None") -> "float | None":
    """
    Highest bid on the book, or None.

    Same trap as the ask side: the exchange sends bids ascending from 0.001, so
    bids[0] is the WORST bid. Scan for the maximum.
    """
    best: float | None = None
    for lvl in bids or []:
        try:
            price = float(lvl["price"] if isinstance(lvl, dict) else lvl)
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0.0:
            continue
        if best is None or price > best:
            best = price
    return best


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
        # The bid side was parsed and thrown away. Without it the detector
        # cannot tell whether its synthetic (ask - TICK) quote IMPROVES the book
        # or merely JOINS the existing best bid at the back of the queue. Median
        # spread on Polymarket is one tick, so ask - TICK usually lands exactly
        # on the best bid — which is why maker quotes rest for their full TTL and
        # expire unfilled while trades happen every ~30s.
        self._best_bid: dict[str, Optional[float]] = {
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
        self._best_bid = {self.yes_token_id: None, self.no_token_id: None}
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

        # Best bid from the snapshot. Explicit max rather than bids[0] so it is
        # correct whichever way the exchange orders the array.
        bids = event.get("bids", [])
        if bids:
            try:
                _p = [float(e["price"] if isinstance(e, dict) else e) for e in bids]
                _p = [x for x in _p if x > 0.0]
                if _p:
                    self._best_bid[asset_id] = max(_p)
            except (KeyError, TypeError, ValueError):
                pass

        best = _best_ask_level(event.get("asks", []))
        if best is None:
            return False
        price = best[0]
        changed = self._best_ask[asset_id] != price
        self._best_ask[asset_id] = price
        return changed

    def _handle_price_change(self, asset_id: str, event: dict) -> bool:
        # Prefer the best_ask the exchange states outright: it is correct on
        # every kind of book move, including the best level being consumed
        # (where a level delta alone would only tell us it disappeared).
        # Record the stated best bid when present. Observational only — it
        # never gates the ask logic below.
        _bb = event.get("best_bid")
        if _bb is not None:
            try:
                self._best_bid[asset_id] = float(_bb)
            except (TypeError, ValueError):
                pass

        explicit = _explicit_best_ask(event)
        if explicit is not None:
            changed = self._best_ask[asset_id] != explicit
            self._best_ask[asset_id] = explicit
            return changed

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
        best = _best_ask_level(event.get("asks", []))
        if best is not None:
            price = best[0]
            changed = self._best_ask[asset_id] != price
            self._best_ask[asset_id] = price
            return changed
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
            "yes_best_bid": self._best_bid.get(self.yes_token_id),
            "no_best_bid":  self._best_bid.get(self.no_token_id),
            "tick_size":    max(_ticks) if _ticks else None,
            "ts":           now,
        }


class _NegRiskGroupState:
    """
    Best-ask state for ONE NegRisk group — the N mutually exclusive outcomes
    that share a `negRiskMarketID`.

    Only the NO token of each outcome is tracked: the bundle strategy buys NO on
    every selected outcome, so the YES legs are dead weight on the socket (a
    51-outcome group would otherwise cost 102 subscriptions instead of 51).

    Ask *size* is captured alongside price because NegRiskArbDetector sizes the
    bundle from the shallowest leg (arXiv:2508.03474 §6.2) — without depth the
    bundle can be sized past what the book can fill, leaving legs unhedged.

    Emitted tick::

        { "type":             "neg_risk_tick",
          "condition_id":     "<negRiskMarketID>",
          "outcome_token_ids": [no_token, ...],   # only legs with a live ask
          "no_asks":          [float, ...],
          "no_ask_sizes":     [float, ...],
          "tick_size":        float | None,
          "n_group_outcomes": int,                # full group size
          "ts":               float }             # time.monotonic()
    """

    # A group is only actionable once at least this many legs are quoted; a
    # single quoted leg can never form a bundle.
    MIN_QUOTED_LEGS = 2

    def __init__(self, group_id: str, no_token_ids: list[str]) -> None:
        self.group_id     = group_id
        self.no_token_ids = list(no_token_ids)
        self.token_ids    = list(no_token_ids)   # shard subscribes to these

        now = time.monotonic()
        self.created_monotonic   = now
        self.last_tick_monotonic = now

        self._best_ask:  dict[str, Optional[float]] = {t: None for t in no_token_ids}
        self._ask_size:  dict[str, Optional[float]] = {t: None for t in no_token_ids}
        self._tick_size: dict[str, Optional[float]] = {t: None for t in no_token_ids}
        # The bid side was never tracked here, so the detector could not tell
        # whether its synthetic quote would IMPROVE a leg's book or just join
        # the back of an existing queue. On a penny-tick book those are the
        # difference between filling and never filling.
        self._best_bid:  dict[str, Optional[float]] = {t: None for t in no_token_ids}
        self._last_pushed: tuple = ()

    def owns(self, asset_id: str) -> bool:
        return asset_id in self._best_ask

    def reset(self) -> None:
        """Invalidate all legs (called on reconnect → await fresh snapshots)."""
        for t in self.no_token_ids:
            self._best_ask[t] = None
            self._ask_size[t] = None
        self._last_pushed = ()

    def idle_seconds(self, now: Optional[float] = None) -> float:
        ref = now if now is not None else time.monotonic()
        return ref - self.last_tick_monotonic

    # ── Event handling ─────────────────────────────────────────────────────────

    def update_leg(self, asset_id: str, event: dict) -> bool:
        """Update one outcome's best ask/size.  Returns True if it changed."""
        if asset_id not in self._best_ask:
            return False
        event_type = event.get("event_type", "").lower()
        if event_type == "price_change":
            return self._handle_price_change(asset_id, event)
        # "book" snapshots and any unknown shape carrying an asks ladder.
        return self._handle_book(asset_id, event)

    def _handle_book(self, asset_id: str, event: dict) -> bool:
        _ts = event.get("tick_size", event.get("tickSize"))
        if _ts is not None:
            try:
                self._tick_size[asset_id] = float(_ts)
            except (TypeError, ValueError):
                pass

        bid = _best_bid_price(event.get("bids", []))
        if bid is not None:
            self._best_bid[asset_id] = bid

        best = _best_ask_level(event.get("asks", []))
        if best is None:
            return False
        price, size = best
        changed = (
            self._best_ask[asset_id] != price
            or self._ask_size[asset_id] != size
        )
        self._best_ask[asset_id] = price
        self._ask_size[asset_id] = size
        return changed

    def _handle_price_change(self, asset_id: str, event: dict) -> bool:
        # Authoritative top-of-book from the exchange (see _explicit_best_ask).
        explicit = _explicit_best_ask(event)
        if explicit is not None:
            if self._best_ask[asset_id] == explicit:
                return False
            self._best_ask[asset_id] = explicit
            # The batched frame states the price but not the depth AT that
            # price. None = "unknown", which the detector treats as uncapped —
            # distinct from 0.0, which means the level is genuinely empty. The
            # next `book` snapshot restores a real depth reading.
            self._ask_size[asset_id] = None
            return True

        side = event.get("side", "").upper()
        if side not in ("SELL", "ASK"):
            return False
        try:
            price = float(event["price"])
        except (KeyError, TypeError, ValueError):
            return False
        try:
            size = float(event.get("size", "1"))
        except (TypeError, ValueError):
            size = 1.0

        if size <= 0.0:
            # Level removed. Only invalidates the leg when it was *the* best ask;
            # a deeper level being pulled says nothing about the top of book.
            if self._best_ask[asset_id] == price:
                self._best_ask[asset_id] = None
                self._ask_size[asset_id] = None
                return True
            return False

        current = self._best_ask[asset_id]
        if current is None or price < current:
            self._best_ask[asset_id] = price
            self._ask_size[asset_id] = size
            return True
        if price == current and self._ask_size[asset_id] != size:
            # Same price level, depth changed — matters for bundle sizing.
            self._ask_size[asset_id] = size
            return True
        return False

    def build_tick(self) -> Optional[dict]:
        """
        Return a `neg_risk_tick` covering every currently quoted leg, or None
        when fewer than MIN_QUOTED_LEGS are quoted or nothing changed since the
        last push.  Updates dedup/liveness state as a side effect.
        """
        token_ids: list[str]   = []
        asks:      list[float] = []
        sizes:     list[float | None] = []
        bids:      list[float | None] = []
        leg_ticks: list[float | None] = []
        for t in self.no_token_ids:
            ask = self._best_ask.get(t)
            if ask is None:
                continue
            token_ids.append(t)
            asks.append(ask)
            # None propagates as "depth unknown" — see _handle_price_change.
            sizes.append(self._ask_size.get(t))
            bids.append(self._best_bid.get(t))
            leg_ticks.append(self._tick_size.get(t))

        if len(token_ids) < self.MIN_QUOTED_LEGS:
            return None

        fingerprint = (tuple(token_ids), tuple(asks), tuple(sizes), tuple(bids))
        if fingerprint == self._last_pushed:
            return None
        self._last_pushed = fingerprint

        now = time.monotonic()
        self.last_tick_monotonic = now
        _ticks = [v for v in self._tick_size.values() if v is not None]
        return {
            "type":              "neg_risk_tick",
            "condition_id":      self.group_id,
            "outcome_token_ids": token_ids,
            "no_asks":           asks,
            "no_ask_sizes":      sizes,
            # Per-leg best bids and tick grids. The single coarse tick below is
            # kept for order validity, but the detector needs the real per-leg
            # grid to know what improving a book actually costs.
            "no_best_bids":      bids,
            "leg_tick_sizes":    leg_ticks,
            # Coarsest observed grid — valid on every member market.
            "tick_size":         max(_ticks) if _ticks else None,
            "n_group_outcomes":  len(self.no_token_ids),
            "ts":                now,
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


def _expand_price_changes(event: dict) -> list[dict]:
    """
    Fan a batched `price_change` frame out into one event per asset.

    Polymarket does NOT send the flat shape this module originally assumed
    ({"event_type": "price_change", "asset_id": …, "price": …, "side": …}).
    The live frame batches every affected asset into a `price_changes` array
    and carries NO top-level asset_id:

        {"market": "0x…",
         "price_changes": [
             {"asset_id": "…", "price": "0.5", "size": "2430", "side": "BUY",
              "best_bid": "0.997", "best_ask": "0.998"}, …],
         "timestamp": "…"}

    Routing keys off asset_id, so an unexpanded frame resolves to "" and is
    dropped outright — the feed then only ever updates from `book` snapshots
    and goes minutes at a time without seeing a quote move.  Expanding here
    keeps every downstream handler working on plain per-asset events.

    Each entry also carries an authoritative `best_ask`, which beats inferring
    the top of book from level deltas: when the best level is consumed, a delta
    tells us only that it vanished, while `best_ask` names the new top.
    """
    changes = event.get("price_changes")
    if not isinstance(changes, list):
        return [event]
    out: list[dict] = []
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        asset_id = ch.get("asset_id") or ch.get("token_id") or ""
        if not asset_id:
            continue
        out.append({
            "event_type": "price_change",
            "asset_id":   asset_id,
            "price":      ch.get("price"),
            "size":       ch.get("size"),
            "side":       ch.get("side", ""),
            "best_ask":   ch.get("best_ask"),
            "best_bid":   ch.get("best_bid"),
        })
    return out


# Plain-text frames the CLOB WS emits as a matter of course. Polymarket answers
# aiohttp's protocol-level keepalive PING (heartbeat=20s) with the text
# "INVALID OPERATION" rather than a pong, so one arrives per shard roughly every
# 20-30s: ~101k WARNING lines per day across 34 shards, which buried real errors
# and grew the journal by ~160 MB. The feeds are unaffected, so these are logged
# at DEBUG. Anything NOT on this list is still a WARNING with its payload.
_BENIGN_WS_TEXT: frozenset[str] = frozenset({
    "INVALID OPERATION",
    "PONG",
    "PING",
})


def _parse_events(raw: str) -> list[dict]:
    """Parse a raw WS text frame into a list of per-asset event dicts."""
    _parse_t0 = time.monotonic()
    try:
        data = _loads(raw)
    except json.JSONDecodeError:
        if raw.strip().upper() in _BENIGN_WS_TEXT:
            logger.debug("WS control frame: %s", raw[:40])
        else:
            logger.warning("Non-JSON WS message dropped: %s", raw[:120])
        return []
    WS_PARSE_SECONDS.observe(time.monotonic() - _parse_t0)
    events = data if isinstance(data, list) else [data]
    out: list[dict] = []
    for e in events:
        if isinstance(e, dict):
            out.extend(_expand_price_changes(e))
    return out


def _explicit_best_ask(event: dict) -> Optional[float]:
    """Authoritative best ask carried on the event, when present and sane."""
    raw = event.get("best_ask")
    if raw is None:
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    return price if 0.0 < price < 1.0 else None


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

        # condition_id (or negRiskMarketID) → _MarketState | _NegRiskGroupState
        self._states:  dict[str, object] = {}
        # asset_id → every state that wants this token's events.  A NO token can
        # belong to BOTH a binary market and a NegRisk group, so routing is
        # one-to-many; a plain dict would silently drop one of the two.
        self._routing: dict[str, list] = {}

    # ── Membership ─────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._states)

    def condition_ids(self) -> list[str]:
        return list(self._states.keys())

    def contains(self, condition_id: str) -> bool:
        return condition_id in self._states

    def _route(self, asset_id: str, state: object) -> None:
        self._routing.setdefault(asset_id, []).append(state)

    def _unroute(self, asset_id: str, state: object) -> None:
        handlers = self._routing.get(asset_id)
        if not handlers:
            return
        try:
            handlers.remove(state)
        except ValueError:
            return
        if not handlers:
            self._routing.pop(asset_id, None)

    def add(self, condition_id: str, yes_token_id: str, no_token_id: str) -> None:
        """Add a market to this shard and (if connected) subscribe incrementally."""
        if condition_id in self._states:
            return
        st = _MarketState(condition_id, yes_token_id, no_token_id)
        self._states[condition_id] = st
        self._route(yes_token_id, st)
        self._route(no_token_id,  st)
        if self._ws is not None and not self._ws.closed:
            # Incremental subscribe on the live socket — no reconnect churn.
            asyncio.ensure_future(self._subscribe([yes_token_id, no_token_id]))

    def add_neg_risk_group(self, group_id: str, no_token_ids: list[str]) -> None:
        """Track a NegRisk group's NO legs on this shard."""
        if group_id in self._states or len(no_token_ids) < 2:
            return
        st = _NegRiskGroupState(group_id, no_token_ids)
        self._states[group_id] = st
        for tid in no_token_ids:
            self._route(tid, st)
        if self._ws is not None and not self._ws.closed:
            asyncio.ensure_future(self._subscribe(list(no_token_ids)))

    def remove(self, condition_id: str) -> None:
        """Stop tracking a market (its subscription is cleaned on next reconnect)."""
        st = self._states.pop(condition_id, None)
        if st is None:
            return
        for tid in st.token_ids:
            self._unroute(tid, st)

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
        changed: list = []
        for event in _parse_events(raw):
            asset_id = event.get("asset_id") or event.get("token_id") or ""
            for st in self._routing.get(asset_id, ()):   # () = no longer tracked
                if st.update_leg(asset_id, event) and st not in changed:
                    changed.append(st)
        for st in changed:
            tick = st.build_tick()
            if tick is not None:
                _queue_push(self._queue, tick)
