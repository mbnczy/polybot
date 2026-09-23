"""
execution/cross_ws.py
─────────────────────
CrossBookWatch — the books of the pairs nearest their threshold, pushed.

CrossGuard prices its pairs by polling. Batched, one pass reads 800 books, but
the reader exports twelve thousand pairs, so on 2026-09-23 the stalest price
the guard acted on was three minutes old. The arbitrages it exists for do not
wait that long: the ones that were real (NVDA, WTI, Micron, XRP) were price
ladders that opened when the underlying moved, and closed again as soon as
someone repriced the other rung.

This keeps a WebSocket subscription on the two tokens a taker entry buys — NO
on the narrow market, YES on the broad — for the pairs the guard last priced
closest to its threshold, and calls back the moment their asks add up to an
edge. It decides nothing: the guard reads both books over REST and runs every
gate before anything is sent. A stale socket can therefore cost a book read,
never a trade.

READ ONLY: subscribes to the public market channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

import aiohttp

from strategy.cross_exit import buy_cost

logger = logging.getLogger(__name__)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Tokens per connection: 250 held for days in scripts/record_ws_books.py.
CROSS_WS_PER_CONN: int = int(os.environ.get("CROSS_WS_PER_CONN", 250))
# A callback fires this far below the guard's own threshold: the socket's view
# is only a hint, and a hint that is slightly early costs one book read.
CROSS_WS_SLACK: float = float(os.environ.get("CROSS_WS_SLACK", 0.01))
# The same quotes on a pair are not news again for this long.
CROSS_WS_REPEAT_S: float = float(os.environ.get("CROSS_WS_REPEAT_S", 30.0))
# New tokens wait at most this long for a resubscription. The nearest pairs
# shift a little every pass; resubscribing on each shift reconnected all the
# sockets every 15 s on the first live start.
CROSS_WS_RESUB_S: float = float(os.environ.get("CROSS_WS_RESUB_S", 300.0))


@dataclass(frozen=True)
class WatchPair:
    key:       tuple
    no_token:  str             # NO on the narrow market, bought at its ask
    yes_token: str             # YES on the broad market, bought at its ask
    no_rate:   "float | None" = None
    yes_rate:  "float | None" = None


class _Top:
    """Best bid and ask of one token, from snapshots and deltas."""
    __slots__ = ("bids", "asks", "bid", "ask")

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.bid: "float | None" = None
        self.ask: "float | None" = None

    def load(self, bids, asks) -> None:
        self.bids, self.asks = _levels(bids), _levels(asks)
        self.bid = max(self.bids) if self.bids else None
        self.ask = min(self.asks) if self.asks else None

    def change(self, side: str, price, size, best_bid, best_ask) -> None:
        try:
            p, s = float(price), float(size)
        except (TypeError, ValueError):
            p = s = None
        if p is not None:
            levels = self.bids if side.upper() == "BUY" else self.asks
            if s <= 0.0:
                levels.pop(p, None)
            else:
                levels[p] = s
        # The exchange names the new top outright; a delta alone cannot say
        # what replaced a level that was consumed.
        self.bid = _price(best_bid, max(self.bids) if self.bids else None)
        self.ask = _price(best_ask, min(self.asks) if self.asks else None)


def _levels(raw) -> dict[float, float]:
    out: dict[float, float] = {}
    for lvl in raw or []:
        try:
            p, s = float(lvl["price"]), float(lvl["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if s > 0.0:
            out[p] = s
    return out


def _price(raw, fallback):
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    return v if 0.0 < v < 1.0 else fallback


class CrossBookWatch:
    """
    Subscribes to the watched pairs' entry tokens and calls
    `on_edge(key, edge)` when a pair's two asks, fees included, come within
    `slack` of `min_edge`. `watch()` replaces the set; `run()` keeps the
    sockets up until cancelled.
    """

    def __init__(self, on_edge: Callable[[tuple, float], None], *, min_edge: float,
                 slack: float = CROSS_WS_SLACK, per_conn: int = CROSS_WS_PER_CONN,
                 repeat_s: float = CROSS_WS_REPEAT_S, resub_s: float = CROSS_WS_RESUB_S,
                 url: str = _WS_URL) -> None:
        self._on_edge = on_edge
        self._min_edge = min_edge
        self._slack = slack
        self._per_conn = max(1, per_conn)
        self._repeat_s = repeat_s
        self._url = url
        self._pairs: dict[tuple, WatchPair] = {}
        self._by_token: dict[str, list[WatchPair]] = {}
        self._tops: dict[str, _Top] = {}
        self._fired: dict[tuple, tuple[float, float, float]] = {}   # key → (no, yes, when)
        self._changed = asyncio.Event()
        self._resub_s = resub_s
        self._resub_at = float("-inf")
        # Tokens on the live sockets. Books of tokens no longer watched stay
        # current until the next resubscription drops them, so a pair that
        # comes back is priced at once rather than after a new snapshot.
        self._subscribed: set[str] = set()
        self.stats = {"frames": 0, "fired": 0, "reconnects": 0}

    # ── the set ──────────────────────────────────────────────────────────────

    @property
    def tokens(self) -> "set[str]":
        return set(self._by_token)

    def watch(self, pairs: "list[WatchPair]") -> None:
        """Replace the watched pairs; the sockets resubscribe when the tokens change."""
        before = set(self._by_token)
        self._pairs = {p.key: p for p in pairs}
        by_token: dict[str, list[WatchPair]] = {}
        for p in pairs:
            by_token.setdefault(p.no_token, []).append(p)
            by_token.setdefault(p.yes_token, []).append(p)
        self._by_token = by_token
        self._fired = {k: v for k, v in self._fired.items() if k in self._pairs}
        for t in before - set(by_token) - self._subscribed:
            self._tops.pop(t, None)
        if (set(by_token) - self._subscribed
                and time.monotonic() - self._resub_at >= self._resub_s):
            self._changed.set()

    def _tracked(self, token: str) -> bool:
        return token in self._by_token or token in self._subscribed

    # ── events ───────────────────────────────────────────────────────────────

    def handle(self, raw: "str | bytes") -> None:
        """One frame from the market channel."""
        self.stats["frames"] += 1
        try:
            data = json.loads(raw)
        except ValueError:
            return                                   # "INVALID OPERATION" and the like
        moved: set[str] = set()
        for e in data if isinstance(data, list) else [data]:
            if not isinstance(e, dict):
                continue
            if str(e.get("event_type") or "").lower() == "book":
                token = str(e.get("asset_id") or "")
                if self._tracked(token):
                    self._tops.setdefault(token, _Top()).load(e.get("bids"), e.get("asks"))
                    moved.add(token)
                continue
            changes = e.get("price_changes")
            if isinstance(changes, list):
                for ch in changes:
                    token = str(ch.get("asset_id") or "")
                    if not self._tracked(token):
                        continue
                    self._tops.setdefault(token, _Top()).change(
                        str(ch.get("side") or ""), ch.get("price"), ch.get("size"),
                        ch.get("best_bid"), ch.get("best_ask"))
                    moved.add(token)
        for token in moved:
            for p in self._by_token.get(token, ()):
                self._check(p)

    def _check(self, p: WatchPair) -> None:
        no, yes = self._tops.get(p.no_token), self._tops.get(p.yes_token)
        if no is None or yes is None or no.ask is None or yes.ask is None:
            return
        if no.bid is not None and no.bid >= no.ask or yes.bid is not None and yes.bid >= yes.ask:
            return                                   # a crossed book is a stale one
        edge = 1.0 - buy_cost(no.ask, p.no_rate) - buy_cost(yes.ask, p.yes_rate)
        if edge < self._min_edge - self._slack:
            return
        now = time.monotonic()
        last = self._fired.get(p.key)
        if last and last[:2] == (no.ask, yes.ask) and now - last[2] < self._repeat_s:
            return                                   # the same quotes: already said
        self._fired[p.key] = (no.ask, yes.ask, now)
        self.stats["fired"] += 1
        try:
            self._on_edge(p.key, edge)
        except Exception as exc:  # noqa: BLE001 — a callback must not end the feed
            logger.warning("CrossBookWatch | callback failed: %s", exc)

    # ── sockets ──────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Keep one socket per `per_conn` tokens; resubscribe when the set changes."""
        tasks: list[asyncio.Task] = []
        try:
            while True:
                self._changed.clear()
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                tokens = sorted(self._by_token)
                self._subscribed = set(tokens)
                self._resub_at = time.monotonic()
                for t in [t for t in self._tops if t not in self._subscribed]:
                    del self._tops[t]
                chunks = [tokens[i:i + self._per_conn]
                          for i in range(0, len(tokens), self._per_conn)]
                tasks = [asyncio.create_task(self._socket(c, i)) for i, c in enumerate(chunks)]
                if tokens:
                    logger.info("CrossBookWatch | %d pair(s), %d token(s) on %d socket(s)",
                                len(self._pairs), len(tokens), len(chunks))
                await self._changed.wait()
                # Let a burst of watch() calls settle into one resubscription.
                await asyncio.sleep(1.0)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _socket(self, tokens: "list[str]", sid: int) -> None:
        backoff = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(self._url, heartbeat=20, max_msg_size=0) as ws:
                        await ws.send_str(json.dumps({"assets_ids": tokens, "type": "market"}))
                        connected = time.monotonic()
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self.handle(msg.data)
                                if time.monotonic() - connected > 10.0:
                                    backoff = 1.0
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — reconnect whatever it was
                logger.warning("CrossBookWatch | socket %d: %s — reconnecting in %.0fs",
                               sid, exc, backoff)
            self.stats["reconnects"] += 1
            # Books from before the gap are not books any more.
            for t in tokens:
                self._tops.pop(t, None)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
