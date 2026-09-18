#!/usr/bin/env python3
"""
record_ws_books.py — record the top of book of live and upcoming matches, READ ONLY.

Why
───
Every snapshot so far says the implications are priced monotonically: of 320
confirmed pairs with a quotable book on both legs, none was violated, and the
best edge is +0.0000. If gaps open at all, they open in the seconds after a goal,
when one sub-market reprices before another — and a match market's endDate is
its kick-off, so the reader and the guard never look at a match once it starts.

This records what the market WebSocket says during those minutes, so any set of
pairs — the model's, or ones generated from the titles — can be evaluated
afterwards against it: how often a gap opens, how wide, and for how long.

What is recorded
────────────────
The YES token of every market (bar exact scores) in every match that kicked off
in the last --behind-h hours or kicks off in the next --ahead-h. The NO book is
the YES book mirrored, so it needs no subscription of its own. A line is written
when a token's best bid or best ask moves:

    epoch_ms,token_index,bid,ask,bid_size,ask_size

and every --snapshot-s every book is read again from REST (POST /books) and
written with an R in front when it differs from its last R line, every book
once per ten minutes. Those lines are the ground truth: the WebSocket
state can go stale — on 2026-09-18 a consumed level stayed in a replayed book
for thirteen minutes, showing a 0.99 bid against a 0.54 ask — and the REST read
replaces it. tokens.jsonl maps each index to its market. Files are
gzipped per hour, and recording stops at --max-mb on disk.

It never builds a signing client and holds no key: market data only.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import aiohttp  # noqa: E402

logger = logging.getLogger("recorder")

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_BOOKS_URL = "https://clob.polymarket.com/books"
_EXACT = re.compile(r"exact score", re.I)


# ── which markets ─────────────────────────────────────────────────────────────

def select(markets: list[dict]) -> list[dict]:
    """Every market of every match that has an over/under set, bar exact scores."""
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        ev = (m.get("events") or [{}])[0].get("id")
        if ev is not None:
            by_event.setdefault(str(ev), []).append(m)
    out = []
    for ev, ms in by_event.items():
        qs = [str(m.get("question") or "") for m in ms]
        if not any(" vs. " in q and "O/U" in q for q in qs):
            continue
        for m in ms:
            q = str(m.get("question") or "")
            if _EXACT.search(q):
                continue
            try:
                yes = json.loads(m.get("clobTokenIds") or "[]")[0]
                outcomes = json.loads(m.get("outcomes") or "[]")
            except (TypeError, ValueError, IndexError):
                continue
            out.append({"token": str(yes), "cid": m.get("conditionId"), "ev": ev, "q": q,
                        "kick": m.get("endDate"), "outcomes": outcomes})
    return out


# ── books ─────────────────────────────────────────────────────────────────────

class Book:
    __slots__ = ("bids", "asks", "bid", "ask")

    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.bid: "float | None" = None
        self.ask: "float | None" = None

    def load(self, bids, asks):
        self.bids = _levels(bids)
        self.asks = _levels(asks)
        self.bid = max(self.bids) if self.bids else None
        self.ask = min(self.asks) if self.asks else None

    def change(self, side: str, price, size, best_bid, best_ask):
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
        # The exchange names the new top outright; a delta alone cannot say what
        # replaced a level that was consumed.
        self.bid = _price(best_bid, max(self.bids) if self.bids else None)
        self.ask = _price(best_ask, min(self.asks) if self.asks else None)

    def top(self):
        return (self.bid, self.ask, self.bids.get(self.bid, 0.0) if self.bid else 0.0,
                self.asks.get(self.ask, 0.0) if self.ask else 0.0)


def _levels(raw) -> dict[float, float]:
    out = {}
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


# ── output ────────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self, out_dir: Path, max_bytes: int):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.index: dict[str, int] = {}
        self.books: dict[str, Book] = {}
        self.last: dict[str, tuple] = {}
        self.last_r: dict[str, tuple] = {}      # the last REST top written per token
        self._fh = None
        self._hour = None
        self.lines = 0
        self.frames = 0
        self.full = False
        self._tokens_fh = open(self.dir / "tokens.jsonl", "a", encoding="utf-8")
        for line in (self.dir / "tokens.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                self.index[row["token"]] = row["i"]
            except (ValueError, KeyError):
                continue

    def register(self, rows: list[dict]) -> None:
        for row in rows:
            if row["token"] not in self.index:
                self.index[row["token"]] = len(self.index)
                self._tokens_fh.write(json.dumps({"i": self.index[row["token"]], **row}) + "\n")
        self._tokens_fh.flush()

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("*.csv.gz"))

    def _file(self):
        hour = time.strftime("%Y%m%d-%H", time.gmtime())
        if hour != self._hour:
            if self._fh is not None:
                self._fh.close()
            self._fh = gzip.open(self.dir / f"{hour}.csv.gz", "at", encoding="ascii")
            self._hour = hour
        return self._fh

    def write(self, token: str, snapshot: bool = False, mark: str = "S",
              force: bool = False) -> None:
        if self.full:
            return
        book = self.books.get(token)
        if book is None:
            return
        top = book.top()
        if not snapshot and self.last.get(token, (None, None))[:2] == top[:2]:
            return                               # size alone moved: not a line
        if snapshot and not force and self.last_r.get(token, (None, None))[:2] == top[:2]:
            return                               # the REST read says what it said before
        if snapshot:
            self.last_r[token] = top
        self.last[token] = top
        bid, ask, bs, az = top
        self._file().write(
            f"{mark if snapshot else ''}{int(time.time() * 1000)},{self.index[token]},"
            f"{'' if bid is None else bid},{'' if ask is None else ask},{bs:g},{az:g}\n")
        self.lines += 1

    def event(self, e: dict) -> None:
        etype = str(e.get("event_type") or "").lower()
        if etype == "book":
            token = str(e.get("asset_id") or "")
            if token in self.index:
                self.books.setdefault(token, Book()).load(e.get("bids"), e.get("asks"))
                self.write(token)
            return
        changes = e.get("price_changes")
        if isinstance(changes, list):
            for ch in changes:
                token = str(ch.get("asset_id") or "")
                if token not in self.index:
                    continue
                self.books.setdefault(token, Book()).change(
                    str(ch.get("side") or ""), ch.get("price"), ch.get("size"),
                    ch.get("best_bid"), ch.get("best_ask"))
                self.write(token)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()


# ── connections ───────────────────────────────────────────────────────────────

async def shard(tokens: list[str], rec: Recorder, sid: int) -> None:
    backoff = 1.0
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(_WS_URL, heartbeat=20, max_msg_size=0) as ws:
                    await ws.send_str(json.dumps({"assets_ids": tokens, "type": "market"}))
                    connected = time.monotonic()
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            continue
                        rec.frames += 1
                        try:
                            data = json.loads(msg.data)
                        except ValueError:
                            continue                     # "INVALID OPERATION" and the like
                        for e in data if isinstance(data, list) else [data]:
                            if isinstance(e, dict):
                                rec.event(e)
                        if time.monotonic() - connected > 10.0:
                            backoff = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — reconnect whatever it was
            logger.warning("shard %d: %s — reconnecting in %.0fs", sid, exc, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, 30.0)


async def snapshots(rec: Recorder, every: float, anchor_every: float = 600.0) -> None:
    """
    Every `every` seconds, replace each book with a REST read, and write it as R
    when its top differs from the last R line — every book once per
    `anchor_every`. Writing every book every minute was 16 MB an hour of lines
    saying nothing had changed; the R timeline is the same without them.
    """
    last_anchor = 0.0
    while True:
        await asyncio.sleep(every)
        force = time.time() - last_anchor >= anchor_every
        if force:
            last_anchor = time.time()
        tokens = list(rec.books)
        async with aiohttp.ClientSession() as s:
            for i in range(0, len(tokens), 100):
                part = tokens[i:i + 100]
                try:
                    async with s.post(_BOOKS_URL, json=[{"token_id": t} for t in part],
                                      timeout=aiohttp.ClientTimeout(total=20)) as r:
                        books = await r.json(content_type=None) if r.status == 200 else []
                except Exception as exc:  # noqa: BLE001 — the next round tries again
                    logger.warning("REST resync failed: %s", exc)
                    books = []
                for b in books or ():
                    token = str(b.get("asset_id") or "")
                    if token in rec.books:
                        rec.books[token].load(b.get("bids"), b.get("asks"))
                        rec.write(token, snapshot=True, mark="R", force=force)
        rec.flush()
        if rec.disk_bytes() > rec.max_bytes:
            rec.full = True
            logger.error("recording reached %.0f MB — stopping", rec.max_bytes / 1e6)


async def main(args) -> int:
    import scripts.demo_cross_market as dm   # noqa: PLC0415 — the reader's window fetch

    rec = Recorder(Path(args.out), int(args.max_mb * 1e6))
    stop_at = time.time() + args.hours * 3600.0
    while time.time() < stop_at and not rec.full:
        markets = await asyncio.to_thread(
            dm.fetch_window, args.ahead_h / 24.0, -args.behind_h, 4, "Up or Down")
        rows = select(markets)
        rec.register(rows)
        tokens = [r["token"] for r in rows]
        live = {t for t in tokens}
        for t in [t for t in rec.books if t not in live]:
            rec.books.pop(t, None)
            rec.last.pop(t, None)
            rec.last_r.pop(t, None)
        events = len({r["ev"] for r in rows})
        chunks = [tokens[i:i + args.per_conn] for i in range(0, len(tokens), args.per_conn)]
        logger.info("watching %d market(s) in %d match(es) on %d connection(s)",
                    len(tokens), events, len(chunks))
        tasks = [asyncio.create_task(shard(c, rec, i)) for i, c in enumerate(chunks)]
        tasks.append(asyncio.create_task(snapshots(rec, args.snapshot_s)))
        until = min(time.time() + args.refresh_s, stop_at)
        lines0, frames0, t0 = rec.lines, rec.frames, time.time()
        while time.time() < until and not rec.full:
            await asyncio.sleep(min(300.0, max(1.0, until - time.time())))
            dt = max(1.0, time.time() - t0)
            logger.info("%d line(s), %d frame(s) in %.0fs · %d book(s) live · %.1f MB on disk",
                        rec.lines - lines0, rec.frames - frames0, dt, len(rec.books),
                        rec.disk_bytes() / 1e6)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        rec.flush()
    rec.flush()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/home/ubuntu/polybot-dev/recordings")
    ap.add_argument("--hours", type=float, default=80.0, help="how long to record")
    ap.add_argument("--behind-h", type=float, default=3.0,
                    help="matches that kicked off up to this long ago (in play)")
    ap.add_argument("--ahead-h", type=float, default=6.0,
                    help="and matches kicking off within this long")
    ap.add_argument("--refresh-s", type=float, default=1800.0,
                    help="how often the set of matches is chosen again")
    ap.add_argument("--per-conn", type=int, default=250, help="tokens per connection")
    ap.add_argument("--snapshot-s", type=float, default=60.0)
    ap.add_argument("--max-mb", type=float, default=600.0, help="stop at this much on disk")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(main(args)))
