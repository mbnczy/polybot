#!/usr/bin/env python3
"""
fill_model.py — does a resting bid at the top of the book fill, and what is it
worth once it has?

Both maker strategies rest on the same bet: an order a tick inside the spread
gets filled, and holds its value after. The main strategy's record says the
second half fails — 20 pairs placed, none completed, 15 left with the one leg
that filled as the market moved through it — and the cross module's paper rests
could only be judged at kick-off, days away. This measures the bet directly on
every market the WebSocket recorder watched (record_ws_books.py).

Every --every seconds, for every market whose recorded book is two-sided with
room inside the spread, two hypothetical orders go up:

    buy  YES at bid + tick     filled by a later trade at a YES price ≤ it
    sell YES at ask − tick     filled by a later trade at a YES price ≥ it
                               (the same as buying NO at 1 − that)

Being a tick better than the book, any such trade would have met ours first.
Trades come from the public feed, NO-token trades turned into YES prices. For a
filled order the mark-out is the recorded mid 1, 5 and 30 minutes after the fill
against the price we traded at — positive when the fill was worth having.

Split by where the order stood in the match (before kick-off, by hours, or in
play), by how far out the price was (a tail below 0.10 or above 0.90, or the
middle), and by market family. READ ONLY: public data, nothing is sent.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from scripts.analyze_ws_recording import _lines, load_tokens  # noqa: E402

_TRADES = "https://data-api.polymarket.com/trades"
HORIZONS = (180.0, 1800.0, 7200.0)
MARKOUTS = (60.0, 300.0, 1800.0)


def family(q: str) -> str:
    if q.startswith("Spread:"):
        return "spread"
    if "O/U" in q:
        return "match O/U" if q.split(": ", 1)[-1].startswith("O/U") else "half/team O/U"
    if "Both Teams" in q:
        return "btts"
    if "win" in q.lower() or "draw" in q.lower():
        return "winner/draw"
    return "other"


def stage(t: float, kick: "float | None") -> str:
    if kick is None:
        return "?"
    h = (kick - t) / 3600.0
    return ("in play" if h <= 0 else "<1h" if h < 1 else "1-6h" if h < 6 else ">6h")


def tick_of(bid: float, ask: float) -> float:
    return 0.01 if abs(bid * 100 - round(bid * 100)) < 1e-6 and \
        abs(ask * 100 - round(ask * 100)) < 1e-6 else 0.001


def load_books(rec: Path, wanted: set) -> dict:
    """index → sorted [(t, bid, ask)], crossed books dropped."""
    out: dict[int, list] = defaultdict(list)
    for f in sorted(rec.glob("*.csv.gz")):
        for line in _lines(f):
            body = line[1:] if line[:1] in ("S", "R") else line
            p = body.rstrip("\n").split(",")
            if len(p) != 6:
                continue
            i = int(p[1])
            if i not in wanted:
                continue
            bid = float(p[2]) if p[2] else None
            ask = float(p[3]) if p[3] else None
            if bid is not None and ask is not None and bid >= ask:
                continue
            out[i].append((int(p[0]) / 1000.0, bid, ask))
    for s in out.values():
        s.sort(key=lambda x: x[0])
    return out


def state_at(series: list, times: list, t: float):
    i = bisect.bisect_right(times, t) - 1
    return series[i] if i >= 0 else None


def trades_of(cid: str, since: float) -> list:
    out, offset = [], 0
    with httpx.Client(timeout=30) as c:
        while offset < 5000:
            r = c.get(_TRADES, params={"market": cid, "limit": 500, "offset": offset})
            if r.status_code != 200:
                break
            page = r.json()
            out += page
            if len(page) < 500 or page[-1]["timestamp"] < since:
                break
            offset += 500
    return sorted((t for t in out if t["timestamp"] >= since), key=lambda t: t["timestamp"])


def main(args) -> int:
    rec = Path(args.recording)
    tokens = load_tokens(rec / "tokens.jsonl")
    random.seed(args.seed)
    pick = random.sample(sorted(tokens), min(args.markets, len(tokens)))
    books = load_books(rec, set(pick))
    pick = [i for i in pick if len(books.get(i, ())) >= 2]
    t0 = min(books[i][0][0] for i in pick)
    with ThreadPoolExecutor(6) as ex:
        trades = dict(zip(pick, ex.map(lambda i: trades_of(tokens[i]["cid"], t0), pick)))
    traded = sum(1 for i in pick if trades[i])
    print(f"{len(pick)} recorded markets sampled, {traded} with trades during the recording; "
          f"orders every {args.every:.0f}s\n")

    cells: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    for i in pick:
        # A market nobody traded in still counts: its orders simply never fill.
        row, series, tr = tokens[i], books[i], trades[i]
        times = [s[0] for s in series]
        kick = None
        try:
            kick = datetime.fromisoformat(str(row["kick"]).replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
        yes = row["token"]
        tpx = [(float(x["timestamp"]), float(x["price"]) if str(x["asset"]) == yes
                else 1.0 - float(x["price"])) for x in tr]
        t_end = series[-1][0]
        t = series[0][0]
        while t < t_end:
            st = state_at(series, times, t)
            t += args.every
            if not st or st[1] is None or st[2] is None:
                continue
            _, bid, ask = st
            tick = tick_of(bid, ask)
            if ask - bid < 2 * tick - 1e-9:
                continue                               # no room to stand a tick better
            mid0 = (bid + ask) / 2.0
            tail = "tail" if mid0 < 0.10 or mid0 > 0.90 else "middle"
            for side, price in (("buy YES", bid + tick), ("sell YES", ask - tick)):
                key = (stage(st[0], kick), tail, family(row["q"]), side)
                c = cells[key]
                c["orders"] += 1
                fill = next((ft for ft, yp in tpx if ft > st[0] and
                             (yp <= price + 1e-9 if side == "buy YES" else yp >= price - 1e-9)),
                            None)
                if fill is None:
                    continue
                for h in HORIZONS:
                    if fill - st[0] <= h:
                        c[f"fill{int(h)}"] += 1
                if fill - st[0] > HORIZONS[-1]:
                    continue
                for m in MARKOUTS:
                    later = state_at(series, times, fill + m)
                    if later and later[1] is not None and later[2] is not None:
                        mid = (later[1] + later[2]) / 2.0
                        c[f"mo{int(m)}"] += (mid - price) if side == "buy YES" else (price - mid)
                        c[f"mo{int(m)}n"] += 1

    def report(title, keyf):
        agg: dict = defaultdict(lambda: defaultdict(float))
        for key, c in cells.items():
            for k, v in c.items():
                agg[keyf(key)][k] += v
        print(f"── by {title}")
        print(f"   {'':22} {'orders':>7} {'fill 3m':>8} {'30m':>7} {'2h':>7}   "
              f"{'mark-out 1m':>11} {'5m':>7} {'30m':>7}")
        for k in sorted(agg, key=lambda k: -agg[k]["orders"]):
            c = agg[k]
            n = c["orders"]
            if n < args.min_orders:
                continue
            mo = [(c[f"mo{int(m)}"] / c[f"mo{int(m)}n"]) if c[f"mo{int(m)}n"] else float("nan")
                  for m in MARKOUTS]
            print(f"   {str(k)[:22]:22} {int(n):7d} {c['fill180'] / n:8.1%} {c['fill1800'] / n:7.1%} "
                  f"{c['fill7200'] / n:7.1%}   {mo[0]:+11.4f} {mo[1]:+7.4f} {mo[2]:+7.4f}")
        print()

    report("stage", lambda k: k[0])
    report("stage and price", lambda k: f"{k[0]} {k[1]}")
    report("family", lambda k: k[2])
    report("side", lambda k: k[3])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default="/home/ubuntu/polybot-dev/recordings")
    ap.add_argument("--markets", type=int, default=800, help="markets to sample")
    ap.add_argument("--every", type=float, default=600.0, help="seconds between orders")
    ap.add_argument("--min-orders", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    sys.exit(main(ap.parse_args()))
