#!/usr/bin/env python3
"""
edge_study.py — two edge classes the book-top arbitrage never looked at.

Everything measured so far hit one wall: at the top of the book, Polymarket's
prices are efficient to within the fee. Two classes do not live at that wall:

1. Calibration. A price is a probability; if outcomes priced at 0.05 happen
   less than 5% of the time, buying the other side has an edge over many small
   bets. Prediction markets are known to overprice longshots. For each resolved
   market: the volume-weighted YES price over the six hours before the event
   (a sports market's game start, otherwise the day before it closed), against
   how it resolved.

2. Resolution lag. After a game has certainly ended and before the market pays,
   the winning side is worth 1. Every trade on it well below that is money left
   on the table for whoever knows the result. For sports markets: trades on the
   eventual winner below --snipe-below, later than the game can still be going.

READ ONLY: public Gamma and data-api reads, nothing is sent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

_GAMMA = "https://gamma-api.polymarket.com/markets"
_TRADES = "https://data-api.polymarket.com/trades"
_ISO = "%Y-%m-%dT%H:%M:%SZ"
_LOCAL = threading.local()

# How long after its start a game can still be going, generously.
_GAME_HOURS = (("table_tennis", 1.5), ("esports", 5.0), ("tennis", 5.0), ("cricket", 9.0),
               ("", 4.0))


def _get(url, params):
    c = getattr(_LOCAL, "c", None)
    if c is None:
        c = _LOCAL.c = httpx.Client(timeout=30)
    for attempt in range(4):
        try:
            r = c.get(url, params=params)
        except httpx.HTTPError:
            time.sleep(1 + 2 * attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1 + 2 * attempt)
            continue
        return r
    return None


def _ts(raw) -> "float | None":
    if not raw:
        return None
    s = str(raw).replace("Z", "+00:00").replace(" ", "T")
    if re.search(r"[+-]\d\d$", s):
        s += ":00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def family(m: dict) -> str:
    kind = str(m.get("sportsMarketType") or "")
    q = str(m.get("question") or "")
    if kind:
        if "tennis" in kind and "table" not in kind:
            return "tennis"
        if "table_tennis" in kind:
            return "table tennis"
        if kind in ("totals", "team_totals"):
            return "sports totals"
        if "spread" in kind or "handicap" in kind:
            return "sports spreads"
        if kind in ("moneyline", "draw", "both_teams_to_score"):
            return "sports moneyline/draw"
        return "sports other"
    if re.search(r"\((HIGH|LOW)\)|finish week|close above|closes above", q):
        return "stock/commodity ladders"
    if re.search(r"price of (Bitcoin|Ethereum|Solana|XRP)|Bitcoin|Ethereum|Solana|XRP", q):
        return "crypto"
    return "other events"


def sample_markets(days: float, slices: int, per_slice: int, seed: int) -> list[dict]:
    random.seed(seed)
    now = datetime.now(timezone.utc)
    out, seen = [], set()
    starts = sorted(random.uniform(0.2, days) for _ in range(slices))
    for d in starts:
        hi = now - timedelta(days=d)
        lo = hi - timedelta(hours=2)
        got = []
        for offset in range(0, 1000, 100):
            r = _get(_GAMMA, {"closed": "true", "limit": 100, "offset": offset,
                              "end_date_min": lo.strftime(_ISO), "end_date_max": hi.strftime(_ISO),
                              "order": "endDate", "ascending": "true"})
            page = r.json() if r is not None and r.status_code == 200 else []
            got += page
            if len(page) < 100:
                break
        got = [m for m in got if "Up or Down" not in (m.get("question") or "")
               and m.get("conditionId") not in seen]
        random.shuffle(got)
        for m in got[:per_slice]:
            seen.add(m.get("conditionId"))
            out.append(m)
    return out


def resolved_yes(m: dict) -> "int | None":
    try:
        p = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except (TypeError, ValueError):
        return None
    if len(p) != 2 or sorted(p) != [0.0, 1.0]:
        return None                               # a 50-50, or not resolved yet
    return 1 if p[0] == 1.0 else 0


def trades(cid: str) -> list:
    out = []
    for offset in range(0, 1500, 500):
        r = _get(_TRADES, {"market": cid, "limit": 500, "offset": offset})
        page = r.json() if r is not None and r.status_code == 200 else []
        out += page
        if len(page) < 500:
            break
    return out


def study(m: dict, args) -> "dict | None":
    y = resolved_yes(m)
    if y is None:
        return None
    try:
        yes_token = json.loads(m.get("clobTokenIds") or "[]")[0]
    except (TypeError, ValueError, IndexError):
        return None
    closed = _ts(m.get("closedTime"))
    start = _ts(m.get("gameStartTime"))
    ref = start if start else (closed - 86_400.0 if closed else None)
    if ref is None:
        return None
    tr = trades(m["conditionId"])
    if not tr:
        return None
    # (time, YES price, size, whether the taker bought YES)
    px = [(float(t["timestamp"]),
           float(t["price"]) if str(t["asset"]) == str(yes_token) else 1.0 - float(t["price"]),
           float(t["size"]),
           (str(t["asset"]) == str(yes_token)) == (str(t.get("side")).upper() == "BUY"))
          for t in tr]
    pre = [(p, s) for t, p, s, _ in px if ref - 6 * 3600 <= t <= ref - 300]
    row = {"family": family(m), "y": y, "pre": None, "snipe_n": 0, "snipe_usdc": 0.0,
           "snipe_edge": 0.0, "lift_n": 0, "lift_edge": 0.0, "q": m.get("question", "")}
    if len(pre) >= 2 and sum(s for _, s in pre) > 0:
        row["pre"] = sum(p * s for p, s in pre) / sum(s for _, s in pre)
    if start and closed:
        kind = str(m.get("sportsMarketType") or "")
        hours = next(h for key, h in _GAME_HOURS if key in kind)
        done = start + hours * 3600
        for t, p, s, bought_yes in px:
            if done <= t < closed:
                p_win = p if y == 1 else 1.0 - p            # the winner's price
                if p_win <= args.snipe_below:
                    row["snipe_n"] += 1
                    row["snipe_usdc"] += s * p_win
                    row["snipe_edge"] += s * (1.0 - p_win)
                    # Proof an offer stood there: the taker bought the winner.
                    if bought_yes == (y == 1):
                        row["lift_n"] += 1
                        row["lift_edge"] += s * (1.0 - p_win)
    return row


def ci(k: int, n: int) -> str:
    if n == 0:
        return ""
    p = k / n
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
    return f"±{half:.3f}"


def main(args) -> int:
    ms = sample_markets(args.days, args.slices, args.per_slice, args.seed)
    print(f"{len(ms)} resolved markets sampled from {args.slices} two-hour slices of the last "
          f"{args.days:g} days (Up or Down excluded)")
    with ThreadPoolExecutor(3) as ex:
        rows = [r for r in ex.map(lambda m: study(m, args), ms) if r]
    cal = [r for r in rows if r["pre"] is not None]
    print(f"{len(rows)} resolved with trades; {len(cal)} with a price in the six hours before\n")

    edges = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 1.0001]
    print("── calibration: price before the event against how often YES happened")
    print(f"   {'bucket':12} {'n':>5} {'avg price':>9} {'happened':>9} {'':>7} {'gap':>7}")
    for lo, hi in zip(edges, edges[1:]):
        b = [r for r in cal if lo <= r["pre"] < hi]
        if not b:
            continue
        k = sum(r["y"] for r in b)
        avg = sum(r["pre"] for r in b) / len(b)
        print(f"   {lo:.2f}-{min(hi, 1):.2f}  {len(b):5d} {avg:9.3f} {k / len(b):9.3f} "
              f"{ci(k, len(b)):>7} {k / len(b) - avg:+7.3f}")
    print()
    print("── calibration by family: longshots (below 0.15) and favourites (above 0.85)")
    fams = sorted({r["family"] for r in cal})
    for f in fams:
        for name, sel in (("longshots", lambda r: r["pre"] < 0.15),
                          ("favourites", lambda r: r["pre"] > 0.85)):
            b = [r for r in cal if r["family"] == f and sel(r)]
            if len(b) < args.min_n:
                continue
            k = sum(r["y"] for r in b)
            avg = sum(r["pre"] for r in b) / len(b)
            print(f"   {f[:24]:24} {name:10} n={len(b):4d} priced {avg:.3f} happened "
                  f"{k / len(b):.3f} {ci(k, len(b))}  gap {k / len(b) - avg:+.3f}")
    print()
    sn = [r for r in rows if r["snipe_n"]]
    sports = [r for r in rows if r["family"].startswith(("sports", "tennis", "table"))]
    print(f"── resolution lag: trades on the eventual winner at ≤ {args.snipe_below:.2f} after the "
          f"game had certainly ended")
    print(f"   {len(sn)} of {len(sports)} sports markets; {sum(r['snipe_n'] for r in sn)} trades, "
          f"{sum(r['snipe_usdc'] for r in sn):.0f} USDC paid for "
          f"{sum(r['snipe_edge'] for r in sn):.0f} USDC of certain profit")
    print(f"   of which a taker bought the winner (an offer provably stood there): "
          f"{sum(r['lift_n'] for r in sn)} trades, {sum(r['lift_edge'] for r in sn):.0f} USDC")
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for r in sn:
        by[r["family"]][0] += 1
        by[r["family"]][1] += r["snipe_usdc"]
        by[r["family"]][2] += r["snipe_edge"]
    for f, (n, usd, edge) in sorted(by.items(), key=lambda kv: -kv[1][2]):
        print(f"   {f[:26]:26} {n:4d} market(s) · {usd:8.0f} USDC traded · {edge:7.0f} USDC left")
    for r in sorted(sn, key=lambda r: -r["snipe_edge"])[:6]:
        print(f"      {r['snipe_edge']:7.1f}  {r['q'][:80]}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--slices", type=int, default=40)
    ap.add_argument("--per-slice", type=int, default=80)
    ap.add_argument("--snipe-below", type=float, default=0.97)
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    sys.exit(main(ap.parse_args()))
