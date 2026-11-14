#!/usr/bin/env python3
"""
reward_study.py — what would a minimum quote earn from the liquidity rewards?

Polymarket pays a daily amount per market to makers resting orders within
rewards_max_spread of the midpoint, whether or not they fill. Each order scores

    size × ((max_spread − distance) / max_spread)²      (distance in cents)

summed per side; a maker's score is the smaller side (two-sided), or a third of
the larger when only one side is quoted and the midpoint is inside
[0.10, 0.90]. The day's rate is split in proportion to scores.

For every market paying today this reads the book, scores the liquidity already
there, and prices a two-sided quote of the minimum size joining the touch on
both sides: what share it would take, what that pays a day, and the collateral
it holds. It also counts the market's trades in the last day — the flow that
would fill the quote, which is where the cost is.

READ ONLY: public CLOB, Gamma and data-api reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

_CLOB = "https://clob.polymarket.com"
_GAMMA = "https://gamma-api.polymarket.com/markets"
_TRADES = "https://data-api.polymarket.com/trades"


def reward_markets(c: httpx.Client) -> list[dict]:
    out, cursor = [], ""
    for _ in range(40):
        r = c.get(f"{_CLOB}/rewards/markets/current", params={"next_cursor": cursor} if cursor else None)
        d = r.json()
        out += d.get("data") or []
        cursor = d.get("next_cursor") or ""
        if not cursor or cursor == "LTE=":
            break
    return out


def score(levels, mid: float, v: float, side: str) -> float:
    """Σ size × ((v − s)/v)² over the side's levels within v cents of the midpoint."""
    total = 0.0
    for lvl in levels or []:
        p, sz = float(lvl["price"]), float(lvl["size"])
        s = (mid - p if side == "bids" else p - mid) * 100.0
        if 0.0 <= s < v:
            total += sz * ((v - s) / v) ** 2
    return total


def q_min(q1: float, q2: float, mid: float) -> float:
    if 0.10 <= mid <= 0.90:
        return max(min(q1, q2), max(q1, q2) / 3.0)
    return min(q1, q2)


def main(args) -> int:
    with httpx.Client(timeout=30) as c:
        rew = [m for m in reward_markets(c) if float(m.get("total_daily_rate") or 0) >= args.min_rate]
        print(f"{len(rew)} markets pay at least {args.min_rate:g} USDC a day today; "
              f"{sum(float(m['total_daily_rate']) for m in rew):.0f} USDC a day between them")
        meta = {}
        cids = [m["condition_id"] for m in rew]
        for i in range(0, len(cids), 50):
            r = c.get(_GAMMA, params=[("condition_ids", x) for x in cids[i:i + 50]])
            for g in r.json() if r.status_code == 200 else []:
                meta[g["conditionId"]] = g
        toks = {}
        for cid, g in meta.items():
            try:
                toks[cid] = json.loads(g.get("clobTokenIds") or "[]")[0]
            except (TypeError, ValueError, IndexError):
                pass
        books = {}
        tl = list(toks.values())
        for i in range(0, len(tl), 100):
            r = c.post(f"{_CLOB}/books", json=[{"token_id": t} for t in tl[i:i + 100]])
            for b in r.json() if r.status_code == 200 else []:
                books[str(b["asset_id"])] = b

    rows = []
    for m in rew:
        cid = m["condition_id"]
        g, tok = meta.get(cid), toks.get(cid)
        b = books.get(str(tok)) if tok else None
        if not g or not b or not b.get("bids") or not b.get("asks"):
            continue
        bid = max(float(x["price"]) for x in b["bids"])
        ask = min(float(x["price"]) for x in b["asks"])
        if bid >= ask:
            continue
        mid = (bid + ask) / 2.0
        v = float(m.get("rewards_max_spread") or 0)
        size = max(float(m.get("rewards_min_size") or 0), 5.0)
        if v <= 0:
            continue
        q_there = q_min(score(b["bids"], mid, v, "bids"), score(b["asks"], mid, v, "asks"), mid)
        s_touch = max(mid - bid, ask - mid) * 100.0         # joining the touch on both sides
        if s_touch >= v:
            continue                                        # the touch is outside the band
        ours = size * ((v - s_touch) / v) ** 2
        share = ours / (q_there + ours)
        rate = float(m["total_daily_rate"])
        capital = size * bid + size * (1.0 - ask)           # YES at the bid, NO at 1 − ask
        rows.append({"cid": cid, "q": g.get("question", ""), "rate": rate, "mid": mid,
                     "ours": ours, "q_there": q_there,
                     "spread_c": (ask - bid) * 100, "v": v, "size": size, "share": share,
                     "per_day": rate * share, "capital": capital,
                     "roi": rate * share / capital if capital > 0 else 0.0,
                     "end": g.get("endDate")})
    rows.sort(key=lambda r: -r["roi"])

    # the flow that would fill us: trades in the last day on the best candidates
    top = rows[:args.top]
    since = time.time() - 86_400.0

    def flow(r):
        with httpx.Client(timeout=30) as c:
            t = c.get(_TRADES, params={"market": r["cid"], "limit": 500}).json()
        day = [x for x in t if x["timestamp"] >= since]
        return len(day), sum(float(x["size"]) for x in day)

    def competitiveness(r):
        with httpx.Client(timeout=30) as c:
            d = c.get(f"{_CLOB}/rewards/markets/{r['cid']}").json().get("data") or [{}]
        return float(d[0].get("market_competitiveness") or 0.0)

    with ThreadPoolExecutor(3) as ex:
        for r, (n, sh) in zip(top, ex.map(flow, top)):
            r["trades_day"], r["shares_day"] = n, sh
        for r, comp in zip(top, ex.map(competitiveness, top)):
            # Polymarket's own score of the makers already there, in place of ours
            r["comp"] = comp
            r["pm_share"] = r["ours"] / (comp + r["ours"]) if comp + r["ours"] > 0 else 0.0
            r["pm_day"] = r["rate"] * r["pm_share"]

    print(f"{len(rows)} of them priced from their books.\n")
    print(f"{'ours/day':>8} {'book Q':>7} {'PM comp':>7} {'PM/day':>7} {'capital':>7} {'rate':>5} "
          f"{'mid':>5} {'size':>5} {'trades/day':>10}  market")
    for r in top:
        print(f"{r['per_day']:8.2f} {r['q_there']:7.1f} {r.get('comp', 0):7.1f} "
              f"{r.get('pm_day', 0):7.2f} {r['capital']:7.1f} {r['rate']:5.0f} {r['mid']:5.2f} "
              f"{r['size']:5.0f} {r.get('trades_day', 0):5d} ({r.get('shares_day', 0):6.0f})  "
              f"{r['q'][:46]}")
    budget = args.capital
    picked, used, earn = [], 0.0, 0.0
    for r in rows:
        if used + r["capital"] <= budget:
            picked.append(r)
            used += r["capital"]
            earn += r["per_day"]
    print(f"\nwith {budget:g} USDC spread over the best-paying quotes: {len(picked)} market(s), "
          f"{used:.1f} USDC held, {earn:.2f} USDC a day gross, before the cost of being filled")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-rate", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--capital", type=float, default=75.0)
    sys.exit(main(ap.parse_args()))
