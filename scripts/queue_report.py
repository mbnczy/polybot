#!/usr/bin/env python3
"""
scripts/queue_report.py
───────────────────────
What fraction of the live universe can a maker quote actually reach?

Whether a post-only order fills is decided by the shares already resting at its
price, and until 2026-09-10 the bot never read that number. Measured across 257
live two-sided books that day:

    one-tick spread                  79% of active markets
    median queue at the touch    10,558 shares against a ten-share order
    p90 queue                   171,338 shares

An order behind 10,558 shares does not fill slowly. It does not fill.

This script prices that for the current book: for every liquid market it
computes the quote `lead_book_bid` would place, asks whether that quote opens
its own price level or joins a queue, and reports what share of the universe is
reachable at a given order size.

    python scripts/queue_report.py --size 10 --markets 200

Read-only: fetches public book data, places nothing, touches no live state.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from strategy.arbitrage import (  # noqa: E402
    lead_book_bid,
    maker_quote_is_reachable,
    quote_opens_new_level,
    spread_fraction,
)

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com"
_GAMMA_PAGE = 100      # Gamma caps `limit` at 100 however much you ask for


def fetch_books(limit: int) -> list[dict]:
    """Live two-sided books, busiest first."""
    out: list[dict] = []
    with httpx.Client(timeout=30.0) as c:
        markets: list[tuple] = []
        offset = 0
        while len(markets) < limit and offset < 2_000:
            r = c.get(GAMMA, params={
                "closed": "false", "active": "true", "limit": _GAMMA_PAGE,
                "offset": offset, "order": "volume24hr", "ascending": "false",
            })
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            for m in batch:
                try:
                    toks = json.loads(m.get("clobTokenIds") or "[]")
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if len(toks) == 2:
                    markets.append((
                        str(m.get("question") or "")[:42], toks[0],
                        float(m.get("volume24hr") or 0.0),
                        float(m.get("orderPriceMinTickSize") or 0.01),
                    ))
            offset += len(batch)

        for question, token, vol, tick in markets[:limit]:
            try:
                b = c.get(f"{CLOB}/book", params={"token_id": token}).json()
            except (httpx.HTTPError, ValueError):
                continue
            bids, asks = b.get("bids") or [], b.get("asks") or []
            if not bids or not asks:
                continue
            # The exchange sends bids ascending, so bids[0] is the WORST bid.
            best_bid = max(float(x["price"]) for x in bids)
            best_ask = min(float(x["price"]) for x in asks)
            queue = sum(float(x["size"]) for x in bids
                        if abs(float(x["price"]) - best_bid) < 1e-12)
            out.append(dict(q=question, vol=vol, tick=tick,
                            bid=best_bid, ask=best_ask, queue=queue))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=10.0,
                    help="order size the reachability test is measured against")
    ap.add_argument("--markets", type=int, default=200)
    a = ap.parse_args()

    print(f"\n  fetching up to {a.markets} live books…")
    books = fetch_books(a.markets)
    if not books:
        print("  no two-sided book returned")
        return 1
    print(f"    got {len(books)} with quotes on both sides\n")

    reach = collections.Counter()
    spreads = collections.Counter()
    for b in books:
        quote = lead_book_bid(b["ask"], b["bid"], b["tick"])
        b["quote"] = quote
        b["leads"] = quote_opens_new_level(quote, b["bid"], b["tick"])
        b["ok"] = maker_quote_is_reachable(
            quote, b["bid"], b["tick"], b["queue"], a.size)
        spreads[round((b["ask"] - b["bid"]) / b["tick"])] += 1
        reach["own price level" if b["leads"] else
              ("thin queue" if b["ok"] else "behind a wall")] += 1

    n = len(books)
    print(f"  REACHABLE AT {a.size:g} SHARES")
    for k in ("own price level", "thin queue", "behind a wall"):
        v = reach[k]
        bar = "█" * round(v / n * 40)
        print(f"    {k:<18}{v:>4} ({v / n * 100:>3.0f}%)  {bar}")
    ok = reach["own price level"] + reach["thin queue"]
    print(f"    {'':<18}{'':>4}  {'':>5} {ok}/{n} = {ok / n * 100:.0f}% quotable\n")

    print("  SPREAD (ticks)")
    for k in sorted(spreads)[:6]:
        print(f"    {k:>3} tick{'s' if k != 1 else ' '}: {spreads[k]:>4} "
              f"({spreads[k] / n * 100:>3.0f}%)")

    one = sorted(b["queue"] for b in books
                 if round((b["ask"] - b["bid"]) / b["tick"]) == 1)
    if one:
        print("\n  QUEUE AT THE TOUCH — one-tick books")
        for pct in (10, 25, 50, 75, 90):
            v = one[min(int(len(one) * pct / 100), len(one) - 1)]
            print(f"    p{pct:<3}{v:>12,.0f} shares  ({v / a.size:>9,.0f}x our order)")
        print(f"    median {st.median(one):,.0f}")

    # Ticks do not compare across the universe. Half of live markets trade on a
    # 0.001 grid where one tick is 0.1%, the other half on 0.01 where it is ten
    # times that — so ten ticks on the fine grid is a THINNER spread than one
    # tick on the coarse. Ranking "wide spread" targets in ticks picks the wrong
    # markets.
    print("\n  SPREAD IN PRICE, not ticks")
    fr = sorted(
        (spread_fraction(b["ask"], b["bid"]) or 0.0) for b in books)
    for pct in (50, 75, 90, 99):
        v = fr[min(int(len(fr) * pct / 100), len(fr) - 1)]
        print(f"    p{pct:<3}{v * 100:>8.2f}%")

    print("\n  BEST MAKER TARGETS — reachable, busiest first")
    tgt = sorted((b for b in books if b["ok"]), key=lambda b: -b["vol"])[:10]
    if not tgt:
        print("    none — every book in this sample is behind a wall")
    for b in tgt:
        kind = "new level" if b["leads"] else "thin queue"
        frac = (spread_fraction(b["ask"], b["bid"]) or 0.0) * 100
        print(f"    {b['q']:<44} quote {b['quote']:.3f}  "
              f"{round((b['ask'] - b['bid']) / b['tick']):>2}t "
              f"({frac:>4.1f}%)  queue {b['queue']:>8,.0f}  {kind}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
