#!/usr/bin/env python3
"""
scripts/fill_report.py
──────────────────────
Summarise telemetry/fill_log.jsonl: which maker legs actually fill.

The bot has never completed a bundle, and the working theory is queue position
— on a one-tick spread `ask - tick` IS the best bid, so a post-only quote joins
the back of a queue rather than leading it. This turns that theory into a
measurement.

    python scripts/fill_report.py [--path fill_log.jsonl] [--days 7]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time


def load(path: str, since: float) -> list[dict]:
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ts", 0) >= since:
                    rows.append(r)
    except FileNotFoundError:
        print(f"  no fill log at {path} yet — nothing has been recorded")
        sys.exit(0)
    return rows


def table(title: str, buckets: dict) -> None:
    if not buckets:
        return
    print(f"\n  {title}")
    print(f"    {'bucket':<22}{'legs':>7}{'filled':>8}{'partial':>9}{'fill rate':>11}")
    for key in sorted(buckets):
        c = buckets[key]
        n = sum(c.values())
        if not n:
            continue
        got = c.get("filled", 0)
        part = c.get("partial", 0)
        print(f"    {str(key):<22}{n:>7}{got:>8}{part:>9}"
              f"{(got + part) / n * 100:>10.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="fill_log.jsonl")
    ap.add_argument("--days", type=float, default=7.0)
    a = ap.parse_args()

    rows = load(a.path, time.time() - a.days * 86_400)
    if not rows:
        print("  nothing recorded in that window")
        return 0

    n = len(rows)
    got = sum(1 for r in rows if r["outcome"] == "filled")
    part = sum(1 for r in rows if r["outcome"] == "partial")
    print(f"  {n} maker leg(s) over {a.days:.0f} day(s)")
    print(f"    fully filled  {got:>6}  ({got / n * 100:.1f}%)")
    print(f"    partial       {part:>6}  ({part / n * 100:.1f}%)")
    print(f"    never filled  {n - got - part:>6}  "
          f"({(n - got - part) / n * 100:.1f}%)")

    # The question the file exists for: does leading the book matter, and how
    # much? If the two rows differ sharply, queue position is the constraint and
    # bidding to lead is worth paying for. If they do not, it is something else.
    by_lead = collections.defaultdict(collections.Counter)
    for r in rows:
        lead = r.get("leads")
        key = "led the book" if lead else ("joined queue" if lead is False else "unknown")
        by_lead[key][r["outcome"]] += 1
    table("by whether our quote led the book", by_lead)

    by_spread = collections.defaultdict(collections.Counter)
    for r in rows:
        st = r.get("spread_ticks")
        key = "unknown" if st is None else f"{int(st)} tick" if st < 4 else "4+ ticks"
        by_spread[key][r["outcome"]] += 1
    table("by spread at placement", by_spread)

    by_price = collections.defaultdict(collections.Counter)
    for r in rows:
        p = r.get("price", 0)
        band = ("<0.10" if p < 0.10 else "0.10-0.30" if p < 0.30 else
                "0.30-0.70" if p < 0.70 else "0.70-0.90" if p < 0.90 else ">=0.90")
        by_price[band][r["outcome"]] += 1
    table("by price band", by_price)

    by_path = collections.defaultdict(collections.Counter)
    for r in rows:
        by_path[r.get("path", "?")][r["outcome"]] += 1
    table("by strategy", by_path)

    rest = sorted(r.get("rested_s", 0) for r in rows if r["outcome"] in ("filled", "partial"))
    if rest:
        print(f"\n  time on the book before filling")
        for q, lbl in ((0.5, "median"), (0.9, "p90"), (1.0, "max")):
            print(f"    {lbl:>7}: {rest[min(int(q * len(rest)), len(rest) - 1)]:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
