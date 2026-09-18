#!/usr/bin/env python3
"""
paper_fills.py — would the guard's rests have filled, and what would they have made?

The guard, disabled, writes every rest it would have made to
<positions>_paper_rests.jsonl: both markets, our two bids, the size. This replays
each one against the public trade feed (data-api /trades) and, once the markets
resolve, against the result.

A leg counts as filled by trades at or through our price after the rest went up:

    NO on narrow at p   ←  narrow traded at a YES price ≥ 1 − p  (someone bought
                           YES at our implied offer, or sold NO into our bid)
    YES on broad at q   ←  broad traded at a YES price ≤ q

Our bid stood a tick above the best one, so any such trade would have met it
first. Trade sizes count toward our size; less is a partial fill.

Each horizon is a policy: rest for that long, then cancel. The guard re-rests a
pair every 15 minutes, so for each pair a rest is only taken once the previous
one has expired — or, once both legs filled, not again until resolution — and
nothing is counted twice.

P&L, per filled share, at resolution: NO on narrow pays 1 − the narrow's YES
price, YES on broad pays the broad's. A single filled leg is held to resolution:
the guard would try to complete it or sell it back, and holding is the plain
version of that.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from strategy.resolution_audit import resolved_prices  # noqa: E402

_TRADES = "https://data-api.polymarket.com/trades"
_GAMMA = "https://gamma-api.polymarket.com/markets"
HORIZONS = (("3 min", 180.0), ("30 min", 1800.0), ("2 h", 7200.0), ("to kick-off", None))


def load_rests(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return sorted(out, key=lambda r: r["ts"])


def trades_since(cid: str, since: float) -> list[dict]:
    """Every trade on the market since `since`, oldest first."""
    out: list[dict] = []
    with httpx.Client(timeout=30) as c:
        offset = 0
        while offset < 10_000:
            r = c.get(_TRADES, params={"market": cid, "limit": 500, "offset": offset})
            if r.status_code != 200:
                break
            page = r.json()
            out += page
            if len(page) < 500 or (page and page[-1]["timestamp"] < since):
                break
            offset += 500
    return sorted((t for t in out if t["timestamp"] >= since), key=lambda t: t["timestamp"])


def resolutions(cids: list[str]) -> dict[str, "tuple[float, float] | None"]:
    out: dict[str, "tuple[float, float] | None"] = {}
    with httpx.Client(timeout=30) as c:
        for i in range(0, len(cids), 50):
            part = cids[i:i + 50]
            r = c.get(_GAMMA, params=[("condition_ids", x) for x in part] + [("closed", "true")])
            for m in (r.json() if r.status_code == 200 else []):
                out[str(m.get("conditionId"))] = resolved_prices(m)
    return out


def yes_price(t: dict, yes_token: str) -> float:
    p = float(t["price"])
    return p if str(t["asset"]) == yes_token else 1.0 - p


def filled(trades: list[dict], start: float, end: float, size: float, hits) -> float:
    got = 0.0
    for t in trades:
        if t["timestamp"] <= start:
            continue
        if t["timestamp"] > end:
            break
        if hits(t):
            got += float(t["size"])
            if got >= size:
                return size
    return got


def main(args) -> int:
    rests = load_rests(Path(args.rests))
    if not rests:
        print("no paper rests yet")
        return 0
    first = defaultdict(lambda: float("inf"))
    for r in rests:
        first[r["narrow"]] = min(first[r["narrow"]], r["ts"])
        first[r["broad"]] = min(first[r["broad"]], r["ts"])
    cids = list(first)
    with ThreadPoolExecutor(6) as ex:
        trades = dict(zip(cids, ex.map(lambda c: trades_since(c, first[c]), cids)))
    resolved = resolutions(cids)
    now = time.time()
    pairs = {(r["narrow"], r["broad"]) for r in rests}
    span = (rests[-1]["ts"] - rests[0]["ts"]) / 3600.0
    print(f"{len(rests)} paper rest(s) on {len(pairs)} pair(s) over {span:.1f} h; "
          f"{sum(v is not None for v in resolved.values())}/{len(cids)} market(s) resolved\n")

    for label, horizon in HORIZONS:
        counts = defaultdict(int)
        pnl_done, pnl_open_legs = 0.0, 0
        examples = []
        busy: dict[tuple, float] = {}
        for r in rests:
            key = (r["narrow"], r["broad"])
            if r["ts"] < busy.get(key, 0.0):
                continue
            kick = r.get("resolves_ts") or now
            end = min(kick, r["ts"] + horizon) if horizon else kick
            end = min(end, now)
            p, q, size = r["no_price"], r["yes_price"], r["shares"]
            n_yes, b_yes = r["narrow_yes_token"], r["broad_yes_token"]
            fn = filled(trades[r["narrow"]], r["ts"], end, size,
                        lambda t: yes_price(t, n_yes) >= 1.0 - p - 1e-9)
            fb = filled(trades[r["broad"]], r["ts"], end, size,
                        lambda t: yes_price(t, b_yes) <= q + 1e-9)
            counts["rests"] += 1
            both = fn >= size - 1e-9 and fb >= size - 1e-9
            if both:
                counts["both legs"] += 1
                busy[key] = kick
            elif fn > 0 or fb > 0:
                counts["narrow only" if fn > 0 and fb == 0 else
                       "broad only" if fb > 0 and fn == 0 else "both, partly"] += 1
                busy[key] = end
            else:
                counts["nothing"] += 1
                busy[key] = end
            if fn == 0 and fb == 0:
                continue
            rn, rb = resolved.get(r["narrow"]), resolved.get(r["broad"])
            if rn is None or rb is None:
                pnl_open_legs += 1
                continue
            pnl = fn * ((1.0 - rn[0]) - p) + fb * (rb[0] - q)
            pnl_done += pnl
            examples.append((pnl, both, fn, fb, r))
        n = counts["rests"] or 1
        print(f"── rest {label}: {counts['rests']} rest(s) — both legs {counts['both legs']} "
              f"({counts['both legs'] / n:.0%}), narrow only {counts['narrow only']}, "
              f"broad only {counts['broad only']}, partly {counts['both, partly']}, "
              f"nothing {counts['nothing']} ({counts['nothing'] / n:.0%})")
        print(f"   realised {pnl_done:+.2f} USDC on the resolved fills; "
              f"{pnl_open_legs} filled rest(s) still waiting for their markets")
        for pnl, both, fn, fb, r in sorted(examples, key=lambda e: e[0])[:args.show]:
            print(f"   {pnl:+.2f}  {'both' if both else 'one '} NO {fn:.1f}@{r['no_price']:.3f} "
                  f"YES {fb:.1f}@{r['yes_price']:.3f}  {r['narrow_title'][:50]}")
        print()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rests", default="/home/ubuntu/polybot/cross_positions_live_paper_rests.jsonl")
    ap.add_argument("--show", type=int, default=5, help="worst resolved fills to list")
    sys.exit(main(ap.parse_args()))
