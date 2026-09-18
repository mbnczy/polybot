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
version of that. A pair with both legs is locked the moment it is complete, so
its guaranteed minimum is reported before anything resolves.

One leg. For rests whose one-leg edge cleared +2% the replay also tries resting
NO on narrow alone: when it fills, YES on broad is bought at its ask at that
moment — read off the WebSocket recording (record_ws_books.py), or, where the
recording does not reach, the ask when the rest went up (counted apart as
approximate). If completing no longer pays by then, the leg is held.

Which rests fill. Every rest on its own — not the policies' sequence — split by
how long before kick-off it went up, and by how many trades its narrow market
had seen in the day before: the answers the resting window and a ranking by
activity have to be built on.
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

import bisect  # noqa: E402

import httpx  # noqa: E402

from scripts.analyze_ws_recording import _lines, load_tokens  # noqa: E402
from strategy.resolution_audit import resolved_prices  # noqa: E402

_TRADES = "https://data-api.polymarket.com/trades"
_GAMMA = "https://gamma-api.polymarket.com/markets"
HORIZONS = (("3 min", 180.0), ("30 min", 1800.0), ("2 h", 7200.0), ("to kick-off", None))


def load_rests(path: Path) -> list[dict]:
    """The rests, oldest first — bar those in a one-sided book, which the guard
    stopped making on 2026-09-18 (a bid with no offer on a leg)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("no_ask", True) is None or r.get("yes_ask", True) is None:
            continue
        out.append(r)
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
    return fill_at(trades, start, end, size, hits)[0]


def fill_at(trades: list[dict], start: float, end: float, size: float,
            hits) -> "tuple[float, float | None]":
    """(shares filled, when the last of them filled) within (start, end]."""
    got, when = 0.0, None
    for t in trades:
        if t["timestamp"] <= start:
            continue
        if t["timestamp"] > end:
            break
        if hits(t):
            got += float(t["size"])
            when = float(t["timestamp"])
            if got >= size:
                return size, when
    return got, when


def fee(p: float, rate: "float | None") -> float:
    return (0.05 if rate is None else rate) * p * (1.0 - p)


def recorded_asks(rec_dir: Path, yes_tokens: set) -> dict:
    """token → [(time, best ask or None)] from the recording, crossed books dropped."""
    path = rec_dir / "tokens.jsonl"
    if not path.exists():
        return {}
    idx = {i: row["token"] for i, row in load_tokens(path).items() if row["token"] in yes_tokens}
    out: dict[str, list] = defaultdict(list)
    for f in sorted(rec_dir.glob("*.csv.gz")):
        for line in _lines(f):
            body = line[1:] if line[:1] in ("S", "R") else line
            parts = body.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            tok = idx.get(int(parts[1]))
            if tok is None:
                continue
            bid = float(parts[2]) if parts[2] else None
            ask = float(parts[3]) if parts[3] else None
            if bid is not None and ask is not None and bid >= ask:
                continue
            out[tok].append((int(parts[0]) / 1000.0, ask))
    for series in out.values():
        series.sort(key=lambda x: x[0])
    return out


def ask_at(series: "list | None", t: float) -> "tuple[float | None, bool]":
    """The recorded ask at t, and whether the recording covered t at all."""
    if not series:
        return None, False
    i = bisect.bisect_right([x[0] for x in series], t) - 1
    if i < 0:
        return None, False
    return series[i][1], True


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
        # a day earlier than the first rest: the activity before each rest is a column
        trades = dict(zip(cids, ex.map(lambda c: trades_since(c, first[c] - 86_400.0), cids)))
    resolved = resolutions(cids)
    now = time.time()
    pairs = {(r["narrow"], r["broad"]) for r in rests}
    span = (rests[-1]["ts"] - rests[0]["ts"]) / 3600.0
    print(f"{len(rests)} paper rest(s) on {len(pairs)} pair(s) over {span:.1f} h; "
          f"{sum(v is not None for v in resolved.values())}/{len(cids)} market(s) resolved\n")

    for label, horizon in HORIZONS:
        counts = defaultdict(int)
        pnl_done, pnl_open_legs, locked2 = 0.0, 0, 0.0
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
                locked2 += size * (1.0 - (p + q))
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
        print(f"   realised {pnl_done:+.2f} USDC on the resolved fills; locked {locked2:+.2f} "
              f"on complete pairs; {pnl_open_legs} filled rest(s) still waiting for their markets")
        for pnl, both, fn, fb, r in sorted(examples, key=lambda e: e[0])[:args.show]:
            print(f"   {pnl:+.2f}  {'both' if both else 'one '} NO {fn:.1f}@{r['no_price']:.3f} "
                  f"YES {fb:.1f}@{r['yes_price']:.3f}  {r['narrow_title'][:50]}")
        print()

    # ── one leg: rest NO on narrow alone, buy YES on broad at its ask on the fill ──
    eligible = [r for r in rests if (r.get("one_leg_no_edge") or -1.0) >= args.one_leg_min]
    series = recorded_asks(Path(args.recording), {r["broad_yes_token"] for r in eligible})
    print(f"── one leg (rest NO on narrow, take YES on broad): {len(eligible)} rest(s) "
          f"with a one-leg edge ≥ {args.one_leg_min:+.2f}")
    for label, horizon in HORIZONS:
        c = defaultdict(int)
        locked, realised, waiting = 0.0, 0.0, 0
        busy: dict[tuple, float] = {}
        for r in eligible:
            key = (r["narrow"], r["broad"])
            if r["ts"] < busy.get(key, 0.0):
                continue
            kick = r.get("resolves_ts") or now
            end = min(min(kick, r["ts"] + horizon) if horizon else kick, now)
            p, n_yes = r["no_price"], r["narrow_yes_token"]
            fn, when = fill_at(trades[r["narrow"]], r["ts"], end, r["shares"],
                               lambda t: yes_price(t, n_yes) >= 1.0 - p - 1e-9)
            c["rests"] += 1
            if fn <= 0:
                c["nothing"] += 1
                busy[key] = end
                continue
            busy[key] = kick
            ask, covered = ask_at(series.get(r["broad_yes_token"]), when)
            if not covered:
                ask = (r.get("yes_ask") or [None])[0]
                c["approx"] += 1
            cost = None if ask is None else p + ask + fee(ask, r.get("broad_fee_rate"))
            complete = cost is not None and cost <= 1.0 and fn >= 5.0
            c["completed" if complete else "held alone"] += 1
            if complete:
                locked += fn * (1.0 - cost)
            rn, rb = resolved.get(r["narrow"]), resolved.get(r["broad"])
            if rn is None or rb is None:
                waiting += 1
                continue
            realised += (fn * ((1.0 - rn[0]) + rb[0] - cost) if complete
                         else fn * ((1.0 - rn[0]) - p))
        n = c["rests"] or 1
        print(f"   {label:12} {c['rests']:4d} rest(s): filled {c['rests'] - c['nothing']} "
              f"({(c['rests'] - c['nothing']) / n:.0%}) — completed {c['completed']}, held alone "
              f"{c['held alone']} ({c['approx']} priced off the rest-time ask) · locked "
              f"{locked:+.2f} · realised {realised:+.2f} · {waiting} waiting")
    print()

    # ── which rests fill: every rest on its own ──
    def narrow_fill(r, horizon):
        kick = r.get("resolves_ts") or now
        end = min(min(kick, r["ts"] + horizon) if horizon else kick, now)
        p, n_yes = r["no_price"], r["narrow_yes_token"]
        return fill_at(trades[r["narrow"]], r["ts"], end, r["shares"],
                       lambda t: yes_price(t, n_yes) >= 1.0 - p - 1e-9)[0] > 0

    def bucket_ko(r):
        h = ((r.get("resolves_ts") or now) - r["ts"]) / 3600.0
        return "<1h" if h < 1 else "1-3h" if h < 3 else "3-6h" if h < 6 else \
            "6-24h" if h < 24 else ">24h"

    def bucket_act(r):
        n = sum(1 for t in trades[r["narrow"]] if r["ts"] - 86_400.0 <= t["timestamp"] < r["ts"])
        return "0" if n == 0 else "1-9" if n < 10 else "10+"

    for title, fn_b, order in (("time to kick-off", bucket_ko, ("<1h", "1-3h", "3-6h", "6-24h", ">24h")),
                               ("narrow's trades in the day before", bucket_act, ("0", "1-9", "10+"))):
        groups = defaultdict(list)
        for r in rests:
            groups[fn_b(r)].append(r)
        print(f"── NO on narrow filled, by {title}")
        for b in order:
            g = groups.get(b, [])
            if not g:
                continue
            f30 = sum(narrow_fill(r, 1800.0) for r in g)
            fko = sum(narrow_fill(r, None) for r in g)
            print(f"   {b:6} {len(g):5d} rest(s): within 30 min {f30 / len(g):6.1%} · "
                  f"by kick-off {fko / len(g):6.1%}")
        print()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rests", default="/home/ubuntu/polybot/cross_positions_live_paper_rests.jsonl")
    ap.add_argument("--show", type=int, default=5, help="worst resolved fills to list")
    ap.add_argument("--recording", default="/home/ubuntu/polybot-dev/recordings",
                    help="record_ws_books.py output, for the ask at the moment of a fill")
    ap.add_argument("--one-leg-min", type=float, default=0.02)
    sys.exit(main(ap.parse_args()))
