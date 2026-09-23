#!/usr/bin/env python3
"""
coverage_watch.py — would watching more markets find more arbitrage?

The reader builds pairs only from quotable markets (both sides within 0.10) in
the one-week window. The taker path has no spread gate, so a certain
implication in a wider book is tradeable too — and on 2026-09-23 there were
7.4 times as many of those (43,728 against 5,879). One snapshot found no
violation on either side, but violations last seconds to minutes, so one
snapshot proves little. This repeats the scan every --every seconds for
--hours and writes one JSON line per scan.

Each scan: every market resolving within --days, the structural links
(strategy/structural_implications.py: match over/unders, spreads, price
ladders) over ALL of them, and the taker edge from the Gamma snapshot —
narrow YES bid − broad YES ask − both taker fees. Any pair at or above
--min-edge has its two books read straight away, and the edge and depth that
could actually be bought are recorded beside the snapshot's.

READ ONLY: public Gamma and CLOB reads, nothing is sent.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import httpx  # noqa: E402

import demo_cross_market as d  # noqa: E402
import strategy.structural_implications as si  # noqa: E402

_CLOB = "https://clob.polymarket.com"


def _fee(p: float, m: dict) -> float:
    r = d._fee_rate(m)
    return (0.05 if r is None else r) * p * (1.0 - p)


def _books(tokens: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with httpx.Client(timeout=20) as c:
        for i in range(0, len(tokens), 100):
            try:
                r = c.post(f"{_CLOB}/books", json=[{"token_id": t} for t in tokens[i:i + 100]])
                for b in r.json() if r.status_code == 200 else []:
                    out[str(b.get("asset_id"))] = b
            except (httpx.HTTPError, ValueError):
                pass
    return out


def _executable(n: dict, b: dict, books: dict) -> "dict | None":
    """Edge and depth at the live books: sell into the narrow YES bid (= buy its
    NO), buy the broad YES ask."""
    nt, bt = d._tokens_of(n), d._tokens_of(b)
    if not nt or not bt:
        return None
    nb, bb = books.get(str(nt[0])), books.get(str(bt[0]))
    if not nb or not bb or not nb.get("bids") or not bb.get("asks"):
        return None
    bid = max(nb["bids"], key=lambda x: float(x["price"]))
    ask = min(bb["asks"], key=lambda x: float(x["price"]))
    pb, pa = float(bid["price"]), float(ask["price"])
    return {"edge": round(pb - pa - _fee(1 - pb, n) - _fee(pa, b), 4),
            "bid": pb, "ask": pa, "depth": min(float(bid["size"]), float(ask["size"]))}


def scan(args) -> dict:
    t0 = time.time()
    markets = d.fetch_window(args.days, 1.0, 4, "Up or Down")
    by_id = {str(m.get("conditionId")): m for m in markets}
    quot = {str(m.get("conditionId")) for m in d.quotable(markets, 0.10)}
    links = d.drop_cross_fixture(si.links(markets), markets)
    row = {"ts": round(t0), "markets": len(markets), "quotable": len(quot),
           "links_seen": 0, "links_wide": 0, "pos_seen": 0, "pos_wide": 0, "hits": []}
    flagged = []
    for r in links:
        n, b = by_id.get(r.narrow), by_id.get(r.broad)
        if not n or not b:
            continue
        seen = r.narrow in quot and r.broad in quot
        row["links_seen" if seen else "links_wide"] += 1
        nb, ba = d._num(n.get("bestBid")), d._num(b.get("bestAsk"))
        if nb is None or ba is None:
            continue
        e = nb - ba - _fee(1 - nb, n) - _fee(ba, b)
        if e > 0:
            row["pos_seen" if seen else "pos_wide"] += 1
        if e >= args.min_edge:
            flagged.append((e, seen, n, b))
    if flagged:
        toks = []
        for _, _, n, b in flagged:
            for m in (n, b):
                t = d._tokens_of(m)
                if t:
                    toks.append(str(t[0]))
        books = _books(sorted(set(toks)))
        for e, seen, n, b in sorted(flagged, key=lambda x: -x[0])[:args.max_hits]:
            row["hits"].append({"snapshot_edge": round(e, 4), "seen_by_reader": seen,
                                "live": _executable(n, b, books),
                                "narrow": str(n.get("question"))[:90],
                                "broad": str(b.get("question"))[:90]})
    row["secs"] = round(time.time() - t0, 1)
    return row


def main(args) -> int:
    out = Path(args.out)
    end = time.time() + args.hours * 3600.0
    while True:
        start = time.time()
        try:
            row = scan(args)
        except Exception as exc:                        # noqa: BLE001 — keep watching
            row = {"ts": round(start), "error": repr(exc)[:300]}
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "hits"}), flush=True)
        gc.collect()
        if time.time() >= end:
            return 0
        time.sleep(max(0.0, args.every - (time.time() - start)))


def summary(path: str) -> int:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ok = [r for r in rows if "error" not in r]
    print(f"{len(ok)} scans ({len(rows) - len(ok)} failed) over "
          f"{(ok[-1]['ts'] - ok[0]['ts']) / 3600:.1f} h" if ok else "no scans")
    for side in ("seen", "wide"):
        links = sum(r[f"links_{side}"] for r in ok) / max(1, len(ok))
        pos = sum(r[f"pos_{side}"] for r in ok)
        hits = [h for r in ok for h in r["hits"] if h["seen_by_reader"] == (side == "seen")]
        real = [h for h in hits if h["live"] and h["live"]["edge"] >= 0.02]
        print(f"  {side:4}: {links:8.0f} links a scan · {pos} snapshot(s) with a taker edge > 0 · "
              f"{len(hits)} ≥ min edge · {len(real)} still ≥ +0.02 at the live books")
        for h in sorted(real, key=lambda h: -h["live"]["edge"])[:10]:
            lv = h["live"]
            print(f"     {lv['edge']:+.3f} depth {lv['depth']:.0f}  {h['narrow'][:60]} ⊆ {h['broad'][:50]}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--every", type=float, default=600.0)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--max-hits", type=int, default=40)
    ap.add_argument("--out", default=str(REPO / "coverage_watch.jsonl"))
    ap.add_argument("--summary", action="store_true", help="summarise --out and exit")
    a = ap.parse_args()
    sys.exit(summary(a.out) if a.summary else main(a))
