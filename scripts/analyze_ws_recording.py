#!/usr/bin/env python3
"""
analyze_ws_recording.py — replay a book recording against the structural pairs.

Question it answers: when a match's over/unders fall out of order, how often,
how wide, for how long, how deep — and is it in play or before kick-off?

For every link "X over k ⊆ Y over j" (strategy/structural_implications.py) the
pair is bought as NO on narrow + YES on broad. From the YES books recorded by
record_ws_books.py:

    crossing:  (1 − narrow YES bid) + broad YES ask + taker fees on both
    resting:   (1 − narrow YES ask + tick) + (broad YES bid + tick), no fee,
               only while both books are within --max-spread, as the guard does

A crossed book (bid at or above ask) is dropped as stale: the exchange cannot
hold one. --rest-only replays just the periodic REST reads, which cannot go
stale, at their coarser cadence.

An episode is a stretch during which the edge stays at or above --min-edge; it
ends at the first update that takes it below. Depth is the smaller of the two
touches the crossing would take (narrow YES bid size, broad YES ask size).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import strategy.structural_implications as si  # noqa: E402


def _kick_ts(raw) -> "float | None":
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load_tokens(path: Path) -> dict[int, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        out[row["i"]] = row
    return out


def _lines(path: Path):
    """The file's lines; the hour being recorded ends mid-block, which is not an error."""
    try:
        with gzip.open(path, "rt", encoding="ascii") as fh:
            yield from fh
    except EOFError:
        return


def fee(p: float, rate: float) -> float:
    return rate * p * (1.0 - p)


class Episode:
    __slots__ = ("start", "end", "best", "depth", "inplay")

    def __init__(self, t, edge, depth, inplay):
        self.start = self.end = t
        self.best, self.depth, self.inplay = edge, depth, inplay


def main(args) -> int:
    rec = Path(args.dir)
    tokens = load_tokens(rec / "tokens.jsonl")
    by_cid = {}
    for i, row in tokens.items():
        by_cid.setdefault(row["cid"], (i, row))
    markets = [{"conditionId": cid, "question": row["q"], "outcomes": row["outcomes"]}
               for cid, (i, row) in by_cid.items()]
    links = si.links(markets)
    idx = {cid: i for cid, (i, _) in by_cid.items()}
    kick = {cid: _kick_ts(row["kick"]) for cid, (_, row) in by_cid.items()}
    touching: dict[int, list] = defaultdict(list)
    for r in links:
        touching[idx[r.narrow]].append(r)
        touching[idx[r.broad]].append(r)
    print(f"{len(tokens)} recorded markets, {len(links)} structural links among them")

    top: dict[int, tuple] = {}
    open_ep: dict[tuple, Episode] = {}
    done: list[tuple] = []
    kinds = ("cross", "rest")
    files = sorted(rec.glob("*.csv.gz"))
    lines = 0
    for f in files:
        for line in _lines(f):
            lines += 1
            mark = line[0] if line[0] in "SR" else ""
            if args.rest_only and mark != "R":
                continue
            parts = (line[1:] if mark else line).rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            t = int(parts[0]) / 1000.0
            i = int(parts[1])
            bid = float(parts[2]) if parts[2] else None
            ask = float(parts[3]) if parts[3] else None
            top[i] = (bid, ask, float(parts[4] or 0), float(parts[5] or 0))
            if bid is not None and ask is not None and bid >= ask:
                top.pop(i)           # a crossed book cannot exist: the state went stale
            for r in touching.get(i, ()):
                n, b = top.get(idx[r.narrow]), top.get(idx[r.broad])
                if not n or not b:
                    continue
                k = kick.get(r.narrow)
                inplay = k is not None and t >= k
                edges = {}
                if n[0] is not None and b[1] is not None:
                    no_ask = 1.0 - n[0]
                    edges["cross"] = (1.0 - no_ask - b[1] - fee(no_ask, args.rate)
                                      - fee(b[1], args.rate), min(n[2], b[3]))
                # The guard's own gate: a bid a tick above an almost empty book
                # is not an offer anyone takes (CROSS_MAKER_MAX_SPREAD).
                if (None not in (n[0], n[1], b[0], b[1])
                        and n[1] - n[0] <= args.max_spread + 1e-9
                        and b[1] - b[0] <= args.max_spread + 1e-9):
                    edges["rest"] = (1.0 - (1.0 - n[1] + args.tick) - (b[0] + args.tick),
                                     0.0)
                for kind in kinds:
                    key = (r.narrow, r.broad, kind)
                    e = edges.get(kind)
                    ep = open_ep.get(key)
                    if e is not None and e[0] >= args.min_edge:
                        if ep is None:
                            open_ep[key] = Episode(t, e[0], e[1], inplay)
                        else:
                            ep.end = t
                            if e[0] > ep.best:
                                ep.best, ep.depth = e[0], e[1]
                            ep.inplay = ep.inplay or inplay
                    elif ep is not None:
                        ep.end = t
                        done.append((key, ep))
                        del open_ep[key]
    for key, ep in open_ep.items():
        done.append((key, ep))
    print(f"{lines} lines replayed from {len(files)} file(s)\n")

    for kind in kinds:
        eps = [(k, e) for k, e in done if k[2] == kind]
        live = [e for _, e in eps if e.inplay]
        pre = [e for _, e in eps if not e.inplay]
        print(f"── {kind}: {len(eps)} episode(s) at edge ≥ {args.min_edge:+.3f} "
              f"({len(live)} in play, {len(pre)} before kick-off)")
        for label, group in (("in play", live), ("before", pre)):
            if not group:
                continue
            durs = sorted(e.end - e.start for e in group)
            print(f"   {label:8} duration median {durs[len(durs)//2]:.0f}s, "
                  f"p90 {durs[int(len(durs)*0.9)]:.0f}s · best {max(e.best for e in group):+.3f}")
        for (nc, bc, _), e in sorted(eps, key=lambda ke: -ke[1].best)[:args.top]:
            nq = by_cid[nc][1]["q"]
            bq = by_cid[bc][1]["q"].split(": ", 1)[-1]
            print(f"   {e.best:+.3f} for {e.end - e.start:5.0f}s depth {e.depth:6.0f} "
                  f"{'LIVE' if e.inplay else 'pre '}  {nq[:58]} ⊆ {bq}")
        print()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/home/ubuntu/polybot-dev/recordings")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--rate", type=float, default=0.05, help="taker fee rate")
    ap.add_argument("--tick", type=float, default=0.01)
    ap.add_argument("--max-spread", type=float, default=0.05,
                    help="resting only in books this tight, as the guard does")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--rest-only", action="store_true",
                    help="replay only the REST reads (R lines): coarser, but the ground truth")
    sys.exit(main(ap.parse_args()))
