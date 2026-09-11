#!/usr/bin/env python3
"""
scripts/backtest_cross.py
─────────────────────────
A larger backtest of the implication mapper, on the population we now trade.

The first backtest checked 2,000 resolved markets and produced six testable
assertions — which proves nothing either way. This one is built to find out:

  • more history. Gamma caps offset pagination at 2,000 rows per query and every
    WEEK of resolved markets exceeds that, so the universe is fetched one DAY at
    a time, each window paged to its own ceiling.
  • the traded population. Execution only takes pairs that resolve inside a
    one-week window, so pairs whose two end dates lie further apart than
    --max-span-days are left out. A relation between a match sub-market and a
    year-end market is not one we would ever hold.
  • the live pipeline. Candidates go through the same prefilter, the same model,
    the same confidence threshold, and then the ladder direction guard — and the
    guard's rejections are scored too, to check it removes what is wrong.

An implication narrow ⊆ broad is only informative when narrow resolved YES; then
broad must have too. Where narrow resolved NO it holds vacuously and is counted
apart. Counterexamples inside a threshold family the exchange itself resolved
inconsistently are reported apart from the rate.

    python scripts/backtest_cross.py --days 42 --pairs 200 --env-file ../cross-market/.env

Read-only: fetches history and calls the model. Places nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from scripts.backtest_implications import (  # noqa: E402
    _family_key,
    _outcome,
    inconsistent_families,
)

GAMMA = "https://gamma-api.polymarket.com/markets"
_PAGE = 100
_OFFSET_CEILING = 2000
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def _end(m: dict) -> "float | None":
    try:
        return datetime.fromisoformat(
            str(m.get("endDate")).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def fetch_resolved(days: int, limit: int, gap_s: float = 0.15) -> list[dict]:
    """Resolved markets with a clean 0/1 outcome, one day-window at a time."""
    now = datetime.now(timezone.utc)
    out: list[dict] = []
    seen: set[str] = set()
    capped = 0
    with httpx.Client(timeout=30.0) as h:
        for d in range(days):
            hi, lo = now - timedelta(days=d), now - timedelta(days=d + 1)
            offset = 0
            while offset < _OFFSET_CEILING:
                r = h.get(GAMMA, params={
                    "closed": "true", "limit": _PAGE, "offset": offset,
                    "end_date_min": lo.strftime(_ISO), "end_date_max": hi.strftime(_ISO),
                })
                if r.status_code == 429:
                    time.sleep(2.0)
                    continue
                if r.status_code != 200:
                    break
                page = r.json()
                if not page:
                    break
                for m in page:
                    cid = str(m.get("conditionId") or "")
                    if cid and cid not in seen and _outcome(m) is not None and m.get("question"):
                        seen.add(cid)
                        out.append(m)
                offset += len(page)
                time.sleep(gap_s)
            if offset >= _OFFSET_CEILING:
                capped += 1
            if len(out) >= limit:
                break
    if capped:
        print(f"    {capped} day-window(s) hit Gamma's 2,000-row ceiling — sampled, not complete")
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=42, help="history to fetch, in days")
    ap.add_argument("--markets", type=int, default=20000, help="cap on markets fetched")
    ap.add_argument("--pairs", type=int, default=200, help="pairs sent to the model")
    ap.add_argument("--per-event", type=int, default=3)
    ap.add_argument("--max-span-days", type=float, default=7.0,
                    help="drop pairs whose end dates lie further apart than this")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--env-file", default=str(REPO / ".env"))
    ap.add_argument("--out", default="backtest_cross_report.json")
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(a.env_file)
    # implication_mapper reads IMPLICATION_* at import time — after the env.
    from strategy.implication_mapper import (  # noqa: PLC0415
        build_candidates, build_client, classify_candidates,
        resolve_model, resolve_provider,
    )
    from strategy.ladder_direction import check_direction, CONTRADICTS  # noqa: PLC0415

    t0 = time.time()
    print(f"\n  fetching resolved markets, {a.days} days back, one day at a time…")
    markets = fetch_resolved(a.days, a.markets)
    by_id = {str(m["conditionId"]): m for m in markets}
    truth = {cid: _outcome(m) for cid, m in by_id.items()}
    print(f"    {len(markets)} markets with a clean outcome "
          f"({sum(1 for v in truth.values() if v)} resolved YES) in {time.time() - t0:.0f}s")

    # Oversample, then keep the traded population.
    cands = build_candidates(markets, max_pairs=a.pairs * 4, max_per_event=a.per_event)

    def span(c) -> "float | None":
        ea, eb = _end(by_id[c.a_id]), _end(by_id[c.b_id])
        return None if ea is None or eb is None else abs(ea - eb) / 86_400.0

    in_window = [c for c in cands if (span(c) is not None and span(c) <= a.max_span_days)]
    shape_of = {frozenset((c.a_id, c.b_id)): c.shape for c in in_window}
    cands = in_window[:a.pairs]
    print(f"  prefilter : {len(cands)} pair(s) within a {a.max_span_days:g}-day span  "
          f"{dict(collections.Counter(c.shape for c in cands))}")
    if not cands:
        return 0

    provider = resolve_provider()
    model = resolve_model(provider)
    client = build_client(provider)
    print(f"  classifying {len(cands)} pair(s) on {provider.name}/{model}…")
    t1 = time.time()
    rels = [r for r in classify_candidates(cands, provider=provider, model=model,
                                           client=client, concurrency=a.concurrency)
            if r.confidence >= a.threshold]
    print(f"    {len(rels)} implication(s) asserted at ≥{a.threshold} "
          f"in {(time.time() - t1) / 60:.1f} min")

    suspect = inconsistent_families(markets)
    kept, backwards = [], []
    for r in rels:
        (backwards if check_direction(r.narrow, r.broad, markets) == CONTRADICTS
         else kept).append(r)

    def score(rs):
        testable, vacuous, bad, tainted = [], 0, [], []
        for r in rs:
            tn, tb = truth.get(r.narrow), truth.get(r.broad)
            if tn is None or tb is None:
                continue
            if not tn:
                vacuous += 1
                continue
            testable.append(r)
            if not tb:
                q = str(by_id[r.narrow].get("question") or "")
                qb = str(by_id[r.broad].get("question") or "")
                (tainted if (_family_key(q) in suspect or _family_key(qb) in suspect)
                 else bad).append(r)
        return testable, vacuous, bad, tainted

    t_k, v_k, bad_k, taint_k = score(kept)
    t_b, v_b, bad_b, _ = score(backwards)

    print(f"\n  {'':26}{'asserted':>10}{'vacuous':>10}{'testable':>10}{'violated':>10}")
    print(f"  {'kept (traded)':26}{len(kept):>10}{v_k:>10}{len(t_k):>10}{len(bad_k):>10}")
    print(f"  {'refused as backwards':26}{len(backwards):>10}{v_b:>10}{len(t_b):>10}{len(bad_b):>10}")
    if taint_k:
        print(f"    + {len(taint_k)} counterexample(s) in families the exchange "
              f"resolved inconsistently — reported apart")

    rate = len(bad_k) / len(t_k) if t_k else None
    if rate is not None:
        # Wilson upper bound: with few testable pairs a zero rate is weak evidence.
        n, k, z = len(t_k), len(bad_k), 1.96
        centre = (k + z * z / 2) / (n + z * z)
        half = z * ((k * (n - k) / n + z * z / 4) ** 0.5) / (n + z * z)
        upper = centre + half
        print(f"\n  VIOLATION RATE {rate * 100:.1f}%  ({len(bad_k)}/{len(t_k)} testable)"
              f"  — 95% upper bound {upper * 100:.1f}%")
    else:
        print("\n  no testable assertion — narrow never resolved YES in this sample")

    by_shape = collections.defaultdict(lambda: [0, 0])
    for r in t_k:
        by_shape[shape_of.get(frozenset((r.narrow, r.broad)), "?")][0] += 1
    for r in bad_k:
        by_shape[shape_of.get(frozenset((r.narrow, r.broad)), "?")][1] += 1
    if by_shape:
        print("  by shape  " + "   ".join(f"{s}: {v[1]}/{v[0]}" for s, v in sorted(by_shape.items())))

    if bad_k:
        print("\n  counterexamples among KEPT implications — the mapper was wrong here:")
        for r in bad_k[:10]:
            print(f"    narrow YES: {str(by_id[r.narrow].get('question'))[:70]}")
            print(f"    broad  NO : {str(by_id[r.broad].get('question'))[:70]}\n")
    if t_b:
        print(f"  the guard's refusals, scored: {len(bad_b)}/{len(t_b)} testable ones "
              f"were violations — {'it removed wrong relations' if bad_b else 'none were wrong here'}")

    report = {
        "days": a.days, "markets": len(markets), "pairs": len(cands),
        "asserted": len(rels), "kept": len(kept), "refused_backwards": len(backwards),
        "testable": len(t_k), "violations": len(bad_k), "exchange_inconsistent": len(taint_k),
        "violation_rate": rate, "by_shape": {k: {"testable": v[0], "violated": v[1]}
                                             for k, v in by_shape.items()},
        "counterexamples": [{"narrow": str(by_id[r.narrow].get("question")),
                             "broad": str(by_id[r.broad].get("question"))} for r in bad_k],
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\n  report → {a.out}  ({report['runtime_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
