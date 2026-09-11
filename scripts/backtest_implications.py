#!/usr/bin/env python3
"""
scripts/backtest_implications.py
────────────────────────────────
Check the implication mapper against markets that have already resolved.

Why this exists
───────────────
The strategy's only loss mode is asserting an implication that is not true. Yet
the confidence attached to every assertion is the model's own opinion of itself:
`confidence: 0.97` is a claim, not a measurement. Three days of live running
produced 52 discovery rounds, 290 price checks and zero violations, which tells
us nothing about whether the mapper is right — only that nothing was mispriced
while we watched.

Resolved markets settle it. If the model says A implies B, then in every world
where A resolved YES, B must also have resolved YES. A single counterexample
proves the implication false, and counting them measures what the confidence
score is actually worth.

What a result means
───────────────────
Only pairs where the NARROW side resolved YES carry information. If A resolved
NO the implication is vacuously satisfied and proves nothing — a mapper that
asserts nonsense will look perfect on those. The report separates the two, and
the number that matters is the violation rate among TESTABLE pairs.

    python scripts/backtest_implications.py --markets 600 --pairs 40

Read-only: fetches history, calls the model, prints. Places no orders and
touches no live state.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

# strategy.implication_mapper reads IMPLICATION_* at import time, so it must be
# imported AFTER load_dotenv or it captures the wrong provider — the same reason
# demo_cross_market imports it inside a function.

GAMMA = "https://gamma-api.polymarket.com/markets"
_GAMMA_PAGE = 100     # Gamma's hard ceiling; asking for more silently returns 100
logger = logging.getLogger("backtest")


def fetch_resolved(limit: int) -> list[dict]:
    """Closed markets with an unambiguous 0/1 outcome, newest first."""
    # Gamma silently caps `limit` at 100 however much you ask for. Advancing the
    # offset by the REQUESTED page size therefore skipped 400 markets per page:
    # the sample was 20% of history with holes in it. Worse for this backtest
    # than a plain shortfall — an endDate-ordered listing keeps an event's
    # markets adjacent, so striding across it splits event clusters and leaves
    # behind whatever pairs up within a single page, which is precisely the
    # numeric ladders. Advance by what the page actually returned.
    out: list[dict] = []
    with httpx.Client(timeout=30.0) as c:
        offset = 0
        while len(out) < limit and offset < 20_000:
            r = c.get(GAMMA, params={
                "closed": "true", "limit": _GAMMA_PAGE, "offset": offset,
                "order": "endDate", "ascending": "false",
            })
            if r.status_code != 200:
                logger.warning("gamma %s at offset %d — stopping with %d market(s)",
                               r.status_code, offset, len(out))
                break
            batch = r.json()
            if not batch:
                break
            for m in batch:
                if _outcome(m) is not None and m.get("question"):
                    out.append(m)
            offset += len(batch)
    return out[:limit]


# ── ground truth is not always self-consistent ────────────────────────────────
# Polymarket resolved "Reppo FDV above $100M one day after launch?" NO while
# resolving the $500M and $800M markets in the same event, under an identical
# description, YES. No implication mapper can be right about that pair, and
# counting it as a mapper error puts the blame in the wrong place — it moved the
# headline from 0 to 16.7%.
#
# The check has to stay narrow. "Will 47 senators vote Yea?" is also non-monotone
# across its family, and correctly so: it asks for an exact count, not a
# threshold, and a mapper that reads it as a ladder IS wrong and must be scored
# that way. So only families whose wording states a comparison are checked.
_COMPARATIVE = re.compile(
    r"\b(above|below|over|under|greater|less|at least|at most|or more|or fewer|"
    r"reach|reaches|dip to|exceed|surpass)\b", re.I)
_FAMILY_NUM = re.compile(r"\$?([\d,]+(?:\.\d+)?)\s*([kmbt])?\b", re.I)
_MAGNITUDE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _family_key(question: str) -> str | None:
    """Skeleton of a threshold family: the title with its one number blanked."""
    if not _COMPARATIVE.search(question):
        return None
    nums = _FAMILY_NUM.findall(question)
    if len(nums) != 1:
        return None
    return _FAMILY_NUM.sub("#", question)


def _threshold_value(question: str) -> float:
    n, sfx = _FAMILY_NUM.findall(question)[0]
    return float(n.replace(",", "")) * _MAGNITUDE.get((sfx or "").lower(), 1.0)


def inconsistent_families(markets: list[dict]) -> set[str]:
    """
    Threshold families whose recorded outcomes contradict themselves.

    Sorted by threshold, a comparative family must be monotone in one direction
    or the other — "above $X" gets harder as X rises, "below $X" easier. A family
    that is monotone in neither cannot all be true.
    """
    fam: dict[str, list[tuple[float, bool]]] = collections.defaultdict(list)
    for m in markets:
        q, out = str(m.get("question") or ""), _outcome(m)
        key = _family_key(q) if out is not None else None
        if key:
            fam[key].append((_threshold_value(q), out))

    bad = set()
    for key, rows in fam.items():
        if len(rows) < 3:            # two points are monotone by construction
            continue
        rows.sort()
        outs = [o for _, o in rows]
        rising  = all(a <= b for a, b in zip(outs, outs[1:]))
        falling = all(a >= b for a, b in zip(outs, outs[1:]))
        if not rising and not falling:
            bad.add(key)
    return bad


def _outcome(m: dict) -> bool | None:
    """True if YES resolved, False if NO, None if not cleanly resolved."""
    try:
        pr = json.loads(m.get("outcomePrices") or "[]")
        if len(pr) != 2:
            return None
        yes, no = float(pr[0]), float(pr[1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if {yes, no} != {0.0, 1.0}:
        return None
    return yes == 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=600)
    ap.add_argument("--pairs",   type=int, default=40,
                    help="candidate pairs sent to the model (costs money)")
    ap.add_argument("--per-event", type=int, default=2,
                    help="cap pairs drawn from any one event. The overlap "
                         "ranking otherwise fills the whole sample with a "
                         "single threshold ladder — the easiest case there is, "
                         "and no measure of the mapper at all")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--env-file", default=str(REPO / ".env"))
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from dotenv import load_dotenv
    load_dotenv(a.env_file)

    from strategy.implication_mapper import (   # noqa: PLC0415 — after the env
        build_candidates, build_client, classify_candidates,
        resolve_model, resolve_provider,
    )

    print(f"\n  fetching up to {a.markets} resolved markets…")
    markets = fetch_resolved(a.markets)
    truth = {str(m["conditionId"]): _outcome(m) for m in markets
             if m.get("conditionId")}
    titles = {str(m.get("conditionId")): str(m.get("question") or "")
              for m in markets}
    print(f"    got {len(markets)} with a clean outcome "
          f"({sum(1 for v in truth.values() if v)} resolved YES)")

    cands = build_candidates(markets, max_pairs=a.pairs,
                             max_per_event=a.per_event)
    print(f"  prefilter  : {len(cands)} candidate pair(s)")
    if not cands:
        return 0

    provider = resolve_provider()
    model    = resolve_model(provider)
    client   = build_client(provider)
    print(f"  classifying on {provider.name}/{model}…")
    rels = classify_candidates(
        cands, provider=provider, model=model, client=client,
        concurrency=a.concurrency,
    )
    rels = [r for r in rels if r.confidence >= a.threshold]
    print(f"    {len(rels)} implication(s) asserted at ≥{a.threshold}")
    if not rels:
        print("\n  nothing asserted — no accuracy to measure")
        return 0

    # An implication A ⊆ B is only informative when A resolved YES. Where A
    # resolved NO it holds vacuously and would flatter a mapper asserting
    # nonsense, so those are counted separately and excluded from the rate.
    testable, vacuous, violations = [], 0, []
    for r in rels:
        tn, tb = truth.get(r.narrow), truth.get(r.broad)
        if tn is None or tb is None:
            continue
        if not tn:
            vacuous += 1
            continue
        testable.append(r)
        if not tb:
            violations.append(r)

    # Split the violations: a mapper error and an exchange that contradicts
    # itself are both counterexamples, and only one of them is our problem.
    suspect = inconsistent_families(markets)
    tainted = [r for r in violations
               if _family_key(titles.get(r.narrow, "")) in suspect
               or _family_key(titles.get(r.broad, "")) in suspect]
    genuine = [r for r in violations if r not in tainted]

    print(f"\n  {'':4}{'asserted':>10}{'vacuous':>10}{'testable':>10}"
          f"{'violated':>10}")
    print(f"  {'':4}{len(rels):>10}{vacuous:>10}{len(testable):>10}"
          f"{len(violations):>10}")

    # How many assertions are numeric ladders — two titles identical but for a
    # number. The prefilter ranks by token overlap, and the only implications
    # that look lexically near-identical ARE ladders, so they crowd out
    # everything else. They are also the case the market is least likely to
    # misprice, being obvious to everyone: a strategy that only ever sees
    # ladders is looking exactly where no arbitrage survives.
    import re as _re
    def _ladder(a: str, b: str) -> bool:
        strip = lambda t: _re.sub(r"[\d.,$]+", "#", t.lower())
        return strip(a) == strip(b)

    ladders = sum(1 for r in rels
                  if _ladder(titles.get(r.narrow, ""), titles.get(r.broad, "")))
    print(f"\n  shape of what the model was shown")
    print(f"    numeric ladders   {ladders:>4}/{len(rels)} "
          f"({ladders / len(rels) * 100:.0f}%)")
    print(f"    everything else   {len(rels) - ladders:>4}/{len(rels)}")
    if ladders == len(rels):
        print("    -> the prefilter never offered a non-ladder pair. Ladders are")
        print("       the easiest relations to get right AND the least likely to")
        print("       be mispriced, so this measures accuracy on the cases that")
        print("       can never pay.")

    print(f"\n  a sample of what was asserted:")
    for r in rels[:6]:
        tn, tb = truth.get(r.narrow), truth.get(r.broad)
        mark = lambda v: "YES" if v else ("NO " if v is False else " ? ")
        print(f"    [{mark(tn)}] {titles.get(r.narrow,'')[:52]}")
        print(f"      implies")
        print(f"    [{mark(tb)}] {titles.get(r.broad,'')[:52]}")
        print()

    if not testable:
        print("\n  none of the assertions were testable — the narrow side never")
        print("  resolved YES in this sample, so nothing was proved either way.")
        print("  Take a larger sample before trusting any of them.")
        return 0

    if tainted:
        print(f"\n  {len(tainted)} of the {len(violations)} counterexample(s) sit in a threshold")
        print(f"  family whose own recorded outcomes contradict each other — the")
        print(f"  exchange resolved a lower threshold NO and a higher one YES under")
        print(f"  an identical description. No mapper can be right about those, so")
        print(f"  they are reported apart from the rate rather than charged to it.")
        for r in tainted:
            print(f"    · {titles.get(r.narrow,'')[:60]}")

    rate = len(genuine) / len(testable)
    violations = genuine
    print(f"\n  VIOLATION RATE {rate * 100:.1f}%  "
          f"({len(violations)}/{len(testable)} testable assertions were false)")

    by_conf = collections.defaultdict(lambda: [0, 0])
    for r in testable:
        b = f"{r.confidence:.2f}"
        by_conf[b][0] += 1
        if r in violations:
            by_conf[b][1] += 1
    print(f"\n  by the model's own confidence")
    print(f"    {'claimed':>9}{'testable':>10}{'violated':>10}{'actual':>9}")
    for b in sorted(by_conf, reverse=True):
        n, v = by_conf[b]
        print(f"    {b:>9}{n:>10}{v:>10}{(1 - v / n) * 100:>8.0f}%")

    if violations:
        print(f"\n  counterexamples — the model asserted these and reality "
              f"disagreed:")
        for r in violations[:5]:
            print(f"\n    narrow resolved YES: {titles.get(r.narrow, r.narrow)[:64]}")
            print(f"    broad  resolved NO : {titles.get(r.broad,  r.broad)[:64]}")
            print(f"    claimed confidence : {r.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
