"""
scripts/smoke_negrisk_feeds.py
──────────────────────────────
Read-only smoke test for the NegRisk pipeline against the LIVE Polymarket API.

Places NO orders and touches no wallet: it exercises only
MarketScanner group discovery → FeedRegistry → MarketShard → NegRiskArbDetector,
then prints what the detector would have done.

Usage::

    .venv/bin/python scripts/smoke_negrisk_feeds.py [--seconds 45] [--groups 5]

Reports, per group: how many neg_risk_ticks arrived, how many legs were quoted,
the best (highest) relative edge observed, and why signals were rejected.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PAPER_TRADE_MODE", "true")   # belt and braces

from core.scanner import FeedRegistry, MarketScanner   # noqa: E402
from strategy.arbitrage import NegRiskArbDetector      # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("smoke-negrisk")


async def main(seconds: float, max_groups: int, pair_cap: float) -> int:
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=4096)
    registry = FeedRegistry(queue=queue, max_feeds=max_groups)

    registered: list[tuple[str, int]] = []

    async def _on_group(group_id: str, no_ids: list[str]) -> bool:
        if len(registered) >= max_groups:
            return False
        ok = await registry.add_neg_risk_group(group_id, no_ids)
        if ok:
            registered.append((group_id, len(no_ids)))
        return ok

    async def _on_market(*_a, **_kw) -> bool:
        return False        # binary feeds are out of scope for this smoke test

    scanner = MarketScanner(
        on_market_added=_on_market,
        on_neg_risk_group=_on_group,
        max_feeds=max_groups,
        feed_registry=registry,
    )

    logger.info("Scanning Gamma for NegRisk groups …")
    await scanner._scan_once()

    if not registered:
        logger.error("No NegRisk groups discovered — pipeline cannot be verified")
        await registry.stop_all()
        return 1

    logger.info("Registered %d group(s):", len(registered))
    for gid, n in registered:
        logger.info("  %s — %d outcome legs", gid[:20], n)

    detector = NegRiskArbDetector()

    ticks       = 0
    per_group   = Counter()
    legs_seen   = defaultdict(int)
    signals     = 0
    best_edge   = 0.0
    best_detail = ""

    logger.info("Listening for neg_risk_ticks for %.0f s …", seconds)
    deadline = asyncio.get_event_loop().time() + seconds
    while asyncio.get_event_loop().time() < deadline:
        timeout = deadline - asyncio.get_event_loop().time()
        try:
            tick = await asyncio.wait_for(queue.get(), timeout=max(0.1, timeout))
        except asyncio.TimeoutError:
            break
        if tick.get("type") != "neg_risk_tick":
            continue

        ticks += 1
        gid = tick["condition_id"]
        per_group[gid] += 1
        legs_seen[gid] = max(legs_seen[gid], len(tick["no_asks"]))

        sig = detector.evaluate_neg_risk(
            condition_id=gid,
            outcome_token_ids=tick["outcome_token_ids"],
            no_asks=tick["no_asks"],
            max_position_usdc=pair_cap,
            maker_rebate=0.0,
            tick_size=tick.get("tick_size"),
            no_ask_sizes=tick.get("no_ask_sizes"),
        )
        if sig is not None:
            signals += 1
            if sig.relative_edge > best_edge:
                best_edge = sig.relative_edge
                best_detail = (
                    f"{gid[:20]} legs={sig.n_outcomes} "
                    f"combined_bid={sig.combined_bid:.4f} "
                    f"bundles={sig.n_bundles:.2f}"
                )

        # Diagnostic: how far the group is from an arb, regardless of gating.
        implied_sum = sum(1.0 - a for a in tick["no_asks"])
        logger.debug(
            "tick group=%s legs=%d Σimplied_yes=%.4f %s",
            gid[:16], len(tick["no_asks"]), implied_sum,
            "ARB" if implied_sum > 1.0 else "",
        )

    await registry.stop_all()

    print()
    print("═" * 72)
    print(f"groups registered : {len(registered)}")
    print(f"neg_risk_ticks    : {ticks}")
    print(f"groups that ticked: {len(per_group)}")
    for gid, n in per_group.most_common(10):
        print(f"    {gid[:20]}  ticks={n:<5} max_quoted_legs={legs_seen[gid]}")
    print(f"signals passing all gates : {signals}")
    if best_detail:
        print(f"best relative_edge        : {best_edge:.4f}  ({best_detail})")
    print("═" * 72)

    if ticks == 0:
        logger.error("Pipeline produced NO neg_risk_ticks — feed path is broken")
        return 1
    logger.info("NegRisk feed path verified end-to-end (no orders placed).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--groups",  type=int,   default=5)
    ap.add_argument("--cap",     type=float,
                    default=float(os.environ.get("MAX_ARB_PAIR_USDC", "20")))
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.seconds, args.groups, args.cap)))
