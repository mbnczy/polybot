"""
scripts/benchmark_max_feeds.py
──────────────────────────────
Benchmark MarketScanner._scan_once() at different MAX_FEEDS values.

Measures wall-clock time and peak RSS memory for each configuration using
either a live Gamma API snapshot (default) or a local JSON file, so the
benchmark reflects realistic data volumes.

Usage
─────
    # Live snapshot (fetches once, then reuses across all trials):
    python scripts/benchmark_max_feeds.py

    # Local snapshot (avoids network on repeated runs):
    python scripts/benchmark_max_feeds.py --snapshot markets.json

    # Custom max_feeds list and trial count:
    python scripts/benchmark_max_feeds.py --max-feeds 10 25 50 100 200 --trials 5

Output
──────
    max_feeds │ trials │ mean_ms │ p95_ms │ peak_rss_mb │ admitted/scan
    ──────────┼────────┼─────────┼────────┼─────────────┼──────────────
           10 │      5 │    4.12 │   4.38 │       42.1  │          10
           25 │      5 │    5.89 │   6.10 │       42.3  │          25
           50 │      5 │    8.34 │   8.91 │       42.6  │          50
          100 │      5 │   14.72 │  15.43 │       43.1  │         100
            0 │      5 │    3.01 │   3.19 │       41.8  │         712  ← passthrough

Target: mean_ms < 200 for chosen max_feeds value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import aiohttp

# Make project root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from core.scanner import MarketScorer, _extract_condition_id, _extract_token_ids, _is_binary

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during benchmark

_GAMMA_URL  = "https://gamma-api.polymarket.com/markets"
_PAGE_LIMIT = 100
_END_CURSOR = "LTE="


async def _fetch_snapshot() -> list[dict]:
    """Fetch all active binary markets from the Gamma API (one-shot)."""
    results: list[dict] = []
    cursor = ""
    async with aiohttp.ClientSession() as session:
        while True:
            params: dict[str, object] = {
                "active": "true", "closed": "false",
                "archived": "false", "limit": _PAGE_LIMIT,
            }
            if cursor:
                params["next_cursor"] = cursor
            async with session.get(
                _GAMMA_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            if isinstance(data, list):
                results.extend(m for m in data if _is_binary(m))
                break
            results.extend(m for m in data.get("data", []) if _is_binary(m))
            cursor = data.get("next_cursor", "")
            if not cursor or cursor == _END_CURSOR:
                break
    return results


def _score_and_admit(markets: list[dict], max_feeds: int) -> tuple[float, int]:
    """
    Run the scoring + admission logic (CPU only, no I/O) and return
    (wall_seconds, admitted_count).
    """
    scorer = MarketScorer()
    t0 = time.perf_counter()

    if max_feeds == 0:
        # passthrough path
        admitted = 0
        for m in markets:
            cid = _extract_condition_id(m)
            if not cid:
                continue
            yes_id, no_id = _extract_token_ids(m)
            if yes_id and no_id:
                admitted += 1
        return time.perf_counter() - t0, admitted

    # scoring path
    scored: list[tuple[float, dict]] = []
    for m in markets:
        cid = _extract_condition_id(m)
        if not cid:
            continue
        yes_id, no_id = _extract_token_ids(m)
        if not yes_id or not no_id:
            continue
        scored.append((scorer.score(m), m))

    scored.sort(key=lambda t: t[0], reverse=True)
    admitted = len(scored[:max_feeds])
    return time.perf_counter() - t0, admitted


def _peak_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        # Windows: use psutil if available
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return float("nan")


def _run_benchmark(
    markets: list[dict],
    max_feeds_list: list[int],
    trials: int,
) -> None:
    header = (
        f"{'max_feeds':>10} │ {'trials':>6} │ {'mean_ms':>8} │ "
        f"{'p95_ms':>7} │ {'peak_rss_mb':>12} │ {'admitted/scan':>14}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for mf in max_feeds_list:
        durations: list[float] = []
        last_admitted = 0
        for _ in range(trials):
            elapsed, admitted = _score_and_admit(markets, mf)
            durations.append(elapsed * 1000)
            last_admitted = admitted

        mean_ms = statistics.mean(durations)
        p95_ms  = sorted(durations)[int(len(durations) * 0.95)]
        rss     = _peak_rss_mb()
        flag    = "  ← TARGET EXCEEDED" if mean_ms > 200 else ""
        print(
            f"{mf:>10} │ {trials:>6} │ {mean_ms:>8.2f} │ "
            f"{p95_ms:>7.2f} │ {rss:>12.1f} │ {last_admitted:>14}{flag}"
        )

    print(sep)
    print("Target: mean_ms < 200 ms for chosen max_feeds value.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MarketScanner scoring at various max_feeds values")
    parser.add_argument(
        "--snapshot", metavar="FILE",
        help="Path to a local JSON file (list of Gamma API market objects). "
             "If omitted, a live snapshot is fetched from the Gamma API.",
    )
    parser.add_argument(
        "--max-feeds", nargs="+", type=int,
        default=[0, 10, 25, 50, 100],
        metavar="N",
        help="max_feeds values to benchmark (0 = passthrough). Default: 0 10 25 50 100",
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Number of timing trials per max_feeds value (default: 5)",
    )
    parser.add_argument(
        "--save-snapshot", metavar="FILE",
        help="Save the fetched live snapshot to a JSON file for reuse.",
    )
    args = parser.parse_args()

    if args.snapshot:
        path = Path(args.snapshot)
        print(f"Loading snapshot from {path} …")
        markets = json.loads(path.read_text())
        print(f"Loaded {len(markets)} binary markets.")
    else:
        print("Fetching live snapshot from Gamma API …")
        markets = await _fetch_snapshot()
        print(f"Fetched {len(markets)} binary markets.")
        if args.save_snapshot:
            Path(args.save_snapshot).write_text(json.dumps(markets, indent=2))
            print(f"Snapshot saved to {args.save_snapshot}")

    if not markets:
        print("No markets found — nothing to benchmark.")
        return

    print(f"\nBenchmarking {args.max_feeds} with {args.trials} trial(s) each …\n")
    _run_benchmark(markets, args.max_feeds, args.trials)


if __name__ == "__main__":
    asyncio.run(main())
