"""
scripts/bench_hotpath.py
────────────────────────
Local micro-benchmark for the detection hot path — no network, no event loop.

Measures the two pure, per-tick stages that dominate detection cost:
  1. WS message parse + best-ask update   (MarketFeed._dispatch)
  2. arbitrage evaluation                  (DutchBookPricer.evaluate_maker /
                                            ArbDetector.evaluate)

Reports p50 / p99 / mean nanoseconds-per-op so each refactor phase can be
compared against a recorded baseline (see docs/REFACTOR_PLAN.md §9).

Run:
    python scripts/bench_hotpath.py            # default 200k iterations
    ITERS=1000000 python scripts/bench_hotpath.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.ws_feed import _loads, MarketFeed                  # noqa: E402
from strategy.arbitrage import ArbDetector, DutchBookPricer  # noqa: E402

_ITERS = int(os.environ.get("ITERS", "200000"))


def _pctl(samples_ns: list[float], q: float) -> float:
    s = sorted(samples_ns)
    return s[min(len(s) - 1, int(q * len(s)))]


def _report(label: str, samples_ns: list[float]) -> None:
    print(
        f"  {label:<28} p50={_pctl(samples_ns,0.50):8.0f} ns  "
        f"p99={_pctl(samples_ns,0.99):9.0f} ns  "
        f"mean={statistics.mean(samples_ns):8.0f} ns  (n={len(samples_ns):,})"
    )


def bench_parse(iters: int) -> list[float]:
    """Parse + best-ask update via a real (unconnected) MarketFeed."""
    book = (
        '{"event_type":"book","asset_id":"Y",'
        '"asks":[{"price":"0.47","size":"300"},{"price":"0.48","size":"100"}],'
        '"bids":[{"price":"0.45","size":"200"}]}'
    )
    feed = MarketFeed("Y", "N", "0xc", asyncio.Queue())
    samples: list[float] = []
    loop = asyncio.new_event_loop()
    try:
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            loop.run_until_complete(feed._dispatch(book))
            samples.append(time.perf_counter_ns() - t0)
    finally:
        loop.close()
    return samples


def bench_loads(iters: int) -> list[float]:
    raw = '{"event_type":"price_change","asset_id":"Y","price":"0.47","side":"SELL","size":"50"}'
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _loads(raw)
        samples.append(time.perf_counter_ns() - t0)
    return samples


def bench_eval_maker(iters: int) -> list[float]:
    pricer = DutchBookPricer()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        pricer.evaluate_maker("0xc", "Y", "N", 0.47, 0.50, maker_rebate=0.01)
        samples.append(time.perf_counter_ns() - t0)
    return samples


def bench_eval_taker(iters: int) -> list[float]:
    det = ArbDetector(default_fee_rate=0.0)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        det.evaluate("0xc", "Y", "N", 0.47, 0.50, fee_rate=0.0)
        samples.append(time.perf_counter_ns() - t0)
    return samples


def main() -> None:
    n = _ITERS
    print(f"\n  Hot-path micro-benchmark  (iters={n:,}, orjson={_orjson_active()})")
    print("  " + "─" * 70)
    _report("json _loads()",        bench_loads(n))
    _report("evaluate_maker()",     bench_eval_maker(n))
    _report("evaluate() (taker)",   bench_eval_taker(n))
    # _dispatch drives an event loop per call, so use fewer iterations.
    _report("_dispatch (parse+upd)", bench_parse(min(n, 50_000)))
    print()


def _orjson_active() -> bool:
    try:
        import orjson  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
