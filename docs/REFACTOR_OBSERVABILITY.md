# Refactoring & Observability

Branch: `feature/refactor-and-observability` (builds on `feature/efficiency-and-reliability`)

Goal: reduce duplication, fix latent bugs, and make the reliability/efficiency
safety paths **measurable** rather than log-only. Net result: the full test
suite goes from **75 passed / 4 failed → 90 passed / 0 failed**.

---

## R1 — DRY the signal-quality guard

The extreme-price check was copy-pasted in `ArbDetector.evaluate` and
`DutchBookPricer.evaluate_maker`. Extracted into one shared helper in
`strategy/arbitrage.py`:

```python
def _within_quality_band(yes_ask, no_ask, lo, hi) -> bool:
    return lo <= yes_ask <= hi and lo <= no_ask <= hi
```

Both detectors now call it — the quality rule lives in exactly one place.

## R2 — Single source of truth for `TICK_SIZE`

`TICK_SIZE` was defined twice (`strategy/arbitrage.py` **and**
`core/clob_client.py`). `clob_client` now imports it from `arbitrage` (a clean
one-way dependency), so the two can never drift apart.

## R3 — Fix `_days_to_close` (clears 4 pre-existing test failures)

`MarketScorer._days_to_close` returned `0.0` for expired/missing/bad dates and
the raw fractional days near expiry — which (a) failed 4 committed tests that
expect a **1.0 floor**, and (b) risked **score explosion** in the final hours
(dividing a market's score by e.g. 0.02 days).

Fix, matching the documented `score = volume × rebate / max(days, 1)` intent:
- `_days_to_close` now **floors at 1.0** (safe scoring denominator; neutral 1.0
  for missing/unparseable/expired).
- Expiry *gating* moved to a new explicit `MarketScorer._is_expired(market)`
  (parseable endDate in the past). The scanner skip now uses `_is_expired`
  instead of `_days_to_close(m) <= 0`. Missing/bad dates are treated as
  not-expired (Gamma already filters `active=true&closed=false`).

No behaviour regression: expired markets are still skipped before scoring; no
test scores an expired market, and score tests use multi-day futures.

## F1 — Observability metrics for the safety paths

The reliability/efficiency events were previously **log-only**. Added Prometheus
metrics (`telemetry/metrics.py`) and wired them in:

| Metric | Type | Incremented where |
|---|---|---|
| `polly_arb_detected_total` | Counter | every detected signal (`strategy_loop`) |
| `polly_real_edge_bps` | Gauge | `(1−(yes_ask+no_ask))·10000` per detection |
| `polly_arb_half_fills_total` | Counter | single-leg fill → unwind |
| `polly_arb_unwind_failures_total` | Counter | unwind raised (manual-intervention) |
| `polly_feeds_pruned_total` | Counter | `FeedRegistry.prune_stale` |
| `polly_stale_ticks_skipped_total` | Counter | freshness guard skip |

`real_edge_bps` is the key quality signal — it lets a dashboard distinguish a
genuine sub-$1 combined cost from a rebate-inflated near-1.0 "edge".

---

## Tests

```
tests/test_refactor_observability.py → 11 passed  (R1/R2/R3 helpers + FEEDS_PRUNED)
tests/test_market_scorer.py          → 16 passed  (incl. the 4 formerly-failing)
Full suite                           → 90 passed, 0 failed
```

## Files changed

| File | Change |
|---|---|
| `strategy/arbitrage.py` | `_within_quality_band` helper; both detectors use it |
| `core/clob_client.py` | import `TICK_SIZE` from arbitrage (de-duplicated) |
| `core/scanner.py` | `_days_to_close` 1.0 floor; new `_is_expired`; skip uses it; `FEEDS_PRUNED` |
| `telemetry/metrics.py` | 6 new reliability/efficiency metrics |
| `main.py` | wire detection / half-fill / unwind-fail / stale-tick metrics |
| `tests/test_refactor_observability.py` | 11 new tests |
