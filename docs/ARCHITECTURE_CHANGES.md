# Architecture Changes vs. Original `main`

Compares the current development branch against the original code on `main`
(`920a3bf "runner"`). Scope: **11 commits, +4,138 / −130 lines across 26 files.**

> Lineage note: the first delta from `main` was the pre-existing
> `feature/dutch-book-negrisk-integration` base (`b4990bf`), which introduced the
> **DutchBookPricer** (maker path) and **NegRiskArbDetector** strategy layer. The
> remaining six rounds (reliability, efficiency, refactor, observability, latency,
> Phase 0/1 perf) are the subject of this document and are flagged **[new]**.

---

## 1. At a glance — what changed by subsystem

| Subsystem | `main` (original) | Now |
|---|---|---|
| Strategy / pricing | Taker `ArbDetector` only | + maker **DutchBookPricer**, **NegRiskArbDetector**, **signal-quality guards** (extreme-price, min-real-edge) **[new]** |
| Execution | Fire dual FOK, **book profit unconditionally** | **Leg-fill reconciliation** + naked-leg **unwind**; profit booked only on confirmed both-fill **[new]** |
| Market discovery | Scan + admit `max_feeds` **per scan** (unbounded growth) | **Global feed cap**, **stale-feed pruning**, **liquidity/volume floor**, **cache pre-warm on admit** **[new]** |
| Real-time feed | stdlib `json`, push every event | **orjson** parse, **tick dedup**, **stale-tick guard**, monotonic-clock `ts`, `idle_seconds` liveness **[new]** |
| Hot-path latency | per-tick `await` fee/rebate fetch | **cache pre-warm** + **sync `peek_*`** (no event-loop bounce when warm) **[new]** |
| Risk | `CircuitBreaker` gates (unchanged core) | same gates; `_days_to_close` **floored at 1.0** + explicit `_is_expired` **[new]** |
| Settlement | `auto_redeem` at resolution | `InventoryManager` (instant `mergePositions`) **exists but not yet wired** — planned |
| Observability | 5 Prometheus metrics | **17 metrics** incl. per-hop latency histograms, half-fills, pruning, real-edge **[new]** |
| Alerting | execution-only Telegram | + **real-time detection alerts** (throttled) + standalone **live monitor** **[new]** |
| Tooling | Kaggle backtest only | + **orderFilled/parquet adapters**, **live smoke**, **hot-path benchmark** **[new]** |
| Config / hygiene | minimal `.gitignore`, dead `ARB_THRESHOLD` | hardened `.gitignore` (secrets), documented env, `TICK_SIZE` single source **[new]** |
| Hot objects | plain dataclasses | `slots=True` on `ArbSignal`/`ArbLeg`/`NegRiskSignal`/`OrderBookTick` **[new]** |

---

## 2. Data flow — before vs after

**Original (`main`):**
```
Scanner → FeedRegistry → MarketFeed(WS, json) → Queue → strategy_loop:
   await fee  →  ArbDetector(taker)  →  size  →  gate  →  execute_arb_pair
   →  book profit unconditionally  →  Telegram (on execution)
```

**Now:**
```
Scanner (global cap + prune + volume floor + on_admit PRE-WARM caches)
   → FeedRegistry → MarketFeed (orjson, dedup, freshness ts, liveness)
   → Queue → strategy_loop:
       freshness guard (drop stale)
       fee/rebate via SYNC PEEK (await only on cold miss)
       ArbDetector(taker)  →  DutchBookPricer(maker, quality-guarded)
       → detection metric + REAL_EDGE + throttled Telegram ALERT
       → size  →  gate
       → execute_arb_pair / execute_arb_maker_pair
       → RECONCILE fills: both → book P&L ; half → UNWIND, no P&L ; none → defer
   (per-hop latency histograms throughout)
```

---

## 3. Subsystem deep-dives

### 3.1 Strategy / pricing — `strategy/arbitrage.py`
- **[base]** Maker `DutchBookPricer` (synthetic post-only bids, rebate-aware) and
  `NegRiskArbDetector` (N-outcome `payout = n−1`) added alongside the original
  taker `ArbDetector`.
- **[new]** **Signal-quality guards**, factored into one shared
  `_within_quality_band(yes,no,lo,hi)` used by both detectors:
  - **Extreme-price filter** `[EXTREME_PRICE_LO, EXTREME_PRICE_HI]` — skip
    near-resolved markets whose "edge" is just the rebate subsidy.
  - **`MIN_REAL_EDGE`** (maker) — optionally require a genuine pre-rebate gap.
- **[new]** `FeeEngine`/`MakerRebateEngine` gain `prime_cache` + **sync
  `peek_*`**; `TICK_SIZE` becomes the single source of truth (re-exported by
  `clob_client`).
- **[new]** `slots=True` on `ArbSignal`/`ArbLeg`/`NegRiskSignal`.

### 3.2 Execution & reconciliation — `core/clob_client.py`, `main.py`
- **Original behavior (a correctness bug):** after firing both FOK legs, the loop
  booked `net_profit` **unconditionally**, assuming both filled.
- **[new]** `classify_fills(yes_resp,no_resp) → both|yes_only|no_only|none` and
  `unwind_leg(...)`. `strategy_loop` now:
  - `both` → book P&L + execution alert,
  - half → **flatten the naked leg**, alert, **book nothing**,
  - none → defer (resting maker orders).
  This converts a silent naked-exposure-as-profit bug into a safe, observable path.

### 3.3 Market discovery & feeds — `core/scanner.py`
- **Original:** `max_feeds` was a **per-scan** admission limit → feeds grew
  without bound across re-scans (observed 50→94 live).
- **[new]** `FeedRegistry` enforces a **global** active-feed cap (`add_market`
  returns `bool`); `prune_stale(max_idle_s)` drops dead feeds;
  cap-aware admission only fills free slots and marks markets known on success.
- **[new]** **Liquidity floor** (`min_volume_24h`) skips dead books.
- **[new]** `_days_to_close` floored at 1.0 (safe scoring denominator) with
  expiry gating moved to explicit `_is_expired` (fixed 4 latent test failures).
- **[new]** `on_admit(condition_id, market)` hook → **cache pre-warm** before the
  first tick.

### 3.4 Real-time feed — `core/ws_feed.py`
- **[new]** **orjson** parse (optional, stdlib fallback) + `WS_PARSE_SECONDS`.
- **[new]** **Tick dedup** (skip unchanged best-ask), **`_last_tick_monotonic`/
  `idle_seconds`** liveness for pruning, monotonic `ts` for correct latency.

### 3.5 Hot-path latency — `main.py`
- **[new]** **Freshness guard** (`MAX_TICK_AGE_S`) drops stale quotes.
- **[new]** **Peek-first** fee/rebate (sync cache hit → no `await` bounce);
  network fetch only on a cold miss.
- **[new]** End-to-end + per-hop latency histograms.

### 3.6 Observability — `telemetry/metrics.py`
- **Original:** `usdc_balance`, `cumulative_pnl`, `ws_latency_ms`,
  `active_markets`, `match_orders_total` (+scanner gauges).
- **[new]:** `arb_detected_total`, `real_edge_bps`, `arb_half_fills_total`,
  `arb_unwind_failures_total`, `feeds_pruned_total`, `stale_ticks_skipped_total`,
  `eval_latency_seconds`, `ws_parse_seconds`, `sign_seconds`, `submit_seconds`,
  `tick_to_ack_seconds`.

### 3.7 Alerting — `telemetry/telegram.py`, `scripts/live_arb_monitor.py`
- **[new]** `send_arb_detected(...)` fires on **detection** (independent of
  execution), throttled per-market (`ARB_ALERT_COOLDOWN_S`) + min-edge floor.
- **[new]** `live_arb_monitor.py`: persistent detection-only service (no trading).

### 3.8 Tooling — `backtest/`, `scripts/`
- **[new]** `OrderFilledDataFeed` (warproxxx CSV) + `TradesParquetDataFeed`
  (HuggingFace) backtest adapters; `run_backtest.py` CLI flags.
- **[new]** `smoke_live_arb.py` (live connectivity/arb-search) and
  `bench_hotpath.py` (network-free p50/p99 micro-benchmark).

---

## 4. Behavioral / semantic changes (not just additions)
1. **Profit is booked only on a confirmed two-leg fill** (was: always). Half-fills
   are flattened and **not** counted — the single most important correctness fix.
2. **Feeds are globally capped and pruned** (was: unbounded growth).
3. **Near-resolved / illiquid markets are skipped** (was: traded on rebate-inflated edge).
4. **Stale quotes are dropped** before acting (was: acted on regardless of age).
5. **Scoring no longer explodes near expiry** (1.0 floor) and expiry-skip is explicit.
6. **`ARB_THRESHOLD` is documented as dead**; the live trigger is `DESIRED_NET_MARGIN`.

## 5. What deliberately did NOT change
- The core arbitrage math (Dutch-book identity; taker/maker formulas).
- The 8-coroutine `asyncio.gather` topology and the shared-queue model
  (a deeper restructure is **planned** in `REFACTOR_PLAN.md` Phase 2).
- The `CircuitBreaker` risk gates and limits.
- Paper-trade safety semantics.

## 6. Config surface added (all in `.env.example`)
`EXTREME_PRICE_LO/HI`, `MIN_REAL_EDGE`, `MAX_TICK_AGE_S`, `FEED_PRUNE_IDLE_S`,
`MIN_VOLUME_24H`, `ARB_ALERT_COOLDOWN_S`, `ARB_ALERT_MIN_BPS`, plus the now-correct
`DESIRED_NET_MARGIN`/`DEFAULT_TAKER_FEE`. New dep: `orjson` (optional at runtime).

## 7. Test surface
`main` shipped a small integration set. Now: **101 tests** including
`test_efficiency_reliability.py`, `test_refactor_observability.py`,
`test_latency_optimization.py`, plus expanded `test_main_mock.py` /
`test_08_telemetry.py`. Full suite green.

## 8. Still open (see REFACTOR_PLAN.md)
- **Phase 2** — decouple detection from execution (inline-in-feed evaluation +
  bounded execution workers + concurrency-safe breaker).
- **Phase 3** — pre-signed orders, keep-alive sessions.
- **Phase 4** — wire `InventoryManager` for instant `mergePositions` capital
  recycling (the biggest earnings lever; needs a P&L-ownership refactor).
- **Phase 5** — typed central `Config`; pure `strategy/` layer.
