# Efficiency & Reliability Improvements

Branch: `feature/efficiency-and-reliability`

Goal: make the bot **more efficient** (less wasted compute / fewer wasted feeds)
and bring in returns **more reliably** (never book phantom profit, never trade
on stale or low-quality signals). Every change is config-gated and unit-tested.

> TL;DR of the strategy reality this work is grounded in: on live order books a
> genuine YES+NO arb is **rare** (asks almost always sum to > 1.0). The levers
> that actually matter are therefore (1) not *losing* money on half-fills or
> stale quotes, (2) not wasting resources on dead markets, and (3) not flooding
> yourself with rebate-inflated "signals" from near-resolved markets. That is
> exactly what these changes target.

---

## 1. Leg-fill reconciliation + unwind  *(reliability — highest impact)*

**Problem.** `execute_arb_pair` fires both FOK legs concurrently but the old
`strategy_loop` booked `net_profit` **unconditionally**, assuming both filled.
If only one leg fills you are left holding a **naked, directional position** —
the opposite of risk-free — and it was being recorded as guaranteed profit.

**Solution.**
- `core/clob_client.py`: `classify_fills(yes_resp, no_resp) → {both, yes_only, no_only, none}`
  plus `_resp_filled()` (`status ∈ {matched, filled, paper}`).
- `core/clob_client.py`: `unwind_leg(token_id, size)` market-sells a stranded leg.
- `main.py` `strategy_loop`:
  - `both` → book profit + Telegram execution alert (unchanged happy path).
  - `yes_only` / `no_only` → **flatten the naked leg** via `unwind_leg`, alert,
    **book no profit**. If the unwind itself fails → `send_critical_error`
    ("MANUAL INTERVENTION REQUIRED").
  - `none` → log only (maker GTC orders may be resting; P&L deferred to fill
    confirmation rather than assumed).

**Tests.** `TestClassifyFills` (7 cases) + the 3 mock-pipeline integration tests.

---

## 2. Signal-quality guards  *(reliability)*

**Problem.** The maker path fired on **near-resolved** markets (e.g. YES≈0.02 /
NO≈0.98) where the headline edge is almost entirely the **maker-rebate subsidy**
on a thin book that rarely fills — not a durable price gap. Observed live: 2
"arbs" at combined 0.999 whose *real* gap was only 10 bps, the rest rebate.

**Solution (`strategy/arbitrage.py`).**
- **Extreme-price filter** (`EXTREME_PRICE_LO=0.05`, `EXTREME_PRICE_HI=0.95`):
  skip a market unless **both** legs' asks sit inside the band. Applied to
  `DutchBookPricer.evaluate_maker` **and** `ArbDetector.evaluate`.
- **Minimum real edge** (`MIN_REAL_EDGE`, default `0.0` = off): when > 0, the
  maker path requires `real_edge = 1 − (yes_ask + no_ask) ≥ MIN_REAL_EDGE`, so
  the rebate is *upside*, not the basis for entry. Recommended: `0.002` (20 bps).

**Tests.** `TestExtremePriceGuard` (4) + `TestMinRealEdgeGuard` (3).

---

## 3. Feed global cap + stale-feed pruning  *(efficiency — fixes a real leak)*

**Problem.** `max_feeds` was a **per-scan** admission limit, not a global cap.
Each re-scan admitted up to `max_feeds` *new* markets, so active feeds grew
without bound across scans (observed live: 50 → 94 in ~15 min). Unbounded
WebSocket connections = wasted CPU, memory, and bandwidth.

**Solution.**
- `core/scanner.py` `FeedRegistry(queue, max_feeds=0)`: hard **global** ceiling.
  `add_market` now returns `bool` (`False` when at cap, so the scanner keeps the
  market eligible for a later scan instead of marking it permanently seen).
- `FeedRegistry.prune_stale(max_idle_s) → [condition_id]`: drops feeds that
  produced **no two-sided quote** within `max_idle_s` (dead / illiquid / one-sided),
  freeing the slot.
- `core/ws_feed.py` `MarketFeed.idle_seconds()` + `_last_tick_monotonic`: liveness
  tracking (a feed that never produces a tick ages from construction → prunable).
- `core/scanner.py` `MarketScanner`: prunes before admitting, computes
  `free_slots = max_feeds − active_count`, and only marks a market known on a
  **successful** add.

**Config.** `MAX_FEEDS` (global cap), `FEED_PRUNE_IDLE_S` (default 600, 0=off).

**Tests.** `test_feed_registry_global_cap`, `test_feed_registry_prune_stale`,
`test_feed_registry_prune_disabled`.

---

## 4. Liquidity / 24h-volume floor  *(efficiency)*

**Problem.** The rebate-yield scorer admitted illiquid markets (live: many
0.999/0.999 books with no real ask-side liquidity) — a wasted feed slot.

**Solution (`core/scanner.py`).** `MarketScanner(min_volume_24h=…)` skips
candidates whose 24h volume is below the floor before scoring/admission.

**Config.** `MIN_VOLUME_24H` (default 0 = off).  **Tests.** `test_scanner_volume_floor`.

---

## 5. Tick dedup + stale-tick freshness guard  *(efficiency + returns)*

**Problem.** The CLOB WS emits an event on *every* book/price_change even when
the best ask is unchanged — re-evaluated needlessly and could re-alert. Also,
acting on a quote that has been sitting in the queue risks **adverse selection**
(the price has already moved).

**Solution.**
- `core/ws_feed.py`: `_maybe_push_tick` skips enqueueing when `(yes_ask, no_ask)`
  is unchanged since the last push (`_last_pushed`).
- `main.py`: **freshness guard** — skip a tick when
  `time.monotonic() − tick["ts"] > MAX_TICK_AGE_S` (default 2.0s, 0=off).

> ⚠️ Bug fixed during this work: the latency/age math must use `time.monotonic()`
> (the same clock `ws_feed` stamps with), **not** `loop.time()`, which differs
> under uvloop and made every tick look stale. See §"Bugs found & fixed".

**Tests.** `test_marketfeed_dedup_and_liveness`.

---

## 6. Real-time arb-detection Telegram alerts  *(observability)*

`telemetry/telegram.py` `send_arb_detected(...)` fires the moment an arb is
detected — **independent of execution** (so breaker-blocked opportunities are
still surfaced). Wired into `strategy_loop` and the standalone
`scripts/live_arb_monitor.py` (detection + alerts, **no trading**).

**Throttling** (essential — detection can be frequent): per-market cooldown
`ARB_ALERT_COOLDOWN_S` (default 300) + minimum-edge floor `ARB_ALERT_MIN_BPS`
(default 0). **Tests.** `TestArbDetectedThrottle` (4).

---

## 7. Config hygiene

- `.env.example`: documented that **`ARB_THRESHOLD` is deprecated / unread** —
  the live trigger is `DESIRED_NET_MARGIN` (mapping note: `0.985 ≈ 0.015`).
- All new knobs added to `.env.example` and wired through `main.py` +
  `scripts/live_arb_monitor.py`.

---

## Configuration reference (new / clarified)

| Env var | Default | Purpose |
|---|---|---|
| `DESIRED_NET_MARGIN` | 0.005 | Live trigger (the bot reads THIS, not `ARB_THRESHOLD`) |
| `EXTREME_PRICE_LO` / `_HI` | 0.05 / 0.95 | Skip near-resolved markets (both legs must be in band) |
| `MIN_REAL_EDGE` | 0.0 (off) | Require genuine pre-rebate gap on the maker path |
| `MAX_TICK_AGE_S` | 2.0 | Drop stale quotes (adverse-selection guard); 0=off |
| `MAX_FEEDS` | 50 | **Global** active-feed cap (no longer per-scan) |
| `FEED_PRUNE_IDLE_S` | 600 | Drop feeds with no two-sided quote in this window; 0=off |
| `MIN_VOLUME_24H` | 0 (off) | Skip dead books below this 24h volume |
| `ARB_ALERT_COOLDOWN_S` | 300 | Per-market Telegram detection-alert cooldown |
| `ARB_ALERT_MIN_BPS` | 0 | Suppress detection alerts below this edge |

**Recommended "reliable returns" profile:**
`MIN_REAL_EDGE=0.002`, `MIN_VOLUME_24H=10000`, `FEED_PRUNE_IDLE_S=600`,
`MAX_TICK_AGE_S=2.0`, keep `EXTREME_PRICE_*` at defaults.

---

## Bugs found & fixed (during testing)

1. **Freshness-guard clock mismatch** — used `loop.time()` vs `ws_feed`'s
   `time.monotonic()`; under uvloop these differ, making every tick "stale".
   Fixed to `time.monotonic()` in `main.py`.
2. **Mock gap** — `FakeTelegramNotifier` lacked `send_arb_detected`, so the new
   detection call crashed `strategy_loop` in the integration tests. Added the
   method to `tests/mocks/fake_telegram.py`.

---

## Test status

```
tests/test_efficiency_reliability.py  → 23 passed   (all new features)
tests/integration/test_main_mock.py   →  4 passed   (full pipeline incl. reconciliation)
Full suite                            → 75 passed, 4 failed
```

The **4 failures are pre-existing** in `tests/test_market_scorer.py::TestDaysToClose`
(they expect `_days_to_close` to floor at 1.0 while the committed code floors at
0.0). They fail identically on a clean `HEAD` worktree and were **not introduced
by this work** — left untouched because changing `_days_to_close` would break the
scanner's `<= 0` expired-market skip. Flagged here for a separate fix.

---

## Files changed

| File | Change |
|---|---|
| `strategy/arbitrage.py` | extreme-price + min-real-edge guards |
| `core/clob_client.py` | `classify_fills`, `_resp_filled`, `unwind_leg` |
| `core/scanner.py` | global feed cap, `prune_stale`, volume floor, cap-aware admission |
| `core/ws_feed.py` | tick dedup, `idle_seconds` liveness |
| `main.py` | leg reconciliation, freshness guard, detection alert, env wiring |
| `telemetry/telegram.py` | `send_arb_detected` (throttled) |
| `scripts/live_arb_monitor.py` | persistent detection-only monitor (no trading) |
| `.env.example` | documented new knobs + deprecated `ARB_THRESHOLD` |
| `tests/test_efficiency_reliability.py` | 23 new unit tests |
| `tests/mocks/fake_telegram.py` | `send_arb_detected` mock |
