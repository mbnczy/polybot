# Full Refactor Plan — Minimum Latency, Maximum Efficiency & Performance

Branch lineage: builds on the integrated development branch
`feature/dutch-book-negrisk-integration` (which already contains the
efficiency-and-reliability, refactor-and-observability and latency-optimization
rounds). This document is the master plan; **Phase 1 is implemented on
`feature/perf-refactor-phase1`** and the rest is sequenced behind it.

---

## 0. Executive summary

The bot's job is to detect a fleeting YES+NO mispricing and fire two legs before
it disappears. After the latency work already merged (cache pre-warming + sync
cache peeks), **detection is sub-millisecond**. The remaining wins are:

1. **Architectural** — a single `strategy_loop` serializes every feed; a slow
   execution blocks all evaluation. Decouple detection (parallel, in-feed) from
   execution (bounded worker pool).
2. **Execution path** — network RTT + ECDSA signing dominate the real-world
   tick→ack time. Pre-sign/template orders and keep connections warm.
3. **Capital efficiency** — wire the existing `InventoryManager` so a filled
   set is merged back to USDC in seconds (capital velocity = more arbs/day).
4. **Micro/throughput** — `orjson`, `__slots__`, fewer allocations (Phase 1, done).

Guiding principle: **measure every phase** against a Phase-0 baseline; never
regress the 99-test suite or the circuit-breaker risk semantics; preserve paper
-trade safety.

---

## 1. Current architecture (baseline)

### 1.1 Runtime topology
`main()` runs ~8 coroutines under `asyncio.gather` (uvloop installed):

```
scanner_loop ── Gamma poll ──> FeedRegistry.add_market ──> MarketFeed (1 per market)
                                                              │ WS best-ask
                                                              ▼
                                       shared asyncio.Queue(maxsize=2048)
                                                              │
                                                   strategy_loop (SINGLE consumer)
                                                   ├─ fee/rebate peek
                                                   ├─ ArbDetector / DutchBookPricer
                                                   ├─ CircuitBreaker (size + gate)
                                                   ├─ PolyClient.execute_* (sign+submit)
                                                   └─ classify_fills + book P&L
auto_redeem_loop · heartbeat_loop · telegram_loop · sig_logger · metrics_server · tuner_loop
```

### 1.2 The hot path, hop by hop
| # | Hop | File / symbol | Cost class |
|---|-----|---------------|-----------|
| 1 | WS frame → `_loads` | `core/ws_feed.py:_dispatch` | µs (orjson: lower) |
| 2 | update best-ask, dedup | `_handle_book/_handle_price_change`, `_maybe_push_tick` | µs |
| 3 | enqueue tick | shared `asyncio.Queue` | µs + loop bounce |
| 4 | dequeue (single consumer) | `main.py:strategy_loop` | **queueing delay under burst** |
| 5 | fee/rebate | `peek_*` (sync, warm) | ~0 |
| 6 | evaluate | `ArbDetector/DutchBookPricer` (pure) | µs |
| 7 | size + risk gate | `CircuitBreaker` | µs |
| 8 | sign orders | `clob_client` ECDSA (threadpool) | **~10s of ms** |
| 9 | submit orders | CLOB REST (gather) | **network RTT, 10s–100s ms** |
| 10 | reconcile + book | `classify_fills`, breaker | µs |

**Dominant costs: hops 8–9 (execution) and hop 4 (single-consumer serialization
under load).** Hops 1–7 are already near-optimal after prior rounds.

### 1.3 Key data structures
`arb_tick: dict` → `ArbSignal` (frozen dataclass) → `ArbOrderIntent` → order
response dicts. (`ArbSignal`, `ArbLeg`, `NegRiskSignal`, `OrderBookTick` now use
`slots=True` — Phase 1.)

---

## 2. Goals, non-goals, invariants

**Goals:** minimize tick→ack latency; maximize detection throughput across many
feeds; maximize capital velocity (arbs/day per unit bankroll); keep the codebase
testable and observable.

**Non-goals:** changing the arbitrage math; adding speculative/directional
strategies; chasing micro-optimizations that don't show in the Phase-0 metrics.

**Invariants (must hold after every phase):**
- Full test suite green (currently 99).
- No order fires without passing `CircuitBreaker.check_arb`.
- A single-leg fill is never booked as profit (leg reconciliation preserved).
- Paper-trade mode performs zero network/chain writes.
- `.env`/secrets never enter git.

---

## 3. Phase 0 — Measurement & baseline  *(prereq · risk: none)*

You cannot optimize what you don't measure.

**Tasks**
- Add per-hop Histograms (extend `telemetry/metrics.py`):
  `ws_parse_seconds`, `tick_enqueue_seconds`, `eval_latency_seconds` (done),
  `sign_seconds`, `submit_seconds`, `tick_to_ack_seconds` (end-to-end).
- Add a debug-level per-tick correlation id (`condition_id` + monotonic ns) so a
  single opportunity can be traced across hops in logs.
- Add a `scripts/bench_hotpath.py` micro-benchmark: synthesise N book events,
  run them through `MarketFeed._dispatch` + `DutchBookPricer.evaluate_maker`,
  report p50/p99 ns/op (no network).

**Acceptance:** a Grafana-ready latency breakdown and a repeatable local
benchmark; numbers recorded in this doc's §9 table.

**Test plan:** unit-test that each histogram observes once per relevant event
(reuse the `REGISTRY.get_sample_value` pattern from `test_refactor_observability.py`).

---

## 4. Phase 1 — Hot-path micro-optimizations  *(risk: low · STATUS: ✅ implemented)*

**Done on `feature/perf-refactor-phase1`:**
- **`orjson` fast-parse** (`core/ws_feed.py`): optional import with a stdlib
  `json` fallback (`_loads`); `orjson` added to `requirements.txt`. ~2–5× faster
  per WS message; zero behavioural change when absent.
- **`slots=True`** on `ArbSignal`, `ArbLeg`, `NegRiskSignal` (`strategy/arbitrage.py`)
  and `OrderBookTick` (`backtest/data_handler.py`): faster attribute access, no
  per-instance `__dict__` → lower memory & GC pressure on the per-tick objects.
- Fixed an intrusive test (`test_e2e_full_pipeline.py`) that reached into
  `ArbSignal.__dict__`; now uses `dataclasses.replace` (slots-safe).

**Remaining Phase-1 candidates (planned):**
- Use `orjson.dumps` for the WS subscribe payload (cold path — low priority).
- Precompute `combined = yes_ask + no_ask` once and thread it through evaluate.
- Avoid building the `arb_tick` dict when a feed will evaluate inline (see Phase 2).

**Impact:** µs–ms per message; compounds at high feed counts. **Test:** full
suite green (99) with `orjson` active and with the fallback.

---

## 5. Phase 2 — Decouple detection from execution  *(risk: medium · biggest throughput win)*

**Problem.** One `strategy_loop` consumes all feeds and runs execution inline, so
a single order's network RTT (hops 8–9) stalls evaluation of every other market.

**Design.**
1. **Inline evaluation in `MarketFeed`.** The detector/pricer are pure and sync;
   call them the instant the best-ask updates, inside the feed's own coroutine.
   Only *actionable signals* (rare) leave the feed — not every tick.
2. **Signal channel → bounded executor.** Replace the tick queue with a small
   `asyncio.Queue` of `ArbSignal`s drained by an `asyncio.Semaphore`-bounded set
   of execution workers; execution runs as background tasks so detection never
   blocks.
3. **Concurrency-safe `CircuitBreaker`.** Wrap size+gate+book mutations in an
   `asyncio.Lock`; enforce that `check_arb` → execute → `on_fill` is atomic per
   opportunity so concurrent fills can't both pass a gate that only one should.

**Refactor surface**
- `core/ws_feed.py`: inject the detector/pricer/fee+rebate engines (or a thin
  `Evaluator` facade) into `MarketFeed`; emit `ArbSignal` instead of `arb_tick`.
- `main.py`: replace the monolithic `strategy_loop` with
  `evaluator` (in feeds) + `execution_worker(s)` draining a signal queue.
- `risk/circuit_breaker.py`: add `asyncio.Lock`; make the
  size→gate→reserve→fill sequence atomic.

**Impact:** evaluation parallelizes across all feeds; a fresh arb's path to
execution no longer waits behind another market's RTT.
**Risk:** shared risk-state concurrency. **Mitigation:** the lock + new tests for
concurrent-fill ordering and double-gate prevention.
**Acceptance:** new `test_concurrent_execution.py`: N concurrent signals respect
`MAX_POSITIONS`/daily-loss exactly; suite stays green.

---

## 6. Phase 3 — Execution-path latency  *(risk: medium · biggest real-world win)*

**Problem.** Hops 8–9 (sign + submit) dominate wall-clock latency.

**Tasks**
- **Pre-signed / templated orders.** Precompute the static portion of each
  market's order payload off the hot path (per feed admission); at fire time
  only fill price/size and sign. Investigate whether the SDK supports caching
  the EIP-712 domain/struct to cut signing cost.
- **Persistent keep-alive session.** Reuse one HTTP/2 `aiohttp` session per CLOB
  host (no per-call TLS handshake); pre-warm on the first feed admission.
  Currently some paths open a session per call (e.g. `MakerRebateEngine._fetch_*`).
- **Threadpool sizing.** `clob_client._executor` is 4 workers; size to expected
  concurrent fires and benchmark (signing is CPU-bound ECDSA).
- **Parallel submit** is already done via `asyncio.gather`.
- **Infra note (out of code scope):** co-locate the bot near Polymarket's
  ingress and use a low-latency Polygon RPC — the single biggest external lever.

**Impact:** ~30–50% lower tick→ack (data-dependent). **Test:** mock-timed
`sign_seconds`/`submit_seconds` stay within budget; paper-mode unaffected.

---

## 7. Phase 4 — Capital efficiency: wire `InventoryManager`  *(risk: med-high · biggest $ lever)*

**Problem.** `execution/inventory_manager.py` is fully built (tracks paired
fills, calls `mergePositions` to recycle a YES+NO set into **$1 USDC in seconds**)
but is **never instantiated in `main.py`**. Today capital is locked in CTF tokens
until market *resolution* (possibly weeks). Recycling it in seconds multiplies
how many arbs the same bankroll can fund.

**Blocker — P&L double-booking.** `InventoryManager._on_settled` books P&L into
the breaker, but `strategy_loop` already books it on the "both filled" branch.

**Refactor (P&L-ownership):**
1. `strategy_loop` (or the Phase-2 execution worker): on a confirmed paired fill,
   **detect + register only** — call `inventory.register_paired_fill(signal)`;
   do **not** call `breaker.on_fill` there.
2. Add `register_paired_fill(signal)` for the binary FOK/maker path (existing
   `register_matched_pair` assumes a single matchOrders `tx_hash`).
3. Move P&L booking into `_on_settled` (single owner) — booked when the merge
   confirms (paper: immediate).
4. Wire `InventoryManager(client, breaker, notifier)` into `main()`; add
   `inventory.run()` to the gather list and `_halt_tasks`.
5. **Maker-order TTL/cancel:** resting GTC maker orders that don't fill within a
   window are cancelled (frees intent; avoids fills at a moved price).

**Impact:** step-change in arbs/day per unit bankroll. **Risk:** on-chain writes
in live mode + P&L accounting. **Mitigation:** paper-mode is a no-op merge;
extensive tests that P&L is booked exactly once and capital is freed; canary
ramp.
**Acceptance:** `test_settlement_pnl.py`: paper settle books P&L once, frees the
position, integration suite green.

---

## 8. Phase 5 — Structural refactor (maintainability enabling speed)  *(risk: low)*

- **Centralized typed `Config`** (`config.py`): load+validate all env once at
  startup (replaces ~30 scattered `os.environ.get` across 6 modules). Faster
  (no repeated lookups) and safer (fail-fast on bad config).
- **Pure `strategy/` layer:** no network/IO in pricing modules so they stay
  trivially unit-testable and branch-predictable. Move `FeeEngine`/
  `MakerRebateEngine` network bits behind an interface injected at the edge.
- **Split `strategy_loop`** into composable, individually-tested stages
  (`evaluate`, `size_and_gate`, `execute`, `reconcile`) — partly done
  (`classify_fills`). Enables Phase 2 cleanly.
- **Module boundaries:** `core/` (transport), `strategy/` (pure), `risk/`,
  `execution/`, `telemetry/`.

---

## 9. Latency budget (to be filled by Phase 0)

| Hop | Baseline p50 | Baseline p99 | Target p99 |
|-----|-------------:|-------------:|-----------:|
| WS parse | _tbd_ | _tbd_ | _tbd_ |
| enqueue→dequeue | _tbd_ | _tbd_ | ~0 (Phase 2 inline) |
| evaluate | _tbd_ | _tbd_ | <50 µs |
| sign | _tbd_ | _tbd_ | Phase 3 |
| submit (RTT) | _tbd_ | _tbd_ | Phase 3 + infra |
| **tick→ack** | _tbd_ | _tbd_ | — |

---

## 10. Cross-cutting: testing, rollout, rollback

- **Testing:** every phase keeps the suite green; add the per-phase tests above;
  keep `live_arb_monitor.py` as a paper-mode end-to-end smoke.
- **Rollout:** branch → implement → suite green → benchmark vs §9 → commit →
  paper-mode validation → canary (low `MAX_ARB_PAIR_USDC` live, watch metrics) →
  ramp.
- **Rollback:** each phase is an isolated branch/commit; revert is a single
  `git revert`. Phase 4's on-chain path is feature-flagged via `PAPER_TRADE_MODE`.
- **Metrics to watch in canary:** `tick_to_ack_seconds`, `arb_half_fills_total`,
  `arb_unwind_failures_total`, `feeds_pruned_total`, `real_edge_bps`,
  cumulative P&L, USDC balance recycled.

---

## 11. Sequencing & status

| Phase | Title | Risk | Status |
|------:|-------|------|--------|
| 0 | Measurement & baseline | none | planned (next) |
| 1 | Hot-path micro-opts (orjson, slots) | low | **✅ implemented** |
| 2 | Decouple detection/execution | medium | planned |
| 3 | Execution-path latency | medium | planned |
| 4 | Capital recycling (InventoryManager) | med-high | **✅ implemented** (binary path = recycle-only; P&L stays booked at fill; NegRisk path books at settle) |
| 5 | Structural refactor / typed Config | low | planned |

Recommended order: **0 → 1 (done) → 2 → 3 → 4 → 5**, doing 0 first so 2–4 are
measured, not guessed.

---

## 12. File-by-file change map (cumulative)

| File | Phase(s) | Nature |
|------|----------|--------|
| `core/ws_feed.py` | 1,2 | orjson; inline evaluation; emit signals |
| `strategy/arbitrage.py` | 1,5 | slots; pure-layer extraction |
| `backtest/data_handler.py` | 1 | slots on OrderBookTick |
| `main.py` | 2,4,5 | evaluator+workers; InventoryManager wiring; Config |
| `risk/circuit_breaker.py` | 2 | asyncio.Lock; atomic size→gate→fill |
| `core/clob_client.py` | 3 | pre-signing, keep-alive, threadpool sizing |
| `execution/inventory_manager.py` | 4 | `register_paired_fill`; P&L ownership; TTL |
| `telemetry/metrics.py` | 0 | per-hop histograms |
| `config.py` (new) | 5 | typed central config |
| `scripts/bench_hotpath.py` (new) | 0 | local micro-benchmark |
