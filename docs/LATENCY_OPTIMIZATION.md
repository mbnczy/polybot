# Latency Optimization & Capital-Recycling Plan

Branch: `feature/latency-optimization` (builds on `feature/refactor-and-observability`)

Live arbs are rare and fleeting — whoever evaluates and reacts **fastest** wins
them. This round removes the network and event-loop overhead from the hot path.

---

## L1 — Cache pre-warming on market admission

**Problem.** The first `arb_tick` for a freshly-admitted market triggered a
**network round-trip** inside the hot path: `FeeEngine.get_taker_fee` and
`MakerRebateEngine.get_maker_rebate` both hit the Gamma API on a cold cache. The
first tick is often where the juiciest (just-appeared) arb is — and it's exactly
the one we were slowest on.

**Solution.** `MarketScanner` gained an optional `on_admit(condition_id, market)`
hook, fired the moment a market is admitted. `main.py` wires it to
`prime_cache(...)` for **both** engines using the category/`feeRate` already
present in the Gamma dict — **zero extra network** (synchronous dict writes). By
the time the first tick arrives, fee and rebate are already cached.

## L2 — Synchronous cache-hit fast path

**Problem.** Even on a warm cache, `await get_taker_fee(...)` / `await
get_maker_rebate(...)` are coroutines — each `await` yields to the event loop,
adding scheduling latency to **every** tick.

**Solution.** New synchronous `peek_taker_fee()` / `peek_maker_rebate()` return
the fresh cached value (TTL-aware) or `None` without ever fetching. The hot path
peeks first and only `await`s on a genuine miss:

```python
fee_rate = fee_engine.peek_taker_fee(cid) or await fee_engine.get_taker_fee(cid)
```

After L1 the peek almost always hits, so the taker/maker/NegRisk evaluation runs
**fully synchronously** — no event-loop bounce between dequeue and decision.

## L7 — Eval-latency histogram

`polly_eval_latency_seconds` (Histogram) measures dequeue→decision time, so the
hot-path latency is now observable (buckets from 10 µs to 100 ms). Combined with
the existing `polly_ws_latency_ms`, the full tick→decision path is measurable.

---

## Tests

```
tests/test_latency_optimization.py → 9 passed (peeks TTL-aware; on_admit pre-warm)
Full suite                         → 99 passed, 0 failed
```

## Files changed

| File | Change |
|---|---|
| `strategy/arbitrage.py` | `peek_taker_fee`, `peek_maker_rebate` (sync, TTL-aware) |
| `core/scanner.py` | `on_admit(condition_id, market)` hook on both admission paths |
| `main.py` | `_prewarm_caches` wired to `on_admit`; peek-first hot path (taker/maker/NegRisk); `EVAL_LATENCY` |
| `telemetry/metrics.py` | `polly_eval_latency_seconds` histogram |

---

## Further latency ideas (not yet implemented)

- **orjson for WS parsing** — `ws_feed` parses every book/price_change with
  stdlib `json`; `orjson` is ~2–5× faster. Cheap win if the dependency is added.
- **Non-blocking execution dispatch** — today a slow execution blocks the single
  consumer from evaluating the next tick. Dispatching execution as a bounded
  background task would let evaluation continue (needs care: the `CircuitBreaker`
  is not yet concurrency-safe).
- **Persistent signed-order templates** — pre-compute as much of the order
  payload as possible per market so only price/size differ at fire time.

---

## ★ Next big earn-money lever: capital recycling (InventoryManager)

A complete `InventoryManager` already exists (`execution/inventory_manager.py`)
that, on a paired fill, calls **`mergePositions`** to burn a YES+NO set back into
**$1 USDC immediately** — but **it is not wired into `main.py`**. Today every
arb locks ~$50 of capital until the market *resolves* (possibly weeks). Merging
recycles it in **seconds**, multiplying capital velocity → far more arbs per day
on the same bankroll. This is the single highest-leverage profit improvement
left.

**Why it's deferred to its own round:** `InventoryManager._on_settled` books P&L
into the `CircuitBreaker`, but `strategy_loop` *already* books P&L on the "both
filled" path. Wiring it naively would **double-count profit**. Doing it correctly
requires a small **P&L-ownership refactor**: `strategy_loop` should only
*detect + register* the paired fill; `InventoryManager` should own
*settlement + P&L booking*. That touches the integration tests' immediate-pnl
assertions and deserves a dedicated, carefully-tested change.

**Plan:**
1. Instantiate `InventoryManager(client, breaker, notifier)` in `main()`; add
   `inventory.run()` to the `asyncio.gather` task list and `_halt_tasks`.
2. Add `register_paired_fill(signal)` for the binary FOK/maker path (the
   existing `register_matched_pair` assumes a single matchOrders tx_hash).
3. Move P&L booking out of `strategy_loop`'s "both" branch into
   `_on_settled` (single owner) to prevent double-counting.
4. Add a maker-order **TTL/cancel** so resting GTC orders that never fill are
   cancelled instead of tying up intent / filling later at a moved price.
5. Tests: paper-mode settlement books P&L exactly once; capital is freed; the
   integration suite stays green.
