# Polymarket Arb Bot

Autonomous, fee-aware arbitrage bot for [Polymarket](https://polymarket.com) binary markets.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in credentials
python main.py
```

Set `PAPER_TRADE_MODE=true` in `.env` to simulate orders without spending real USDC.

---

## Architecture

Eight concurrent asyncio loops run under `asyncio.gather`:

| Loop | Purpose |
|------|---------|
| `scanner_loop` | Polls Gamma API → `FeedRegistry` |
| `strategy_loop` | Queue consumer → fee engine → arb detector → circuit breaker → dual-FOK execution |
| `auto_redeem_loop` | Polls resolved markets → CTF position redemption |
| `heartbeat_loop` | 60-second health ping (Telegram + metrics refresh) |
| `telegram_loop` | `/status` and `/halt` command listener |
| `sig_logger` | Async SQLite WAL signal logger |
| `metrics_server` | Prometheus scrape endpoint on `:8000/metrics` |
| `tuner_loop` | Real-time parameter tuning |

---

## Scoring & Feed Limiting

### What `MAX_FEEDS` does

`MAX_FEEDS` (default `50`) caps the number of new markets admitted per scanner cycle.

- `MAX_FEEDS=0` — **unlimited / passthrough**: every discovered binary market is
  registered with a WebSocket feed in arrival order. This is the original behaviour.
- `MAX_FEEDS=N` — **priority scoring**: all new candidates are scored by `MarketScorer`,
  ranked highest-first, and only the top N are admitted. Existing feeds are never evicted.

Set it in `.env`:

```
MAX_FEEDS=50   # 0 = unlimited
```

### How `MarketScorer` works

The scorer estimates **daily maker-rebate yield** for each market:

```
score = (volume_24h × maker_rebate_rate) / max(days_to_close, 1)
```

| Factor | Source |
|--------|--------|
| `volume_24h` | `volume24hr` field from Gamma API (falls back to `volume`) |
| `maker_rebate_rate` | Per-category schedule (politics 1.00 %, crypto 1.44 %, sports 0.75 %, economy 0.80 %, science 0.60 %, default 0.50 %) |
| `days_to_close` | Calendar days until `endDate`; floored at 1 to avoid division by zero |

Markets with **high daily volume**, **expiring soon**, and **in high-rebate categories**
score highest. These are precisely the conditions where arb opportunities are most frequent
and the maker rebate most meaningfully offsets taker costs.

### How to interpret the log lines

Every scan cycle emits the following at `INFO` level:

```
MarketScorer | top-5 condition=0x1234… category=politics score=8.4210 volume_24h=50000.00 days_to_close=2.3 rebate=1.00%
MarketScorer | total_candidates=312 admitted=50 (max_feeds=50)
MarketScanner | new market condition=0x1234… yes=0xABC… no=0xDEF… score=8.4210
```

- **`top-5`** lines show the highest-ranked candidates before admission.
- **`total_candidates`** is the number of unseen binary markets found this cycle.
- **`admitted`** is how many were registered (≤ `max_feeds`).
- The **`score=`** suffix on `new market` lines lets you correlate admitted markets
  with their expected daily rebate yield.

### Prometheus metrics

The scanner exposes these on the `/metrics` endpoint alongside existing bot metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `polly_scanner_feeds_fetched_total` | Counter | Total binary markets fetched from Gamma API |
| `polly_scanner_scan_duration_seconds` | Histogram | Wall time of each `_scan_once()` call |
| `polly_scanner_score_min` | Gauge | Min score seen in last scan (scoring path) |
| `polly_scanner_score_avg` | Gauge | Mean score in last scan (scoring path) |
| `polly_scanner_score_max` | Gauge | Max score in last scan (scoring path) |
| `polly_scanner_candidates` | Gauge | Unseen markets found in last scan |
| `polly_scanner_admitted` | Gauge | Markets whose callback fired in last scan |

---

## Tuning `MAX_FEEDS` for Production

Run the bundled benchmark to measure scoring overhead at different values:

```bash
# Fetch a live snapshot and benchmark (saves snapshot for reuse):
python scripts/benchmark_max_feeds.py --save-snapshot markets.json

# Re-run from local file (no network):
python scripts/benchmark_max_feeds.py --snapshot markets.json --max-feeds 10 25 50 100 200 --trials 10
```

Target: `mean_ms < 200` for the chosen value. Typical results on a 700-market snapshot:

```
max_feeds │ mean_ms │ p95_ms │ admitted/scan
──────────┼─────────┼────────┼──────────────
        0 │    2.1  │   2.4  │           700   ← passthrough (no scoring)
       10 │    3.8  │   4.1  │            10
       50 │    7.2  │   7.9  │            50
      100 │   12.6  │  13.4  │           100
```

---

## Back-testing Scorer Impact

Compare scored vs. unscored selection on historical data:

```bash
# Auto-download Kaggle dataset (requires KAGGLE_API_TOKEN in .env):
python scripts/backtest_scorer_impact.py --max-feeds 50

# Use a pre-downloaded CSV:
python scripts/backtest_scorer_impact.py --csv polymarket_markets.csv --max-feeds 50
```

Sample output:

```
Strategy A (unscored, all 700 markets):
  trades=1842  net_pnl=+$38.10  win_rate=71.2%  max_drawdown=-$22.40

Strategy B (scored, top-50):
  trades=241   net_pnl=+$19.83  win_rate=84.6%  max_drawdown=-$6.30

Delta (B − A):
  trades=-1601  net_pnl=-$18.27  win_rate=+13.4%  max_drawdown=+$16.10
  Verdict: Scored strategy shows higher win-rate and lower drawdown — scorer validated.
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FEEDS` | `50` | Markets admitted per scan (0 = unlimited) |
| `SCAN_INTERVAL` | `300` | Seconds between Gamma API polls |
| `ARB_THRESHOLD` | `0.985` | Combined ask threshold to trigger a trade |
| `STARTING_BALANCE` | `500.0` | Initial USDC for drawdown calculations |
| `PAPER_TRADE_MODE` | `false` | Simulate orders without real execution |
| `METRICS_PORT` | `8000` | Prometheus scrape port |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Risk Controls

| Rule | Limit | Action |
|------|-------|--------|
| Daily loss | −$250 USDC | Hard halt (exception) |
| Session drawdown | −15 % of starting balance | Block new orders |
| Open positions | 5 pairs max | Block new orders |
| Per-trade cost | $50 USDC combined | Reject order intent |

---

## CI

GitHub Actions runs `pytest tests/` on every push and pull request to `main`.
The workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
Merges are blocked if any test fails.
