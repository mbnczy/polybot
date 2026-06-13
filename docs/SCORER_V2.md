# V2 Market Scorer

Branch: `feature/scorer-v2` → merged into the development trunk.
Spec: `docs/Scorer machine.docx`.

## Problem

The original scorer ranked markets by `volume24h × maker_rebate / days` — i.e. it
preferred the **biggest** markets. In practice those (US presidential election,
marquee events) carry huge capital but are **extremely efficient**: tight
spreads, professional market makers and low-latency algos leave almost no real
arbitrage. We don't want the biggest markets — we want markets **liquid enough
for $100–500 positions but not yet fully efficient**.

## Solution — three independent components

```
SCORE = (L_factor × I_factor × P_eff) / sqrt(days_to_close)
```

| Component | Formula | Purpose |
|---|---|---|
| **Liquidity** `L_factor` | `volume24h^0.3 × liquidity^0.2` | How big a position fits without slippage. **Dampened exponents** so giant markets don't auto-dominate. |
| **Inefficiency** `I_factor` | `rel_spread×100 + ineff_edge×1000` | The market's "dumbness" / arb-propensity. `rel_spread=(ask−bid)/mid`; `ineff_edge=｜yes+no−1.0｜`. Direct arb dislocations weighted 10×. |
| **Penalty** `P_eff` | `1 / (1 + (volume24h/100000)^2)` | Suppresses too-big / too-efficient markets — at ~$100k 24h volume the score is roughly halved, at ~$200k about a fifth. |
| **Time** | `/ sqrt(days_to_close)` | Near-expiry markets stay favoured without dominating the ranking. |

### Hard exclusions (score → 0, never admitted)
- `liquidity < $500` — too thin to trade.
- `volume24h < $100` — effectively dead.

## Critical engineering constraint — scanner stays fast

Every input is read from a **single Gamma `/markets` response**:
`volume24hr`, `liquidityNum` (or `liquidity`), `bestBid`/`bestAsk`/`spread`, and
`outcomePrices` (`'["0.51","0.49"]'` → `ineff_edge = |0.51+0.49−1|`). **No
per-market CLOB `/book` calls are made during scanning** — issuing thousands of
them would be slow and rate-limit-prone. Order-book depth, fill probability and
live arbitrage analysis happen only for the admitted **top-N** feeds, over the
WebSocket layer (`ws_feed.py`). The scanner therefore concentrates resources on
the genuinely promising markets while remaining fast.

## Behaviour (observed)

| Market | vol / liq / prices | V2 score |
|---|---|---|
| mid-size, inefficient | 5k / 8k / 0.52+0.50 (edge .02, spread .07) | **667** |
| huge, efficient | 5M / 2M / 0.50+0.50 (edge 0, spread .002) | **0.07** |
| illiquid | 50 / 100 | **0 (excluded)** |

The mid-size inefficient market outranks the huge efficient one by ~9000×, and
the illiquid one is excluded entirely — exactly the intended selection.

## Implementation

- `core/scanner.py`:
  - `MarketScorer.score()` rewritten to the V2 formula.
  - Helpers `_volume_24h`, `_liquidity`, `_relative_spread`, `_inefficiency_edge`
    (robust to Gamma's float/string field variants).
  - Tunable module constants `V2_VOL_EXP`, `V2_LIQ_EXP`, `V2_MIN_LIQUIDITY`,
    `V2_MIN_VOLUME_24H`, `V2_SPREAD_WEIGHT`, `V2_INEFF_WEIGHT`, `V2_PENALTY_PIVOT`.
  - Scanner admission skips any market scoring `≤ 0` (excluded) regardless of
    free feed slots.
- `_days_to_close` (floored at 1.0) is reused as the `sqrt` divisor.
- Maker rebate is **no longer part of market selection** (it still applies to
  P&L and the maker pricing path) — selection is now purely about capturable
  arbitrage potential.

## Tests
`tests/test_market_scorer.py` rewritten for V2 (formula, component weighting,
exclusions, giant-suppression, string-field coercion). Fakes (`make_market` and
inline fixtures) updated to carry the Gamma V2 fields. Full suite: **118 passed**.
