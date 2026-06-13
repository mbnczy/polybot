# Arbitrage Duration Reporting

Branch: `feature/arb-duration`.

## What it does

Reports **how long a market's arbitrage opportunity lasts** on Telegram. A
market is "in arb" on any tick where the detector/pricer produces a signal; the
*arbitrage duration* is the wall-clock time from the first such tick until the
market transitions back to no-arb (the window closes).

Why it's useful: window length tells you whether an opportunity is actually
**capturable**. Multi-second windows are worth chasing; sub-second flickers were
already arbitraged away by faster players — so this is direct feedback on which
markets (and what latency budget) are worth your effort.

On close you get a Telegram message:

```
ARB WINDOW CLOSED (MAKER)
Market:    0x7a1fe597ac9644…
Duration:  3.42 s
Peak edge: 187.5 bps
Ticks:     5
```

## How it works

`strategy/arb_duration.py` — `ArbDurationTracker` (pure, synchronous, fully
unit-tested):

- Fed every evaluated `arb_tick` via `update(condition_id, in_arb, ts, edge_bps,
  is_maker)`.
- Opens a window on the first in-arb tick; tracks **peak edge** and **tick count**
  across the window.
- On the first no-arb tick for that market it **closes** the window and returns
  an `ArbWindow(duration_s, peak_edge_bps, ticks, is_maker_peak)`; the caller
  sends it via `TelegramNotifier.send_arb_duration(...)`.
- `flush(ts)` closes all still-open windows at shutdown (reported with
  `still_open=True`).

Durations use the monotonic clock that `ws_feed` already stamps on
`arb_tick["ts"]`, so they are accurate.

### Wiring
- `main.py` `strategy_loop`: updates the tracker on every tick (arb + no-arb),
  before the early `continue`, and fires `send_arb_duration` on close.
- `scripts/live_arb_monitor.py`: same, in the detection-only monitor (the natural
  place to *observe* durations without trading).
- `telemetry/telegram.py`: `send_arb_duration(...)` — fire-and-forget HTML alert.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `ARB_DURATION_MIN_S` | `0` | Only report windows lasting ≥ this many seconds (raise to filter sub-second flickers). Validated in `BotConfig`. |

## Limitation

The WS feed pushes a tick only when the best ask **changes** (dedup). If an arb
opens and the price then stops moving, no closing tick arrives and the window is
reported only at shutdown (via `flush`, marked "still open"). This is inherent to
event-driven feeds and is acceptable — a window with no further ticks is, by
definition, not actively changing.

## Tests
`tests/test_arb_duration.py` — open→close duration, peak/maker tracking,
multi-market independence, re-open, min-duration flicker filter, flush. Full
suite: **125 passed**.
