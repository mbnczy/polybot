"""
telemetry/fill_log.py
─────────────────────
Append-only record of what happens to every maker leg we post.

Why
───
The bot has never completed a bundle. The working theory is queue position:
63% of live legs sit on a one-tick spread, where `ask - tick` IS the best bid,
so a post-only quote joins the back of a queue instead of leading it. That is a
theory, and the honest way to settle it is a week of outcomes rather than
another argument.

Each line records one leg's fate with the conditions it faced, so the question
"which legs actually fill" can be answered by price band, tick size, spread and
queue depth rather than guessed at.

Deliberately dumb: one JSON object per line, opened and closed per write, no
buffering. A telemetry file must never be able to lose the bot money, and the
volumes here are trivial — a few hundred lines a day.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

FILL_LOG_PATH: str = os.environ.get("FILL_LOG_PATH", "fill_log.jsonl")
FILL_LOG_ENABLED: bool = os.environ.get(
    "FILL_LOG_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")


def record(
    *,
    path:      str,
    outcome:   str,
    price:     float,
    tick:      float | None = None,
    best_bid:  float | None = None,
    best_ask:  float | None = None,
    size:      float = 0.0,
    matched:   float = 0.0,
    rested_s:  float = 0.0,
    condition_id: str = "",
    queue_ahead: float | None = None,
    reachable:   bool | None = None,
    expected_fill_s: float | None = None,
) -> None:
    """
    Record one maker leg's outcome. Never raises: a telemetry failure must not
    interrupt trading.

    `outcome` is one of filled / partial / expired / cancelled.

    `queue_ahead` is the shares already resting at the price we joined, and it
    is the number this file was written for. Leading the book is not the same
    question as being reachable: a quote can lead by a tick on a book so thick
    behind it that nothing trades, and a quote can join a queue of forty and
    fill in a minute. `leads` records the price relationship; `queue_ahead`
    records the wait.
    """
    if not FILL_LOG_ENABLED:
        return
    row = {
        "ts":        round(time.time(), 3),
        "path":      path,                       # "pair" | "negrisk"
        "outcome":   outcome,
        "condition": condition_id[:24],
        "price":     round(price, 4),
        "tick":      tick,
        "best_bid":  best_bid,
        "best_ask":  best_ask,
        # Whether our quote LED the book or merely joined it — the whole
        # question this file exists to answer.
        "leads":     (None if best_bid is None else bool(price > best_bid + 1e-12)),
        "spread_ticks": (
            None if (best_bid is None or best_ask is None or not tick)
            else round((best_ask - best_bid) / tick, 1)
        ),
        # Shares resting at the price we joined. None = the exchange stated a
        # price without its depth (a batched price_change), which is distinct
        # from 0 — an empty level.
        "queue_ahead": queue_ahead,
        # What the detector predicted at quote time. Comparing this against
        # `outcome` is the whole point: it is how we find out whether the
        # reachability rule is any good.
        "reachable":   reachable,
        # Seconds the queue was predicted to take. Recorded beside rested_s and
        # the outcome so the prediction can be scored against what happened.
        "expected_fill_s": (
            None if expected_fill_s is None
            else (None if expected_fill_s == float("inf")
                  else round(expected_fill_s, 1))
        ),
        "size":      round(size, 2),
        "matched":   round(matched, 2),
        "rested_s":  round(rested_s, 1),
    }
    try:
        with open(FILL_LOG_PATH, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.debug("fill_log write failed: %s", exc)
