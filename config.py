"""
config.py
─────────
Single, typed, **validated** source of truth for all env-driven bot settings.

Why this exists
───────────────
Configuration was previously read with ~30 scattered `os.environ.get(...)` calls
across six modules, with no validation — a typo like `DESIRED_NET_MARGIN=1.5`
(150 %) or `MAX_FEEDS=-3` would silently misconfigure the bot. `BotConfig.from_env()`
reads everything in one place and **fails fast at startup** on an invalid value,
so misconfiguration can never reach the hot path.

Usage (in main()):
    from config import BotConfig
    cfg = BotConfig.from_env()        # raises ValueError on bad config
    logger.info("Config: %s", cfg)

The strategy/risk modules keep their own constant defaults; this object is the
entry-point's validated view and the place to see every knob at a glance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from strategy.arbitrage import (
    DEFAULT_TAKER_FEE,
    DESIRED_NET_MARGIN,
    EXTREME_PRICE_HI,
    EXTREME_PRICE_LO,
    MAX_TAKER_FEE,
    MIN_REAL_EDGE,
)


def _f(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid float") from exc


def _i(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid int") from exc


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Validated, immutable snapshot of all env-driven settings."""

    # Capital / risk
    starting_balance:   float
    # Strategy thresholds
    desired_net_margin: float
    default_taker_fee:  float
    extreme_lo:         float
    extreme_hi:         float
    min_real_edge:      float
    max_tick_age_s:     float
    # Scanner / feeds
    scan_interval:      float
    max_feeds:          int
    prune_idle_s:       float
    min_volume_24h:     float
    # Execution concurrency (Phase 2 non-blocking dispatch)
    exec_concurrency:   int
    # Arb-duration reporting: only report windows lasting at least this long
    arb_duration_min_s: float
    # Single-market ENV seed (optional, backward-compatible)
    yes_token_id:       str
    no_token_id:        str
    condition_id:       str

    @classmethod
    def from_env(cls) -> "BotConfig":
        cfg = cls(
            starting_balance   = _f("STARTING_BALANCE",   500.0),
            desired_net_margin = _f("DESIRED_NET_MARGIN", DESIRED_NET_MARGIN),
            default_taker_fee  = _f("DEFAULT_TAKER_FEE",  DEFAULT_TAKER_FEE),
            extreme_lo         = _f("EXTREME_PRICE_LO",   EXTREME_PRICE_LO),
            extreme_hi         = _f("EXTREME_PRICE_HI",   EXTREME_PRICE_HI),
            min_real_edge      = _f("MIN_REAL_EDGE",      MIN_REAL_EDGE),
            max_tick_age_s     = _f("MAX_TICK_AGE_S",     2.0),
            scan_interval      = _f("SCAN_INTERVAL",      300.0),
            max_feeds          = _i("MAX_FEEDS",          50),
            prune_idle_s       = _f("FEED_PRUNE_IDLE_S",  600.0),
            min_volume_24h     = _f("MIN_VOLUME_24H",     0.0),
            exec_concurrency   = _i("EXEC_CONCURRENCY",   6),
            arb_duration_min_s = _f("ARB_DURATION_MIN_S", 0.0),
            yes_token_id       = os.environ.get("YES_TOKEN_ID", "").strip(),
            no_token_id        = os.environ.get("NO_TOKEN_ID",  "").strip(),
            condition_id       = os.environ.get("CONDITION_ID", "").strip(),
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if not (0.0 < self.desired_net_margin < 1.0):
            raise ValueError(
                f"DESIRED_NET_MARGIN must be in (0, 1); got {self.desired_net_margin}"
            )
        if not (0.0 <= self.default_taker_fee <= MAX_TAKER_FEE):
            raise ValueError(
                f"DEFAULT_TAKER_FEE must be in [0, {MAX_TAKER_FEE}]; "
                f"got {self.default_taker_fee}"
            )
        if not (0.0 <= self.extreme_lo < self.extreme_hi <= 1.0):
            raise ValueError(
                f"EXTREME_PRICE band must satisfy 0 ≤ lo < hi ≤ 1; "
                f"got [{self.extreme_lo}, {self.extreme_hi}]"
            )
        if self.min_real_edge < 0.0:
            raise ValueError(f"MIN_REAL_EDGE must be ≥ 0; got {self.min_real_edge}")
        if self.max_tick_age_s < 0.0:
            raise ValueError(f"MAX_TICK_AGE_S must be ≥ 0; got {self.max_tick_age_s}")
        if self.starting_balance <= 0.0:
            raise ValueError(f"STARTING_BALANCE must be > 0; got {self.starting_balance}")
        if self.max_feeds < 0:
            raise ValueError(f"MAX_FEEDS must be ≥ 0; got {self.max_feeds}")
        if self.scan_interval <= 0.0:
            raise ValueError(f"SCAN_INTERVAL must be > 0; got {self.scan_interval}")
        if self.prune_idle_s < 0.0:
            raise ValueError(f"FEED_PRUNE_IDLE_S must be ≥ 0; got {self.prune_idle_s}")
        if self.min_volume_24h < 0.0:
            raise ValueError(f"MIN_VOLUME_24H must be ≥ 0; got {self.min_volume_24h}")
        if self.exec_concurrency < 1:
            raise ValueError(f"EXEC_CONCURRENCY must be ≥ 1; got {self.exec_concurrency}")
        if self.arb_duration_min_s < 0.0:
            raise ValueError(
                f"ARB_DURATION_MIN_S must be ≥ 0; got {self.arb_duration_min_s}"
            )
