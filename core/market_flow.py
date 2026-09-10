"""
core/market_flow.py
───────────────────
Remember how much each market trades, so the strategy can estimate how long a
resting order would wait.

Whether a post-only quote fills is queue_ahead divided by the rate at which the
queue drains, and the bot has never had the second number. The WS feed carries
`book` and `price_change` only — no trade tape — and a level shrinking in a
price_change is indistinguishable from a cancel. The scanner, meanwhile,
already computes 24h volume per NegRisk group to enforce its liquidity floor
and then throws it away.

This keeps it. Same shape as market_titles: a process-wide dict filled by the
scanner, read by whoever needs it, with no lookup on the hot path.

It is an ESTIMATE and should be read as one. 24h volume says nothing about the
last five minutes, counts both sides of every trade, and does not say what
fraction executed at the touch. It is the right order of magnitude for "will
this fill inside a 45-second TTL", and nothing finer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_volume_24h: dict[str, float] = {}


def remember(condition_id: str, volume_24h: float) -> None:
    """Record a market's or group's 24h traded volume in USDC."""
    if not condition_id:
        return
    try:
        v = float(volume_24h)
    except (TypeError, ValueError):
        return
    if v >= 0.0:
        _volume_24h[str(condition_id)] = v


def get(condition_id: str) -> float | None:
    """24h volume in USDC, or None when the market was never scanned."""
    return _volume_24h.get(str(condition_id))


def clear() -> None:
    """Drop everything — tests only."""
    _volume_24h.clear()


def tracked() -> int:
    return len(_volume_24h)
