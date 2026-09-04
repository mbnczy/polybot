"""
core/market_titles.py
─────────────────────
Condition ID → human-readable market question.

The scanner already receives the full Gamma `/markets` payload, which carries
the `question` field ("Fed rate hike in 2026?"). That text was previously
discarded, leaving every log line and Telegram alert identified only by a
truncated hex condition ID.

This module keeps a process-local map so signals, alerts and the daily summary
can name the market. Populated as a side effect of the scan the bot already
performs, so it costs no extra API calls.

Deliberately dependency-free: core.scanner imports strategy.arbitrage, so any
registry those two share must not import either of them.

Bounded to _MAX_TITLES entries; the scanner tracks up to MAX_FEEDS (2000)
markets and prunes idle ones, so an unbounded dict would leak across a long
uptime.
"""

from __future__ import annotations

import threading

_MAX_TITLES: int = 8000

_titles: dict[str, str] = {}
_lock = threading.Lock()


def remember(condition_id: str, market: dict) -> None:
    """
    Record the question text for `condition_id`, if the payload carries one.

    Accepts the raw Gamma market dict so callers can hand over whatever they
    already fetched. Silently does nothing when the field is absent.
    """
    if not condition_id:
        return
    text = (
        market.get("question")
        or market.get("title")
        or market.get("groupItemTitle")
        or ""
    )
    text = str(text).strip()
    if not text:
        return
    with _lock:
        if len(_titles) >= _MAX_TITLES and condition_id not in _titles:
            # Cheap bound: drop an arbitrary entry rather than grow forever.
            _titles.pop(next(iter(_titles)), None)
        _titles[condition_id] = text


def title(condition_id: str, *, default: str = "") -> str:
    """Return the remembered question text, or `default` if unknown."""
    if not condition_id:
        return default
    with _lock:
        return _titles.get(condition_id, default)


def label(condition_id: str, *, width: int = 48) -> str:
    """
    Human-friendly identifier for logs: the question when known, otherwise the
    truncated condition ID. Always safe to interpolate into a log line.
    """
    text = title(condition_id)
    if not text:
        return condition_id[:16]
    if len(text) > width:
        text = text[: width - 1].rstrip() + "…"
    return f"{text} [{condition_id[:10]}]"


def count() -> int:
    """Number of titles currently remembered."""
    with _lock:
        return len(_titles)


def clear() -> None:
    """Drop every remembered title (tests)."""
    with _lock:
        _titles.clear()
