"""
tests/test_market_scorer.py
───────────────────────────
Unit tests for MarketScorer and its integration with MarketScanner.max_feeds.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from core.scanner import MarketScorer, MarketScanner


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _future_date(days: float) -> str:
    """ISO date string `days` days from now (UTC)."""
    dt = datetime.now(timezone.utc) + timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _past_date(days: float) -> str:
    """ISO date string `days` days ago (UTC)."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _market(
    condition_id: str = "0xABC",
    category: str = "crypto",
    volume: float = 10_000.0,
    days_until_close: float | None = 10.0,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Minimal Gamma API market dict for testing."""
    m: dict[str, Any] = {
        "conditionId":   condition_id,
        "clobTokenIds":  [f"0xYES_{condition_id}", f"0xNO_{condition_id}"],
        "active":        True,
        "closed":        False,
        "category":      category,
        "volume24hr":    volume,
    }
    if end_date is not None:
        m["endDate"] = end_date
    elif days_until_close is not None:
        m["endDate"] = _future_date(days_until_close)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# MarketScorer — score formula
# ═══════════════════════════════════════════════════════════════════════════

class TestMarketScorerFormula:
    def test_score_basic(self) -> None:
        """score = (volume × rebate) / days_to_close"""
        scorer = MarketScorer()
        m = _market(category="crypto", volume=10_000.0, days_until_close=10.0)
        # crypto rebate = 0.0144; days = 10
        expected = (10_000.0 * 0.0144) / 10.0
        assert abs(scorer.score(m) - expected) < 1e-3

    def test_score_politics_category(self) -> None:
        scorer = MarketScorer()
        m = _market(category="politics", volume=5_000.0, days_until_close=5.0)
        # politics rebate = 0.0100
        expected = (5_000.0 * 0.0100) / 5.0
        assert abs(scorer.score(m) - expected) < 1e-3

    def test_score_unknown_category_uses_default(self) -> None:
        scorer = MarketScorer()
        m = _market(category="unknown_xyz", volume=1_000.0, days_until_close=2.0)
        # DEFAULT_MAKER_REBATE = 0.0050
        expected = (1_000.0 * 0.0050) / 2.0
        assert abs(scorer.score(m) - expected) < 1e-3

    def test_score_zero_volume_is_zero(self) -> None:
        scorer = MarketScorer()
        m = _market(category="crypto", volume=0.0, days_until_close=5.0)
        assert scorer.score(m) == 0.0

    def test_score_volume_fallback_field(self) -> None:
        """Falls back to 'volume' when 'volume24hr' is absent."""
        scorer = MarketScorer()
        m = _market(volume=2_000.0, days_until_close=4.0)
        m.pop("volume24hr")
        m["volume"] = 2_000.0
        expected = (2_000.0 * 0.0144) / 4.0
        assert abs(scorer.score(m) - expected) < 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# MarketScorer — days_to_close edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestDaysToClose:
    def test_future_date_parsed(self) -> None:
        m = {"endDate": _future_date(10.0)}
        days = MarketScorer._days_to_close(m)
        # Allow a couple of seconds of float drift
        assert 9.99 < days < 10.01

    def test_expired_market_floor_is_one(self) -> None:
        """Already-expired markets should return the floor value 1.0."""
        m = {"endDate": _past_date(5.0)}
        assert MarketScorer._days_to_close(m) == 1.0

    def test_missing_end_date_floor(self) -> None:
        assert MarketScorer._days_to_close({}) == 1.0

    def test_bad_end_date_floor(self) -> None:
        m = {"endDate": "not-a-date"}
        assert MarketScorer._days_to_close(m) == 1.0

    def test_end_date_snake_case_field(self) -> None:
        """Accepts end_date (snake_case) as well as endDate (camelCase)."""
        m = {"end_date": _future_date(7.0)}
        days = MarketScorer._days_to_close(m)
        assert 6.99 < days < 7.01

    def test_days_never_below_one(self) -> None:
        """Even a market expiring in one hour must not return < 1.0."""
        m = {"endDate": _future_date(1 / 48)}  # 30 minutes from now
        assert MarketScorer._days_to_close(m) == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# MarketScanner — max_feeds=0 passthrough
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxFeedsZeroPassthrough:
    """With max_feeds=0, all markets are admitted (original behaviour)."""

    @pytest.mark.asyncio
    async def test_all_markets_admitted(self) -> None:
        admitted: list[str] = []

        async def on_added(cond_id: str, _yes: str, _no: str) -> None:
            admitted.append(cond_id)  # noqa: B023

        markets = [_market(f"0x{i:04X}") for i in range(10)]

        scanner = MarketScanner(on_market_added=on_added, max_feeds=0)
        with patch.object(scanner, "_fetch_all_active", AsyncMock(return_value=markets)):
            await scanner._scan_once()

        assert len(admitted) == 10

    @pytest.mark.asyncio
    async def test_dedup_still_works(self) -> None:
        admitted: list[str] = []

        async def on_added(cond_id: str, _yes: str, _no: str) -> None:
            admitted.append(cond_id)  # noqa: B023

        m = _market("0x0001")
        scanner = MarketScanner(on_market_added=on_added, max_feeds=0)
        with patch.object(scanner, "_fetch_all_active", AsyncMock(return_value=[m])):
            await scanner._scan_once()
            await scanner._scan_once()  # second scan — already known

        assert admitted.count("0x0001") == 1


# ═══════════════════════════════════════════════════════════════════════════
# MarketScanner — top-N selection
# ═══════════════════════════════════════════════════════════════════════════

class TestTopNSelection:
    """With max_feeds=N, only the top-N by score are admitted."""

    @pytest.mark.asyncio
    async def test_top3_admitted_from_five(self) -> None:
        admitted: list[str] = []

        async def on_added(cond_id: str, _yes: str, _no: str) -> None:
            admitted.append(cond_id)  # noqa: B023

        # Build 5 markets with clearly different scores.
        # score = volume * rebate / days;  all crypto (rebate=0.0144), same days=10
        # So score rank == volume rank.
        markets = [
            _market("0x0001", volume=1_000.0,  days_until_close=10.0),
            _market("0x0002", volume=5_000.0,  days_until_close=10.0),   # rank 2
            _market("0x0003", volume=10_000.0, days_until_close=10.0),   # rank 1
            _market("0x0004", volume=500.0,    days_until_close=10.0),
            _market("0x0005", volume=8_000.0,  days_until_close=10.0),   # rank 3
        ]

        scanner = MarketScanner(on_market_added=on_added, max_feeds=3)
        with patch.object(scanner, "_fetch_all_active", AsyncMock(return_value=markets)):
            await scanner._scan_once()

        assert set(admitted) == {"0x0003", "0x0002", "0x0005"}
        assert len(admitted) == 3

    @pytest.mark.asyncio
    async def test_fewer_candidates_than_max_feeds(self) -> None:
        """If fewer candidates than max_feeds exist, all are admitted."""
        admitted: list[str] = []

        async def on_added(cond_id: str, _yes: str, _no: str) -> None:
            admitted.append(cond_id)  # noqa: B023

        markets = [_market(f"0x{i:04X}") for i in range(3)]
        scanner = MarketScanner(on_market_added=on_added, max_feeds=10)
        with patch.object(scanner, "_fetch_all_active", AsyncMock(return_value=markets)):
            await scanner._scan_once()

        assert len(admitted) == 3

    @pytest.mark.asyncio
    async def test_dedup_respected_in_scoring_path(self) -> None:
        """Already-known markets are excluded before scoring."""
        admitted: list[str] = []

        async def on_added(cond_id: str, _yes: str, _no: str) -> None:
            admitted.append(cond_id)  # noqa: B023

        markets = [_market(f"0x{i:04X}") for i in range(5)]
        scanner = MarketScanner(on_market_added=on_added, max_feeds=5)
        with patch.object(scanner, "_fetch_all_active", AsyncMock(return_value=markets)):
            await scanner._scan_once()   # admits all 5
            await scanner._scan_once()   # all 5 now known → 0 new

        assert len(admitted) == 5
