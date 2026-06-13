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
    liquidity: float = 8_000.0,
    best_bid: float = 0.47,
    best_ask: float = 0.52,
    outcome_prices: str | None = '["0.52", "0.50"]',
) -> dict[str, Any]:
    """Minimal Gamma API market dict for testing (V2 scoring fields included)."""
    m: dict[str, Any] = {
        "conditionId":   condition_id,
        "clobTokenIds":  [f"0xYES_{condition_id}", f"0xNO_{condition_id}"],
        "active":        True,
        "closed":        False,
        "category":      category,
        "volume24hr":    volume,
        "liquidityNum":  liquidity,
        "bestBid":       best_bid,
        "bestAsk":       best_ask,
    }
    if outcome_prices is not None:
        m["outcomePrices"] = outcome_prices
    if end_date is not None:
        m["endDate"] = end_date
    elif days_until_close is not None:
        m["endDate"] = _future_date(days_until_close)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# MarketScorer — score formula
# ═══════════════════════════════════════════════════════════════════════════

class TestMarketScorerFormula:
    """V2: SCORE = (L_factor × I_factor × P_eff) / sqrt(days_to_close)."""

    @staticmethod
    def _expected(volume, liquidity, bid, ask, yes, no, days) -> float:
        import core.scanner as sc
        L = (volume ** sc.V2_VOL_EXP) * (liquidity ** sc.V2_LIQ_EXP)
        mid = (ask + bid) / 2.0
        rel_spread = (ask - bid) / mid
        ineff = abs((yes + no) - 1.0)
        I = rel_spread * sc.V2_SPREAD_WEIGHT + ineff * sc.V2_INEFF_WEIGHT
        P = 1.0 / (1.0 + (volume / sc.V2_PENALTY_PIVOT) ** 2)
        return (L * I * P) / (max(days, 1.0) ** 0.5)

    def test_score_matches_formula(self) -> None:
        scorer = MarketScorer()
        m = _market(volume=5_000.0, days_until_close=16.0, liquidity=8_000.0,
                    best_bid=0.45, best_ask=0.52, outcome_prices='["0.52", "0.50"]')
        exp = self._expected(5_000.0, 8_000.0, 0.45, 0.52, 0.52, 0.50, 16.0)
        assert scorer.score(m) == pytest.approx(exp, rel=1e-6)

    def test_inefficient_midsize_beats_huge_efficient(self) -> None:
        """The whole point: a liquid-but-inefficient market outranks a giant
        efficient one."""
        scorer = MarketScorer()
        good = _market(volume=5_000.0, liquidity=8_000.0, best_bid=0.45,
                       best_ask=0.52, outcome_prices='["0.52", "0.50"]')
        huge = _market(volume=5_000_000.0, liquidity=2_000_000.0, best_bid=0.499,
                       best_ask=0.501, outcome_prices='["0.50", "0.50"]')
        assert scorer.score(good) > scorer.score(huge) * 100

    def test_excluded_below_liquidity_floor(self) -> None:
        scorer = MarketScorer()
        m = _market(volume=5_000.0, liquidity=499.0)   # < V2_MIN_LIQUIDITY (500)
        assert scorer.score(m) == 0.0

    def test_excluded_below_volume_floor(self) -> None:
        scorer = MarketScorer()
        m = _market(volume=99.0, liquidity=8_000.0)    # < V2_MIN_VOLUME_24H (100)
        assert scorer.score(m) == 0.0

    def test_higher_inefficiency_scores_higher(self) -> None:
        scorer = MarketScorer()
        base = _market(outcome_prices='["0.50", "0.50"]')           # edge 0.00
        ineff = _market(outcome_prices='["0.55", "0.50"]')          # edge 0.05
        assert scorer.score(ineff) > scorer.score(base)

    def test_volume_penalty_suppresses_giants(self) -> None:
        scorer = MarketScorer()
        small = _market(volume=10_000.0)
        giant = _market(volume=1_000_000.0)
        # Same inefficiency/liquidity shape, but the penalty crushes the giant
        # despite its larger L_factor.
        assert scorer.score(giant) < scorer.score(small)

    def test_volume_fallback_field(self) -> None:
        """Falls back to 'volume' when 'volume24hr' is absent."""
        scorer = MarketScorer()
        m = _market(volume=2_000.0)
        m.pop("volume24hr")
        m["volume"] = 2_000.0
        assert scorer.score(m) > 0.0

    def test_liquidity_string_field(self) -> None:
        """Gamma returns liquidity as a string too — must coerce."""
        scorer = MarketScorer()
        m = _market(liquidity=8_000.0)
        m.pop("liquidityNum")
        m["liquidity"] = "8000.0"
        assert scorer.score(m) > 0.0


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
