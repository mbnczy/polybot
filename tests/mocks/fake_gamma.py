"""Scripted fake of the Polymarket Gamma /markets REST endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _future_iso(days: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def make_market(
    condition_id: str,
    yes_token_id: str,
    no_token_id: str,
    *,
    category: str = "crypto",
    volume_24h: float = 10_000.0,
    days_until_close: float = 10.0,
    resolved: bool = False,
    winner: str = "",
    fee_rate: float = 0.02,
    liquidity: float = 8_000.0,
    best_bid: float = 0.47,
    best_ask: float = 0.52,
    outcome_prices: str = '["0.52", "0.50"]',
) -> dict[str, Any]:
    return {
        "conditionId":  condition_id,
        "clobTokenIds": [yes_token_id, no_token_id],
        "active":       True,
        "closed":       False,
        "category":     category,
        "volume24hr":   volume_24h,
        "endDate":      _future_iso(days_until_close),
        "resolved":     resolved,
        "winner":       winner,
        "feeRate":      fee_rate,
        # V2 scoring fields (Gamma /markets returns these on a single query).
        "liquidityNum": liquidity,
        "bestBid":      best_bid,
        "bestAsk":      best_ask,
        "outcomePrices": outcome_prices,
    }


class FakeGamma:
    """In-memory state for Gamma API mocks."""

    def __init__(self) -> None:
        self.markets: list[dict[str, Any]] = []

    def add(self, market: dict[str, Any]) -> None:
        self.markets.append(market)

    def resolve(self, condition_id: str, winner: str) -> None:
        for m in self.markets:
            if m.get("conditionId") == condition_id:
                m["resolved"] = True
                m["winner"]   = winner

    def get_all(self) -> list[dict[str, Any]]:
        return list(self.markets)

    def get_by_condition(self, condition_id: str) -> list[dict[str, Any]]:
        return [m for m in self.markets if m.get("conditionId") == condition_id]
