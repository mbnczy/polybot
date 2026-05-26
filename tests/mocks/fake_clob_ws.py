"""Fake Polymarket CLOB WebSocket — scripted message generator."""

from __future__ import annotations

import asyncio
import json
from typing import Any


class FakeWSMessage:
    """Mimics aiohttp.WSMessage."""

    def __init__(self, data: str) -> None:
        import aiohttp
        self.type = aiohttp.WSMsgType.TEXT
        self.data = data


class FakeWebSocket:
    def __init__(self, script: list[dict | list]) -> None:
        self._script  = list(script)
        self._sent: list[str] = []

    async def send_str(self, payload: str) -> None:
        self._sent.append(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._script:
            # Hold open forever so the consumer awaits cancellation.
            await asyncio.Event().wait()
        msg = self._script.pop(0)
        return FakeWSMessage(json.dumps(msg))


class _WSConnectCtx:
    def __init__(self, ws: FakeWebSocket) -> None:
        self._ws = ws

    async def __aenter__(self) -> FakeWebSocket:
        return self._ws

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class FakeAiohttpSession:
    """Replacement for `aiohttp.ClientSession()` used by ws_feed.MarketFeed."""

    def __init__(self, ws: FakeWebSocket) -> None:
        self._ws = ws

    async def __aenter__(self) -> "FakeAiohttpSession":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    def ws_connect(self, *_args: Any, **_kwargs: Any) -> _WSConnectCtx:
        return _WSConnectCtx(self._ws)


def make_book_message(asset_id: str, ask_price: float) -> dict:
    return {
        "event_type": "book",
        "asset_id":   asset_id,
        "asks":       [{"price": f"{ask_price:.4f}", "size": "100"}],
        "bids":       [{"price": f"{max(0.01, ask_price - 0.01):.4f}", "size": "100"}],
    }


def make_price_change_message(asset_id: str, ask_price: float) -> dict:
    return {
        "event_type": "price_change",
        "asset_id":   asset_id,
        "price":      f"{ask_price:.4f}",
        "side":       "SELL",
        "size":       "100",
    }
