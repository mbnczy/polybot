"""
Gamma returns a closed market only when asked for closed ones.

Without that, a settled cross position never saw its market settle: Brighton/Albion
resolved on 2026-10-18 and on 2026-11-04 still held one of two slots and 8.22 of 15
USDC, which blocked 586 entries in a day. The paper book in the reader had the same
query, and showed 0 resolved positions out of 73.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from execution.cross_guard import _gamma_resolution
from scripts import demo_cross_market as demo

SETTLED = {"conditionId": "0xc", "closed": True, "outcomePrices": json.dumps(["1", "0"])}
RUNNING = {"conditionId": "0xr", "closed": False, "outcomePrices": json.dumps(["0.5", "0.5"])}


def _gamma(params):
    """Behaves as the live API did on 2026-11-04 for these two markets."""
    wanted = params.get("condition_ids")
    closed = params.get("closed") == "true"
    return [m for m in (SETTLED, RUNNING)
            if m["conditionId"] == wanted and m["closed"] == closed]


class _Resp:
    status_code = 200

    def __init__(self, rows):
        self._rows = rows

    def json(self):
        return self._rows


def _patch_async(monkeypatch):
    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            return _Resp(_gamma(params))

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def test_the_guard_sees_a_settled_market_settle(monkeypatch):
    _patch_async(monkeypatch)
    assert asyncio.run(_gamma_resolution("0xc")) is True


def test_the_guard_still_waits_on_a_running_market(monkeypatch):
    _patch_async(monkeypatch)
    assert asyncio.run(_gamma_resolution("0xr")) is None


def test_the_paper_book_sees_a_settled_market_settle(monkeypatch):
    class _Http:
        def get(self, url, params=None):
            return _Resp(_gamma(params))

    monkeypatch.setattr(demo, "_http", lambda: _Http())
    assert demo._resolved_outcome("0xc") is True
    assert demo._resolved_outcome("0xr") is None
