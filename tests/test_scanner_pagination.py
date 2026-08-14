"""
tests/test_scanner_pagination.py
────────────────────────────────
Regression for the Gamma-pagination bug: `_fetch_all_active` treated the
plain-list response as complete and stopped after the first ~100 markets, so
the scanner only ever saw a small slice of the liquid universe (~30 candidates
instead of ~600). It must page by `offset` until a short/empty page.
"""

from __future__ import annotations

import pytest

import core.scanner as scanner_mod
from core.scanner import MarketScanner, _PAGE_LIMIT
from tests.mocks.fake_gamma import make_market


def _page(n: int, start: int) -> list[dict]:
    return [
        make_market("0x" + f"{start + i:064x}", f"y{start+i}", f"n{start+i}",
                    volume_24h=1_000)
        for i in range(n)
    ]


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self, content_type=None):
        return self._payload


class _FakeSession:
    """Serves full pages by offset, then a short final page (end of universe)."""

    def __init__(self, total: int):
        self.total = total
        self.offsets_seen: list[int] = []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    def get(self, url, params=None, timeout=None):
        offset = int((params or {}).get("offset", 0))
        self.offsets_seen.append(offset)
        remaining = max(0, self.total - offset)
        n = min(_PAGE_LIMIT, remaining)
        if offset >= self.total:
            return _FakeResp(None, status=422)   # past the last page
        return _FakeResp(_page(n, offset))


@pytest.mark.asyncio
async def test_fetch_all_active_pages_through_full_universe(monkeypatch):
    # 250 markets → pages at offset 0 (100), 100 (100), 200 (50 = short → stop).
    total = 250
    session = _FakeSession(total)
    monkeypatch.setattr(scanner_mod.aiohttp, "ClientSession",
                        lambda *a, **k: session, raising=True)

    scanner = MarketScanner(on_market_added=None, scan_interval=999, max_feeds=0)
    markets = await scanner._fetch_all_active()

    assert len(markets) == total, "did not page through the whole universe"
    assert session.offsets_seen[:3] == [0, 100, 200], session.offsets_seen


@pytest.mark.asyncio
async def test_fetch_all_active_single_short_page(monkeypatch):
    # Fewer than one page → one request, immediate stop (no offset spin).
    session = _FakeSession(12)
    monkeypatch.setattr(scanner_mod.aiohttp, "ClientSession",
                        lambda *a, **k: session, raising=True)

    scanner = MarketScanner(on_market_added=None, scan_interval=999, max_feeds=0)
    markets = await scanner._fetch_all_active()

    assert len(markets) == 12
    assert session.offsets_seen == [0]


@pytest.mark.asyncio
async def test_fetch_all_active_exact_multiple_stops_on_422(monkeypatch):
    # Exactly 200 markets: pages at 0 and 100 are full; offset 200 → 422 → stop.
    session = _FakeSession(200)
    monkeypatch.setattr(scanner_mod.aiohttp, "ClientSession",
                        lambda *a, **k: session, raising=True)

    scanner = MarketScanner(on_market_added=None, scan_interval=999, max_feeds=0)
    markets = await scanner._fetch_all_active()

    assert len(markets) == 200
    assert session.offsets_seen == [0, 100, 200]   # 200 → 422 → clean stop
