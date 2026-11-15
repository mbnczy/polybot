"""
Data API v2. v1 was retired on 2026-11-02 and answers everything with HTTP 410,
which left the reconciler blind and the fee calibration stale without anything
failing loudly. Rows here are copied from what the live API returned on 2026-11-04.
"""

from __future__ import annotations

import asyncio
import pathlib

import httpx
import pytest

from core import data_api
from core.clob_client import PolyClient


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


class _Http:
    """Answers each GET from a list, and records what was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return self.answers.pop(0)


def _page(rows, cursor=None):
    return _Resp({"data": rows, "pagination": {"limit": 500, "offset": 0,
                                               "has_more": cursor is not None,
                                               "next_cursor": cursor}})


# ── paging ────────────────────────────────────────────────────────────────────

def test_a_walk_follows_the_cursor_and_keeps_the_filters():
    http = _Http(_page([{"a": 1}], "CUR"), _page([{"a": 2}]))
    rows = asyncio.run(data_api.fetch_all(http, "positions", {"user": "0xw"}))
    assert rows == [{"a": 1}, {"a": 2}]
    assert http.calls[0][0] == "https://data-api.polymarket.com/v2/positions"
    assert http.calls[0][1] == {"user": "0xw", "limit": 500}
    # `limit` goes on the first page only; the anchor is sent again; no offset ever.
    assert http.calls[1][1] == {"user": "0xw", "cursor": "CUR"}


def test_half_a_wallet_is_no_answer():
    """The reconciler reads a missing position as shares that moved."""
    http = _Http(_page([{"a": 1}], "CUR"), _Resp({"error": "x"}, status=503))
    assert asyncio.run(data_api.fetch_all(http, "positions", {"user": "0xw"})) is None


def test_a_retired_route_is_no_answer():
    http = _Http(_Resp({"error": "Data API v1 has been retired"}, status=410))
    assert asyncio.run(data_api.fetch_all(http, "positions", {})) is None


def test_an_answer_without_rows_is_no_answer():
    http = _Http(_Resp({"unexpected": True}))
    assert asyncio.run(data_api.fetch_all(http, "positions", {})) is None


def test_a_walk_that_never_ends_is_no_answer():
    http = _Http(*[_page([{}], "CUR") for _ in range(30)])
    assert asyncio.run(data_api.fetch_all(http, "positions", {})) is None


def test_market_trades_walk_back_to_since_and_name_the_asset():
    calls = []

    def get(url, params):
        calls.append((url, dict(params)))
        if len(calls) == 1:
            return _page([{"timestamp": 300, "token_id": "t1"}, {"timestamp": 200, "token_id": "t2"}], "C")
        return _page([{"timestamp": 100, "token_id": "t1"}], "D")

    rows = data_api.market_trades(get, "0xc", since=150.0)
    assert [r["asset"] for r in rows] == ["t1", "t2", "t1"]
    assert calls[0] == ("https://data-api.polymarket.com/v2/trades", {"condition": "0xc", "limit": 500})
    assert calls[1][1] == {"condition": "0xc", "cursor": "C"}
    assert len(calls) == 2, "the page that reached `since` was the last one needed"


def test_market_trades_keep_what_came_before_a_failed_page():
    answers = [_page([{"timestamp": 300, "token_id": "t1"}], "C"), None]
    assert len(data_api.market_trades(lambda u, p: answers.pop(0), "0xc")) == 1


def test_no_live_code_still_calls_v1():
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for f in root.rglob("*.py"):
        if {".venv", "tests"} & set(f.parts):
            continue
        text = f.read_text()
        for route in ("positions", "activity", "trades", "value"):
            if f"data-api.polymarket.com/{route}" in text:
                offenders.append(f"{f.relative_to(root)}: {route}")
    assert not offenders, offenders


# ── the client ────────────────────────────────────────────────────────────────

POSITION = {
    "proxy_wallet": "0x3bfc05406b16eb3eec686df6c0ab47ba321aa390",
    "token_id": "115612176759954686609185778731175618682728838103045563263916988149010940588876",
    "condition_id": "0x6e7b8ccfec3ff88324ad266e841e8c8925bb4dc294c66ba15bffc3b883876863",
    "current_size": 5.0, "avg_price": 0.54, "current_price": 0.505, "current_value": 2.525,
    "total_size": 5.0, "total_pnl": -0.175, "status": "OPEN", "redeemable": False,
    "negative_risk": False, "title": "Abha Saudi Club vs. Al Shabab Saudi Club: O/U 0.5",
    "outcome": "Over",
}
LOSER = {**POSITION, "token_id": "114616", "condition_id": "0xcf27", "current_size": 5.2497,
         "current_price": 0.0, "current_value": 0.0, "status": "REDEEMABLE", "redeemable": True,
         "title": "Galway United FC vs. Shelbourne FC: O/U 7.5"}
REDEEM = {"proxy_wallet": "0x3bfc", "timestamp": 1793778468, "condition_id": "0x86bf",
          "type": "REDEEM", "size": 10.23091, "usdc_size": 10.23091,
          "transaction_hash": "0xe0972c59", "price": 0.0, "token_id": "108587",
          "side": "", "outcome_index": 1, "outcome": "No",
          "title": "Will the Republican Party control the Senate after the 2026 Midterm elections?"}


@pytest.fixture
def client(monkeypatch):
    class _Client(PolyClient):
        trading_wallet = "0xw"

    c = _Client.__new__(_Client)
    http = _Http()

    class _Ctx:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return http

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Ctx)
    c.http = http
    return c


def test_positions_read_the_v2_names(client):
    client.http.answers = [_page([POSITION, LOSER])]
    rows = asyncio.run(client.open_positions_detail())
    assert rows[0]["size"] == 5.0 and rows[0]["cur_price"] == 0.505 and rows[0]["value"] == 2.525
    assert rows[0]["asset"] == POSITION["token_id"]
    assert rows[0]["condition_id"] == POSITION["condition_id"]
    assert rows[0]["redeemable"] is False
    assert rows[1]["redeemable"] is True and rows[1]["value"] == 0.0


def test_positions_ask_for_dust_too(client):
    """v2's default floor of 0.1 shares would read a position crossing it as a trade."""
    client.http.answers = [_page([])]
    asyncio.run(client.open_positions_detail())
    query = client.http.calls[0][1]
    assert query["filter_type"] == "TOKENS" and query["filter_amount"] == 0.0001
    assert "sizeThreshold" not in query


def test_a_failed_positions_read_is_none_not_empty(client):
    client.http.answers = [_Resp({}, status=503)]
    assert asyncio.run(client.open_positions_detail()) is None


def test_redemptions_read_the_feed(client):
    client.http.answers = [_page([REDEEM])]
    rows = asyncio.run(client.recent_redemptions(1793778000.4))
    assert rows == [{"title": REDEEM["title"][:80], "outcome": "No", "usdc": 10.23091,
                     "tx": "0xe0972c59", "condition_id": "0x86bf", "ts": 1793778468}]
    query = client.http.calls[0][1]
    assert query["type"] == "REDEEM" and query["start"] == 1793778000 and query["user"] == "0xw"


def test_a_failed_redemption_read_is_none(client):
    client.http.answers = [_Resp({}, status=503)]
    assert asyncio.run(client.recent_redemptions(1.0)) is None
