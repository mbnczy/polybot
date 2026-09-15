"""
The reader fetches the whole one-week window, not its first page.

On 2026-09-15 the 400 soonest-ending markets covered 90 minutes — temperature
and five-minute crypto markets — and the first football market sat at position
1,390. The window holds ~40,000 markets; one Gamma query pages at most 2,000.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import scripts.demo_cross_market as demo


class _Resp:
    def __init__(self, rows):
        self.status_code, self._rows, self.text = 200, rows, ""

    def json(self):
        return self._rows


class _FakeGamma:
    """Filters by end date, orders by it, pages with limit/offset — like Gamma."""

    def __init__(self, markets):
        self.markets = sorted(markets, key=lambda m: m["endDate"])
        self.requests = 0

    def get(self, url, params):
        self.requests += 1
        lo, hi = params["end_date_min"], params["end_date_max"]
        rows = [m for m in self.markets if lo <= m["endDate"] <= hi]
        off, lim = int(params["offset"]), int(params["limit"])
        return _Resp([dict(m, description="x" * 500) for m in rows[off:off + lim]])


def _market(i, when, title=None):
    return {"conditionId": f"0x{i:06x}", "question": title or f"Team {i} vs. Team {i+1}: O/U 2.5",
            "endDate": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "outcomes": '["Over", "Under"]',
            "clobTokenIds": '["1", "2"]', "bestAsk": 0.55, "events": [{"id": i // 10, "title": "big"}]}


def test_a_crowded_window_is_split_until_every_market_is_fetched(monkeypatch):
    now = datetime.now(timezone.utc)
    markets = [_market(i, now + timedelta(hours=2, minutes=i // 3)) for i in range(1500)]
    markets += [_market(10_000 + i, now + timedelta(days=3, minutes=i)) for i in range(120)]
    fake = _FakeGamma(markets)
    monkeypatch.setattr(demo, "_thread_client", lambda: fake)
    monkeypatch.setattr(demo, "_GAMMA_MAX_OFFSET", 300)          # a small ceiling forces splits
    got = demo.fetch_window(7, min_hours=1.0, concurrency=3)
    assert sorted(m["conditionId"] for m in got) == sorted(m["conditionId"] for m in markets)


def test_markets_are_slimmed_to_the_fields_that_are_read(monkeypatch):
    now = datetime.now(timezone.utc)
    fake = _FakeGamma([_market(1, now + timedelta(hours=5))])
    monkeypatch.setattr(demo, "_thread_client", lambda: fake)
    (m,) = demo.fetch_window(1, min_hours=1.0, concurrency=1)
    assert "description" not in m
    assert m["events"] == [{"id": "0"}]
    assert m["outcomes"] == '["Over", "Under"]' and m["bestAsk"] == 0.55


def test_title_series_can_be_excluded(monkeypatch):
    now = datetime.now(timezone.utc)
    fake = _FakeGamma([_market(1, now + timedelta(hours=5)),
                       _market(2, now + timedelta(hours=5), "Bitcoin Up or Down - 8:50AM-8:55AM ET")])
    monkeypatch.setattr(demo, "_thread_client", lambda: fake)
    got = demo.fetch_window(1, min_hours=1.0, concurrency=1, exclude="Up or Down")
    assert [m["conditionId"] for m in got] == ["0x000001"]


def test_pricing_reads_the_snapshot_not_the_book():
    assert demo.snapshot_yes_ask({"bestAsk": "0.42"}) == 0.42
    assert demo.snapshot_yes_ask({"bestAsk": 1.0}) is None
    assert demo.snapshot_yes_ask({}) is None


def test_a_transient_gamma_error_is_retried_not_skipped(monkeypatch):
    """A 5xx mid-slice used to end the slice: its later pages were never fetched."""
    now = datetime.now(timezone.utc)
    fake = _FakeGamma([_market(i, now + timedelta(hours=3)) for i in range(250)])
    calls = {"n": 0}
    real_get = fake.get

    def flaky(url, params):
        calls["n"] += 1
        if calls["n"] == 2:                      # the second request fails once
            r = _Resp([]); r.status_code = 502
            return r
        return real_get(url, params)

    fake.get = flaky
    monkeypatch.setattr(demo, "_thread_client", lambda: fake)
    monkeypatch.setattr(demo.time, "sleep", lambda s: None)
    got = demo.fetch_window(1, min_hours=1.0, concurrency=1)
    assert len(got) == 250
