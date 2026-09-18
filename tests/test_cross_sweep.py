"""
How fast the guard sees a move.

On 2026-09-17 the file held 2,200 pairs, 84% of them on books too wide to trade,
and the guard's book-read budget prices 30 pairs a poll. Even with a pair 30
points from the threshold left alone for ten minutes, the queue never emptied:
a full sweep took 19 minutes, while a goal moves a match's over/under by 30
points at once. The reader now exports only quotable books; the guard spends
what the due pairs leave of its budget on the deferred ones, and says how old
its stalest price is.
"""

from __future__ import annotations

import time

import pytest

import scripts.demo_cross_market as reader
from tests.test_cross_guard import NOW, _book, _guard, _imp, _write_imps

FAR = {"nn": _book(asks=[(0.83, 50)], bids=[(0.80, 50)]),
       "by": _book(asks=[(0.96, 50)], bids=[(0.94, 50)])}          # edge -0.79
WIDE = {"nn2": _book(asks=[(0.50, 50)], bids=[(0.01, 50)]),
        "by2": _book(asks=[(0.60, 50)], bids=[(0.01, 50)])}         # nothing to rest in
SECOND = _imp(narrow="0xn2", broad="0xb2", narrow_no_token="nn2",
              broad_yes_token="by2", narrow_yes_token="ny2", broad_no_token="bn2")


def _counting(client):
    reads: list[str] = []
    inner = client.get_orderbook

    async def counting(token):
        reads.append(token)
        return await inner(token)

    client.get_orderbook = counting
    return reads


def _clock(monkeypatch, start=NOW):
    t = [start]
    monkeypatch.setattr(time, "time", lambda: t[0])
    return t


# ── the leftover budget ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_deferred_pair_is_priced_when_the_budget_has_room(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, FAR, [_imp()], enabled=False)
    reads = _counting(client)
    await g.poll_once()
    await g.poll_once()
    assert reads == ["nn", "by", "nn", "by"]


@pytest.mark.asyncio
async def test_a_tight_book_gets_the_leftover_read_before_a_wide_one(tmp_path, monkeypatch):
    """Both pairs are deferred and there is room for one. The wide one waited
    longer, but a book nobody quotes is the last place a trade appears."""
    g, client, _, _ = _guard(tmp_path, monkeypatch, {**WIDE, **FAR}, [SECOND, _imp()],
                             enabled=False, max_book_reads=2)
    reads = _counting(client)
    await g.poll_once()                              # the wide pair, first in the file
    await g.poll_once()                              # the far one, never priced yet
    assert g.stop_summary().count("book_too_wide 1") == 1
    await g.poll_once()                              # both deferred, one read of room
    assert reads[4:] == ["nn", "by"]


# ── the stalest price ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_summary_says_how_old_the_stalest_price_is(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, {**WIDE, **FAR}, [_imp(), SECOND],
                             enabled=False, max_book_reads=2)
    t = _clock(monkeypatch)
    await g.poll_once()                              # prices the first, queues the second
    t[0] = NOW + 50.0
    await g.poll_once()                              # prices the second; the first waits
    assert "stalest price 50s" in g.stop_summary()


@pytest.mark.asyncio
async def test_a_pair_never_priced_counts_from_the_first_look(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, {**WIDE, **FAR}, [_imp(), SECOND],
                             enabled=False, max_book_reads=2)
    t = _clock(monkeypatch)
    g._first_scan = NOW - 120.0                      # the file was read two minutes ago
    await g.poll_once()
    assert "stalest price 120s" in g.stop_summary()


# ── pairs that leave the file ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_pair_that_left_the_file_is_forgotten(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, {**WIDE, **FAR}, [_imp(), SECOND],
                             enabled=False)
    await g.poll_once()
    assert SECOND.key in g._last_stop and SECOND.key in g._priced_at
    _write_imps(tmp_path, [_imp()])                  # the reader dropped it
    await g.poll_once()
    for table in (g._last_stop, g._last_edge, g._next_check, g._priced_at):
        assert SECOND.key not in table
    assert _imp().key in g._last_stop


@pytest.mark.asyncio
async def test_a_cooldown_survives_the_pair_leaving_the_file(tmp_path, monkeypatch):
    """A losing unwind's cooldown is safety state, not a schedule."""
    g, client, _, _ = _guard(tmp_path, monkeypatch, {**WIDE, **FAR}, [_imp(), SECOND],
                             enabled=False)
    await g.poll_once()
    g._cooldown[SECOND.key] = NOW + 3600.0
    _write_imps(tmp_path, [_imp()])
    await g.poll_once()
    assert g._cooldown[SECOND.key] == NOW + 3600.0


# ── the reader exports quotable books only ────────────────────────────────────

def _m(cid, bid, ask):
    return {"conditionId": cid, "bestBid": bid, "bestAsk": ask}


def test_only_markets_quoted_within_the_ceiling_are_kept():
    kept = reader.quotable([_m("tight", 0.40, 0.42), _m("wide", 0.01, 0.99),
                            _m("one-sided", None, 0.30), _m("dead", 0, 0)], 0.10)
    assert [m["conditionId"] for m in kept] == ["tight"]


def test_a_book_exactly_at_the_ceiling_is_kept():
    """0.55 - 0.45 is 0.10000000000000009."""
    assert len(reader.quotable([_m("edge", 0.45, 0.55)], 0.10)) == 1


def test_the_reader_ceiling_leaves_room_above_the_makers_gate():
    """Watch a book while it is near tradeable, or a pair drops out of the file
    every time its spread brushes the gate."""
    from execution.cross_guard import CROSS_MAKER_MAX_SPREAD
    assert reader.READER_MAX_SPREAD > CROSS_MAKER_MAX_SPREAD


# ── books in batches ──────────────────────────────────────────────────────────

import asyncio
import json

import execution.cross_guard as cg
from core.clob_client import PolyClient
from execution.cross_guard import load_implications
from tests.test_cross_guard import _Client


class _BatchClient(_Client):
    """A client that can read many books in one request, and counts both kinds."""

    def __init__(self, books, fail=False):
        super().__init__(books)
        self.batches: list[list[str]] = []
        self.singles: list[str] = []
        self.fail = fail

    async def get_orderbook(self, token):
        self.singles.append(token)
        return await super().get_orderbook(token)

    async def get_orderbooks(self, token_ids, chunk=100):
        self.batches.append(list(token_ids))
        if self.fail:
            raise RuntimeError("503 from /books")
        return {t: self.books[t] for t in token_ids if t in self.books}


def _pairs(n):
    """n pairs on distinct markets, every one 79 points from the threshold."""
    books, imps = {}, []
    for i in range(n):
        books[f"nn{i}"] = _book(asks=[(0.83, 50)], bids=[(0.80, 50)])
        books[f"by{i}"] = _book(asks=[(0.96, 50)], bids=[(0.94, 50)])
        imps.append(_imp(narrow=f"0xn{i}", broad=f"0xb{i}", narrow_no_token=f"nn{i}",
                         broad_yes_token=f"by{i}", narrow_yes_token=f"ny{i}",
                         broad_no_token=f"bn{i}"))
    return books, imps


def _batch_guard(tmp_path, monkeypatch, books, imps, *, fail=False, **kw):
    g, _, _, _ = _guard(tmp_path, monkeypatch, books, imps, enabled=False, **kw)
    client = _BatchClient(books, fail=fail)
    g._client = client
    return g, client


@pytest.mark.asyncio
async def test_the_whole_file_is_priced_in_one_poll_from_one_batch(tmp_path, monkeypatch):
    """With single reads a poll prices 30 pairs; 100 pairs took four polls."""
    books, imps = _pairs(100)
    g, client = _batch_guard(tmp_path, monkeypatch, books, imps)
    await g.poll_once()
    assert len(client.batches) == 1 and client.singles == []
    line = g.stop_summary()
    assert "maker_edge_below_min 100" in line and "not re-priced" not in line


@pytest.mark.asyncio
async def test_the_batch_stops_at_its_token_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "CROSS_BOOK_BATCH", 4)       # two pairs' worth
    books, imps = _pairs(5)
    g, client = _batch_guard(tmp_path, monkeypatch, books, imps, book_batches=1)
    await g.poll_once()
    assert client.batches == [["nn0", "by0", "nn1", "by1"]]
    assert "3 not re-priced" in g.stop_summary()


@pytest.mark.asyncio
async def test_a_market_shared_by_two_pairs_costs_one_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "CROSS_BOOK_BATCH", 3)
    books, imps = _pairs(2)
    shared = _imp(narrow="0xn9", broad="0xb9", narrow_no_token="nn0",
                  broad_yes_token="by1", narrow_yes_token="ny9", broad_no_token="bn9")
    g, client = _batch_guard(tmp_path, monkeypatch, books, [imps[0], shared, imps[1]],
                             book_batches=1)
    await g.poll_once()
    # nn0, by0 for the first pair; the shared one adds only by1 — three tokens.
    assert set(client.batches[0]) == {"nn0", "by0", "by1"}
    assert "maker_edge_below_min 2" in g.stop_summary()


@pytest.mark.asyncio
async def test_a_failed_batch_falls_back_to_single_reads_at_the_old_budget(tmp_path, monkeypatch):
    books, imps = _pairs(10)
    g, client = _batch_guard(tmp_path, monkeypatch, books, imps, fail=True,
                             max_book_reads=6)
    await g.poll_once()
    assert len(client.batches) == 1
    assert len(client.singles) == 6                      # three pairs, two reads each
    assert "7 not re-priced" in g.stop_summary()


@pytest.mark.asyncio
async def test_a_token_missing_from_the_batch_is_a_pair_with_no_book(tmp_path, monkeypatch):
    books, imps = _pairs(2)
    del books["by1"]                                      # the exchange has no book
    g, client = _batch_guard(tmp_path, monkeypatch, books, imps)
    await g.poll_once()
    line = g.stop_summary()
    assert "no_book 1" in line and "maker_edge_below_min 1" in line
    assert client.singles == []                           # not re-asked one at a time


# ── the fee the reader exported ───────────────────────────────────────────────

class _Fees:
    def __init__(self):
        self.asked: list[str] = []

    async def get_taker_rate(self, condition_id):
        self.asked.append(condition_id)
        return 0.04


@pytest.mark.asyncio
async def test_an_exported_fee_rate_saves_the_lookup(tmp_path, monkeypatch):
    fees = _Fees()
    imp = _imp(narrow_fee_rate=0.03, broad_fee_rate=0.0)
    g, _, _, _ = _guard(tmp_path, monkeypatch, FAR, [imp], enabled=False, fee_engine=fees)
    await g.poll_once()
    assert fees.asked == []


@pytest.mark.asyncio
async def test_without_an_exported_rate_the_fee_engine_is_asked(tmp_path, monkeypatch):
    fees = _Fees()
    g, _, _, _ = _guard(tmp_path, monkeypatch, FAR, [_imp()], enabled=False, fee_engine=fees)
    await g.poll_once()
    assert fees.asked == ["0xn", "0xb"]


def test_fees_off_survives_the_round_trip_as_zero(tmp_path):
    """0.0 is a rate, not a missing one: a truthiness test would read it as None."""
    p = tmp_path / "imps.json"
    row = {"narrow": "0xn", "broad": "0xb", "narrow_title": "n", "broad_title": "b",
           "narrow_yes_token": "ny", "narrow_no_token": "nn", "broad_yes_token": "by",
           "broad_no_token": "bn", "narrow_end_ts": NOW + 86_400, "broad_end_ts": NOW + 86_400,
           "narrow_outcomes": ["Yes", "No"], "broad_outcomes": ["Yes", "No"],
           "narrow_fee_rate": 0.0, "broad_fee_rate": 0.05}
    p.write_text(json.dumps({"implications": [row]}))
    (imp,) = load_implications(p)
    assert imp.narrow_fee_rate == 0.0 and imp.broad_fee_rate == 0.05


def test_the_reader_reads_the_fee_the_way_the_fee_engine_does():
    assert reader._fee_rate({"feesEnabled": False, "feeSchedule": {"rate": 0.05}}) == 0.0
    assert reader._fee_rate({"feesEnabled": True, "feeSchedule": {"rate": 0.03}}) == 0.03
    assert reader._fee_rate({"feesEnabled": True}) is None           # FeeEngine decides
    assert reader._fee_rate({"feeSchedule": {"rate": 7}}) is None      # not a rate
    slim = reader._slim({"conditionId": "c", "feesEnabled": True,
                         "feeSchedule": {"rate": 0.07, "exponent": 1, "takerOnly": True}})
    assert slim["feeRate"] == 0.07 and "feeSchedule" not in slim


# ── the client's batch read ───────────────────────────────────────────────────

class _Book:
    def __init__(self, token_id, bid, ask):
        self.token_id = token_id
        self.bids = [type("L", (), {"price": str(bid), "size": "10"})()]
        self.asks = [type("L", (), {"price": str(ask), "size": "10"})()]


def test_the_client_dedupes_chunks_and_keys_books_by_token():
    calls: list[list[str]] = []

    class _Sdk:
        def get_order_books(self, *, token_ids):
            calls.append(list(token_ids))
            return tuple(_Book(t, 0.40, 0.45) for t in token_ids if t != "dead")

    c = PolyClient.__new__(PolyClient)
    c._client = _Sdk()

    async def run(fn, *a, **k):
        return fn()

    c._run_with_retry = run
    out = asyncio.run(c.get_orderbooks(["a", "b", "a", "dead", "c"], chunk=2))
    assert calls == [["a", "b"], ["dead", "c"]]
    assert set(out) == {"a", "b", "c"}
    assert out["a"] == {"bids": [{"price": 0.40, "size": 10.0}],
                        "asks": [{"price": 0.45, "size": 10.0}]}


@pytest.mark.asyncio
async def test_fee_lookups_are_capped_per_poll(tmp_path, monkeypatch):
    """An export without rates must not become a Gamma request per market."""
    monkeypatch.setattr(cg, "CROSS_MAX_FEE_LOOKUPS_PER_POLL", 3)
    fees = _Fees()
    books, imps = _pairs(10)
    g, client = _batch_guard(tmp_path, monkeypatch, books, imps, fee_engine=fees)
    await g.poll_once()
    assert len(fees.asked) == 3
    assert "maker_edge_below_min 10" in g.stop_summary()   # every pair still priced


@pytest.mark.asyncio
async def test_a_rate_fee_engine_already_holds_costs_no_lookup(tmp_path, monkeypatch):
    class _Cached(_Fees):
        def peek_taker_rate(self, condition_id):
            return 0.03

    fees = _Cached()
    g, _, _, _ = _guard(tmp_path, monkeypatch, FAR, [_imp()], enabled=False, fee_engine=fees)
    await g.poll_once()
    assert fees.asked == []
