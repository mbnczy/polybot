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
