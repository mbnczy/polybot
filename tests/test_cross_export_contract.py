"""
The contract between the read-only reader and the trading bot is one file.

The reader writes it with export_implications; the bot's cross guard reads it
with load_implications. They are separate programs, and a field renamed on one
side would leave the bot silently holding "nothing to trade" forever — so the
round trip is tested end to end, with the reader's real writer.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

from execution.cross_guard import load_implications
from scripts.demo_cross_market import export_implications
from strategy.cross_market import Implication


def _iso(days):
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mkt(cid, q, yes, no, end=None):
    return {"conditionId": cid, "question": q, "endDate": end or _iso(2),
            "clobTokenIds": json.dumps([yes, no])}


MARKETS = [
    _mkt("0xn", "Imisli FK vs. Neftchi Baku PFC: 1st Half O/U 0.5", "ny", "nn"),
    _mkt("0xb", "Imisli FK vs. Neftchi Baku PFC: O/U 0.5", "by", "bn", end=_iso(2.1)),
]


def test_what_the_reader_writes_the_bot_reads(tmp_path):
    path = tmp_path / "cross_implications.json"
    n = export_implications([Implication("0xn", "0xb", 0.97, "first half ⊆ match")],
                            MARKETS, str(path))
    assert n == 1
    (imp,) = load_implications(path)
    assert (imp.narrow, imp.broad) == ("0xn", "0xb")
    assert (imp.narrow_yes_token, imp.narrow_no_token) == ("ny", "nn")
    assert (imp.broad_yes_token, imp.broad_no_token) == ("by", "bn")
    assert imp.narrow_title.startswith("Imisli FK")
    # The later leg sets the lockup.
    assert imp.resolves_ts == imp.broad_end_ts > imp.narrow_end_ts
    assert imp.confidence == 0.97


def test_a_relation_whose_markets_are_unknown_is_not_exported(tmp_path):
    path = tmp_path / "cross_implications.json"
    n = export_implications([Implication("0xn", "0xmissing", 0.97)], MARKETS, str(path))
    assert n == 0
    assert load_implications(path) == []


def test_the_write_is_atomic(tmp_path):
    """The bot must never read half a file: no temp file is left behind."""
    path = tmp_path / "cross_implications.json"
    export_implications([Implication("0xn", "0xb", 0.97)], MARKETS, str(path))
    assert not (tmp_path / "cross_implications.json.tmp").exists()


def test_a_pair_that_has_already_ended_is_not_exported(tmp_path):
    """
    The bot prices what is in this file. Inside a one-week window the match
    sub-markets expire hourly, and on 2026-09-12 whole passes read "past_end 25"
    off a file 90 minutes old.
    """
    ended = [_mkt("0xn", "narrow", "ny", "nn", end=_iso(-0.5)),
             _mkt("0xb", "broad", "by", "bn", end=_iso(-0.4))]
    path = tmp_path / "imps.json"
    n = export_implications([Implication("0xn", "0xb", 0.97, "over ⊆ over")],
                            ended, str(path))
    assert n == 0
    assert load_implications(path) == []


def test_a_pair_still_running_survives_the_expiry_filter(tmp_path):
    """One leg already ended does not end the pair: the later date decides."""
    mixed = [_mkt("0xn", "narrow", "ny", "nn", end=_iso(-0.5)),
             _mkt("0xb", "broad", "by", "bn", end=_iso(1.0))]
    path = tmp_path / "imps.json"
    assert export_implications([Implication("0xn", "0xb", 0.97, "e")],
                               mixed, str(path)) == 1
