"""
The prefilter was rewritten to cover the whole one-week window — about 40,000
markets, 800 million pairs — where the full cross product could not run. It must
still return exactly what the cross product returned. The old implementation is
kept verbatim in tests/_prefilter_reference.py and compared pair for pair here,
across the parameters production and the other tests use.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone

import pytest

import strategy.implication_mapper as im
from tests._prefilter_reference import build_candidates_reference

NOW = 1_800_000_000.0
OU = ("Over", "Under")
YN = ("Yes", "No")
TEAMS = ["Arsenal FC", "Chelsea FC", "Real Betis", "FK Teplice", "SK Slavia Praha",
         "Gimcheon Sangmu", "Gangwon FC", "Cesena FC", "US Cremonese", "Riga FC",
         "FK Auda", "Lions FC", "South Melbourne", "Al Faisaly", "Al Ittihad"]


def _universe(seed: int = 7, events: int = 14) -> list[dict]:
    rnd = random.Random(seed)
    markets: list[dict] = []

    def add(q, ev=None, group=None, days=1.0, outcomes=YN, cid=None):
        m = {"conditionId": cid or f"0x{len(markets):05x}", "question": q,
             "outcomes": json.dumps(list(outcomes))}
        if days is not None:
            m["endDate"] = datetime.fromtimestamp(NOW + days * 86_400, timezone.utc) \
                .strftime("%Y-%m-%dT%H:%M:%SZ")
        if ev:
            m["events"] = [{"id": ev}]
        if group:
            m["negRiskMarketID"] = group
        markets.append(m)

    for e in range(events):
        h, a = rnd.sample(TEAMS, 2)
        ev, days, match = f"ev{e}", rnd.choice([0.2, 1.0, 3.0, 6.5, 9.0]), f"{h} vs. {a}"
        for line in (0.5, 1.5, 2.5, 3.5):
            add(f"{match}: O/U {line}", ev, days=days, outcomes=OU)
        for line in (7.5, 8.5, 9.5):
            add(f"{match}: O/U {line} Total Corners", ev, days=days, outcomes=OU)
        add(f"{match}: 1st Half O/U 0.5", ev, days=days, outcomes=OU)
        add(f"{match}: {h} O/U 0.5", ev, days=days, outcomes=OU)
        add(f"{match}: {a} O/U 0.5", ev, days=days, outcomes=OU)
        for score in ("0 - 0", "1 - 0", "0 - 1", "1 - 1", "2 - 1"):
            add(f"Exact Score: {h} {score} {a}?", ev, group=f"g{e}", days=days)
        add(f"Will {match} end in a draw?", ev, days=days)
        add(f"Will {h} win?", ev, days=days)
        add(f"{match}: O/U 2.5", ev, days=days, outcomes=OU)              # duplicate wording
    for k in range(40):                                                    # no event: ladders
        coin = rnd.choice(["Bitcoin", "Ethereum", "Solana"])
        add(f"Will {coin} reach ${rnd.choice([60, 70, 80, 90, 100])}k by September 20?",
            days=rnd.choice([2.0, 5.0, 8.0, None]))
    for k in range(20):                                                    # same team, other events
        t = rnd.choice(TEAMS)
        add(f"Will {t} score first?", f"other{k % 4}", days=rnd.choice([1.0, 4.0]))
    add("?!", "ev0")                                                       # no usable word
    add("Will Arsenal FC win?", "ev0", cid="0x00000")                      # repeated condition id
    return markets


GRID = [(0.5, 400, 20), (0.5, 50, 3), (0.3, 1000, 8), (0.15, 5000, 0),
        (0.5, 5000, 1), (0.0, 300, 5)]


@pytest.mark.parametrize("lockup_cap", [0.0, 7.0])
@pytest.mark.parametrize("min_overlap,max_pairs,max_per_event", GRID)
def test_the_scaled_prefilter_matches_the_cross_product(
        monkeypatch, min_overlap, max_pairs, max_per_event, lockup_cap):
    monkeypatch.setattr(im, "_MAX_LOCKUP_DAYS", lockup_cap)
    monkeypatch.setattr(time, "time", lambda: NOW)
    markets = _universe()
    kw = dict(min_overlap=min_overlap, max_pairs=max_pairs, max_per_event=max_per_event)
    expected = build_candidates_reference(markets, **kw)
    assert expected, "the universe should produce candidates"
    assert im.build_candidates(markets, **kw) == expected


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_it_matches_on_other_universes_too(monkeypatch, seed):
    monkeypatch.setattr(im, "_MAX_LOCKUP_DAYS", 7.0)
    monkeypatch.setattr(time, "time", lambda: NOW)
    markets = _universe(seed=seed, events=10)
    kw = dict(min_overlap=0.5, max_pairs=800, max_per_event=8)
    assert im.build_candidates(markets, **kw) == build_candidates_reference(markets, **kw)
