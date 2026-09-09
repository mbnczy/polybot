"""
tests/test_backtest_consistency.py
──────────────────────────────────
The backtest measures whether the implication mapper tells the truth. That
number is only meaningful if a counterexample really is the mapper's fault.

Polymarket resolved "Reppo FDV above $100M one day after launch?" NO while
resolving $500M and $800M in the same event, under a byte-identical
description, YES. Charging that to the mapper moved the headline violation rate
from 0% to 16.7% and pointed the blame at the wrong component.

The excuse has to be narrow, though, or it becomes a way for real errors to
escape — which is what these tests pin down.
"""

from scripts.backtest_implications import _family_key, inconsistent_families


def _mkt(question: str, yes: bool) -> dict:
    return {
        "question": question,
        "conditionId": "0x" + str(abs(hash(question)))[:8],
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["1", "0"]' if yes else '["0", "1"]',
    }


def test_contradictory_threshold_family_is_flagged():
    markets = [
        _mkt("Reppo FDV above $20M one day after launch?", True),
        _mkt("Reppo FDV above $50M one day after launch?", True),
        _mkt("Reppo FDV above $100M one day after launch?", False),   # the defect
        _mkt("Reppo FDV above $500M one day after launch?", True),
    ]
    assert inconsistent_families(markets) == {"Reppo FDV above # one day after launch?"}


def test_exact_count_family_is_not_excused():
    """
    "Will 47 senators vote Yea?" asks for an exact count. Its family is
    non-monotone by design — exactly 50 voted, so only that market is YES — and
    a mapper reading it as a threshold ladder IS wrong. Excusing it here would
    hide a real error, so these titles must not form a family at all.
    """
    markets = [_mkt(f'Will {n} senators vote "Yea" for the nominee?', n == 50)
               for n in range(47, 58)]
    assert inconsistent_families(markets) == set()
    assert _family_key('Will 50 senators vote "Yea" for the nominee?') is None


def test_a_monotone_ladder_is_left_alone():
    markets = [
        _mkt("Will BTC reach $50,000 by December 31?", True),
        _mkt("Will BTC reach $70,000 by December 31?", True),
        _mkt("Will BTC reach $90,000 by December 31?", False),
        _mkt("Will BTC reach $120,000 by December 31?", False),
    ]
    assert inconsistent_families(markets) == set()


def test_two_market_family_is_never_contradictory():
    """Two points are monotone in one direction or the other by construction."""
    markets = [
        _mkt("Will ETH reach $4,000 by June 30?", False),
        _mkt("Will ETH reach $2,000 by June 30?", True),
    ]
    assert inconsistent_families(markets) == set()
