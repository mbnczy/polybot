"""
A taker pair where one leg filled and the other did not.

On 2026-11-04 01:04 UTC the bot asked for 11.13 NO at 0.747, got 11.87 at 0.701,
sold back 11.13 and left 0.74 stranded. It said "no profit booked" and booked
nothing, for a real loss of 0.43 USDC.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.clob_client import filled_buy
from strategy.arbitrage import round_trip_pnl


def test_a_market_buy_can_get_more_shares_than_asked():
    resp = {"status": "matched", "making_amount": Decimal("8.32"),
            "taking_amount": Decimal("11.868759")}
    assert filled_buy(resp, 11.13, 0.747) == pytest.approx((8.32, 11.868759))


def test_a_response_without_amounts_falls_back_on_what_was_asked():
    assert filled_buy({"status": "matched"}, 10.0, 0.5) == pytest.approx((5.0, 10.0))


def test_the_round_trip_of_2026_11_04_comes_out_as_the_wallet_says():
    # BUY  11.868759 at 0.701: the wallet paid 8.4195
    # SELL 11.13 at 0.66:      the wallet received 7.2459
    pnl = round_trip_pnl(8.32, 11.868759, 7.3458, 11.13, rate=0.04)
    cost_of_sold = 8.4195 * 11.13 / 11.868759
    assert pnl == pytest.approx(7.2459 - cost_of_sold, abs=0.001)


def test_selling_all_of_it_books_all_of_the_loss():
    pnl = round_trip_pnl(8.32, 11.868759, 11.868759 * 0.66, 11.868759, rate=0.04)
    assert pnl == pytest.approx(11.868759 * 0.66 * (1 - 0.04 * 0.34) - 8.4195, abs=0.001)


def test_nothing_sold_is_no_result():
    assert round_trip_pnl(8.32, 11.87, 0.0, 0.0, rate=0.04) == 0.0
