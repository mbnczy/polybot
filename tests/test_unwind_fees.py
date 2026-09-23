"""
An unwind's realised P&L is what the wallet received, not what the book paid.

The order response's taking_amount is before the taker fee. On 2026-09-23 the
bot booked −0.04 USDC over four NegRisk unwinds and completions while the wallet
lost −0.31: every unwind was short by its fee.
"""

from __future__ import annotations

import pytest

from strategy.arbitrage import taker_sell_fee
from tests.test_cross_guard import ARB, _guard, _imp


@pytest.mark.parametrize("gross, sold, received", [
    (2.7975, 10.21, 2.7163),      # Texas governor NO, 10.21 at 0.274
    (6.898, 10.30, 6.80687),      # TX-35 NO, 10.30 at 0.6697
    (3.4505, 5.15, 3.40496),      # TX-35 NO, half of a FOK retry
])
def test_the_fee_is_what_the_wallet_did_not_receive(gross, sold, received):
    assert gross - taker_sell_fee(gross, sold, 0.04) == pytest.approx(received, abs=6e-4)


def test_nothing_sold_pays_no_fee():
    assert taker_sell_fee(0.0, 5.0, 0.04) == 0.0
    assert taker_sell_fee(1.0, 0.0, 0.04) == 0.0


@pytest.mark.asyncio
async def test_a_cross_sell_books_the_fee_at_the_markets_rate(tmp_path, monkeypatch):
    g, client, _, _ = _guard(tmp_path, monkeypatch, ARB, [_imp()])
    # 10 NO shares bought at 0.40, sold at the 0.38 bid, market rate 0.05:
    # 3.80 − 4.00 − 10 × 0.05 × 0.38 × 0.62 = −0.2 − 0.1178
    pnl = await g._sell("nn", 10.0, 0.40, 0.05)
    assert client.sells == [("nn", 10.0)]
    assert pnl == pytest.approx(-0.3178, abs=1e-6)
