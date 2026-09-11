"""
Leaving a cross-market arbitrage early — the arithmetic and the persistence.

The position is NO on narrow plus YES on broad: a guaranteed 1.00 and a lottery
ticket on the middle outcome, paid only at resolution. Completing both sets
merges to 2.00 immediately and sells the ticket at the market's price. These
tests pin that the exit never beats itself on paper and loses in practice: it
fires only when it captures what holding already guarantees.
"""

from __future__ import annotations

import pytest

from strategy.cross_exit import (
    ROUTE_COMPLETE,
    ROUTE_SELL,
    CrossPosition,
    ExitQuote,
    PositionBook,
    best_exit,
    exit_by_completion,
    exit_by_sale,
    resolution_profit,
    should_exit,
    taker_fee_per_share,
)


def _pos(narrow_no=0.40, broad_yes=0.39, size=10.0, **kw) -> CrossPosition:
    """The worked example: narrow at 0.60, broad at 0.39, fees off for clarity."""
    base = dict(
        narrow="0xnarrow", broad="0xbroad",
        narrow_title="X wins by more than 60%", broad_title="X wins",
        narrow_yes_token="ny", narrow_no_token="nn",
        broad_yes_token="by", broad_no_token="bn",
        size=size, narrow_no_paid=narrow_no, broad_yes_paid=broad_yes,
        opened_ts=1_000_000.0,
    )
    base.update(kw)
    return CrossPosition(**base)


# ── the fee ───────────────────────────────────────────────────────────────────

def test_fee_follows_the_bell():
    """rate x p x (1-p): largest at 0.5, vanishing at the edges."""
    assert taker_fee_per_share(0.5, 0.05) == pytest.approx(0.0125)
    assert taker_fee_per_share(0.02, 0.05) < taker_fee_per_share(0.5, 0.05) / 10
    assert taker_fee_per_share(0.0, 0.05) == 0.0


# ── the position ──────────────────────────────────────────────────────────────

def test_guaranteed_edge_is_one_minus_entry():
    p = _pos()
    assert p.entry_cost == pytest.approx(0.79)
    assert p.guaranteed_edge == pytest.approx(0.21)


# ── completion: the core claim ────────────────────────────────────────────────

def test_completing_at_the_mispriced_state_returns_nothing():
    """
    While the violation persists, the complementary pair is overpriced by the
    same amount the entry was underpriced: 0.60 + 0.61 = 1.21. It is one
    mispricing seen from both sides, so completing now just unwinds it.
    """
    profit = exit_by_completion(_pos(), 0.60, 0.61, rate=0.0, gas=0.0)
    assert profit == pytest.approx(0.0, abs=1e-9)


def test_completing_after_a_correction_pays_edge_plus_the_middle_outcome():
    """
    Corrected market: narrow 0.20, broad 0.55, so the middle outcome is priced
    at 0.35. Completing costs 0.20 + 0.45 = 0.65 and returns 0.21 + 0.35.
    """
    profit = exit_by_completion(_pos(), 0.20, 0.45, rate=0.0, gas=0.0)
    assert profit == pytest.approx(0.56)


def test_the_complementary_pair_is_mutually_exclusive():
    """
    YES on narrow and NO on broad cannot both pay — narrow happening means broad
    happens — so a sanely priced market charges under 1.00 for the two together,
    and the shortfall is exactly the middle outcome's price.
    """
    narrow_yes, broad_yes = 0.20, 0.55
    completion = narrow_yes + (1.0 - broad_yes)
    middle = broad_yes - narrow_yes
    assert completion == pytest.approx(1.0 - middle)
    assert completion < 1.0


def test_gas_is_charged_per_transaction_not_per_share():
    """Two merges weigh far more on a 1-pair position than on a 100-pair one."""
    small = exit_by_completion(_pos(size=1.0), 0.20, 0.45, rate=0.0, gas=0.05)
    large = exit_by_completion(_pos(size=100.0), 0.20, 0.45, rate=0.0, gas=0.05)
    assert large > small
    assert large == pytest.approx(0.56 - 0.001)


def test_missing_books_give_no_quote():
    assert exit_by_completion(_pos(), None, 0.45) is None
    assert exit_by_sale(_pos(), 0.80, None) is None


# ── the sale route ────────────────────────────────────────────────────────────

def test_sale_is_the_same_trade_up_to_spread_and_fees():
    """
    1 − ask(YES_narrow) ≈ bid(NO_narrow). On a book with no spread and no fee
    the two routes return the same thing; the real difference is costs.
    """
    comp = exit_by_completion(_pos(), 0.20, 0.45, rate=0.0, gas=0.0)
    sale = exit_by_sale(_pos(), 0.80, 0.55, rate=0.0)
    assert comp == pytest.approx(sale)


def test_best_exit_picks_the_cheaper_route():
    q = best_exit(_pos(), narrow_yes_ask=0.20, broad_no_ask=0.45,
                  narrow_no_bid=0.70, broad_yes_bid=0.50, rate=0.0, gas=0.0)
    assert q.route == ROUTE_COMPLETE
    assert q.profit == pytest.approx(0.56)

    q = best_exit(_pos(), narrow_yes_ask=0.30, broad_no_ask=0.55,
                  narrow_no_bid=0.80, broad_yes_bid=0.55, rate=0.0, gas=0.0)
    assert q.route == ROUTE_SELL


def test_best_exit_with_no_books_has_no_route():
    q = best_exit(_pos(), narrow_yes_ask=None, broad_no_ask=None,
                  narrow_no_bid=None, broad_yes_bid=None)
    assert q.route == ""


# ── the rule ──────────────────────────────────────────────────────────────────

def test_it_holds_while_the_violation_persists():
    q = best_exit(_pos(), narrow_yes_ask=0.60, broad_no_ask=0.61,
                  narrow_no_bid=0.39, broad_yes_bid=0.38, rate=0.0, gas=0.0)
    assert should_exit(_pos(), q, min_capture=1.0) is False


def test_it_exits_once_the_market_corrects():
    q = best_exit(_pos(), narrow_yes_ask=0.20, broad_no_ask=0.45,
                  narrow_no_bid=0.79, broad_yes_bid=0.54, rate=0.0, gas=0.0)
    assert should_exit(_pos(), q, min_capture=1.0) is True


def test_it_never_exits_for_less_than_resolution_guarantees():
    """At min_capture 1.0 a partial correction is not enough — hold instead."""
    q = ExitQuote(ROUTE_COMPLETE, 0.15, 0.15, None)
    assert should_exit(_pos(), q, min_capture=1.0) is False


def test_a_lower_capture_trades_edge_for_capital_back_sooner():
    q = ExitQuote(ROUTE_COMPLETE, 0.18, 0.18, None)
    assert should_exit(_pos(), q, min_capture=0.8) is True


def test_an_entry_that_was_never_an_arbitrage_takes_any_break_even():
    bad = _pos(narrow_no=0.55, broad_yes=0.50)           # cost 1.05
    assert should_exit(bad, ExitQuote(ROUTE_SELL, 0.0, None, 0.0)) is True
    assert should_exit(bad, ExitQuote(ROUTE_SELL, -0.01, None, -0.01)) is False


# ── resolution ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("narrow_yes,broad_yes,payout", [
    (True, True, 1.0),     # narrow happened, so broad did
    (False, True, 2.0),    # the middle outcome — the ticket pays
    (False, False, 1.0),   # neither happened
])
def test_resolution_pays_one_or_two(narrow_yes, broad_yes, payout):
    assert resolution_profit(_pos(), narrow_yes, broad_yes) == pytest.approx(
        payout - 0.79)


def test_a_false_implication_is_the_only_way_to_lose():
    """narrow YES with broad NO cannot happen if the mapping was right."""
    assert resolution_profit(_pos(), True, False) == pytest.approx(-0.79)


# ── persistence ───────────────────────────────────────────────────────────────

def test_positions_survive_a_restart(tmp_path):
    path = tmp_path / "positions.json"
    book = PositionBook(path)
    book.open(_pos())
    book.save()

    again = PositionBook(path).load()
    assert again.has_open("0xnarrow", "0xbroad")
    assert again.open_positions()[0].guaranteed_edge == pytest.approx(0.21)


def test_a_pair_cannot_be_opened_twice():
    book = PositionBook("/nonexistent/unused.json")
    assert book.open(_pos()) is True
    assert book.open(_pos()) is False


def test_closing_moves_it_to_history(tmp_path):
    book = PositionBook(tmp_path / "p.json")
    pos = _pos()
    book.open(pos)
    book.close(pos, route=ROUTE_COMPLETE, profit=0.56, status="exited",
               now=1_000_000.0 + 14 * 86_400)
    assert not book.has_open("0xnarrow", "0xbroad")
    s = book.summary()
    assert s["exited_early"] == 1
    assert s["realised_usdc"] == pytest.approx(5.6)


def test_a_corrupt_file_starts_empty_rather_than_crashing(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json")
    assert PositionBook(path).load().open_positions() == []


def test_unknown_fields_in_an_older_file_are_ignored(tmp_path):
    """A file written by a later version must not stop this one loading."""
    import json
    path = tmp_path / "p.json"
    row = {**_pos().__dict__, "some_future_field": 1}
    path.write_text(json.dumps({"open": [row], "closed": []}))
    assert PositionBook(path).load().has_open("0xnarrow", "0xbroad")
