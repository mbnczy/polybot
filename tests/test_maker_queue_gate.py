"""
tests/test_maker_queue_gate.py
──────────────────────────────
Whether a post-only quote can fill is decided by the queue in front of it, and
until now the bot never looked.

Measured across 257 live two-sided books on 2026-09-10:

    one-tick spread                     79% of active markets
    median queue at the touch       10,558 shares
    our order                           10 shares          -> 1,056x behind
    p90 queue                      171,338 shares          -> 17,134x behind

On a one-tick book there is no price improvement to buy: the tick IS the
minimum increment, so the only price above the resting bid is the ask, and
quoting there makes us a taker. Those maker legs do not fill slowly. They do
not fill.

What this changes is routing, not admission. Section 7b of the detector already
prices a leg we will have to cross at the ask plus the taker fee, so a bundle
that clears is genuinely profitable — it is just executed badly, resting an
order that cannot fill until the TTL expires and the guard crosses it anyway.
`ArbLeg.reachable` names the legs that applies to.
"""

import math

import pytest

from core.ws_feed import _best_bid_level, _depth_bucket
from strategy.arbitrage import (
    NegRiskArbDetector,
    maker_quote_is_reachable,
    quote_opens_new_level,
)


def _det(**kw) -> NegRiskArbDetector:
    params = dict(desired_net_margin=0.005, min_leg_shares=0.0)
    params.update(kw)
    return NegRiskArbDetector(**params)


# ── the book reading ─────────────────────────────────────────────────────────

class TestBidLevel:

    def test_size_comes_back_with_the_price(self):
        """The queue is the number that decides everything; it must survive."""
        assert _best_bid_level([{"price": "0.45", "size": "250"}]) == (0.45, 250.0)

    def test_the_best_bid_is_the_highest_not_the_first(self):
        """
        The exchange sends bids ascending from 0.001, so bids[0] is the WORST
        bid. Reading it as the touch reports a queue from the wrong price level
        entirely — the same trap already documented on the ask side.
        """
        level = _best_bid_level([
            {"price": "0.02", "size": "9999"},
            {"price": "0.45", "size": "250"},
            {"price": "0.31", "size": "40"},
        ])
        assert level == (0.45, 250.0)

    def test_a_missing_size_is_unknown_not_zero(self):
        price, size = _best_bid_level([{"price": "0.45"}])
        assert price == 0.45
        assert size is None

    def test_empty_book(self):
        assert _best_bid_level([]) is None
        assert _best_bid_level(None) is None


class TestDepthBucket:
    """
    Raw sizes in the tick fingerprint would push an update on every one-share
    flicker; omitting them entirely would hide a book draining from 10,000 to
    90 while the price stands still. Order of magnitude splits the difference.
    """

    def test_a_real_drain_changes_bucket(self):
        assert _depth_bucket(10_558) != _depth_bucket(90)

    def test_jitter_does_not(self):
        assert _depth_bucket(10_558) == _depth_bucket(10_200)

    def test_unknown_is_distinct_from_empty(self):
        assert _depth_bucket(None) != _depth_bucket(0)


# ── the decision ─────────────────────────────────────────────────────────────

class TestQuoteOpensNewLevel:

    def test_inside_a_two_tick_spread_opens_its_own_level(self):
        # ask 0.87, bid 0.85 -> lead_book_bid quotes 0.86, alone.
        assert quote_opens_new_level(0.86, 0.85, 0.01) is True

    def test_on_a_one_tick_spread_it_joins_the_touch(self):
        # ask 0.86, bid 0.85 -> the post-only ceiling IS 0.85.
        assert quote_opens_new_level(0.85, 0.85, 0.01) is False

    def test_an_empty_book_is_a_new_level(self):
        assert quote_opens_new_level(0.85, None, 0.01) is True


class TestReachability:

    def test_a_new_price_level_is_reachable_however_deep_the_book(self):
        """Alone at its own price, an order is first by construction."""
        assert maker_quote_is_reachable(0.86, 0.85, 0.01, 999_999, 10) is True

    def test_the_median_one_tick_queue_is_not_reachable(self):
        """10,558 shares ahead of ten. This is the common case, not the tail."""
        assert maker_quote_is_reachable(0.85, 0.85, 0.01, 10_558, 10) is False

    def test_a_thin_queue_is_reachable(self):
        assert maker_quote_is_reachable(0.85, 0.85, 0.01, 90, 10) is True

    def test_the_threshold_scales_with_our_own_order(self):
        # 20x the order by default: 200 shares for a 10-share leg.
        assert maker_quote_is_reachable(0.85, 0.85, 0.01, 200, 10) is True
        assert maker_quote_is_reachable(0.85, 0.85, 0.01, 201, 10) is False

    def test_an_unknown_queue_is_permitted(self):
        """
        A batched price_change states a price without its depth. Refusing to
        quote on missing data would silence the bot on exactly the fast-moving
        books worth quoting, so unknown is allowed through — and recorded, so
        the cost of that choice is measurable rather than assumed.
        """
        assert maker_quote_is_reachable(0.85, 0.85, 0.01, None, 10) is True


# ── end to end through the detector ──────────────────────────────────────────

class TestDetectorCarriesTheBook:

    def test_a_deep_touch_marks_the_leg_unreachable(self):
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.58, 0.63, 0.68],
            maker_rebate=0.0, tick_size=0.01,
            # one-tick spreads: the quote can only join the touch
            no_best_bids=[0.57, 0.62, 0.67],
            no_bid_sizes=[10_558, 8_000, 12_000],
        )
        assert sig is not None
        assert all(not leg.leads for leg in sig.legs)
        assert all(not leg.reachable for leg in sig.legs)

    def test_a_wide_spread_leg_is_reachable(self):
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.58, 0.63, 0.68],
            maker_rebate=0.0, tick_size=0.01,
            # two-tick spreads: the quote opens a level of its own
            no_best_bids=[0.56, 0.61, 0.66],
            no_bid_sizes=[10_558, 8_000, 12_000],
        )
        assert sig is not None
        assert all(leg.leads for leg in sig.legs)
        assert all(leg.reachable for leg in sig.legs)

    def test_the_book_is_carried_for_the_fill_log(self):
        """
        Every field the fill log needs to explain a miss afterwards. The log has
        recorded tick/best_bid/best_ask as null on all 67 real rows since it was
        written, which made its lead-versus-queue report unanswerable.
        """
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.58, 0.63, 0.68],
            maker_rebate=0.0, tick_size=0.01,
            no_best_bids=[0.57, 0.62, 0.67],
            no_bid_sizes=[300, None, 12_000],
        )
        assert sig is not None
        by_id = {leg.token_id: leg for leg in sig.legs}
        assert by_id["A"].book_bid == 0.57
        assert by_id["A"].queue_ahead == 300
        assert by_id["A"].tick == 0.01
        assert by_id["B"].queue_ahead is None      # unknown survives as unknown

    def test_absent_book_data_changes_nothing(self):
        """Callers that pass no bid sizes must behave exactly as before."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.58, 0.63, 0.68],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is not None
        assert all(leg.reachable for leg in sig.legs)
        assert all(leg.queue_ahead is None for leg in sig.legs)


# ── the record that has to survive to next week ──────────────────────────────

class TestFillLogCarriesTheBook:

    def _rows(self, tmp_path, monkeypatch, **kw):
        import json as _json

        import telemetry.fill_log as fl
        path = tmp_path / "fill_log.jsonl"
        monkeypatch.setattr(fl, "FILL_LOG_PATH", str(path))
        monkeypatch.setattr(fl, "FILL_LOG_ENABLED", True)
        fl.record(path="negrisk", outcome="expired", price=0.85, **kw)
        return [_json.loads(l) for l in path.read_text().splitlines()]

    def test_the_queue_is_written(self, tmp_path, monkeypatch):
        row = self._rows(tmp_path, monkeypatch,
                         tick=0.01, best_bid=0.85, best_ask=0.86,
                         queue_ahead=10_558, reachable=False)[0]
        assert row["queue_ahead"] == 10_558
        assert row["reachable"] is False
        assert row["leads"] is False
        assert row["spread_ticks"] == 1.0

    def test_an_unknown_queue_stays_unknown(self, tmp_path, monkeypatch):
        """None means the exchange stated a price without its depth. It is not
        an empty level, and writing it as 0 would invent a clear book."""
        row = self._rows(tmp_path, monkeypatch,
                         tick=0.01, best_bid=0.85, best_ask=0.86,
                         queue_ahead=None, reachable=True)[0]
        assert row["queue_ahead"] is None

    def test_telemetry_failure_never_reaches_the_caller(self, tmp_path, monkeypatch):
        """A log that cannot be written must not interrupt trading."""
        import telemetry.fill_log as fl
        monkeypatch.setattr(fl, "FILL_LOG_PATH", "/proc/nonexistent/fill.jsonl")
        monkeypatch.setattr(fl, "FILL_LOG_ENABLED", True)
        fl.record(path="negrisk", outcome="expired", price=0.85, queue_ahead=5)


class TestGuardPassesTheBookThrough:

    def test_a_watched_leg_keeps_what_the_signal_measured(self):
        """
        The guard is where the book state goes to be forgotten: it builds its
        own per-leg state and, until now, kept only price and size. Everything
        the fill log needs has to survive that copy.
        """
        from execution.negrisk_guard import _BundleLegState

        leg = _BundleLegState(
            0, "tok", "oid", 0.85, 10.0,
            book_bid=0.85, book_ask=0.86, tick=0.01,
            queue_ahead=10_558, leads=False, reachable=False,
        )
        assert leg.queue_ahead == 10_558
        assert leg.reachable is False
        assert leg.book_ask == 0.86

    def test_the_defaults_keep_older_construction_sites_valid(self):
        from execution.negrisk_guard import _BundleLegState

        leg = _BundleLegState(0, "tok", "oid", 0.85, 10.0)
        assert leg.queue_ahead is None
        assert leg.reachable is True
