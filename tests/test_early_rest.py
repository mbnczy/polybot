"""
tests/test_early_rest.py
────────────────────────
Resting before the arbitrage exists.

Under price-time priority a queue cannot be jumped, but it can be JOINED EARLY.
An order resting at the price where the bundle would still clear fills only if
someone sells into it, and at that moment the arbitrage is real by
construction. The prediction turns into patience, which is the only legitimate
form of queue priority there is.

It is also the most dangerous idea in the set, because a leg that fills alone is
a naked position. The four round trips visible in the account on 2026-09-08/09
cost -0.25, -0.25, -0.16 and +0.03, averaging about -0.20 USDC per leg unwound.

The arithmetic that follows from that is uncomfortable enough to be the main
result of this file: at the 0.02-0.03 edge these bundles actually carry, early
resting needs the bundle to complete 87-91% of the time merely to break even.
Zero bundles have ever completed as makers. So the feature ships refusing to
run without a measured completion rate — a default here would be a number
nobody measured deciding how much money to risk.
"""

import pytest

from strategy.arbitrage import (
    arb_limit_price,
    break_even_completion_rate,
    early_rest_expected_value,
    should_rest_early,
)


class TestTheLimitPrice:

    def test_it_is_what_is_left_of_the_payout(self):
        """Three legs pay 2. If two cost 1.30, the third may cost 0.68 at a
        0.02 target edge."""
        assert arb_limit_price(1.30, 3, 0.02) == pytest.approx(0.68)

    def test_a_larger_target_edge_bids_less(self):
        assert arb_limit_price(1.30, 3, 0.05) < arb_limit_price(1.30, 3, 0.02)

    def test_an_expensive_group_leaves_nothing_to_bid(self):
        """When the other legs already cost the whole payout, the limit price
        goes negative — there is no price at which this is an arbitrage."""
        assert arb_limit_price(2.10, 3, 0.0) < 0.0


class TestTheDecisionRule:

    def test_completing_pays_and_failing_costs(self):
        assert early_rest_expected_value(0.03, 1.0, 0.20) == pytest.approx(0.03)
        assert early_rest_expected_value(0.03, 0.0, 0.20) == pytest.approx(-0.20)

    def test_the_break_even_rate_at_a_realistic_edge(self):
        """
        The number that decides whether this is worth building on. A 0.03 edge
        against a 0.20 unwind needs 87% completion.
        """
        assert break_even_completion_rate(0.03, 0.20) == pytest.approx(0.870, abs=0.005)
        assert break_even_completion_rate(0.02, 0.20) == pytest.approx(0.909, abs=0.005)

    def test_a_fat_edge_tolerates_far_more_failure(self):
        """Where this idea would actually pay: a 0.50 edge needs only 29%."""
        assert break_even_completion_rate(0.50, 0.20) < 0.30

    def test_the_rule_matches_the_break_even(self):
        edge = 0.03
        p = break_even_completion_rate(edge, 0.20)
        assert should_rest_early(edge, p + 0.02, 0.20) is True
        assert should_rest_early(edge, p - 0.02, 0.20) is False


class TestItRefusesToGuess:

    def test_without_a_completion_rate_it_stays_off(self):
        """
        No default. The completion rate is the input that decides how much money
        is at risk, and nothing has measured it — the fill log now records the
        prediction beside the outcome, and that is where the number comes from.
        """
        assert should_rest_early(0.03) is False
        assert should_rest_early(10.0) is False, "not even a huge edge overrides it"

    def test_an_explicit_rate_is_honoured(self):
        assert should_rest_early(0.03, p_complete=0.99, naked_cost=0.20) is True

    def test_a_probability_outside_zero_to_one_is_clamped(self):
        assert early_rest_expected_value(0.03, 1.5, 0.20) == pytest.approx(0.03)
        assert early_rest_expected_value(0.03, -0.5, 0.20) == pytest.approx(-0.20)
