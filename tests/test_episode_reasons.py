"""
Every arb episode summary says what actually happened.

"What happened: nothing — no execution attempted" went out for most maker
episodes while the logs showed 15 of 20 maker signals placed. Four gaps in the
reporting, pinned here:
  1. the gates (busy market, breaker, size) refused a signal and said nothing;
  2. the episode started waiting for an outcome only once orders were
     acknowledged, 0.1–0.3 s after the signal, and a window closing in that gap
     reported "no execution attempted";
  3. a pair that rested and expired unfilled sent no outcome, so its summary
     never went out;
  4. an outcome arriving while the window was still open left the episode
     pending for good.
"""

from __future__ import annotations

import logging
import re

import pytest

from risk.circuit_breaker import CircuitBreaker
from tests.test_pair_guard import FakeGuardClient, _build_guard, _maker_signal, _resting_resp


def _notifier():
    import os
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy:token")
    os.environ.setdefault("TELEGRAM_CHAT_ID", "1")
    from telemetry.telegram import TelegramNotifier
    n = TelegramNotifier()
    n._enabled = True
    n._arb_min_bps = 0.0
    n._sent = []
    n._fire = lambda t, **k: n._sent.append(re.sub(r"</?(b|code)>", "", t))
    return n


def _detect(n, cid):
    n.arb_detected(condition_id=cid, combined_cost=0.94, net_edge=0.06,
                   is_maker=True, yes_price=0.17, no_price=0.77)


def test_a_refused_signal_says_why_not_nothing():
    n = _notifier()
    for _ in range(3):
        _detect(n, "0xa")
        n.arb_skipped("0xa", "market busy — cooling down after a losing unwind")
    n.send_arb_summary("0xa", 4.0, 600.0, 3, True)
    t = n._sent[0]
    assert "not placed 3× — market busy — cooling down after a losing unwind" in t
    assert "no execution attempted" not in t


def test_an_outcome_while_the_window_is_open_releases_the_summary_at_close():
    n = _notifier()
    _detect(n, "0xb")
    n.arb_execution_started("0xb")
    n.arb_event("0xb", "🔻 Maker pair FAILED — directional result", -0.15)
    assert n._sent == []                          # the window is still open
    n.send_arb_summary("0xb", 30.0, 600.0, 9, True)
    assert len(n._sent) == 1 and "FAILED" in n._sent[0] and "-0.1500" in n._sent[0]


def test_a_window_closing_while_orders_are_on_their_way_waits_for_them():
    """Pending from dispatch: the close in the 0.1–0.3 s before the ack waits."""
    n = _notifier()
    _detect(n, "0xc")
    n.arb_execution_started("0xc")               # at dispatch, before the ack
    n.send_arb_summary("0xc", 0.2, 600.0, 2, True)
    assert n._sent == []
    n.arb_event("0xc", "⏹ Maker pair rested 900s with no fill — cancelled")
    assert len(n._sent) == 1 and "no fill" in n._sent[0]
    assert "no execution attempted" not in n._sent[0]


def test_every_summary_sent_is_logged(caplog):
    n = _notifier()
    _detect(n, "0xd")
    n.arb_skipped("0xd", "blocked by the circuit breaker — max positions")
    with caplog.at_level(logging.INFO, logger="telemetry.telegram"):
        n.send_arb_summary("0xd", 2.0, 600.0, 1, True)
    assert any("ARB EPISODE sent" in r.getMessage() and "max positions" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_a_pair_that_expires_unfilled_reports_it():
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "live", "size_matched": 0.0}
    client.orders["on"] = {"status": "live", "size_matched": 0.0}
    guard, _, notifier = _build_guard(client, order_ttl_s=0.0)
    guard.watch_pair(_maker_signal(10.0), 10.0, _resting_resp("oy"), _resting_resp("on"))
    await guard.poll_once()
    assert any("no fill" in str(e) for e in notifier.events)


@pytest.mark.asyncio
async def test_the_guard_says_why_a_market_is_busy():
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "live", "size_matched": 0.0}
    client.orders["on"] = {"status": "live", "size_matched": 0.0}
    guard, _, _ = _build_guard(client)
    assert guard.busy_reason("0xcond") is None
    guard.watch_pair(_maker_signal(10.0), 10.0, _resting_resp("oy"), _resting_resp("on"))
    assert guard.busy_reason("0xcond") == "maker pair still resting"
    guard._pairs.clear()
    guard._cooldown["0xcond"] = 9e18
    assert guard.busy_reason("0xcond") == "cooling down after a losing unwind"


def test_the_breaker_keeps_its_last_reason(monkeypatch):
    import risk.circuit_breaker as cb
    b = CircuitBreaker(starting_balance=100.0)
    monkeypatch.setattr(b, "_arb_blocking_reason", lambda intent: "max open positions reached")
    assert b.check_arb(object()) is False
    assert b.last_block_reason == "max open positions reached"
