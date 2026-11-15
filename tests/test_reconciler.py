"""
WalletReconciler — the check that does not trust the bot's own account.

Every serious fault of 2026-09-05/06 was internally consistent in the logs and
plainly wrong in the wallet. The invariant here is deliberately narrow:
inventory must not move while the bot believes it has nothing open.
"""

from __future__ import annotations

import pytest

from execution.reconciler import WalletReconciler


class _Client:
    def __init__(self, rows, cash=100.0):
        self.rows, self.cash = rows, cash

    async def open_positions_detail(self):
        return self.rows

    async def collateral_balance(self):
        return self.cash


class _Breaker:
    def __init__(self, open_positions=0):
        self._open = open_positions

    def status_dict(self):
        return {"open_positions": self._open}


class _Notifier:
    def __init__(self):
        self.sent = []

    async def notify(self, msg):
        self.sent.append(msg)


def _row(title, outcome, size):
    return {"title": title, "outcome": outcome, "size": size}


def _rig(rows, cash=100.0, open_positions=0):
    c, b, n = _Client(rows, cash), _Breaker(open_positions), _Notifier()
    return WalletReconciler(c, b, n, poll_s=10.0), c, b, n


@pytest.mark.asyncio
async def test_first_pass_only_takes_a_baseline():
    r, *_ = _rig([_row("Market A", "No", 10.0)])
    assert await r.check_once() is None


@pytest.mark.asyncio
async def test_quiet_wallet_reports_nothing():
    r, c, _, n = _rig([_row("Market A", "No", 10.0)])
    await r.check_once()
    n.sent.clear()          # discard the startup unmanaged-inventory notice
    assert await r.check_once() is None
    assert n.sent == []


@pytest.mark.asyncio
async def test_inventory_moving_with_nothing_open_is_flagged():
    """
    The live signature: guards released every slot and reported nothing filled
    while abandoned orders kept getting hit.
    """
    r, c, _, n = _rig([_row("Beriont", "No", 35.49)])
    await r.check_once()
    c.rows = [_row("Beriont", "No", 65.91)]      # +30.42 with nothing watching
    c.cash = 100.0 - 29.22
    report = await r.check_once()

    assert report is not None
    assert r.unexplained_events == 1
    assert n.sent, "an unsupervised fill must raise an alert"
    assert "Beriont" in n.sent[0]


@pytest.mark.asyncio
async def test_movement_is_expected_while_a_position_is_open():
    """No false alarm when the bot is legitimately trading."""
    r, c, b, n = _rig([_row("Beriont", "No", 10.0)], open_positions=1)
    await r.check_once()
    n.sent.clear()
    c.rows = [_row("Beriont", "No", 20.0)]
    assert await r.check_once() is None
    assert n.sent == []


@pytest.mark.asyncio
async def test_a_new_position_appearing_counts_as_movement():
    r, c, *_ = _rig([_row("A", "No", 5.0)])
    await r.check_once()
    c.rows = [_row("A", "No", 5.0), _row("B", "No", 7.0)]
    report = await r.check_once()
    assert report and any("B" in k for k in report["moves"])


@pytest.mark.asyncio
async def test_a_position_disappearing_counts_as_movement():
    r, c, *_ = _rig([_row("A", "No", 5.0)])
    await r.check_once()
    c.rows = []
    report = await r.check_once()
    assert report and report["moves"]


@pytest.mark.asyncio
async def test_dust_moves_are_ignored():
    """Share counts carry rounding; sub-0.01 drift is not a trade."""
    r, c, _, n = _rig([_row("A", "No", 5.0)])
    await r.check_once()
    n.sent.clear()
    c.rows = [_row("A", "No", 5.001)]
    assert await r.check_once() is None
    assert n.sent == []


@pytest.mark.asyncio
async def test_an_unreadable_snapshot_is_not_an_alert():
    """A failed read must never be reported as a phantom fill."""
    r, c, _, n = _rig([_row("A", "No", 5.0)])
    await r.check_once()
    n.sent.clear()

    async def _fail():
        return None
    c.open_positions_detail = _fail
    assert await r.check_once() is None
    assert n.sent == []


@pytest.mark.asyncio
async def test_the_reconciler_never_trades():
    """It has no order surface at all — the point is that it only observes."""
    r, c, *_ = _rig([_row("A", "No", 5.0)])
    for forbidden in ("post_order", "unwind_leg", "cancel_order", "cancel_all_orders"):
        assert not hasattr(c, forbidden)
    await r.check_once()


@pytest.mark.asyncio
async def test_positions_held_at_startup_are_reported_as_unmanaged():
    """
    The guards only resolve bundles they registered, and their watch list is in
    memory — empty after every restart. Anything already held is therefore
    nobody's: no guard will flatten it, complete it, or notice it. The positions
    left by the 2026-09-05 leak sat untouched for 37 hours exactly this way.
    """
    r, _, _, n = _rig([_row("Beriont", "No", 25.36), _row("Sullivan", "No", 20.28)])
    assert await r.check_once() is None      # baseline pass returns no report
    assert n.sent, "startup inventory must be surfaced"
    assert "Beriont" in n.sent[0] and "Sullivan" in n.sent[0]


@pytest.mark.asyncio
async def test_a_clean_start_says_nothing():
    r, _, _, n = _rig([])
    await r.check_once()
    assert n.sent == []


@pytest.mark.asyncio
async def test_a_redemption_is_not_reported_as_an_escape():
    """
    Regression for 2026-09-08 07:43.

        RECONCILE | inventory moved with NOTHING open — an order escaped
        supervision | cash +20.2900 | Will Maura Sullivan ... |No -20.29

    Nothing escaped. The market resolved and AutoRedeemer redeemed the winning
    position: shares to zero, cash up by their face value. This check is the
    only thing that catches a real leak, and one that cries wolf stops being
    read — so a settlement must not look like a fault.
    """
    r, c, _, n = _rig([_row("Will Maura Sullivan be the Democratic nominee for NH-01?", "No", 20.29)], cash=57.23)
    await r.check_once()
    n.sent.clear()
    r.note_redemption("Will Maura Sullivan be the Democratic nominee for NH-01?")   # as market_titles reports it  # AutoRedeemer says so
    c.rows = []                      # redeemed away
    c.cash = 57.23 + 20.29           # paid out at 1.00/share
    assert await r.check_once() is None
    assert n.sent == [], "a settlement was reported as an escaped order"


@pytest.mark.asyncio
async def test_a_real_escape_is_still_caught_alongside_a_redemption():
    """One leg settling must not mask another leg being bought unsupervised."""
    r, c, _, n = _rig([_row("Will Maura Sullivan be the Democratic nominee for NH-01?", "No", 20.29)], cash=57.23)
    await r.check_once()
    n.sent.clear()
    r.note_redemption("Will Maura Sullivan be the Democratic nominee for NH-01?")   # as market_titles reports it
    c.rows = [_row("Sabalenka", "No", 10.0)]      # appeared from nowhere
    c.cash = 57.23 + 20.29 - 6.96
    report = await r.check_once()
    assert report is not None
    assert any("Sabalenka" in k for k in report["moves"])
    assert not any("Sullivan" in k for k in report["moves"])


@pytest.mark.asyncio
async def test_shares_vanishing_without_cash_is_still_an_alert():
    """Position gone and no money arrived is not a redemption — it is a loss."""
    r, c, _, n = _rig([_row("A", "No", 20.0)], cash=50.0)
    await r.check_once()
    n.sent.clear()
    c.rows = []
    c.cash = 50.0                    # nothing came back
    assert await r.check_once() is not None
    assert n.sent


@pytest.mark.asyncio
async def test_an_unannounced_disappearance_still_alerts():
    """
    Only AutoRedeemer can vouch for a settlement. A position that vanishes with
    no announcement is exactly the leak this check exists for, whatever the cash
    happens to have done.
    """
    r, c, _, n = _rig([_row("A", "No", 20.0)], cash=50.0)
    await r.check_once()
    n.sent.clear()
    c.rows = []
    c.cash = 70.0                    # looks like a payout, but nobody said so
    assert await r.check_once() is not None
    assert n.sent


@pytest.mark.asyncio
async def test_an_announcement_is_consumed_not_reusable():
    """A second disappearance must not ride on the first one's notice."""
    r, c, _, n = _rig([_row("A", "No", 20.0), _row("B", "No", 5.0)], cash=50.0)
    await r.check_once()
    n.sent.clear()
    r.note_redemption("A")
    c.rows = []                      # BOTH vanished, only one was announced
    c.cash = 75.0
    report = await r.check_once()
    assert report is not None, "the unannounced one must still be reported"
    assert any("B" in k for k in report["moves"])


# ── the bot's own orders, settling after the position closed ─────────────────

class _TradingClient(_Client):
    """A client that remembers which tokens the bot traded, as PolyClient does."""

    def __init__(self, rows, cash=100.0, traded=None):
        super().__init__(rows, cash)
        self.traded = dict(traded or {})          # token -> unix time

    def traded_since(self, ts):
        return {t for t, at in self.traded.items() if at >= ts}


def _arow(title, outcome, size, asset):
    return {**_row(title, outcome, size), "asset": asset}


@pytest.mark.asyncio
async def test_the_bots_own_completion_landing_a_read_late_is_not_an_escape():
    """
    2026-09-23 10:16: a NegRisk bundle completed and closed; the completion's
    10.30 Republicans NO reached the data API at the next read, with nothing
    open, and was reported as an escaped order.
    """
    import time as _time
    c = _TradingClient([_arow("Texas governor D", "No", 10.28, "tokD")])
    r = WalletReconciler(c, _Breaker(0), _Notifier(), poll_s=120.0)
    await r.check_once()
    c.rows = c.rows + [_arow("Texas governor R", "No", 10.30, "tokR")]
    c.cash -= 3.1765
    c.traded["tokR"] = _time.time() - 125.0      # bought two minutes ago
    assert await r.check_once() is None
    assert r.unexplained_events == 0


@pytest.mark.asyncio
async def test_a_move_on_a_token_the_bot_did_not_trade_still_alarms():
    import time as _time
    c = _TradingClient([_arow("Market A", "No", 10.0, "tokA")],
                       traded={"tokA": _time.time()})
    r = WalletReconciler(c, _Breaker(0), _Notifier(), poll_s=120.0)
    await r.check_once()
    c.rows = [_arow("Market A", "No", 10.0, "tokA"), _arow("Market B", "Yes", 20.0, "tokB")]
    report = await r.check_once()
    assert report is not None and list(report["moves"]) == ["Market B|Yes"]


@pytest.mark.asyncio
async def test_a_token_traded_long_ago_does_not_explain_a_new_move():
    import time as _time
    c = _TradingClient([_arow("Market A", "No", 10.0, "tokA")],
                       traded={"tokA": _time.time() - 3_600.0})
    r = WalletReconciler(c, _Breaker(0), _Notifier(), poll_s=120.0)
    await r.check_once()
    c.rows = [_arow("Market A", "No", 30.0, "tokA")]
    assert await r.check_once() is not None


@pytest.mark.asyncio
async def test_a_sale_that_empties_a_position_is_attributed_by_its_last_token():
    """2026-09-23 11:46: the unwind sold all 10.30 TX-35 NO; the row vanished."""
    import time as _time
    c = _TradingClient([_arow("TX-35 R", "No", 10.30, "tok35")])
    r = WalletReconciler(c, _Breaker(0), _Notifier(), poll_s=120.0)
    await r.check_once()
    c.rows = []
    c.traded["tok35"] = _time.time() - 110.0
    assert await r.check_once() is None


# ── resolution: redeemable rows and the exchange's own operator ───────────────

def _resolved(title, outcome, size, value=0.0):
    return {**_row(title, outcome, size), "redeemable": True, "value": value}


@pytest.mark.asyncio
async def test_a_market_resolving_is_not_shares_moving():
    """A loser stays listed for good. It left live inventory without any order."""
    r, c, _, n = _rig([_row("A", "No", 20.0)])
    await r.check_once()
    n.sent.clear()
    c.rows = [_resolved("A", "No", 20.0)]
    assert await r.check_once() is None
    assert n.sent == []


@pytest.mark.asyncio
async def test_a_redeemable_row_appearing_is_not_shares_appearing():
    """A cross leg leaving the managed set after its position closed."""
    r, c, _, n = _rig([])
    await r.check_once()
    c.rows = [_resolved("Brighton", "No", 10.0)]
    assert await r.check_once() is None
    assert n.sent == []


@pytest.mark.asyncio
async def test_a_resolved_row_excuses_only_itself():
    r, c, _, n = _rig([_row("A", "No", 20.0), _row("B", "Yes", 5.0)])
    await r.check_once()
    n.sent.clear()
    c.rows = [_resolved("A", "No", 20.0)]                # B vanished with no reason
    report = await r.check_once()
    assert report is not None and list(report["moves"]) == ["B|Yes"]


class _FeedClient(_Client):
    def __init__(self, rows, feed, cash=100.0):
        super().__init__(rows, cash)
        self.feed = feed
        self.asked = 0

    async def recent_redemptions(self, since):
        self.asked += 1
        return self.feed


def _feed_rig(rows, feed):
    c, b, n = _FeedClient(rows, feed), _Breaker(), _Notifier()
    return WalletReconciler(c, b, n, poll_s=10.0), c, n


@pytest.mark.asyncio
async def test_the_operator_redeeming_between_two_reads_is_not_a_leak():
    """
    Regression for what would have happened on 2026-11-04 with the reconciler able
    to see: the operator redeemed within seconds of the market closing, so the shares
    went from held to gone with nothing here told about it.
    """
    title = "Will the Republicans win the Texas Senate race in 2026?"
    r, c, n = _feed_rig([_row(title, "No", 20.44)], [])
    await r.check_once()
    n.sent.clear()
    c.rows = []
    c.feed = [{"title": title, "outcome": "No", "usdc": 20.44}]
    assert await r.check_once() is None
    assert n.sent == []


@pytest.mark.asyncio
async def test_the_feed_excuses_only_the_position_it_names():
    r, c, n = _feed_rig([_row("A", "No", 20.0), _row("B", "No", 5.0)], [])
    await r.check_once()
    n.sent.clear()
    c.rows = []
    c.feed = [{"title": "A", "outcome": "No", "usdc": 20.0}]
    report = await r.check_once()
    assert report is not None and list(report["moves"]) == ["B|No"]


@pytest.mark.asyncio
async def test_the_same_market_on_the_other_outcome_is_not_excused():
    r, c, n = _feed_rig([_row("A", "Yes", 5.0)], [])
    await r.check_once()
    c.rows = []
    c.feed = [{"title": "A", "outcome": "No", "usdc": 20.0}]
    assert await r.check_once() is not None


@pytest.mark.asyncio
async def test_the_feed_never_excuses_shares_appearing():
    r, c, n = _feed_rig([], [{"title": "A", "outcome": "No", "usdc": 20.0}])
    await r.check_once()
    c.rows = [_row("A", "No", 5.0)]
    assert await r.check_once() is not None
    assert c.asked == 0, "nothing vanished, so there is nothing to ask the feed about"


@pytest.mark.asyncio
async def test_a_failed_feed_read_still_alerts():
    """Better a false alarm than a leak passing because the feed was down."""
    r, c, n = _feed_rig([_row("A", "No", 20.0)], None)
    await r.check_once()
    n.sent.clear()
    c.rows = []
    assert await r.check_once() is not None
    assert n.sent


# ── the start-up report ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_worthless_leftovers_do_not_page_anyone_at_startup():
    """11 positions were reported on 2026-11-02, every one of them worth nothing."""
    r, _, _, n = _rig([_resolved("Galway O/U 7.5", "Over", 5.25),
                       {**_row("Treasury 5.5%", "Yes", 0.37), "value": 0.04}])
    await r.check_once()
    assert n.sent == []


@pytest.mark.asyncio
async def test_a_position_worth_something_is_still_reported_at_startup():
    r, _, _, n = _rig([{**_row("Galway O/U 7.5", "Over", 5.25), "value": 0.0},
                       {**_row("Texas", "No", 10.0), "value": 8.0}])
    await r.check_once()
    assert len(n.sent) == 1 and "Texas" in n.sent[0] and "Galway" not in n.sent[0]


@pytest.mark.asyncio
async def test_small_positions_are_still_watched():
    r, c, _, n = _rig([{**_row("Treasury 5.5%", "Yes", 0.37), "value": 0.04}])
    await r.check_once()
    c.rows = []
    assert await r.check_once() is not None
