"""
Step 7 — AutoRedeemer reports what Polymarket's operator redeemed, and redeems
what it left behind.

On 2026-11-04 the operator redeemed three winning positions for 41.71 pUSD within
seconds of each market closing. The bot's old scan was not involved: it read Gamma
fields that no longer exist and looked for USDC.e position ids no pUSD position has.
"""

from __future__ import annotations

import pytest

from execution.auto_redeem import AutoRedeemer

T0 = 1_793_800_000.0
GRACE = 1800.0


class _Client:
    def __init__(self, rows=(), feed=()):
        self.rows = list(rows)
        self.feed = list(feed)
        self.redeemed: list[str] = []
        self.raises: "Exception | None" = None
        self.asked_since: list[float] = []

    async def open_positions_detail(self):
        return self.rows

    async def recent_redemptions(self, since):
        self.asked_since.append(since)
        return self.feed

    async def redeem_positions(self, condition_id):
        if self.raises:
            raise self.raises
        self.redeemed.append(condition_id)
        return "0xtx"


class _Notifier:
    def __init__(self):
        self.sent: list[str] = []

    async def notify(self, msg):
        self.sent.append(msg)


def _rig(rows=(), feed=(), **kw):
    client, notifier = _Client(rows, feed), _Notifier()
    announced: list[str] = []
    r = AutoRedeemer(notifier, client, on_redeemed=announced.append, grace_s=GRACE, **kw)
    r._started_at = T0 - 60.0
    return r, client, notifier, announced


def _winner(cid="0xwin", value=5.0, title="Abha O/U 0.5", outcome="Over"):
    return {"title": title, "outcome": outcome, "condition_id": cid, "size": value,
            "value": value, "redeemable": True}


def _loser(cid="0xlose"):
    return {"title": "Galway O/U 7.5", "outcome": "Over", "condition_id": cid, "size": 5.25,
            "value": 0.0, "redeemable": True}


def _paid(usdc=20.44, tx="0xaaa", cid="0xwin", outcome="No"):
    return {"title": "Will the Republicans win the Texas Senate race in 2026?",
            "outcome": outcome, "usdc": usdc, "tx": tx, "condition_id": cid, "ts": int(T0)}


# ── reports ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_redemption_by_the_operator_is_reported_once():
    r, _, n, _ = _rig(feed=[_paid()])
    await r.check_once(now=T0)
    await r.check_once(now=T0 + 120)
    assert len(n.sent) == 1
    assert "CTF REDEEMED" in n.sent[0] and "20.44" in n.sent[0] and "Texas Senate" in n.sent[0]


@pytest.mark.asyncio
async def test_the_report_counts_as_a_transaction_on_telegram():
    from telemetry.telegram import message_kind
    r, _, n, _ = _rig(feed=[_paid()])
    await r.check_once(now=T0)
    assert message_kind(n.sent[0]) == "trade"


@pytest.mark.asyncio
async def test_the_losing_side_burned_in_the_same_tx_is_not_news():
    r, _, n, _ = _rig(feed=[_paid(), _paid(usdc=0.0, outcome="Yes")])
    await r.check_once(now=T0)
    assert len(n.sent) == 1


@pytest.mark.asyncio
async def test_only_what_was_paid_since_this_run_started_is_read():
    r, c, *_ = _rig()
    await r.check_once(now=T0)
    assert c.asked_since == [T0 - 60.0]


@pytest.mark.asyncio
async def test_a_late_row_inside_the_window_is_still_reported():
    r, c, n, _ = _rig(feed=[_paid(tx="0xaaa")])
    await r.check_once(now=T0)
    c.feed = [_paid(tx="0xaaa"), _paid(tx="0xbbb")]
    await r.check_once(now=T0 + 120)
    assert len(n.sent) == 2


@pytest.mark.asyncio
async def test_a_failed_feed_read_changes_nothing():
    r, c, n, _ = _rig(feed=[_paid()])
    c.feed = None
    await r.check_once(now=T0)
    assert n.sent == []
    c.feed = [_paid()]
    await r.check_once(now=T0 + 120)
    assert len(n.sent) == 1


# ── stand-in ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_operator_gets_the_grace_period_first():
    r, c, *_ = _rig(rows=[_winner()])
    await r.check_once(now=T0)
    await r.check_once(now=T0 + GRACE - 1)
    assert c.redeemed == []


@pytest.mark.asyncio
async def test_a_winner_it_left_behind_is_redeemed_and_announced():
    r, c, _, announced = _rig(rows=[_winner()])
    await r.check_once(now=T0)
    await r.check_once(now=T0 + GRACE)
    assert c.redeemed == ["0xwin"]
    assert announced == ["Abha O/U 0.5"]


@pytest.mark.asyncio
async def test_a_loser_is_never_redeemed():
    """It pays nothing, and it stays listed as redeemable for good."""
    r, c, *_ = _rig(rows=[_loser()])
    await r.check_once(now=T0)
    await r.check_once(now=T0 + GRACE * 10)
    assert c.redeemed == []


@pytest.mark.asyncio
async def test_a_position_not_yet_resolved_is_never_redeemed():
    r, c, *_ = _rig(rows=[{**_winner(), "redeemable": False}])
    await r.check_once(now=T0)
    await r.check_once(now=T0 + GRACE * 10)
    assert c.redeemed == []


@pytest.mark.asyncio
async def test_a_winner_the_operator_got_to_needs_nothing_more():
    r, c, *_ = _rig(rows=[_winner()])
    await r.check_once(now=T0)
    c.rows = []                                     # paid out meanwhile
    await r.check_once(now=T0 + GRACE)
    assert c.redeemed == []
    assert r._first_seen == {}


@pytest.mark.asyncio
async def test_a_failed_attempt_waits_then_asks_a_person_once():
    from telemetry.telegram import message_kind
    r, c, n, announced = _rig(rows=[_winner()])
    c.raises = RuntimeError("relayer down")
    await r.check_once(now=T0)
    for i in range(1, 6):
        await r.check_once(now=T0 + GRACE + i * 3600.0)
    assert announced == []
    assert len(n.sent) == 1 and "REDEEM FAILED" in n.sent[0]
    assert message_kind(n.sent[0]) == "critical"


@pytest.mark.asyncio
async def test_attempts_are_spaced_out():
    r, c, *_ = _rig(rows=[_winner()])
    c.raises = RuntimeError("relayer down")
    tries = []
    orig = c.redeem_positions

    async def counting(cid):
        tries.append(cid)
        return await orig(cid)

    c.redeem_positions = counting
    await r.check_once(now=T0)
    await r.check_once(now=T0 + GRACE)
    await r.check_once(now=T0 + GRACE + 120)
    assert len(tries) == 1


@pytest.mark.asyncio
async def test_a_failed_wallet_read_touches_nothing():
    r, c, *_ = _rig(rows=[_winner()])
    await r.check_once(now=T0)
    c.rows = None
    await r.check_once(now=T0 + GRACE)
    assert c.redeemed == []
    assert "0xwin" in r._first_seen
