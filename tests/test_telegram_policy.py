"""
TELEGRAM_TRADES_ONLY: the chat carries transactions that happened, nothing else.

Asked for on 2026-10-13. By then the chat was a feed of arbitrage notices that
never traded — resting orders every three minutes, episode summaries reading
"nothing — no execution attempted", the reader's signal alerts — and the few
messages about real fills were lost among them.
"""

from __future__ import annotations

import asyncio

import pytest

from telemetry.telegram import TelegramNotifier, message_kind, says_traded


@pytest.mark.parametrize("text,kind", [
    # ── a transaction happened ──
    ("💰 CROSS POSITION OPENED\n  NO  X @ 0.56\n  YES Y @ 0.26", "trade"),
    ("💰 CROSS POSITION OPENED (maker + taker completion)\n  NO  X", "trade"),
    ("⚠️ CROSS HALF-FILL — YES on broad did not fill; NO on narrow sold back, -0.12 USDC.", "trade"),
    ("⚠️ CROSS MAKER HALF-FILL — 5.24 shares filled on one leg only, sold back for -0.05 USDC.", "trade"),
    ("🚨 CROSS MAKER HALF-FILL STUCK — 5.24 shares of YES X filled alone", "trade"),
    ("🔓 CROSS POSITION EXITED EARLY — +0.21 USDC after 1.2d", "trade"),
    ("🏁 CROSS POSITION RESOLVED — +0.20 USDC", "trade"),
    ("<b>ARB EXECUTED</b>\n<b>Market:</b> X", "trade"),
    ("⚠️ Half-fill on 0xabc (YES_ONLY) — naked leg flattened", "trade"),
    ("CTF REDEEMED\ncondition=0xabc...", "trade"),
    ("✅ NegRisk bundle on Texas completed as taker — 10.28 bundles, profit=+0.0164 USDC", "trade"),
    # ── nothing traded, but someone has to look ──
    ("<b>CRITICAL ERROR</b>\n<code>daily loss limit hit</code>", "critical"),
    ("All orders cancelled. Bot halted.", "critical"),
    ("⚠️ RECONCILE — inventory changed while the bot had nothing open.", "critical"),
    ("⚠️ CROSS MAKER — the exchange would not cancel 0xabc for 10 min", "critical"),
    ("⚠️ CROSS — 4.99 shares of YES X still held 10 min after a sell-back", "critical"),
    ("ARB EXECUTION ERROR: timeout", "critical"),
    # ── neither ──
    ("🪧 CROSS MAKER ORDERS RESTING\n  NO  X @ 0.41\n  YES Y @ 0.51", "info"),
    ("Polymarket ARB Bot v7 online\nbalance=73.65 USDC", "info"),
    ("<b>📊 Polybot — Daily Heartbeat</b>", "info"),
    ("<b>ARB DETECTED</b> (TAKER)\n<b>Market:</b> X", "info"),
    ("<b>ARB WINDOW CLOSED</b> (MAKER)", "info"),
])
def test_what_each_message_is_about(text, kind):
    assert message_kind(text) == kind


@pytest.mark.parametrize("line,traded", [
    ("✅ both legs filled at submission — 10.30 pairs", True),
    ("⚠️ half-fill (YES_ONLY) — naked leg flattened, no profit booked", True),
    ("🔻 Maker pair on 0xabc FAILED — no shares paired. Directional result, pnl=-0.16 USDC", True),
    ("🔻 NegRisk bundle on 0xabc incomplete — 1/2 legs filled, unwound, pnl=-0.05 USDC", True),
    ("no fill — both FOK legs killed, no exposure", False),
    ("⏹ Maker pair on 0xabc rested 900s with no fill — cancelled", False),
    ("❌ execution failed — timeout", False),
    ("bundle expired unfilled — releasing slot", False),
])
def test_an_outcome_line_trades_or_does_not(line, traded):
    assert says_traded(line) is traded


def _notifier(monkeypatch, trades_only="true", critical=None):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setenv("TELEGRAM_TRADES_ONLY", trades_only)
    if critical is None:
        monkeypatch.delenv("TELEGRAM_CRITICAL", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_CRITICAL", critical)
    n = TelegramNotifier()
    n._enabled = True
    n.posted = []

    class _Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        closed = False

        def post(self, url, json=None):
            n.posted.append(json["text"])
            return _Resp()

    async def _get_session():
        return _Session()
    n._get_session = _get_session
    return n


@pytest.mark.asyncio
async def test_only_transactions_reach_the_chat(monkeypatch):
    n = _notifier(monkeypatch)
    assert await n.notify("🪧 CROSS MAKER ORDERS RESTING\n  NO  X @ 0.41") is False
    assert await n.notify("Polymarket ARB Bot v7 online") is False
    assert await n.notify("💰 CROSS POSITION OPENED\n  NO  X @ 0.56") is True
    assert await n.notify("⚠️ CROSS HALF-FILL — YES did not fill; NO sold back") is True
    assert [t.split("\n")[0] for t in n.posted] == [
        "💰 CROSS POSITION OPENED", "⚠️ CROSS HALF-FILL — YES did not fill; NO sold back"]


@pytest.mark.asyncio
async def test_critical_alerts_still_go_out_unless_switched_off(monkeypatch):
    n = _notifier(monkeypatch)
    assert await n.notify("All orders cancelled. Bot halted.") is True
    quiet = _notifier(monkeypatch, critical="false")
    assert await quiet.notify("All orders cancelled. Bot halted.") is False
    assert await quiet.notify("💰 CROSS POSITION OPENED\n  NO  X") is True


@pytest.mark.asyncio
async def test_without_the_switch_everything_goes_out_as_before(monkeypatch):
    n = _notifier(monkeypatch, trades_only="false")
    assert await n.notify("🪧 CROSS MAKER ORDERS RESTING") is True
    assert await n.notify("Polymarket ARB Bot v7 online") is True


@pytest.mark.asyncio
async def test_a_caller_can_say_what_its_message_is(monkeypatch):
    n = _notifier(monkeypatch)
    assert await n.notify("bought 5 of X", kind="trade") is True
    assert await n.notify("💰 CROSS POSITION OPENED", kind="info") is False


def _detect(n, cid):
    n.arb_detected(condition_id=cid, combined_cost=0.94, net_edge=0.06,
                   is_maker=True, yes_price=0.17, no_price=0.77)


@pytest.mark.asyncio
async def test_an_episode_is_sent_only_if_something_traded_in_it(monkeypatch):
    n = _notifier(monkeypatch)
    n._arb_min_bps = 0.0
    # 1. a signal a gate refused
    _detect(n, "0xa")
    n.arb_skipped("0xa", "market busy")
    n.send_arb_summary("0xa", 4.0, 600.0, 3, True)
    # 2. orders that rested and expired
    _detect(n, "0xb")
    n.arb_event("0xb", "⏹ Maker pair on 0xb rested 900s with no fill — cancelled")
    n.send_arb_summary("0xb", 4.0, 600.0, 3, True)
    # 3. a leg that filled and was sold back
    _detect(n, "0xc")
    n.arb_event("0xc", "🔻 Maker pair on 0xc FAILED — no shares paired.", pnl=-0.16)
    n.send_arb_summary("0xc", 4.0, 600.0, 3, True)
    await asyncio.sleep(0.05)
    assert len(n.posted) == 1 and "0xc" in n.posted[0] and "-0.1600" in n.posted[0]
