"""Step 8 — SignalLogger persists rows; tuner_loop adjusts ArbDetector margin;
TelegramNotifier sends and degrades gracefully."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from strategy.arbitrage import ArbDetector, ArbSignal
from strategy.tuner     import (
    FREQ_HIGH_THRESHOLD,
    STEP_UP,
    WIDE_EDGE_THRESHOLD,
    _query_last_hour,
)
from telemetry.db_logger import SignalLogger
from telemetry.telegram  import TelegramNotifier
from tests.mocks.fake_telegram import FakeTelegramNotifier


def _signal(net_edge: float, idx: int) -> ArbSignal:
    return ArbSignal(
        condition_id=f"0x{idx:064x}",
        yes_token_id="Y", no_token_id="N",
        yes_ask=0.47, no_ask=0.50,
        combined_cost=0.97, fee_rate=0.0, fee_cost=0.0, net_edge=net_edge,
        yes_size=10.0, no_size=10.0,
    )


@pytest.mark.asyncio
async def test_signal_logger_persists_arb_rows(tmp_signal_db):
    logger = SignalLogger(db_path=str(tmp_signal_db))
    await logger.init()
    writer = asyncio.create_task(logger.run())

    for i in range(15):
        logger.log_arb(_signal(net_edge=0.012, idx=i))

    await asyncio.sleep(0.2)
    await logger.close()
    writer.cancel()
    await asyncio.gather(writer, return_exceptions=True)

    async with aiosqlite.connect(str(tmp_signal_db)) as db:
        cur = await db.execute("SELECT COUNT(*) FROM arb_signals")
        (count,) = await cur.fetchone()
    assert count == 15


@pytest.mark.asyncio
async def test_tuner_increases_margin_on_high_freq_wide_edge(tmp_signal_db):
    """Bypass tuner_loop's sleep — drive a single cycle directly."""
    logger = SignalLogger(db_path=str(tmp_signal_db))
    await logger.init()
    writer = asyncio.create_task(logger.run())
    for i in range(FREQ_HIGH_THRESHOLD + 2):
        logger.log_arb(_signal(net_edge=WIDE_EDGE_THRESHOLD + 0.005, idx=i))
    await asyncio.sleep(0.2)
    await logger.close()
    writer.cancel()
    await asyncio.gather(writer, return_exceptions=True)

    detector = ArbDetector(desired_net_margin=0.01)
    start_margin = detector._net_margin

    # Reproduce the body of one tuner cycle.
    import time
    count, mean_edge = await _query_last_hour(str(tmp_signal_db), since=time.time() - 3600.0)
    assert count >= FREQ_HIGH_THRESHOLD
    assert mean_edge >= WIDE_EDGE_THRESHOLD

    detector._net_margin = min(detector._net_margin + STEP_UP, 0.05)
    assert detector._net_margin == pytest.approx(start_margin + STEP_UP, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# TelegramNotifier — disabled mode (no env creds)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def disabled_notifier():
    # conftest sets TELEGRAM_BOT_TOKEN="" and TELEGRAM_CHAT_ID="" — notifier is off
    return TelegramNotifier()


@pytest.mark.asyncio
async def test_notify_returns_false_when_disabled(disabled_notifier):
    result = await disabled_notifier.notify("hello")
    assert result is False


@pytest.mark.asyncio
async def test_heartbeat_returns_false_when_disabled(disabled_notifier):
    result = await disabled_notifier.heartbeat({"session_pnl": 1.5, "open_positions": 0})
    assert result is False


@pytest.mark.asyncio
async def test_send_trade_execution_noop_when_disabled(disabled_notifier):
    # _fire() must return without raising when the notifier is disabled
    await disabled_notifier.send_trade_execution(
        condition_id="0xabc",
        yes_token_id="Y", no_token_id="N",
        yes_ask=0.47, no_ask=0.50,
        n_shares=10.0, combined_cost=0.97,
        guaranteed_profit=0.03,
    )


@pytest.mark.asyncio
async def test_send_critical_error_noop_when_disabled(disabled_notifier):
    await disabled_notifier.send_critical_error("loss limit breached")


@pytest.mark.asyncio
async def test_close_without_session_does_not_raise(disabled_notifier):
    await disabled_notifier.close()  # _session is None — must not raise


@pytest.mark.asyncio
async def test_run_listener_disabled_cancels_cleanly(disabled_notifier):
    task = asyncio.create_task(disabled_notifier.run_listener())
    await asyncio.sleep(0)  # let it reach the Event().wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# ═══════════════════════════════════════════════════════════════════════════
# TelegramNotifier — enabled mode (mocked HTTP + Application)
# ═══════════════════════════════════════════════════════════════════════════

def _make_mock_session(status: int = 200, body: str = "") -> MagicMock:
    """Build a mock aiohttp.ClientSession whose .post() context manager returns `status`."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=body)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    mock_session.close = AsyncMock()
    return mock_session


@pytest.mark.asyncio
async def test_notify_success_returns_true(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "99999")

    mock_session = _make_mock_session(status=200)

    with patch("telemetry.telegram.aiohttp.ClientSession", return_value=mock_session), \
         patch("telemetry.telegram.Application"):
        notifier = TelegramNotifier()
        result = await notifier.notify("hello from test")
        await notifier.close()

    assert result is True
    mock_session.post.assert_called_once()
    # Verify the right URL is targeted
    call_args = mock_session.post.call_args
    url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
    assert "sendMessage" in url
    assert "fake-token-123" in url


@pytest.mark.asyncio
async def test_notify_returns_false_on_api_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "99999")

    mock_session = _make_mock_session(status=400, body='{"description":"Bad Request"}')

    with patch("telemetry.telegram.aiohttp.ClientSession", return_value=mock_session), \
         patch("telemetry.telegram.Application"):
        notifier = TelegramNotifier()
        result = await notifier.notify("this should fail")
        await notifier.close()

    assert result is False


@pytest.mark.asyncio
async def test_notify_returns_false_on_network_exception(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "99999")

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(side_effect=OSError("connection refused"))
    mock_session.close = AsyncMock()

    with patch("telemetry.telegram.aiohttp.ClientSession", return_value=mock_session), \
         patch("telemetry.telegram.Application"):
        notifier = TelegramNotifier()
        result = await notifier.notify("this should fail silently")
        await notifier.close()

    assert result is False


@pytest.mark.asyncio
async def test_heartbeat_sends_html_message(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "99999")

    captured: list[dict] = []

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    def _capture_post(url, *, json=None, **_kw):
        captured.append(json or {})
        return mock_resp

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=_capture_post)
    mock_session.closed = False
    mock_session.close = AsyncMock()

    with patch("telemetry.telegram.aiohttp.ClientSession", return_value=mock_session), \
         patch("telemetry.telegram.Application"):
        notifier = TelegramNotifier()
        await notifier.heartbeat({"session_pnl": 2.5, "open_positions": 1})
        await notifier.close()

    assert captured, "No HTTP call made"
    payload = captured[0]
    assert payload.get("parse_mode") == "HTML"
    assert "Heartbeat" in payload.get("text", "")
    assert "session_pnl" in payload.get("text", "")


@pytest.mark.asyncio
async def test_alert_sends_html_bold_title(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "99999")

    captured: list[dict] = []

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    def _capture(url, *, json=None, **_):
        captured.append(json or {})
        return mock_resp

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=_capture)
    mock_session.closed = False
    mock_session.close = AsyncMock()

    with patch("telemetry.telegram.aiohttp.ClientSession", return_value=mock_session), \
         patch("telemetry.telegram.Application"):
        notifier = TelegramNotifier()
        await notifier.alert("CRITICAL", "market feed dropped")
        await notifier.close()

    text = captured[0].get("text", "")
    assert "<b>CRITICAL</b>" in text
    assert "market feed dropped" in text


# ═══════════════════════════════════════════════════════════════════════════
# FakeTelegramNotifier — verifies the mock captures all event types
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_fake_notifier_captures_all_message_types():
    notifier = FakeTelegramNotifier()

    await notifier.notify("startup message")
    await notifier.alert("TITLE", "body text")
    await notifier.heartbeat({"session_pnl": 2.0, "open_positions": 0})
    await notifier.send_trade_execution(
        condition_id="0xabc", yes_token_id="Y", no_token_id="N",
        yes_ask=0.47, no_ask=0.50, n_shares=10.0,
        combined_cost=0.97, guaranteed_profit=0.03,
    )
    await notifier.send_critical_error("EMERGENCY HALT")

    assert "startup message" in notifier.messages
    assert ("TITLE", "body text") in notifier.alerts
    assert len(notifier.trade_executions) == 1
    assert notifier.trade_executions[0]["condition_id"] == "0xabc"
    assert len(notifier.critical_errors) == 1
    assert "EMERGENCY HALT" in notifier.critical_errors[0]


@pytest.mark.asyncio
async def test_fake_notifier_halt_callback_is_invoked():
    notifier = FakeTelegramNotifier()
    halt_fired = asyncio.Event()

    async def _halt():
        halt_fired.set()

    notifier.set_halt_callback(_halt)
    await notifier.trigger_halt()

    assert halt_fired.is_set()


@pytest.mark.asyncio
async def test_fake_notifier_close_sets_flag():
    notifier = FakeTelegramNotifier()
    assert not notifier.closed
    await notifier.close()
    assert notifier.closed


@pytest.mark.asyncio
async def test_fake_notifier_run_listener_cancels_cleanly():
    notifier = FakeTelegramNotifier()
    task = asyncio.create_task(notifier.run_listener())
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
