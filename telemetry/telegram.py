"""
telemetry/telegram.py
─────────────────────
Two-way async Telegram Bot — Arbitrage Edition.

Outbound (push notifications)
──────────────────────────────
Messages are sent via the raw Telegram Bot HTTP API through a persistent
aiohttp.ClientSession.  This path is fully independent of the command listener
and works from the first line of main() with zero initialisation overhead.

Inbound (command & control)
────────────────────────────
A python-telegram-bot v20 Application polls the Bot API for updates and
dispatches command handlers.  The polling loop runs inside `run_listener()`,
a long-running coroutine that is added to asyncio.gather alongside the trading
loops so it never blocks the hot path.

All inbound commands are gated on TELEGRAM_CHAT_ID.  Any update from a
different chat is silently dropped before handler logic executes.

Supported commands
──────────────────
  /status   — Returns current USDC balance (starting_balance + session_pnl),
               daily PnL from the CircuitBreaker, and count of open positions.
  /halt     — Manual kill switch: cancels all orders, sets the shutdown event,
               and cancels all asyncio tasks.  The callback is supplied by
               main.py via TelegramNotifier.set_halt_callback().

Parse-mode policy
─────────────────
All structured alerts use HTML parse mode.  HTML is safer than MarkdownV2 for
numeric bot output because floats, dashes, and USDC values contain characters
(- . +) that MarkdownV2 requires escaping.

Graceful degradation
────────────────────
If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are absent or blank the notifier
silently drops all messages and the command listener sleeps indefinitely.

Environment variables
─────────────────────
  TELEGRAM_BOT_TOKEN   bot token from @BotFather
  TELEGRAM_CHAT_ID     numeric chat / channel ID (authorised operator chat)
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
from typing import Callable, Coroutine, Optional

import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"


def _scrub(text: object) -> str:
    """
    Strip bot tokens out of text destined for the log.

    aiohttp/httpx embed the full request URL in their exception messages, and
    the Telegram API puts the bot token in the path -- so logging a raw
    exception leaks the credential into the journal.
    """
    out = str(text)
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if tok:
        out = out.replace(tok, "<REDACTED>")
    # belt and braces: catch any bot<digits>:<secret> pattern, incl. a stale token
    out = re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot<REDACTED>", out)
    return out


def _h(value: object) -> str:
    """HTML-escape a value for safe insertion into a Telegram HTML message."""
    return html.escape(str(value))


class TelegramNotifier:
    """
    Two-way async Telegram client.

    Outbound messages use a persistent aiohttp.ClientSession (low overhead,
    no initialisation dependency).  Inbound commands are handled by a
    python-telegram-bot Application whose polling loop runs inside
    `run_listener()`.

    Usage::

        notifier = TelegramNotifier(on_status=breaker.status_dict)
        notifier.set_halt_callback(do_halt_coroutine)

        # In asyncio.gather:
        await asyncio.gather(
            ...,
            notifier.run_listener(),
        )

        # Send from anywhere:
        await notifier.notify("Bot online")
        await notifier.close()
    """

    def __init__(
        self,
        *,
        on_status: Optional[Callable[[], dict]] = None,
    ) -> None:
        self._token:   Optional[str] = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or None
        self._chat_id: Optional[str] = os.environ.get("TELEGRAM_CHAT_ID",   "").strip() or None
        self._session: Optional[aiohttp.ClientSession] = None
        self._enabled  = bool(self._token and self._chat_id)

        # Callbacks — on_halt is set after construction to avoid circular refs
        self._on_status: Optional[Callable[[], dict]]           = on_status
        self._on_halt:   Optional[Callable[[], Coroutine]]      = None

        # ── Arb-detection alert throttling ────────────────────────────────────
        # At high signal frequency, alerting on every detection would flood the
        # chat and trip Telegram's rate limit. Gate alerts behind a per-market
        # cooldown and a minimum-edge floor (both env-tunable).
        self._arb_cooldown_s: float = float(
            os.environ.get("ARB_ALERT_COOLDOWN_S", "300")
        )
        self._arb_min_bps: float = float(
            os.environ.get("ARB_ALERT_MIN_BPS", "0")
        )
        self._last_arb_alert: dict[str, float] = {}   # condition_id → monotonic ts

        # python-telegram-bot Application (command listener only)
        self._closing: bool = False
        self._app: Optional[Application] = None
        if self._enabled:
            self._app = self._build_app()
            logger.info("TelegramNotifier ready | chat_id=%s", self._chat_id)
        else:
            logger.warning(
                "TelegramNotifier disabled — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Halt callback (set after construction to break circular reference)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_app(self) -> Application:
        """
        Construct the python-telegram-bot Application with explicit timeouts.

        The defaults (5 s connect) are too tight for a lossy path to
        api.telegram.org: a dropped SYN alone exceeds the budget before TCP
        can retransmit, which killed the listener at startup every boot.
        Rebuilt on each supervisor retry -- a part-initialised Application
        cannot safely be re-entered.
        """
        app = (
            Application.builder()
            .token(self._token)
            .connect_timeout(20.0)
            .read_timeout(20.0)
            .write_timeout(20.0)
            .pool_timeout(20.0)
            .get_updates_connect_timeout(20.0)
            .get_updates_read_timeout(40.0)
            .build()
        )
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("halt",   self._cmd_halt))
        return app

    def set_halt_callback(self, cb: Callable[[], Coroutine]) -> None:
        """Attach the coroutine that will be awaited when /halt is received."""
        self._on_halt = cb

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # The path to api.telegram.org intermittently resets/blackholes
            # connections. Cap connect separately so a dead attempt fails fast
            # and the retry in notify() gets a fresh connection quickly,
            # rather than burning the whole 10 s budget on one hung connect.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15, sock_connect=4, sock_read=10)
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session (call on shutdown)."""
        # Signals run_listener()'s supervisor to stop retrying. Without this the
        # retry loop keeps the process alive after main() has finished, and
        # systemd SIGKILLs the unit at TimeoutStopSec.
        self._closing = True
        if self._session and not self._session.closed:
            await self._session.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Core outbound send (aiohttp — no dependency on the Application lifecycle)
    # ──────────────────────────────────────────────────────────────────────────

    async def notify(self, text: str, *, parse_mode: Optional[str] = None) -> bool:
        """
        Send a message to the configured chat.

        Returns True on success, False on any error (non-raising).
        """
        if not self._enabled:
            logger.debug("Telegram notify (noop): %s", text[:80])
            return False

        url     = f"{_API_BASE}/bot{self._token}/sendMessage"
        payload: dict = {"chat_id": self._chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        # ~35 % of connections to api.telegram.org fail from this host, so a
        # single attempt silently drops roughly a third of all alerts. Retry
        # transient failures only; 4xx (bad token/chat_id, rate limit) is
        # permanent and must not be hammered. _fire() detaches delivery, so
        # these waits never touch the trading hot path.
        attempts  = 5
        last_err: object = None
        for attempt in range(1, attempts + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        logger.debug("Telegram OK | %s", text[:60])
                        return True
                    body = await resp.text()
                    if 400 <= resp.status < 500:
                        logger.error(
                            "Telegram error %d (permanent): %s",
                            resp.status, body[:200],
                        )
                        return False
                    last_err = f"HTTP {resp.status}: {body[:200]}"
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            if attempt < attempts:
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))
        logger.error(
            "Telegram send failed after %d attempts: %s",
            attempts, _scrub(last_err),
        )
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def alert(self, title: str, body: str) -> bool:
        """Send a two-line HTML alert: bold title + body."""
        return await self.notify(
            f"<b>{_h(title)}</b>\n{_h(body)}",
            parse_mode="HTML",
        )

    async def heartbeat(self, status: dict) -> bool:
        """Format and send a system health heartbeat via HTML."""
        lines = ["<b>Polymarket Arb Bot — Heartbeat</b>"]
        for k, v in status.items():
            lines.append(f"  {_h(k)}: <code>{_h(v)}</code>")
        return await self.notify("\n".join(lines), parse_mode="HTML")

    # ──────────────────────────────────────────────────────────────────────────
    # Fire-and-forget helpers (never block the trading loop)
    # ──────────────────────────────────────────────────────────────────────────

    def _fire(self, text: str, *, parse_mode: Optional[str] = None) -> None:
        """
        Schedule a background delivery task via asyncio.create_task().

        Returns immediately; the HTTP round-trip is fully detached from the
        caller so the hot trading path is never stalled on network I/O.
        """
        if not self._enabled:
            logger.debug("Telegram fire (noop): %s", text[:80])
            return
        asyncio.create_task(
            self.notify(text, parse_mode=parse_mode),
            name="tg-deliver",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Typed trade-execution alert — Arbitrage Edition
    # ──────────────────────────────────────────────────────────────────────────

    async def send_trade_execution(
        self,
        condition_id:      str,
        yes_token_id:      str,
        no_token_id:       str,
        yes_ask:           float,
        no_ask:            float,
        n_shares:          float,
        combined_cost:     float,
        guaranteed_profit: float,
    ) -> None:
        """Non-blocking arbitrage execution notification via HTML."""
        total_deployed = round(n_shares * combined_cost, 4)
        spread_bps     = round((1.0 - combined_cost) * 10_000, 1)

        text = (
            f"<b>ARB EXECUTED</b>\n"
            f"<b>Market:</b>     <code>{_h(condition_id[:24])}…</code>\n"
            f"<b>YES token:</b>  <code>{_h(yes_token_id[:16])}…</code>\n"
            f"<b>NO token:</b>   <code>{_h(no_token_id[:16])}…</code>\n"
            f"<b>YES price:</b>  <code>{yes_ask:.4f} USDC</code>\n"
            f"<b>NO price:</b>   <code>{no_ask:.4f} USDC</code>\n"
            f"<b>Combined:</b>   <code>{combined_cost:.6f} USDC/pair</code>\n"
            f"<b>Edge:</b>       <code>{spread_bps:.1f} bps</code>\n"
            f"<b>Shares:</b>     <code>{n_shares:.2f}</code>\n"
            f"<b>Deployed:</b>   <code>{total_deployed:.4f} USDC</code>\n"
            f"<b>Profit:</b>     <code>+{guaranteed_profit:.6f} USDC (guaranteed)</code>"
        )
        self._fire(text, parse_mode="HTML")

    async def send_critical_error(self, error_message: str) -> None:
        """Non-blocking critical error notification (circuit trip, timeout, etc.)."""
        text = f"<b>CRITICAL ERROR</b>\n<code>{_h(error_message)}</code>"
        self._fire(text, parse_mode="HTML")

    # ──────────────────────────────────────────────────────────────────────────
    # Real-time arb DETECTION alert (fires on signal, before/independent of fill)
    # ──────────────────────────────────────────────────────────────────────────

    def send_arb_detected(
        self,
        condition_id:  str,
        combined_cost: float,
        net_edge:      float,
        is_maker:      bool,
        yes_price:     float,
        no_price:      float,
        *,
        category:      str = "",
    ) -> None:
        """
        Fire a non-blocking alert the moment an arbitrage signal is detected —
        independent of whether it later executes or is blocked by the circuit
        breaker.  Distinct from `send_trade_execution` (which fires only on a
        confirmed fill).

        Throttling
        ──────────
        • Per-market cooldown (ARB_ALERT_COOLDOWN_S, default 300 s): the same
          condition_id will not alert again until the cooldown elapses.
        • Minimum-edge floor (ARB_ALERT_MIN_BPS, default 0): signals below this
          edge are not alerted (still traded — this only gates the notification).

        Returns immediately; delivery is detached via `_fire`.
        """
        if not self._enabled:
            return

        edge_bps = round(net_edge * 10_000, 1)
        if edge_bps < self._arb_min_bps:
            return

        now = time.monotonic()
        last = self._last_arb_alert.get(condition_id)
        if last is not None and (now - last) < self._arb_cooldown_s:
            return   # still in cooldown for this market
        self._last_arb_alert[condition_id] = now

        path = "MAKER" if is_maker else "TAKER"
        lines = [
            f"<b>ARB DETECTED</b> ({path})",
            f"<b>Market:</b>   <code>{_h(condition_id[:24])}…</code>",
        ]
        if category:
            lines.append(f"<b>Category:</b> <code>{_h(category)}</code>")
        lines += [
            f"<b>YES:</b>      <code>{yes_price:.4f}</code>",
            f"<b>NO:</b>       <code>{no_price:.4f}</code>",
            f"<b>Combined:</b> <code>{combined_cost:.6f} USDC/pair</code>",
            f"<b>Edge:</b>     <code>{edge_bps:.1f} bps</code>",
        ]
        self._fire("\n".join(lines), parse_mode="HTML")

    def send_arb_duration(
        self,
        condition_id:  str,
        duration_s:    float,
        peak_edge_bps: float,
        ticks:         int,
        is_maker:      bool,
        *,
        still_open:    bool = False,
    ) -> None:
        """
        Non-blocking alert reporting how long a market's arbitrage window lasted
        (first detected → closed).  Fired by the duration tracker on close (or,
        with still_open=True, for windows still open at shutdown).
        """
        if not self._enabled:
            return
        path  = "MAKER" if is_maker else "TAKER"
        title = "ARB WINDOW OPEN AT SHUTDOWN" if still_open else "ARB WINDOW CLOSED"
        text = (
            f"<b>{title}</b> ({path})\n"
            f"<b>Market:</b>    <code>{_h(condition_id[:24])}…</code>\n"
            f"<b>Duration:</b>  <code>{duration_s:.2f} s</code>\n"
            f"<b>Peak edge:</b> <code>{peak_edge_bps:.1f} bps</code>\n"
            f"<b>Ticks:</b>     <code>{ticks}</code>"
        )
        self._fire(text, parse_mode="HTML")

    # ──────────────────────────────────────────────────────────────────────────
    # Inbound command handlers (python-telegram-bot)
    # ──────────────────────────────────────────────────────────────────────────

    async def _is_authorized(self, update: Update) -> bool:
        """Return True only if the update originates from the configured chat."""
        return (
            update.effective_chat is not None
            and str(update.effective_chat.id) == self._chat_id
        )

    async def _cmd_status(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        /status — report USDC balance, daily PnL, and open positions.

        Data is sourced from the CircuitBreaker.status_dict() callback supplied
        at construction time.  If no callback is configured, reports unavailable.
        """
        if not await self._is_authorized(update):
            return

        if self._on_status is not None:
            try:
                raw = self._on_status()
                # Derive current USDC balance from starting_balance + session_pnl
                usdc_balance = round(
                    raw.get("starting_balance", 0.0) + raw.get("session_pnl", 0.0), 4
                )
                daily_pnl      = raw.get("daily_pnl", 0.0)
                open_positions = raw.get("open_positions", 0)

                text = (
                    "<b>Bot Status</b>\n"
                    f"  USDC balance:    <code>{usdc_balance:.4f}</code>\n"
                    f"  Daily PnL:       <code>{daily_pnl:+.4f}</code>\n"
                    f"  Open positions:  <code>{open_positions}</code>\n"
                    f"  Session PnL:     <code>{raw.get('session_pnl', 0.0):+.4f}</code>\n"
                    f"  Daily loss limit:<code>{raw.get('daily_loss_limit', 'n/a')}</code>"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Error building /status reply: %s", exc)
                text = f"<b>Status error</b>\n<code>{_h(exc)}</code>"
        else:
            text = "<b>Status unavailable</b> — no callback configured"

        try:
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send /status reply: %s", exc)

    async def _cmd_halt(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        /halt — manual kill switch.

        Sends an acknowledgement, then schedules the halt coroutine as a
        background asyncio task so this handler returns immediately and the
        updater is not blocked.  The halt coroutine (supplied by main.py)
        cancels all open orders and all asyncio tasks.
        """
        if not await self._is_authorized(update):
            return

        logger.critical("Manual HALT triggered via Telegram /halt command")

        try:
            await update.message.reply_text(
                "<b>HALT INITIATED</b>\n"
                "Cancelling all open orders and shutting down the bot...",
                parse_mode="HTML",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send /halt ack: %s", exc)

        if self._on_halt is not None:
            # Fire-and-forget: schedule halt without blocking the updater
            asyncio.ensure_future(self._on_halt())
        else:
            logger.error("/halt received but no halt callback is configured")

    # ──────────────────────────────────────────────────────────────────────────
    # Command listener coroutine
    # ──────────────────────────────────────────────────────────────────────────

    async def run_listener(self) -> None:
        """
        Long-running coroutine that polls the Telegram Bot API for commands.

        Designed to run concurrently inside asyncio.gather alongside the
        trading loops.  It does NOT block the hot path: the updater's polling
        tasks are dispatched as asyncio background tasks by python-telegram-bot.

        Cancelled cleanly when the parent gather is cancelled or when /halt
        fires task cancellation across the bot.
        """
        if not self._enabled or self._app is None:
            logger.info("Telegram command listener disabled — sleeping")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return
            return

        logger.info(
            "Telegram command listener starting (long-polling) | "
            "commands: /status /halt"
        )
        # Supervised: a transient failure must never permanently disarm /halt.
        # The path to api.telegram.org drops a significant share of connections,
        # so start_polling() can fail on any given attempt. Retry with backoff
        # instead of letting the coroutine die (which silently lost the kill
        # switch while the bot kept trading).
        backoff  = 5.0
        attempt  = 0
        while True:
            attempt += 1
            try:
                async with self._app:
                    await self._app.start()
                    await self._app.updater.start_polling(drop_pending_updates=True)
                    logger.info(
                        "Telegram command listener active (attempt %d)", attempt
                    )
                    backoff = 5.0          # reset once a start has succeeded
                    try:
                        # Hold here; cancelled by the gather when the bot shuts down.
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        pass
                    finally:
                        await self._app.updater.stop()
                        await self._app.stop()
                        logger.info("Telegram command listener stopped")
                return                     # clean shutdown -- do not retry
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self._closing:
                    logger.info(
                        "Telegram command listener stopping (shutdown in "
                        "progress): %s", _scrub(exc),
                    )
                    return
                logger.warning(
                    "Telegram command listener failed (attempt %d): %s -- "
                    "retrying in %.0fs", attempt, _scrub(exc), backoff,
                )
                await asyncio.sleep(backoff)
                if self._closing:          # shutdown may land during the wait
                    return
                backoff = min(backoff * 2, 60.0)
                try:
                    self._app = self._build_app()
                except Exception as rebuild_exc:  # noqa: BLE001
                    logger.error(
                        "Telegram listener rebuild failed: %s", rebuild_exc
                    )
