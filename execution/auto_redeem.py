"""
execution/auto_redeem.py
────────────────────────
Follows winning shares after a market resolves.

Polymarket pays them out itself. Trading setup approves its auto-redeem operator
(0xa1200000d0002264C9a1698e001292D00E1b00af) on the wallet, and the chain shows it
burning our winning shares within seconds of a market closing: three positions and
41.71 pUSD on 2026-11-04, with nothing here involved.

So this module does two things:
  1.  Reports each redemption, whoever performed it, from the wallet's activity feed.
  2.  Stands in if the operator does not come: a winning position still held
      REDEEM_GRACE_S after it was first seen is redeemed through the SDK, which
      picks the collateral adapter for standard or negative-risk markets. SDK 0.7.1
      only knows the CTF route, so a Protocol V2 market would fail and raise the
      alert below; every market on Gamma is still v1.

Losing shares stay listed as redeemable at value 0. Redeeming them pays nothing, so
they are left alone.

This used to scan up to 2000 markets on Gamma for `resolved` and `winner` fields
Gamma no longer returns, then look for shares under USDC.e position ids that no
pUSD position has. It never redeemed anything, and it was not the reason winnings
arrived.

Environment variables
─────────────────────
  REDEEM_POLL_INTERVAL   seconds between looks at the wallet (default 120)
  REDEEM_GRACE_S         how long the operator gets before the bot steps in (default 1800)
  REDEEM_RETRY_S         seconds between attempts on one market (default 3600)
  REDEEM_ALERT_AFTER     failed attempts before asking a person (default 3)
  REDEEM_MIN_VALUE_USD   a position worth less is not worth redeeming (default 0.01)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.clob_client import PolyClient
    from telemetry.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

_POLL_INTERVAL: float = float(os.environ.get("REDEEM_POLL_INTERVAL", 120.0))
_GRACE_S: float = float(os.environ.get("REDEEM_GRACE_S", 1800.0))
_RETRY_S: float = float(os.environ.get("REDEEM_RETRY_S", 3600.0))
_ALERT_AFTER: int = int(os.environ.get("REDEEM_ALERT_AFTER", 3))
_MIN_VALUE_USD: float = float(os.environ.get("REDEEM_MIN_VALUE_USD", 0.01))

# How far back each look reads the feed. Well past the poll, so a row the indexer
# delivers late is still reported; rows already reported are remembered.
_FEED_WINDOW_S: float = 3600.0


class AutoRedeemer:
    """
    Reports redemptions and redeems what the operator left behind.

    Parameters
    ----------
    notifier    : carries the redemption and failure messages.
    clob_client : reads the wallet and performs a redemption through the SDK.
    on_redeemed : told the market title of each redemption performed here, so the
                  reconciler does not read the vanished shares as an escaped order.
    """

    def __init__(
        self,
        notifier:    "TelegramNotifier",
        clob_client: "PolyClient",
        on_redeemed: "Callable[[str], None] | None" = None,
        poll_s:      float = _POLL_INTERVAL,
        grace_s:     float = _GRACE_S,
    ) -> None:
        self._notifier = notifier
        self._client = clob_client
        self._on_redeemed = on_redeemed
        self._poll = max(10.0, poll_s)
        self._grace = max(0.0, grace_s)
        # Reports start now: what was paid out before this run is not news.
        self._started_at = time.time()
        self._reported: set[tuple[str, str, str]] = set()
        # condition_id -> when a winning position there was first seen / last tried
        self._first_seen: dict[str, float] = {}
        self._last_try: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        logger.info("AutoRedeemer init | poll=%.0fs grace=%.0fs", self._poll, self._grace)

    async def run(self) -> None:
        """Long-running coroutine. Cancel to stop cleanly."""
        logger.info("AutoRedeemer started")
        try:
            while True:
                await asyncio.sleep(self._poll)
                try:
                    await self.check_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("AutoRedeemer check failed: %s", exc)
        except asyncio.CancelledError:
            logger.info("AutoRedeemer stopped")
            raise

    async def check_once(self, now: "float | None" = None) -> None:
        now = time.time() if now is None else now
        await self._report_redemptions(now)
        await self._redeem_what_was_left(now)

    # ── reports ───────────────────────────────────────────────────────────────

    async def _report_redemptions(self, now: float) -> None:
        since = max(self._started_at, now - _FEED_WINDOW_S)
        rows = await self._client.recent_redemptions(since)
        for r in rows or []:
            key = (r["tx"], r["condition_id"], r["outcome"])
            if key in self._reported:
                continue
            self._reported.add(key)
            # A redemption burns the losing side too, at nothing.
            if r["usdc"] <= 0.0:
                continue
            logger.info("AutoRedeemer | redeemed %s | %s | %.4f pUSD | tx=%s",
                        r["outcome"], r["title"][:50], r["usdc"], r["tx"][:20])
            await self._notifier.notify(
                f"💰 CTF REDEEMED — {r['usdc']:.2f} pUSD\n"
                f"  {r['outcome']} {r['title'][:60]}\n"
                f"  tx={r['tx']}"
            )

    # ── stand-in ──────────────────────────────────────────────────────────────

    async def _redeem_what_was_left(self, now: float) -> None:
        rows = await self._client.open_positions_detail()
        if rows is None:
            return
        winners: dict[str, dict] = {}
        for r in rows:
            if r.get("redeemable") and r.get("condition_id") \
                    and float(r.get("value") or 0.0) >= _MIN_VALUE_USD:
                winners[r["condition_id"]] = r
        # Whatever was paid out since the last look needs no more watching.
        for seen in (self._first_seen, self._last_try, self._failures):
            for cid in [c for c in seen if c not in winners]:
                del seen[cid]
        for cid, r in winners.items():
            first = self._first_seen.setdefault(cid, now)
            if now - first < self._grace:
                continue
            if now - self._last_try.get(cid, 0.0) < _RETRY_S:
                continue
            self._last_try[cid] = now
            await self._redeem(cid, r)

    async def _redeem(self, condition_id: str, row: dict) -> None:
        title = str(row.get("title") or condition_id[:20])
        logger.warning(
            "AutoRedeemer | %s still unredeemed after %.0fs (%.4f pUSD) — redeeming",
            title[:50], self._grace, float(row.get("value") or 0.0),
        )
        try:
            tx = await self._client.redeem_positions(condition_id)
        except Exception as exc:  # noqa: BLE001
            n = self._failures[condition_id] = self._failures.get(condition_id, 0) + 1
            logger.error("AutoRedeemer | redeem failed for %s (%d): %s",
                         condition_id[:16], n, exc)
            if n == _ALERT_AFTER:
                await self._notifier.notify(
                    f"⚠️ REDEEM FAILED {n}× — {row.get('outcome')} {title[:60]}, "
                    f"{float(row.get('value') or 0.0):.2f} pUSD still unpaid: {exc}\n"
                    "MANUAL INTERVENTION REQUIRED — redeem it on polymarket.com."
                )
            return
        logger.info("AutoRedeemer | redeem sent for %s | tx=%s", condition_id[:16], str(tx)[:20])
        # Announced so the reconciler does not read the vanished shares as a leak.
        # The Telegram report comes from the activity feed, like any other redemption.
        if self._on_redeemed is not None:
            try:
                self._on_redeemed(title)
            except Exception as exc:  # noqa: BLE001
                logger.debug("AutoRedeemer | redemption notice failed: %s", exc)
