"""
execution/reward_quoter.py
──────────────────────────
RewardQuoter — a small, bounded pilot of quoting for liquidity rewards.

Why this exists
───────────────
The wallet never earned a liquidity reward: its orders were 5-10 shares and the
programme counts orders of a market's minimum size, usually 20. Measured on
2026-11-15, the reward farmers with the fewest losing days rest exactly that
minimum on hundreds of markets and hold the cheaper token to offer from.

What it does
────────────
On a dozen quiet markets far from their end it buys the minimum lot of the cheaper
token, then keeps a bid and an ask on that token at the best prices others rest at
(strategy/reward_quotes.py). Both sides score. All a quote can lose is what its
shares and its bid cost, and REWARD_MAX_COST_USDC caps that a market.

What stops it (the owner's terms, 2026-11-15)
─────────────────────────────────────────────
  • REWARD_BUDGET_USDC      25   shares held plus resting bids, across all markets
  • REWARD_MAX_LOSS_USDC     8   realised result plus shares marked at the bid
  • REWARD_PROOF_DAYS        3   no day's rewards reached REWARD_MIN_DAILY_USDC by then
  • REWARD_PILOT_DAYS        7   the pilot is over
Stopping cancels the bids. The asks keep resting, so the shares can still be sold.

Off unless REWARD_QUOTING_ENABLED=true. With REWARD_QUOTING_DRY_RUN=true it picks
its markets and logs them, and sends nothing.

It shares the wallet with the cross path, which only trades markets inside its
seven-day window: nothing here is quoted within REWARD_MIN_DAYS_TO_END (15) of its
end, so the two never hold the same token while the pilot runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from core.clob_client import _FILLED_STATUSES, filled_buy, prime_market_meta
from risk.circuit_breaker import CircuitBreakerTripped
from strategy.arbitrage import effective_taker_fee
from strategy.reward_quotes import (
    Plan,
    RewardMarket,
    best_other,
    bid_shares_within,
    eligible,
    order_score,
    parse_reward_market,
    plan_quote,
    scoring_now,
    select,
)

if TYPE_CHECKING:
    from core.clob_client import PolyClient
    from risk.circuit_breaker import CircuitBreaker
    from telemetry.telegram import TelegramNotifier

logger = logging.getLogger(__name__)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return default if not raw else raw in ("1", "true", "yes", "on")


REWARD_QUOTING_ENABLED: bool = _flag("REWARD_QUOTING_ENABLED", False)
REWARD_QUOTING_DRY_RUN: bool = _flag("REWARD_QUOTING_DRY_RUN", False)
REWARD_BUDGET_USDC:    float = float(os.environ.get("REWARD_BUDGET_USDC", 25.0))
REWARD_MAX_MARKETS:    int   = int(os.environ.get("REWARD_MAX_MARKETS", 12))
REWARD_MAX_COST_USDC:  float = float(os.environ.get("REWARD_MAX_COST_USDC", 2.5))
REWARD_MAX_LOSS_USDC:  float = float(os.environ.get("REWARD_MAX_LOSS_USDC", 8.0))
REWARD_PILOT_DAYS:     float = float(os.environ.get("REWARD_PILOT_DAYS", 7.0))
REWARD_PROOF_DAYS:     float = float(os.environ.get("REWARD_PROOF_DAYS", 3.0))
REWARD_MIN_DAILY_USDC: float = float(os.environ.get("REWARD_MIN_DAILY_USDC", 1.0))
REWARD_POLL_S:         float = float(os.environ.get("REWARD_POLL_S", 60.0))
# Which markets are read at all: the cheaper token at most this price, the end at
# least this far off, at most this much traded in a day, a pool of at least this.
REWARD_MAX_PRICE:        float = float(os.environ.get("REWARD_MAX_PRICE", 0.08))
REWARD_MIN_DAYS_TO_END:  float = float(os.environ.get("REWARD_MIN_DAYS_TO_END", 15.0))
REWARD_MAX_VOLUME_24H:   float = float(os.environ.get("REWARD_MAX_VOLUME_24H", 3000.0))
REWARD_MIN_RATE:         float = float(os.environ.get("REWARD_MIN_RATE", 2.0))
REWARD_MAX_MIN_SIZE:     float = float(os.environ.get("REWARD_MAX_MIN_SIZE", 50.0))
# A quote is only opened where both sides would score at least this much of the
# most a share can, and where buying the shares gives up at most this to the spread.
REWARD_MIN_SCORE:        float = float(os.environ.get("REWARD_MIN_SCORE", 0.5))
REWARD_MAX_SPREAD_COST:  float = float(os.environ.get("REWARD_MAX_SPREAD_COST", 0.25))
# An order is left where it is this long before it is moved to follow the book.
REWARD_REQUOTE_MIN_S:  float = float(os.environ.get("REWARD_REQUOTE_MIN_S", 90.0))
# Markets that emptied are replaced this often, while the budget allows.
REWARD_REFILL_S:       float = float(os.environ.get("REWARD_REFILL_S", 6 * 3600.0))
REWARD_STATE_PATH:     str   = os.environ.get("REWARD_STATE_PATH", "reward_quotes_live.json")

_REWARDS_URL = "https://clob.polymarket.com/rewards/markets/multi"
_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
# The exchange takes no order under this many shares.
_MIN_ORDER_SHARES: float = 5.0
_SHARE_EPS: float = 0.005
# Fills are read this far back, and remembered twice as long so none is applied twice.
_FILL_WINDOW_S: float = 6 * 3600.0
_SCORING_EVERY_S: float = 600.0
_INCOME_EVERY_S: float = 1800.0
_BALANCE_EVERY_S: float = 600.0
_SELECT_RETRY_S: float = 600.0


@dataclass
class Quote:
    """One market being quoted: the shares held, and the two resting orders."""
    condition_id: str
    title:        str
    outcome:      str
    token_id:     str
    min_size:     float
    max_spread:   float
    rate:         float
    lot:          float                    # shares each side is meant to show
    end_ts:       "float | None" = None
    shares:       float = 0.0              # held
    cost:         float = 0.0              # what the held shares cost, fees included
    realised:     float = 0.0
    bid_id:       str = ""
    bid_price:    float = 0.0
    bid_shares:   float = 0.0              # still resting
    ask_id:       str = ""
    ask_price:    float = 0.0
    ask_shares:   float = 0.0
    opened_ts:    float = 0.0
    scoring:      bool = False             # what the exchange last said of the two orders
    mismatch:     int = 0                  # wallet reads in a row that disagreed with `shares`

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "Quote":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in known})

    @property
    def avg_cost(self) -> float:
        return self.cost / self.shares if self.shares > _SHARE_EPS else 0.0


async def fetch_reward_markets(keep: "Callable[[RewardMarket], bool]") -> "list[RewardMarket] | None":
    """
    The reward markets `keep` accepts, or None if the listing cannot be read.

    The listing is asked for richest pool first, so the first page without a
    paying market ends the walk: about 17 pages of 60 on 2026-11-15.
    """
    import httpx  # noqa: PLC0415 — optional at import time

    out: list[RewardMarket] = []
    cursor = ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            for _ in range(80):
                params = {"order_by": "rate_per_day", "position": "DESC", "page_size": 500}
                if cursor:
                    params["next_cursor"] = cursor
                resp = await http.get(_REWARDS_URL, params=params)
                if resp.status_code != 200:
                    logger.warning("RewardQuoter | listing returned HTTP %d", resp.status_code)
                    return None
                body = resp.json()
                paying = [m for m in map(parse_reward_market, body.get("data") or []) if m]
                out += [m for m in paying if keep(m)]
                cursor = body.get("next_cursor") or ""
                if not paying or not cursor or cursor == "LTE=":
                    break
    except Exception as exc:  # noqa: BLE001
        logger.warning("RewardQuoter | listing read failed: %s", exc)
        return None
    return out


async def fetch_market_info(condition_id: str) -> "dict | None":
    """Gamma's row for one open market, or None."""
    import httpx  # noqa: PLC0415

    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            resp = await http.get(_GAMMA_URL, params={"condition_ids": condition_id})
        rows = resp.json() if resp.status_code == 200 else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("RewardQuoter | market lookup failed for %s: %s", condition_id[:14], exc)
        return None
    for row in rows if isinstance(rows, list) else []:
        if str(row.get("conditionId")) == condition_id:
            return row
    return None


class RewardQuoter:
    """Keeps minimum-size two-sided quotes on a few reward markets, inside hard limits."""

    def __init__(
        self,
        client:   "PolyClient",
        breaker:  "CircuitBreaker",
        notifier: "TelegramNotifier",
        *,
        fee_engine=None,
        enabled:  "bool | None" = None,
        dry_run:  "bool | None" = None,
        state_path: "str | None" = None,
        budget:   "float | None" = None,
        max_markets: "int | None" = None,
        max_cost: "float | None" = None,
        max_loss: "float | None" = None,
        fetch_markets: "Callable[..., Awaitable[list[RewardMarket] | None]]" = fetch_reward_markets,
        fetch_info: "Callable[[str], Awaitable[dict | None]]" = fetch_market_info,
    ) -> None:
        self._client = client
        self._breaker = breaker
        self._notifier = notifier
        self._fee_engine = fee_engine
        self._enabled = REWARD_QUOTING_ENABLED if enabled is None else enabled
        self._dry_run = REWARD_QUOTING_DRY_RUN if dry_run is None else dry_run
        self._path = Path(state_path or REWARD_STATE_PATH)
        self._budget = REWARD_BUDGET_USDC if budget is None else budget
        self._max_markets = REWARD_MAX_MARKETS if max_markets is None else max_markets
        self._max_cost = REWARD_MAX_COST_USDC if max_cost is None else max_cost
        self._max_loss = REWARD_MAX_LOSS_USDC if max_loss is None else max_loss
        self._fetch_markets = fetch_markets
        self._fetch_info = fetch_info
        self._poll = max(5.0, REWARD_POLL_S)

        self._quotes: dict[str, Quote] = {}
        # order id → {cid, side, price}: every order of ours, resting or recently
        # cancelled, so a fill that lands after a cancel is still attributed.
        self._orders: dict[str, dict] = {}
        self._applied: dict[str, float] = {}       # "trade:order" → when it was applied
        self._reported: dict[str, float] = {}      # income rows already announced
        self._income: dict[str, dict] = {}         # UTC day paid → {type: usdc}
        self._started_ts: "float | None" = None
        self._stopped: str = ""
        self._realised: float = 0.0                # every market, closed ones included
        self._closed: list[dict] = []
        # Nothing below is kept across restarts.
        self._orphans: list[str] = []
        self._moved: dict[str, float] = {}         # "cid:side" → monotonic time of last placement
        self._marks: dict[str, float] = {}         # cid → best bid others rest at
        self._next: dict[str, float] = {}
        self._percent: dict[str, float] = {}
        logger.info(
            "RewardQuoter init | %s | budget %.2f USDC over %d market(s), %.2f a market, "
            "stop at -%.2f | proof %.0fd, pilot %.0fd",
            "QUOTING" if self._enabled and not self._dry_run
            else "DRY RUN" if self._enabled else "off",
            self._budget, self._max_markets, self._max_cost, self._max_loss,
            REWARD_PROOF_DAYS, REWARD_PILOT_DAYS,
        )

    # ── state ─────────────────────────────────────────────────────────────────

    def restore(self) -> int:
        """Load what a previous process left. Returns the number of markets held."""
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return 0
        except (OSError, ValueError) as exc:
            logger.error("RewardQuoter | %s is unreadable (%s): starting empty", self._path, exc)
            return 0
        for row in raw.get("quotes") or []:
            q = Quote.from_dict(row)
            # Orders the previous process left are cancelled on the first poll.
            self._orphans += [o for o in (q.bid_id, q.ask_id) if o]
            q.bid_id = q.ask_id = ""
            q.bid_price = q.bid_shares = q.ask_price = q.ask_shares = 0.0
            q.scoring = False
            self._quotes[q.condition_id] = q
        self._orders = dict(raw.get("orders") or {})
        for row in self._orders.values():                # none of them rests any more
            row.setdefault("retired", time.time())
        self._applied = dict(raw.get("applied") or {})
        self._reported = dict(raw.get("reported") or {})
        self._income = dict(raw.get("income") or {})
        self._started_ts = raw.get("started_ts")
        self._stopped = str(raw.get("stopped") or "")
        self._realised = float(raw.get("realised") or 0.0)
        self._closed = list(raw.get("closed") or [])
        held = sum(1 for q in self._quotes.values() if q.shares > _SHARE_EPS)
        if self._quotes:
            logger.info(
                "RewardQuoter | restored %d market(s), %d with shares (cost %.4f USDC)%s",
                len(self._quotes), held, sum(q.cost for q in self._quotes.values()),
                f" | stopped: {self._stopped}" if self._stopped else "",
            )
        return len(self._quotes)

    def _save(self) -> None:
        body = {
            "version": 1, "saved_at": time.time(), "started_ts": self._started_ts,
            "stopped": self._stopped, "realised": self._realised,
            "quotes": [q.as_dict() for q in self._quotes.values()],
            "orders": self._orders, "applied": self._applied, "reported": self._reported,
            "income": self._income, "closed": self._closed[-200:],
        }
        try:
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(body))
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning("RewardQuoter | cannot write %s: %s", self._path, exc)

    # ── what the rest of the bot asks ─────────────────────────────────────────

    def managed_titles(self) -> set[str]:
        """Titles whose inventory this owns, for WalletReconciler."""
        return {q.title for q in self._quotes.values() if q.title}

    def committed(self) -> float:
        """Collateral in use: the shares held at cost, and the resting bids."""
        return sum(q.cost + q.bid_price * q.bid_shares for q in self._quotes.values())

    def result(self) -> float:
        """Realised result plus the shares held, marked at the bid others rest at."""
        held = sum(q.shares * self._marks.get(q.condition_id, 0.0) - q.cost
                   for q in self._quotes.values())
        return self._realised + held

    def rewards(self) -> float:
        return sum(float(day.get("REWARD") or 0.0) for day in self._income.values())

    def summary(self) -> str:
        if not self._enabled and not self._quotes:
            return "off"
        qs = list(self._quotes.values())
        both = sum(1 for q in qs if q.bid_id and q.ask_id)
        state = (f"stopped ({self._stopped})" if self._stopped
                 else "dry run" if self._dry_run else "quoting" if self._enabled else "off")
        return (
            f"{state} | {len(qs)} market(s), {both} two-sided, "
            f"{sum(1 for q in qs if q.scoring)} scoring | in use {self.committed():.2f} of "
            f"{self._budget:.2f} USDC | result {self.result():+.4f} "
            f"(realised {self._realised:+.4f}) | rewards {self.rewards():.2f}"
        )

    # ── loop ──────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("RewardQuoter started | poll=%.0fs", self._poll)
        try:
            while True:
                try:
                    await self.poll_once()
                except (CircuitBreakerTripped, asyncio.CancelledError):
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("RewardQuoter poll failed: %s", exc)
                await asyncio.sleep(self._poll)
        except asyncio.CancelledError:
            logger.info("RewardQuoter stopped")
            raise

    def _due(self, what: str, every: float) -> bool:
        now = time.monotonic()
        if now < self._next.get(what, 0.0):
            return False
        self._next[what] = now + every
        return True

    async def poll_once(self, now: "float | None" = None) -> None:
        now = time.time() if now is None else now
        await self._cancel_orphans()
        if not self._enabled:
            return
        if self._dry_run:
            if self._due("select", _SELECT_RETRY_S):
                await self._choose(now)
                await self._probe(now)
            return
        try:
            await self._apply_fills(now)
            if not self._stopped:
                await self._enter(now)
            await self._requote_all(now)
            await self._check_balances()
            await self._check_scoring()
            await self._check_income(now)
            await self._check_stops(now)
        finally:
            self._save()

    async def _cancel_orphans(self) -> None:
        """Cancel orders a previous process left resting: nothing follows the book for them."""
        while self._orphans:
            oid = self._orphans.pop()
            try:
                await self._client.cancel_order(oid)
                logger.warning("RewardQuoter | cancelled an order left resting: %s", oid[:14])
            except Exception as exc:  # noqa: BLE001
                logger.warning("RewardQuoter | could not cancel %s: %s", oid[:14], exc)

    async def _probe(self, now: float) -> None:
        """In a dry run, the three account reads the live loop depends on, once each."""
        fills = await self._client.fills_since(now - 14 * 86_400.0)
        percent = await self._client.reward_percentages()
        income = await self._client.recent_income(now - 14 * 86_400.0)
        logger.info(
            "RewardQuoter | dry run reads: fills in 14 days %s | reward shares %s | "
            "income rows %s",
            "unreadable" if fills is None else len(fills),
            "unreadable" if percent is None else len(percent),
            "unreadable" if income is None else len(income),
        )

    # ── choosing and opening ──────────────────────────────────────────────────

    async def _choose(self, now: float) -> list[Plan]:
        """The quotes worth opening now, inside what is left of the budget."""
        held = set(self._quotes)

        def keep(m: RewardMarket) -> bool:
            return eligible(m, now, max_price=REWARD_MAX_PRICE, min_days=REWARD_MIN_DAYS_TO_END,
                            max_volume=REWARD_MAX_VOLUME_24H, min_rate=REWARD_MIN_RATE,
                            max_min_size=REWARD_MAX_MIN_SIZE, excluded=held)

        markets = await self._fetch_markets(keep)
        if markets is None:
            return []
        books = await self._client.get_orderbooks([m.cheaper()[0] for m in markets])
        plans = [p for m in markets
                 if (p := self._plan(m, books.get(m.cheaper()[0])))]
        room = self._budget - self.committed()
        slots = self._max_markets - len(self._quotes)
        chosen = select(plans, budget=room, max_markets=max(0, slots))
        logger.info(
            "RewardQuoter | %d market(s) eligible, %d can be quoted, %d chosen for %.2f of "
            "%.2f USDC free",
            len(markets), len(plans), len(chosen), sum(p.cost for p in chosen), room,
        )
        for p in chosen:
            logger.info(
                "RewardQuoter | %s %s %s | bid %.3f × %.0f, ask %.3f × %.2f | cost %.2f | "
                "pool %.0f/day, share now %.0f%% | ends %s",
                "would quote" if self._dry_run else "plan", p.outcome, p.market.question[:52],
                p.bid, p.bid_shares, p.ask, p.buy_shares, p.cost, p.market.rate,
                p.share * 100.0,
                datetime.fromtimestamp(p.market.end_ts, timezone.utc).strftime("%Y-%m-%d")
                if p.market.end_ts else "?",
            )
        return chosen

    def _plan(self, market: RewardMarket, book: "dict | None") -> "Plan | None":
        return plan_quote(market, book, max_cost=self._max_cost, min_score=REWARD_MIN_SCORE,
                          max_spread_cost=REWARD_MAX_SPREAD_COST)

    async def _enter(self, now: float) -> None:
        """Open quotes: all of them once, then replacements for markets that emptied."""
        first = self._started_ts is None
        if len(self._quotes) >= self._max_markets:
            return
        if not self._due("select", _SELECT_RETRY_S if first else REWARD_REFILL_S):
            return
        chosen = await self._choose(now)
        opened = [q for p in chosen if (q := await self._open(p, now))]
        if not opened:
            return
        if first:
            self._started_ts = now
        self._save()
        await self._say(
            f"🎯 REWARD QUOTES {'STARTED' if first else 'ADDED'} — {len(opened)} market(s), "
            f"{sum(q.cost for q in opened):.2f} USDC of shares bought\n"
            + "\n".join(f"  {q.outcome} {q.title[:48]} — {q.shares:.2f} @ {q.avg_cost:.3f}"
                        for q in opened[:12]),
            kind="trade")

    async def _open(self, plan: Plan, now: float) -> "Quote | None":
        """Buy the shares for one quote. The orders follow on the same poll."""
        m = plan.market
        info = await self._fetch_info(m.condition_id)
        if not info or info.get("closed") or not info.get("acceptingOrders", True) \
                or not info.get("enableOrderBook", True):
            logger.info("RewardQuoter | %s is not taking orders — left", m.question[:48])
            return None
        prime_market_meta(info)
        fresh = self._plan(m, await self._client.get_orderbook(plan.token_id))
        if fresh is None or fresh.ask > plan.ask + 1e-9:
            logger.info("RewardQuoter | %s moved since it was chosen — left", m.question[:48])
            return None
        if self.committed() + fresh.cost > self._budget + 1e-9:
            return None
        try:
            resp = await self._client.post_order(
                token_id=fresh.token_id, side="BUY", price=fresh.ask, size=fresh.buy_shares,
                _order_type="FAK")
        except Exception as exc:  # noqa: BLE001
            logger.warning("RewardQuoter | buy failed on %s: %s", m.question[:48], exc)
            return None
        if str((resp or {}).get("status", "")).strip().lower() not in _FILLED_STATUSES:
            logger.info("RewardQuoter | buy not filled on %s: %s", m.question[:48], resp)
            return None
        spent, bought = filled_buy(resp, fresh.buy_shares, fresh.ask)
        if bought <= _SHARE_EPS:
            return None
        rate = None
        if self._fee_engine is not None:
            try:
                rate = await self._fee_engine.get_taker_rate(m.condition_id)
            except Exception:  # noqa: BLE001 — the live estimate will do
                rate = None
        cost = spent * (1.0 + effective_taker_fee(spent / bought, rate))
        q = Quote(condition_id=m.condition_id, title=m.question, outcome=fresh.outcome,
                  token_id=fresh.token_id, min_size=m.min_size, max_spread=m.max_spread,
                  rate=m.rate, lot=fresh.bid_shares, end_ts=m.end_ts, shares=bought, cost=cost,
                  opened_ts=now)
        self._quotes[q.condition_id] = q
        self._save()
        logger.info("RewardQuoter | bought %.2f %s for %.4f USDC (fee %.4f) | %s",
                    bought, q.outcome, spent, cost - spent, q.title[:56])
        return q

    # ── fills ─────────────────────────────────────────────────────────────────

    async def _apply_fills(self, now: float) -> None:
        if not self._orders:
            return
        since = max(self._started_ts or now, now - _FILL_WINDOW_S)
        fills = await self._client.fills_since(since)
        if fills is None:
            return
        for f in sorted(fills, key=lambda f: f.get("ts") or 0.0):
            known = self._orders.get(str(f.get("order") or "").lower())
            key = f"{f.get('trade')}:{f.get('order')}"
            if known is None or key in self._applied:
                continue
            self._applied[key] = now
            q = self._quotes.get(known["cid"])
            if q is None:
                continue
            await self._apply(q, known["side"], float(f["shares"]), float(f["price"]),
                              str(f["order"]).lower())
        for key in [k for k, at in self._applied.items() if now - at > 2 * _FILL_WINDOW_S]:
            del self._applied[key]
        for oid in [o for o, row in self._orders.items()
                    if row.get("retired") and now - row["retired"] > 2 * _FILL_WINDOW_S]:
            del self._orders[oid]

    async def _apply(self, q: Quote, side: str, shares: float, price: float, oid: str) -> None:
        if side == "BUY":
            q.shares += shares
            q.cost += shares * price                         # a maker pays no fee
            if oid == q.bid_id:
                q.bid_shares = max(0.0, q.bid_shares - shares)
                if q.bid_shares <= _SHARE_EPS:
                    q.bid_id, q.bid_price, q.bid_shares = "", 0.0, 0.0
            logger.info("RewardQuoter | bid filled: bought %.2f %s @ %.3f | %s",
                        shares, q.outcome, price, q.title[:56])
            await self._say(f"🎯 REWARD QUOTE — bought {shares:.2f} {q.outcome} @ {price:.3f}\n"
                            f"  {q.title[:60]}", kind="trade")
            return
        sold = min(shares, q.shares)
        realised = sold * (price - q.avg_cost)
        q.cost -= sold * q.avg_cost
        q.shares -= sold
        if q.shares <= _SHARE_EPS:
            q.shares, q.cost = 0.0, 0.0
        q.realised += realised
        self._realised += realised
        if oid == q.ask_id:
            q.ask_shares = max(0.0, q.ask_shares - shares)
            if q.ask_shares <= _SHARE_EPS:
                q.ask_id, q.ask_price, q.ask_shares = "", 0.0, 0.0
        logger.info("RewardQuoter | ask filled: sold %.2f %s @ %.3f, %+.4f USDC | %s",
                    sold, q.outcome, price, realised, q.title[:56])
        await self._say(f"🎯 REWARD QUOTE — sold {sold:.2f} {q.outcome} @ {price:.3f}, "
                        f"{realised:+.4f} USDC\n  {q.title[:60]}", kind="trade")
        self._save()
        self._breaker.book_pnl(round(realised, 6))           # raises when the day is through its limit

    # ── keeping the two orders at the book ────────────────────────────────────

    async def _requote_all(self, now: float) -> None:
        if not self._quotes:
            return
        books = await self._client.get_orderbooks([q.token_id for q in self._quotes.values()])
        for q in list(self._quotes.values()):
            book = books.get(q.token_id)
            if book is None:
                continue
            await self._requote(q, book)
            # Everything held was sold: a bid alone scores nothing at these prices,
            # so the market is closed and its slot goes to whichever ranks best now.
            if q.shares <= _SHARE_EPS and not q.ask_id and await self._cancel(q, "BUY"):
                self._close(q, now)
            elif self._stopped and q.shares < _MIN_ORDER_SHARES and not q.ask_id:
                self._close(q, now)

    def _close(self, q: Quote, now: float) -> None:
        """A market with nothing left to offer and no bid: off the list, its slot free."""
        self._closed.append({"title": q.title, "outcome": q.outcome, "realised": q.realised,
                             "left": q.shares, "left_cost": q.cost, "closed_ts": now})
        # Shares too few to offer are written off: they cannot be sold through the book.
        self._realised -= q.cost
        del self._quotes[q.condition_id]
        self._marks.pop(q.condition_id, None)
        logger.info("RewardQuoter | closed %s | realised %+.4f, %.2f share(s) left",
                    q.title[:56], q.realised, q.shares)

    async def _requote(self, q: Quote, book: dict) -> None:
        bid_lvl = best_other(book, "bids", q.bid_price if q.bid_id else None, q.bid_shares,
                             min_level=q.min_size)
        ask_lvl = best_other(book, "asks", q.ask_price if q.ask_id else None, q.ask_shares,
                             min_level=q.min_size)
        any_bid = best_other(book, "bids", q.bid_price if q.bid_id else None, q.bid_shares)
        if ask_lvl is None:
            # Nobody offers a full lot: the shares are still offered, beside whoever is.
            ask_lvl = best_other(book, "asks", q.ask_price if q.ask_id else None, q.ask_shares)
        self._marks[q.condition_id] = any_bid[0] if any_bid else 0.0

        # The ask: everything held, at the best price others ask.
        offer = math.floor(q.shares * 100.0 + 1e-6) / 100.0
        if offer < _MIN_ORDER_SHARES or ask_lvl is None:
            if q.ask_id and offer < _MIN_ORDER_SHARES:
                await self._cancel(q, "SELL")
        elif not self._same(q.ask_id, q.ask_price, q.ask_shares, ask_lvl[0], offer):
            if self._may_move(q, "SELL") and await self._cancel(q, "SELL"):
                await self._place(q, "SELL", ask_lvl[0], offer)

        # The bid: the lot, at the best price others bid, while it scores and fits.
        want = 0.0
        if not self._stopped and bid_lvl is not None and ask_lvl is not None \
                and order_score(q.max_spread, (ask_lvl[0] - bid_lvl[0]) * 50.0) > 0.0:
            want = bid_shares_within(self._max_cost, q.cost, bid_lvl[0],
                                     max(q.lot, q.min_size), q.min_size)
            room = self._budget - (self.committed() - q.bid_price * q.bid_shares)
            if want * bid_lvl[0] > room + 1e-9:
                want = 0.0
        if want <= 0.0:
            if q.bid_id:
                await self._cancel(q, "BUY")
        elif not self._same(q.bid_id, q.bid_price, q.bid_shares, bid_lvl[0], want):
            if self._may_move(q, "BUY") and await self._cancel(q, "BUY"):
                await self._place(q, "BUY", bid_lvl[0], want)

    @staticmethod
    def _same(oid: str, price: float, shares: float, want_price: float, want: float) -> bool:
        return bool(oid) and abs(price - want_price) < 1e-9 and abs(shares - want) < 0.01

    def _may_move(self, q: Quote, side: str) -> bool:
        """Whether this side has rested long enough to be moved, or is not resting."""
        if not (q.bid_id if side == "BUY" else q.ask_id):
            return True
        return time.monotonic() - self._moved.get(f"{q.condition_id}:{side}", 0.0) \
            >= REWARD_REQUOTE_MIN_S

    async def _cancel(self, q: Quote, side: str) -> bool:
        oid = q.bid_id if side == "BUY" else q.ask_id
        if not oid:
            return True
        try:
            await self._client.cancel_order(oid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RewardQuoter | cancel failed on %s: %s", q.title[:40], exc)
            return False
        if oid in self._orders:
            self._orders[oid]["retired"] = time.time()
        if side == "BUY":
            q.bid_id, q.bid_price, q.bid_shares = "", 0.0, 0.0
        else:
            q.ask_id, q.ask_price, q.ask_shares = "", 0.0, 0.0
        q.scoring = False
        return True

    async def _place(self, q: Quote, side: str, price: float, shares: float) -> bool:
        if side == "SELL":
            # You cannot offer what the wallet does not hold yet: a buy settles on
            # the chain seconds after it matches.
            held = await self._client.share_balance(q.token_id)
            if held is not None and held + _SHARE_EPS < shares:
                shares = math.floor(held * 100.0 + 1e-6) / 100.0
                if shares < _MIN_ORDER_SHARES:
                    return False
        try:
            resp = await self._client.post_maker_order(
                token_id=q.token_id, side=side, desired_price=price, size=shares)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RewardQuoter | %s order failed on %s: %s", side, q.title[:40], exc)
            return False
        oid = str((resp or {}).get("order_id") or (resp or {}).get("orderID") or "").lower()
        if not oid:
            logger.warning("RewardQuoter | %s order refused on %s: %s", side, q.title[:40], resp)
            return False
        placed = float((resp or {}).get("price") or price)
        self._orders[oid] = {"cid": q.condition_id, "side": side, "price": placed}
        self._moved[f"{q.condition_id}:{side}"] = time.monotonic()
        if side == "BUY":
            q.bid_id, q.bid_price, q.bid_shares = oid, placed, shares
        else:
            q.ask_id, q.ask_price, q.ask_shares = oid, placed, shares
        logger.info("RewardQuoter | %s %.2f %s @ %.3f resting | %s",
                    "bid" if side == "BUY" else "ask", shares, q.outcome, placed, q.title[:52])
        return True

    # ── checks ────────────────────────────────────────────────────────────────

    async def _check_balances(self) -> None:
        """
        Shares the wallet no longer holds were sold in a fill the feed did not show.

        Only a shortfall is taken over, and only when two reads in a row agree. More
        shares than expected are not ours to offer: a bid's fill is attributed by its
        order id, so a surplus belongs to something else in the wallet.
        """
        if not self._due("balances", _BALANCE_EVERY_S):
            return
        for q in self._quotes.values():
            if time.time() - q.opened_ts < 120.0:
                continue
            held = await self._client.share_balance(q.token_id)
            if held is None:
                continue
            if held + 0.5 >= q.shares:
                q.mismatch = 0
                if held > q.shares + 0.5:
                    logger.warning("RewardQuoter | wallet holds %.2f of %s, %.2f are tracked",
                                   held, q.title[:40], q.shares)
                continue
            q.mismatch += 1
            if q.mismatch < 2:
                continue
            gone = q.shares - held
            price = q.ask_price or self._marks.get(q.condition_id, 0.0)
            logger.warning("RewardQuoter | %.2f share(s) of %s left the wallet unseen — "
                           "booked as sold @ %.3f", gone, q.title[:40], price)
            q.mismatch = 0
            await self._apply(q, "SELL", gone, price, "")

    async def _check_scoring(self) -> None:
        if not self._due("scoring", _SCORING_EVERY_S):
            return
        ids = [o for q in self._quotes.values() for o in (q.bid_id, q.ask_id) if o]
        got = await self._client.orders_scoring(ids) if ids else {}
        if got is None:
            return
        for q in self._quotes.values():
            q.scoring = bool(q.bid_id and q.ask_id and got.get(q.bid_id) and got.get(q.ask_id))
        percent = await self._client.reward_percentages()
        if percent is not None:
            self._percent = percent
        formed = sum(1 for q in self._quotes.values() if scoring_now(
            q.bid_price if q.bid_id else None, q.bid_shares,
            q.ask_price if q.ask_id else None, q.ask_shares, q.min_size, q.max_spread))
        logger.info(
            "RewardQuoter | the exchange scores %d of %d resting order(s); %d of %d quote(s) "
            "score on both sides, %d by our own reading | share of pools: %s",
            sum(1 for v in got.values() if v), len(ids),
            sum(1 for q in self._quotes.values() if q.scoring), len(self._quotes), formed,
            ", ".join(f"{q.title[:22]} {self._percent.get(q.condition_id.lower(), 0.0):.1f}%"
                      for q in self._quotes.values()) or "none",
        )

    async def _check_income(self, now: float) -> None:
        if self._started_ts is None or not self._due("income", _INCOME_EVERY_S):
            return
        rows = await self._client.recent_income(self._started_ts)
        for r in sorted(rows or [], key=lambda r: r["ts"]):
            key = f"{r['type']}:{r['ts']}"
            if key in self._reported or r["usdc"] <= 0.0:
                continue
            self._reported[key] = now
            day = datetime.fromtimestamp(r["ts"], timezone.utc).strftime("%Y-%m-%d")
            self._income.setdefault(day, {})
            self._income[day][r["type"]] = self._income[day].get(r["type"], 0.0) + r["usdc"]
            what = {"REWARD": "liquidity rewards", "MAKER_REBATE": "maker rebate",
                    "YIELD": "holding yield"}.get(r["type"], r["type"])
            logger.info("RewardQuoter | paid %s: %.4f USDC of %s", day, r["usdc"], what)
            await self._say(f"💰 REWARDS PAID — {r['usdc']:.4f} USDC of {what} ({day})",
                            kind="trade")

    async def _check_stops(self, now: float) -> None:
        if self._stopped or self._started_ts is None:
            return
        days = (now - self._started_ts) / 86_400.0
        best_day = max((float(d.get("REWARD") or 0.0) for d in self._income.values()),
                       default=0.0)
        result = self.result()
        if result <= -self._max_loss:
            await self._stop(f"the result reached {result:+.2f} USDC, the limit is "
                             f"-{self._max_loss:.2f}", now)
        # An hour past the day, so the payout for the last full day has arrived.
        elif days >= REWARD_PROOF_DAYS + 1.0 / 24.0 and best_day < REWARD_MIN_DAILY_USDC:
            await self._stop(f"no day paid {REWARD_MIN_DAILY_USDC:.2f} USDC of rewards in "
                             f"{REWARD_PROOF_DAYS:.0f} days (best {best_day:.2f})", now)
        elif days >= REWARD_PILOT_DAYS:
            await self._stop(f"the {REWARD_PILOT_DAYS:.0f}-day pilot is over", now)

    async def _stop(self, why: str, now: float) -> None:
        """Stop bidding. The asks stay, so what is held can still be sold."""
        self._stopped = why
        for q in self._quotes.values():
            await self._cancel(q, "BUY")
        self._save()
        held = [q for q in self._quotes.values() if q.shares > _SHARE_EPS]
        logger.warning("RewardQuoter | STOPPED: %s | result %+.4f, rewards %.2f, %d market(s) "
                       "still held", why, self.result(), self.rewards(), len(held))
        await self._say(
            f"⚠️ REWARD QUOTES STOPPED — {why}\n"
            f"  result {self.result():+.2f} USDC (realised {self._realised:+.2f}), "
            f"rewards paid {self.rewards():.2f} USDC\n"
            f"  {len(held)} market(s) still held, {sum(q.cost for q in held):.2f} USDC at "
            f"cost; their asks keep resting, no new bids.", kind="critical")

    async def _say(self, text: str, *, kind: str) -> None:
        try:
            await self._notifier.notify(text, kind=kind)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RewardQuoter | notify failed: %s", exc)
