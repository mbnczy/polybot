"""
execution/cross_guard.py
────────────────────────
Trade the cross-market implications the reader finds, and carry each position to
an early exit or to resolution.

The trade
─────────
For an implication narrow ⊆ broad priced the wrong way round:

    buy NO on narrow  +  buy YES on broad        pays ≥ 1.00 in every outcome

The reader discovers implications with a language model and cannot trade — its
systemd unit cannot even see the wallet's .env. It writes what it found, after
the ladder direction guard, to CROSS_IMPLICATIONS_PATH. This guard reads that
file, prices both legs from the live book, and trades.

What must hold before a single order goes out
──────────────────────────────────────────────
  • both markets resolve inside CROSS_MAX_LOCKUP_DAYS. The arbitrage pays only
    at resolution, and the reader's long-dated violations sat 112 days out; an
    unknown end date cannot be shown to meet the window, so it does not trade.
  • the edge survives the asks actually payable, fees included, per market.
    The detector prices NO on narrow as 1 − its YES ask, which on the live book
    understated the entry by a median 100 bps and on a one-sided in-play book by
    an order of magnitude (a "+200 bps" signal whose real entry was 1.79).
  • the size fits the depth at the best ask on BOTH legs. A FOK order for more
    than the touch holds either walks the book or fails.
  • the breaker's cross ledger has room: its own slots and capital ceiling, the
    shared daily loss limit and drawdown.

Execution
─────────
The two legs are separate markets and cannot be filled atomically. The thinner
leg — the one likelier to fail — goes first: if it does not fill, nothing is
held and the attempt costs nothing. If it fills and the second leg does not,
the first is sold straight back and the result booked. That half-fill is the
loss this path can take by construction, and ordering the legs this way makes
it the rarer case.

Afterwards
──────────
Every poll, each open position's exit is priced by selling both legs at the bid,
and it closes when that captures what resolution already guarantees
(strategy/cross_exit.should_exit). Completing both sets and merging is the other
way out and prices the same up to spread and fees, but needs the NegRisk
adapter on half the markets; this version exits by sale and leaves the merge
route for later. Past its end date a position waits for resolution, is closed
at the realised payoff, and its markets are handed to AutoRedeemer.

OFF unless CROSS_EXECUTION_ENABLED=true. While off it evaluates everything and
logs what it would have done, so the gates can be watched against live books
before any money depends on them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from core.clob_client import _FILLED_STATUSES
from risk.circuit_breaker import CircuitBreakerTripped
from strategy.quantity_guard import same_quantity
from strategy.cross_exit import (
    ROUTE_SELL,
    CrossPosition,
    ExitQuote,
    PositionBook,
    buy_cost,
    resolution_profit,
    sell_proceeds,
    should_exit,
)

logger = logging.getLogger(__name__)


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


CROSS_EXECUTION_ENABLED: bool = _flag("CROSS_EXECUTION_ENABLED", "false")
CROSS_IMPLICATIONS_PATH: str = os.environ.get(
    "CROSS_IMPLICATIONS_PATH",
    "/home/ubuntu/polybot-dev/cross-exec/cross_implications.json",
)
CROSS_POSITIONS_PATH: str = os.environ.get(
    "CROSS_POSITIONS_PATH", "cross_positions_live.json"
)
CROSS_POLL_S: float = float(os.environ.get("CROSS_POLL_S", 30.0))
# The whole point of the window: capital back within a week, not a quarter.
CROSS_MAX_LOCKUP_DAYS: float = float(os.environ.get("CROSS_MAX_LOCKUP_DAYS", 7.0))
CROSS_MIN_EDGE: float = float(os.environ.get("CROSS_MIN_EDGE", 0.02))
# Not lower than this: the exchange's minimum order is 5 shares, and 5 shares of
# a pair costing 0.85 is 4.25 USDC. A smaller cap does not mean smaller trades,
# it means no trades at all. Exposure is held down by the breaker's cross ledger
# (one position at a time) instead.
CROSS_MAX_POSITION_USDC: float = float(os.environ.get("CROSS_MAX_POSITION_USDC", 5.0))
# A position we could not sell back is the half-fill loss, so both legs must
# have a bid to exit into. Minutes before an event closes the book is a few
# stale quotes nobody will honour, and an "edge" far above what these markets
# ever really offer is that, not an arbitrage: it must survive a second poll.
CROSS_MIN_TIME_TO_END_S: float = float(os.environ.get("CROSS_MIN_TIME_TO_END_S", 1800.0))
CROSS_SUSPICIOUS_EDGE: float = float(os.environ.get("CROSS_SUSPICIOUS_EDGE", 0.15))
CROSS_SPIKE_CONFIRM_S: float = float(os.environ.get("CROSS_SPIKE_CONFIRM_S", 300.0))
# Gamma's orderMinSize on live markets.
CROSS_MIN_SHARES: float = float(os.environ.get("CROSS_MIN_SHARES", 5.0))
# After a failed or declined attempt, leave the pair alone this long.
CROSS_ENTRY_COOLDOWN_S: float = float(os.environ.get("CROSS_ENTRY_COOLDOWN_S", 900.0))
# How long a resolved position's markets stay on the redemption list.
CROSS_REDEEM_TAIL_DAYS: float = 14.0

_GAMMA = "https://gamma-api.polymarket.com/markets"


# ═══════════════════════════════════════════════════════════════════════════════
# What the reader hands over
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Implication:
    narrow:           str
    broad:            str
    narrow_title:     str
    broad_title:      str
    narrow_yes_token: str
    narrow_no_token:  str
    broad_yes_token:  str
    broad_no_token:   str
    narrow_end_ts:    "float | None"
    broad_end_ts:     "float | None"
    confidence:       float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.narrow, self.broad)

    @property
    def resolves_ts(self) -> "float | None":
        if self.narrow_end_ts is None or self.broad_end_ts is None:
            return None
        return max(self.narrow_end_ts, self.broad_end_ts)


def load_implications(path: "str | Path") -> list[Implication]:
    """Read the reader's export. Missing or unreadable means nothing to trade."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        logger.warning("CrossGuard | cannot read %s: %s", p, exc)
        return []
    out = []
    for r in raw.get("implications", []):
        try:
            out.append(Implication(
                narrow=str(r["narrow"]), broad=str(r["broad"]),
                narrow_title=str(r.get("narrow_title", "")),
                broad_title=str(r.get("broad_title", "")),
                narrow_yes_token=str(r["narrow_yes_token"]),
                narrow_no_token=str(r["narrow_no_token"]),
                broad_yes_token=str(r["broad_yes_token"]),
                broad_no_token=str(r["broad_no_token"]),
                narrow_end_ts=(float(r["narrow_end_ts"])
                               if r.get("narrow_end_ts") is not None else None),
                broad_end_ts=(float(r["broad_end_ts"])
                              if r.get("broad_end_ts") is not None else None),
                confidence=float(r.get("confidence", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Pricing — pure, so every gate can be tested without a network
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Level:
    price: float
    size:  float


def best_level(book: dict, side: str) -> "Level | None":
    """
    The best price on one side and the shares resting AT it.

    Scanned for the extreme rather than read off the end: the exchange sends
    bids ascending and asks descending, so the first entry is the WORST price.
    """
    rows = []
    for r in (book or {}).get(side) or []:
        try:
            p, s = float(r["price"]), float(r["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if p > 0.0 and s > 0.0:
            rows.append((p, s))
    if not rows:
        return None
    best = min(p for p, _ in rows) if side == "asks" else max(p for p, _ in rows)
    return Level(best, sum(s for p, s in rows if abs(p - best) < 1e-12))


@dataclass(frozen=True)
class Opportunity:
    imp:            Implication
    no_ask:         Level      # NO on narrow
    yes_ask:        Level      # YES on broad
    rate_narrow:    "float | None"
    rate_broad:     "float | None"
    shares:         float
    entry_per_pair: float      # fees included
    lockup_days:    float

    @property
    def edge_per_pair(self) -> float:
        return 1.0 - self.entry_per_pair

    @property
    def committed(self) -> float:
        return self.shares * self.entry_per_pair


def size_position(no_ask: Level, yes_ask: Level, entry_per_pair: float, *,
                  max_usdc: float, min_shares: float) -> float:
    """Pairs to buy: bounded by the touch on both legs and the capital cap."""
    if entry_per_pair <= 0.0:
        return 0.0
    cap = min(no_ask.size, yes_ask.size, max_usdc / entry_per_pair)
    shares = math.floor(cap * 100.0) / 100.0
    return shares if shares >= min_shares else 0.0


def quoted_entry(
    no_book: dict, yes_book: dict,
    rate_narrow: "float | None" = None, rate_broad: "float | None" = None,
) -> "tuple[Level, Level, float] | None":
    """Both touches and the fee-inclusive cost of one pair; None without an ask on each leg."""
    no_ask, yes_ask = best_level(no_book, "asks"), best_level(yes_book, "asks")
    if no_ask is None or yes_ask is None:
        return None
    return no_ask, yes_ask, buy_cost(no_ask.price, rate_narrow) + buy_cost(yes_ask.price, rate_broad)


# Where an implication stopped, in a word, for the heartbeat's summary line.
# Matched against evaluate()'s reasons in order; a test pins every refusal to
# its key, so a reworded reason cannot silently fall through to "other".
_STOP_KEYS: tuple[tuple[str, str], ...] = (
    ("different statistics", "different_statistic"),
    ("date unknown", "end_unknown"),
    ("past its end date", "past_end"),
    ("window", "outside_window"),
    ("dying book", "too_close_to_end"),
    ("no ask", "no_ask"),
    ("no bid", "no_bid"),
    ("at the real asks", "edge_below_min"),
    ("depth at the best ask", "thin_touch"),
)
# Stops reached with both asks priced on a genuine implication: only these
# contribute to "best edge", so a pseudo-implication's fake edge cannot show.
_PRICED_STOPS = frozenset({"edge_below_min", "thin_touch", "ok"})


def stop_key(reason: str) -> str:
    if reason == "ok":
        return "ok"
    for needle, key in _STOP_KEYS:
        if needle in reason:
            return key
    return "other"


def evaluate(
    imp: Implication, no_book: dict, yes_book: dict, *,
    now: float,
    rate_narrow: "float | None" = None,
    rate_broad:  "float | None" = None,
    max_lockup_days: float = CROSS_MAX_LOCKUP_DAYS,
    min_edge: float = CROSS_MIN_EDGE,
    max_usdc: float = CROSS_MAX_POSITION_USDC,
    min_shares: float = CROSS_MIN_SHARES,
    min_time_to_end_s: float = CROSS_MIN_TIME_TO_END_S,
) -> "tuple[Opportunity | None, str]":
    """Every entry gate, from the two books. Returns (opportunity, reason)."""
    # Belt and braces: the reader's prefilter already drops these, but this guard
    # trades whatever the file says, and the first pair ever to clear every
    # other gate here was three corners "implying" three goals.
    if same_quantity(imp.narrow_title, imp.broad_title) is False:
        return None, "the two markets count different statistics — not an implication"
    ends = imp.resolves_ts
    if ends is None:
        return None, "resolution date unknown — cannot show it fits the window"
    lockup = (ends - now) / 86_400.0
    if lockup <= 0.0:
        return None, "already past its end date"
    if lockup > max_lockup_days:
        return None, f"locks capital {lockup:.1f}d > {max_lockup_days:g}d window"
    if ends - now < min_time_to_end_s:
        return None, (f"resolves in {(ends - now) / 60.0:.0f} min, under the "
                      f"{min_time_to_end_s / 60.0:.0f} min floor — dying book")
    quote = quoted_entry(no_book, yes_book, rate_narrow, rate_broad)
    if quote is None:
        return None, "a leg has no ask — nothing to buy"
    no_ask, yes_ask, entry = quote
    # Both legs must be sellable. If the second leg fails, the first is sold
    # straight back — into a book with no bid that is impossible, and the
    # half-fill becomes a position we hold blind to resolution.
    if best_level(no_book, "bids") is None or best_level(yes_book, "bids") is None:
        return None, "a leg has no bid — a half-fill could not be sold back"
    if 1.0 - entry < min_edge:
        return None, f"edge {1.0 - entry:+.4f} < {min_edge:.4f} at the real asks"
    shares = size_position(no_ask, yes_ask, entry, max_usdc=max_usdc,
                           min_shares=min_shares)
    if shares <= 0.0:
        return None, (f"depth at the best ask ({no_ask.size:.0f} / {yes_ask.size:.0f}) "
                      f"does not reach the {min_shares:g}-share minimum")
    return Opportunity(imp, no_ask, yes_ask, rate_narrow, rate_broad, shares,
                       entry, lockup), "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# The guard
# ═══════════════════════════════════════════════════════════════════════════════

ResolutionLookup = Callable[[str], Awaitable["bool | None"]]


class CrossGuard:
    def __init__(
        self, client, breaker, notifier, *,
        fee_engine=None,
        implications_path: "str | None" = None,
        positions_path:    "str | None" = None,
        enabled:           "bool | None" = None,
        poll_s:            "float | None" = None,
        max_lockup_days:   "float | None" = None,
        min_edge:          "float | None" = None,
        max_position_usdc: "float | None" = None,
        min_shares:        "float | None" = None,
        min_time_to_end_s: "float | None" = None,
        suspicious_edge:   "float | None" = None,
        resolution_lookup: "ResolutionLookup | None" = None,
    ) -> None:
        self._client = client
        self._breaker = breaker
        self._notifier = notifier
        self._fees = fee_engine
        self._imp_path = implications_path or CROSS_IMPLICATIONS_PATH
        self._enabled = CROSS_EXECUTION_ENABLED if enabled is None else enabled
        self._poll = CROSS_POLL_S if poll_s is None else poll_s
        self._max_lockup = CROSS_MAX_LOCKUP_DAYS if max_lockup_days is None else max_lockup_days
        self._min_edge = CROSS_MIN_EDGE if min_edge is None else min_edge
        self._max_usdc = CROSS_MAX_POSITION_USDC if max_position_usdc is None else max_position_usdc
        self._min_shares = CROSS_MIN_SHARES if min_shares is None else min_shares
        self._min_time_to_end = (CROSS_MIN_TIME_TO_END_S if min_time_to_end_s is None
                                 else min_time_to_end_s)
        self._suspicious_edge = (CROSS_SUSPICIOUS_EDGE if suspicious_edge is None
                                 else suspicious_edge)
        self._spike: dict[tuple[str, str], float] = {}
        self._resolved = resolution_lookup or _gamma_resolution
        self._book = PositionBook(positions_path or CROSS_POSITIONS_PATH).load()
        self._cooldown: dict[tuple[str, str], float] = {}
        self._stuck: set[tuple[str, str]] = set()
        self.last_reason: dict[tuple[str, str], str] = {}
        self.stats = {"evaluated": 0, "would_enter": 0, "entered": 0,
                      "half_filled": 0, "exited": 0, "resolved": 0, "stuck": 0}
        # The last complete pass over the reader's file, for stop_summary():
        # (when, implications read, stops by key, (best edge, narrow title)).
        self._last_pass: "tuple[float, int, dict[str, int], tuple[float, str] | None] | None" = None
        logger.info(
            "CrossGuard init | %s | window=%gd min_edge=%.3f max_pos=%.2f USDC "
            "open=%d file=%s",
            "EXECUTING" if self._enabled else "evaluate only (disabled)",
            self._max_lockup, self._min_edge, self._max_usdc,
            len(self._book.open_positions()), self._imp_path,
        )

    # ── wiring ────────────────────────────────────────────────────────────────

    def restore(self) -> int:
        """
        Re-register persisted open positions in the breaker's cross ledger.

        Their capital is still committed after a restart, and a ledger that
        forgot it would let new entries exceed the ceiling.
        """
        n = 0
        for pos in self._book.open_positions():
            self._breaker.on_cross_open(pos.entry_cost * pos.size)
            n += 1
        if n:
            logger.info("CrossGuard | restored %d open position(s) into the breaker", n)
        return n

    def _recent(self) -> list[CrossPosition]:
        cutoff = time.time() - CROSS_REDEEM_TAIL_DAYS * 86_400.0
        return self._book.open_positions() + [
            p for p in self._book.closed if (p.closed_ts or 0.0) >= cutoff
        ]

    def condition_ids(self) -> set[str]:
        """Markets AutoRedeemer should watch: held now, or resolved recently."""
        return {c for p in self._recent() for c in (p.narrow, p.broad)}

    def managed_titles(self) -> set[str]:
        """Titles whose inventory this guard owns, for WalletReconciler."""
        return {t for p in self._recent() for t in (p.narrow_title, p.broad_title) if t}

    def open_positions(self) -> list[CrossPosition]:
        return self._book.open_positions()

    def stop_summary(self) -> str:
        """
        One line for the heartbeat: where the last pass over the reader's file
        stopped, commonest first, and the best edge the real asks quoted.

        Entries are logged only when a pair clears every gate, so without this a
        quiet journal cannot tell "no opportunity" from "not looking".
        """
        if self._last_pass is None:
            return "no pass yet"
        ts, total, stops, best = self._last_pass
        head = f"last pass {time.time() - ts:.0f}s ago"
        if not total:
            return f"{head} | no implications in {self._imp_path}"
        parts = ", ".join(f"{k} {v}" for k, v in
                          sorted(stops.items(), key=lambda kv: (-kv[1], kv[0])))
        line = f"{head} | {total} implication(s): {parts}"
        if best is not None:
            line += f" | best edge {best[0]:+.4f} (min {self._min_edge:.4f}) {best[1][:50]}"
        return (f"{line} | would_enter {self.stats['would_enter']} "
                f"entered {self.stats['entered']} open {len(self._book.open_positions())}")

    # ── loop ──────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("CrossGuard started | poll=%.0fs", self._poll)
        try:
            while True:
                try:
                    await self.poll_once()
                except CircuitBreakerTripped:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("CrossGuard poll failed: %s", exc)
                await asyncio.sleep(self._poll)
        except asyncio.CancelledError:
            logger.info("CrossGuard stopped")
            raise

    async def poll_once(self) -> None:
        now = time.time()
        await self._watch_open(now)
        await self._scan_entries(now)

    # ── entries ───────────────────────────────────────────────────────────────

    async def _rate(self, condition_id: str) -> "float | None":
        if self._fees is None:
            return None
        try:
            return await self._fees.get_taker_rate(condition_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("CrossGuard | fee lookup failed for %s: %s", condition_id[:12], exc)
            return None

    async def _scan_entries(self, now: float) -> None:
        imps = load_implications(self._imp_path)
        stops: dict[str, int] = {}
        best: "tuple[float, str] | None" = None
        for imp in imps:
            stop, edge = await self._consider(imp, now)
            stops[stop] = stops.get(stop, 0) + 1
            if edge is not None and (best is None or edge > best[0]):
                best = (edge, imp.narrow_title)
        self._last_pass = (now, len(imps), stops, best)

    async def _consider(self, imp: Implication, now: float) -> "tuple[str, float | None]":
        """
        One implication through every gate. Returns where it stopped and, once
        both asks are priced on a genuine implication, the edge they quote.
        """
        key = imp.key
        if self._book.has_open(*key):
            return "held", None
        if key in self._stuck:
            return "stuck", None
        if self._cooldown.get(key, 0.0) > now:
            return "cooldown", None
        ends = imp.resolves_ts
        # Cheap window check before spending two book reads on it.
        if ends is None:
            return "end_unknown", None
        if ends <= now:
            return "past_end", None
        if (ends - now) / 86_400.0 > self._max_lockup:
            return "outside_window", None
        self.stats["evaluated"] += 1
        # A book can be gone — the market closed, or the token was never
        # tradeable. That is this pair's answer, not the pass's: letting it
        # raise aborted the whole scan, and on 2026-09-12 the guard went 12
        # minutes without completing one, exits included.
        try:
            no_book = await self._client.get_orderbook(imp.narrow_no_token)
            yes_book = await self._client.get_orderbook(imp.broad_yes_token)
        except Exception as exc:  # noqa: BLE001
            logger.debug("CrossGuard | no book for %s: %s", imp.narrow_title[:40], exc)
            self.last_reason[key] = f"orderbook unavailable: {exc}"
            return "no_book", None
        rate_narrow, rate_broad = await self._rate(imp.narrow), await self._rate(imp.broad)
        opp, reason = evaluate(
            imp, no_book, yes_book, now=now,
            rate_narrow=rate_narrow, rate_broad=rate_broad,
            max_lockup_days=self._max_lockup, min_edge=self._min_edge,
            max_usdc=self._max_usdc, min_shares=self._min_shares,
            min_time_to_end_s=self._min_time_to_end,
        )
        self.last_reason[key] = reason
        stop, edge = stop_key(reason), None
        if stop in _PRICED_STOPS:
            quote = quoted_entry(no_book, yes_book, rate_narrow, rate_broad)
            edge = None if quote is None else 1.0 - quote[2]
        if opp is None:
            return stop, edge
        if not self._breaker.check_cross(opp.committed):
            self.last_reason[key] = "blocked by the breaker"
            return "breaker", edge
        # These markets do not offer 47% arbitrages. One quoted on 2026-09-12,
        # two hours from resolution, in a pass where 16 of 17 other pairs had no
        # ask at all — a stale book, not an edge. A real one is still there on
        # the next poll; a ghost is not.
        if opp.edge_per_pair >= self._suspicious_edge:
            seen = self._spike.get(key, 0.0)
            self._spike[key] = now
            if now - seen > CROSS_SPIKE_CONFIRM_S:
                logger.info(
                    "CrossGuard | %+.4f/pair on %s is above %.2f — waiting for a "
                    "second poll to confirm it is not a stale book",
                    opp.edge_per_pair, imp.narrow_title[:40], self._suspicious_edge,
                )
                self.last_reason[key] = "edge above the plausible band, unconfirmed"
                return "unconfirmed_spike", edge
        if not self._enabled:
            self.stats["would_enter"] += 1
            self._cooldown[key] = now + CROSS_ENTRY_COOLDOWN_S
            logger.info(
                "CrossGuard | WOULD ENTER (disabled) %s ⊆ %s | %.2f pairs @ %.4f "
                "→ edge %+.4f/pair · %.2f USDC · %.1fd",
                imp.narrow_title[:40], imp.broad_title[:40], opp.shares,
                opp.entry_per_pair, opp.edge_per_pair, opp.committed,
                opp.lockup_days,
            )
            return "would_enter", edge
        await self._enter(opp, now)
        return "attempted", edge

    async def _buy(self, token: str, price: float, shares: float) -> "float | None":
        """FOK buy. Returns shares received, or None when nothing filled."""
        try:
            resp = await self._client.post_order(
                token_id=token, side="BUY", price=price, size=shares)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossGuard | buy failed on %s: %s", token[:12], exc)
            return None
        status = str((resp or {}).get("status", "")).strip().lower()
        if status not in _FILLED_STATUSES:
            logger.info("CrossGuard | buy not filled on %s: %s", token[:12], status or resp)
            return None
        # On a BUY the taker gives collateral (making) and gets shares (taking).
        try:
            got = float(resp.get("taking_amount") or 0.0)
        except (TypeError, ValueError):
            got = 0.0
        if not 0.5 * shares <= got <= 2.0 * shares:
            got = shares
        return got

    async def _sell(self, token: str, shares: float, cost_per_share: float
                    ) -> "float | None":
        """Sell at the bid. Returns realised P&L, or None when it did not fill."""
        try:
            resp = await self._client.unwind_leg(token, shares)
        except Exception as exc:  # noqa: BLE001
            logger.error("CrossGuard | sell failed on %s: %s", token[:12], exc)
            return None
        status = str((resp or {}).get("status", "")).strip().lower()
        if status not in _FILLED_STATUSES:
            logger.error("CrossGuard | sell not filled on %s: %s", token[:12], status or resp)
            return None
        # On a SELL the taker gives shares (making) and gets collateral (taking).
        proceeds = float(resp.get("taking_amount") or 0.0)
        sold = float(resp.get("making_amount") or shares)
        return round(proceeds - sold * cost_per_share, 6)

    async def _enter(self, opp: Opportunity, now: float) -> None:
        imp, key = opp.imp, opp.imp.key
        no_cost = buy_cost(opp.no_ask.price, opp.rate_narrow)
        yes_cost = buy_cost(opp.yes_ask.price, opp.rate_broad)
        legs = [
            ("NO on narrow", imp.narrow_no_token, opp.no_ask, no_cost),
            ("YES on broad", imp.broad_yes_token, opp.yes_ask, yes_cost),
        ]
        # The thinner leg first: if it is the one that fails, nothing is held.
        legs.sort(key=lambda leg: leg[2].size)
        (l1, t1, lv1, c1), (l2, t2, lv2, c2) = legs

        self._breaker.on_cross_open(opp.committed)
        got1 = await self._buy(t1, lv1.price, opp.shares)
        if got1 is None:
            self._breaker.release_cross(opp.committed)
            self._cooldown[key] = now + CROSS_ENTRY_COOLDOWN_S
            return
        got2 = await self._buy(t2, lv2.price, opp.shares)
        if got2 is None:
            self.stats["half_filled"] += 1
            self._cooldown[key] = now + CROSS_ENTRY_COOLDOWN_S
            pnl = await self._sell(t1, got1, c1)
            if pnl is None:
                self.stats["stuck"] += 1
                self._stuck.add(key)
                await self._say(
                    f"🚨 CROSS HALF-FILL STUCK — {l1} filled ({got1:.2f}), {l2} did not, "
                    f"and selling it back failed.\n  {imp.narrow_title[:60]}\n"
                    f"  {imp.broad_title[:60]}\n  Manual decision needed.")
                return
            self._breaker.on_cross_close(pnl, opp.committed)
            await self._say(
                f"⚠️ CROSS HALF-FILL — {l2} did not fill; {l1} sold back, "
                f"{pnl:+.4f} USDC.\n  {imp.narrow_title[:60]}\n  {imp.broad_title[:60]}")
            return

        size = math.floor(min(got1, got2) * 100.0) / 100.0
        pos = CrossPosition(
            narrow=imp.narrow, broad=imp.broad,
            narrow_title=imp.narrow_title, broad_title=imp.broad_title,
            narrow_yes_token=imp.narrow_yes_token, narrow_no_token=imp.narrow_no_token,
            broad_yes_token=imp.broad_yes_token, broad_no_token=imp.broad_no_token,
            size=size, narrow_no_paid=no_cost, broad_yes_paid=yes_cost,
            opened_ts=now, resolves_ts=imp.resolves_ts, paper=False,
        )
        # The ledger reserved the planned size; hold exactly what was bought.
        actual = pos.entry_cost * pos.size
        if abs(actual - opp.committed) > 1e-6:
            self._breaker.release_cross(opp.committed)
            self._breaker.on_cross_open(actual)
        self._book.open(pos)
        self._book.save()
        self.stats["entered"] += 1
        await self._say(
            f"💰 CROSS POSITION OPENED\n"
            f"  NO  {imp.narrow_title[:56]} @ {no_cost:.4f}\n"
            f"  YES {imp.broad_title[:56]} @ {yes_cost:.4f}\n"
            f"  {size:.2f} pairs · entry {pos.entry_cost:.4f} → guaranteed "
            f"{pos.guaranteed_edge:+.4f}/pair ({pos.guaranteed_edge * size:+.2f} USDC) "
            f"· resolves in {opp.lockup_days:.1f}d")

    # ── open positions ────────────────────────────────────────────────────────

    async def _watch_open(self, now: float) -> None:
        for pos in self._book.open_positions():
            if pos.key in self._stuck:
                continue
            if pos.resolves_ts is not None and now >= pos.resolves_ts:
                await self._try_resolve(pos, now)
                continue
            await self._try_exit(pos, now)

    async def _try_resolve(self, pos: CrossPosition, now: float) -> None:
        ny, by = await self._resolved(pos.narrow), await self._resolved(pos.broad)
        if ny is None or by is None:
            return                                   # not settled yet — keep waiting
        per_pair = resolution_profit(pos, ny, by)
        committed = pos.entry_cost * pos.size
        self._book.close(pos, route="resolution", profit=per_pair,
                         status="resolved", now=now)
        self._book.save()
        self.stats["resolved"] += 1
        await self._say(
            f"🏁 CROSS POSITION RESOLVED — {per_pair * pos.size:+.4f} USDC\n"
            f"  {pos.narrow_title[:56]} → {'YES' if ny else 'NO'}\n"
            f"  {pos.broad_title[:56]} → {'YES' if by else 'NO'}"
            + ("\n  ⚠️ narrow YES with broad NO: the implication was FALSE"
               if ny and not by else ""))
        self._breaker.on_cross_close(per_pair * pos.size, committed)

    async def _try_exit(self, pos: CrossPosition, now: float) -> None:
        no_bid = best_level(await self._client.get_orderbook(pos.narrow_no_token), "bids")
        yes_bid = best_level(await self._client.get_orderbook(pos.broad_yes_token), "bids")
        if no_bid is None or yes_bid is None:
            return
        # Both legs must sell in full at the touch, or the exit leaves one naked.
        if no_bid.size < pos.size or yes_bid.size < pos.size:
            return
        rn, rb = await self._rate(pos.narrow), await self._rate(pos.broad)
        per_pair = (sell_proceeds(no_bid.price, rn) + sell_proceeds(yes_bid.price, rb)
                    - pos.entry_cost)
        quote = ExitQuote(ROUTE_SELL, per_pair, None, per_pair)
        if not should_exit(pos, quote):
            return
        if not self._enabled:
            logger.info("CrossGuard | WOULD EXIT (disabled) %s | %+.4f/pair against "
                        "%+.4f guaranteed", pos.narrow_title[:40], per_pair,
                        pos.guaranteed_edge)
            return
        committed = pos.entry_cost * pos.size
        pnl1 = await self._sell(pos.narrow_no_token, pos.size, pos.narrow_no_paid)
        if pnl1 is None:
            return                                   # nothing sold — try again later
        pnl2 = await self._sell(pos.broad_yes_token, pos.size, pos.broad_yes_paid)
        if pnl2 is None:
            self.stats["stuck"] += 1
            self._stuck.add(pos.key)
            await self._say(
                f"🚨 CROSS EXIT HALF-DONE — NO on narrow sold ({pnl1:+.4f}), YES on "
                f"broad did not.\n  {pos.broad_title[:60]}\n  Manual decision needed.")
            return
        realised = pnl1 + pnl2
        self._book.close(pos, route=ROUTE_SELL, profit=realised / pos.size,
                         status="exited", now=now)
        self._book.save()
        self.stats["exited"] += 1
        await self._say(
            f"🔓 CROSS POSITION EXITED EARLY — {realised:+.4f} USDC after "
            f"{pos.days_held(now):.1f}d\n  {pos.narrow_title[:56]}\n  {pos.broad_title[:56]}")
        self._breaker.on_cross_close(realised, committed)

    async def _say(self, text: str) -> None:
        try:
            await self._notifier.notify(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossGuard | notify failed: %s", exc)


async def _gamma_resolution(condition_id: str) -> "bool | None":
    """
    True if YES won, False if NO won, None while unresolved or unclear.

    Filtered on `condition_ids` — the camel-case key is silently ignored and
    returns an unfiltered page.
    """
    import httpx  # noqa: PLC0415

    try:
        async with httpx.AsyncClient(timeout=20.0) as h:
            r = await h.get(_GAMMA, params={"condition_ids": condition_id})
        rows = r.json() if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        return None
    for m in rows:
        if str(m.get("conditionId")) != condition_id:
            continue
        try:
            yes, no = (float(x) for x in json.loads(m.get("outcomePrices") or "[]"))
        except (TypeError, ValueError):
            return None
        if {yes, no} != {0.0, 1.0}:
            return None
        return yes == 1.0
    return None
