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
from strategy.outcomes import is_yes_no
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
    taker_fee_per_share,
)

logger = logging.getLogger(__name__)


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


CROSS_EXECUTION_ENABLED: bool = _flag("CROSS_EXECUTION_ENABLED", "false")
# Restrict trading to markets whose first token is provably "Yes". It went in on
# 2026-09-13, when the model read an O/U line as Under while the token bought
# would have been Over (strategy/outcomes.py). Off since 2026-09-15: with each
# market's YES outcome stated in the prompt, all 355 over/under implications
# asserted over the next 19 hours scored correct for the first token, and the
# restriction was blocking 68% of pairs — at whole-window scale, 189 of 189.
# A leg whose outcome labels are unknown is still refused: nothing stated which
# outcome its first token is.
CROSS_YES_NO_ONLY: bool = _flag("CROSS_YES_NO_ONLY", "false")
CROSS_IMPLICATIONS_PATH: str = os.environ.get(
    "CROSS_IMPLICATIONS_PATH",
    "/home/ubuntu/polybot-dev/cross-exec/cross_implications.json",
)
CROSS_POSITIONS_PATH: str = os.environ.get(
    "CROSS_POSITIONS_PATH", "cross_positions_live.json"
)
# Halved once the adaptive re-check landed: most pairs are deferred on any given
# poll, so looking twice as often costs little and halves the delay on the ones
# sitting at the threshold.
CROSS_POLL_S: float = float(os.environ.get("CROSS_POLL_S", 15.0))
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
# Resting our own bids instead of crossing the spread. A taker pays the ask plus
# the fee on both legs; measured on the live book that is 1.5–3.4 points of fee
# and a spread on top, and the best edge the taker path has seen in six days is
# −1%. A maker fill pays no fee and buys a tick above the bid, which is where
# that same pair can become tradeable. The cost is queue risk: an order can rest
# unfilled, or one leg fills and the other does not.
CROSS_MAKER_ENABLED: bool = _flag("CROSS_MAKER_ENABLED", "true")
CROSS_MAKER_TTL_S: float = float(os.environ.get("CROSS_MAKER_TTL_S", 180.0))
CROSS_MAKER_MIN_EDGE: float = float(os.environ.get("CROSS_MAKER_MIN_EDGE", CROSS_MIN_EDGE))
# A bid a tick above an almost empty book is not an offer anyone will take: on
# 2026-09-17 that produced "+0.96 edge" pairs whose books stood at 0.01 / 0.50.
# Only a book tight enough for our bid to be near the market can be rested in.
CROSS_MAKER_MAX_SPREAD: float = float(os.environ.get("CROSS_MAKER_MAX_SPREAD", 0.05))
# Cross slots the maker path may never take. On 2026-09-23 its resting pairs held
# the single cross slot nearly all day, 62 rests without one fill, and 37,794
# entries were refused behind them — a taker arbitrage like that morning's
# NVDA +2.13% would have been one of them. A slot kept free for the taker path
# (and one position's capital with it) means resting never blocks crossing.
CROSS_TAKER_RESERVE_SLOTS: int = int(os.environ.get("CROSS_TAKER_RESERVE_SLOTS", 1))
# How close to its end a pair may still get a resting bid. The taker path keeps
# thirty minutes: near a market's end its book is a few stale quotes. A match
# market's end is its kick-off, and there the book is the busiest it will be —
# of the trades before kick-off on finished matches' tail markets, 36% came in
# the last hour. A rest is withdrawn at kick-off either way.
CROSS_MAKER_MIN_TIME_TO_END_S: float = float(
    os.environ.get("CROSS_MAKER_MIN_TIME_TO_END_S", 300.0))
# Polymarket's tick is 0.01 in the middle of the range and 0.001 at the extremes.
# The reader exports each market's own; this is the fallback.
CROSS_DEFAULT_TICK: float = 0.01
# With a wide net the file holds hundreds of pairs, and pricing every one of
# them every poll is two book reads each against the exchange's rate limit. A
# pair 30 points from the threshold does not become tradeable in 30 seconds, so
# attention goes where the gap is small: near the threshold every poll, mid
# band every few minutes, the rest rarely.
CROSS_RECHECK_MID_S: float = float(os.environ.get("CROSS_RECHECK_MID_S", 120.0))
CROSS_RECHECK_FAR_S: float = float(os.environ.get("CROSS_RECHECK_FAR_S", 600.0))
CROSS_NEAR_BAND: float = float(os.environ.get("CROSS_NEAR_BAND", 0.03))
# Book reads per poll, two a pair. With the whole one-week window in the file
# there are more due pairs than a poll can price without leaning on the
# exchange's rate limit; the rest wait for the next poll, nearest first.
CROSS_MAX_BOOK_READS_PER_POLL: int = int(os.environ.get("CROSS_MAX_BOOK_READS_PER_POLL", 60))
# When the client can, books come in batches instead (POST /books, 100 tokens a
# request). One read at a time, the budget priced 87 pairs a minute and a sweep
# of the file took ten. Eight batches a poll is up to 800 books in eight
# requests, against the 60 requests of the single reads they replace — sized for
# the ~8,700 pairs the file holds once the match over/unders' structural
# implications are in it. The single reads stay as the fallback, at their old
# budget, for a poll whose batch read fails.
CROSS_BOOK_BATCH: int = int(os.environ.get("CROSS_BOOK_BATCH", 100))
CROSS_MAX_BOOK_BATCHES_PER_POLL: int = int(os.environ.get("CROSS_MAX_BOOK_BATCHES_PER_POLL", 8))
# Fee lookups on Gamma, per poll, for pairs the reader exported without a rate.
# Hundreds of pairs a poll would otherwise be hundreds of Gamma requests, and the
# main strategy's market scanner shares that rate limit. Past the budget a pair
# is priced at CROSS_TAKER_RATE until FeeEngine's cache has its market.
CROSS_MAX_FEE_LOOKUPS_PER_POLL: int = int(os.environ.get("CROSS_MAX_FEE_LOOKUPS_PER_POLL", 20))
# How much longer a book too wide to rest in may wait for a spare read than a
# tight one. An absolute preference starved them: with 504 tight pairs ahead,
# the 471 wide ones were priced only when their ten-minute re-check fell due,
# and the stalest price climbed a minute every minute. With a head start instead,
# tight books are priced about every 40 s and wide ones about every 100 s.
CROSS_WIDE_BOOK_PENALTY_S: float = float(os.environ.get("CROSS_WIDE_BOOK_PENALTY_S", 60.0))
CROSS_MID_BAND: float = float(os.environ.get("CROSS_MID_BAND", 0.10))
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
    narrow_outcomes:  "tuple[str, ...] | None" = None
    broad_outcomes:   "tuple[str, ...] | None" = None
    narrow_tick:      "float | None" = None
    broad_tick:       "float | None" = None
    # Taker fee rates as the reader read them off Gamma; None means ask FeeEngine.
    narrow_fee_rate:  "float | None" = None
    broad_fee_rate:   "float | None" = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.narrow, self.broad)

    @property
    def resolves_ts(self) -> "float | None":
        if self.narrow_end_ts is None or self.broad_end_ts is None:
            return None
        return max(self.narrow_end_ts, self.broad_end_ts)


def _take_batch(order: "list[Implication]", capacity: int) -> "list[Implication]":
    """The pairs, in order, whose books fit in `capacity` distinct tokens. Pairs
    share markets, so a token already in the batch costs nothing again."""
    seen: set[str] = set()
    batch: list[Implication] = []
    for imp in order:
        new = {imp.narrow_no_token, imp.broad_yes_token} - seen
        if len(seen) + len(new) > capacity:
            break
        seen |= new
        batch.append(imp)
    return batch


def _opt_rate(v) -> "float | None":
    try:
        rate = float(v)
    except (TypeError, ValueError):
        return None
    return rate if 0.0 <= rate <= 1.0 else None


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
                narrow_outcomes=(tuple(str(o) for o in r["narrow_outcomes"])
                                 if r.get("narrow_outcomes") else None),
                broad_outcomes=(tuple(str(o) for o in r["broad_outcomes"])
                                if r.get("broad_outcomes") else None),
                narrow_tick=(float(r["narrow_tick"]) if r.get("narrow_tick") else None),
                broad_tick=(float(r["broad_tick"]) if r.get("broad_tick") else None),
                # 0.0 is a rate (fees off), so this one is tested against None.
                narrow_fee_rate=_opt_rate(r.get("narrow_fee_rate")),
                broad_fee_rate=_opt_rate(r.get("broad_fee_rate")),
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
    ("too wide a book to rest in", "book_too_wide"),
    ("a one-sided book to rest in", "one_sided_book"),
    ("at our own bids", "maker_edge_below_min"),
    ("nothing to rest above", "no_bid"),
    ("outcome labels are unknown", "outcomes_unknown"),
    ("not a Yes/No market", "not_yes_no"),
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
_PRICED_STOPS = frozenset({"edge_below_min", "thin_touch", "ok", "maker_edge_below_min"})
_SHARE_EPS: float = 0.005


def stop_key(reason: str) -> str:
    if reason == "ok":
        return "ok"
    for needle, key in _STOP_KEYS:
        if needle in reason:
            return key
    return "other"


def tick_of(token_id: str, stated: "float | None") -> float:
    """The market's price increment: what the reader exported, else the client's cache."""
    if stated and stated > 0:
        return float(stated)
    try:
        from core.clob_client import peek_market_meta  # noqa: PLC0415
        meta = peek_market_meta(token_id)
        if meta and meta[0] > 0:
            return float(meta[0])
    except Exception:  # noqa: BLE001 — a missing cache is not an error
        pass
    return CROSS_DEFAULT_TICK


def maker_price(book: dict, tick: float) -> "float | None":
    """
    Where to rest a BUY: one tick above the best bid, or alongside it when that
    tick would cross the spread. None without a bid to rest above — a leg with
    no bid has nobody to sell back to either.
    """
    bid, ask = best_level(book, "bids"), best_level(book, "asks")
    if bid is None:
        return None
    price = round(bid.price + tick, 4)
    if ask is not None and price >= ask.price - 1e-9:
        price = round(bid.price, 4)
    return price if 0.0 < price < 1.0 else None


@dataclass
class MakerAttempt:
    """Two of our own bids resting on two different markets."""
    narrow:        str
    broad:         str
    narrow_title:  str
    broad_title:   str
    no_token:      str
    yes_token:     str
    no_price:      float
    yes_price:     float
    shares:        float
    no_order:      str
    yes_order:     str
    placed_at:     float
    resolves_ts:   "float | None"
    committed:     float
    no_filled:     float = 0.0
    yes_filled:    float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.narrow, self.broad)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "narrow", "broad", "narrow_title", "broad_title", "no_token", "yes_token",
            "no_price", "yes_price", "shares", "no_order", "yes_order", "placed_at",
            "resolves_ts", "committed", "no_filled", "yes_filled")}


@dataclass(frozen=True)
class MakerOpportunity:
    imp:            Implication
    no_price:       float      # our bid for NO on narrow
    yes_price:      float      # our bid for YES on broad
    shares:         float
    entry_per_pair: float      # no fee: a maker fill pays none
    lockup_days:    float

    @property
    def edge_per_pair(self) -> float:
        return 1.0 - self.entry_per_pair

    @property
    def committed(self) -> float:
        return self.shares * self.entry_per_pair


def maker_entry(imp: Implication, no_book: dict, yes_book: dict) -> "tuple[float, float, float] | None":
    """(NO price, YES price, cost per pair) if both legs can be quoted, else None."""
    no_price = maker_price(no_book, tick_of(imp.narrow_no_token, imp.narrow_tick))
    yes_price = maker_price(yes_book, tick_of(imp.broad_yes_token, imp.broad_tick))
    if no_price is None or yes_price is None:
        return None
    return no_price, yes_price, round(no_price + yes_price, 6)


def evaluate_maker(
    imp: Implication, no_book: dict, yes_book: dict, *,
    now: float,
    max_lockup_days: float = CROSS_MAX_LOCKUP_DAYS,
    min_edge: float = CROSS_MAKER_MIN_EDGE,
    max_usdc: float = CROSS_MAX_POSITION_USDC,
    min_shares: float = CROSS_MIN_SHARES,
    min_time_to_end_s: float = CROSS_MIN_TIME_TO_END_S,
    yes_no_only: bool = CROSS_YES_NO_ONLY,
    max_spread: float = CROSS_MAKER_MAX_SPREAD,
) -> "tuple[MakerOpportunity | None, str]":
    """The same gates as a taker entry, priced at the bids we would rest."""
    gate = _entry_gates(imp, now, max_lockup_days, min_time_to_end_s, yes_no_only)
    if gate is not None:
        return None, gate
    for book, leg in ((no_book, "NO on narrow"), (yes_book, "YES on broad")):
        bid, ask = best_level(book, "bids"), best_level(book, "asks")
        # No offer at all is the widest book there is, not a book the spread gate
        # may skip: on 2026-09-18 such legs let "+0.959" through — a bid of 0.03
        # for a NO worth 0.999, which only a fat finger would ever sell into.
        if bid is not None and ask is None:
            return None, f"{leg} has a bid and no offer — a one-sided book to rest in"
        # A book exactly at the cap must pass: 0.55 - 0.50 is 0.05000000000000004.
        if bid is not None and ask is not None and ask.price - bid.price > max_spread + 1e-9:
            return None, (f"{leg} quotes {bid.price:.3f}/{ask.price:.3f} — too wide a "
                          f"book to rest in")
    quote = maker_entry(imp, no_book, yes_book)
    if quote is None:
        return None, "a leg has no bid — nothing to rest above"
    no_price, yes_price, entry = quote
    if 1.0 - entry < min_edge:
        return None, (f"maker edge {1.0 - entry:+.4f} < {min_edge:.4f} at our own bids")
    shares = math.floor(max_usdc / entry * 100.0) / 100.0 if entry > 0 else 0.0
    if shares < min_shares:
        return None, (f"depth at the best ask ({shares:.2f} shares) does not reach the "
                      f"{min_shares:g}-share minimum")
    lockup = ((imp.resolves_ts or now) - now) / 86_400.0
    return MakerOpportunity(imp, no_price, yes_price, shares, entry, lockup), "ok"


def _entry_gates(imp: Implication, now: float, max_lockup_days: float,
                 min_time_to_end_s: float, yes_no_only: bool) -> "str | None":
    """The gates that do not depend on how the legs are priced."""
    if same_quantity(imp.narrow_title, imp.broad_title) is False:
        return "the two markets count different statistics — not an implication"
    if imp.narrow_outcomes is None or imp.broad_outcomes is None:
        return "a leg's outcome labels are unknown — its YES token is not proven"
    if yes_no_only and not (is_yes_no(imp.narrow_outcomes) and is_yes_no(imp.broad_outcomes)):
        return "a leg is not a Yes/No market — its YES token is not proven"
    ends = imp.resolves_ts
    if ends is None:
        return "resolution date unknown — cannot show it fits the window"
    if (ends - now) <= 0.0:
        return "already past its end date"
    if (ends - now) / 86_400.0 > max_lockup_days:
        return f"locks capital {(ends - now) / 86_400.0:.1f}d > {max_lockup_days:g}d window"
    if ends - now < min_time_to_end_s:
        return (f"resolves in {(ends - now) / 60.0:.0f} min, under the "
                f"{min_time_to_end_s / 60.0:.0f} min floor — dying book")
    return None


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
    yes_no_only: bool = CROSS_YES_NO_ONLY,
) -> "tuple[Opportunity | None, str]":
    """Every entry gate, from the two books. Returns (opportunity, reason)."""
    gate = _entry_gates(imp, now, max_lockup_days, min_time_to_end_s, yes_no_only)
    if gate is not None:
        return None, gate
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
    lockup = ((imp.resolves_ts or now) - now) / 86_400.0
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
        yes_no_only:       "bool | None" = None,
        max_book_reads:    "int | None" = None,
        book_batches:      "int | None" = None,
        maker_enabled:     "bool | None" = None,
        maker_ttl_s:       "float | None" = None,
        maker_min_edge:    "float | None" = None,
        taker_reserve:     "int | None" = None,
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
        self._yes_no_only = CROSS_YES_NO_ONLY if yes_no_only is None else yes_no_only
        self._max_reads = (CROSS_MAX_BOOK_READS_PER_POLL if max_book_reads is None
                           else max_book_reads)
        self._book_batches = (CROSS_MAX_BOOK_BATCHES_PER_POLL if book_batches is None
                              else book_batches)
        # This poll's batched books, token → book; None outside a scan or when
        # the batch read failed and the pairs are read one at a time.
        self._poll_books: "dict[str, dict] | None" = None
        self._fee_lookups_left = CROSS_MAX_FEE_LOOKUPS_PER_POLL
        self._imp_cache: "tuple[tuple[int, int] | None, list[Implication]]" = (None, [])
        self._next_check: dict[tuple[str, str], float] = {}
        # What each pair was last judged to be, so a deferred pair still reports
        # its price rather than vanishing behind "deferred".
        self._maker_enabled = CROSS_MAKER_ENABLED if maker_enabled is None else maker_enabled
        self._maker_ttl = CROSS_MAKER_TTL_S if maker_ttl_s is None else maker_ttl_s
        self._maker_min_edge = (CROSS_MAKER_MIN_EDGE if maker_min_edge is None
                                else maker_min_edge)
        self._taker_reserve = CROSS_TAKER_RESERVE_SLOTS if taker_reserve is None else taker_reserve
        self._attempts: dict[tuple[str, str], MakerAttempt] = {}
        self._attempts_path = Path(positions_path or CROSS_POSITIONS_PATH).with_name(
            Path(positions_path or CROSS_POSITIONS_PATH).stem + "_maker.json")
        # Every rest the guard would have made while disabled, one JSON line each,
        # for scripts/paper_fills.py to replay against the public trade feed.
        self._paper_path = Path(positions_path or CROSS_POSITIONS_PATH).with_name(
            Path(positions_path or CROSS_POSITIONS_PATH).stem + "_paper_rests.jsonl")
        self._orphans: list[MakerAttempt] = []
        self._last_stop: dict[tuple[str, str], str] = {}
        self._last_edge: dict[tuple[str, str], "float | None"] = {}
        # When each pair's books were last read, and when this guard first looked
        # at the file: the stalest price is how late we can be to a dislocation.
        self._priced_at: dict[tuple[str, str], float] = {}
        self._first_scan: "float | None" = None
        self._resolved = resolution_lookup or _gamma_resolution
        self._book = PositionBook(positions_path or CROSS_POSITIONS_PATH).load()
        self._cooldown: dict[tuple[str, str], float] = {}
        self._stuck: set[tuple[str, str]] = set()
        self.last_reason: dict[tuple[str, str], str] = {}
        self.stats = {"evaluated": 0, "would_enter": 0, "entered": 0,
                      "half_filled": 0, "exited": 0, "resolved": 0, "stuck": 0,
                      "would_rest": 0, "rested": 0, "maker_filled": 0}
        # The last complete pass over the reader's file, for stop_summary():
        # (when, implications read, stops by key, (best edge, narrow title),
        #  pairs not priced this pass, age of the stalest price in seconds).
        self._last_pass: ("tuple[float, int, dict[str, int], "
                          "tuple[float, str] | None, int, float | None] | None") = None
        logger.info(
            "CrossGuard init | %s | maker=%s (ttl %.0fs, min_edge %.3f, %d slot(s) kept "
            "for takers) | window=%gd min_edge=%.3f max_pos=%.2f USDC open=%d file=%s",
            "EXECUTING" if self._enabled else "evaluate only (disabled)",
            "on" if self._maker_enabled else "off", self._maker_ttl, self._maker_min_edge,
            self._taker_reserve, self._max_lockup, self._min_edge, self._max_usdc,
            len(self._book.open_positions()), self._imp_path,
        )
        headroom = getattr(breaker, "cross_headroom", None)
        if (self._enabled and self._maker_enabled and self._taker_reserve > 0
                and callable(headroom) and headroom()[0] <= self._taker_reserve):
            logger.warning(
                "CrossGuard | maker path can never rest: CROSS_MAX_POSITIONS leaves no "
                "slot beyond the %d kept for taker entries", self._taker_reserve)

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
        # Orders we left resting belong to a process that is gone: nothing would
        # watch them fill. They are cancelled on the first poll.
        try:
            raw = json.loads(self._attempts_path.read_text())
        except (OSError, ValueError):
            raw = []
        for row in raw if isinstance(raw, list) else []:
            try:
                self._orphans.append(MakerAttempt(**row))
            except TypeError:
                continue
        if self._orphans:
            logger.warning("CrossGuard | %d maker order pair(s) left resting by the "
                           "previous process — cancelling on the first poll",
                           len(self._orphans))
        return n

    def _save_attempts(self) -> None:
        try:
            tmp = self._attempts_path.with_suffix(self._attempts_path.suffix + ".tmp")
            tmp.write_text(json.dumps([a.as_dict() for a in self._attempts.values()]))
            tmp.replace(self._attempts_path)
        except OSError as exc:
            logger.warning("CrossGuard | cannot write %s: %s", self._attempts_path, exc)

    def _recent(self) -> list[CrossPosition]:
        cutoff = time.time() - CROSS_REDEEM_TAIL_DAYS * 86_400.0
        return self._book.open_positions() + [
            p for p in self._book.closed if (p.closed_ts or 0.0) >= cutoff
        ]

    def condition_ids(self) -> set[str]:
        """Markets AutoRedeemer should watch: held now, or resolved recently."""
        out = {c for p in self._recent() for c in (p.narrow, p.broad)}
        out |= {c for a in self._attempts.values() for c in (a.narrow, a.broad)}
        return out

    def managed_titles(self) -> set[str]:
        """Titles whose inventory this guard owns, for WalletReconciler."""
        out = {t for p in self._recent() for t in (p.narrow_title, p.broad_title) if t}
        # A maker order that fills puts shares in the wallet before any position
        # exists; without this the reconciler would call them an escape.
        out |= {t for a in self._attempts.values()
                for t in (a.narrow_title, a.broad_title) if t}
        return out

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
        ts, total, stops, best, deferred, stalest = self._last_pass
        head = f"last pass {time.time() - ts:.0f}s ago"
        if not total:
            return f"{head} | no implications in {self._imp_path}"
        parts = ", ".join(f"{k} {v}" for k, v in
                          sorted(stops.items(), key=lambda kv: (-kv[1], kv[0])))
        line = f"{head} | {total} implication(s): {parts}"
        if deferred:
            line += f" | {deferred} not re-priced this pass"
        if stalest is not None:
            line += f" | stalest price {stalest:.0f}s"
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
        await self._cancel_orphans()
        await self._watch_maker(now)
        await self._watch_open(now)
        await self._scan_entries(now)

    # ── resting our own bids ──────────────────────────────────────────────────

    async def _cancel_orphans(self) -> None:
        """Cancel maker orders a previous process left resting; nothing watches them."""
        while self._orphans:
            a = self._orphans.pop()
            for order in (a.no_order, a.yes_order):
                if not order:
                    continue
                try:
                    await self._client.cancel_order(order)
                    logger.warning("CrossGuard | cancelled orphaned maker order %s", order[:12])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("CrossGuard | orphan cancel %s failed: %s", order[:12], exc)
        if self._attempts_path.exists() and not self._attempts:
            self._save_attempts()

    async def _post_maker(self, token: str, price: float, shares: float) -> "str | None":
        try:
            resp = await self._client.post_maker_order(
                token_id=token, side="BUY", desired_price=price, size=shares)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossGuard | maker post failed on %s: %s", token[:12], exc)
            return None
        oid = str((resp or {}).get("order_id") or (resp or {}).get("orderID") or "")
        return oid or None

    async def _rest_maker(self, opp: MakerOpportunity, now: float) -> bool:
        imp = opp.imp
        self._breaker.on_cross_open(opp.committed)
        no_id = await self._post_maker(imp.narrow_no_token, opp.no_price, opp.shares)
        yes_id = await self._post_maker(imp.broad_yes_token, opp.yes_price, opp.shares)
        if not no_id or not yes_id:
            for oid in (no_id, yes_id):
                if oid:
                    try:
                        await self._client.cancel_order(oid)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("CrossGuard | cancel after a failed pair: %s", exc)
            self._breaker.release_cross(opp.committed)
            self._cooldown[imp.key] = now + CROSS_ENTRY_COOLDOWN_S
            logger.warning("CrossGuard | could not rest both legs on %s — nothing left open",
                           imp.narrow_title[:40])
            return False
        self._attempts[imp.key] = MakerAttempt(
            narrow=imp.narrow, broad=imp.broad, narrow_title=imp.narrow_title,
            broad_title=imp.broad_title, no_token=imp.narrow_no_token,
            yes_token=imp.broad_yes_token, no_price=opp.no_price, yes_price=opp.yes_price,
            shares=opp.shares, no_order=no_id, yes_order=yes_id, placed_at=now,
            resolves_ts=imp.resolves_ts, committed=opp.committed)
        self._save_attempts()
        self.stats["rested"] += 1
        await self._say(
            f"🪧 CROSS MAKER ORDERS RESTING\n"
            f"  NO  {imp.narrow_title[:56]} @ {opp.no_price:.4f}\n"
            f"  YES {imp.broad_title[:56]} @ {opp.yes_price:.4f}\n"
            f"  {opp.shares:.2f} pairs · entry {opp.entry_per_pair:.4f} → "
            f"{opp.edge_per_pair:+.4f}/pair if both fill · {self._maker_ttl:.0f}s to fill")
        return True

    async def _refresh_attempt(self, a: MakerAttempt) -> None:
        for leg in ("no", "yes"):
            order, token = getattr(a, f"{leg}_order"), getattr(a, f"{leg}_token")
            if not order:
                continue
            try:
                filled = await self._client.order_filled_size(order, token)
            except Exception as exc:  # noqa: BLE001
                logger.debug("CrossGuard | fill lookup failed for %s: %s", order[:12], exc)
                continue
            if filled is not None:
                setattr(a, f"{leg}_filled",
                        max(getattr(a, f"{leg}_filled"), min(a.shares, float(filled))))

    async def _watch_maker(self, now: float) -> None:
        for a in list(self._attempts.values()):
            await self._refresh_attempt(a)
            if min(a.no_filled, a.yes_filled) >= a.shares - _SHARE_EPS:
                await self._settle_attempt(a, now)           # both legs filled
                continue
            if now - a.placed_at < self._maker_ttl:
                continue
            for order in (a.no_order, a.yes_order):
                try:
                    await self._client.cancel_order(order)
                except Exception as exc:  # noqa: BLE001
                    logger.info("CrossGuard | cancel %s failed (may have filled): %s",
                                order[:12], exc)
            # A cancel can race a fill, and the trade feed trails the match.
            await self._refresh_attempt(a)
            await self._settle_attempt(a, now)

    async def _settle_attempt(self, a: MakerAttempt, now: float) -> None:
        """What the resting pair actually became: a position, an excess, or nothing."""
        self._attempts.pop(a.key, None)
        self._save_attempts()
        self._cooldown[a.key] = now + CROSS_ENTRY_COOLDOWN_S
        pairs = math.floor(min(a.no_filled, a.yes_filled) * 100.0) / 100.0
        self._breaker.release_cross(a.committed)
        if pairs > _SHARE_EPS:
            await self._open_from_maker(a, pairs, a.no_price, a.yes_price, now, "maker")
        excess = round(abs(a.no_filled - a.yes_filled), 4)
        if excess > _SHARE_EPS:
            await self._resolve_excess(a, excess, now)
        elif pairs <= _SHARE_EPS:
            logger.info("CrossGuard | maker pair on %s expired unfilled — nothing held",
                        a.narrow_title[:40])

    async def _open_from_maker(self, a: MakerAttempt, size: float, no_paid: float,
                               yes_paid: float, now: float, how: str) -> None:
        imp_pos = CrossPosition(
            narrow=a.narrow, broad=a.broad, narrow_title=a.narrow_title,
            broad_title=a.broad_title, narrow_yes_token="", narrow_no_token=a.no_token,
            broad_yes_token=a.yes_token, broad_no_token="", size=size,
            narrow_no_paid=no_paid, broad_yes_paid=yes_paid, opened_ts=now,
            resolves_ts=a.resolves_ts, paper=False)
        self._breaker.on_cross_open(imp_pos.entry_cost * size)
        self._book.open(imp_pos)
        self._book.save()
        self.stats["maker_filled"] += 1
        await self._say(
            f"💰 CROSS POSITION OPENED ({how})\n"
            f"  NO  {a.narrow_title[:56]} @ {no_paid:.4f}\n"
            f"  YES {a.broad_title[:56]} @ {yes_paid:.4f}\n"
            f"  {size:.2f} pairs · entry {imp_pos.entry_cost:.4f} → guaranteed "
            f"{imp_pos.guaranteed_edge:+.4f}/pair ({imp_pos.guaranteed_edge * size:+.2f} USDC)")

    async def _resolve_excess(self, a: MakerAttempt, excess: float, now: float) -> None:
        """One leg filled further than the other: complete it if that still pays, else sell it back."""
        no_rich = a.no_filled > a.yes_filled
        held_token = a.no_token if no_rich else a.yes_token
        held_paid = a.no_price if no_rich else a.yes_price
        other_token = a.yes_token if no_rich else a.no_token
        ask = best_level(await self._client.get_orderbook(other_token), "asks")
        if ask is not None and ask.size >= excess:
            completed = held_paid + buy_cost(ask.price, None)
            if completed < 1.0:
                got = await self._buy(other_token, ask.price, excess)
                if got:
                    size = math.floor(min(excess, got) * 100.0) / 100.0
                    no_paid = held_paid if no_rich else ask.price
                    yes_paid = ask.price if no_rich else held_paid
                    await self._open_from_maker(a, size, no_paid, yes_paid, now,
                                                "maker + taker completion")
                    return
        self.stats["half_filled"] += 1
        pnl = await self._sell(held_token, excess, held_paid,
                               await self._rate(a.narrow if no_rich else a.broad))
        if pnl is None:
            self.stats["stuck"] += 1
            self._stuck.add(a.key)
            await self._say(
                f"🚨 CROSS MAKER HALF-FILL STUCK — {excess:.2f} shares of "
                f"{'NO ' + a.narrow_title[:50] if no_rich else 'YES ' + a.broad_title[:50]} "
                f"filled alone and could not be sold back. Manual decision needed.")
            return
        self._breaker.book_pnl(pnl)
        await self._say(
            f"⚠️ CROSS MAKER HALF-FILL — {excess:.2f} shares filled on one leg only, "
            f"sold back for {pnl:+.4f} USDC.\n  {a.narrow_title[:56]}\n  {a.broad_title[:56]}")

    # ── entries ───────────────────────────────────────────────────────────────

    async def _pair_rate(self, condition_id: str, exported: "float | None") -> "float | None":
        """The reader's rate, else FeeEngine's cache, else a lookup while this
        poll's budget lasts, else None (priced at CROSS_TAKER_RATE)."""
        if exported is not None:
            return exported
        peek = getattr(self._fees, "peek_taker_rate", None)
        cached = peek(condition_id) if callable(peek) else None
        if cached is not None:
            return cached
        if self._fee_lookups_left <= 0:
            return None
        self._fee_lookups_left -= 1
        return await self._rate(condition_id)

    async def _rate(self, condition_id: str) -> "float | None":
        if self._fees is None:
            return None
        try:
            return await self._fees.get_taker_rate(condition_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("CrossGuard | fee lookup failed for %s: %s", condition_id[:12], exc)
            return None

    def _implications(self) -> list[Implication]:
        """
        The reader's export, re-read only when the file changes. Covering the
        whole one-week window it holds thousands of pairs, and parsing that every
        15 seconds would be most of the guard's work.
        """
        try:
            st = os.stat(self._imp_path)
        except OSError:
            self._imp_cache = (None, [])
            return []
        sig = (st.st_mtime_ns, st.st_size)
        if self._imp_cache[0] != sig:
            self._imp_cache = (sig, load_implications(self._imp_path))
            self._forget_departed({imp.key for imp in self._imp_cache[1]})
        return self._imp_cache[1]

    def _forget_departed(self, present: "set[tuple[str, str]]") -> None:
        """
        Drop the schedule of pairs that left the file. The reader now exports
        only quotable books, so pairs come and go with their spreads, and each
        departure used to leave four entries behind for the life of the process.
        A pair that returns is simply priced as new. Cooldowns and stuck pairs
        are safety state and are kept whatever the file says.
        """
        for table in (self._last_stop, self._last_edge, self._next_check,
                      self._priced_at, self.last_reason):
            for key in [k for k in table if k not in present]:
                del table[key]

    def _spare_priority(self, imp: Implication) -> float:
        """Which deferred pair gets a leftover read: whichever has waited longest,
        a book last refused as too wide counted as if it had been priced
        CROSS_WIDE_BOOK_PENALTY_S later than it was."""
        key = imp.key
        at = self._priced_at.get(key, 0.0)
        if self._last_stop.get(key) == "book_too_wide":
            at += CROSS_WIDE_BOOK_PENALTY_S
        return at

    def _priority(self, imp: Implication) -> tuple:
        """Where a book read is worth most: near the threshold, then never priced, then longest waiting."""
        key = imp.key
        if key not in self._last_stop:
            return (1, 0.0)
        edge = self._last_edge.get(key)
        if edge is not None and self._min_edge - edge <= CROSS_NEAR_BAND:
            return (0, -edge)
        return (2, self._next_check.get(key, 0.0))

    async def _scan_entries(self, now: float) -> None:
        imps = self._implications()
        if self._first_scan is None:
            self._first_scan = now
        self._fee_lookups_left = CROSS_MAX_FEE_LOOKUPS_PER_POLL
        judged: "list[tuple[Implication, str, float | None]]" = []
        due: list[Implication] = []
        spare: list[Implication] = []
        for imp in imps:
            stop = self._cheap_stop(imp, now)
            if stop is None:
                due.append(imp)
            elif stop == "deferred":
                spare.append(imp)
            else:
                judged.append((imp, stop, None))
        due.sort(key=self._priority)
        spare.sort(key=self._spare_priority)
        watched = due + spare                       # every pair a book read could price
        # Due pairs first, then what the deferred ones can have of the budget
        # rather than leaving it unspent. A pair 30 points from the threshold is
        # deferred for ten minutes, and a goal moves a match's over/under by
        # that much at once.
        order = due + spare
        single = order[:max(1, self._max_reads // 2)]
        batch = single
        if self._book_batches > 0 and hasattr(self._client, "get_orderbooks"):
            batch = _take_batch(order, self._book_batches * CROSS_BOOK_BATCH)
            tokens = [t for imp in batch for t in (imp.narrow_no_token, imp.broad_yes_token)]
            try:
                self._poll_books = await self._client.get_orderbooks(
                    tokens, chunk=CROSS_BOOK_BATCH)
            except Exception as exc:  # noqa: BLE001 — the single reads still work
                logger.warning("CrossGuard | batched book read failed, reading %d pair(s) "
                               "one at a time this poll: %s", len(single), exc)
                self._poll_books = None
                batch = single
        try:
            for imp in batch:
                stop, edge = await self._priced(imp, now)
                self._priced_at[imp.key] = now
                judged.append((imp, stop, edge))
        finally:
            self._poll_books = None
        taken = {id(imp) for imp in batch}
        judged.extend((imp, "queued", None) for imp in due if id(imp) not in taken)
        judged.extend((imp, "deferred", None) for imp in spare if id(imp) not in taken)
        stalest = max((now - self._priced_at.get(imp.key, self._first_scan)
                       for imp in watched), default=None)

        stops: dict[str, int] = {}
        best: "tuple[float, str] | None" = None
        waiting = 0
        for imp, stop, edge in judged:
            if stop in ("deferred", "queued"):
                # Report what it was when last priced; a pair 30 points away is
                # not news every 15 seconds, but it is still the fact about it.
                waiting += 1
                stop = self._last_stop.get(imp.key, "not yet priced")
                edge = self._last_edge.get(imp.key)
            else:
                self._last_stop[imp.key] = stop
                self._last_edge[imp.key] = edge
                wait = self._recheck_delay(stop, edge)
                if wait > 0.0:
                    self._next_check[imp.key] = now + wait
            stops[stop] = stops.get(stop, 0) + 1
            if edge is not None and (best is None or edge > best[0]):
                best = (edge, imp.narrow_title)
        self._last_pass = (now, len(imps), stops, best, waiting, stalest)

    def _recheck_delay(self, stop: str, edge: "float | None") -> float:
        """
        How long this pair may be left alone. Distance from the threshold is
        the only thing that matters: a pair 30 points away cannot cross it in
        one poll, and pricing it again costs two book reads that a near pair
        needs more.
        """
        if stop in ("held", "stuck", "cooldown", "past_end", "outside_window",
                    "end_unknown", "different_statistic", "too_close_to_end",
                    "not_yes_no", "outcomes_unknown", "resting"):
            return 0.0                      # judged without a book read anyway
        if edge is None:
            return CROSS_RECHECK_MID_S      # no ask, no bid, or no book at all
        gap = self._min_edge - edge
        if gap <= CROSS_NEAR_BAND:
            return 0.0                      # close: look every poll
        if gap <= CROSS_MID_BAND:
            return CROSS_RECHECK_MID_S
        return CROSS_RECHECK_FAR_S

    async def _consider(self, imp: Implication, now: float) -> "tuple[str, float | None]":
        """
        One implication through every gate. Returns where it stopped and, once
        both asks are priced on a genuine implication, the edge they quote.
        """
        stop = self._cheap_stop(imp, now)
        if stop is not None:
            return stop, None
        return await self._priced(imp, now)

    def _cheap_stop(self, imp: Implication, now: float) -> "str | None":
        """The gates that need no book read; None when the pair is due for pricing."""
        key = imp.key
        if self._book.has_open(*key):
            return "held"
        if key in self._attempts:
            return "resting"
        if key in self._stuck:
            return "stuck"
        if self._cooldown.get(key, 0.0) > now:
            return "cooldown"
        ends = imp.resolves_ts
        # Cheap window check before spending two book reads on it.
        if ends is None:
            return "end_unknown"
        if ends <= now:
            return "past_end"
        if (ends - now) / 86_400.0 > self._max_lockup:
            return "outside_window"
        if imp.narrow_outcomes is None or imp.broad_outcomes is None:
            return "outcomes_unknown"
        if self._yes_no_only and not (is_yes_no(imp.narrow_outcomes)
                                      and is_yes_no(imp.broad_outcomes)):
            return "not_yes_no"
        if self._next_check.get(key, 0.0) > now:
            return "deferred"
        return None

    async def _priced(self, imp: Implication, now: float) -> "tuple[str, float | None]":
        """The gates that need both books."""
        key = imp.key
        self.stats["evaluated"] += 1
        # A book can be gone — the market closed, or the token was never
        # tradeable. That is this pair's answer, not the pass's: letting it
        # raise aborted the whole scan, and on 2026-09-12 the guard went 12
        # minutes without completing one, exits included.
        try:
            if self._poll_books is not None:
                no_book = self._poll_books.get(imp.narrow_no_token)
                yes_book = self._poll_books.get(imp.broad_yes_token)
                if no_book is None or yes_book is None:
                    # The batch leaves out a token with no book; a single read
                    # would have raised for it.
                    raise LookupError("no book for this token in the batch")
            else:
                no_book = await self._client.get_orderbook(imp.narrow_no_token)
                yes_book = await self._client.get_orderbook(imp.broad_yes_token)
        except Exception as exc:  # noqa: BLE001
            logger.debug("CrossGuard | no book for %s: %s", imp.narrow_title[:40], exc)
            self.last_reason[key] = f"orderbook unavailable: {exc}"
            return "no_book", None
        rate_narrow = await self._pair_rate(imp.narrow, imp.narrow_fee_rate)
        rate_broad = await self._pair_rate(imp.broad, imp.broad_fee_rate)
        opp, reason = evaluate(
            imp, no_book, yes_book, now=now,
            rate_narrow=rate_narrow, rate_broad=rate_broad,
            max_lockup_days=self._max_lockup, min_edge=self._min_edge,
            max_usdc=self._max_usdc, min_shares=self._min_shares,
            min_time_to_end_s=self._min_time_to_end,
            yes_no_only=self._yes_no_only,
        )
        self.last_reason[key] = reason
        stop, edge = stop_key(reason), None
        if stop in _PRICED_STOPS:
            quote = quoted_entry(no_book, yes_book, rate_narrow, rate_broad)
            edge = None if quote is None else 1.0 - quote[2]
        if opp is None:
            # The taker path pays the ask plus a fee on both legs. Our own bid
            # pays neither, and that is where these pairs can become tradeable.
            if self._maker_enabled and stop in ("edge_below_min", "thin_touch", "no_ask",
                                                "too_close_to_end"):
                return await self._maker_attempt(imp, no_book, yes_book, now, stop, edge)
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

    def _leaves_taker_room(self, committed: float) -> bool:
        """Whether resting `committed` still leaves the reserved taker slot(s) free."""
        reserve = max(0, self._taker_reserve)
        if reserve == 0:
            return True
        headroom = getattr(self._breaker, "cross_headroom", None)
        if not callable(headroom):
            return True
        slots, usdc = headroom()
        return slots > reserve and usdc - committed >= reserve * self._max_usdc - 1e-9

    async def _maker_attempt(self, imp: Implication, no_book: dict, yes_book: dict,
                             now: float, taker_stop: str,
                             taker_edge: "float | None") -> "tuple[str, float | None]":
        """What resting our own bids would cost, and — when it pays — resting them."""
        key = imp.key
        opp, reason = evaluate_maker(
            imp, no_book, yes_book, now=now, max_lockup_days=self._max_lockup,
            min_edge=self._maker_min_edge, max_usdc=self._max_usdc,
            min_shares=self._min_shares, min_time_to_end_s=CROSS_MAKER_MIN_TIME_TO_END_S,
            yes_no_only=self._yes_no_only)
        self.last_reason[key] = reason
        # The same discipline the taker path keeps: a price only counts once the
        # book it came from has passed the gates. Pricing first put the +0.9600
        # of an 0.01/0.99 book into the heartbeat's "best edge" on 2026-09-17,
        # on a pair the wide-book gate had already refused.
        edge = taker_edge
        if stop_key(reason) in _PRICED_STOPS:
            quote = maker_entry(imp, no_book, yes_book)
            if quote is not None:
                maker_edge = 1.0 - quote[2]
                edge = maker_edge if edge is None else max(edge, maker_edge)
        if opp is None:
            return (stop_key(reason) if reason != "ok" else taker_stop), edge
        if self._enabled and not self._leaves_taker_room(opp.committed):
            self.last_reason[key] = "the last cross slot is kept for a taker arbitrage"
            return "taker_reserve", edge
        if not self._breaker.check_cross(opp.committed):
            self.last_reason[key] = "blocked by the breaker"
            return "breaker", edge
        if opp.edge_per_pair >= self._suspicious_edge:
            seen = self._spike.get(key, 0.0)
            self._spike[key] = now
            if now - seen > CROSS_SPIKE_CONFIRM_S:
                logger.info(
                    "CrossGuard | maker %+.4f/pair on %s is above %.2f — waiting for a "
                    "second poll to confirm it is not a stale book",
                    opp.edge_per_pair, imp.narrow_title[:40], self._suspicious_edge)
                return "unconfirmed_spike", edge
        if not self._enabled:
            self.stats["would_rest"] += 1
            self._cooldown[key] = now + CROSS_ENTRY_COOLDOWN_S
            logger.info(
                "CrossGuard | WOULD REST (disabled) %s ⊆ %s | %.2f pairs, bids "
                "%.4f + %.4f = %.4f → edge %+.4f/pair · %.2f USDC · %.1fd "
                "(taker was %s)",
                imp.narrow_title[:40], imp.broad_title[:40], opp.shares, opp.no_price,
                opp.yes_price, opp.entry_per_pair, opp.edge_per_pair, opp.committed,
                opp.lockup_days, taker_stop)
            self._record_paper_rest(opp, no_book, yes_book, now)
            return "maker_would_rest", edge
        rested = await self._rest_maker(opp, now)
        return ("maker_resting" if rested else "maker_failed"), edge

    def _record_paper_rest(self, opp: "MakerOpportunity", no_book: dict, yes_book: dict,
                           now: float) -> None:
        """
        What would have rested, and the books it rested in. Whether a disabled
        guard's rests would ever fill is the question no log line answered: on
        2026-09-18, 35 of 48 pairs clearing every maker gate had not traded once
        in seven days on either leg.
        """
        imp = opp.imp

        def touch(book, side):
            lvl = best_level(book, side)
            return None if lvl is None else [lvl.price, lvl.size]

        # The one-leg version: rest one side only and, the moment it fills, buy
        # the other at its ask. One fill instead of two, and of the 289 pairs
        # signalling on 2026-09-18, 32 cleared +2% that way — always by resting
        # NO on narrow, i.e. offering the longshot that retail buys.
        no_ask, yes_ask = best_level(no_book, "asks"), best_level(yes_book, "asks")
        one_no = (None if yes_ask is None else
                  1.0 - opp.no_price - buy_cost(yes_ask.price, imp.broad_fee_rate))
        one_yes = (None if no_ask is None else
                   1.0 - buy_cost(no_ask.price, imp.narrow_fee_rate) - opp.yes_price)
        row = {"ts": now, "narrow": imp.narrow, "broad": imp.broad,
               "narrow_title": imp.narrow_title, "broad_title": imp.broad_title,
               "narrow_yes_token": imp.narrow_yes_token, "narrow_no_token": imp.narrow_no_token,
               "broad_yes_token": imp.broad_yes_token, "broad_no_token": imp.broad_no_token,
               "no_price": opp.no_price, "yes_price": opp.yes_price, "shares": opp.shares,
               "entry": opp.entry_per_pair, "edge": opp.edge_per_pair,
               "resolves_ts": imp.resolves_ts,
               "one_leg_no_edge": one_no, "one_leg_yes_edge": one_yes,
               "broad_fee_rate": imp.broad_fee_rate, "narrow_fee_rate": imp.narrow_fee_rate,
               "no_bid": touch(no_book, "bids"), "no_ask": touch(no_book, "asks"),
               "yes_bid": touch(yes_book, "bids"), "yes_ask": touch(yes_book, "asks")}
        try:
            with open(self._paper_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError as exc:
            logger.warning("CrossGuard | cannot append %s: %s", self._paper_path, exc)

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

    async def _sell(self, token: str, shares: float, cost_per_share: float,
                    rate: "float | None" = None) -> "float | None":
        """Sell at the bid. Returns realised P&L after the taker fee, or None
        when it did not fill. `rate` is the market's taker rate."""
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
        # taking_amount is what the book paid, before the fee the wallet pays.
        if sold > 0.0 and proceeds > 0.0:
            proceeds -= sold * taker_fee_per_share(proceeds / sold, rate)
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
            pnl = await self._sell(t1, got1, c1, opp.rate_narrow
                                   if t1 == imp.narrow_no_token else opp.rate_broad)
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
        pnl1 = await self._sell(pos.narrow_no_token, pos.size, pos.narrow_no_paid, rn)
        if pnl1 is None:
            return                                   # nothing sold — try again later
        pnl2 = await self._sell(pos.broad_yes_token, pos.size, pos.broad_yes_paid, rb)
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
