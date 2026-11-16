"""
strategy/reward_quotes.py
─────────────────────────
What a minimum two-sided quote earns from Polymarket's liquidity rewards, and
which markets can be quoted with very little at risk.

The programme (docs: programs/liquidity-rewards): each market pays a daily amount
to orders resting within `max_spread` cents of the midpoint, sampled every minute.
An order scores

    size × ((v − s) / v)²        v = max spread, s = its distance from the midpoint

a maker's score is the smaller of its two sides, and with the midpoint under 0.10
or over 0.90 only a two-sided quote scores at all. The day's pool is split in
proportion to scores. Orders smaller than the market's minimum size do not count.

The quote planned here is one token's bid and ask: a bid for the cheaper token, and
an ask from shares of that same token bought for the purpose. Both sides score, and
all the quote can lose is what the shares and the bid cost. The other way to quote
two-sided, a bid on each token, ties up a dollar a share.

Measured on 2026-11-15: across 5,764 markets such a quote scored a median 0.17 USDC
a day at a median cost of 7 USDC, and the calmest of 118 reward farmers (15% losing
days or fewer) quote exactly this size on hundreds to thousands of markets.

Nothing here does I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

_EPS = 1e-9


@dataclass(frozen=True)
class RewardMarket:
    """One row of GET /rewards/markets/multi that pays something today."""
    condition_id: str
    question:     str
    rate:         float                              # USDC a day
    max_spread:   float                              # cents
    min_size:     float                              # shares
    tokens:       tuple[tuple[str, str, float], ...]  # (token id, outcome, price)
    volume_24h:   float
    end_ts:       "float | None"
    event_id:     str

    def cheaper(self) -> tuple[str, str, float]:
        """The token that costs less to hold, as (token id, outcome, price)."""
        return min(self.tokens, key=lambda t: t[2])


@dataclass(frozen=True)
class Plan:
    """A quote that could be placed on one market right now."""
    market:     RewardMarket
    token_id:   str
    outcome:    str
    bid:        float      # where the bid rests
    ask:        float      # where the ask rests
    buy_shares: float      # shares bought for the ask
    bid_shares: float
    cost:       float      # the shares and the bid: the most the quote can lose
    ours:       float      # our score
    theirs:     float      # what the book already scores
    per_day:    float      # rate × ours / (ours + theirs), at this snapshot

    @property
    def share(self) -> float:
        total = self.ours + self.theirs
        return self.ours / total if total > 0.0 else 0.0


def _end_ts(raw: object) -> "float | None":
    if not raw:
        return None
    text = str(raw).strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    elif text.endswith("+00"):
        text += ":00"
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


def parse_reward_market(row: dict) -> "RewardMarket | None":
    """A listing row as a RewardMarket, or None if it pays nothing or is malformed."""
    try:
        rate = sum(float(c.get("rate_per_day") or 0.0) for c in row.get("rewards_config") or [])
        tokens = tuple(
            (str(t["token_id"]), str(t.get("outcome") or ""), float(t.get("price") or 0.0))
            for t in row.get("tokens") or []
        )
        market = RewardMarket(
            condition_id=str(row["condition_id"]),
            question=str(row.get("question") or ""),
            rate=rate,
            max_spread=float(row.get("rewards_max_spread") or 0.0),
            min_size=float(row.get("rewards_min_size") or 0.0),
            tokens=tokens,
            volume_24h=float(row.get("volume_24hr") or 0.0),
            end_ts=_end_ts(row.get("end_date")),
            event_id=str(row.get("event_id") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if market.rate <= 0.0 or market.max_spread <= 0.0 or market.min_size <= 0.0:
        return None
    if len(market.tokens) != 2 or not all(t[0] for t in market.tokens):
        return None
    return market


# ── scoring ───────────────────────────────────────────────────────────────────

def order_score(v: float, s: float) -> float:
    """S(v, s): what one share scores `s` cents from the midpoint, 0 outside the band."""
    if v <= 0.0 or s < -_EPS or s >= v:
        return 0.0
    return ((v - max(s, 0.0)) / v) ** 2


def q_min(one: float, two: float, mid: float) -> float:
    """A maker's score from its two sides. Single-sided scores a third, mid-range only."""
    if 0.10 <= mid <= 0.90:
        return max(min(one, two), max(one, two) / 3.0)
    return min(one, two)


def levels(book: "dict | None", side: str) -> list[tuple[float, float]]:
    """(price, size) rows of one side of a book, bad rows left out."""
    out: list[tuple[float, float]] = []
    for row in (book or {}).get(side) or []:
        try:
            price, size = float(row["price"]), float(row["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if price > 0.0 and size > 0.0:
            out.append((price, size))
    return out


def best_other(book: "dict | None", side: str, own_price: "float | None" = None,
               own_size: float = 0.0, min_level: float = 0.0) -> "tuple[float, float] | None":
    """
    The best price on a side where others rest at least `min_level` shares, and
    how many they rest there.

    The book includes our own order: joining "the best bid" when the best bid is
    us would mean following ourselves, so our size is taken out first. A level
    under `min_level` is dust. The programme leaves orders under a market's
    minimum size out of its midpoint, and a single share offered far inside the
    spread must not be what a whole lot is priced off.
    """
    rows: dict[float, float] = {}
    for price, size in levels(book, side):
        key = round(price, 6)
        rows[key] = rows.get(key, 0.0) + size
    if own_price is not None:
        key = round(own_price, 6)
        if key in rows:
            rows[key] -= own_size
    rows = {p: s for p, s in rows.items() if s > 0.005 and s + _EPS >= min_level}
    if not rows:
        return None
    price = min(rows) if side == "asks" else max(rows)
    return price, rows[price]


def depth_to(book: "dict | None", side: str, price: float) -> float:
    """Shares resting on a side at `price` or better."""
    if side == "asks":
        return sum(size for p, size in levels(book, side) if p <= price + _EPS)
    return sum(size for p, size in levels(book, side) if p >= price - _EPS)


def book_scores(book: "dict | None", mid: float, v: float) -> tuple[float, float]:
    """What the bids and the asks in a book score against `mid`."""
    bids = sum(size * order_score(v, (mid - price) * 100.0) for price, size in levels(book, "bids"))
    asks = sum(size * order_score(v, (price - mid) * 100.0) for price, size in levels(book, "asks"))
    return bids, asks


# ── planning ──────────────────────────────────────────────────────────────────

def shares_to_buy(ask: float, min_size: float, min_buy_usdc: float) -> float:
    """
    Shares to buy for the ask: the market's minimum size, or more where that many
    would cost less than the exchange accepts for a marketable order — and then
    the next size up whose cost is a whole number of cents.

    A marketable BUY is sent as an amount to spend, in cents. Where amount / price
    is not a whole number of hundredths of a share, the order's rounded price
    lands a hair under the ask and nothing matches: on 2026-11-16 the three buys
    that came out exact filled (20.40 @ 0.05, 20 @ 0.06, 60 @ 0.017) and all seven
    that did not were killed (35.18 @ 0.029 is 1.03 / 0.029 = 35.517...).
    """
    if ask <= 0.0:
        return 0.0
    need = max(min_size, min_buy_usdc / ask)
    k = round(ask * 10_000)                                # the price in hundredths of a cent
    if k <= 0:
        return 0.0
    step = 10_000 // math.gcd(k, 10_000)                   # in hundredths of a share
    return math.ceil(need * 100.0 / step - 1e-9) * step / 100.0


def plan_quote(
    market: RewardMarket,
    book: "dict | None",
    *,
    max_cost: float,
    min_score: float = 0.5,
    max_spread_cost: float = 0.25,
    min_buy_usdc: float = 1.02,
) -> "Plan | None":
    """
    The quote to place on `market`'s cheaper token given that token's book, or None
    where one cannot be placed within `max_cost`.

    Both prices join the book, at the best bid and the best ask that hold at least
    the market's minimum size. The shares for the ask are bought from the asks up
    to that price, so they must hold enough.

    The shares are bought at the ask and are worth the bid the moment they are
    held, so the spread on them is lost on the way in: `max_spread_cost` caps it.
    On 2026-11-16 the first list chosen without it included a market quoted
    0.014 / 0.083, and twelve such entries would have marked the pilot down by
    7.6 USDC of its 8 USDC limit before a single order rested.
    """
    token_id, outcome, _ = market.cheaper()
    bid_lvl = best_other(book, "bids", min_level=market.min_size)
    ask_lvl = best_other(book, "asks", min_level=market.min_size)
    if bid_lvl is None or ask_lvl is None:
        return None
    bid, ask = bid_lvl[0], ask_lvl[0]
    if not 0.0 < bid < ask < 1.0:
        return None
    mid = (bid + ask) / 2.0
    half = (ask - bid) * 50.0                              # each side's distance, in cents
    score = order_score(market.max_spread, half)
    if score < min_score:
        return None
    buy = shares_to_buy(ask, market.min_size, min_buy_usdc)
    if depth_to(book, "asks", ask) + _EPS < buy:
        return None                                        # the asks cannot fill the buy
    if buy * (ask - bid) > max_spread_cost + _EPS:
        return None
    spend = buy * ask
    # Bid for as many shares as the ask offers, so neither side caps the score,
    # as far as the cost allows.
    bid_shares = min(buy, math.floor((max_cost - spend) / bid + 1e-6))
    if bid_shares + _EPS < market.min_size:
        return None
    cost = spend + bid_shares * bid
    ours = q_min(bid_shares * score, buy * score, mid)
    theirs = q_min(*book_scores(book, mid, market.max_spread), mid)
    total = ours + theirs
    per_day = market.rate * ours / total if total > 0.0 else 0.0
    return Plan(market=market, token_id=token_id, outcome=outcome, bid=bid, ask=ask,
                buy_shares=buy, bid_shares=float(bid_shares), cost=cost, ours=ours,
                theirs=theirs, per_day=per_day)


def eligible(
    market: RewardMarket,
    now: float,
    *,
    max_price: float,
    min_days: float,
    max_volume: float,
    min_rate: float,
    max_min_size: float,
    excluded: "Iterable[str]" = (),
) -> bool:
    """Whether a market is worth reading the book of."""
    if market.condition_id in excluded:
        return False
    if market.rate < min_rate or market.min_size > max_min_size:
        return False
    if market.volume_24h > max_volume:
        return False
    # Far from its end: nothing here should be decided while the quote rests, and
    # the cross path only trades markets inside its seven-day window.
    if market.end_ts is None or market.end_ts < now + min_days * 86_400.0:
        return False
    return 0.0 < market.cheaper()[2] <= max_price


def select(plans: "Iterable[Plan]", *, budget: float, max_markets: int,
           max_share: float = 0.5) -> list[Plan]:
    """
    The plans to place: most reward for the cost first, one market an event,
    inside the budget.

    A snapshot of a nearly empty book promises most of a pool. `max_share` caps
    what a plan is ranked on, so a thin moment does not decide the list.
    """
    def worth(p: Plan) -> float:
        return p.market.rate * min(p.share, max_share) / p.cost if p.cost > 0.0 else 0.0

    chosen: list[Plan] = []
    events: set[str] = set()
    spent = 0.0
    for p in sorted(plans, key=worth, reverse=True):
        if len(chosen) >= max_markets:
            break
        if worth(p) <= 0.0 or spent + p.cost > budget + _EPS:
            continue
        key = p.market.event_id or p.market.condition_id
        if key in events:
            continue
        events.add(key)
        chosen.append(p)
        spent += p.cost
    return chosen


# ── while a quote rests ───────────────────────────────────────────────────────

def bid_shares_within(max_cost: float, held_cost: float, bid: float, want: float,
                      min_size: float) -> float:
    """
    Shares to bid for: `want`, cut to what keeps the quote inside `max_cost` with
    `held_cost` already paid for the shares held. 0 when the minimum size no longer fits.
    """
    if bid <= 0.0:
        return 0.0
    shares = min(want, math.floor((max_cost - held_cost) / bid + 1e-6))
    return float(shares) if shares + _EPS >= min_size else 0.0


def scoring_now(bid: "float | None", bid_shares: float, ask: "float | None", ask_shares: float,
                market_min: float, v: float) -> bool:
    """Whether a resting bid and ask form a quote the programme scores: both there,
    both at least the minimum size, both inside the band around their midpoint."""
    if bid is None or ask is None or not 0.0 < bid < ask:
        return False
    if bid_shares + _EPS < market_min or ask_shares + _EPS < market_min:
        return False
    return order_score(v, (ask - bid) * 50.0) > 0.0
