"""
strategy/cross_exit.py
──────────────────────
When to leave a cross-market arbitrage before it resolves.

The position
────────────
For an implication narrow ⊆ broad priced the wrong way round, the arbitrage is

    buy NO on narrow  +  buy YES on broad

which pays at least 1.00 in every outcome, and 2.00 in the middle one:

    narrow YES (so broad YES)      NO_narrow 0  +  YES_broad 1   =  1
    narrow NO,  broad YES          NO_narrow 1  +  YES_broad 1   =  2
    broad NO    (so narrow NO)     NO_narrow 1  +  YES_broad 0   =  1

So holding it is a guaranteed 1.00 plus a lottery ticket on the middle outcome.
The catch is that it only pays at resolution, and the live reader's violations
sit on pairs that resolve 112 days out.

The exit
────────
Buying the two missing sides — YES on narrow, NO on broad — completes a full set
on each market, and a full set merges back to 1.00 immediately. Two sets, 2.00,
now. Per pair:

    profit = 2 − entry − completion
           = guaranteed_edge + (1 − completion)

The second term is the whole story. YES_narrow and NO_broad are two MUTUALLY
EXCLUSIVE outcomes — narrow happening and broad not happening cannot coincide —
so in any sanely priced market they cost less than 1.00 together, by exactly the
market's price for the middle outcome. Completing sells the lottery ticket to the
market at that price.

While the violation persists the complementary pair is overpriced by the same
amount the entry was underpriced — it is one mispricing seen from both sides —
so completing then returns nothing. The moment the market corrects, it returns
the guaranteed edge plus the middle outcome's price, immediately. That is the
point of watching: the guaranteed part is identical, it just arrives in weeks
instead of months.

Selling the two legs at the bid is the other way out, and it is nearly the same
trade — 1 − ask(YES_narrow) ≈ bid(NO_narrow) — so both are priced and the better
one wins. Completion pays two taker fees and two on-chain merges; a sale pays two
taker fees.

The rule
────────
Exit when the exit captures at least CROSS_EXIT_MIN_CAPTURE of the guaranteed
edge. At the default 1.0 that is "never leave for less than resolution already
guarantees", which makes the monitor impossible to lose to: it either exits with
everything holding would have given plus the ticket's price, or it holds.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

# Taker rate used to price the exit legs. The fee is rate x p x (1-p) per share —
# the same formula the main strategy uses — so it is largest at 0.5 and vanishes
# at the extremes. 0.05 is Polymarket's general-category rate; sports trade at
# 0.03 and crypto at 0.07.
CROSS_TAKER_RATE: float = float(os.environ.get("CROSS_TAKER_RATE", 0.05))

# Gas for one mergePositions on Polygon, in USDC. Completion needs two. Small,
# but charged per transaction rather than per share, so it matters on a small
# position and not at all on a large one.
CROSS_MERGE_GAS_USDC: float = float(os.environ.get("CROSS_MERGE_GAS_USDC", 0.02))

# Fraction of the guaranteed edge an exit must capture. 1.0 never exits for less
# than resolution guarantees. Lower it to trade some guaranteed edge for capital
# back sooner — the right number depends on what that capital would earn next.
CROSS_EXIT_MIN_CAPTURE: float = float(
    os.environ.get("CROSS_EXIT_MIN_CAPTURE", 1.0)
)

ROUTE_COMPLETE = "complete"
ROUTE_SELL = "sell"


def taker_fee_per_share(price: float, rate: "float | None" = None) -> float:
    """Polymarket's taker fee in USDC for one share at `price`."""
    r = CROSS_TAKER_RATE if rate is None else rate
    if not 0.0 < price < 1.0:
        return 0.0
    return r * price * (1.0 - price)


def buy_cost(ask: float, rate: "float | None" = None) -> float:
    """What one share actually costs to buy at the ask, fee included."""
    return ask + taker_fee_per_share(ask, rate)


def sell_proceeds(bid: float, rate: "float | None" = None) -> float:
    """What one share actually returns when sold at the bid, fee deducted."""
    return bid - taker_fee_per_share(bid, rate)


# ═══════════════════════════════════════════════════════════════════════════════
# The position
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossPosition:
    """One held arbitrage: NO on narrow plus YES on broad, `size` pairs of it."""
    narrow:           str
    broad:            str
    narrow_title:     str
    broad_title:      str
    narrow_yes_token: str
    narrow_no_token:  str
    broad_yes_token:  str
    broad_no_token:   str
    size:             float
    # Per share, fee included. Priced at the ask actually paid, NOT at 1 − the
    # YES ask: the NO ask is about 1 − YES bid, and pricing it off the YES ask
    # understates the entry by the narrow market's spread — a median of 100 bps
    # on the live book, which halves a 200 bps edge.
    narrow_no_paid:   float
    broad_yes_paid:   float
    opened_ts:        float
    resolves_ts:      "float | None" = None
    paper:            bool = True
    status:           str = "open"          # open | exited | resolved
    exit_route:       str = ""
    exit_profit:      float = 0.0            # per pair, once closed
    closed_ts:        "float | None" = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.narrow, self.broad)

    @property
    def entry_cost(self) -> float:
        """USDC per pair."""
        return self.narrow_no_paid + self.broad_yes_paid

    @property
    def guaranteed_edge(self) -> float:
        """USDC per pair that resolution pays for certain."""
        return 1.0 - self.entry_cost

    def days_held(self, now: "float | None" = None) -> float:
        end = self.closed_ts if self.closed_ts else (now or time.time())
        return max(0.0, (end - self.opened_ts) / 86_400.0)

    def days_to_resolution(self, now: "float | None" = None) -> "float | None":
        if self.resolves_ts is None:
            return None
        return max(0.0, (self.resolves_ts - (now or time.time())) / 86_400.0)


# ═══════════════════════════════════════════════════════════════════════════════
# The two ways out
# ═══════════════════════════════════════════════════════════════════════════════

def exit_by_completion(
    pos:            CrossPosition,
    narrow_yes_ask: "float | None",
    broad_no_ask:   "float | None",
    *,
    rate: "float | None" = None,
    gas:  "float | None" = None,
) -> "float | None":
    """
    Profit per pair from buying the missing sides and merging both sets.

    2.00 comes back from the two merges. Out go the entry, the two completing
    legs at their asks with fees, and two merge transactions spread over the
    position's size.
    """
    if narrow_yes_ask is None or broad_no_ask is None:
        return None
    g = CROSS_MERGE_GAS_USDC if gas is None else gas
    completion = buy_cost(narrow_yes_ask, rate) + buy_cost(broad_no_ask, rate)
    gas_per_pair = 2.0 * g / max(pos.size, 1e-9)
    return 2.0 - pos.entry_cost - completion - gas_per_pair


def exit_by_sale(
    pos:           CrossPosition,
    narrow_no_bid: "float | None",
    broad_yes_bid: "float | None",
    *,
    rate: "float | None" = None,
) -> "float | None":
    """Profit per pair from selling both legs at the bid."""
    if narrow_no_bid is None or broad_yes_bid is None:
        return None
    got = sell_proceeds(narrow_no_bid, rate) + sell_proceeds(broad_yes_bid, rate)
    return got - pos.entry_cost


def resolution_profit(pos: CrossPosition, narrow_yes: bool, broad_yes: bool) -> float:
    """
    Profit per pair once both markets have resolved.

    NO on narrow pays when narrow resolved NO; YES on broad pays when broad
    resolved YES. The implication rules out narrow YES with broad NO, so the
    payout is 1 or 2 — never 0 — whenever the mapping was right. A 0 here means
    the implication was false, and that is the one loss this strategy can take.
    """
    payout = (0.0 if narrow_yes else 1.0) + (1.0 if broad_yes else 0.0)
    return payout - pos.entry_cost


@dataclass(frozen=True)
class ExitQuote:
    route:      str               # ROUTE_COMPLETE | ROUTE_SELL | "" (no quote)
    profit:     float             # per pair, for the chosen route
    completion: "float | None"    # per pair, if priceable
    sale:       "float | None"    # per pair, if priceable


def best_exit(
    pos: CrossPosition,
    *,
    narrow_yes_ask: "float | None",
    broad_no_ask:   "float | None",
    narrow_no_bid:  "float | None",
    broad_yes_bid:  "float | None",
    rate: "float | None" = None,
    gas:  "float | None" = None,
) -> ExitQuote:
    """Price both routes and keep the better one."""
    comp = exit_by_completion(pos, narrow_yes_ask, broad_no_ask, rate=rate, gas=gas)
    sale = exit_by_sale(pos, narrow_no_bid, broad_yes_bid, rate=rate)
    options = [(p, r) for p, r in ((comp, ROUTE_COMPLETE), (sale, ROUTE_SELL))
               if p is not None]
    if not options:
        return ExitQuote("", 0.0, comp, sale)
    profit, route = max(options)
    return ExitQuote(route, profit, comp, sale)


def should_exit(
    pos:   CrossPosition,
    quote: ExitQuote,
    *,
    min_capture: "float | None" = None,
) -> bool:
    """
    Leave when the exit captures enough of what resolution guarantees.

    At min_capture 1.0 this can never do worse than holding: it only fires when
    the exit returns at least the guaranteed edge, and for completion that is the
    same as saying the complementary pair has stopped being overpriced — the
    market has corrected.
    """
    if not quote.route:
        return False
    cap = CROSS_EXIT_MIN_CAPTURE if min_capture is None else min_capture
    edge = pos.guaranteed_edge
    if edge <= 0.0:
        # The entry was never an arbitrage at the prices actually paid. Take any
        # exit that at least breaks even rather than hold a bet.
        return quote.profit >= 0.0
    return quote.profit >= cap * edge


def annualised(profit: float, capital: float, days: float) -> "float | None":
    """Return on capital per year. None when the holding period is zero."""
    if capital <= 0.0 or days <= 0.0:
        return None
    return (profit / capital) * (365.0 / days)


# ═══════════════════════════════════════════════════════════════════════════════
# The book of positions, kept across restarts
# ═══════════════════════════════════════════════════════════════════════════════

class PositionBook:
    """
    Every cross-market position, open and closed, persisted to one JSON file.

    Written atomically (temp file then rename) so a crash mid-write cannot leave
    a truncated file that loses every open position — the same way the reader
    already keeps its announced implications.
    """

    def __init__(self, path: "str | Path") -> None:
        self._path = Path(path)
        self._positions: dict[tuple[str, str], CrossPosition] = {}
        self._closed: list[CrossPosition] = []

    # ── persistence ───────────────────────────────────────────────────────────

    def load(self) -> "PositionBook":
        if not self._path.exists():
            return self
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.error("cross positions | cannot read %s: %s — starting empty",
                         self._path, exc)
            return self
        names = {f.name for f in fields(CrossPosition)}
        for row in raw.get("open", []):
            pos = CrossPosition(**{k: v for k, v in row.items() if k in names})
            self._positions[pos.key] = pos
        for row in raw.get("closed", []):
            self._closed.append(
                CrossPosition(**{k: v for k, v in row.items() if k in names}))
        return self

    def save(self) -> None:
        payload = {
            "open": [asdict(p) for p in self._positions.values()],
            "closed": [asdict(p) for p in self._closed],
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(self._path)
        except OSError as exc:
            logger.error("cross positions | cannot write %s: %s", self._path, exc)

    # ── positions ─────────────────────────────────────────────────────────────

    def has_open(self, narrow: str, broad: str) -> bool:
        return (narrow, broad) in self._positions

    def open(self, pos: CrossPosition) -> bool:
        """Register a position. Refuses a second one on the same pair."""
        if pos.key in self._positions:
            return False
        self._positions[pos.key] = pos
        return True

    def open_positions(self) -> list[CrossPosition]:
        return list(self._positions.values())

    def close(
        self, pos: CrossPosition, *, route: str, profit: float, status: str,
        now: "float | None" = None,
    ) -> None:
        pos.status = status
        pos.exit_route = route
        pos.exit_profit = profit
        pos.closed_ts = now or time.time()
        self._positions.pop(pos.key, None)
        self._closed.append(pos)

    @property
    def closed(self) -> list[CrossPosition]:
        return list(self._closed)

    def summary(self) -> dict:
        """Totals across closed positions, in USDC."""
        closed = self._closed
        realised = sum(p.exit_profit * p.size for p in closed)
        early = [p for p in closed if p.status == "exited"]
        return {
            "open": len(self._positions),
            "closed": len(closed),
            "exited_early": len(early),
            "resolved": len(closed) - len(early),
            "realised_usdc": round(realised, 4),
            "median_days_held": (
                sorted(p.days_held() for p in closed)[len(closed) // 2]
                if closed else None
            ),
        }
