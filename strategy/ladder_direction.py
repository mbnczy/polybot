"""
strategy/ladder_direction.py
────────────────────────────
Refuse a threshold implication whose direction is provably backwards.

What went wrong
───────────────
The ten largest arbitrages the live reader ever reported were one pair, read
backwards:

    narrow  "Will Trump's approval rating hit 35% in 2026?"   0.35
    broad   "Will Trump's approval rating hit 25% in 2026?"   0.05
                                        -> "+3000 bps, 140% APR"

"Hit" here means FALL to. Falling to 25% requires passing 35% on the way, so
hitting 25% implies hitting 35%: the narrow market is 25%, not 35%. The prices
were right all along, and a trade on the "violation" pays nothing on either leg
if the rating settles at 30%. The strategy's floor is only guaranteed when the
implication is true.

Two sources of direction, in order
──────────────────────────────────
1. What the title SAYS. "Above", "over", "at least" make a rising ladder — the
   higher threshold is the narrower event. "Below", "under", "dip", "fall" make
   a falling one. When the words settle it, the words win, and no family is
   needed: a two-market "above" pair read backwards is caught too.

2. What the family's PRICES say — only when the verb does not commit. "Hit" and
   "reach" are a rise on one side of the current value and a fall on the other.
   Measured on the Trump ladder:

       hit 20%  0.035    hit 25%  0.045    hit 30%  0.050    hit 35%  0.320

   Price rises with the threshold in all six pairs — coherent only if a lower
   threshold is harder, a falling ladder. A genuine arbitrage is LOCAL, one pair
   out of order, and moves Kendall's sign by one pair of many; a direction error
   is GLOBAL. So for an ambiguous verb with at least three priced members, the
   trend decides.

Why the order matters
─────────────────────
The first version used prices for everything, and the live survey showed where
that goes wrong: "Metamask FDV above $700M" and "Consensys IPO market cap above
$1B" both came back as falling ladders, because their whole families were priced
with higher thresholds costing MORE. "Above" is unambiguous — the higher
threshold is narrower — so those families are not reversed; they are mispriced
across the board. Letting prices overrule the title there would have thrown out
the implication the model got RIGHT, on exactly the pairs where the reader was
seeing real edges. Prices are only trusted where the words are silent.

What it cannot do
─────────────────
An ambiguous verb on a family of two stays UNKNOWN: the pair is the whole family
and a mispricing and a reversal look alike. UNKNOWN passes through unchanged —
this check only removes a relation it can show is backwards.
"""

from __future__ import annotations

import json
import re
from itertools import combinations

CONSISTENT = "consistent"
CONTRADICTS = "contradicts"
UNKNOWN = "unknown"

RISING = "rising"     # higher threshold is the narrower event
FALLING = "falling"   # lower threshold is the narrower event

SOURCE_STATED = "stated"
SOURCE_PRICES = "prices"

# A number as it appears in a market title: optional sign, optional $, digits
# with thousands separators, optional decimals, optional magnitude or percent.
_NUM = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\s?(?:[kmbt](?![a-z])|%)?", re.I)
_MAGNITUDE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

_RISING_WORDS = re.compile(
    r"\b(above|over|greater than|more than|at least|exceeds?|exceeding|"
    r"surpass(?:es)?|or more|or higher|higher than)\b", re.I)
_FALLING_WORDS = re.compile(
    r"\b(below|under|less than|fewer than|at most|dips?|falls?|drops?|sinks?|"
    r"or less|or lower|lower than)\b", re.I)

# Fewest priced members that can establish a direction from prices alone.
MIN_FAMILY = 3


def _value(token: str) -> "float | None":
    t = token.strip().lower().replace(",", "").replace("$", "").replace(" ", "")
    mult = 1.0
    if t.endswith("%"):
        t = t[:-1]
    elif t and t[-1] in _MAGNITUDE:
        mult, t = _MAGNITUDE[t[-1]], t[:-1]
    try:
        return float(t) * mult
    except ValueError:
        return None


def numbers(title: str) -> list[float]:
    """Every number in a title, in order."""
    out = []
    for m in _NUM.finditer(title or ""):
        v = _value(m.group(0))
        if v is not None:
            out.append(v)
    return out


def skeleton(title: str) -> str:
    """The title with every number blanked — members of a ladder share it."""
    return _NUM.sub("#", (title or "").strip().lower())


def stated_direction(title: str) -> "str | None":
    """RISING or FALLING from the title's own words; None when they do not commit."""
    r = bool(_RISING_WORDS.search(title or ""))
    f = bool(_FALLING_WORDS.search(title or ""))
    if r != f:
        return RISING if r else FALLING
    return None


def _yes_price(market: dict) -> "float | None":
    try:
        p = float(json.loads(market.get("outcomePrices") or "[]")[0])
    except (TypeError, ValueError, IndexError):
        return None
    return p if 0.0 < p < 1.0 else None


def _question(market: dict) -> str:
    return str(market.get("question") or market.get("title") or "")


def family_trend(members: list[tuple[float, float]]) -> int:
    """
    Kendall's sign of price against threshold, over every pair.

    Positive: price rises with the threshold — a lower threshold is harder, a
    FALLING ladder. Negative: a RISING ladder.
    """
    s = 0
    for (xa, pa), (xb, pb) in combinations(members, 2):
        dx, dp = xb - xa, pb - pa
        if dx == 0 or dp == 0:
            continue
        s += 1 if (dx > 0) == (dp > 0) else -1
    return s


def _price_direction(sk: str, idx: int, markets: list[dict]) -> "str | None":
    """Direction from the family's prices, or None when they cannot say."""
    members = []
    for m in markets:
        q = _question(m)
        if skeleton(q) != sk:
            continue
        n = numbers(q)
        p = _yes_price(m)
        if p is None or idx >= len(n):
            continue
        members.append((n[idx], p))
    if len({x for x, _ in members}) < MIN_FAMILY:
        return None
    t = family_trend(members)
    if t == 0:
        return None
    return FALLING if t > 0 else RISING


def direction_of(narrow_id: str, broad_id: str, markets: list[dict]
                 ) -> "tuple[str | None, str | None, int | None]":
    """
    (direction, source, threshold index) for a ladder pair.

    (None, None, None) when the two markets are not a clean ladder: different
    skeletons, or not exactly one number differing between them.
    """
    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    mn, mb = by_id.get(narrow_id), by_id.get(broad_id)
    if not mn or not mb:
        return None, None, None
    qn, qb = _question(mn), _question(mb)
    sk = skeleton(qn)
    if sk != skeleton(qb):
        return None, None, None
    xn, xb = numbers(qn), numbers(qb)
    if len(xn) != len(xb):
        return None, None, None
    # The threshold is the one position that differs. The year in "hit 35% in
    # 2026" is a number too, and it is the same on both sides.
    diff = [i for i in range(len(xn)) if xn[i] != xb[i]]
    if len(diff) != 1:
        return None, None, None
    idx = diff[0]

    stated = stated_direction(qn)
    if stated is not None:
        return stated, SOURCE_STATED, idx
    priced = _price_direction(sk, idx, markets)
    if priced is not None:
        return priced, SOURCE_PRICES, idx
    return None, None, idx


def check_direction(narrow_id: str, broad_id: str, markets: list[dict]) -> str:
    """CONSISTENT / CONTRADICTS / UNKNOWN for one asserted implication."""
    direction, _source, idx = direction_of(narrow_id, broad_id, markets)
    if direction is None or idx is None:
        return UNKNOWN
    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    x_narrow = numbers(_question(by_id[narrow_id]))[idx]
    x_broad = numbers(_question(by_id[broad_id]))[idx]
    narrow_is_higher = x_narrow > x_broad
    ok = narrow_is_higher if direction == RISING else not narrow_is_higher
    return CONSISTENT if ok else CONTRADICTS


def filter_implications(rels: list, markets: list[dict]) -> tuple[list, list]:
    """
    Split asserted implications into (kept, rejected-as-backwards).

    Only CONTRADICTS is removed. UNKNOWN passes, because this check exists to
    remove relations it can show are wrong, not to demand proof of every one.
    """
    kept, rejected = [], []
    for r in rels:
        if check_direction(r.narrow, r.broad, markets) == CONTRADICTS:
            rejected.append(r)
        else:
            kept.append(r)
    return kept, rejected
