"""
strategy/cross_market.py
────────────────────────
Cross-market implication arbitrage — DETECTION ONLY.

Background
──────────
Single-market Dutch Book arbs (YES + NO < $1) are rare. future_arb_strategies.md
cites 7 per month across 173 NBA games, versus ~290 between *related* markets —
roughly 40x the opportunity count.

The relationship that matters is strict IMPLICATION, not correlation:

    A implies B   ⟹   P(A) ≤ P(B)   always

e.g. "Team X wins by 5+" implies "Team X wins". When the book prices A above B
that ordering is violated, and the violation is tradeable with a payout floor:

    buy B_YES at price_B  +  buy A_NO at (1 − price_A)
    cost = price_B + 1 − price_A  <  1        (since price_A > price_B)

      A occurs      → B occurs too : B_YES 1.00 + A_NO 0.00 = 1.00
      B but not A   →               : B_YES 1.00 + A_NO 1.00 = 2.00
      neither       →               : B_YES 0.00 + A_NO 1.00 = 1.00

Payout ≥ 1.00 in every branch for a cost below 1.00 — contract-guaranteed, the
same shape as the YES+NO pair the bot already trades. There is no directional
exposure.

The risk is therefore NOT market movement. It is that the implication itself is
wrong: two markets that look related but resolve off different events, sources
or timeframes. A mapping error turns a "guaranteed" trade into an unhedged bet.

That is why this module ships detection-only. It logs what it *would* have
traded so the implications can be audited against reality before a single order
is placed.

Design
──────
`RelationRegistry` is deliberately a dumb store. The prototype fills it with
conservative title heuristics (`discover_from_markets`); the LLM mapper proposed
in the research doc is a drop-in replacement for that one function, and nothing
downstream needs to change. Slow inference stays off the hot path by construction
— the registry is consulted, never computed, during price evaluation.

Nothing here places orders.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Minimum violation, in probability terms, before a mispricing is worth logging.
# Below this the "edge" is inside the tick grid and fee noise.
CROSS_MIN_EDGE: float = float(os.environ.get("CROSS_MIN_EDGE", 0.02))

# Confidence floor for a discovered implication. Heuristic matches below this are
# recorded but never evaluated, so a weak guess cannot produce a signal.
CROSS_MIN_CONFIDENCE: float = float(os.environ.get("CROSS_MIN_CONFIDENCE", 0.90))


# ═══════════════════════════════════════════════════════════════════════════════
# Relationships
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Implication:
    """
    `narrow` occurring guarantees `broad` occurring:  narrow ⊆ broad.

    Both are Polymarket condition IDs. `confidence` is how sure we are that the
    implication actually holds — 1.0 for a hand-verified pair, lower for a
    heuristic or model-proposed one.

    `evidence` records WHY the relation was asserted, so a bad mapping can be
    traced back to the rule that produced it. That audit trail is the whole
    safety story for this strategy.
    """
    narrow:     str
    broad:      str
    confidence: float
    evidence:   str = ""

    def __post_init__(self) -> None:
        if self.narrow == self.broad:
            raise ValueError("an implication needs two distinct markets")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1]; got {self.confidence}")


@dataclass
class CrossMarketSignal:
    """A violated implication, priced. Informational — nothing executes it."""
    narrow:        str
    broad:         str
    narrow_title:  str
    broad_title:   str
    narrow_price:  float     # P(narrow) implied by its YES ask
    broad_price:   float     # P(broad)  implied by its YES ask
    violation:     float     # narrow_price − broad_price, > 0 when inconsistent
    cost:          float     # broad_price + (1 − narrow_price)
    min_payout:    float     # guaranteed floor (1.00)
    edge:          float     # min_payout − cost
    confidence:    float
    evidence:      str

    def describe(self) -> str:
        return (
            f"{self.narrow_title[:44]!r} priced {self.narrow_price:.3f} > "
            f"{self.broad_title[:44]!r} at {self.broad_price:.3f} "
            f"(implies ≤) | cost {self.cost:.4f} → floor {self.min_payout:.2f} "
            f"| edge {self.edge:+.4f} ({self.edge * 10_000:+.0f} bps) "
            f"| confidence {self.confidence:.2f}"
        )


class RelationRegistry:
    """
    Stores implications, indexed for O(1) lookup by either side.

    Intentionally has no discovery logic of its own: relations arrive from
    whatever source the caller trusts (heuristics today, an LLM mapper later).
    """

    def __init__(self) -> None:
        self._by_narrow: dict[str, list[Implication]] = {}
        self._all: list[Implication] = []

    def add(self, rel: Implication) -> None:
        if any(r.narrow == rel.narrow and r.broad == rel.broad for r in self._all):
            return
        self._all.append(rel)
        self._by_narrow.setdefault(rel.narrow, []).append(rel)

    def extend(self, rels: Iterable[Implication]) -> None:
        for r in rels:
            self.add(r)

    def for_narrow(self, condition_id: str) -> list[Implication]:
        return self._by_narrow.get(condition_id, [])

    def __len__(self) -> int:
        return len(self._all)

    @property
    def all(self) -> list[Implication]:
        return list(self._all)


# ═══════════════════════════════════════════════════════════════════════════════
# Prototype discovery — conservative title heuristics
# ═══════════════════════════════════════════════════════════════════════════════

# "by 5+ points", "by 10 or more", "by at least 3"
# A margin qualifier needs a UNIT ("by 5 points") or an explicit open bound
# ("by 5+", "by at least 5"). Without that constraint "by 2027" — a deadline —
# matches, which is how the first draft produced two false implications out of a
# 2,100-market scan. Four-digit years are excluded outright.
_MARGIN_RE = re.compile(
    r"\bby\s+(?:"
    r"at\s+least\s+(?!\d{4}\b)\d{1,3}(?:\.\d+)?"                # by at least 3
    r"|(?!\d{4}\b)\d{1,3}(?:\.\d+)?\s*\+"                        # by 5+
    r"|(?!\d{4}\b)\d{1,3}(?:\.\d+)?\s+or\s+more"                 # by 5 or more
    r"|(?!\d{4}\b)\d{1,3}(?:\.\d+)?\s+"                          # by 5 points
    r"(?:points?|goals?|runs?|games?|sets?|lengths?|strokes?)"
    r")"
    r"(?:\s*(?:points?|goals?|runs?|games?|sets?|lengths?|strokes?))?",
    re.IGNORECASE,
)

_NOISE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalise(title: str) -> str:
    return _NOISE_RE.sub(" ", title.lower()).strip()


def _strip_margin(title: str) -> str:
    return _normalise(_MARGIN_RE.sub(" ", title))


def discover_from_markets(markets: list[dict]) -> list[Implication]:
    """
    Propose implications from market titles — the prototype stand-in for the
    LLM mapper in future_arb_strategies.md #3.

    One deliberately narrow rule: a market whose title is another's title plus a
    winning-margin qualifier is strictly narrower.

        "Will the Lakers win by 5+ points?"  ⊆  "Will the Lakers win?"

    Conservative on purpose. Both titles must reduce to the SAME string once the
    margin phrase is removed, and both must belong to the same event when that
    field is present. A false implication is the only way this strategy loses
    money, so the prototype would rather find nothing than guess.

    Returns proposals — the caller decides what to trust.
    """
    by_stripped: dict[str, list[dict]] = {}
    out: list[Implication] = []

    for m in markets:
        title = str(m.get("question") or m.get("title") or "").strip()
        cid   = str(m.get("conditionId") or m.get("condition_id") or "").strip()
        if not title or not cid:
            continue
        if _MARGIN_RE.search(title):
            by_stripped.setdefault(_strip_margin(title), []).append(m)

    plain: dict[str, dict] = {}
    for m in markets:
        title = str(m.get("question") or m.get("title") or "").strip()
        cid   = str(m.get("conditionId") or m.get("condition_id") or "").strip()
        if not title or not cid or _MARGIN_RE.search(title):
            continue
        plain[_normalise(title)] = m

    for stripped, narrows in by_stripped.items():
        broad = plain.get(stripped)
        if broad is None:
            continue
        broad_cid   = str(broad.get("conditionId") or broad.get("condition_id"))
        broad_event = broad.get("eventId") or broad.get("event_id")
        for n in narrows:
            n_cid   = str(n.get("conditionId") or n.get("condition_id"))
            n_event = n.get("eventId") or n.get("event_id")
            if n_cid == broad_cid:
                continue
            # Same-event check when the field exists; refuse to link across events.
            if broad_event and n_event and broad_event != n_event:
                logger.debug(
                    "cross-market | title match but different events "
                    "(%s vs %s) — refusing", n_event, broad_event,
                )
                continue
            same_event = bool(broad_event and n_event and broad_event == n_event)
            out.append(Implication(
                narrow=n_cid,
                broad=broad_cid,
                # Same-event confirmation is what separates a safe mapping from
                # a coincidental title match.
                confidence=0.95 if same_event else 0.85,
                evidence=(
                    f"title '{str(n.get('question'))[:48]}' is "
                    f"'{str(broad.get('question'))[:48]}' + margin qualifier"
                    + ("; same eventId" if same_event else "; event unconfirmed")
                ),
            ))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════════════════════════════════════

class CrossMarketDetector:
    """
    Watches implication pairs for price-ordering violations.

    Feed it YES prices as they arrive; it reports pairs where the narrower
    market is priced ABOVE the broader one, which is logically impossible and
    therefore tradeable.

    Pure lookup — no I/O, no inference — so it is safe on the hot path.
    """

    def __init__(
        self,
        registry:       RelationRegistry,
        min_edge:       float = CROSS_MIN_EDGE,
        min_confidence: float = CROSS_MIN_CONFIDENCE,
    ) -> None:
        self._reg   = registry
        self._edge  = min_edge
        self._conf  = min_confidence
        self._price: dict[str, float] = {}
        self._title: dict[str, str] = {}

    def update_price(
        self, condition_id: str, yes_price: float, title: str = ""
    ) -> list[CrossMarketSignal]:
        """
        Record a market's YES price and return any violations it creates.

        Both directions are checked: this market may be the narrow side of one
        implication and the broad side of another.
        """
        if not (0.0 < yes_price < 1.0):
            return []
        self._price[condition_id] = yes_price
        if title:
            self._title[condition_id] = title

        signals: list[CrossMarketSignal] = []
        for rel in self._reg.for_narrow(condition_id):
            sig = self._check(rel)
            if sig:
                signals.append(sig)
        for rel in self._reg.all:
            if rel.broad == condition_id:
                sig = self._check(rel)
                if sig:
                    signals.append(sig)
        return signals

    def _check(self, rel: Implication) -> Optional[CrossMarketSignal]:
        if rel.confidence < self._conf:
            return None
        pn = self._price.get(rel.narrow)
        pb = self._price.get(rel.broad)
        if pn is None or pb is None:
            return None

        violation = pn - pb
        if violation <= 0.0:
            return None                      # ordering intact, nothing to do

        # buy broad YES + narrow NO; floor payout is 1.00 in every branch
        cost = pb + (1.0 - pn)
        edge = 1.0 - cost                    # == violation, kept explicit
        if edge < self._edge:
            return None

        return CrossMarketSignal(
            narrow=rel.narrow, broad=rel.broad,
            narrow_title=self._title.get(rel.narrow, rel.narrow[:16]),
            broad_title=self._title.get(rel.broad, rel.broad[:16]),
            narrow_price=pn, broad_price=pb,
            violation=violation, cost=cost, min_payout=1.0, edge=edge,
            confidence=rel.confidence, evidence=rel.evidence,
        )
