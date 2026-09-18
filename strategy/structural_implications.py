"""
strategy/structural_implications.py
───────────────────────────────────
Implications between a match's over/under markets, read off the titles.

Of the 968 implications the model had exported on 2026-09-18, 93% sat inside one
match and 91% were two shapes: "1st Half O/U k ⊆ O/U k" and "O/U ⊆ O/U". Those
are not judgments. A half's goals never exceed the match's, and a team's never
exceed both teams', so

    X ≤ Y always, and j ≤ k   ⇒   "X over k" ⊆ "Y over j"

Feyenoord–Utrecht alone lists 47 over/under markets, from which a few hundred
such implications follow; the model had found three. It spends its budget one
pair at a time on facts a rule states for every match at once, and the per-event
cap threw most of them away before it saw them.

What is and is not read
───────────────────────
Only titles of the exact shapes

    "<A> vs. <B>: [<A or B>] [1st Half | 2nd Half] O/U <k.5>"

whose first outcome is "Over". The team must be one of the two named in the
event, so "Corners O/U 9.5" or a player's statistic never parses. The line must
be a half: an integer line can land exactly and refund, and "over 2" is then
not the event the rule reasons about.

Only the links are emitted: every dominance at the same line, and each family's
ladder between neighbouring lines. "X over k ⊆ Y over j" is the chain X over k ⊆
Y over k ⊆ Y over k−1 ⊆ … ⊆ Y over j, so when it is violated at least one link
is — provided Y lists line k, as the bounding family, the one with the larger
count, does. The rest would multiply the pairs the guard prices for nothing new.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations

from strategy.cross_market import Implication

SOURCE = "structural"

_TITLE = re.compile(
    # The team group is lazy (??): "1st Half O/U 0.5" must read as a period,
    # not as a team called "1st Half".
    r"^(?P<a>.+?) vs\. (?P<b>.+?): (?:(?P<team>.+?) )??(?:(?P<period>1st Half|2nd Half) )?"
    r"O/U (?P<line>\d+\.5)$")

TOTAL, FULL = "total", "full"


@dataclass(frozen=True)
class OverUnder:
    cid:    str
    event:  str        # "A vs. B" — the match both markets must share
    scope:  str        # TOTAL, or the team's name as the event states it
    period: str        # FULL, "1st Half" or "2nd Half"
    line:   float

    @property
    def family(self) -> tuple[str, str]:
        return (self.scope, self.period)


def parse(market: dict) -> "OverUnder | None":
    """The market as an over/under on goals, or None when it is anything else."""
    q = str(market.get("question") or "").strip()
    m = _TITLE.match(q)
    if not m or not market.get("conditionId"):
        return None
    team = m.group("team")
    if team is not None and team not in (m.group("a"), m.group("b")):
        return None                  # "Corners", a player, a statistic
    if _first_outcome(market) != "over":
        return None                  # the YES token has to be Over
    return OverUnder(cid=str(market["conditionId"]),
                     event=f"{m.group('a')} vs. {m.group('b')}",
                     scope=team or TOTAL, period=m.group("period") or FULL,
                     line=float(m.group("line")))


def _first_outcome(market: dict) -> str:
    raw = market.get("outcomes")
    try:
        outcomes = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    except (TypeError, ValueError):
        return ""
    return str(outcomes[0]).strip().lower() if outcomes else ""


def dominated(x: OverUnder, y: OverUnder) -> bool:
    """X's count never exceeds Y's: same match, and X's team and period inside Y's."""
    return (x.event == y.event
            and (x.scope == y.scope or y.scope == TOTAL)
            and (x.period == y.period or y.period == FULL))


def implies(x: OverUnder, y: OverUnder) -> bool:
    """ "X over x.line" ⊆ "Y over y.line"."""
    return (x.cid != y.cid and dominated(x, y) and y.line <= x.line
            and (x.family, x.line) != (y.family, y.line))


def decides(a: dict, b: dict) -> bool:
    """True when the rule settles this pair either way, so the model need not."""
    x, y = parse(a), parse(b)
    return x is not None and y is not None and x.event == y.event


def links(markets: list[dict]) -> list[Implication]:
    """The linking implications of every match's over/under set."""
    by_event: dict[str, list[OverUnder]] = {}
    for m in markets:
        ou = parse(m)
        if ou is not None:
            by_event.setdefault(ou.event, []).append(ou)
    out: list[Implication] = []
    for event, ous in by_event.items():
        by_family: dict[tuple[str, str], list[OverUnder]] = {}
        for ou in ous:
            by_family.setdefault(ou.family, []).append(ou)
        # A ladder: each line inside the next lower one of the same family.
        for fam in by_family.values():
            fam.sort(key=lambda o: o.line)
            for lower, upper in zip(fam, fam[1:]):
                out.append(_imp(upper, lower, "ladder"))
        # Across families at the same line, wherever one count bounds the other.
        by_line: dict[float, list[OverUnder]] = {}
        for ou in ous:
            by_line.setdefault(ou.line, []).append(ou)
        for same in by_line.values():
            for x, y in combinations(same, 2):
                if dominated(x, y) and x.family != y.family:
                    out.append(_imp(x, y, "nested"))
                elif dominated(y, x) and x.family != y.family:
                    out.append(_imp(y, x, "nested"))
    return out


def _imp(narrow: OverUnder, broad: OverUnder, kind: str) -> Implication:
    def name(o):
        return " ".join(p for p in (o.scope if o.scope != TOTAL else "",
                                    o.period if o.period != FULL else "") if p) or "match"
    return Implication(narrow.cid, broad.cid, 1.0,
                       f"{SOURCE}:{kind}: {name(narrow)} over {narrow.line:g} ⊆ "
                       f"{name(broad)} over {broad.line:g}")
