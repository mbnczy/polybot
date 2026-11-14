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

Spreads join them. "Spread: A (-k.5)" whose first outcome is A is "A wins by
k+1 or more", which needs A to have scored k+1: it sits inside "A O/U k.5", and
inside the same team's spread one line lower. A spread's title does not name the
match, so it is tied to the match's over/unders by the market's event.

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

_SPREAD = re.compile(r"^Spread: (?P<team>.+) \(-(?P<line>\d+\.5)\)$")

TOTAL, FULL = "total", "full"


def _event_id(market: dict) -> "str | None":
    ev = market.get("events")
    if isinstance(ev, list) and ev and isinstance(ev[0], dict) and ev[0].get("id") is not None:
        return str(ev[0]["id"])
    return None


@dataclass(frozen=True)
class Spread:
    cid:   str
    ev_id: str         # the market's event: a spread's title does not name the match
    team:  str
    line:  float       # wins by line + 0.5 or more


def parse_spread(market: dict) -> "Spread | None":
    """ "Spread: A (-k.5)" with A as the first outcome, else None."""
    m = _SPREAD.match(str(market.get("question") or "").strip())
    ev_id = _event_id(market)
    if not m or ev_id is None or not market.get("conditionId"):
        return None
    raw = market.get("outcomes")
    try:
        outcomes = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    except (TypeError, ValueError):
        return None
    if not outcomes or str(outcomes[0]).strip() != m.group("team"):
        return None                  # the YES token has to be the named team covering
    return Spread(str(market["conditionId"]), ev_id, m.group("team"), float(m.group("line")))


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
    # The match is the event, not the title. A baseball series lists the same
    # two teams on consecutive days under identical titles: keyed on the title,
    # "O/U 8.5 ⊆ O/U 7.5" linked Saturday's game to Friday's, 35 of those pairs
    # were violated at resolution in September 2026, and their "+5.5% taker
    # edges" were two different games' prices.
    fixture = _event_id(market) or str(market.get("endDate") or "")
    return OverUnder(cid=str(market["conditionId"]),
                     event=f"{m.group('a')} vs. {m.group('b')}#{fixture}",
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
    pa, pb = parse_strike(a), parse_strike(b)
    if pa is not None and pb is not None and pa.group == pb.group:
        narrow, broad = ((pa, pb) if (pa.strike > pb.strike) == pa.rising else (pb, pa))
        return _creation_safe(narrow, broad)
    x, y = parse(a), parse(b)
    if x is not None and y is not None:
        return x.event == y.event
    sx, sy = parse_spread(a), parse_spread(b)
    if (sx or x) is None or (sy or y) is None or (sx is None and sy is None):
        return False
    ea, eb = _event_id(a), _event_id(b)
    return ea is not None and ea == eb


def links(markets: list[dict]) -> list[Implication]:
    """The linking implications of every match's over/under and spread set."""
    by_event: dict[str, list[OverUnder]] = {}
    by_ev_id: dict[str, list[OverUnder]] = {}
    spreads: dict[tuple[str, str], list[Spread]] = {}
    for m in markets:
        ou = parse(m)
        if ou is not None:
            by_event.setdefault(ou.event, []).append(ou)
            if _event_id(m) is not None:
                by_ev_id.setdefault(_event_id(m), []).append(ou)
            continue
        sp = parse_spread(m)
        if sp is not None:
            spreads.setdefault((sp.ev_id, sp.team), []).append(sp)
    out: list[Implication] = price_links(markets)
    for (ev_id, team), sps in spreads.items():
        sps.sort(key=lambda s_: s_.line)
        for lower, upper in zip(sps, sps[1:]):
            out.append(Implication(upper.cid, lower.cid, 1.0,
                                   f"{SOURCE}:spread-ladder: {team} -{upper.line:g} ⊆ "
                                   f"{team} -{lower.line:g}"))
        goals = {(o.scope, o.period, o.line): o for o in by_ev_id.get(ev_id, ())}
        for sp in sps:
            tgt = goals.get((team, FULL, sp.line)) or goals.get((TOTAL, FULL, sp.line))
            if tgt is not None:
                out.append(Implication(sp.cid, tgt.cid, 1.0,
                                       f"{SOURCE}:spread-goals: {team} -{sp.line:g} ⊆ "
                                       f"{'match' if tgt.scope == TOTAL else team} over "
                                       f"{tgt.line:g}"))
    for event, ous in by_event.items():
        by_family: dict[tuple[str, str], list[OverUnder]] = {}
        for ou in ous:
            by_family.setdefault(ou.family, []).append(ou)
        # A ladder: each line inside the next lower one of the same family.
        for fam in by_family.values():
            fam.sort(key=lambda o: o.line)
            for lower, upper in zip(fam, fam[1:]):
                if upper.line > lower.line:          # a listed twin is no ladder step
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


# ── price ladders ─────────────────────────────────────────────────────────────
#
# The model's own maker signals on 2026-09-18 were mostly weekly stock ladders —
# "hit (HIGH) $X", "finish week above $X" — found a pair at a time. Their rules
# make them ladders outright:
#
#   hit (HIGH) $X     any 1-minute candle's High ≥ X during the week's regular
#   hit (LOW) $X      hours, after the market was created (Low ≤ X)
#   finish/close/     one observation: the week's (the day's) close, or the
#   price above $X    price at a stated time, above X
#
# so a higher strike sits inside a lower one (a lower one inside a higher, for
# LOW). "After the market was created" is the catch for the hit ladders: a
# broader strike created later could have missed a move the narrower one
# caught. They are linked only when the broader strike existed before the
# narrower one did, or before the week's first session opened — ladders are
# created together on the Friday before, and that is the normal case.

_PRICE = r"\$(?P<x>\d[\d,]*(?:\.\d+)?)"
_DAY = r"(?P<p>[A-Z][a-z]+ \d{1,2}(?:,? \d{4})?)"
_PRICE_TITLES = (
    ("hit", re.compile(rf"^Will (?P<u>.+?) \((?P<t>[A-Z][A-Z0-9.\-]*)\) hit \((?P<d>HIGH|LOW)\) "
                       rf"{_PRICE} Week of {_DAY}\?$")),
    ("finish", re.compile(rf"^Will (?P<u>.+?) \((?P<t>[A-Z][A-Z0-9.\-]*)\) finish week of "
                          rf"{_DAY} above {_PRICE}\?$")),
    ("close", re.compile(rf"^Will (?P<u>.+?) \((?P<t>[A-Z][A-Z0-9.\-]*)\) close above "
                         rf"{_PRICE} on {_DAY}\?$")),
    ("closes", re.compile(rf"^(?P<u>.+?) \((?P<t>[A-Z]+)\) closes above {_PRICE} on {_DAY}\?$")),
    ("price", re.compile(rf"^Will the price of (?P<u>[A-Za-z][A-Za-z .]*) be above {_PRICE} "
                         rf"on {_DAY}\?$")),
)
_WEEK_OPEN_UTC_H = 13          # 09:30 New York is 13:30 UTC in summer; earlier is stricter


@dataclass(frozen=True)
class Strike:
    cid:     str
    group:   tuple             # (kind, underlying, period, end date, HIGH/LOW)
    strike:  float
    rising:  bool              # a higher strike is the narrower event
    created: "float | None"
    opens:   "float | None"    # when prices start to count, for path-dependent kinds


def _ts(raw) -> "float | None":
    from datetime import datetime  # noqa: PLC0415
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _week_open(period: str) -> "float | None":
    from datetime import datetime, timezone  # noqa: PLC0415
    for fmt in ("%B %d %Y", "%B %d, %Y"):
        try:
            d = datetime.strptime(period, fmt).replace(tzinfo=timezone.utc)
            return d.replace(hour=_WEEK_OPEN_UTC_H).timestamp()
        except ValueError:
            continue
    return None


def parse_strike(market: dict) -> "Strike | None":
    """The market as one strike of a price ladder, or None."""
    q = str(market.get("question") or "").strip()
    if not market.get("conditionId") or _first_outcome(market) != "yes":
        return None
    for kind, rx in _PRICE_TITLES:
        m = rx.match(q)
        if not m:
            continue
        d = m.groupdict().get("d")
        under = (m.groupdict().get("t") or m.group("u")).strip()
        try:
            x = float(m.group("x").replace(",", ""))
        except ValueError:
            return None
        return Strike(cid=str(market["conditionId"]),
                      group=(kind, under, m.group("p"), str(market.get("endDate") or ""), d),
                      strike=x, rising=(d != "LOW"),
                      created=_ts(market.get("createdAt")),
                      opens=_week_open(m.group("p")) if kind == "hit" else None)
    return None


def price_ladder_kind(title: str) -> "str | None":
    """Which kind of price-ladder rung a title is ("hit", "close", …), or None."""
    q = str(title or "").strip()
    for kind, rx in _PRICE_TITLES:
        if rx.match(q):
            return kind
    return None


def _creation_safe(narrow: Strike, broad: Strike) -> bool:
    """For a path-dependent ladder: did the broad strike exist whenever the narrow could have been hit?"""
    if narrow.opens is None and narrow.group[0] != "hit":
        return True                              # one observation: creation cannot matter
    if narrow.created is None or broad.created is None:
        return False
    since = max(narrow.created, narrow.opens or narrow.created)
    return broad.created <= since


def price_links(markets: list[dict]) -> list[Implication]:
    groups: dict[tuple, list[Strike]] = {}
    for m in markets:
        s = parse_strike(m)
        if s is not None:
            groups.setdefault(s.group, []).append(s)
    out: list[Implication] = []
    for group, strikes in groups.items():
        strikes.sort(key=lambda s: s.strike)
        for lo, hi in zip(strikes, strikes[1:]):
            if lo.strike == hi.strike:
                continue
            narrow, broad = (hi, lo) if lo.rising else (lo, hi)
            if not _creation_safe(narrow, broad):
                continue
            kind, under = group[0], group[1]
            label = f"{kind} {group[4]}" if group[4] else kind
            out.append(Implication(narrow.cid, broad.cid, 1.0,
                                   f"{SOURCE}:price-ladder: {under} {label} {narrow.strike:g} ⊆ "
                                   f"{broad.strike:g}"))
    return out
