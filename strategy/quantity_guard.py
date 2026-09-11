"""
strategy/quantity_guard.py
──────────────────────────
Refuse an "implication" between two markets that count different things.

The first pair to clear every execution gate on the live book was:

    narrow  "Al Faisaly vs. Al Ittihad: Al Faisaly O/U 2.5 Corners"
    broad   "Al Faisaly vs. Al Ittihad: Al Faisaly O/U 2.5"          (goals)
    entry 0.7585 → "+0.2415 per pair", resolving in two hours

It is not an implication. Three corners say nothing about three goals. The
prices were right — over 2.5 corners is far likelier than over 2.5 goals — and in
the common case of plenty of corners and few goals the trade pays nothing on
either leg.

The shape prefilter calls it a NEST because one title is the other plus a word,
and for "1st Half" or a team name that word does narrow the claim. "Corners"
does not: it swaps the statistic being counted. The one-week execution window
makes this the dominant risk rather than an edge case, because what resolves
within a week is mostly match sub-markets — goals, corners, cards, shots,
games — all sharing the "O/U N" template. The model is the only other defence,
and the ladder direction guard does not see it: this is not a ladder.

The rule: two over/under markets can only nest if they count the same thing.

  • statistics are read from a fixed vocabulary, anywhere in the title
    ("Total Games O/U 3.5", "O/U 2.5 Corners");
  • goals and points are the unnamed default, so "O/U 2.5 Goals" and "O/U 2.5"
    count the same thing;
  • period qualifiers ("1st Half", "2nd Set") are stripped first — they narrow
    the claim, which is exactly what a real nest does;
  • it only judges pairs that are BOTH over/under markets. "Lakers win by 5+
    points" under "Lakers win" is a genuine nest, and the word "points" there is
    a margin, not a statistic.

A false refusal costs an opportunity; a false pass costs the position. The
vocabulary errs toward refusing.
"""

from __future__ import annotations

import re

_OVER_UNDER = re.compile(r"\bO/U\b|\bover/under\b", re.I)

# Period qualifiers narrow a claim in time; they are not a different statistic.
_PERIOD = re.compile(
    r"\b(1st|2nd|3rd|4th|first|second|third|fourth)\s+"
    r"(half|set|quarter|period|map|inning|innings|round|game)\b", re.I)

# Statistic -> canonical name. Goals and points are deliberately absent: they are
# what an unnamed over/under counts.
_STATISTICS: tuple[tuple[str, str], ...] = (
    (r"corners?", "corners"),
    (r"(?:yellow |red )?cards?|bookings?", "cards"),
    (r"shots?(?: on target)?", "shots"),
    (r"fouls?", "fouls"),
    (r"offsides?", "offsides"),
    (r"saves?", "saves"),
    (r"tackles?", "tackles"),
    (r"assists?", "assists"),
    (r"rebounds?", "rebounds"),
    (r"games?", "games"),
    (r"sets?", "sets"),
    (r"maps?", "maps"),
    (r"rounds?", "rounds"),
    (r"kills?", "kills"),
    (r"aces?", "aces"),
    (r"strikeouts?", "strikeouts"),
    (r"touchdowns?", "touchdowns"),
)
_STAT_RES = tuple((re.compile(rf"\b(?:{p})\b", re.I), name) for p, name in _STATISTICS)


def is_over_under(title: str) -> bool:
    return bool(_OVER_UNDER.search(title or ""))


def statistics(title: str) -> frozenset[str]:
    """What an over/under market counts. Empty means the default: goals, points."""
    text = _PERIOD.sub(" ", title or "")
    return frozenset(name for rx, name in _STAT_RES if rx.search(text))


def same_quantity(a_title: str, b_title: str) -> "bool | None":
    """
    True / False for two over/under markets; None when either is not one.

    None is not a pass or a refusal — it means this rule has nothing to say.
    """
    if not (is_over_under(a_title) and is_over_under(b_title)):
        return None
    return statistics(a_title) == statistics(b_title)
