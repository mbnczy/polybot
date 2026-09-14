"""
strategy/outcomes.py
────────────────────
Which outcome of a market is its "YES".

Everything downstream reads an implication as "the FIRST token of narrow
resolving implies the FIRST token of broad resolving": the export hands over
clobTokenIds[0] as the YES token, and the guard buys it. For a Yes/No market
that is the question as written. For anything else — an over/under line, a
spread, a pick between two teams — the title does not say which outcome is
first, and the model guessed.

On 2026-09-13 it guessed wrong. For "Cesena FC vs. US Cremonese: O/U 5.5" it
reasoned "a 0-0 means Under, so it resolves YES", while the first token — the
one the bot would have bought — is Over. Every WOULD ENTER in two days of
evaluate-only running was that mistake: a bet against a 0-0, dressed as an
arbitrage, losing the whole position whenever the match ended goalless.

These helpers make the YES side explicit where the title cannot.
"""

from __future__ import annotations

import json
import re


def market_outcomes(market: "dict | None") -> "tuple[str, str] | None":
    """The two outcome labels in token order, or None when they cannot be read."""
    raw = (market or {}).get("outcomes")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    return str(raw[0]).strip(), str(raw[1]).strip()


def is_yes_no(outcomes: "tuple[str, ...] | list[str] | None") -> bool:
    """True only for a market whose first token is provably "Yes"."""
    return (outcomes is not None and len(outcomes) == 2
            and (str(outcomes[0]).lower(), str(outcomes[1]).lower()) == ("yes", "no"))


def _names(text: str, label: str) -> bool:
    return bool(label) and re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text, re.I) is not None


def contradicts_yes_side(text: str, outcomes: "tuple[str, ...] | None") -> "str | None":
    """
    The NO label, when a verdict's reasoning is about that outcome and never
    names the YES one — the Cesena mistake. None when there is nothing to check.
    """
    if outcomes is None or is_yes_no(outcomes):
        return None
    yes, no = outcomes[0], outcomes[1]
    if _names(text or "", no) and not _names(text or "", yes):
        return no
    return None
