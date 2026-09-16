"""
strategy/market_rules.py
────────────────────────
The part of a market's resolution rules an implication depends on.

A title says what a market is about; the rules say what it pays on. Every
implication the model got wrong at whole-window scale was right about the
titles and wrong about the rules:

  "Julia Adams vs Rositsa Dencheva"  ⊆  "Completed Match: Julia Adams vs ..."
      A player can advance on a walkover or a retirement. The completed-match
      market then resolves No — "If a forfeit of any kind occurs, including a
      walkover or retirement..." — and a match not played at all splits the
      winner market 50-50. Three such pairs resolved against the trade within
      a day.

So the model is shown the rules. They are long — a Gamma description runs to a
thousand characters and more — and 70,000 of them would not fit the reader's
memory, so each is cut to a digest: the opening sentences, which say what
resolves YES, and every sentence about the cases where markets part ways —
cancellation, postponement, forfeit, ties, splits, the period that counts.
"""

from __future__ import annotations

import re

# A sentence ends at . ! or ? — possibly followed by a closing quote or bracket,
# as in: resolve to "No." If a forfeit occurs... Missing that left the forfeit
# clause glued to the sentence before it, too long to keep, and dropped.
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[A-Z\"'“(])")

_EDGE = re.compile(
    r"cancel|postpon|reschedul|walkover|retire|forfeit|abandon|suspend|void|"
    r"not (?:be )?(?:played|completed|held)|tie\b|tied|draw|50-50|50/50|split|"
    r"refund|regular (?:time|play)|stoppage|extra time|overtime|penalt|"
    r"first half|halftime|half-time|period|quarter|inning|set\b|"
    r"according to|source|official|timezone|\bET\b|UTC|deadline|by .*20\d\d",
    re.I,
)


def rules_digest(description: "str | None", limit: int = 900) -> str:
    """The opening sentences plus every sentence about an edge case, within `limit`."""
    text = re.sub(r"\s+", " ", str(description or "")).strip()
    if not text:
        return ""
    sentences, start = [], 0
    for m in _BOUNDARY.finditer(text):
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    sentences.append(text[start:].strip())
    keep = sentences[:2] + [x for x in sentences[2:] if _EDGE.search(x)]
    out = ""
    for x in keep:
        if len(x) > limit // 2:
            x = x[:limit // 2].rsplit(" ", 1)[0] + " …"
        if len(out) + len(x) + 1 > limit:
            continue                  # a shorter edge-case sentence may still fit
        out = f"{out} {x}".strip()
    return out or text[:limit]
