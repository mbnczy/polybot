"""
strategy/verdict_cache.py
─────────────────────────
Remember what the model already said about a pair of markets.

The prefilter builds 2,500–4,000 candidate pairs a pass and we sent 30 to the
model — one percent — because 30 pairs cost three minutes of a 31b model. The
cap was the breadth limit, and breadth is where the arbitrage is.

But a verdict does not expire: "1st half O/U 0.5 ⊆ match O/U 0.5" is a fact
about two questions, and both questions are immutable once listed. So the model
should see a pair ONCE, ever, and every later pass should get it for free. The
cost of covering thousands of pairs then stops being per pass and becomes
one-time — which is what makes a wide net affordable at all.

Negative verdicts are cached too, and they are the bulk of it: most candidate
pairs are not implications, and re-asking about them is where the time went.

Entries are pruned by age so a cache that has run for months does not grow
without bound; the markets themselves are long gone by then.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from strategy.cross_market import Implication

logger = logging.getLogger(__name__)

_TTL_DAYS = 30.0


def _key(a_id: str, b_id: str) -> str:
    """One key per unordered pair: the model decides which way it nests."""
    return "|".join(sorted((str(a_id), str(b_id))))


class VerdictCache:
    def __init__(self, path: "str | Path", ttl_days: float = _TTL_DAYS) -> None:
        self._path = Path(path)
        self._ttl = ttl_days * 86_400.0
        self._rows: dict[str, dict] = {}
        self._dirty = False

    # ── persistence ───────────────────────────────────────────────────────────

    def load(self) -> "VerdictCache":
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return self
        except (OSError, ValueError) as exc:
            logger.warning("verdict cache unreadable (%s): starting empty", exc)
            return self
        if isinstance(raw, dict):
            self._rows = {k: v for k, v in raw.get("verdicts", {}).items()
                          if isinstance(v, dict)}
        return self

    def save(self) -> None:
        if not self._dirty:
            return
        self.prune()
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps({"saved_at": time.time(), "verdicts": self._rows}))
            tmp.replace(self._path)          # atomic
            self._dirty = False
        except OSError as exc:
            logger.warning("cannot write verdict cache %s: %s", self._path, exc)

    def prune(self, now: "float | None" = None) -> int:
        now = time.time() if now is None else now
        stale = [k for k, v in self._rows.items()
                 if now - float(v.get("ts") or 0.0) > self._ttl]
        for k in stale:
            del self._rows[k]
        return len(stale)

    def __len__(self) -> int:
        return len(self._rows)

    # ── use ───────────────────────────────────────────────────────────────────

    def split(self, candidates: list) -> "tuple[list[Implication], list]":
        """Known verdicts as implications, and the candidates still to classify."""
        known, unknown = [], []
        for c in candidates:
            row = self._rows.get(_key(c.a_id, c.b_id))
            if row is None:
                unknown.append(c)
                continue
            if row.get("narrow"):
                known.append(Implication(str(row["narrow"]), str(row["broad"]),
                                         float(row.get("confidence") or 0.0),
                                         str(row.get("evidence") or "")))
        return known, unknown

    def remember(self, candidates: list, rels: list) -> None:
        """Record one classification round: every candidate, verdict or not."""
        found = {_key(r.narrow, r.broad): r for r in rels}
        now = time.time()
        for c in candidates:
            k = _key(c.a_id, c.b_id)
            r = found.get(k)
            self._rows[k] = ({"narrow": r.narrow, "broad": r.broad,
                              "confidence": float(getattr(r, "confidence", 0.0)),
                              "evidence": str(getattr(r, "evidence", ""))[:300],
                              "ts": now}
                             if r is not None else {"narrow": None, "ts": now})
        self._dirty = True
