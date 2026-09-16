"""
strategy/resolution_audit.py
────────────────────────────
Hold every exported implication to how its two markets actually resolved.

The model judges implications; reality settles them within a week. On
2026-09-15 it asserted "W50 Pazardzhik: Julia Adams vs Rositsa Dencheva" ⊆
"...: Completed Match". A walkover advances a player without the match being
completed, and three such pairs resolved against the trade within a day — each
winner market split 50-50, each completed-match market resolved No. No prompt
anticipates every mistake of that kind. The resolutions catch them all.

For a pair narrow ⊆ broad the trade is NO on narrow plus YES on broad, and at
resolution it pays

    payout = price of narrow's second token + price of broad's first token

with a winning token at 1, a losing one at 0 and a 50-50 split at 0.5. A true
implication pays at least 1. Less is a violation — a false implication, or two
markets whose rules part ways on a cancellation.

A violation acts at two levels:
  • the pair: its verdict becomes a refusal, so it is never exported again;
  • its template — the kind and period of each market, e.g.
    "head-to-head ⊆ completed-match" — is blocked once its violations are both
    repeated and not rare, so the same mistake on another match is refused
    before that one resolves too. One violation among hundreds of held ladders
    (an exchange mis-resolution) blocks nothing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BLOCK_MIN_VIOLATIONS = 2
BLOCK_MIN_RATE = 0.25
GRACE_S = 2 * 3600.0            # resolution lands some time after the end date
GIVE_UP_DAYS = 14.0             # unresolved this long after its end: stop asking
# Pairs not yet at their end date are checked too, a rotating batch per run. The
# walkover pairs settled days before their official end — Pazardzhik's market
# said September 22 and had resolved on the 15th. Waiting for the date would
# have let the same mistake trade for a week.
EARLY_BATCH = 1000
KEEP_RESULTS_DAYS = 30.0

_KINDS: tuple[tuple[str, str], ...] = (
    (r"completed match", "completed-match"),
    (r"exact score", "exact-score"),
    (r"both teams to score", "btts"),
    (r"\bspread\b|handicap", "spread"),
    (r"\bo/u\b|over/under", "over-under"),
    (r"#\d+|\bnumber one\b|top \d+|\bsong\b|\balbum\b", "ranking"),
    (r"\bdraw\b", "draw"),
    (r"\bvs\.?\b", "head-to-head"),
    (r"\b(above|below|reach|hit|exceed|at least|more than|less than)\b", "threshold"),
    (r"\bwin\b|\bwinner\b", "win"),
)
_KIND_RES = tuple((re.compile(p, re.I), k) for p, k in _KINDS)
_PERIOD = re.compile(r"\b(1st|2nd|first|second)\s+(half|set|quarter|period)\b|half-?time", re.I)


def kind_of(title: str) -> str:
    t = title or ""
    kind = next((k for rx, k in _KIND_RES if rx.search(t)), "other")
    p = _PERIOD.search(t)
    return f"{kind}:{p.group(2).lower() if p and p.group(2) else 'half'}" if p else kind


def template_of(narrow_title: str, broad_title: str) -> str:
    return f"{kind_of(narrow_title)} ⊆ {kind_of(broad_title)}"


def pair_key(narrow: str, broad: str) -> str:
    return f"{narrow}|{broad}"


def resolved_prices(market: "dict | None") -> "tuple[float, float] | None":
    """Final token prices (1, 0, or 0.5 each) of a resolved market, else None."""
    if not market or not market.get("closed"):
        return None
    raw = market.get("outcomePrices")
    try:
        prices = [float(x) for x in (json.loads(raw) if isinstance(raw, str) else raw)]
    except (TypeError, ValueError):
        return None
    if len(prices) != 2:
        return None

    def settle(p: float) -> "float | None":
        if p >= 0.99:
            return 1.0
        if p <= 0.01:
            return 0.0
        if abs(p - 0.5) <= 0.01:
            return 0.5
        return None

    a, b = settle(prices[0]), settle(prices[1])
    if a is None or b is None or abs(a + b - 1.0) > 1e-9:
        return None
    return a, b


def trade_payout(narrow: "tuple[float, float]", broad: "tuple[float, float]") -> float:
    """What NO on narrow plus YES on broad paid."""
    return narrow[1] + broad[0]


class ResolutionAudit:
    def __init__(self, path: "str | Path", *, block_min_violations: int = BLOCK_MIN_VIOLATIONS,
                 block_min_rate: float = BLOCK_MIN_RATE) -> None:
        self._path = Path(path)
        self._min_violations = block_min_violations
        self._min_rate = block_min_rate
        self.pending: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self.templates: dict[str, dict] = {}
        self._dirty = False

    # ── persistence ───────────────────────────────────────────────────────────

    def load(self) -> "ResolutionAudit":
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return self
        except (OSError, ValueError) as exc:
            logger.warning("resolution audit %s unreadable (%s): starting empty", self._path, exc)
            return self
        self.pending = dict(raw.get("pending") or {})
        self.results = dict(raw.get("results") or {})
        self.templates = dict(raw.get("templates") or {})
        return self

    def save(self) -> None:
        if not self._dirty:
            return
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps({"version": 1, "saved_at": time.time(),
                                       "pending": self.pending, "results": self.results,
                                       "templates": self.templates}))
            tmp.replace(self._path)
            self._dirty = False
        except OSError as exc:
            logger.warning("cannot write resolution audit %s: %s", self._path, exc)

    # ── bookkeeping ───────────────────────────────────────────────────────────

    def record(self, rows: list[dict], now: "float | None" = None) -> int:
        """Remember exported pairs so they can be checked once they resolve."""
        now = time.time() if now is None else now
        added = 0
        for r in rows:
            key = pair_key(r["narrow"], r["broad"])
            if key in self.pending or key in self.results:
                continue
            ends = [e for e in (r.get("narrow_end_ts"), r.get("broad_end_ts")) if e is not None]
            if not ends:
                continue
            self.pending[key] = {"narrow": r["narrow"], "broad": r["broad"],
                                 "narrow_title": r.get("narrow_title", ""),
                                 "broad_title": r.get("broad_title", ""),
                                 "end_ts": max(ends), "first_seen": now}
            added += 1
        self._dirty |= bool(added)
        return added

    def due(self, now: "float | None" = None) -> list[dict]:
        now = time.time() if now is None else now
        return [p for p in self.pending.values() if p["end_ts"] + GRACE_S <= now]

    def violated(self, narrow: str, broad: str) -> bool:
        return self.results.get(pair_key(narrow, broad), {}).get("verdict") == "violated"

    def is_blocked(self, narrow_title: str, broad_title: str) -> bool:
        return bool(self.templates.get(template_of(narrow_title, broad_title), {}).get("blocked"))

    def blocked_templates(self) -> list[str]:
        return sorted(t for t, s in self.templates.items() if s.get("blocked"))

    # ── the check ─────────────────────────────────────────────────────────────

    def apply(self, p: dict, narrow_market: "dict | None", broad_market: "dict | None",
              now: "float | None" = None) -> "str | None":
        """'held', 'violated', or None while either market is unresolved."""
        now = time.time() if now is None else now
        key = pair_key(p["narrow"], p["broad"])
        n, b = resolved_prices(narrow_market), resolved_prices(broad_market)
        if n is None or b is None:
            if p["end_ts"] + GIVE_UP_DAYS * 86_400.0 < now:
                self.pending.pop(key, None)
                self._dirty = True
            return None
        payout = trade_payout(n, b)
        verdict = "held" if payout >= 1.0 - 1e-9 else "violated"
        template = template_of(p["narrow_title"], p["broad_title"])
        self.results[key] = {"verdict": verdict, "payout": payout, "template": template,
                             "narrow_title": p["narrow_title"], "broad_title": p["broad_title"],
                             "ts": now}
        self.pending.pop(key, None)
        stats = self.templates.setdefault(template, {"held": 0, "violated": 0, "blocked": False,
                                                     "examples": []})
        stats[verdict] += 1
        if verdict == "violated" and len(stats["examples"]) < 3:
            stats["examples"].append(f"{p['narrow_title'][:70]} ⊆ {p['broad_title'][:70]} "
                                     f"paid {payout:.2f}")
        self._dirty = True
        return verdict

    def run(self, fetch_resolved, now: "float | None" = None) -> dict:
        """
        Check every due pair. `fetch_resolved(ids)` returns market dicts for the
        condition ids given, resolved ones among them. Returns what happened.
        """
        now = time.time() if now is None else now
        due = self.due(now)
        early = sorted((p for p in self.pending.values() if p["end_ts"] + GRACE_S > now),
                       key=lambda p: p.get("last_checked", 0.0))[:EARLY_BATCH]
        for p in early:
            p["last_checked"] = now
        due = due + early
        before = set(self.blocked_templates())
        summary = {"checked": 0, "held": 0, "violated": [], "pending": len(self.pending),
                   "newly_blocked": []}
        if due:
            ids = sorted({p["narrow"] for p in due} | {p["broad"] for p in due})
            markets = {str(m.get("conditionId")): m for m in (fetch_resolved(ids) or [])}
            for p in due:
                v = self.apply(p, markets.get(p["narrow"]), markets.get(p["broad"]), now)
                if v is None:
                    continue
                summary["checked"] += 1
                if v == "held":
                    summary["held"] += 1
                else:
                    summary["violated"].append(p)
        for t, s in self.templates.items():
            total = s["held"] + s["violated"]
            s["blocked"] = bool(total and s["violated"] >= self._min_violations
                                and s["violated"] / total >= self._min_rate)
        summary["newly_blocked"] = [t for t in self.blocked_templates() if t not in before]
        cutoff = now - KEEP_RESULTS_DAYS * 86_400.0
        for key in [k for k, r in self.results.items() if r.get("ts", now) < cutoff]:
            del self.results[key]
            self._dirty = True
        summary["pending"] = len(self.pending)
        return summary
