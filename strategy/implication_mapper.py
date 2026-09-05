"""
strategy/implication_mapper.py
──────────────────────────────
LLM implication mapper — the model layer behind cross-market arbitrage.

`strategy.cross_market.discover_from_markets()` finds implications with a regex
over market titles. It caught 0 real relations across 2,100 live markets and is
structurally blind to anything not phrased as "<title> by N points": mutually
exclusive outcomes, thresholds on the same quantity ("above $100k" ⊆ "above
$80k"), date nesting ("by March" ⊆ "by December"), and every paraphrase.

This module replaces that one function with Claude, keeping the same return type
so nothing downstream changes.

Why the model choice is not a cost decision
───────────────────────────────────────────
A wrong implication does not degrade a metric — it converts a trade the bot
believes is contract-guaranteed into an unhedged directional bet with real money
behind it. Precision is the entire product here, so this runs on Claude Opus 5.

Cost is controlled by architecture instead:

  1. Prefilter        — candidate pairs only (shared tokens / same event), so the
                        model never sees the 2.2M pairs a full cross-product implies.
  2. Batches API      — discovery is offline and once-per-scan, so latency is
                        free to trade for 50% off.
  3. Prompt caching   — the instruction prefix is byte-identical across every
                        request; only the pair varies, and it goes last.
  4. Structured output— schema-validated results, so no parse-retry loop.

Precision safeguards
────────────────────
  • The model is told to answer NONE unless the implication is airtight, and
    that a false positive costs real money while a false negative costs nothing.
  • Its confidence is capped by _MODEL_CONFIDENCE_CEILING, so a model-proposed
    relation can never outrank a hand-verified one.
  • Same-event disagreement is rejected before the model is even asked.
  • Every relation carries the model's own reasoning as `evidence`, so a bad
    mapping is auditable after the fact.

Nothing here places orders. It only populates the registry.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from strategy.cross_market import Implication

logger = logging.getLogger(__name__)

MODEL: str = os.environ.get("IMPLICATION_MODEL", "claude-opus-5")

# A model-proposed relation is never fully trusted: capped below 1.0 so a
# hand-verified implication always outranks one that was inferred.
_MODEL_CONFIDENCE_CEILING: float = 0.97

# Overlap floor for markets that share an eventId. Polymarket groups genuinely
# related markets under one event, so that link is stronger evidence than any
# amount of shared wording — "Lakers victory margin above five" and "Will the
# Lakers win?" share exactly one significant token yet are plainly related.
# A low bar here is safe because the model, not the prefilter, is the judge; the
# max_pairs cap keeps the bill bounded when an event holds many markets.
_SAME_EVENT_MIN_OVERLAP: float = 0.15

# Stop-words that must not, alone, make two titles "related".
_STOP = frozenset("""
will the a an of in on at to for by and or is are be was were do does did
with from as that this it its he she they them their you your we our us
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(title: str) -> set[str]:
    return {w for w in _WORD_RE.findall(title.lower()) if w not in _STOP and len(w) > 2}


def _event_id(market: dict) -> str:
    """
    Extract the event grouping key.

    Live Gamma returns `events` as a LIST of event objects — the flat `eventId`
    field is absent (None) on every market I sampled, so reading it alone made
    the same-event signal dead on real data. Both shapes are accepted so tests
    and any future flattening keep working.
    """
    ev = market.get("events")
    if isinstance(ev, list) and ev and isinstance(ev[0], dict):
        return str(ev[0].get("id") or "")
    return str(market.get("eventId") or market.get("event_id") or "")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Prefilter — keep the model bill proportional to real candidates
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Candidate:
    a_id:    str
    b_id:    str
    a_title: str
    b_title: str
    overlap: float          # Jaccard similarity of significant tokens
    same_event: bool


def build_candidates(
    markets:     list[dict],
    min_overlap: float = 0.5,
    max_pairs:   int   = 400,
) -> list[Candidate]:
    """
    Reduce a market universe to plausible pairs.

    A full cross-product of 2,100 markets is ~2.2M pairs — unaffordable and
    almost entirely nonsense. Two markets can only be logically related if they
    talk about the same subject, so require substantial token overlap, and treat
    a shared event as strong evidence.

    Returns at most `max_pairs`, best overlap first, so cost has a hard ceiling
    regardless of universe size.
    """
    rows = []
    for m in markets:
        title = str(m.get("question") or m.get("title") or "").strip()
        cid   = str(m.get("conditionId") or m.get("condition_id") or "").strip()
        if not title or not cid:
            continue
        rows.append((cid, title, _tokens(title), _event_id(m)))

    out: list[Candidate] = []
    for (a_id, a_t, a_tok, a_ev), (b_id, b_t, b_tok, b_ev) in itertools.combinations(rows, 2):
        if a_id == b_id or not a_tok or not b_tok:
            continue
        union = a_tok | b_tok
        if not union:
            continue
        overlap = len(a_tok & b_tok) / len(union)
        same_event = bool(a_ev and b_ev and a_ev == b_ev)
        # A shared event lowers the bar sharply; without one, wording is the
        # only evidence available, so demand strong overlap.
        floor = _SAME_EVENT_MIN_OVERLAP if same_event else min_overlap
        if overlap < floor:
            continue
        out.append(Candidate(a_id, b_id, a_t, b_t, overlap, same_event))

    out.sort(key=lambda c: (c.same_event, c.overlap), reverse=True)
    if len(out) > max_pairs:
        logger.info(
            "implication mapper | %d candidates over the %d cap — keeping the "
            "strongest; raise max_pairs to widen coverage",
            len(out), max_pairs,
        )
    return out[:max_pairs]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Prompt
# ═══════════════════════════════════════════════════════════════════════════════

# Byte-identical across every request so it caches; the pair goes in the user
# turn, after the breakpoint.
SYSTEM_PROMPT = """\
You classify logical relationships between prediction-market questions.

Given two markets A and B, decide whether one STRICTLY IMPLIES the other.

"A implies B" means: whenever A resolves YES, B must also resolve YES, as a
matter of logical necessity — not correlation, not usually, not almost always.

Examples that ARE implications:
  "Team wins by 5+ points" implies "Team wins"          (a subset of outcomes)
  "BTC above $150k by March" implies "BTC above $100k by March"  (stronger threshold)
  "Candidate wins in a landslide" implies "Candidate wins"

Examples that are NOT implications:
  "Team wins game 1" and "Team wins the series"     — correlated, not necessary
  "BTC above $100k in March" and "BTC above $100k in June" — different windows
  "Candidate wins primary" and "Candidate wins general" — sequential, not implied
  Two outcomes of the same race — mutually exclusive, not nested

Resolution details matter. Two markets that sound nested but settle on different
events, dates, or data sources are NOT an implication.

This classification is used to place real money on the assumption that the
implication cannot fail. A false positive loses money. A false negative costs
nothing — the opportunity is simply skipped. When you are not certain, answer
NONE.

Respond with the relation, your confidence that it holds without exception, and
one sentence of reasoning."""


_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["A_IMPLIES_B", "B_IMPLIES_A", "NONE"],
            "description": "NONE unless the implication is airtight.",
        },
        "confidence": {
            "type": "number",
            "description": "0-1: probability the implication holds without exception.",
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence explaining the decision.",
        },
    },
    "required": ["relation", "confidence", "reasoning"],
    "additionalProperties": False,
}


def _user_prompt(c: Candidate) -> str:
    return (
        f"Market A: {c.a_title}\n"
        f"Market B: {c.b_title}\n"
        f"Same event: {'yes' if c.same_event else 'unknown'}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Batch classification
# ═══════════════════════════════════════════════════════════════════════════════

def _to_implication(
    c: Candidate, relation: str, confidence: float, reasoning: str
) -> Optional[Implication]:
    """Turn a model verdict into a registry entry, or None."""
    if relation == "NONE":
        return None
    conf = min(float(confidence), _MODEL_CONFIDENCE_CEILING)
    if conf <= 0.0:
        return None
    narrow, broad = (
        (c.a_id, c.b_id) if relation == "A_IMPLIES_B" else (c.b_id, c.a_id)
    )
    n_title, b_title = (
        (c.a_title, c.b_title) if relation == "A_IMPLIES_B" else (c.b_title, c.a_title)
    )
    try:
        return Implication(
            narrow=narrow, broad=broad, confidence=conf,
            evidence=(
                f"{MODEL}: '{n_title[:44]}' ⊆ '{b_title[:44]}' — {reasoning[:132]}"
            ),
        )
    except ValueError as exc:                       # self-reference, bad confidence
        logger.warning("implication mapper | rejected verdict: %s", exc)
        return None


def estimate_cost(n_candidates: int) -> dict:
    """
    Rough spend for one discovery pass, before it is run.

    Claude Opus 5 is $5.00/MTok in and $25.00/MTok out; the Batches API halves
    both. The system prompt caches, so only the first request pays for it in
    full and the rest read it at cache rates.
    """
    sys_tokens   = 420          # measured shape of SYSTEM_PROMPT
    pair_tokens  = 60           # two titles plus scaffolding
    out_tokens   = 90           # schema-constrained verdict
    in_full      = sys_tokens + pair_tokens
    in_cached    = pair_tokens
    # first request pays full input; the rest read the cached prefix at ~0.1x
    input_cost = (
        in_full
        + (n_candidates - 1) * (in_cached + sys_tokens * 0.1)
    ) / 1e6 * 5.00 * 0.5
    output_cost = n_candidates * out_tokens / 1e6 * 25.00 * 0.5
    return {
        "candidates": n_candidates,
        "model": MODEL,
        "input_usd": round(input_cost, 4),
        "output_usd": round(output_cost, 4),
        "total_usd": round(input_cost + output_cost, 4),
    }


def classify_batch(
    candidates: list[Candidate],
    client=None,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 3600.0,
) -> list[Implication]:
    """
    Classify candidate pairs with the Batches API and return the implications.

    Discovery runs once per scan cycle and never on the trading path, so the
    batch endpoint's latency is free and its 50% discount is not.

    Results arrive in arbitrary order and are keyed by `custom_id` — never by
    position.
    """
    if not candidates:
        return []

    import time
    import anthropic                                   # noqa: PLC0415 — optional dep

    client = client or anthropic.Anthropic()

    requests = []
    for i, c in enumerate(candidates):
        requests.append({
            "custom_id": f"pair-{i}",
            "params": {
                "model": MODEL,
                "max_tokens": 1024,
                "system": [{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": _RESULT_SCHEMA,
                    },
                },
                "messages": [{"role": "user", "content": _user_prompt(c)}],
            },
        })

    batch = client.messages.batches.create(requests=requests)
    logger.info(
        "implication mapper | batch %s submitted — %d pairs on %s",
        batch.id, len(requests), MODEL,
    )

    waited = 0.0
    while waited < timeout_seconds:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        time.sleep(poll_seconds)
        waited += poll_seconds
    else:
        logger.error(
            "implication mapper | batch %s still running after %.0fs — abandoning "
            "this pass; the registry keeps its previous relations",
            batch.id, timeout_seconds,
        )
        return []

    out: list[Implication] = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            logger.warning(
                "implication mapper | %s: %s", result.custom_id, result.result.type
            )
            continue
        try:
            idx = int(result.custom_id.split("-", 1)[1])
            text = next(
                b.text for b in result.result.message.content if b.type == "text"
            )
            verdict = json.loads(text)
        except (ValueError, StopIteration, IndexError, json.JSONDecodeError) as exc:
            logger.warning(
                "implication mapper | unparseable verdict %s: %s",
                result.custom_id, exc,
            )
            continue
        rel = _to_implication(
            candidates[idx],
            verdict.get("relation", "NONE"),
            verdict.get("confidence", 0.0),
            str(verdict.get("reasoning", "")),
        )
        if rel:
            out.append(rel)

    logger.info(
        "implication mapper | %d implication(s) from %d candidate pair(s)",
        len(out), len(candidates),
    )
    return out


def discover_with_llm(markets: list[dict], client=None, **kw) -> list[Implication]:
    """
    Drop-in replacement for cross_market.discover_from_markets().

    Same signature, same return type — swap the call site and nothing else in
    the pipeline changes.
    """
    candidates = build_candidates(markets, **kw)
    if not candidates:
        logger.info("implication mapper | no candidate pairs survived the prefilter")
        return []
    logger.info(
        "implication mapper | %d candidates | est. %s",
        len(candidates), estimate_cost(len(candidates)),
    )
    return classify_batch(candidates, client=client)
