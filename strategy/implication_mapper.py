"""
strategy/implication_mapper.py
──────────────────────────────
LLM implication mapper — the model layer behind cross-market arbitrage.

`strategy.cross_market.discover_from_markets()` finds implications with a regex
over market titles. It caught 0 real relations across 2,100 live markets and is
structurally blind to anything not phrased "<title> by N points": threshold
nesting ("above $150k" ⊆ "above $100k"), date nesting, and every paraphrase.

This module replaces that one function with an LLM, keeping the same return type
so nothing downstream changes.

Provider-neutral by design
──────────────────────────
Everything speaks the OpenAI chat-completions protocol, so Anthropic, OpenAI and
Google models are reachable through one client by pointing `base_url` at the
right compatibility endpoint. Switch providers with an env var; no code change.

That portability costs money, and it is worth being explicit about why. The
Anthropic-native version of this module (git 3be498b) used two features the
compatibility surface does not expose:

  • the Batches API — 50% off, and discovery is offline so latency was free
  • prompt caching   — an identical instruction prefix on every request

Losing both roughly doubles the bill for the same 400-pair pass. If you settle on
one vendor and stop needing portability, that vendor's native SDK is materially
cheaper for exactly this workload.

Why precision governs the model choice
──────────────────────────────────────
A wrong implication does not degrade a metric — it converts a trade the bot
believes is contract-guaranteed into an unhedged directional bet with real money
behind it. Whichever provider you pick, pick its strong model.

Precision safeguards
────────────────────
  • The prompt states that a false positive loses money and a false negative
    costs nothing, and to answer NONE when unsure.
  • Model confidence is capped by _MODEL_CONFIDENCE_CEILING, so an inferred
    relation can never outrank a hand-verified one.
  • Every relation carries the model's own reasoning as `evidence`, so a bad
    mapping is auditable after the fact.
  • CROSS_MIN_CONFIDENCE still gates the detector downstream.

Nothing here places orders. It only populates the registry.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from strategy.cross_market import Implication

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Providers — one protocol, three vendors
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Provider:
    """
    A chat-completions endpoint reachable with the `openai` client.

    `strict_schema` records whether the endpoint honours json_schema response
    formats. Where it does not, the prompt still demands bare JSON and the parser
    tolerates a fence — the difference is a guarantee versus a convention.
    """
    name:          str
    base_url:      Optional[str]      # None = the openai default
    api_key_env:   str
    default_model: str
    strict_schema: bool


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        name="anthropic",
        # Anthropic's OpenAI-compatibility endpoint. This surface does NOT expose
        # prompt caching or the Batches API — see the module docstring.
        base_url="https://api.anthropic.com/v1/",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-opus-5",
        strict_schema=False,
    ),
    "openai": Provider(
        name="openai",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-5",
        strict_schema=True,
    ),
    "google": Provider(
        name="google",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-pro",
        strict_schema=True,
    ),
}

# A fully env-configured endpoint. Anything that speaks the OpenAI
# chat-completions protocol works — a self-hosted vLLM/Ollama/TGI gateway, an
# internal proxy, or a vendor not listed above. Selected with
# IMPLICATION_PROVIDER=custom.
#
# Configure in .env:
#   IMPLICATION_PROVIDER=custom
#   IMPLICATION_BASE_URL=http://host/v1
#   IMPLICATION_API_KEY=...
#   IMPLICATION_MODEL=vendor/model-name
#   IMPLICATION_STRICT_SCHEMA=false   # most self-hosted servers ignore json_schema
PROVIDERS["custom"] = Provider(
    name="custom",
    base_url=os.environ.get("IMPLICATION_BASE_URL", "").strip() or None,
    api_key_env="IMPLICATION_API_KEY",
    default_model=os.environ.get("IMPLICATION_MODEL", "").strip(),
    strict_schema=os.environ.get(
        "IMPLICATION_STRICT_SCHEMA", "false"
    ).strip().lower() in ("1", "true", "yes", "on"),
)

PROVIDER: str = os.environ.get("IMPLICATION_PROVIDER", "anthropic").strip().lower()
MODEL:    str = os.environ.get("IMPLICATION_MODEL", "").strip()


def resolve_provider(name: str = "") -> Provider:
    key = (name or PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ValueError(
            f"unknown provider {key!r}; expected one of {sorted(PROVIDERS)}"
        )
    p = PROVIDERS[key]
    if key == "custom" and not p.base_url:
        raise RuntimeError(
            "IMPLICATION_PROVIDER=custom requires IMPLICATION_BASE_URL "
            "(e.g. http://host/v1)"
        )
    return p


def resolve_model(provider: Provider, model: str = "") -> str:
    """
    Resolve the model name: explicit argument > env > the provider's default.

    The env override applies ONLY to the provider that IMPLICATION_PROVIDER
    selects. The env vars describe one coherent configuration, so a model name
    set for a self-hosted endpoint must not leak into a hosted vendor — asking
    OpenAI for 'sztaki_pipeline-gemma4:31b' because both were configured in the
    same .env would be a confusing failure at best.
    """
    env_model = MODEL if provider.name == PROVIDER else ""
    resolved  = model or env_model or provider.default_model
    if not resolved:
        raise RuntimeError(
            f"no model for provider {provider.name!r} — set IMPLICATION_MODEL"
        )
    return resolved


def build_client(provider: Provider, api_key: str = ""):
    """
    Construct an `openai` client aimed at `provider`.

    Imported lazily so the trading bot runs without the dependency installed —
    discovery is an offline concern and must never become a hot-path requirement.
    """
    from openai import OpenAI                       # noqa: PLC0415 — optional dep

    key = api_key or os.environ.get(provider.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"{provider.api_key_env} is not set — cannot reach {provider.name}"
        )
    kwargs: dict = {"api_key": key}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    return OpenAI(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Tunables
# ═══════════════════════════════════════════════════════════════════════════════

# A model-proposed relation is never fully trusted: capped below 1.0 so a
# hand-verified implication always outranks one that was inferred.
_MODEL_CONFIDENCE_CEILING: float = 0.97

# Overlap floor for markets sharing a Gamma event. Polymarket groups genuinely
# related markets under one event, so that link beats any amount of shared
# wording — "Lakers victory margin above five" and "Will the Lakers win?" share
# one significant token yet are plainly related. Safe because the model, not the
# prefilter, is the judge; max_pairs keeps the bill bounded.
_SAME_EVENT_MIN_OVERLAP: float = 0.15

_STOP = frozenset("""
will the a an of in on at to for by and or is are be was were do does did
with from as that this it its he she they them their you your we our us
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(title: str) -> set[str]:
    """
    Significant words in a title.

    Deliberately no stemming or lemmatisation. Measured over 2,100 live markets,
    a suffix stemmer moved the candidate count from 39,595 to 39,660 — +0.2%,
    entirely swallowed by max_pairs. It would be ceremony, not recall.
    """
    return {w for w in _WORD_RE.findall(title.lower()) if w not in _STOP and len(w) > 2}


def _event_id(market: dict) -> str:
    """
    Extract the event grouping key.

    Live Gamma returns `events` as a LIST of event objects and leaves the flat
    `eventId` null, so reading eventId alone left this signal dead on real data.
    Both shapes are accepted.
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
    a_id:       str
    b_id:       str
    a_title:    str
    b_title:    str
    overlap:    float            # Jaccard similarity of significant tokens
    same_event: bool
    event_id:   str = ""


def build_candidates(
    markets:       list[dict],
    min_overlap:   float = 0.5,
    max_pairs:     int   = 400,
    max_per_event: int   = 20,
) -> list[Candidate]:
    """
    Reduce a market universe to plausible pairs.

    A full cross-product of 2,100 markets is ~2.2M pairs. Two markets can only be
    logically related if they discuss the same subject, so require token overlap,
    and treat a shared event as the stronger signal.

    `max_per_event` matters more than it looks. Measured on the live universe,
    39,595 pairs clear the 0.5 threshold and ~19,000 of those have IDENTICAL
    token sets — Polymarket generates dozens of near-identical titles per event
    ("Balance of Power: D Senate, D House", "...D Senate, R House", ...). Without
    a per-event cap one 40-outcome election fills every slot with mutually
    exclusive pairs that all come back NONE, and the rest of the universe is
    never examined.
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
        floor = _SAME_EVENT_MIN_OVERLAP if same_event else min_overlap
        if overlap < floor:
            continue
        out.append(Candidate(
            a_id, b_id, a_t, b_t, overlap, same_event,
            a_ev if same_event else "",
        ))

    out.sort(key=lambda c: (c.same_event, c.overlap), reverse=True)

    if max_per_event > 0:
        seen: dict[str, int] = {}
        capped: list[Candidate] = []
        for c in out:
            if c.event_id:
                n = seen.get(c.event_id, 0)
                if n >= max_per_event:
                    continue
                seen[c.event_id] = n + 1
            capped.append(c)
        if len(capped) < len(out):
            logger.info(
                "implication mapper | per-event cap dropped %d pair(s) so no "
                "single event monopolises the budget", len(out) - len(capped),
            )
        out = capped

    if len(out) > max_pairs:
        logger.info(
            "implication mapper | %d candidates over the %d cap — keeping the "
            "strongest; raise max_pairs to widen coverage", len(out), max_pairs,
        )
    return out[:max_pairs]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Prompt
# ═══════════════════════════════════════════════════════════════════════════════

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
  Two similarly-named but DIFFERENT subjects (a person and their namesake)

Resolution details matter. Two markets that sound nested but settle on different
events, dates, or data sources are NOT an implication.

This classification is used to place real money on the assumption that the
implication cannot fail. A false positive loses money. A false negative costs
nothing — the opportunity is simply skipped. When you are not certain, answer
NONE.

Reply with ONLY a JSON object, no prose and no code fence:
{"relation": "A_IMPLIES_B" | "B_IMPLIES_A" | "NONE",
 "confidence": <number 0-1>,
 "reasoning": "<one sentence>"}"""


_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["A_IMPLIES_B", "B_IMPLIES_A", "NONE"],
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
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


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_verdict(text: str) -> Optional[dict]:
    """
    Parse a model reply into a verdict dict, or None.

    Providers differ in how strictly they honour a response format, so tolerate a
    code fence or surrounding prose rather than discarding an otherwise good
    answer. Returns None when nothing parseable is present — the caller treats
    that as "no relation", which is the safe direction.
    """
    if not text:
        return None
    raw = text.strip()
    fenced = _FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if i == -1 or j <= i:
            return None
        raw = raw[i:j + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Classification
# ═══════════════════════════════════════════════════════════════════════════════

def _to_implication(
    c: Candidate, relation: str, confidence: float, reasoning: str, model: str
) -> Optional[Implication]:
    """Turn a model verdict into a registry entry, or None."""
    if relation not in ("A_IMPLIES_B", "B_IMPLIES_A"):
        return None
    try:
        conf = min(float(confidence), _MODEL_CONFIDENCE_CEILING)
    except (TypeError, ValueError):
        return None
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
            evidence=f"{model}: '{n_title[:44]}' ⊆ '{b_title[:44]}' — {reasoning[:132]}",
        )
    except ValueError as exc:                        # self-reference / bad confidence
        logger.warning("implication mapper | rejected verdict: %s", exc)
        return None


def classify_one(
    client, provider: Provider, model: str, c: Candidate
) -> Optional[Implication]:
    """Classify a single pair. Returns None on refusal, error, or NONE."""
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _user_prompt(c)},
        ],
        "max_completion_tokens": 512,
    }
    if provider.strict_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "implication",
                "strict": True,
                "schema": _RESULT_SCHEMA,
            },
        }
    try:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
    except Exception as exc:                         # noqa: BLE001 — provider-agnostic
        logger.warning("implication mapper | request failed: %s", exc)
        return None

    verdict = parse_verdict(text)
    if verdict is None:
        logger.warning(
            "implication mapper | unparseable reply for %s / %s: %r",
            c.a_id[:10], c.b_id[:10], text[:120],
        )
        return None
    return _to_implication(
        c,
        str(verdict.get("relation", "NONE")),
        verdict.get("confidence", 0.0),
        str(verdict.get("reasoning", "")),
        model,
    )


def classify_candidates(
    candidates:  list[Candidate],
    provider:    Optional[Provider] = None,
    model:       str = "",
    client=None,
    concurrency: int = 8,
) -> list[Implication]:
    """
    Classify candidate pairs concurrently.

    The Anthropic Batches API (50% off) is not reachable through the
    compatibility surface, so throughput comes from concurrency instead. Bounded,
    because these endpoints rate-limit and discovery is offline — there is no
    deadline worth tripping a 429 for.
    """
    if not candidates:
        return []

    from concurrent.futures import ThreadPoolExecutor    # noqa: PLC0415

    prov = provider or resolve_provider()
    mdl  = resolve_model(prov, model)
    cli  = client or build_client(prov)

    logger.info(
        "implication mapper | classifying %d pair(s) on %s/%s",
        len(candidates), prov.name, mdl,
    )

    out: list[Implication] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for rel in pool.map(lambda c: classify_one(cli, prov, mdl, c), candidates):
            if rel:
                out.append(rel)

    logger.info(
        "implication mapper | %d implication(s) from %d candidate pair(s)",
        len(out), len(candidates),
    )
    return out


def discover_with_llm(
    markets:     list[dict],
    provider:    str = "",
    model:       str = "",
    client=None,
    concurrency: int = 8,
    **prefilter_kw,
) -> list[Implication]:
    """
    Drop-in replacement for cross_market.discover_from_markets().

    Same return type, so the call site is a one-line swap.
    """
    prov  = resolve_provider(provider)
    cands = build_candidates(markets, **prefilter_kw)
    if not cands:
        logger.info("implication mapper | no candidate pairs survived the prefilter")
        return []
    return classify_candidates(
        cands, provider=prov, model=model, client=client, concurrency=concurrency,
    )
