"""
LLM implication mapper — everything testable without spending money.

The API call is stubbed rather than mocked into a fake "it works" test. What is
worth guarding is the logic around it: which pairs reach the model, how a verdict
becomes a registry entry, the caps that stop a confident-but-wrong model from
being trusted, and the reply parsing — which has to survive three providers with
different ideas about whether "return JSON" means bare JSON.
"""

from __future__ import annotations

import pytest

from strategy.implication_mapper import (
    Candidate,
    PROVIDERS,
    Provider,
    _MODEL_CONFIDENCE_CEILING,
    _to_implication,
    build_candidates,
    classify_candidates,
    classify_one,
    parse_verdict,
    resolve_model,
    resolve_provider,
)


def _mkt(cid, q, event=None):
    d = {"conditionId": cid, "question": q}
    if event:
        d["events"] = [{"id": event}]
    return d


# ── providers ─────────────────────────────────────────────────────────────────

def test_all_vendors_are_reachable():
    """Three hosted vendors plus a fully env-configured custom endpoint."""
    assert set(PROVIDERS) == {"anthropic", "openai", "google", "custom"}


def test_anthropic_and_google_are_pointed_at_compat_endpoints():
    assert "api.anthropic.com" in PROVIDERS["anthropic"].base_url
    assert "generativelanguage" in PROVIDERS["google"].base_url
    # OpenAI itself needs no override
    assert PROVIDERS["openai"].base_url is None


def test_unknown_provider_is_rejected_loudly():
    with pytest.raises(ValueError):
        resolve_provider("bedrock")


def test_explicit_model_beats_the_provider_default():
    p = PROVIDERS["openai"]
    assert resolve_model(p, "gpt-5-mini") == "gpt-5-mini"
    assert resolve_model(p) == p.default_model


# ── prefilter ─────────────────────────────────────────────────────────────────

def test_unrelated_markets_never_reach_the_model():
    markets = [
        _mkt("0xa", "Will it rain in Paris tomorrow?"),
        _mkt("0xb", "Will the Lakers win the championship?"),
    ]
    assert build_candidates(markets) == []


def test_related_markets_become_candidates():
    markets = [
        _mkt("0xa", "Will the Lakers win the game?"),
        _mkt("0xb", "Will the Lakers win the game by 5 points?"),
    ]
    assert len(build_candidates(markets)) == 1


def test_shared_event_lowers_the_overlap_bar():
    """An explicit event link is stronger evidence than wording alone."""
    markets = [
        _mkt("0xa", "Lakers victory margin above five", event="game-42"),
        _mkt("0xb", "Will the Lakers win?",             event="game-42"),
    ]
    assert len(build_candidates(markets)) == 1


def test_event_id_read_from_the_live_list_shape():
    """
    Gamma returns `events` as a LIST and leaves flat eventId null. Reading only
    eventId left the same-event signal dead on real data.
    """
    markets = [
        _mkt("0xa", "Lakers victory margin above five", event="e1"),
        _mkt("0xb", "Will the Lakers win?",             event="e1"),
    ]
    assert build_candidates(markets)[0].same_event is True


def test_stopwords_alone_do_not_relate_two_markets():
    markets = [
        _mkt("0xa", "Will the price of tea in China rise?"),
        _mkt("0xb", "Will the winner of the race be disqualified?"),
    ]
    assert build_candidates(markets) == []


def test_per_event_cap_stops_one_event_eating_the_budget():
    """
    ~19,000 of 39,595 live candidate pairs have identical token sets, nearly all
    from a handful of many-outcome events. Without this cap a single election
    fills every slot with mutually exclusive pairs that all return NONE.
    """
    markets = [
        _mkt(f"0x{i}", f"2026 Balance of Power outcome {i}", event="bop")
        for i in range(12)
    ]
    cands = build_candidates(markets, max_per_event=5)
    assert len([c for c in cands if c.event_id == "bop"]) == 5


def test_global_cap_is_enforced():
    markets = [_mkt(f"0x{i}", f"Will team alpha beta win match {i}?") for i in range(60)]
    assert len(build_candidates(markets, max_pairs=25, max_per_event=0)) == 25


# ── reply parsing: three providers, three habits ──────────────────────────────

def test_parses_bare_json():
    v = parse_verdict('{"relation":"NONE","confidence":0.9,"reasoning":"x"}')
    assert v["relation"] == "NONE"


def test_parses_fenced_json():
    """Models that ignore 'no code fence' must not cost us the answer."""
    v = parse_verdict('```json\n{"relation":"A_IMPLIES_B","confidence":0.9,"reasoning":"x"}\n```')
    assert v["relation"] == "A_IMPLIES_B"


def test_parses_json_with_surrounding_prose():
    v = parse_verdict('Here is my answer:\n{"relation":"NONE","confidence":0.5,"reasoning":"x"}\nHope that helps.')
    assert v["relation"] == "NONE"


@pytest.mark.parametrize("junk", ["", "   ", "no json at all", "{unclosed", "[1,2,3]"])
def test_unparseable_replies_return_none(junk):
    assert parse_verdict(junk) is None


# ── verdict -> registry entry ─────────────────────────────────────────────────

def _cand():
    return Candidate("0xnarrow", "0xbroad", "A wins by 5+", "A wins", 0.9, True, "e1")


def test_none_verdict_produces_no_relation():
    assert _to_implication(_cand(), "NONE", 0.99, "unrelated", "m") is None


def test_a_implies_b_maps_the_right_way_round():
    rel = _to_implication(_cand(), "A_IMPLIES_B", 0.95, "margin is a subset", "m")
    assert rel.narrow == "0xnarrow" and rel.broad == "0xbroad"


def test_b_implies_a_reverses_the_direction():
    rel = _to_implication(_cand(), "B_IMPLIES_A", 0.95, "reversed", "m")
    assert rel.narrow == "0xbroad" and rel.broad == "0xnarrow"


def test_model_confidence_is_capped_below_certainty():
    """A model claiming 1.0 must still rank below a hand-verified relation."""
    rel = _to_implication(_cand(), "A_IMPLIES_B", 1.0, "certain", "m")
    assert rel.confidence == pytest.approx(_MODEL_CONFIDENCE_CEILING)
    assert rel.confidence < 1.0


def test_non_numeric_confidence_is_rejected():
    assert _to_implication(_cand(), "A_IMPLIES_B", "very sure", "x", "m") is None


def test_reasoning_and_model_are_retained_for_audit():
    rel = _to_implication(_cand(), "A_IMPLIES_B", 0.9, "winning by 5 implies winning", "gpt-5")
    assert "winning by 5 implies winning" in rel.evidence
    assert "gpt-5" in rel.evidence


def test_self_referential_verdict_is_rejected():
    bad = Candidate("0xsame", "0xsame", "A", "A", 1.0, True, "")
    assert _to_implication(bad, "A_IMPLIES_B", 0.95, "same market", "m") is None


# ── classification plumbing, with a stub client ───────────────────────────────

class _StubClient:
    """Minimal stand-in for the openai client. Records what it was asked."""

    def __init__(self, reply: str = '{"relation":"NONE","confidence":0.9,"reasoning":"x"}',
                 raise_exc: Exception | None = None):
        self.reply = reply
        self.raise_exc = raise_exc
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                if outer.raise_exc:
                    raise outer.raise_exc
                msg = type("M", (), {"content": outer.reply})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_strict_schema_provider_sends_a_response_format():
    cli = _StubClient()
    classify_one(cli, PROVIDERS["openai"], "gpt-5", _cand())
    assert "response_format" in cli.calls[0]


def test_non_strict_provider_omits_response_format():
    """Anthropic's compat surface does not honour json_schema — do not send it."""
    cli = _StubClient()
    classify_one(cli, PROVIDERS["anthropic"], "claude-opus-5", _cand())
    assert "response_format" not in cli.calls[0]


def test_request_failure_yields_no_relation_rather_than_raising():
    """A provider outage must degrade to 'found nothing', never crash discovery."""
    cli = _StubClient(raise_exc=RuntimeError("503 upstream"))
    assert classify_one(cli, PROVIDERS["openai"], "gpt-5", _cand()) is None


def test_classify_candidates_collects_only_real_relations():
    cli = _StubClient('{"relation":"A_IMPLIES_B","confidence":0.9,"reasoning":"subset"}')
    cands = [_cand(), _cand()]
    out = classify_candidates(cands, provider=PROVIDERS["openai"], model="gpt-5", client=cli)
    assert len(out) == 2
    assert len(cli.calls) == 2


def test_empty_candidate_list_makes_no_requests():
    cli = _StubClient()
    assert classify_candidates([], provider=PROVIDERS["openai"], model="gpt-5", client=cli) == []
    assert cli.calls == []


# ── custom provider: any chat-completions endpoint, configured from .env ──────

def test_custom_provider_is_available():
    assert "custom" in PROVIDERS


def test_custom_provider_requires_a_base_url(monkeypatch):
    """
    Selecting `custom` without IMPLICATION_BASE_URL must fail loudly at resolve
    time, not silently fall back to the openai default endpoint with someone
    else's key.
    """
    import strategy.implication_mapper as im
    monkeypatch.setitem(
        im.PROVIDERS, "custom",
        Provider("custom", None, "IMPLICATION_API_KEY", "m", False),
    )
    with pytest.raises(RuntimeError, match="IMPLICATION_BASE_URL"):
        resolve_provider("custom")


def test_missing_model_is_reported_rather_than_guessed():
    """A self-hosted endpoint has no sensible default model name."""
    p = Provider("custom", "http://host/v1", "IMPLICATION_API_KEY", "", False)
    import strategy.implication_mapper as im
    prev = im.MODEL
    im.MODEL = ""
    try:
        with pytest.raises(RuntimeError, match="IMPLICATION_MODEL"):
            resolve_model(p)
    finally:
        im.MODEL = prev


def test_self_hosted_endpoints_default_to_lenient_schema():
    """
    Most self-hosted servers ignore json_schema response formats; sending one
    can error outright. Default off, and rely on the fence-tolerant parser.
    """
    p = Provider("custom", "http://host/v1", "IMPLICATION_API_KEY", "m", False)
    cli = _StubClient()
    classify_one(cli, p, "m", _cand())
    assert "response_format" not in cli.calls[0]


def test_build_client_refuses_without_a_key():
    p = Provider("custom", "http://host/v1", "IMPLICATION_NO_SUCH_KEY", "m", False)
    import strategy.implication_mapper as im
    with pytest.raises(RuntimeError, match="IMPLICATION_NO_SUCH_KEY"):
        im.build_client(p)


def test_env_model_does_not_leak_into_another_provider(monkeypatch):
    """
    Regression: a model name configured for the env-selected provider must not
    be sent to a different one. Setting IMPLICATION_MODEL for a self-hosted
    endpoint and then asking for openai previously sent the self-hosted model
    name to OpenAI — caught only because a test passed alone and failed in the
    suite.
    """
    import strategy.implication_mapper as im
    monkeypatch.setattr(im, "PROVIDER", "custom")
    monkeypatch.setattr(im, "MODEL", "sztaki_pipeline-gemma4:31b")
    # the env-selected provider gets the env model
    custom = Provider("custom", "http://host/v1", "IMPLICATION_API_KEY", "", False)
    assert im.resolve_model(custom) == "sztaki_pipeline-gemma4:31b"
    # a DIFFERENT provider keeps its own default
    assert im.resolve_model(im.PROVIDERS["openai"]) == "gpt-5"
