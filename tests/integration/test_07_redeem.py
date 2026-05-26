"""Step 7 — AutoRedeemer claims winning positions from resolved markets."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.mocks.fake_clob_ws import FakeAiohttpSession, FakeWebSocket


class _FakeResolutionResponse:
    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def raise_for_status(self): return None

    async def json(self, content_type=None):
        return self._payload


class _FakeResolutionSession:
    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    async def __aenter__(self): return self
    async def __aexit__(self, *_e): return None

    def get(self, *_a, **_kw):
        return _FakeResolutionResponse(self._payload)


@pytest.mark.asyncio
async def test_auto_redeemer_skips_when_zero_balance(
    monkeypatch, patched_web3, fake_telegram, fake_w3,
):
    """A resolved market with no held shares is marked redeemed but no tx fires."""
    from core.scanner            import FeedRegistry
    from execution.auto_redeem    import AutoRedeemer

    registry = FeedRegistry(asyncio.Queue())
    await registry.add_market("0x" + "ab" * 32, "Y", "N")

    redeemer = AutoRedeemer(registry, fake_telegram)

    resolved_payload = [{
        "conditionId": "0x" + "ab" * 32,
        "resolved":    True,
        "winner":      "yes",
    }]

    def _session_factory(*_a, **_kw):
        return _FakeResolutionSession(resolved_payload)

    monkeypatch.setattr("execution.auto_redeem.aiohttp.ClientSession", _session_factory, raising=True)

    await redeemer._scan_and_redeem()

    assert "0x" + "ab" * 32 in redeemer._redeemed

    await registry.stop_all()


@pytest.mark.asyncio
async def test_auto_redeemer_calls_redeem_when_balance_positive(
    monkeypatch, patched_web3, fake_telegram, fake_w3,
):
    monkeypatch.setattr("execution.auto_redeem._PAPER_TRADE", False)

    from core.scanner            import FeedRegistry
    from execution.auto_redeem    import AutoRedeemer, _compute_position_id, _INDEX_YES

    cid = "0x" + "ab" * 32
    registry = FeedRegistry(asyncio.Queue())
    await registry.add_market(cid, "Y", "N")

    redeemer = AutoRedeemer(registry, fake_telegram)

    # Seed YES balance in the fake CTF contract.
    pos_id = _compute_position_id(cid, _INDEX_YES)
    contract = next(iter(fake_w3.eth._contracts.values()))
    contract.functions.balances[(redeemer._wallet, pos_id)] = 5 * 10**6   # 5 shares

    payload = [{
        "conditionId": cid,
        "resolved":    True,
        "winner":      "yes",
    }]
    monkeypatch.setattr(
        "execution.auto_redeem.aiohttp.ClientSession",
        lambda *_a, **_kw: _FakeResolutionSession(payload),
        raising=True,
    )

    await redeemer._scan_and_redeem()
    # Yield to allow the executor-launched send to finish.
    await asyncio.sleep(0.1)

    call_names = [c[0] for c in contract.functions.calls]
    assert "redeemPositions" in call_names
    assert cid in redeemer._redeemed

    await registry.stop_all()
