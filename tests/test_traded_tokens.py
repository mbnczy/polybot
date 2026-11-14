"""PolyClient remembers which tokens the bot traded, for the wallet reconciler."""

from __future__ import annotations

import time

import pytest

from core.clob_client import BundleLeg, PolyClient, _trades_tokens


class _Fake(PolyClient):
    """No network, no __init__: only the recording decorator is exercised."""

    def __init__(self):                                 # noqa: D401 — no credentials
        pass

    @_trades_tokens(lambda a: [a.get("token_id")])
    async def post_order(self, token_id, side, price, size, _order_type="FOK"):
        return {"status": "matched", "order_id": "o-" + token_id}

    @_trades_tokens(lambda a: [getattr(leg, "token_id", None) for leg in a.get("legs") or []])
    async def execute_negrisk_clob_bundle(self, legs):
        return [{"order_id": f"b{i}"} for i, _ in enumerate(legs)]


@pytest.mark.asyncio
async def test_an_order_touches_its_token_and_maps_its_id():
    c = _Fake()
    t0 = time.time()
    await c.post_order(token_id="T1", side="BUY", price=0.5, size=5)
    assert c.traded_since(t0) == {"T1"}
    assert c._order_token["o-T1"] == "T1"


@pytest.mark.asyncio
async def test_bundle_legs_map_order_ids_to_their_own_tokens():
    c = _Fake()
    legs = [BundleLeg.__new__(BundleLeg), BundleLeg.__new__(BundleLeg)]
    object.__setattr__(legs[0], "token_id", "A")
    object.__setattr__(legs[1], "token_id", "B")
    await c.execute_negrisk_clob_bundle(legs)
    assert c._order_token == {"b0": "A", "b1": "B"}


def test_a_status_check_on_a_known_order_touches_its_token():
    c = _Fake()
    c._remember_orders(["X"], {"order_id": "ox"})
    t0 = time.time()
    c._touch_order("ox")
    assert c.traded_since(t0) == {"X"}
    c._touch_order("unknown")
    assert c.traded_since(t0) == {"X"}
