"""
tests/test_sdk_compat.py
────────────────────────
Pin compatibility with the INSTALLED Polymarket V2 SDK (polymarket-client).

The April 2026 exchange upgrade (pUSD collateral, new contracts, new order
version) archived py-clob-client; the bot now trades through
polymarket.SecureClient.  These tests exercise the exact SDK surface
core/clob_client.py depends on, so an SDK upgrade that breaks the live bot
fails CI instead of failing at startup (or at first order) on the VPS.
"""

from __future__ import annotations

import inspect

import pytest

import core.clob_client as cc


def test_module_imports_against_installed_sdk():
    from polymarket import SecureClient  # noqa: F401
    assert cc.SecureClient is SecureClient


def test_secure_client_surface():
    """Every method PolyClient calls must exist with the expected kwargs."""
    from polymarket import SecureClient

    required_kwargs = {
        "create":              {"private_key", "wallet"},
        "create_limit_order":  {"token_id", "price", "size", "side"},
        "create_market_order": {"token_id", "side", "shares",
                                "max_price", "min_price", "order_type"},
        "post_order":          set(),          # positional signed order
        "cancel_order":        {"order_id"},
        "cancel_all":          set(),
        "get_order":           {"order_id"},
        "list_open_orders":    set(),
        "get_order_book":      {"token_id"},
        "merge_positions":     {"condition_id", "amount"},
        "redeem_positions":    {"condition_id"},
        "setup_trading_approvals": set(),
    }
    for name, expected in required_kwargs.items():
        fn = getattr(SecureClient, name, None)
        assert fn is not None, f"SecureClient.{name} missing"
        params = set(inspect.signature(fn).parameters)
        missing = expected - params
        assert not missing, f"SecureClient.{name} lost kwargs: {missing}"


def test_market_order_types_include_fok():
    """The taker/unwind path posts FOK orders."""
    from polymarket import SecureClient
    sig = inspect.signature(SecureClient.create_market_order)
    ann = str(sig.parameters["order_type"].annotation)
    assert "FOK" in ann, f"FOK order type no longer supported: {ann}"


def test_response_models_carry_expected_fields():
    """Converter functions read these fields — they must stay present."""
    from polymarket.models.clob.order_response import AcceptedOrder, RejectedOrder
    from polymarket.models.clob.account import OpenOrder
    from polymarket.models.clob.order_book import OrderBook
    from polymarket.models.clob.cancel import CancelOrdersResponse

    def fields(cls) -> set:
        return set(getattr(cls, "__dataclass_fields__", None)
                   or getattr(cls, "model_fields", {}))

    assert {"ok", "order_id", "status"} <= fields(AcceptedOrder)
    assert {"ok", "message"} <= fields(RejectedOrder)
    assert {"id", "status", "size_matched", "original_size"} <= fields(OpenOrder)
    assert {"bids", "asks"} <= fields(OrderBook)
    assert {"canceled", "not_canceled"} <= fields(CancelOrdersResponse)


def test_order_resp_normaliser_accepts_dict_and_rejected_shape():
    ok = cc._order_resp_to_dict({"status": "live", "order_id": "o1"})
    assert ok["status"] == "live" and ok["order_id"] == "o1"

    class _Rejected:
        ok = False
        code = "FOK_KILL"
        message = "not enough liquidity"

    rej = cc._order_resp_to_dict(_Rejected())
    assert rej["status"] == "error"
    assert "liquidity" in rej["error"]
    assert cc.classify_fills(rej, rej) == "none"


def test_open_order_normaliser_feeds_pair_guard():
    class _Open:
        id = "o9"
        status = "LIVE"
        size_matched = "3.5"
        original_size = "10"
        price = "0.48"
        side = "BUY"
        token_id = "t"

    d = cc._open_order_to_dict(_Open())
    assert d == {"id": "o9", "status": "LIVE", "size_matched": 3.5,
                 "original_size": 10.0, "price": 0.48, "side": "BUY",
                 "token_id": "t"}


def test_normalise_book_from_model_like_levels():
    class _Lvl:
        def __init__(self, p, s): self.price, self.size = p, s

    class _Book:
        bids = [_Lvl("0.47", "100")]
        asks = [_Lvl("0.49", "50"), _Lvl("0.50", "10")]

    norm = cc._normalise_book(_Book())
    assert norm["bids"] == [{"price": 0.47, "size": 100.0}]
    assert min(a["price"] for a in norm["asks"]) == pytest.approx(0.49)


def test_normalise_book_handles_empty():
    assert cc._normalise_book({}) == {"bids": [], "asks": []}
    assert cc._normalise_book(None) == {"bids": [], "asks": []}


def test_environment_targets_v2_stack():
    """Sanity-pin the production environment the SDK will trade against."""
    from polymarket import SecureClient
    env = inspect.signature(SecureClient.create).parameters["environment"].default
    assert env.chain_id == 137
    # pUSD collateral — NOT the pre-migration USDC.e
    assert env.collateral_token.lower() == \
        "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
    assert env.clob_url.startswith("https://clob.polymarket.com")
