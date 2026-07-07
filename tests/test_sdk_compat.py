"""
tests/test_sdk_compat.py
────────────────────────
Pin compatibility with the INSTALLED py-clob-client SDK.

Modern py-clob-client (≥ 0.19) renamed LimitOrderArgs → OrderArgs,
create_limit_order() → create_order(), and returns OrderBookSummary
dataclasses instead of dicts.  These tests exercise core/clob_client.py
against the real installed SDK surface (no conftest shim required), so an
SDK upgrade that breaks the live bot fails CI instead of failing at startup
on the VPS.
"""

from __future__ import annotations

import pytest

import core.clob_client as cc


def test_module_imports_against_installed_sdk():
    """core.clob_client must import with the real SDK, without any shim."""
    assert cc.LimitOrderArgs is not None
    # Whichever generation is installed, the args class must accept the
    # exact keywords the bot uses to build limit orders.
    args = cc.LimitOrderArgs(token_id="1", price=0.48, size=10.0, side="BUY")
    assert args.token_id == "1"


def test_installed_sdk_has_limit_order_signer():
    """The SDK must expose create_order (modern) or create_limit_order (legacy)."""
    from py_clob_client.client import ClobClient
    assert (
        hasattr(ClobClient, "create_order")
        or hasattr(ClobClient, "create_limit_order")
    )


def test_installed_sdk_order_types():
    """The bot posts GTC (maker) and FOK (taker) orders."""
    from py_clob_client.clob_types import OrderType
    assert hasattr(OrderType, "GTC")
    assert hasattr(OrderType, "FOK")


def test_normalise_book_from_orderbook_summary():
    """OrderBookSummary (str-typed levels) → plain float dict."""
    from py_clob_client.clob_types import OrderBookSummary, OrderSummary

    book = OrderBookSummary(
        market="0xm", asset_id="1", timestamp="0", hash="",
        bids=[OrderSummary(price="0.47", size="100")],
        asks=[OrderSummary(price="0.49", size="50"),
              OrderSummary(price="0.50", size="10")],
    )
    norm = cc._normalise_book(book)
    assert norm["bids"] == [{"price": 0.47, "size": 100.0}]
    assert norm["asks"][0] == {"price": 0.49, "size": 50.0}
    assert min(a["price"] for a in norm["asks"]) == pytest.approx(0.49)


def test_normalise_book_from_legacy_dict():
    """Legacy dict books (and junk levels) pass through normalisation."""
    norm = cc._normalise_book({
        "bids": [{"price": "0.40", "size": "5"}, {"price": None, "size": "1"}],
        "asks": [],
    })
    assert norm == {"bids": [{"price": 0.40, "size": 5.0}], "asks": []}


def test_normalise_book_handles_empty():
    assert cc._normalise_book({}) == {"bids": [], "asks": []}
    assert cc._normalise_book(None) == {"bids": [], "asks": []}


def test_create_limit_resolves_installed_method():
    """_create_limit must resolve a callable signer on the current client."""
    class _StubSDK:
        def create_order(self, args):
            return ("signed", args)

    poly = cc.PolyClient.__new__(cc.PolyClient)   # skip network-touching __init__
    poly._client = _StubSDK()
    signed = poly._create_limit("ARGS")
    assert signed == ("signed", "ARGS")
