"""Fake `py_clob_client.client.ClobClient` for use as a patch target."""

from __future__ import annotations

import uuid
from typing import Any


class FakeApiCreds:
    def __init__(self) -> None:
        self.api_key       = "fake-api-key"
        self.api_secret    = "fake-secret"
        self.api_passphrase = "fake-passphrase"


class FakeClobClient:
    """Records every call; returns deterministic responses."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.orders_submitted: list[dict] = []
        self.cancel_all_called: int = 0
        self.signed_orders: list[Any] = []

    # ── L2 credential derivation ─────────────────────────────────────────────
    def create_or_derive_api_creds(self) -> FakeApiCreds:
        return FakeApiCreds()

    # ── Order signing ────────────────────────────────────────────────────────
    def create_market_order(self, args: Any) -> dict:
        signed = {
            "token_id": getattr(args, "token_id", None),
            "side":     getattr(args, "side", None),
            "price":    getattr(args, "price", None),
            "amount":   getattr(args, "amount", None),
            "salt":     uuid.uuid4().hex,
        }
        self.signed_orders.append(signed)
        return signed

    def create_order(self, args: Any) -> dict:
        return self.create_market_order(args)

    # ── Order submission ─────────────────────────────────────────────────────
    def post_order(self, signed: dict, _order_type: Any = None) -> dict:
        resp = {
            "order_id": f"fake-{uuid.uuid4()}",
            "status":   "matched",
            "token_id": signed.get("token_id"),
            "price":    signed.get("price"),
            "size":     signed.get("amount"),
            "side":     signed.get("side"),
        }
        self.orders_submitted.append(resp)
        return resp

    # ── Order management ─────────────────────────────────────────────────────
    def cancel_all(self) -> dict:
        self.cancel_all_called += 1
        return {"cancelled": len(self.orders_submitted)}

    def get_order(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "filled"}

    def get_orders(self) -> list[dict]:
        return list(self.orders_submitted)

    def get_order_book(self, _token_id: str) -> dict:
        return {"bids": [{"price": "0.45", "size": "100"}],
                "asks": [{"price": "0.50", "size": "100"}]}

    def get_market(self, condition_id: str) -> dict:
        return {"condition_id": condition_id, "feeRate": 0.02}
