"""Fake `polymarket.SecureClient` (V2 SDK) for use as a patch target.

PolyClient builds its client via `SecureClient.create(private_key=…,
wallet=…)`; the conftest `patched_clob` fixture swaps the class for a factory
returning this fake.  Responses are plain dicts — core.clob_client's
normalisers (`_order_resp_to_dict`, `_open_order_to_dict`, `_normalise_book`,
`_field`) accept dicts and model objects alike.
"""

from __future__ import annotations

import uuid
from typing import Any


class _FakeHandle:
    """Mimics the SDK's transaction handle."""

    def wait(self) -> dict:
        return {"transaction_hash": f"0xfake{uuid.uuid4().hex}"}


class FakeClobClient:
    """Records every call; returns deterministic responses."""

    wallet_type = "EOA"

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.orders_submitted: list[dict] = []
        self.cancel_all_called: int = 0
        self.cancelled_ids: list[str] = []
        self.signed_orders: list[dict] = []
        self.merges: list[tuple[str, Any]] = []
        self.redeems: list[str] = []

    @classmethod
    def create(cls, **kwargs: Any) -> "FakeClobClient":
        return cls(**kwargs)

    def close(self) -> None:
        pass

    # ── Order signing ────────────────────────────────────────────────────────
    def create_limit_order(self, *, token_id: str, price: Any, size: Any,
                           side: str, **_kw: Any) -> dict:
        signed = {
            "token_id": token_id, "side": side,
            "price": float(price), "size": float(size),
            "type": "GTC", "salt": uuid.uuid4().hex,
        }
        self.signed_orders.append(signed)
        return signed

    def create_market_order(self, *, token_id: str, side: str,
                            shares: Any = None, amount: Any = None,
                            max_price: Any = None, min_price: Any = None,
                            order_type: str = "FOK", **_kw: Any) -> dict:
        signed = {
            "token_id": token_id, "side": side,
            "price": float(max_price or min_price or 0.0),
            "size": float(shares if shares is not None else (amount or 0.0)),
            "type": order_type, "salt": uuid.uuid4().hex,
        }
        self.signed_orders.append(signed)
        return signed

    # ── Order submission ─────────────────────────────────────────────────────
    def post_order(self, signed: dict) -> dict:
        resp = {
            "order_id": f"fake-{uuid.uuid4()}",
            "status":   "matched",
            "token_id": signed.get("token_id"),
            "price":    signed.get("price"),
            "size":     signed.get("size"),
            "side":     signed.get("side"),
        }
        self.orders_submitted.append(resp)
        return resp

    # ── Order management ─────────────────────────────────────────────────────
    def cancel_all(self) -> dict:
        self.cancel_all_called += 1
        return {"canceled": [o["order_id"] for o in self.orders_submitted],
                "not_canceled": {}}

    def cancel_order(self, *, order_id: str) -> dict:
        self.cancelled_ids.append(order_id)
        return {"canceled": [order_id], "not_canceled": {}}

    def get_order(self, *, order_id: str) -> dict:
        return {"id": order_id, "status": "matched",
                "size_matched": 0.0, "original_size": 0.0}

    def list_open_orders(self, **_kw: Any) -> list[dict]:
        return list(self.orders_submitted)

    def get_order_book(self, *, token_id: str) -> dict:  # noqa: ARG002
        return {"bids": [{"price": "0.45", "size": "100"}],
                "asks": [{"price": "0.50", "size": "100"}]}

    # ── Settlement ───────────────────────────────────────────────────────────
    def merge_positions(self, *, condition_id: str, amount: Any) -> _FakeHandle:
        self.merges.append((condition_id, amount))
        return _FakeHandle()

    def redeem_positions(self, *, condition_id: str, **_kw: Any) -> _FakeHandle:
        self.redeems.append(condition_id)
        return _FakeHandle()

    def setup_trading_approvals(self) -> _FakeHandle:
        return _FakeHandle()
