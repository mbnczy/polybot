"""
tests/test_pair_guard.py
────────────────────────
MakerPairGuard — one-leg protection for the GTC maker arb path.

Scenarios covered:
  1. Both legs fill while resting            → P&L booked, inventory handoff.
  2. One leg fills, other never does         → lagging order cancelled, naked
     leg completed as taker when profitable.
  3. One leg fills, completion unprofitable  → naked leg unwound at market.
  4. Neither leg fills before TTL            → both orders cancelled,
     reservation released, nothing booked.
  5. Vanished order without our cancel       → treated as filled (safe side).
"""

from __future__ import annotations

import asyncio

import pytest

from execution.pair_guard import MakerPairGuard
from strategy.arbitrage import ArbSignal


# ═══════════════════════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════════════════════

class FakeGuardClient:
    """Scriptable stand-in for PolyClient (only the surface the guard uses)."""

    def __init__(self) -> None:
        # order_id → order dict returned by get_order_status (None = vanished)
        self.orders: dict[str, dict | None] = {}
        self.cancelled: list[str] = []
        self.unwound: list[tuple[str, float]] = []
        self.taker_buys: list[tuple[str, float, float]] = []
        self.best_asks: dict[str, float] = {}
        # order_id -> shares the TRADE FEED attributes to that order. This is
        # per-order, unlike a wallet balance, which is exactly the distinction
        # the guard depends on.
        self.order_fills: dict[str, float] = {}
        # token_id -> wallet balance. Deliberately still here and deliberately
        # NOT consulted for fill decisions: the incident was caused by reading
        # this number as if it were a per-order fill.
        self.share_balances: dict[str, float] = {}
        self.taker_fill_status: str = "matched"
        # Price the naked leg sells at on unwind (proceeds = size × this).
        self.unwind_sell_price: float = 0.40

    async def share_balance(self, token_id: str):
        return self.share_balances.get(token_id, 0.0)

    async def order_filled_size(self, order_id: str, token_id=None):
        return self.order_fills.get(order_id, 0.0)

    async def get_order_status(self, order_id: str):
        return self.orders.get(order_id)

    async def cancel_order(self, order_id: str) -> dict:
        self.cancelled.append(order_id)
        self.orders[order_id] = {"status": "canceled",
                                 "size_matched": self._matched(order_id)}
        return {"canceled": order_id}

    def _matched(self, order_id: str) -> float:
        cur = self.orders.get(order_id)
        if isinstance(cur, dict):
            return float(cur.get("size_matched", 0.0))
        return 0.0

    async def get_orderbook(self, token_id: str) -> dict:
        ask = self.best_asks.get(token_id)
        return {"asks": [{"price": ask}]} if ask is not None else {"asks": []}

    async def post_order(self, token_id: str, side: str, price: float,
                         size: float) -> dict:
        self.taker_buys.append((token_id, price, size))
        return {"status": self.taker_fill_status, "order_id": "taker-1"}

    async def unwind_leg(self, token_id: str, size: float,
                         price: float = 0.0) -> dict:
        self.unwound.append((token_id, size))
        proceeds = round(size * self.unwind_sell_price, 6)
        return {"status": "matched", "making_amount": size,
                "taking_amount": proceeds}


class FakeBreaker:
    def __init__(self) -> None:
        self.fills: list[float] = []
        self.releases: int = 0

    def on_fill(self, pnl: float) -> None:
        self.fills.append(pnl)

    def release_open(self) -> None:
        self.releases += 1


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.critical: list[str] = []

    async def notify(self, msg: str) -> None:
        self.messages.append(msg)

    async def send_critical_error(self, msg: str) -> None:
        self.critical.append(msg)


class FakeInventory:
    def __init__(self) -> None:
        self.paired: list[str] = []
        self.merged: list[tuple[str, float]] = []

    def register_paired_fill(self, signal) -> None:
        self.paired.append(signal.condition_id)

    async def merge_complementary_set(self, condition_id: str, n_shares: float,
                                      *, is_negrisk: bool = False):
        self.merged.append((condition_id, n_shares))
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _maker_signal(n_shares: float = 10.0) -> ArbSignal:
    return ArbSignal(
        condition_id="0xcond",
        yes_token_id="tok-yes",
        no_token_id="tok-no",
        yes_ask=0.48, no_ask=0.50,
        combined_cost=0.958,
        fee_rate=0.0, fee_cost=0.0,
        net_edge=0.042,
        yes_size=n_shares, no_size=n_shares,
        yes_bid=0.479, no_bid=0.499,
        maker_rebate=0.01, maker_net_edge=0.0518,
    )


def _resting_resp(order_id: str) -> dict:
    return {"status": "live", "order_id": order_id}


def _build_guard(client, inventory=None, hedge_timeout_s=0.0, order_ttl_s=60.0):
    breaker  = FakeBreaker()
    notifier = FakeNotifier()
    guard = MakerPairGuard(
        client, breaker, notifier,
        inventory=inventory,
        poll_interval_s=0.05,
        hedge_timeout_s=hedge_timeout_s,
        order_ttl_s=order_ttl_s,
        taker_fee_est=0.02,
    )
    return guard, breaker, notifier


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_both_legs_fill_books_pnl_and_registers_inventory():
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "live", "size_matched": 0.0}
    client.orders["on"] = {"status": "live", "size_matched": 0.0}
    inventory = FakeInventory()
    guard, breaker, _ = _build_guard(client, inventory)

    sig = _maker_signal(10.0)
    guard.watch_pair(sig, 10.0, _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()                      # both still resting
    assert guard.watched_count == 1
    assert breaker.fills == [] and breaker.releases == 0

    client.orders["oy"] = {"status": "matched", "size_matched": 10.0}
    client.orders["on"] = {"status": "matched", "size_matched": 10.0}
    await guard.poll_once()

    assert guard.watched_count == 0
    assert len(breaker.fills) == 1
    # pnl = 10 × (1 − (0.479 + 0.499)) = 0.22
    assert breaker.fills[0] == pytest.approx(0.22, abs=1e-6)
    assert breaker.releases == 0
    assert inventory.paired == ["0xcond"]


@pytest.mark.asyncio
async def test_one_leg_fill_completes_as_taker_when_profitable():
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "matched", "size_matched": 10.0}
    client.orders["on"] = {"status": "live",    "size_matched": 0.0}
    # NO ask so that 0.479 + 0.50 × 1.02 = 0.989 < 1.0 → completion profitable
    client.best_asks["tok-no"] = 0.50
    guard, breaker, notifier = _build_guard(client, hedge_timeout_s=0.0)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()   # imbalance detected, hedge window (0 s) starts
    await guard.poll_once()   # window elapsed → cancel NO leg → finalize

    assert "on" in client.cancelled
    assert client.taker_buys == [("tok-no", 0.50, 10.0)]
    assert client.unwound == []
    assert len(breaker.fills) == 1
    # completed pairs: 10 × (1 − (0.479 + 0.50 × 1.02)) = 0.11
    assert breaker.fills[0] == pytest.approx(0.11, abs=1e-6)
    assert breaker.releases == 0
    assert guard.watched_count == 0


@pytest.mark.asyncio
async def test_one_leg_fill_unwinds_when_completion_unprofitable():
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "matched", "size_matched": 10.0}
    client.orders["on"] = {"status": "live",    "size_matched": 0.0}
    # Price moved away: 0.479 + 0.60 × 1.02 = 1.091 > 1.0 → must unwind
    client.best_asks["tok-no"] = 0.60
    guard, breaker, notifier = _build_guard(client, hedge_timeout_s=0.0)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()
    await guard.poll_once()

    assert "on" in client.cancelled
    assert client.taker_buys == []
    assert client.unwound == [("tok-yes", 10.0)]
    # The unwind's REALISED loss must be booked (not silently dropped):
    # proceeds 10×0.40 − cost 10×0.479 = 4.0 − 4.79 = −0.79
    assert len(breaker.fills) == 1
    assert breaker.fills[0] == pytest.approx(-0.79, abs=1e-6)
    assert breaker.releases == 0          # on_fill released the reservation
    assert guard.watched_count == 0
    assert any("unwound" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_no_fills_ttl_cancels_both_and_releases():
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "live", "size_matched": 0.0}
    client.orders["on"] = {"status": "live", "size_matched": 0.0}
    guard, breaker, _ = _build_guard(client, order_ttl_s=0.0)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()   # TTL (0 s) already expired → cancel both

    assert set(client.cancelled) == {"oy", "on"}
    assert client.unwound == [] and client.taker_buys == []
    assert breaker.fills == []
    assert breaker.releases == 1
    assert guard.watched_count == 0


@pytest.mark.asyncio
async def test_vanished_order_filled_when_trade_feed_attributes_shares():
    """Untracked order + attributed fills => genuinely filled, P&L booked."""
    client = FakeGuardClient()
    client.orders["oy"] = None    # vanished, we never cancelled it
    client.orders["on"] = {"status": "matched", "size_matched": 10.0}
    client.order_fills["oy"] = 10.0           # the trade feed confirms the fill
    inventory = FakeInventory()
    guard, breaker, _ = _build_guard(client, inventory)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()

    assert len(breaker.fills) == 1
    assert breaker.fills[0] == pytest.approx(0.22, abs=1e-6)
    assert inventory.paired == ["0xcond"]
    assert guard.watched_count == 0


@pytest.mark.asyncio
async def test_vanished_order_not_filled_when_wallet_is_empty():
    """
    Untracked order + NO shares on-chain => cancelled, not filled.

    Regression for 2026-09-04: assuming "filled" fabricated P&L and triggered an
    unwind of shares that did not exist.
    """
    client = FakeGuardClient()
    client.orders["oy"] = None
    client.orders["on"] = {"status": "matched", "size_matched": 10.0}
    client.order_fills = {}                   # no fills attributed to it
    inventory = FakeInventory()
    guard, breaker, _ = _build_guard(client, inventory)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()

    # The YES leg must NOT be counted as filled, so no phantom paired P&L.
    assert inventory.paired == []


@pytest.mark.asyncio
async def test_partial_pair_merges_paired_portion_only():
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "matched",  "size_matched": 10.0}
    client.orders["on"] = {"status": "canceled", "size_matched": 4.0}
    client.best_asks["tok-no"] = 0.99   # completion far too expensive
    inventory = FakeInventory()
    guard, breaker, _ = _build_guard(client, inventory, hedge_timeout_s=0.0)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()   # both legs closed, 4 paired + 6 naked YES
    await asyncio.sleep(0)    # let the ensure_future merge task run

    assert client.unwound == [("tok-yes", 6.0)]
    assert len(breaker.fills) == 1
    # paired pnl = 4 × (1 − 0.978) = 0.088; naked unwind realised =
    # 6×0.40 − 6×0.479 = 2.4 − 2.874 = −0.474; total = −0.386
    assert breaker.fills[0] == pytest.approx(-0.386, abs=1e-6)
    assert inventory.paired == []                    # not a full pair
    assert inventory.merged == [("0xcond", 4.0)]     # merge only what's paired


@pytest.mark.asyncio
async def test_double_watch_same_condition_does_not_orphan_first_pair():
    """
    Two watches on the SAME market must coexist — keying by condition_id
    alone silently overwrote (orphaned) the first pair's resting orders.
    """
    client = FakeGuardClient()
    client.orders["oy"]  = {"status": "live", "size_matched": 0.0}
    client.orders["on"]  = {"status": "live", "size_matched": 0.0}
    client.orders["oy2"] = {"status": "live", "size_matched": 0.0}
    client.orders["on2"] = {"status": "live", "size_matched": 0.0}
    guard, breaker, _ = _build_guard(client)

    sig = _maker_signal(10.0)
    guard.watch_pair(sig, 10.0, _resting_resp("oy"),  _resting_resp("on"))
    guard.watch_pair(sig, 10.0, _resting_resp("oy2"), _resting_resp("on2"))

    assert guard.watched_count == 2          # nothing displaced
    assert guard.is_watching(sig.condition_id)

    # Resolve both pairs fully → both must book, not just the survivor.
    for oid in ("oy", "on", "oy2", "on2"):
        client.orders[oid] = {"status": "matched", "size_matched": 10.0}
    await guard.poll_once()

    assert guard.watched_count == 0
    assert not guard.is_watching(sig.condition_id)
    assert len(breaker.fills) == 2           # each pair booked exactly once


@pytest.mark.asyncio
async def test_error_leg_from_submit_pair_resolves_via_ttl_cancel():
    """
    A leg normalised to {"status": "error"} (failed submission, no order id)
    must not wedge the pair: the surviving resting leg is cancelled at TTL
    and the reservation is released.
    """
    client = FakeGuardClient()
    client.orders["on"] = {"status": "live", "size_matched": 0.0}
    guard, breaker, _ = _build_guard(client, order_ttl_s=0.0)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     {"status": "error", "error": "boom"},   # YES failed
                     _resting_resp("on"))

    await guard.poll_once()

    assert "on" in client.cancelled
    assert breaker.fills == []
    assert breaker.releases == 1
    assert guard.watched_count == 0


@pytest.mark.asyncio
async def test_unwind_failure_marks_stuck_and_alerts():
    """
    A failed unwind (naked shares cannot be sold) must NOT book a phantom
    P&L: it fires a critical alert and releases the reservation, leaving the
    stuck shares for manual recovery.
    """
    class _FailUnwindClient(FakeGuardClient):
        async def unwind_leg(self, token_id, size, price=0.0):
            raise RuntimeError("not enough balance / allowance")

    client = _FailUnwindClient()
    client.orders["oy"] = {"status": "matched", "size_matched": 10.0}
    client.orders["on"] = {"status": "live",    "size_matched": 0.0}
    client.best_asks["tok-no"] = 0.60   # completion unprofitable → unwind path
    guard, breaker, notifier = _build_guard(client, hedge_timeout_s=0.0)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()
    await guard.poll_once()

    assert breaker.fills == []            # nothing realised on stuck shares
    assert breaker.releases == 1          # slot freed
    assert any("MANUAL INTERVENTION" in m for m in notifier.critical)
    assert guard.watched_count == 0


@pytest.mark.asyncio
async def test_taker_completion_books_positive_pnl_and_merges():
    """One-leg fill completed as taker: excess becomes a real pair, positive
    P&L booked, full set merged."""
    client = FakeGuardClient()
    client.orders["oy"] = {"status": "matched", "size_matched": 10.0}
    client.orders["on"] = {"status": "live",    "size_matched": 0.0}
    client.best_asks["tok-no"] = 0.50   # 0.479 + 0.50×1.02 = 0.989 < 1 → complete
    inventory = FakeInventory()
    guard, breaker, _ = _build_guard(client, inventory, hedge_timeout_s=0.0)

    guard.watch_pair(_maker_signal(10.0), 10.0,
                     _resting_resp("oy"), _resting_resp("on"))

    await guard.poll_once()
    await guard.poll_once()

    assert client.taker_buys == [("tok-no", 0.50, 10.0)]
    assert client.unwound == []
    assert len(breaker.fills) == 1
    assert breaker.fills[0] == pytest.approx(0.11, abs=1e-6)   # 10×(1−0.989)
    assert breaker.releases == 0
    assert inventory.paired == ["0xcond"]   # completed excess → full pair merged
