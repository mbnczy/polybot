"""
execution/inventory_manager.py
───────────────────────────────
On-chain settlement engine — matchOrders atomic execution edition.

With the Phase 10 upgrade to CTF Exchange V2 `matchOrders`, the bot now
executes arb bundles as a single atomic on-chain transaction.  The exchange
either fully settles every leg or reverts — partial fills are impossible by
construction.

Because matchOrders bundles cannot partially fill, this class carries no
hedge logic.  NOTE: the binary GTC maker path (execute_arb_maker_pair) is NOT
atomic — its one-leg / partial-fill risk is handled by MakerPairGuard
(execution/pair_guard.py), not here.  The InventoryManager focuses
exclusively on:

1. On-Chain Settlement  (mergePositions, triggered on paired fill detection)
   After matchOrders is confirmed, our wallet holds a full YES+NO complementary
   set worth exactly $1.00 USDC at expiry.  InventoryManager polls on-chain
   ERC-1155 balances and calls the appropriate mergePositions function to
   convert the pair back to USDC immediately.

   Binary markets  → CTF.mergePositions (CollateralToken + conditionId)
   NegRisk markets → NegRiskAdapter.mergePositions (conditionId only)

2. Wallet Monitor  (orphan recovery, every WALLET_POLL_INTERVAL seconds)
   Scans on-chain CTF balances for condition IDs not currently tracked in
   the in-memory position map (e.g. after a VPS restart) and merges any
   found complementary sets.

CTF V2 mergePositions
─────────────────────
Contract : 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045 (Polygon mainnet)
Function : mergePositions(address, bytes32, bytes32, uint256[], uint256)
Effect   : burns `amount` YES tokens + `amount` NO tokens;
           transfers `amount` USDC.e (6 decimals) to msg.sender.

Environment variables
─────────────────────
  POLY_PRIVATE_KEY       EOA hex private key
  POLY_FUNDER_ADDRESS    EOA wallet address (checksummed)
  POLYGON_RPC_URL        Polygon JSON-RPC endpoint
  WALLET_POLL_INTERVAL   seconds between on-chain balance checks (default 60)
  PAPER_TRADE_MODE       "true" → log but do not submit any tx or order
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eth_account import Account
from web3 import Web3
from web3.exceptions import ContractLogicError

from risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerTripped,
)
from telemetry.metrics import CAPITAL_RECYCLED_TOTAL

if TYPE_CHECKING:
    from core.clob_client import PolyClient
    from strategy.arbitrage import ArbSignal
    from telemetry.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
WALLET_POLL_INTERVAL: float = float(os.environ.get("WALLET_POLL_INTERVAL", 60.0))
_PAPER_TRADE:         bool  = os.environ.get("PAPER_TRADE_MODE", "false").strip().lower() == "true"

# ── CTF Contract (Gnosis ConditionalTokens V1 — Polygon mainnet) ─────────────
_CTF_ADDRESS  = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
_USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
_NULL_PARENT: bytes = b"\x00" * 32
_PARTITION_BINARY: list[int] = [1, 2]   # [YES indexSet, NO indexSet]

# CTF shares use USDC.e precision (6 decimals): 1 share = 1_000_000 units
_CTF_DECIMALS: int = 6
_CTF_UNIT:     int = 10 ** _CTF_DECIMALS

# ── NegRisk Adapter (Polymarket multi-outcome markets — Polygon mainnet) ──────
_NEGRISK_ADAPTER_ADDRESS = Web3.to_checksum_address(
    "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
)

_NEGRISK_ABI = [
    {
        "name": "mergePositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "amount",      "type": "uint256"},
        ],
        "outputs": [],
    },
]

# ── Minimal CTF ABI — only the functions this module calls ───────────────────
_CTF_ABI = [
    {
        "name": "mergePositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken",    "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId",        "type": "bytes32"},
            {"name": "partition",          "type": "uint256[]"},
            {"name": "amount",             "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "isApprovedForAll",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "account",  "type": "address"},
            {"name": "operator", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "setApprovalForAll",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"},
        ],
        "outputs": [],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id",      "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegFill:
    """Confirmed fill for one leg of an arb pair."""
    token_id:   str
    side:       str      # "YES" | "NO"
    fill_price: float    # actual fill price (USDC/share)
    shares:     float    # number of shares filled
    fill_time:  float    # time.monotonic() at fill confirmation


@dataclass
class Position:
    """
    Tracks both legs of one arb pair through the fill-to-settle lifecycle.

    States:
      PENDING  → matchOrders broadcast; awaiting on-chain confirmation
      PAIRED   → both legs confirmed on-chain; proceed to mergePositions
      SETTLING → merge tx broadcast, awaiting confirmation
      SETTLED  → mergePositions confirmed; USDC recycled
      FAILED   → merge failed; requires manual intervention
    """
    condition_id:  str
    yes_token_id:  str
    no_token_id:   str
    yes_order_id:  str
    no_order_id:   str
    n_shares:      float
    submitted_at:  float           # time.monotonic()

    yes_fill:  LegFill | None = field(default=None)
    no_fill:   LegFill | None = field(default=None)
    status:    str            = field(default="PENDING")
    is_negrisk: bool          = field(default=False)
    # True  → _on_settled books P&L into the breaker (matchOrders/NegRisk path
    #         where the strategy loop did NOT book it).
    # False → P&L was already booked at fill time (binary FOK/maker path); the
    #         settlement here only recycles collateral via mergePositions.
    book_pnl_on_settle: bool  = field(default=True)

    @property
    def is_paired(self) -> bool:
        return self.yes_fill is not None and self.no_fill is not None


# ═══════════════════════════════════════════════════════════════════════════════
# InventoryManager
# ═══════════════════════════════════════════════════════════════════════════════

class InventoryManager:
    """
    Monitors atomic matchOrders fills and triggers on-chain mergePositions.

    Because CTF Exchange V2 matchOrders is all-or-nothing, there are no partial
    fills to hedge.  This class simply detects when a paired position has been
    confirmed on-chain (both legs filled) and calls mergePositions to convert
    the YES+NO complementary set back to USDC.

    Parameters
    ----------
    clob_client     : PolyClient — used for open-order polling (fill detection)
    circuit_breaker : CircuitBreaker — gates every merge action
    notifier        : TelegramNotifier — sends alerts on merge/failure events

    Usage (in main.py):
        inv_mgr = InventoryManager(client, breaker, notifier)
        asyncio.create_task(inv_mgr.run(), name="inventory_manager")

        # After a successful execute_arb_maker_bundle():
        inv_mgr.register_matched_pair(signal, tx_hash)
    """

    def __init__(
        self,
        clob_client:     "PolyClient",
        circuit_breaker: CircuitBreaker,
        notifier:        "TelegramNotifier",
    ) -> None:
        self._client  = clob_client
        self._breaker = circuit_breaker
        self._notifier = notifier

        # ── Web3 / CTF setup ─────────────────────────────────────────────────
        rpc_url       = os.environ.get("POLYGON_RPC_URL", "")
        self._w3      = Web3(Web3.HTTPProvider(rpc_url))
        self._ctf     = self._w3.eth.contract(address=_CTF_ADDRESS, abi=_CTF_ABI)
        self._negrisk = self._w3.eth.contract(
            address=_NEGRISK_ADAPTER_ADDRESS, abi=_NEGRISK_ABI
        )

        raw_pk           = os.environ.get("POLY_PRIVATE_KEY", "")
        self._private_key = raw_pk if raw_pk.startswith("0x") else "0x" + raw_pk
        self._wallet      = Web3.to_checksum_address(
            os.environ.get("POLY_FUNDER_ADDRESS", "")
        )

        # ── In-memory position state ──────────────────────────────────────────
        self._positions: dict[str, Position] = {}
        self._lock = asyncio.Lock()

        # ERC-1155 approval flags — checked and granted once per session.
        self._ctf_approved:     bool = False
        self._negrisk_approved: bool = False

        logger.info(
            "InventoryManager init | wallet=%s paper=%s poll=%.0fs",
            self._wallet[:12] if self._wallet else "?",
            _PAPER_TRADE, WALLET_POLL_INTERVAL,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public control
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Long-running monitoring coroutine.  Run as an asyncio.Task.

        Poll cycle (500 ms):
          1. Fetch open orders from the CLOB to detect fills.
          2. For each newly PAIRED position → merge_complementary_set().

        Wallet poll (every WALLET_POLL_INTERVAL):
          Check on-chain CTF balances for orphaned positions (post-restart
          recovery).
        """
        logger.info("InventoryManager started")
        last_wallet_poll = 0.0

        try:
            while True:
                await asyncio.sleep(0.5)   # 500 ms hot-path cadence

                try:
                    await self._poll_fills()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("InventoryManager poll error: %s", exc)

                now = time.monotonic()
                if now - last_wallet_poll >= WALLET_POLL_INTERVAL:
                    last_wallet_poll = now
                    try:
                        await self._wallet_monitor()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error("InventoryManager wallet monitor error: %s", exc)

        except asyncio.CancelledError:
            logger.info("InventoryManager stopped")
            raise

    def register_matched_pair(
        self,
        signal:     "ArbSignal",
        tx_hash:    str,
        *,
        is_negrisk: bool = False,
    ) -> None:
        """
        Register a matchOrders-confirmed arb pair for settlement monitoring.

        Call immediately after a successful execute_arb_maker_bundle() returns.

        Parameters
        ----------
        signal     : the ArbSignal that triggered execution
        tx_hash    : hex tx hash of the confirmed matchOrders transaction
        is_negrisk : True for NegRisk multi-outcome markets (uses NegRiskAdapter
                     for mergePositions instead of the CTF contract)
        """
        pos = Position(
            condition_id=signal.condition_id,
            yes_token_id=signal.yes_token_id,
            no_token_id=signal.no_token_id,
            yes_order_id=tx_hash,
            no_order_id=tx_hash,
            n_shares=signal.yes_size,
            submitted_at=time.monotonic(),
            # matchOrders is atomic — both legs are immediately paired
            yes_fill=LegFill(
                token_id=signal.yes_token_id, side="YES",
                fill_price=signal.yes_ask, shares=signal.yes_size,
                fill_time=time.monotonic(),
            ),
            no_fill=LegFill(
                token_id=signal.no_token_id, side="NO",
                fill_price=signal.no_ask, shares=signal.yes_size,
                fill_time=time.monotonic(),
            ),
            status="PAIRED",
            is_negrisk=is_negrisk,
        )
        self._positions[signal.condition_id] = pos
        logger.info(
            "Position registered (matched) | condition=%s tx=%s shares=%.2f",
            signal.condition_id[:16], tx_hash[:20], signal.yes_size,
        )
        # Schedule settlement immediately — no hedge timer needed
        asyncio.ensure_future(self._settle(pos))

    def register_paired_fill(
        self,
        signal:     "ArbSignal",
        *,
        is_negrisk: bool = False,
    ) -> None:
        """
        Register a binary FOK/maker arb pair whose BOTH legs are already filled,
        for **capital recycling only** (mergePositions → USDC now, instead of
        waiting until market resolution).

        Unlike `register_matched_pair`, P&L is NOT booked at settlement here —
        the strategy loop already booked it at fill time. Set via
        `book_pnl_on_settle=False` to avoid double-counting.
        """
        if signal.condition_id in self._positions:
            return   # already tracked
        synthetic_id = f"fill-{signal.condition_id[:16]}-{int(time.monotonic()*1e3)}"
        pos = Position(
            condition_id=signal.condition_id,
            yes_token_id=signal.yes_token_id,
            no_token_id=signal.no_token_id,
            yes_order_id=synthetic_id,
            no_order_id=synthetic_id,
            n_shares=signal.yes_size,
            submitted_at=time.monotonic(),
            yes_fill=LegFill(
                token_id=signal.yes_token_id, side="YES",
                fill_price=signal.yes_ask, shares=signal.yes_size,
                fill_time=time.monotonic(),
            ),
            no_fill=LegFill(
                token_id=signal.no_token_id, side="NO",
                fill_price=signal.no_ask, shares=signal.yes_size,
                fill_time=time.monotonic(),
            ),
            status="PAIRED",
            is_negrisk=is_negrisk,
            book_pnl_on_settle=False,   # P&L already booked at fill
        )
        self._positions[signal.condition_id] = pos
        logger.info(
            "Position registered (paired fill, recycle-only) | condition=%s shares=%.2f",
            signal.condition_id[:16], signal.yes_size,
        )
        asyncio.ensure_future(self._settle(pos))

    # ──────────────────────────────────────────────────────────────────────────
    # Fill polling (detects on-chain confirmations for registered positions)
    # ──────────────────────────────────────────────────────────────────────────

    async def _poll_fills(self) -> None:
        """
        Reconcile tracked positions against open-order state.

        For matchOrders positions, both legs are marked PAIRED at registration
        time.  This poll primarily handles any wallet-monitor-recovered
        positions that start in PENDING state.
        """
        async with self._lock:
            if not self._positions:
                return
            condition_ids = list(self._positions.keys())

        try:
            open_orders: list[dict] = await self._client.get_open_orders()
        except Exception as exc:
            logger.warning("InventoryManager | get_open_orders failed: %s", exc)
            return

        open_ids: set[str] = {o.get("id", o.get("order_id", "")) for o in open_orders}

        async with self._lock:
            for cid in condition_ids:
                pos = self._positions.get(cid)
                if pos is None or pos.status not in ("PENDING",):
                    continue

                # An order absent from open_orders has been filled or cancelled.
                now = time.monotonic()
                if pos.yes_fill is None and pos.yes_order_id not in open_ids:
                    pos.yes_fill = LegFill(
                        token_id=pos.yes_token_id, side="YES",
                        fill_price=0.0, shares=pos.n_shares, fill_time=now,
                    )
                if pos.no_fill is None and pos.no_order_id not in open_ids:
                    pos.no_fill = LegFill(
                        token_id=pos.no_token_id, side="NO",
                        fill_price=0.0, shares=pos.n_shares, fill_time=now,
                    )

                if pos.is_paired:
                    pos.status = "PAIRED"
                    asyncio.ensure_future(self._settle(pos))

    # ──────────────────────────────────────────────────────────────────────────
    # On-Chain Settlement — mergePositions
    # ──────────────────────────────────────────────────────────────────────────

    async def _settle(self, pos: Position) -> None:
        """
        Triggered when both legs are confirmed filled.

        1. Ensure CTF ERC-1155 approval for the merge contract.
        2. Call mergePositions to burn YES + NO shares and receive USDC.
        3. Update circuit breaker with the realised profit.
        """
        if not pos.is_paired:
            return

        pos.status = "SETTLING"
        logger.info(
            "SETTLING | condition=%s shares=%.2f — calling mergePositions",
            pos.condition_id[:16], pos.n_shares,
        )

        if _PAPER_TRADE:
            logger.info(
                "PAPER SETTLE | mergePositions condition=%s shares=%.2f",
                pos.condition_id[:16], pos.n_shares,
            )
            pos.status = "SETTLED"
            self._on_settled(pos)
            return

        settle_fn = (
            self._negrisk_merge_sync if pos.is_negrisk
            else self._merge_positions_sync
        )
        contract_label = "NegRiskAdapter" if pos.is_negrisk else "CTF"

        try:
            tx_hash = await asyncio.get_running_loop().run_in_executor(
                None,
                settle_fn,
                pos.condition_id,
                pos.n_shares,
            )
            pos.status = "SETTLED"
            self._on_settled(pos)

            msg = (
                f"💰 SETTLED — {contract_label}.mergePositions confirmed\n"
                f"condition={pos.condition_id[:20]}\n"
                f"shares={pos.n_shares:.2f}  USDC_received≈{pos.n_shares:.4f}\n"
                f"tx={tx_hash.hex()}"
            )
            logger.info(msg)
            await self._notifier.notify(msg)

        except ContractLogicError as exc:
            pos.status = "FAILED"
            logger.error(
                "mergePositions reverted for condition=%s: %s",
                pos.condition_id[:16], exc,
            )
            await self._notifier.notify(
                f"❌ mergePositions REVERTED\ncondition={pos.condition_id[:20]}\n{exc}"
            )
        except Exception as exc:  # noqa: BLE001
            pos.status = "FAILED"
            logger.error("_settle failed for condition=%s: %s", pos.condition_id[:16], exc)
            await self._notifier.notify(
                f"❌ SETTLE FAILED\ncondition={pos.condition_id[:20]}\n{exc}"
            )

    def _on_settled(self, pos: Position) -> None:
        """
        Finalise a settled position: recycle capital (already merged on-chain by
        the caller) and — only when this path owns P&L — book the guaranteed
        profit into the circuit breaker.

        `book_pnl_on_settle` is False for the binary FOK/maker path, where the
        strategy loop already booked P&L at fill time; settling there merely
        recycles collateral, so we must NOT double-count.
        """
        yes_paid   = pos.yes_fill.fill_price * pos.n_shares if pos.yes_fill else 0.0
        no_paid    = pos.no_fill.fill_price  * pos.n_shares if pos.no_fill  else 0.0
        total_paid = yes_paid + no_paid
        net_pnl    = round(pos.n_shares - total_paid, 6)

        if pos.book_pnl_on_settle:
            try:
                self._breaker.on_fill(pnl=net_pnl)
            except CircuitBreakerTripped:
                raise

        CAPITAL_RECYCLED_TOTAL.inc()
        logger.info(
            "Position settled | condition=%s n_shares=%.2f total_paid=%.6f "
            "net_pnl=%.6f USDC booked_pnl=%s",
            pos.condition_id[:16], pos.n_shares, total_paid, net_pnl,
            pos.book_pnl_on_settle,
        )
        self._positions.pop(pos.condition_id, None)

    async def merge_complementary_set(
        self,
        condition_id: str,
        n_shares:     float,
        *,
        is_negrisk:   bool = False,
    ) -> bytes | None:
        """
        Public entry point: burn a full YES+NO complementary set on-chain.

        For standard binary markets, calls CTF.mergePositions.
        For NegRisk multi-outcome markets, calls NegRiskAdapter.mergePositions.

        Returns the transaction hash bytes on success, or None in paper mode.
        Raises on failure so the caller can decide how to handle it.
        """
        contract_label = "NegRiskAdapter" if is_negrisk else "CTF"
        if _PAPER_TRADE:
            logger.info(
                "PAPER %s.mergePositions | condition=%s shares=%.2f",
                contract_label, condition_id[:16], n_shares,
            )
            return None

        sync_fn = self._negrisk_merge_sync if is_negrisk else self._merge_positions_sync
        return await asyncio.get_running_loop().run_in_executor(
            None,
            sync_fn,
            condition_id,
            n_shares,
        )

    def _merge_positions_sync(
        self,
        condition_id: str,
        n_shares:     float,
    ) -> bytes:
        """
        Blocking: approve (if needed) + build + EIP-1559 gas + sign + broadcast
        mergePositions.  Runs in a thread-pool executor.
        """
        if not self._ctf_approved:
            approved: bool = self._ctf.functions.isApprovedForAll(
                self._wallet, _CTF_ADDRESS
            ).call()
            if not approved:
                logger.info("Granting CTF ERC-1155 setApprovalForAll")
                self._send_tx(
                    self._ctf.functions.setApprovalForAll(_CTF_ADDRESS, True)
                )
            self._ctf_approved = True

        cid_bytes    = bytes.fromhex(condition_id.removeprefix("0x").zfill(64))
        amount_units = int(n_shares * _CTF_UNIT)

        logger.info(
            "mergePositions | condition=%s amount=%d units (%.4f shares)",
            condition_id[:16], amount_units, n_shares,
        )
        return self._send_tx(
            self._ctf.functions.mergePositions(
                _USDC_ADDRESS,
                _NULL_PARENT,
                cid_bytes,
                _PARTITION_BINARY,
                amount_units,
            )
        )

    def _negrisk_merge_sync(
        self,
        condition_id: str,
        n_shares:     float,
    ) -> bytes:
        """
        Blocking: approve (if needed) + build + EIP-1559 gas + sign + broadcast
        NegRiskAdapter.mergePositions.  Runs in a thread-pool executor.
        """
        if not self._negrisk_approved:
            approved: bool = self._ctf.functions.isApprovedForAll(
                self._wallet, _NEGRISK_ADAPTER_ADDRESS
            ).call()
            if not approved:
                logger.info(
                    "Granting CTF setApprovalForAll to NegRiskAdapter (%s)",
                    _NEGRISK_ADAPTER_ADDRESS,
                )
                self._send_tx(
                    self._ctf.functions.setApprovalForAll(
                        _NEGRISK_ADAPTER_ADDRESS, True
                    )
                )
            self._negrisk_approved = True

        cid_bytes    = bytes.fromhex(condition_id.removeprefix("0x").zfill(64))
        amount_units = int(n_shares * _CTF_UNIT)

        logger.info(
            "NegRiskAdapter.mergePositions | condition=%s amount=%d units (%.4f shares)",
            condition_id[:16], amount_units, n_shares,
        )
        return self._send_tx(
            self._negrisk.functions.mergePositions(cid_bytes, amount_units)
        )

    def _send_tx(self, fn_call) -> bytes:
        """
        Build → EIP-1559 gas → sign → broadcast → wait for receipt.
        Returns the raw transaction hash bytes on success.
        Raises on revert or timeout.
        """
        base_fee     = self._w3.eth.get_block("pending")["baseFeePerGas"]
        priority_tip = Web3.to_wei(2, "gwei")    # standard Polygon tip
        max_fee      = base_fee * 2 + priority_tip

        tx = fn_call.build_transaction({
            "from":                 self._wallet,
            "nonce":                self._w3.eth.get_transaction_count(self._wallet, "pending"),
            "gas":                  300_000,
            "maxFeePerGas":         max_fee,
            "maxPriorityFeePerGas": priority_tip,
            "chainId":              137,
            "type":                 "0x2",
        })
        signed  = Account.sign_transaction(tx, self._private_key)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        if receipt["status"] != 1:
            raise RuntimeError(
                f"Transaction reverted — status={receipt['status']} tx={tx_hash.hex()}"
            )
        return tx_hash

    # ──────────────────────────────────────────────────────────────────────────
    # Wallet Monitor — orphan position recovery
    # ──────────────────────────────────────────────────────────────────────────

    async def _wallet_monitor(self) -> None:
        """
        Scan on-chain CTF balances for all tracked condition IDs.

        Detects complementary sets (YES balance == NO balance > 0) for markets
        not currently in the SETTLING/SETTLED state, and triggers a merge.
        This handles bot restarts where in-memory position state was lost.
        """
        async with self._lock:
            monitored = {
                cid: pos for cid, pos in self._positions.items()
                if pos.status not in ("SETTLING", "SETTLED", "FAILED")
            }

        if not monitored:
            return

        logger.debug("WalletMonitor | checking %d position(s) on-chain", len(monitored))

        loop = asyncio.get_running_loop()
        for cid, pos in monitored.items():
            try:
                yes_bal, no_bal = await asyncio.gather(
                    loop.run_in_executor(None, self._get_ctf_balance, cid, 1),
                    loop.run_in_executor(None, self._get_ctf_balance, cid, 2),
                )
                if yes_bal > 0 and yes_bal == no_bal and pos.status != "SETTLING":
                    logger.info(
                        "WalletMonitor | orphan complementary set found "
                        "condition=%s yes_bal=%d no_bal=%d — merging",
                        cid[:16], yes_bal, no_bal,
                    )
                    shares = yes_bal / _CTF_UNIT
                    asyncio.ensure_future(
                        self.merge_complementary_set(
                            cid, shares, is_negrisk=pos.is_negrisk
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "WalletMonitor balance check failed condition=%s: %s",
                    cid[:16], exc,
                )

    def _get_ctf_balance(self, condition_id: str, index_set: int) -> int:
        """
        Return the on-chain ERC-1155 balance for a single CTF token.

        index_set: 1 = YES position, 2 = NO position.
        Runs in a thread-pool executor; never called from the event loop.
        """
        cid_bytes = bytes.fromhex(condition_id.removeprefix("0x").zfill(64))
        # positionId = keccak256(collateralToken || collectionId)
        # For simplicity: query by tokenId = uint256(keccak256(parentCollectionId || conditionId || indexSet))
        import hashlib
        collection_id = hashlib.sha256(
            _NULL_PARENT + cid_bytes + index_set.to_bytes(32, "big")
        ).digest()
        token_id = int.from_bytes(collection_id, "big")
        return self._ctf.functions.balanceOf(self._wallet, token_id).call()
