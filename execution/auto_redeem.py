"""
execution/auto_redeem.py
────────────────────────
Automated CTF position redeemer — Phase 7.

After a Polymarket binary market resolves, winning-side CTF ERC-1155 shares
must be explicitly redeemed back into USDC by calling redeemPositions() on
the Gnosis Conditional Tokens Framework contract on Polygon.

This module polls every REDEEM_POLL_INTERVAL seconds:
  1.  Collects all condition IDs currently tracked by the FeedRegistry.
  2.  Queries the Gamma REST API for each market's resolution status.
  3.  For any resolved market where the wallet holds a non-zero CTF balance,
      builds an EIP-1559 transaction calling ConditionalTokens.redeemPositions,
      signs it with the EOA key (web3-core-operations EOA signing protocol),
      and broadcasts it to Polygon.
  4.  Logs the tx hash and notifies Telegram on success; logs errors without
      crashing the main bot loop.

CTF contract (Gnosis ConditionalTokens V1 — Polygon mainnet)
─────────────────────────────────────────────────────────────
  Address: 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
  Function: redeemPositions(
                address  collateralToken,
                bytes32  parentCollectionId,
                bytes32  conditionId,
                uint256[] indexSets
            )

  ERC-1155 position ID derivation (Gnosis CTF V1):
    innerHash    = keccak256(abi.encodePacked(conditionId, indexSet))
    collectionId = bytes32(uint256(innerHash) ^ uint256(parentCollectionId))
    positionId   = uint256(keccak256(abi.encodePacked(collateralToken, collectionId)))

  For Polymarket binary markets:
    parentCollectionId = 0x00…00   (top-level, 32 zero bytes)
    collateralToken    = 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174  (USDC.e on Polygon)
    indexSets          : 1 = YES outcome, 2 = NO outcome

Gamma API resolution fields used:
  market.resolved : bool  — true when the market has settled
  market.winner   : str   — "Yes" | "No"

Environment variables (inherited from main.py / .env)
──────────────────────────────────────────────────────
  POLY_PRIVATE_KEY       hex EOA private key (with or without 0x prefix)
  POLY_FUNDER_ADDRESS    EOA wallet address (checksummed or lowercase)
  POLYGON_RPC_URL        Polygon JSON-RPC endpoint
  REDEEM_POLL_INTERVAL   seconds between scans (default 300)
  REDEEM_GAS_LIMIT       gas limit for redeemPositions (default 200_000)
  PAPER_TRADE_MODE       "true" → simulate; print but do not broadcast txs
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import aiohttp
from web3 import Web3
from web3.exceptions import ContractLogicError
from eth_account import Account

if TYPE_CHECKING:
    from core.scanner import FeedRegistry
    from telemetry.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# ── Contract addresses (Polygon mainnet) ────────────────────────────────────
CTF_ADDRESS   = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
USDC_ADDRESS  = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")

# parentCollectionId is 32 zero bytes for all top-level Polymarket positions
NULL_PARENT: bytes = b"\x00" * 32

# ── Polymarket binary market index-set mapping ───────────────────────────────
_INDEX_YES = 1   # 0b01 — YES outcome
_INDEX_NO  = 2   # 0b10 — NO outcome

# ── Minimal CTF ABI (only the functions we call) ─────────────────────────────
_CTF_ABI = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken",    "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId",        "type": "bytes32"},
            {"name": "indexSets",          "type": "uint256[]"},
        ],
        "outputs": [],
    },
    {
        "name": "payoutDenominator",
        "type": "function",
        "stateMutability": "view",
        "inputs":  [{"name": "conditionId", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "payoutNumerators",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "index",       "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
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

# ── Configuration ────────────────────────────────────────────────────────────
_GAMMA_API_BASE  = "https://gamma-api.polymarket.com"
_POLL_INTERVAL   = float(os.environ.get("REDEEM_POLL_INTERVAL", 300))
_GAS_LIMIT       = int(os.environ.get("REDEEM_GAS_LIMIT",       200_000))
_PAPER_TRADE     = os.environ.get("PAPER_TRADE_MODE", "false").strip().lower() == "true"

# Cache: condition_id → {"resolved": bool, "winner": str | None, "ts": float}
_RESOLUTION_TTL  = 120.0   # seconds; re-check resolved status after this long


# ═══════════════════════════════════════════════════════════════════════════
# CTF position-ID math  (Gnosis ConditionalTokens V1 spec)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_position_id(condition_id_hex: str, index_set: int) -> int:
    """
    Compute the ERC-1155 position token ID for a top-level Polymarket position.

    Formula (Gnosis CTF V1):
      innerHash    = keccak256(abi.encodePacked(conditionId_bytes32, indexSet_uint256))
      collectionId = innerHash XOR parentCollectionId   (parent = 0x00…00 → no change)
      positionId   = keccak256(abi.encodePacked(collateralToken_address20, collectionId_bytes32))
    """
    # Normalise condition_id to exactly 32 bytes
    cid_bytes   = bytes.fromhex(condition_id_hex.removeprefix("0x").zfill(64))
    idx_bytes   = index_set.to_bytes(32, "big")

    # Step 1 — inner hash (32 bytes)
    inner_hash  = Web3.keccak(cid_bytes + idx_bytes)
    # Step 2 — collection ID (XOR with NULL_PARENT = identity for top-level)
    parent_int  = int.from_bytes(NULL_PARENT, "big")
    coll_id     = (int.from_bytes(inner_hash, "big") ^ parent_int).to_bytes(32, "big")

    # Step 3 — position ID (uses 20-byte address, not 32-byte padded)
    token_addr_bytes = bytes.fromhex(USDC_ADDRESS.removeprefix("0x"))   # 20 bytes
    pos_hash    = Web3.keccak(token_addr_bytes + coll_id)
    return int.from_bytes(pos_hash, "big")


# ═══════════════════════════════════════════════════════════════════════════
# AutoRedeemer
# ═══════════════════════════════════════════════════════════════════════════

class AutoRedeemer:
    """
    Continuously scans tracked condition IDs for resolved Polymarket markets
    and redeems winning CTF positions into USDC automatically.

    Parameters
    ----------
    feed_registry : FeedRegistry
        Source of truth for which condition IDs the bot is monitoring.
    notifier      : TelegramNotifier
        Used to send Telegram alerts on successful redemptions and errors.

    All credentials are read from environment variables at construction time.
    """

    def __init__(
        self,
        feed_registry: "FeedRegistry",
        notifier:      "TelegramNotifier",
    ) -> None:
        self._registry  = feed_registry
        self._notifier  = notifier

        # Web3 setup — blocking but done at init time before the event loop
        rpc_url           = os.environ.get("POLYGON_RPC_URL", "")
        self._w3          = Web3(Web3.HTTPProvider(rpc_url))
        self._ctf         = self._w3.eth.contract(address=CTF_ADDRESS, abi=_CTF_ABI)

        # EOA key & wallet (web3-core-operations EOA protocol)
        raw_pk            = os.environ.get("POLY_PRIVATE_KEY", "")
        self._private_key = raw_pk if raw_pk.startswith("0x") else "0x" + raw_pk
        self._wallet      = Web3.to_checksum_address(
            os.environ.get("POLY_FUNDER_ADDRESS", "")
        )

        # Per-market resolution cache: condition_id → (winner_index | None, cached_ts)
        self._resolution_cache: dict[str, tuple[int | None, float]] = {}
        # Markets already fully redeemed (skip on subsequent polls)
        self._redeemed: set[str] = set()

        logger.info(
            "AutoRedeemer init | wallet=%s rpc=%s paper=%s poll=%.0fs gas=%d",
            self._wallet[:12], rpc_url[:40] or "<not set>",
            _PAPER_TRADE, _POLL_INTERVAL, _GAS_LIMIT,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public control
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Long-running coroutine.  Polls every REDEEM_POLL_INTERVAL seconds.
        Cancel to stop cleanly.
        """
        logger.info("AutoRedeemer started")
        try:
            while True:
                await asyncio.sleep(_POLL_INTERVAL)
                try:
                    await self._scan_and_redeem()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("AutoRedeemer scan error: %s", exc)
        except asyncio.CancelledError:
            logger.info("AutoRedeemer stopped")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Internal scan
    # ──────────────────────────────────────────────────────────────────────────

    async def _scan_and_redeem(self) -> None:
        """One full pass over all tracked condition IDs."""
        condition_ids = self._registry.condition_ids - self._redeemed
        if not condition_ids:
            return

        logger.debug(
            "AutoRedeemer | checking %d market(s) for resolution",
            len(condition_ids),
        )

        async with aiohttp.ClientSession() as session:
            for cid in condition_ids:
                try:
                    await self._check_one(cid, session)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "AutoRedeemer | error processing condition=%s: %s",
                        cid[:16], exc,
                    )

    async def _check_one(
        self,
        condition_id: str,
        session:      aiohttp.ClientSession,
    ) -> None:
        """Check a single market and redeem if resolved with a position balance."""
        winner_index = await self._get_winner_index(condition_id, session)
        if winner_index is None:
            return   # market not yet resolved

        # Compute the ERC-1155 position token ID for the winning side
        pos_id  = _compute_position_id(condition_id, winner_index)
        balance = await asyncio.get_running_loop().run_in_executor(
            None, self._ctf.functions.balanceOf(self._wallet, pos_id).call
        )

        if balance == 0:
            # No winning shares to redeem (we may never have held this market,
            # or we already redeemed via a previous run)
            self._redeemed.add(condition_id)
            logger.debug(
                "AutoRedeemer | condition=%s resolved (%s) — zero balance, skipping",
                condition_id[:16],
                "YES" if winner_index == _INDEX_YES else "NO",
            )
            return

        logger.info(
            "AutoRedeemer | REDEEMING condition=%s winner=%s balance=%d",
            condition_id[:16],
            "YES" if winner_index == _INDEX_YES else "NO",
            balance,
        )
        await self._redeem(condition_id, [winner_index], balance)

    # ──────────────────────────────────────────────────────────────────────────
    # Resolution cache + Gamma API fetch
    # ──────────────────────────────────────────────────────────────────────────

    async def _get_winner_index(
        self,
        condition_id: str,
        session:      aiohttp.ClientSession,
    ) -> int | None:
        """
        Return the winning indexSet (1 = YES, 2 = NO) for a resolved market,
        or None if the market has not yet resolved.

        Uses a TTL cache to avoid hammering the Gamma API.
        """
        cached = self._resolution_cache.get(condition_id)
        if cached is not None:
            winner_idx, cached_ts = cached
            if time.monotonic() - cached_ts < _RESOLUTION_TTL:
                return winner_idx  # may be None (unresolved) or int (resolved)

        winner_idx = await self._fetch_resolution(condition_id, session)
        self._resolution_cache[condition_id] = (winner_idx, time.monotonic())
        return winner_idx

    async def _fetch_resolution(
        self,
        condition_id: str,
        session:      aiohttp.ClientSession,
    ) -> int | None:
        """
        Query Gamma API for market resolution.
        Returns 1 (YES), 2 (NO), or None (not resolved / fetch error).
        """
        url = f"{_GAMMA_API_BASE}/markets"
        try:
            async with session.get(
                url,
                params={"conditionId": condition_id},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning(
                "AutoRedeemer | Gamma API fetch failed for %s: %s",
                condition_id[:16], exc,
            )
            return None

        # Response may be a list or a dict with a nested list
        markets: list[dict] = data if isinstance(data, list) else data.get("data", [])
        if not markets:
            return None

        market = markets[0]
        resolved: bool = bool(market.get("resolved", False) or market.get("is_resolved", False))
        if not resolved:
            return None

        # Determine winner from API field
        winner_str: str = str(market.get("winner", "")).strip().lower()
        if winner_str in ("yes", "1", "true"):
            return _INDEX_YES
        if winner_str in ("no", "2", "false"):
            return _INDEX_NO

        # Fallback: read payout numerators from the on-chain contract
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_winner_from_chain, condition_id
        )

    def _get_winner_from_chain(self, condition_id: str) -> int | None:
        """
        Sync fallback: check payoutDenominator and payoutNumerators on-chain.
        Must be called via run_in_executor (blocking web3 call).
        """
        try:
            cid_bytes = bytes.fromhex(condition_id.removeprefix("0x").zfill(64))
            denom = self._ctf.functions.payoutDenominator(cid_bytes).call()
            if denom == 0:
                return None   # contract not yet resolved
            yes_payout = self._ctf.functions.payoutNumerators(cid_bytes, 0).call()
            no_payout  = self._ctf.functions.payoutNumerators(cid_bytes, 1).call()
            if yes_payout > no_payout:
                return _INDEX_YES
            if no_payout > yes_payout:
                return _INDEX_NO
        except Exception as exc:
            logger.warning(
                "AutoRedeemer | on-chain resolution check failed for %s: %s",
                condition_id[:16], exc,
            )
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Transaction building, signing, and broadcasting
    # ──────────────────────────────────────────────────────────────────────────

    async def _redeem(
        self,
        condition_id: str,
        index_sets:   list[int],
        balance:      int,
    ) -> None:
        """
        Build, sign, and broadcast a redeemPositions() transaction.

        Paper mode: logs the intent without sending anything on-chain.
        """
        cid_bytes = bytes.fromhex(condition_id.removeprefix("0x").zfill(64))

        if _PAPER_TRADE:
            logger.info(
                "PAPER REDEEM | condition=%s index_sets=%s balance=%d "
                "— no tx broadcast in paper mode",
                condition_id[:16], index_sets, balance,
            )
            return

        try:
            tx_hash = await asyncio.get_running_loop().run_in_executor(
                None,
                self._build_sign_send,
                cid_bytes,
                index_sets,
            )
        except ContractLogicError as exc:
            logger.error(
                "AutoRedeemer | redeemPositions reverted for %s: %s",
                condition_id[:16], exc,
            )
            return
        except Exception as exc:
            logger.error(
                "AutoRedeemer | tx submission failed for %s: %s",
                condition_id[:16], exc,
            )
            return

        self._redeemed.add(condition_id)
        msg = (
            f"CTF REDEEMED\n"
            f"condition={condition_id[:20]}...\n"
            f"index_sets={index_sets}  balance={balance}\n"
            f"tx={tx_hash.hex()}"
        )
        logger.info("AutoRedeemer | %s", msg)
        await self._notifier.notify(msg)

    def _build_sign_send(
        self,
        cid_bytes:  bytes,
        index_sets: list[int],
    ) -> bytes:
        """
        Synchronous: build → EIP-1559 gas estimate → sign → broadcast.
        Runs in a thread-pool executor; must not call async functions.
        Returns the transaction hash bytes on success.
        """
        # EIP-1559 base fee + 2 gwei priority tip (Polygon is cheap)
        base_fee     = self._w3.eth.get_block("pending")["baseFeePerGas"]
        priority_tip = Web3.to_wei(2, "gwei")
        max_fee      = base_fee * 2 + priority_tip

        tx = self._ctf.functions.redeemPositions(
            USDC_ADDRESS,
            NULL_PARENT,
            cid_bytes,
            index_sets,
        ).build_transaction({
            "from":                 self._wallet,
            "nonce":                self._w3.eth.get_transaction_count(self._wallet, "pending"),
            "gas":                  _GAS_LIMIT,
            "maxFeePerGas":         max_fee,
            "maxPriorityFeePerGas": priority_tip,
            "chainId":              137,   # Polygon mainnet
            "type":                 "0x2",
        })

        signed = Account.sign_transaction(tx, self._private_key)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        # Wait for receipt with a 60-second timeout
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        if receipt["status"] != 1:
            raise RuntimeError(
                f"redeemPositions reverted — receipt status={receipt['status']}"
            )
        return tx_hash
