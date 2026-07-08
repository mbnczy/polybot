"""
scripts/setup_allowances.py
───────────────────────────
One-time wallet preparation for LIVE trading with a direct EOA.

Polymarket CLOB orders settle on-chain, so the bot wallet must pre-approve
the exchange contracts.  Without these approvals every fill (BUY), every
PairGuard unwind (SELL), and every mergePositions call reverts.

What it grants (the canonical Polymarket EOA set)
─────────────────────────────────────────────────
  USDC.e  approve(∞)            → CTF (splits/merges)
  USDC.e  approve(∞)            → CTF Exchange          (binary BUY fills)
  USDC.e  approve(∞)            → NegRisk CTF Exchange  (negrisk-market fills)
  USDC.e  approve(∞)            → NegRisk Adapter
  CTF     setApprovalForAll     → CTF Exchange          (SELL / unwind)
  CTF     setApprovalForAll     → NegRisk CTF Exchange
  CTF     setApprovalForAll     → NegRisk Adapter

Note: binary markets that belong to a NegRisk event settle on the NegRisk
CTF Exchange even in "NEGRISK_EXEC_MODE=off" (py-clob-client picks the
exchange per market), so the NegRisk approvals are NOT optional.

Usage
─────
  # 1. Fill POLY_PRIVATE_KEY / POLY_FUNDER_ADDRESS / POLYGON_RPC_URL in .env
  # 2. Dry-run: report balances + current allowances, send nothing:
  .venv/bin/python scripts/setup_allowances.py

  # 3. Send the missing approval transactions (needs POL for gas):
  .venv/bin/python scripts/setup_allowances.py --apply

  # Read-only inspection of an arbitrary address (no key needed):
  .venv/bin/python scripts/setup_allowances.py --address 0x…
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

# ── Polygon mainnet contract addresses ────────────────────────────────────────
USDC_E           = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_NATIVE      = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
CTF              = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
CTF_EXCHANGE     = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982e")
NEGRISK_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")
NEGRISK_ADAPTER  = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")

SPENDERS = [
    ("CTF (ConditionalTokens)", CTF),
    ("CTF Exchange",            CTF_EXCHANGE),
    ("NegRisk CTF Exchange",    NEGRISK_EXCHANGE),
    ("NegRisk Adapter",         NEGRISK_ADAPTER),
]
# ERC-1155 operators: CTF itself never needs to operate our CTF balance.
OPERATORS = SPENDERS[1:]

MAX_UINT = 2 ** 256 - 1
# "Effectively unlimited" threshold — half of max still counts as approved.
APPROVED_FLOOR = 2 ** 255

ERC20_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "o", "type": "address"}, {"name": "s", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "s", "type": "address"}, {"name": "v", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
]

ERC1155_ABI = [
    {"name": "isApprovedForAll", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}, {"name": "o", "type": "address"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "setApprovalForAll", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "o", "type": "address"}, {"name": "b", "type": "bool"}],
     "outputs": []},
]


def _send(w3: Web3, pk: str, wallet: str, fn_call) -> str:
    base_fee = w3.eth.get_block("pending")["baseFeePerGas"]
    tip      = Web3.to_wei(2, "gwei")
    tx = fn_call.build_transaction({
        "from":                 wallet,
        "nonce":                w3.eth.get_transaction_count(wallet, "pending"),
        "gas":                  120_000,
        "maxFeePerGas":         base_fee * 2 + tip,
        "maxPriorityFeePerGas": tip,
        "chainId":              137,
        "type":                 "0x2",
    })
    signed  = Account.sign_transaction(tx, pk)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt["status"] != 1:
        raise RuntimeError(f"tx reverted: {tx_hash.hex()}")
    return tx_hash.hex()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--apply", action="store_true",
                    help="send the missing approval transactions "
                         "(default: report only)")
    ap.add_argument("--address", default=None,
                    help="inspect this address read-only (ignores .env key)")
    args = ap.parse_args()

    load_dotenv()
    rpc = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    w3  = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        print(f"ERROR: cannot reach Polygon RPC at {rpc}")
        return 1

    pk = ""
    if args.address:
        wallet = Web3.to_checksum_address(args.address)
        if args.apply:
            print("ERROR: --apply needs the private key from .env, not --address")
            return 1
    else:
        pk     = os.environ.get("POLY_PRIVATE_KEY", "").strip()
        funder = os.environ.get("POLY_FUNDER_ADDRESS", "").strip()
        if not pk or "YOUR" in pk.upper():
            print("ERROR: POLY_PRIVATE_KEY is not set in .env "
                  "(or use --address 0x… for a read-only check)")
            return 1
        pk = pk if pk.startswith("0x") else "0x" + pk
        wallet = Account.from_key(pk).address
        if funder and "YOUR" not in funder.upper() \
                and Web3.to_checksum_address(funder) != wallet:
            print(f"ERROR: POLY_FUNDER_ADDRESS ({funder}) is not the address "
                  f"of POLY_PRIVATE_KEY ({wallet}) — fix .env first")
            return 1

    usdc_e   = w3.eth.contract(address=USDC_E,      abi=ERC20_ABI)
    usdc_nat = w3.eth.contract(address=USDC_NATIVE, abi=ERC20_ABI)
    ctf      = w3.eth.contract(address=CTF,         abi=ERC1155_ABI)

    print(f"Wallet   : {wallet}")
    print(f"RPC      : {rpc}\n")

    # ── Balances ──────────────────────────────────────────────────────────────
    pol   = w3.eth.get_balance(wallet) / 1e18
    e_bal = usdc_e.functions.balanceOf(wallet).call() / 1e6
    n_bal = usdc_nat.functions.balanceOf(wallet).call() / 1e6
    print(f"POL (gas)      : {pol:.4f}   {'OK' if pol >= 1 else '⚠ top up ~5 POL'}")
    print(f"USDC.e         : {e_bal:.2f}   {'OK' if e_bal > 0 else '⚠ the bot trades USDC.e'}")
    print(f"USDC (native)  : {n_bal:.2f}"
          + ("   ⚠ native USDC is NOT used by the bot — swap it to USDC.e"
             if n_bal > 0 and e_bal == 0 else ""))
    print()

    # ── Allowances ────────────────────────────────────────────────────────────
    todo: list[tuple[str, object]] = []

    for label, spender in SPENDERS:
        cur = usdc_e.functions.allowance(wallet, spender).call()
        ok  = cur >= APPROVED_FLOOR
        print(f"USDC.e allowance → {label:24s} "
              f"{'✅ unlimited' if ok else ('⚠ ' + str(cur / 1e6) + ' — missing')}")
        if not ok:
            todo.append((f"USDC.e approve → {label}",
                         usdc_e.functions.approve(spender, MAX_UINT)))

    for label, operator in OPERATORS:
        ok = ctf.functions.isApprovedForAll(wallet, operator).call()
        print(f"CTF  operator    → {label:24s} {'✅ approved' if ok else '⚠ missing'}")
        if not ok:
            todo.append((f"CTF setApprovalForAll → {label}",
                         ctf.functions.setApprovalForAll(operator, True)))

    print()
    if not todo:
        print("All approvals in place — the wallet is trade-ready. ✅")
        return 0

    if not args.apply:
        print(f"{len(todo)} approval(s) missing.  Re-run with --apply to send them "
              f"(needs a little POL for gas).")
        return 2

    for label, fn_call in todo:
        print(f"Sending: {label} …", flush=True)
        tx = _send(w3, pk, wallet, fn_call)
        print(f"  confirmed  tx={tx}")

    print("\nDone — all approvals granted. ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
