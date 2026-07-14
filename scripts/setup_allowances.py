"""
scripts/setup_allowances.py
───────────────────────────
One-time wallet preparation for LIVE trading — V2 (post-pUSD-migration).

Polymarket's April 2026 exchange upgrade replaced USDC.e collateral with
pUSD (1:1 USDC-backed wrapper) and deployed new exchange contracts.  A
trade-ready EOA wallet therefore needs:

  1. POL for gas.
  2. pUSD balance — wrapped from USDC.e via the CollateralOnramp:
         USDC.e.approve(ONRAMP) ; ONRAMP.wrap(USDC.e, wallet, amount)
  3. The V2 approval set — granted by the official SDK's
     `SecureClient.setup_trading_approvals()` (pUSD → exchanges/adapters/
     router, CTF ERC-1155 operators; already-approved entries are skipped).

Usage
─────
  # Dry-run: report balances + what would be done, send nothing:
  .venv/bin/python scripts/setup_allowances.py

  # Wrap all USDC.e into pUSD and grant the V2 approvals:
  .venv/bin/python scripts/setup_allowances.py --apply

  # Read-only inspection of an arbitrary address (no key needed):
  .venv/bin/python scripts/setup_allowances.py --address 0x…
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

# ── Polygon mainnet contract addresses ────────────────────────────────────────
USDC_E      = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_NATIVE = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
PUSD        = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
# CollateralOnramp — wraps USDC.e 1:1 into pUSD (docs.polymarket.com/concepts/pusd)
ONRAMP      = Web3.to_checksum_address("0x93070a847efEf7F70739046A929D47a521F5B8ee")

MAX_UINT       = 2 ** 256 - 1
APPROVED_FLOOR = 2 ** 255   # "effectively unlimited"

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

ONRAMP_ABI = [
    {"name": "wrap", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "_asset", "type": "address"},
                {"name": "_to", "type": "address"},
                {"name": "_amount", "type": "uint256"}],
     "outputs": []},
]


def _send(w3: Web3, pk: str, wallet: str, fn_call, gas: int = 200_000) -> str:
    base_fee = w3.eth.get_block("pending")["baseFeePerGas"]
    # Polygon enforces a 25 gwei minimum priority fee; 30 gwei clears it.
    tip = Web3.to_wei(30, "gwei")
    tx = fn_call.build_transaction({
        "from":                 wallet,
        "nonce":                w3.eth.get_transaction_count(wallet, "pending"),
        "gas":                  gas,
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
    ap = argparse.ArgumentParser(
        description="V2 wallet prep: wrap USDC.e→pUSD + grant trading approvals"
    )
    ap.add_argument("--apply", action="store_true",
                    help="send the wrap + approval transactions "
                         "(default: report only)")
    ap.add_argument("--address", default=None,
                    help="inspect this address read-only (ignores .env key)")
    args = ap.parse_args()

    load_dotenv()
    rpc = os.environ.get("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
    w3  = Web3(Web3.HTTPProvider(rpc))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    if not w3.is_connected():
        print("ERROR: cannot reach the Polygon RPC")
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
    pusd     = w3.eth.contract(address=PUSD,        abi=ERC20_ABI)
    onramp   = w3.eth.contract(address=ONRAMP,      abi=ONRAMP_ABI)

    rpc_masked = re.sub(r"(https?://[^/]+/).+", r"\1…(masked)", rpc)
    print(f"Wallet   : {wallet}")
    print(f"RPC      : {rpc_masked}\n")

    # ── Balances ──────────────────────────────────────────────────────────────
    pol    = w3.eth.get_balance(wallet) / 1e18
    e_bal  = usdc_e.functions.balanceOf(wallet).call()
    n_bal  = usdc_nat.functions.balanceOf(wallet).call() / 1e6
    p_bal  = pusd.functions.balanceOf(wallet).call() / 1e6
    print(f"POL (gas)      : {pol:.4f}   {'OK' if pol >= 1 else '⚠ top up ~5 POL'}")
    print(f"pUSD           : {p_bal:.2f}   "
          + ("OK — this is what the bot trades with" if p_bal > 0
             else "⚠ the V2 exchange collateral — wrap USDC.e below"))
    print(f"USDC.e         : {e_bal / 1e6:.2f}"
          + ("   → will be wrapped to pUSD" if e_bal > 0 else ""))
    print(f"USDC (native)  : {n_bal:.2f}"
          + ("   ⚠ not usable — swap to USDC.e first (Uniswap)"
             if n_bal > 0 else ""))
    print()

    if args.address:
        print("(read-only mode — approval state requires the SDK client; done)")
        return 0

    # ── Plan ──────────────────────────────────────────────────────────────────
    todo: list[str] = []
    onramp_allowance = usdc_e.functions.allowance(wallet, ONRAMP).call()
    if e_bal > 0:
        if onramp_allowance < e_bal:
            todo.append(f"USDC.e approve → CollateralOnramp ({ONRAMP[:10]}…)")
        todo.append(f"Onramp.wrap: {e_bal / 1e6:.2f} USDC.e → pUSD")
    todo.append("SDK setup_trading_approvals (skips already-approved entries)")

    print("Plan:")
    for step in todo:
        print(f"  • {step}")
    if not args.apply:
        print("\nDry-run only.  Re-run with --apply to execute.")
        return 2

    # ── Execute: wrap USDC.e → pUSD ───────────────────────────────────────────
    if e_bal > 0:
        if onramp_allowance < e_bal:
            print("Sending: USDC.e approve → CollateralOnramp …", flush=True)
            tx = _send(w3, pk, wallet, usdc_e.functions.approve(ONRAMP, MAX_UINT),
                       gas=120_000)
            print(f"  confirmed  tx={tx}")
        print(f"Sending: Onramp.wrap {e_bal / 1e6:.2f} USDC.e → pUSD …", flush=True)
        tx = _send(w3, pk, wallet, onramp.functions.wrap(USDC_E, wallet, e_bal))
        print(f"  confirmed  tx={tx}")
        print(f"  pUSD balance now: "
              f"{pusd.functions.balanceOf(wallet).call() / 1e6:.2f}")

    # ── Execute: V2 trading approvals via the official SDK ───────────────────
    print("Running SDK setup_trading_approvals …", flush=True)
    from polymarket import SecureClient
    with SecureClient.create(private_key=pk, wallet=wallet) as client:
        client.setup_trading_approvals().wait()
    print("  approvals in place.")

    print("\nDone — the wallet is V2 trade-ready. ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
