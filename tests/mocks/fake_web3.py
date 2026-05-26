"""Fake Web3 + contract objects for execution module tests."""

from __future__ import annotations

import secrets
from typing import Any


class FakeContractFunction:
    def __init__(self, name: str, args: tuple, parent: "FakeContractFunctions") -> None:
        self.name = name
        self.args = args
        self._parent = parent

    def call(self) -> Any:
        self._parent._record_call(self.name, self.args)
        return self._parent._return_for(self.name, self.args)

    def build_transaction(self, _params: dict) -> dict:
        self._parent._record_call(self.name, self.args)
        return {"to": "0x" + "00" * 20, "data": b"\x00", "value": 0, "nonce": 0,
                "gas": 200_000, "chainId": 137, "type": "0x2",
                "maxFeePerGas": 100, "maxPriorityFeePerGas": 10}


class FakeContractFunctions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.balances: dict[tuple[str, int], int] = {}
        self.approvals: dict[tuple[str, str], bool] = {}
        self.winners: dict[bytes, tuple[int, int]] = {}   # cid_bytes → (yes_num, no_num)
        self.denominators: dict[bytes, int] = {}

    def _record_call(self, name: str, args: tuple) -> None:
        self.calls.append((name, args))

    def _return_for(self, name: str, args: tuple) -> Any:
        if name == "balanceOf":
            wallet, token_id = args
            return self.balances.get((wallet, token_id), 0)
        if name == "isApprovedForAll":
            owner, operator = args
            return self.approvals.get((owner, operator), False)
        if name == "payoutDenominator":
            return self.denominators.get(args[0], 0)
        if name == "payoutNumerators":
            cid, idx = args
            payouts = self.winners.get(cid, (0, 0))
            return payouts[idx] if idx < len(payouts) else 0
        return 0

    def __getattr__(self, name: str):
        def _factory(*args):
            return FakeContractFunction(name, args, self)
        return _factory


class FakeContract:
    def __init__(self) -> None:
        self.functions = FakeContractFunctions()


class FakeEth:
    def __init__(self) -> None:
        self._contracts: dict[str, FakeContract] = {}
        self.sent_raw_transactions: list[bytes] = []
        self.tx_nonce = 0

    def contract(self, address: str, abi: list) -> FakeContract:  # noqa: ARG002
        if address not in self._contracts:
            self._contracts[address] = FakeContract()
        return self._contracts[address]

    def get_block(self, _label: str) -> dict:
        return {"baseFeePerGas": 50}

    def get_transaction_count(self, _addr: str, _state: str = "latest") -> int:
        n = self.tx_nonce
        self.tx_nonce += 1
        return n

    def send_raw_transaction(self, raw: bytes) -> bytes:
        self.sent_raw_transactions.append(raw)
        return secrets.token_bytes(32)

    def wait_for_transaction_receipt(self, _tx_hash: bytes, timeout: int = 60) -> dict:  # noqa: ARG002
        return {"status": 1, "transactionHash": _tx_hash}


class FakeWeb3:
    """Mimics `web3.Web3` enough for AutoRedeemer / InventoryManager init+execute."""

    def __init__(self, _provider: Any = None) -> None:
        self.eth = FakeEth()

    @staticmethod
    def HTTPProvider(_url: str) -> object:
        return object()

    @staticmethod
    def to_checksum_address(addr: str) -> str:
        # Pad to 0x + 40 chars if short.
        if not addr:
            return "0x" + "0" * 40
        a = addr.removeprefix("0x").lower()
        if len(a) < 40:
            a = a.rjust(40, "0")
        return "0x" + a[:40]

    @staticmethod
    def to_wei(amount: int, _unit: str) -> int:
        return amount * 10 ** 9

    @staticmethod
    def keccak(data: bytes) -> bytes:
        import hashlib
        return hashlib.sha256(data).digest()
