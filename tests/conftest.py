"""Shared pytest fixtures for integration tests.

This module runs BEFORE any production module is imported, so it sets the
environment variables and patches the `py_clob_client.clob_types` namespace
so that imports succeed against the modern py-clob-client SDK (which renamed
`LimitOrderArgs` → `OrderArgs`).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# ── 1. Env vars (must be set BEFORE any production import) ───────────────────
os.environ.setdefault("PAPER_TRADE_MODE", "true")
os.environ.setdefault("POLY_PRIVATE_KEY",
                      "0x" + "11" * 32)
os.environ.setdefault("POLY_FUNDER_ADDRESS",
                      "0x" + "22" * 20)
os.environ.setdefault("POLYGON_RPC_URL", "http://127.0.0.1:0")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "")
os.environ.setdefault("TELEGRAM_CHAT_ID",   "")
os.environ.setdefault("STARTING_BALANCE",  "500")
os.environ.setdefault("SCAN_INTERVAL",     "0.05")
os.environ.setdefault("WALLET_POLL_INTERVAL", "0.05")
os.environ.setdefault("REDEEM_POLL_INTERVAL", "0.05")
os.environ.setdefault("DESIRED_NET_MARGIN", "0.005")
os.environ.setdefault("DEFAULT_TAKER_FEE",  "0.02")
os.environ.setdefault("MAX_FEEDS",          "10")
os.environ.setdefault("MAX_POSITIONS",      "5")
# Pin the pair cap so the suite is hermetic — main.py's load_dotenv() would
# otherwise leak the operator's live .env value into the tests.
os.environ.setdefault("MAX_ARB_PAIR_USDC",  "50.0")
os.environ.setdefault("METRICS_PORT",       "0")
os.environ.setdefault("LOG_LEVEL",          "WARNING")
# Keep the NegRisk execution path exercised by the integration suite (paper
# mode — no real matchOrders tx).  Live default is "off"; see config.py.
os.environ.setdefault("NEGRISK_EXEC_MODE",  "onchain")

# ── 3. Make repo root importable ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── 4. Fixtures ──────────────────────────────────────────────────────────────
import pytest                                                # noqa: E402

from tests.mocks.fake_clob_client import FakeClobClient      # noqa: E402
from tests.mocks.fake_clob_ws    import FakeAiohttpSession, FakeWebSocket  # noqa: E402
from tests.mocks.fake_gamma      import FakeGamma            # noqa: E402
from tests.mocks.fake_telegram   import FakeTelegramNotifier # noqa: E402
from tests.mocks.fake_web3       import FakeWeb3             # noqa: E402


@pytest.fixture
def tmp_signal_db(tmp_path, monkeypatch):
    db_path = tmp_path / "signals.db"
    monkeypatch.setenv("SIGNAL_DB_PATH", str(db_path))
    return db_path


@pytest.fixture
def fake_gamma() -> FakeGamma:
    return FakeGamma()


@pytest.fixture
def fake_clob() -> FakeClobClient:
    return FakeClobClient()


@pytest.fixture
def fake_ws() -> FakeWebSocket:
    return FakeWebSocket([])


@pytest.fixture
def fake_w3() -> FakeWeb3:
    return FakeWeb3()


@pytest.fixture
def fake_telegram() -> FakeTelegramNotifier:
    return FakeTelegramNotifier()


@pytest.fixture
def arb_tick_queue() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait({
        "condition_id": "0x" + "ab" * 32,
        "yes_token_id": "1",
        "no_token_id":  "2",
        "yes_ask":      0.47,
        "no_ask":       0.50,
        "yes_bid":      0.46,
        "no_bid":       0.49,
        "timestamp":    0.0,
    })
    return q


@pytest.fixture
def patched_clob(monkeypatch, fake_clob):
    """Patch the SecureClient import site so PolyClient never hits the network."""
    class _Factory:
        @staticmethod
        def create(**_kw):
            return fake_clob

    monkeypatch.setattr("core.clob_client.SecureClient", _Factory, raising=True)
    return fake_clob


@pytest.fixture
def patched_web3(monkeypatch, fake_w3):
    """Patch the Web3 import in execution modules to return the shared fake."""
    class _W3Proxy:
        HTTPProvider        = staticmethod(FakeWeb3.HTTPProvider)
        to_checksum_address = staticmethod(FakeWeb3.to_checksum_address)
        to_wei              = staticmethod(FakeWeb3.to_wei)
        keccak              = staticmethod(FakeWeb3.keccak)

        def __new__(cls, *_a, **_kw):
            return fake_w3

    monkeypatch.setattr("execution.auto_redeem.Web3",       _W3Proxy, raising=True)
    monkeypatch.setattr("execution.inventory_manager.Web3", _W3Proxy, raising=True)
    return fake_w3


@pytest.fixture
def patched_aiohttp_ws(monkeypatch, fake_ws):
    """Replace aiohttp.ClientSession in ws_feed with a fake session."""
    def _factory(*_a, **_kw):
        return FakeAiohttpSession(fake_ws)
    monkeypatch.setattr("core.ws_feed.aiohttp.ClientSession", _factory, raising=True)
    return fake_ws
