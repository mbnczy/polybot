"""
core/clob_client.py
───────────────────
Thin async wrapper around the synchronous py-clob-client SDK.
MakerArbitrageMachine Edition — adds:
  • Residential proxy routing via `ProxyRotator` (session-level IP rotation)
  • Synthetic Post-Only logic via `post_maker_order()` / `execute_arb_maker_bundle()`
  • Concurrent ECDSA signing via `asyncio.gather` (eliminates ~1 s temporal drag)
  • Full HTTP error taxonomy with per-code strategy + exponential backoff
  • On-demand L2 credential re-derivation on 401

Auth flow (web3-core-operations EOA protocol)
─────────────────────────────────────────────
  Level 1  →  signs with EOA private key (on-chain ECDSA, signature_type=0)
  Level 2  →  derives API key/secret/passphrase from the L1 signature;
              required for order placement

  signature_type=0 = standard MetaMask EOA — NOT the Polymarket L2 proxy key.

HTTP Error Taxonomy
───────────────────
  400 Bad Request   — malformed order (price/size out of range, bad token ID)
                      → NON-RETRYABLE; raise ClobApiError immediately
  401 Unauthorized  — expired / invalid API key
                      → re-derive L2 creds once, then retry; raise on second 401
  403 Forbidden     — Cloudflare blocks current proxy IP
                      → rotate proxy session FIRST, then exponential back-off
  409 Conflict      — duplicate nonce / order ID collision
                      → wait + retry up to MAX_RETRIES (SDK auto-generates new nonce)
  429 Too Many Req  — rate-limited by Polymarket
                      → exponential back-off with ±10 % jitter
  529 Overloaded    — Polymarket server overload (Cloudflare custom code)
                      → same back-off schedule as 429

Proxy session rotation (403)
────────────────────────────
  Residential proxy providers (Bright Data, Oxylabs, etc.) support session-level
  IP rotation via a `{session}` placeholder in the username component of the URL:

      http://user-session-{session}:pass@proxy.host:port

  Each call to `ProxyRotator.rotate()` substitutes a fresh UUID fragment, causing
  the provider to allocate a new IP.  The updated URL is applied to HTTP_PROXY /
  HTTPS_PROXY env vars so all subsequent `requests` calls (py-clob-client) pick
  it up without client reconstruction.

  If PROXY_URL contains no `{session}` placeholder (datacenter proxies that auto-
  rotate), rotation is logged as a no-op and exponential back-off proceeds normally.

Synthetic Post-Only
───────────────────
  Polymarket CLOB has no native post-only flag.  We simulate it:
    BUY  → clamp bid to (best_ask − TICK_SIZE); order rests below the spread
    SELL → clamp ask to (best_bid + TICK_SIZE); order rests above the spread
  This guarantees maker status at submission time.  If the market crosses our
  resting price before the server accepts the order we may accidentally be a
  taker on a micro-fill; the risk is bounded by TICK_SIZE × size.

Environment variables
─────────────────────
  POLY_PRIVATE_KEY     hex-encoded EOA private key
  POLY_FUNDER_ADDRESS  EOA / funder wallet address
  POLYGON_RPC_URL      Polygon JSON-RPC endpoint
  PROXY_URL            residential proxy URL (optional); supports {session} placeholder
                       for per-request IP rotation on Cloudflare 403s
  PAPER_TRADE_MODE     "true" → simulate orders locally  (default false)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

from eth_account import Account
from web3 import Web3

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, LimitOrderArgs, MarketOrderArgs, OrderType

from telemetry.metrics import MATCH_ORDERS_TOTAL, SIGN_SECONDS, SUBMIT_SECONDS
from strategy.arbitrage import TICK_SIZE   # single source of truth for tick size

logger = logging.getLogger(__name__)

# ── Network constants ────────────────────────────────────────────────────────
_CLOB_HOST = "https://clob.polymarket.com"
_CHAIN_ID  = 137   # Polygon mainnet

# ── Proxy constants ───────────────────────────────────────────────────────────
_PROXY_SESSION_PLACEHOLDER = "{session}"   # embedded in residential proxy URLs

# ── Thread pool for blocking SDK calls (ECDSA signing + order submission) ──────
# Each arb execution signs two legs concurrently; with Phase 2 dispatching up to
# EXEC_CONCURRENCY executions in parallel, a fixed 4-worker pool becomes a
# contention point. Size via CLOB_SIGNER_THREADS (default 8) to keep signing off
# the critical path under concurrent execution.
_SIGNER_THREADS = max(2, int(os.environ.get("CLOB_SIGNER_THREADS", "8")))
_executor = ThreadPoolExecutor(
    max_workers=_SIGNER_THREADS, thread_name_prefix="clob-worker"
)

# ── Paper-trade toggle — evaluated once at module load; never changes at runtime
_PAPER_TRADE: bool = (
    os.environ.get("PAPER_TRADE_MODE", "false").strip().lower() == "true"
)

# ── Fill classification (leg-reconciliation) ──────────────────────────────────
# A leg counts as filled when its order response status is one of these. FOK
# orders either match in full or are killed; "paper" is the simulated fill.
_FILLED_STATUSES: frozenset[str] = frozenset({"matched", "filled", "paper"})


def _resp_filled(resp: dict | None) -> bool:
    """True if an order response indicates the leg was fully filled."""
    if not isinstance(resp, dict):
        return False
    return str(resp.get("status", "")).strip().lower() in _FILLED_STATUSES


def classify_fills(yes_resp: dict | None, no_resp: dict | None) -> str:
    """
    Reconcile a two-leg arb execution into one of four states:

        "both"     — both legs filled (the hedged, profitable case)
        "yes_only" — only YES filled  → naked long YES exposure, must unwind
        "no_only"  — only NO filled   → naked long NO exposure, must unwind
        "none"     — neither filled    → no position, no P&L

    This is the safety gate that prevents a single-leg fill from being booked
    as guaranteed arbitrage profit (it is not — it is unhedged directional risk).
    """
    yes_ok = _resp_filled(yes_resp)
    no_ok  = _resp_filled(no_resp)
    if yes_ok and no_ok:
        return "both"
    if yes_ok:
        return "yes_only"
    if no_ok:
        return "no_only"
    return "none"

# ── Synthetic Post-Only constants ─────────────────────────────────────────────
# TICK_SIZE single source of truth is strategy/arbitrage.py — imported at the top
# of this module and re-exported here for callers importing it from clob_client.

# ── Retry policy ──────────────────────────────────────────────────────────────
_MAX_RETRIES:    int   = 5
_BASE_BACKOFF:   float = 0.25    # seconds for attempt 0
_MAX_BACKOFF:    float = 30.0    # hard ceiling
_JITTER_FACTOR:  float = 0.10    # ±10 % random jitter on each wait
_RETRYABLE:      frozenset[int] = frozenset({409, 429, 529})

# ── Bundle position cap ────────────────────────────────────────────────────────
_MAX_BUNDLE_USDC: float = 50.0   # hard ceiling per NegRisk maker bundle

# ── CTF Exchange V2 (Polymarket on-chain order matching — Polygon mainnet) ─────
# matchOrders: atomically mints YES+NO shares and settles both legs in one tx.
# Caller must be a registered operator OR supply a valid taker signature.
_CTF_EXCHANGE_V2_ADDRESS: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982e"

# Minimal ABI — only matchOrders is called by this module.
_ORDER_COMPONENTS = [
    {"name": "salt",          "type": "uint256"},
    {"name": "maker",         "type": "address"},
    {"name": "signer",        "type": "address"},
    {"name": "taker",         "type": "address"},
    {"name": "tokenId",       "type": "uint256"},
    {"name": "makerAmount",   "type": "uint256"},
    {"name": "takerAmount",   "type": "uint256"},
    {"name": "expiration",    "type": "uint256"},
    {"name": "nonce",         "type": "uint256"},
    {"name": "feeRateBps",    "type": "uint256"},
    {"name": "side",          "type": "uint8"},
    {"name": "signatureType", "type": "uint8"},
]

_CTF_EXCHANGE_V2_ABI = [
    {
        "name": "matchOrders",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "takerOrder",       "type": "tuple",   "components": _ORDER_COMPONENTS},
            {"name": "makerOrders",      "type": "tuple[]", "components": _ORDER_COMPONENTS},
            {"name": "takerFillAmount",  "type": "uint256"},
            {"name": "makerFillAmounts", "type": "uint256[]"},
            {"name": "takerSignature",   "type": "bytes"},
            {"name": "makerSignatures",  "type": "bytes[]"},
        ],
        "outputs": [],
    },
]

# Token precision: CTF shares use 1e6 (same as USDC.e)
_CTF_UNIT: int = 10 ** 6

# ── Regex to extract an HTTP status code from an exception message ────────────
_STATUS_RE = re.compile(r"\b([45]\d{2})\b")

# ── Proxy env-var lock ────────────────────────────────────────────────────────
# HTTP_PROXY / HTTPS_PROXY are process-wide.  This lock serialises the
# set → call → restore cycle executed inside the thread-pool so two concurrent
# SDK calls don't overwrite each other's proxy URL.
_proxy_env_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# Custom exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class ClobApiError(Exception):
    """
    Raised when the CLOB API returns a terminal error (non-retryable or
    exhausted retries).  `status_code` carries the HTTP status; 0 means
    the status could not be determined from the upstream exception.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code

    def __repr__(self) -> str:
        return f"ClobApiError(status={self.status_code}, msg={self.args[0]!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# Proxy session rotator
# ═══════════════════════════════════════════════════════════════════════════════

class ProxyRotator:
    """
    Manages residential proxy session rotation to evade Cloudflare 403 blocks.

    If PROXY_URL contains the literal `{session}` placeholder, each call to
    `rotate()` substitutes a fresh UUID fragment, causing the provider to
    allocate a new IP from its residential pool.

    Without `{session}` (datacenter proxies that auto-rotate), `rotate()` is a
    logged no-op and exponential back-off proceeds normally.

    `apply_env()` writes the current URL to HTTP_PROXY / HTTPS_PROXY so that
    the `requests` library (used by py-clob-client) picks it up on every
    subsequent request without client reconstruction.
    """

    def __init__(self, base_url: str) -> None:
        self._base = base_url
        self._has_session = _PROXY_SESSION_PLACEHOLDER in base_url
        self._session_id: str = uuid.uuid4().hex[:12]

    def current_url(self) -> str:
        if self._has_session:
            return self._base.replace(_PROXY_SESSION_PLACEHOLDER, self._session_id)
        return self._base

    def rotate(self) -> str:
        """Generate a new session ID and return the updated proxy URL."""
        if self._has_session:
            self._session_id = uuid.uuid4().hex[:12]
            logger.info("Proxy session rotated → session=%s", self._session_id)
        else:
            logger.warning(
                "Proxy URL has no {session} placeholder — session rotation is a no-op"
            )
        return self.current_url()

    def apply_env(self) -> None:
        """Apply current proxy URL to HTTP_PROXY / HTTPS_PROXY env vars."""
        url = self.current_url()
        os.environ["HTTP_PROXY"]  = url
        os.environ["HTTPS_PROXY"] = url


# ═══════════════════════════════════════════════════════════════════════════════
# Bundle leg descriptor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BundleLeg:
    """
    A single execution leg for `execute_arb_maker_bundle`.

    Constructed by the caller from `NegRiskSignal.legs` (ArbLeg instances in
    strategy/arbitrage.py).  Kept in clob_client to avoid a circular import.
    """
    token_id: str
    bid:      float   # pre-clamped synthetic post-only bid price
    size:     float   # number of shares per bundle


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_http_status(exc: Exception) -> int:
    """
    Best-effort extraction of an HTTP status code from an exception.

    Priority:
      1. `exc.response.status_code`  (requests HTTPError)
      2. First 4xx/5xx group in str(exc)
      3. 0 (unknown)
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    match = _STATUS_RE.search(str(exc))
    return int(match.group(1)) if match else 0


def _backoff_secs(attempt: int) -> float:
    """Exponential backoff with ±JITTER_FACTOR random jitter."""
    raw = min(_BASE_BACKOFF * (2.0 ** attempt), _MAX_BACKOFF)
    return raw * (1.0 + _JITTER_FACTOR * (random.random() * 2.0 - 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# PolyClient
# ═══════════════════════════════════════════════════════════════════════════════

class PolyClient:
    """
    Async-friendly Polymarket CLOB client.

    Construction is blocking (L2 credential derivation); build before entering
    the asyncio event loop or call via `asyncio.to_thread()`.

    All public methods are coroutines.  Blocking SDK calls are dispatched to a
    thread-pool executor so the event loop is never stalled.
    """

    def __init__(self) -> None:
        pk     = os.environ["POLY_PRIVATE_KEY"]
        funder = os.environ["POLY_FUNDER_ADDRESS"]
        self._pk     = pk if pk.startswith("0x") else "0x" + pk
        self._funder = funder

        # ── Web3 / CTF Exchange V2 ────────────────────────────────────────────
        # Used by _match_orders_sync to broadcast matchOrders on-chain.
        rpc_url = os.environ.get("POLYGON_RPC_URL", "")
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        self._wallet = Web3.to_checksum_address(funder)
        self._exchange = self._w3.eth.contract(
            address=Web3.to_checksum_address(_CTF_EXCHANGE_V2_ADDRESS),
            abi=_CTF_EXCHANGE_V2_ABI,
        )

        # Proxy setup — optional; only active when PROXY_URL is set.
        # apply_env() writes HTTP_PROXY / HTTPS_PROXY so py-clob-client's
        # underlying requests library routes all traffic through the proxy.
        proxy_url = os.environ.get("PROXY_URL", "").strip()
        self._proxy: ProxyRotator | None = None
        if proxy_url:
            self._proxy = ProxyRotator(proxy_url)
            # Do NOT call apply_env() here.  Proxy vars are injected only for
            # the duration of each thread-pool SDK call via _run_in_proxy_context,
            # so the WebSocket (ws_feed.py / trust_env=False) is never affected.
            logger.info(
                "Proxy enabled — CLOB REST calls will use residential proxy "
                "(per-call context; WebSocket is unproxied)"
            )

        self._client = self._build_l2_client(pk, funder)

    # ──────────────────────────────────────────────────────────────────────────
    # Client factory (also used for credential re-derivation on 401)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_l2_client(pk: str, funder: str) -> ClobClient:
        """Derive L2 API credentials and return an authenticated ClobClient."""
        l1: ClobClient = ClobClient(
            host=_CLOB_HOST,
            chain_id=_CHAIN_ID,
            key=pk,
            funder=funder,
            signature_type=0,
        )
        try:
            creds: ApiCreds = l1.create_or_derive_api_creds()
            logger.info("CLOB L2 creds derived for funder %s", funder[:10])
        except Exception as exc:
            logger.error("Failed to derive API creds: %s", exc)
            raise

        return ClobClient(
            host=_CLOB_HOST,
            chain_id=_CHAIN_ID,
            key=pk,
            funder=funder,
            creds=creds,
            signature_type=0,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal dispatch helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _run_in_proxy_context(self, fn, *args, **kwargs) -> Any:
        """
        Execute fn(*args, **kwargs) inside a thread-pool worker with the current
        proxy URL injected into HTTP_PROXY / HTTPS_PROXY for the duration of the
        call, then restored to their previous state.

        _proxy_env_lock serialises the set→call→restore cycle so concurrent
        executor threads don't clobber each other's proxy state.

        The async event loop is never touched, which means ws_feed.py's
        `trust_env=False` ClientSession runs on a permanently clean environment.
        """
        proxy_url  = self._proxy.current_url()   # type: ignore[union-attr]
        with _proxy_env_lock:
            prev_http  = os.environ.get("HTTP_PROXY")
            prev_https = os.environ.get("HTTPS_PROXY")
            os.environ["HTTP_PROXY"]  = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
        try:
            return fn(*args, **kwargs)
        finally:
            with _proxy_env_lock:
                if prev_http is None:
                    os.environ.pop("HTTP_PROXY",  None)
                else:
                    os.environ["HTTP_PROXY"]  = prev_http
                if prev_https is None:
                    os.environ.pop("HTTPS_PROXY", None)
                else:
                    os.environ["HTTPS_PROXY"] = prev_https

    async def _run(self, fn, *args, **kwargs) -> Any:
        """
        Dispatch a blocking SDK call to the shared thread pool.

        When a proxy is configured the call is wrapped in _run_in_proxy_context
        so HTTP_PROXY / HTTPS_PROXY are set only for the duration of the SDK
        call.  The WebSocket path (ws_feed.py) is never exposed to these vars.
        """
        loop = asyncio.get_running_loop()
        if self._proxy is not None:
            f = partial(self._run_in_proxy_context, fn, *args, **kwargs)
        else:
            f = partial(fn, *args, **kwargs)
        return await loop.run_in_executor(_executor, f)

    async def _rederive_creds(self) -> None:
        """Re-derive L2 credentials in response to a 401.  Thread-safe."""
        logger.warning("CLOB 401 — re-deriving L2 API credentials")
        new_client = await self._run(self._build_l2_client, self._pk, self._funder)
        self._client = new_client
        logger.info("CLOB credentials refreshed successfully")

    async def _run_with_retry(self, fn, *args, **kwargs) -> Any:
        """
        Run a blocking SDK call with the full HTTP error taxonomy and retry policy.

        Error handling:
          400  → non-retryable; raises ClobApiError(400, …) immediately
          401  → re-derives creds once; if still 401 after re-derive, raises
          409  → retryable (conflict / nonce collision)
          429  → retryable (rate limit) with exponential back-off
          529  → retryable (Polymarket overload) with exponential back-off
          other→ retryable up to _MAX_RETRIES with back-off
        """
        _rederive_attempted = False
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self._run(fn, *args, **kwargs)

            except ClobApiError:
                raise   # already classified; propagate immediately

            except Exception as exc:
                status = _extract_http_status(exc)

                # ── 400: Bad Request — never retryable ───────────────────────
                if status == 400:
                    logger.error(
                        "CLOB 400 Bad Request (non-retryable) | fn=%s err=%s",
                        getattr(fn, "__name__", fn), exc,
                    )
                    raise ClobApiError(400, str(exc)) from exc

                # ── 401: Unauthorized — re-derive creds, retry once ──────────
                if status == 401:
                    if not _rederive_attempted:
                        _rederive_attempted = True
                        logger.warning(
                            "CLOB 401 Unauthorized (attempt %d) — refreshing creds",
                            attempt,
                        )
                        await self._rederive_creds()
                        # Immediately retry without counting as a backoff attempt
                        continue
                    else:
                        logger.error("CLOB 401 persists after credential refresh")
                        raise ClobApiError(401, f"Auth failed after re-derive: {exc}") from exc

                # ── 403: Forbidden — Cloudflare block; rotate proxy then backoff
                if status == 403:
                    if self._proxy is not None:
                        self._proxy.rotate()
                        # No apply_env() here — the new session ID is picked up
                        # automatically by _run_in_proxy_context on the next attempt.
                        logger.warning(
                            "CLOB 403 Forbidden — proxy session rotated "
                            "(attempt %d/%d) | %s",
                            attempt + 1, _MAX_RETRIES, exc,
                        )
                    else:
                        logger.warning(
                            "CLOB 403 Forbidden — no proxy configured; cannot "
                            "rotate session (attempt %d/%d) | %s",
                            attempt + 1, _MAX_RETRIES, exc,
                        )
                    if attempt >= _MAX_RETRIES:
                        raise ClobApiError(
                            403, f"403 persists after proxy rotation: {exc}"
                        ) from exc
                    wait = _backoff_secs(attempt)
                    await asyncio.sleep(wait)
                    continue

                # ── 409 / 429 / 529: retryable ───────────────────────────────
                if status in _RETRYABLE or status == 0:
                    label = {409: "Conflict", 429: "RateLimit", 529: "Overload"}.get(
                        status, f"HTTP-{status or 'unknown'}"
                    )
                    if attempt >= _MAX_RETRIES:
                        logger.error(
                            "CLOB %s — exhausted %d retries | %s",
                            label, _MAX_RETRIES, exc,
                        )
                        raise ClobApiError(status, str(exc)) from exc

                    wait = _backoff_secs(attempt)
                    logger.warning(
                        "CLOB %s (attempt %d/%d) — backing off %.2f s | %s",
                        label, attempt + 1, _MAX_RETRIES, wait, exc,
                    )
                    await asyncio.sleep(wait)
                    continue

                # ── Any other status — treat as retryable up to limit ────────
                if attempt >= _MAX_RETRIES:
                    raise ClobApiError(status or 0, str(exc)) from exc

                wait = _backoff_secs(attempt)
                logger.warning(
                    "CLOB error status=%d (attempt %d/%d) — backing off %.2f s | %s",
                    status, attempt + 1, _MAX_RETRIES, wait, exc,
                )
                await asyncio.sleep(wait)

        raise ClobApiError(0, "unreachable retry exhaustion")

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — market data
    # ──────────────────────────────────────────────────────────────────────────

    async def get_markets(self, next_cursor: str = "") -> dict:
        """Return a page of active markets."""
        return await self._run_with_retry(self._client.get_markets, next_cursor)

    async def get_market(self, condition_id: str) -> dict:
        """Return full market detail for a single condition ID."""
        return await self._run_with_retry(self._client.get_market, condition_id)

    async def get_orderbook(self, token_id: str) -> dict:
        """Return the current L2 orderbook for a token."""
        return await self._run_with_retry(self._client.get_order_book, token_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — order management
    # ──────────────────────────────────────────────────────────────────────────

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order by ID."""
        return await self._run_with_retry(self._client.cancel, order_id)

    async def get_open_orders(self) -> list[dict]:
        """Return all open orders for the authenticated account."""
        return await self._run_with_retry(self._client.get_orders)

    async def get_order_status(self, order_id: str) -> dict | None:
        """
        Fetch the current status of a single order.

        Returns the order dict if found, or None if the order is no longer
        tracked (fully filled or cancelled).  Uses the REST `/order/{id}`
        endpoint.
        """
        try:
            return await self._run_with_retry(self._client.get_order, order_id)
        except ClobApiError as exc:
            if exc.status_code == 404:
                return None   # order fully consumed or never existed
            raise

    async def cancel_all_orders(self) -> dict:
        """Cancel all open orders for the authenticated account."""
        if _PAPER_TRADE:
            logger.info("PAPER CANCEL ALL | no real orders to cancel")
            return {"status": "paper", "cancelled": 0}

        result = await self._run_with_retry(self._client.cancel_all)
        logger.info("cancel_all_orders result: %s", result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — single-leg taker order (FOK)
    # ──────────────────────────────────────────────────────────────────────────

    async def post_order(
        self,
        token_id:    str,
        side:        str,
        price:       float,
        size:        float,
        _order_type: OrderType = OrderType.FOK,
    ) -> dict:
        """
        Build, sign, and submit a single-leg FOK market order.

        Paper mode: logs intent and returns a mock response — no network call.
        """
        if _PAPER_TRADE:
            cost_usdc = round(price * size, 6)
            logger.info(
                "PAPER ORDER | side=%s token=%s price=%.4f size=%.2f cost=%.6f USDC",
                side, token_id[:12], price, size, cost_usdc,
            )
            return {
                "status":    "paper",
                "order_id":  f"paper-{uuid.uuid4()}",
                "token_id":  token_id,
                "side":      side,
                "price":     price,
                "size":      size,
                "cost_usdc": cost_usdc,
            }

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=size,
            side=side,
            price=price,
        )
        signed_order = await self._run_with_retry(
            self._client.create_market_order, order_args
        )
        response = await self._run_with_retry(
            self._client.post_order, signed_order, OrderType.FOK
        )
        logger.info(
            "Order submitted | side=%s token=%s price=%.4f size=%.2f → %s",
            side, token_id[:12], price, size, response,
        )
        return response

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — synthetic post-only maker order (GTC limit)
    # ──────────────────────────────────────────────────────────────────────────

    async def post_maker_order(
        self,
        token_id:      str,
        side:          str,
        desired_price: float,
        size:          float,
    ) -> dict:
        """
        Synthetic post-only limit order.

        Algorithm:
          1. Fetch current L2 orderbook (one REST call, retried on failure).
          2. Clamp the desired price so it cannot immediately cross the spread:
               BUY  → safe_price = min(desired_price, best_ask − TICK_SIZE)
               SELL → safe_price = max(desired_price, best_bid + TICK_SIZE)
          3. Submit as a GTC limit order.  The order rests on the book as a maker.

        The clamp ensures maker status at submission time.  If the spread crosses
        our resting price between submission and acceptance, we may absorb a tiny
        taker fill (bounded by TICK_SIZE × size); this is the residual risk of
        synthetic rather than native post-only.

        Returns the exchange response dict (contains order_id, status, etc.).
        Paper mode: returns a mock dict without any network calls.
        """
        # ── 1. Determine the safe price ──────────────────────────────────────
        safe_price: float
        if _PAPER_TRADE:
            # In paper mode we don't fetch the live book; use desired_price as-is.
            safe_price = round(max(0.01, min(desired_price, 0.99)), 3)
        else:
            book = await self.get_orderbook(token_id)
            side_upper = side.upper()

            if side_upper == "BUY":
                asks = book.get("asks", [])
                if asks:
                    best_ask = min(
                        float(a["price"] if isinstance(a, dict) else a)
                        for a in asks
                    )
                    safe_price = min(desired_price, best_ask - TICK_SIZE)
                else:
                    # No asks visible — use desired_price; order will rest on book
                    safe_price = desired_price

            else:  # SELL
                bids = book.get("bids", [])
                if bids:
                    best_bid = max(
                        float(b["price"] if isinstance(b, dict) else b)
                        for b in bids
                    )
                    safe_price = max(desired_price, best_bid + TICK_SIZE)
                else:
                    safe_price = desired_price

            # Clamp to the valid CLOB price tick range [0.01, 0.99]
            safe_price = round(max(0.01, min(safe_price, 0.99)), 3)

        if _PAPER_TRADE:
            logger.info(
                "PAPER MAKER ORDER | side=%s token=%s desired=%.4f safe=%.4f size=%.2f",
                side, token_id[:12], desired_price, safe_price, size,
            )
            return {
                "status":   "paper",
                "order_id": f"paper-maker-{uuid.uuid4()}",
                "token_id": token_id,
                "side":     side,
                "price":    safe_price,
                "size":     size,
                "maker":    True,
            }

        # ── 2. Sign and submit as GTC limit ─────────────────────────────────
        limit_args = LimitOrderArgs(
            token_id=token_id,
            price=safe_price,
            size=size,
            side=side.upper(),
        )
        signed = await self._run_with_retry(
            self._client.create_limit_order, limit_args
        )
        response = await self._run_with_retry(
            self._client.post_order, signed, OrderType.GTC
        )
        logger.info(
            "Maker order placed | side=%s token=%s desired=%.4f safe=%.4f "
            "size=%.2f → %s",
            side, token_id[:12], desired_price, safe_price, size, response,
        )
        return response

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — dual-leg taker arbitrage (FOK, zero leg risk)
    # ──────────────────────────────────────────────────────────────────────────

    async def execute_arb_pair(
        self,
        yes_token_id: str,
        yes_price:    float,
        yes_size:     float,
        no_token_id:  str,
        no_price:     float,
        no_size:      float,
    ) -> tuple[dict, dict]:
        """
        Fire two FOK market orders for both legs simultaneously.

        Step 1 — ECDSA sign both orders concurrently via asyncio.gather().
                 CPU-bound signing runs in _executor; eliminates ~1 s serial drag.
        Step 2 — Submit both signed FOK orders simultaneously via asyncio.gather().
        Step 3 — Return (yes_resp, no_resp).

        FOK semantics: if either leg cannot be fully filled at the limit price
        the exchange rejects that leg immediately.  Use the InventoryManager to
        monitor for partial fills in live trading.

        Paper mode: simulates both legs, logs guaranteed profit, no network calls.
        """
        if _PAPER_TRADE:
            yes_cost         = round(yes_price * yes_size, 6)
            no_cost          = round(no_price  * no_size,  6)
            combined_cost    = round(yes_cost + no_cost, 6)
            guaranteed_profit = round(yes_size - combined_cost, 6)

            logger.info(
                "PAPER ARB PAIR | "
                "YES %.4f × %.2f = %.4f USDC | "
                "NO  %.4f × %.2f = %.4f USDC | "
                "combined=%.6f | profit=+%.6f USDC",
                yes_price, yes_size, yes_cost,
                no_price,  no_size,  no_cost,
                combined_cost, guaranteed_profit,
            )
            yes_resp: dict = {
                "status":    "paper", "order_id": f"paper-yes-{uuid.uuid4()}",
                "token_id":  yes_token_id, "side": "BUY",
                "price":     yes_price,    "size": yes_size, "cost_usdc": yes_cost,
            }
            no_resp: dict = {
                "status":    "paper", "order_id": f"paper-no-{uuid.uuid4()}",
                "token_id":  no_token_id,  "side": "BUY",
                "price":     no_price,     "size": no_size,  "cost_usdc": no_cost,
            }
            return yes_resp, no_resp

        # ── Live path ─────────────────────────────────────────────────────────
        yes_args = MarketOrderArgs(token_id=yes_token_id, amount=yes_size, side="BUY", price=yes_price)
        no_args  = MarketOrderArgs(token_id=no_token_id,  amount=no_size,  side="BUY", price=no_price)

        # Step 1 — sign both concurrently (CPU-bound ECDSA in executor)
        _sign_t0 = time.monotonic()
        yes_signed, no_signed = await asyncio.gather(
            self._run_with_retry(self._client.create_market_order, yes_args),
            self._run_with_retry(self._client.create_market_order, no_args),
        )
        SIGN_SECONDS.observe(time.monotonic() - _sign_t0)

        # Step 2 — submit both FOK orders simultaneously
        _submit_t0 = time.monotonic()
        yes_resp, no_resp = await asyncio.gather(
            self._run_with_retry(self._client.post_order, yes_signed, OrderType.FOK),
            self._run_with_retry(self._client.post_order, no_signed,  OrderType.FOK),
        )
        SUBMIT_SECONDS.observe(time.monotonic() - _submit_t0)

        logger.info("ARB PAIR submitted | YES=%s | NO=%s", yes_resp, no_resp)
        MATCH_ORDERS_TOTAL.inc()
        return yes_resp, no_resp

    async def unwind_leg(
        self,
        token_id: str,
        size:     float,
        price:    float = 0.0,
    ) -> dict:
        """
        Flatten a naked leg by market-SELLING `size` shares of `token_id`.

        Called when a two-leg arb only half-fills: holding one leg unhedged is
        directional risk, so we immediately sell it back rather than book a
        non-existent guaranteed profit.  `price` is an optional floor/limit hint
        (0.0 → take whatever the book offers).

        Paper mode: simulates the unwind with no network call.
        """
        if _PAPER_TRADE:
            logger.warning(
                "PAPER UNWIND | SELL %.2f of %s (naked leg flattened)",
                size, token_id[:16],
            )
            return {
                "status": "paper", "order_id": f"paper-unwind-{uuid.uuid4()}",
                "token_id": token_id, "side": "SELL", "size": size,
            }

        sell_args = MarketOrderArgs(
            token_id=token_id, amount=size, side="SELL", price=price,
        )
        signed = await self._run_with_retry(self._client.create_market_order, sell_args)
        resp   = await self._run_with_retry(self._client.post_order, signed, OrderType.FOK)
        logger.warning("UNWIND submitted | SELL %s of %s | resp=%s", size, token_id[:16], resp)
        return resp

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — dual-leg maker arbitrage (GTC limit, synthetic post-only)
    # ──────────────────────────────────────────────────────────────────────────

    async def execute_arb_maker_pair(
        self,
        yes_token_id: str,
        yes_bid:      float,   # pre-clamped by DutchBookPricer; safe to submit directly
        yes_size:     float,
        no_token_id:  str,
        no_bid:       float,
        no_size:      float,
    ) -> tuple[dict, dict]:
        """
        Place two GTC limit orders concurrently (synthetic post-only maker arb).

        Both ECDSA signing operations and both order submissions run inside
        a single asyncio.gather() call, eliminating the ~1 s serial signing drag.

        `yes_bid` / `no_bid` must already be clamped below their respective
        best asks (by DutchBookPricer.evaluate_maker()).  This method trusts
        that prices are valid post-only bids in [0.01, 0.99].

        Unlike execute_arb_pair (FOK), these orders REST on the book as makers.
        The InventoryManager monitors fill status and hedges any naked leg that
        remains unfilled for > HEDGE_TIMEOUT_S seconds.

        Returns (yes_resp, no_resp) — exchange dicts with order_id and status.
        Orders may be PENDING (not yet filled) when returned.
        """
        if _PAPER_TRADE:
            logger.info(
                "PAPER MAKER ARB PAIR | YES bid=%.4f × %.2f | NO bid=%.4f × %.2f",
                yes_bid, yes_size, no_bid, no_size,
            )
            yes_resp = {
                "status": "paper", "order_id": f"paper-maker-yes-{uuid.uuid4()}",
                "token_id": yes_token_id, "side": "BUY",
                "price": yes_bid, "size": yes_size, "maker": True,
            }
            no_resp = {
                "status": "paper", "order_id": f"paper-maker-no-{uuid.uuid4()}",
                "token_id": no_token_id, "side": "BUY",
                "price": no_bid, "size": no_size, "maker": True,
            }
            return yes_resp, no_resp

        yes_args = LimitOrderArgs(
            token_id=yes_token_id, price=yes_bid, size=yes_size, side="BUY"
        )
        no_args = LimitOrderArgs(
            token_id=no_token_id, price=no_bid, size=no_size, side="BUY"
        )

        # Concurrent ECDSA signing — eliminates the ~1 s temporal drag
        yes_signed, no_signed = await asyncio.gather(
            self._run_with_retry(self._client.create_limit_order, yes_args),
            self._run_with_retry(self._client.create_limit_order, no_args),
        )

        # Concurrent GTC submission
        yes_resp, no_resp = await asyncio.gather(
            self._run_with_retry(self._client.post_order, yes_signed, OrderType.GTC),
            self._run_with_retry(self._client.post_order, no_signed,  OrderType.GTC),
        )

        logger.info(
            "MAKER ARB PAIR placed | YES order=%s | NO order=%s",
            yes_resp.get("order_id", "?")[:16],
            no_resp.get("order_id",  "?")[:16],
        )
        return yes_resp, no_resp

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers — CTF Exchange V2 on-chain matchOrders
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_order_struct(signed_order: Any) -> dict:
        """
        Convert a py-clob-client signed order object into a CTF Exchange V2
        Order struct dict suitable for ABI-encoding in matchOrders.

        py-clob-client stores order fields with snake_case and camelCase
        aliases depending on version; `getattr` with fallback handles both.
        """
        def _addr(val: Any, fallback: str = "0x" + "0" * 40) -> str:
            raw = str(val) if val else fallback
            return Web3.to_checksum_address(raw) if raw.startswith("0x") else fallback

        return {
            "salt":          int(getattr(signed_order, "salt",          0)),
            "maker":         _addr(getattr(signed_order, "maker",        None)),
            "signer":        _addr(getattr(signed_order, "signer",       None)),
            "taker":         _addr(getattr(signed_order, "taker",        None)),
            "tokenId":       int(getattr(signed_order, "token_id",
                                  getattr(signed_order, "tokenId",       0))),
            "makerAmount":   int(getattr(signed_order, "maker_amount",
                                  getattr(signed_order, "makerAmount",   0))),
            "takerAmount":   int(getattr(signed_order, "taker_amount",
                                  getattr(signed_order, "takerAmount",   0))),
            "expiration":    int(getattr(signed_order, "expiration",     0)),
            "nonce":         int(getattr(signed_order, "nonce",          0)),
            "feeRateBps":    int(getattr(signed_order, "fee_rate_bps",
                                  getattr(signed_order, "feeRateBps",    0))),
            "side":          int(getattr(signed_order, "side",           0)),
            "signatureType": int(getattr(signed_order, "signature_type",
                                  getattr(signed_order, "signatureType", 0))),
        }

    def _match_orders_sync(
        self,
        taker_struct:       dict,
        maker_structs:      list[dict],
        taker_fill_amount:  int,
        maker_fill_amounts: list[int],
        taker_sig:          bytes,
        maker_sigs:         list[bytes],
    ) -> bytes:
        """
        Build → EIP-1559 gas → sign → broadcast → receipt for matchOrders.

        Runs in the shared thread-pool executor (never called directly from
        the event loop).  Raises RuntimeError if the transaction reverts.

        Returns the raw transaction hash bytes on success.
        """
        base_fee     = self._w3.eth.get_block("pending")["baseFeePerGas"]
        priority_tip = Web3.to_wei(2, "gwei")   # standard Polygon tip
        max_fee      = base_fee * 2 + priority_tip

        fn_call = self._exchange.functions.matchOrders(
            taker_struct,
            maker_structs,
            taker_fill_amount,
            maker_fill_amounts,
            taker_sig,
            maker_sigs,
        )
        tx = fn_call.build_transaction({
            "from":                 self._wallet,
            "nonce":                self._w3.eth.get_transaction_count(
                                        self._wallet, "pending"
                                    ),
            "gas":                  500_000,
            "maxFeePerGas":         max_fee,
            "maxPriorityFeePerGas": priority_tip,
            "chainId":              137,
            "type":                 "0x2",
        })
        signed_tx = Account.sign_transaction(tx, self._pk)
        tx_hash   = self._w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt   = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] != 1:
            raise RuntimeError(
                f"matchOrders reverted — status={receipt['status']} "
                f"tx={tx_hash.hex()}"
            )
        logger.info(
            "matchOrders confirmed | tx=%s block=%s gas_used=%s",
            tx_hash.hex()[:20], receipt["blockNumber"], receipt["gasUsed"],
        )
        return tx_hash

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — atomic on-chain arb via CTF Exchange V2 matchOrders
    # ──────────────────────────────────────────────────────────────────────────

    async def execute_arb_maker_bundle(
        self,
        legs:         list[BundleLeg],
        fee_rate_bps: int = 0,
    ) -> list[dict]:
        """
        Execute an N-leg NegRisk arb bundle via CTF Exchange V2 matchOrders.

        Instead of submitting multiple GTC limit orders to the CLOB REST API,
        this method constructs a single on-chain matchOrders transaction.
        When the combined cost across all legs is < $1.00 (accounting for
        the V2 `fee_rate_bps` structure), the shares are atomically minted
        and the transaction either completely succeeds or reverts — no partial
        fills are possible.

        Algorithm
        ─────────
        1. Enforce $50 USDC position cap and V2 profitability gate.
        2. ECDSA-sign all N limit order structs concurrently (CPU-bound,
           thread-pool dispatch via asyncio.gather).
        3. Extract CTF Exchange V2 Order struct dicts + raw signatures.
        4. Broadcast a single matchOrders transaction via Web3 (EIP-1559).
        5. Wait for on-chain confirmation; raise on revert.
        6. Return per-leg response dicts carrying the confirmed tx hash.

        Parameters
        ----------
        legs         : pre-clamped BundleLeg list from NegRiskArbDetector
        fee_rate_bps : V2 fee rate in basis points (default 0 = maker rebate
                       already netted out by the NegRisk signal math)

        Raises
        ------
        ValueError   — legs list is empty
        ClobApiError — bundle cost exceeds $50 cap or V2 edge check fails
        RuntimeError — matchOrders transaction reverted on-chain
        """
        if not legs:
            raise ValueError("execute_arb_maker_bundle: legs list is empty")

        # ── V2 profitability gate ─────────────────────────────────────────────
        # Per-bundle cost × fee multiplier must be < payout (N − 1 USDC / bundle).
        # Uses per-bundle quantities so the check is independent of position size.
        bundle_cost     = sum(leg.bid for leg in legs)   # cost of 1 bundle
        fee_multiplier  = 1.0 + fee_rate_bps / 10_000
        effective_cost  = bundle_cost * fee_multiplier
        payout          = float(len(legs) - 1)   # NegRisk payout per bundle = N − 1

        if effective_cost >= payout:
            raise ClobApiError(
                0,
                f"V2 effective_cost {effective_cost:.6f} USDC ≥ payout "
                f"{payout:.1f} USDC (fee_rate_bps={fee_rate_bps}) — "
                f"bundle is not profitable; skipping matchOrders",
            )

        total_cost = sum(leg.bid * leg.size for leg in legs)
        if total_cost > _MAX_BUNDLE_USDC:
            raise ClobApiError(
                0,
                f"Bundle cost {total_cost:.4f} USDC exceeds "
                f"${_MAX_BUNDLE_USDC:.0f} cap ({len(legs)} legs)",
            )

        # ── Paper-trade path ──────────────────────────────────────────────────
        if _PAPER_TRADE:
            logger.info(
                "PAPER MATCH_ORDERS | %d legs | total_cost=%.4f USDC "
                "effective=%.4f payout=%.1f",
                len(legs), total_cost, effective_cost, payout,
            )
            responses: list[dict] = []
            for i, leg in enumerate(legs):
                resp = {
                    "status":   "paper_matched",
                    "tx_hash":  f"paper-tx-{uuid.uuid4()}",
                    "leg_idx":  i,
                    "token_id": leg.token_id,
                    "price":    leg.bid,
                    "size":     leg.size,
                    "matched":  True,
                }
                responses.append(resp)
                logger.info(
                    "  leg[%d] token=%s bid=%.4f size=%.2f",
                    i, leg.token_id[:12], leg.bid, leg.size,
                )
            return responses

        # ── Live path: CTF Exchange V2 matchOrders ────────────────────────────
        # Step 1 — ECDSA-sign all N order structs concurrently
        limit_args_list = [
            LimitOrderArgs(
                token_id=leg.token_id,
                price=leg.bid,
                size=leg.size,
                side="BUY",
            )
            for leg in legs
        ]
        signed_orders: list[Any] = list(await asyncio.gather(*[
            self._run_with_retry(self._client.create_limit_order, args)
            for args in limit_args_list
        ]))

        # Step 2 — Extract Order structs and raw ECDSA signatures
        # Convention: first leg is the taker order; subsequent legs are makers.
        taker_struct  = self._extract_order_struct(signed_orders[0])
        maker_structs = [self._extract_order_struct(o) for o in signed_orders[1:]]

        def _sig_bytes(signed_order: Any) -> bytes:
            raw = getattr(signed_order, "signature", "") or ""
            return bytes.fromhex(raw.lstrip("0x"))

        taker_sig  = _sig_bytes(signed_orders[0])
        maker_sigs = [_sig_bytes(o) for o in signed_orders[1:]]

        # Fill amounts in CTF token units (1e6 precision)
        taker_fill_amount  = int(legs[0].size * _CTF_UNIT)
        maker_fill_amounts = [int(leg.size * _CTF_UNIT) for leg in legs[1:]]

        # Step 3 — Broadcast matchOrders (blocking; runs in thread-pool)
        tx_hash: bytes = await asyncio.get_running_loop().run_in_executor(
            _executor,
            self._match_orders_sync,
            taker_struct,
            maker_structs,
            taker_fill_amount,
            maker_fill_amounts,
            taker_sig,
            maker_sigs,
        )

        # Step 4 — Build per-leg response dicts
        tx_hex = tx_hash.hex()
        bundle_responses: list[dict] = [
            {
                "status":   "matched",
                "tx_hash":  tx_hex,
                "leg_idx":  i,
                "token_id": leg.token_id,
                "price":    leg.bid,
                "size":     leg.size,
                "matched":  True,
            }
            for i, leg in enumerate(legs)
        ]

        logger.info(
            "MATCH_ORDERS confirmed | %d legs | total_cost=%.4f USDC "
            "effective=%.4f | tx=%s",
            len(legs), total_cost, effective_cost, tx_hex[:20],
        )
        MATCH_ORDERS_TOTAL.inc()
        return bundle_responses

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — close a single leg (emergency taker exit)
    # ──────────────────────────────────────────────────────────────────────────

    async def close_leg_taker(
        self,
        token_id: str,
        size:     float,
        min_price: float = 0.01,
    ) -> dict:
        """
        Emergency exit: sell `size` shares of `token_id` as a taker (FOK SELL).

        Used by InventoryManager when an emergency hedge is needed.  The `min_price`
        guard prevents selling into a completely empty bid book; the default 0.01
        accepts any nonzero bid.
        """
        if _PAPER_TRADE:
            logger.info(
                "PAPER CLOSE LEG | SELL %s size=%.2f min_price=%.4f",
                token_id[:12], size, min_price,
            )
            return {
                "status":   "paper",
                "order_id": f"paper-close-{uuid.uuid4()}",
                "token_id": token_id,
                "side":     "SELL",
                "size":     size,
            }

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=size,
            side="SELL",
            price=min_price,
        )
        signed = await self._run_with_retry(
            self._client.create_market_order, order_args
        )
        response = await self._run_with_retry(
            self._client.post_order, signed, OrderType.FOK
        )
        logger.info(
            "Emergency leg close | SELL %s size=%.2f → %s",
            token_id[:12], size, response,
        )
        return response
