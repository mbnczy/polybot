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
import json
import logging
import math
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from functools import partial, wraps
from typing import Any

from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

# Polymarket unified V2 SDK (the archived py-clob-client signs pre-migration
# orders that the CLOB now rejects with "invalid order version").
from polymarket import BuilderApiKey, SecureClient
from polymarket.errors import RateLimitError, UnexpectedResponseError

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
_CTF_ADDRESS_1155 = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
_ERC1155_MIN_ABI = [
    {"constant": True,
     "inputs": [{"name": "owner", "type": "address"},
                {"name": "id", "type": "uint256"}],
     "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
]

_ERC20_MIN_ABI = [
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
     "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [],
     "name": "decimals", "outputs": [{"name": "", "type": "uint8"}],
     "stateMutability": "view", "type": "function"},
]

_executor = ThreadPoolExecutor(
    max_workers=_SIGNER_THREADS, thread_name_prefix="clob-worker"
)

# ── Market-meta cache (tick size + neg-risk) ─────────────────────────────────
# The V2 SDK re-fetches /tick-size and /neg-risk from the CLOB on EVERY order
# build (prepare_{limit,market}_order_draft_sync) and again on every post
# (place.py), with no cache of its own.  Measured from the Montreal box that is
# ~112 ms of blocking network I/O per signature — larger than the whole arb
# window we are trying to hit (measured: 119.6 ms cold vs 7.8 ms primed).
#
# Both values are immutable properties of a market, and the scanner already
# fetches them from Gamma (`orderPriceMinTickSize` / `negRisk`), so priming this
# cache on market admission costs zero extra requests.  Verified against the
# CLOB's authoritative /markets/{condition_id} over the top-40 markets by 24h
# volume: 40/40 exact agreement on tick size, neg-risk and token IDs.
_MARKET_META: dict[str, tuple[Decimal, bool]] = {}
_META_LOCK = threading.Lock()

# Set once _install_market_meta_cache() has patched the SDK fetchers.
_META_CACHE_INSTALLED = False


def prime_market_meta(market: dict) -> int:
    """
    Populate the market-meta cache from a Gamma market dict.

    Returns the number of token IDs primed (0 when the dict lacks the fields —
    callers treat that as a no-op, never an error).
    """
    raw_tokens = market.get("clobTokenIds")
    tick_raw   = market.get("orderPriceMinTickSize")
    if raw_tokens is None or tick_raw is None:
        return 0

    try:
        token_ids = (
            json.loads(raw_tokens) if isinstance(raw_tokens, str) else list(raw_tokens)
        )
        # str() first: Decimal(0.001) from a float carries binary-float noise.
        tick = Decimal(str(tick_raw))
    except (ValueError, TypeError, ArithmeticError, json.JSONDecodeError):
        return 0
    if tick <= 0:
        return 0

    neg_risk = bool(market.get("negRisk"))
    primed = 0
    with _META_LOCK:
        for tid in token_ids:
            tid = str(tid).strip()
            if tid:
                _MARKET_META[tid] = (tick, neg_risk)
                primed += 1
    return primed


def peek_market_meta(token_id: str) -> tuple[Decimal, bool] | None:
    """Synchronous cache peek — never fetches.  None on miss."""
    return _MARKET_META.get(str(token_id))


def market_meta_cache_size() -> int:
    """Number of token IDs currently cached (diagnostics / tests)."""
    return len(_MARKET_META)


def clear_market_meta_cache() -> None:
    """Drop every cached entry (tests only)."""
    with _META_LOCK:
        _MARKET_META.clear()


def _install_market_meta_cache() -> None:
    """
    Wrap the SDK's tick-size / neg-risk fetchers with a cache lookup.

    The consuming modules bind these names at import time
    (`from ...market_data import fetch_tick_size_sync`), so patching
    `market_data` alone would not take effect — each consumer's own namespace
    has to be rebound.

    Only the *sync* variants are patched: PolyClient drives the synchronous
    SecureClient through `_executor`, so the async paths are never reached.
    A miss falls through to the original fetcher and stores its result, which
    keeps behaviour identical when the cache has not been primed.
    """
    global _META_CACHE_INSTALLED
    if _META_CACHE_INSTALLED:
        return

    try:
        from polymarket._internal.actions.orders import (  # noqa: PLC0415
            estimate as _estimate,
            limit as _limit,
            market as _market,
            market_data as _market_data,
            place as _place,
        )
    except ImportError as exc:  # SDK layout changed — degrade to no cache.
        logger.warning(
            "Market-meta cache NOT installed (SDK layout changed: %s) — order "
            "signing keeps its per-call network round-trips", exc,
        )
        return

    def _wrap_tick(orig):
        @wraps(orig)
        def wrapper(ctx, *, token_id: str):
            hit = _MARKET_META.get(str(token_id))
            if hit is not None:
                return hit[0]
            tick = orig(ctx, token_id=token_id)
            with _META_LOCK:
                _, neg = _MARKET_META.get(str(token_id), (tick, False))
                _MARKET_META[str(token_id)] = (tick, neg)
            return tick
        return wrapper

    def _wrap_neg(orig):
        @wraps(orig)
        def wrapper(ctx, *, token_id: str):
            hit = _MARKET_META.get(str(token_id))
            if hit is not None:
                return hit[1]
            neg = orig(ctx, token_id=token_id)
            with _META_LOCK:
                tick, _ = _MARKET_META.get(str(token_id), (TICK_SIZE_FALLBACK, neg))
                _MARKET_META[str(token_id)] = (tick, neg)
            return neg
        return wrapper

    tick_cached = _wrap_tick(_market_data.fetch_tick_size_sync)
    neg_cached  = _wrap_neg(_market_data.fetch_neg_risk_sync)

    # Rebind in every module that imported the names directly.
    for mod in (_market_data, _limit, _market, _place, _estimate):
        if hasattr(mod, "fetch_tick_size_sync"):
            mod.fetch_tick_size_sync = tick_cached
        if hasattr(mod, "fetch_neg_risk_sync"):
            mod.fetch_neg_risk_sync = neg_cached

    _META_CACHE_INSTALLED = True
    logger.info(
        "Market-meta cache installed — /tick-size and /neg-risk are served from "
        "cache on the order hot path"
    )


# Fallback used only when neg-risk is cached before tick size (never on the
# primed path, where both land together).  Matches the strategy-wide default.
TICK_SIZE_FALLBACK = Decimal(str(TICK_SIZE))

_install_market_meta_cache()

# ── Paper-trade toggle — evaluated once at module load; never changes at runtime
_PAPER_TRADE: bool = (
    os.environ.get("PAPER_TRADE_MODE", "false").strip().lower() == "true"
)

# ── Fill classification (leg-reconciliation) ──────────────────────────────────
# A leg counts as filled when its order response status is one of these. FOK
# orders either match in full or are killed; "paper" is the simulated fill.
_FILLED_STATUSES: frozenset[str] = frozenset({"matched", "filled", "paper"})
# How many progressively smaller slices to try when a FOK unwind is killed.
_UNWIND_MAX_SLICES: int = int(os.environ.get("UNWIND_MAX_SLICES", 12))

# How many recent account trades to scan when attributing a fill to an order id.
# Fills are looked up seconds after submission, so the answer is always near the
# head of the feed; the cap stops a lookup walking the entire trade history.
_TRADE_SCAN_LIMIT: int = int(os.environ.get("TRADE_SCAN_LIMIT", 200))


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
# Same env-driven per-trade ceiling as binary pairs (MAX_ARB_PAIR_USDC).
from risk.circuit_breaker import MAX_ARB_PAIR_USDC as _MAX_BUNDLE_USDC  # noqa: E402

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

def _classify_sdk_error(exc: Exception) -> int | None:
    """Map V2 SDK exception classes to synthetic HTTP statuses for the retry
    policy (RateLimitError → 429).  Returns None when unrecognised."""
    if isinstance(exc, RateLimitError):
        return 429
    return None


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


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK model object OR a plain dict (test fakes)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _order_resp_to_dict(resp: Any) -> dict:
    """
    Normalise a post_order result (AcceptedOrder | RejectedOrder | dict) to
    the plain dict shape the strategy layer / classify_fills consumes.
    """
    if isinstance(resp, dict):
        return resp
    if _field(resp, "ok", True) is False:      # RejectedOrder
        return {
            "status": "error",
            "error":  str(_field(resp, "message", "rejected")),
            "code":   str(_field(resp, "code", "")),
        }
    return {
        "status":        str(_field(resp, "status", "") or ""),
        "order_id":      str(_field(resp, "order_id", "") or ""),
        "making_amount": _field(resp, "making_amount"),
        "taking_amount": _field(resp, "taking_amount"),
    }


def _open_order_to_dict(order: Any) -> dict:
    """Normalise an OpenOrder model to the dict shape PairGuard polls."""
    if isinstance(order, dict):
        return order
    return {
        "id":            str(_field(order, "id", "") or ""),
        "status":        str(_field(order, "status", "") or ""),
        "size_matched":  float(_field(order, "size_matched", 0.0) or 0.0),
        "original_size": float(_field(order, "original_size", 0.0) or 0.0),
        "price":         float(_field(order, "price", 0.0) or 0.0),
        "side":          str(_field(order, "side", "") or ""),
        "token_id":      str(_field(order, "token_id", "") or ""),
    }


def _normalise_book(book: Any) -> dict:
    """
    Normalise an orderbook response to {"bids": [...], "asks": [...]} with
    float-typed levels.  Accepts a legacy dict or a modern OrderBookSummary
    dataclass whose levels are OrderSummary(price=str, size=str) objects.
    """
    def _levels(raw: Any) -> list[dict]:
        out: list[dict] = []
        for lvl in raw or []:
            if isinstance(lvl, dict):
                price, size = lvl.get("price"), lvl.get("size")
            else:
                price = getattr(lvl, "price", None)
                size  = getattr(lvl, "size",  None)
            try:
                out.append({"price": float(price), "size": float(size)})
            except (TypeError, ValueError):
                continue
        return out

    if isinstance(book, dict):
        bids, asks = book.get("bids"), book.get("asks")
    else:
        bids = getattr(book, "bids", None)
        asks = getattr(book, "asks", None)
    return {"bids": _levels(bids), "asks": _levels(asks)}


def _backoff_secs(attempt: int) -> float:
    """Exponential backoff with ±JITTER_FACTOR random jitter."""
    raw = min(_BASE_BACKOFF * (2.0 ** attempt), _MAX_BACKOFF)
    return raw * (1.0 + _JITTER_FACTOR * (random.random() * 2.0 - 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# PolyClient
# ═══════════════════════════════════════════════════════════════════════════════

def _is_untracked_order(exc: Exception) -> bool:
    """
    True when the SDK raised purely because the CLOB returned a null body.

    The CLOB answers `null` for an order it no longer tracks. The SDK feeds that
    None into model_validate() and raises UnexpectedResponseError instead of
    returning None.

    NOTE: "untracked" means filled OR cancelled — the response cannot tell them
    apart. Callers must resolve the ambiguity against the chain (see
    PolyClient.share_balance); assuming either way is unsafe.

    Narrow on purpose: only a validation failure whose offending input is None
    counts, so genuine schema drift still surfaces as an error.
    """
    cause  = getattr(exc, "__cause__", None)
    errors = getattr(cause, "errors", None)
    if not callable(errors):
        return False
    try:
        rows = errors()
    except Exception:  # noqa: BLE001
        return False
    return bool(rows) and all(r.get("input", "sentinel") is None for r in rows)


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
        # Polygon is POA-style: without this middleware every get_block()
        # (EIP-1559 gas estimation) raises ExtraDataLengthError.
        self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
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

        self._client = self._build_client(self._pk, funder)

    # ──────────────────────────────────────────────────────────────────────────
    # Client factory (also used for credential re-derivation on 401)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_client(pk: str, funder: str) -> "SecureClient":
        """
        Build an authenticated V2 SecureClient.

        Preferred: the DEPOSIT WALLET flow (wallet=None + builder API key) —
        the V2 exchange rejects direct-EOA makers with "maker address not
        allowed, please use the deposit wallet flow".  The EOA remains the
        signer; collateral and positions live in the derived Deposit Wallet,
        and relayer transactions (approvals/merges) are gasless.

        Falls back to the direct-EOA flow when no builder key is configured
        (POLYMARKET_BUILDER_KEY/SECRET/PASSPHRASE) — orders will be rejected
        by the exchange, so this is only useful for read paths.
        """
        b_key  = os.environ.get("POLYMARKET_BUILDER_KEY", "").strip()
        b_sec  = os.environ.get("POLYMARKET_BUILDER_SECRET", "").strip()
        b_pass = os.environ.get("POLYMARKET_BUILDER_PASSPHRASE", "").strip()
        try:
            if b_key and b_sec and b_pass:
                client = SecureClient.create(
                    private_key=pk,
                    api_key=BuilderApiKey(key=b_key, secret=b_sec,
                                          passphrase=b_pass),
                )
            else:
                logger.warning(
                    "No POLYMARKET_BUILDER_KEY configured — falling back to "
                    "direct-EOA flow; the V2 exchange will REJECT orders."
                )
                client = SecureClient.create(private_key=pk, wallet=funder)
            logger.info(
                "Polymarket SecureClient ready | trading_wallet=%s type=%s signer=%s",
                str(getattr(client, "wallet", "?"))[:12],
                getattr(client, "wallet_type", "?"),
                funder[:10],
            )
            return client
        except Exception as exc:
            logger.error("Failed to build SecureClient: %s", exc)
            raise

    @property
    def trading_wallet(self) -> str:
        """
        The address that holds collateral and positions on the V2 stack —
        the Deposit Wallet in the preferred flow (NOT the signer EOA).
        Balance-scanning components must target this address.
        """
        return str(getattr(self._client, "wallet", "") or self._funder)

    async def share_balance(self, token_id: str) -> "float | None":
        """
        On-chain share balance for a CLOB token — ground truth for "did it fill".

        A Polymarket CLOB token_id IS the ERC-1155 position id on the
        ConditionalTokens contract, so balanceOf(wallet, token_id) is the
        authoritative count of shares held.

        This exists because order status alone cannot answer the question. The
        CLOB returns a null body for any order it no longer tracks, which covers
        BOTH "fully filled" and "cancelled" — indistinguishable from the
        response. Guessing either way is harmful: assuming cancelled leaves a
        filled leg naked, assuming filled fabricates P&L and triggers unwinds of
        shares that do not exist (observed live 2026-09-04). The chain knows.

        Returns None on failure so callers can fall back rather than guess.
        """
        def _read() -> float:
            ctf = self._w3.eth.contract(
                address=Web3.to_checksum_address(_CTF_ADDRESS_1155),
                abi=_ERC1155_MIN_ABI,
            )
            wallet = Web3.to_checksum_address(self.trading_wallet)
            raw    = ctf.functions.balanceOf(wallet, int(token_id)).call()
            return raw / 1e6      # CTF positions carry 6 decimals, like the collateral

        try:
            return await asyncio.get_running_loop().run_in_executor(_executor, _read)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Share balance query failed for %s: %s", str(token_id)[:16], exc
            )
            return None

    async def order_filled_size(
        self, order_id: str, token_id: "str | None" = None
    ) -> "float | None":
        """
        How much of ONE order actually filled, attributed by order id.

        This is the correct answer to "did my order fill", and `share_balance`
        is not. A wallet balance is a WALLET-WIDE quantity: it includes shares
        bought by earlier trades, shares from other strategies, and — the case
        that broke us — fills belonging to other bundles resting on the same
        token at the same moment. Every guard reading it credited itself with
        every other guard's fills plus whatever inventory already existed, then
        "unwound" positions it never opened.

        The CLOB trade feed carries the attribution instead. For each trade the
        account took part in, our order id appears either as `taker_order_id`
        (we crossed) or inside `maker_orders` with the exact `matched_amount`
        that order contributed (we were rested). Summing those is exact,
        immune to pre-existing inventory, and immune to concurrency.

        Returns None if the feed cannot be read, so callers keep whatever
        conservative fallback they had rather than treating "unknown" as zero.
        """
        if _PAPER_TRADE:
            return None

        oid = str(order_id or "").strip().lower()
        if not oid:
            return None

        def _read() -> float:
            total = 0.0
            kwargs = {"token_id": str(token_id)} if token_id else {}
            pages  = self._client.list_account_trades(**kwargs)
            seen   = 0
            for page in pages:
                for tr in page.items:
                    seen += 1
                    # A trade the exchange failed to settle transfers nothing.
                    if str(getattr(tr, "status", "")).upper() == "FAILED":
                        continue
                    if str(getattr(tr, "taker_order_id", "")).lower() == oid:
                        total += float(tr.size)
                    for mo in getattr(tr, "maker_orders", ()) or ():
                        if str(getattr(mo, "order_id", "")).lower() == oid:
                            total += float(mo.matched_amount)
                if seen >= _TRADE_SCAN_LIMIT:
                    break
            return total

        try:
            return await asyncio.get_running_loop().run_in_executor(_executor, _read)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Trade lookup failed for order %s: %s", oid[:12], exc
            )
            return None

    async def open_positions_detail(self) -> "list[dict] | None":
        """
        Current share positions held by the trading wallet, with market names.

        Reads Polymarket's data API, which already returns the human-readable
        `title` per position, so the daily summary can say "Fed rate hike in
        2026?" rather than a bare condition ID.

        Returns a list of {title, outcome, size, avg_price, cur_price, value,
        pnl}, or None if the query fails.
        """
        url    = "https://data-api.polymarket.com/positions"
        params = {"user": self.trading_wallet, "sizeThreshold": 0.0001}
        try:
            import httpx  # noqa: PLC0415 — optional at import time
            async with httpx.AsyncClient(timeout=20.0) as http:
                resp = await http.get(url, params=params)
            if resp.status_code != 200:
                logger.warning(
                    "Positions query returned HTTP %d", resp.status_code
                )
                return None
            rows = resp.json()
            if not isinstance(rows, list):
                return None
            out: list[dict] = []
            for r in rows:
                out.append({
                    "title":     str(r.get("title") or "")[:80],
                    "outcome":   str(r.get("outcome") or ""),
                    "size":      float(r.get("size") or 0.0),
                    "avg_price": float(r.get("avgPrice") or 0.0),
                    "cur_price": float(r.get("curPrice") or 0.0),
                    "value":     float(r.get("currentValue") or 0.0),
                    "pnl":       float(r.get("cashPnl") or 0.0),
                })
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("Positions query failed: %s", exc)
            return None

    async def collateral_balance(self) -> "float | None":
        """
        Read the ACTUAL collateral (pUSD) balance held by the trading wallet.

        The circuit breaker anchors its drawdown maths to this figure, so it
        must reflect the wallet rather than a hand-maintained .env constant.
        The collateral token address comes from the SDK's environment config,
        so a future protocol migration does not silently read the wrong token.

        Returns None on any failure so the caller can fall back to .env.
        """
        def _read() -> float:
            ctx = getattr(self._client, "_ctx", None)
            cfg = getattr(ctx, "environment_config", None)
            token_addr = getattr(cfg, "collateral_token", None)
            if not token_addr:
                raise RuntimeError("collateral_token missing from SDK env config")
            erc20 = self._w3.eth.contract(
                address=Web3.to_checksum_address(token_addr),
                abi=_ERC20_MIN_ABI,
            )
            addr = Web3.to_checksum_address(self.trading_wallet)
            raw  = erc20.functions.balanceOf(addr).call()
            dec  = erc20.functions.decimals().call()
            return raw / (10 ** dec)

        try:
            return await asyncio.get_running_loop().run_in_executor(_executor, _read)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Collateral balance query failed: %s", exc)
            return None

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

    async def _submit_pair(self, yes_coro, no_coro) -> tuple[dict, dict]:
        """
        Submit both legs concurrently WITHOUT losing the surviving leg when
        one submission raises.

        A plain asyncio.gather() propagates the first exception and discards
        the sibling result — if the sibling order was accepted, it would live
        on the book with nobody tracking it (orphaned order / naked exposure).
        Instead, a failed leg is normalised to {"status": "error"} so
        classify_fills() treats it as unfilled and the caller's half-fill /
        PairGuard handling resolves the surviving leg.

        Raises only when BOTH legs failed (nothing usable to track).
        """
        yes_resp, no_resp = await asyncio.gather(
            yes_coro, no_coro, return_exceptions=True
        )
        if isinstance(yes_resp, BaseException) and isinstance(no_resp, BaseException):
            raise yes_resp

        def _norm(resp: Any, label: str) -> dict:
            if isinstance(resp, BaseException):
                logger.error(
                    "%s leg submission FAILED (sibling leg survives — "
                    "half-fill handling takes over): %s", label, resp,
                )
                return {"status": "error", "error": str(resp)}
            return resp

        return _norm(yes_resp, "YES"), _norm(no_resp, "NO")

    def _create_limit(self, *, token_id: str, price: float, size: float,
                      side: str) -> Any:
        """Sign a GTC limit order (resolved on the live client at call time)."""
        return self._client.create_limit_order(
            token_id=token_id, price=price, size=size, side=side.upper(),
        )

    def _create_market(self, *, token_id: str, side: str, shares: float,
                       price: float | None) -> Any:
        """
        Sign an FOK market order for `shares`.

        `price` acts as the marketable limit: max price for BUY, min price
        for SELL.  None → let the SDK derive it from the live book.
        """
        side = side.upper()
        kwargs: dict[str, Any] = {
            "token_id": token_id, "side": side,
            "shares": shares, "order_type": "FOK",
        }
        if price is not None and price > 0:
            kwargs["max_price" if side == "BUY" else "min_price"] = price
        return self._client.create_market_order(**kwargs)

    def _post(self, signed_order: Any) -> dict:
        """Submit a signed order and normalise the response to a dict."""
        return _order_resp_to_dict(self._client.post_order(signed_order))

    async def _post_once(self, signed_order: Any) -> dict:
        """
        Submit a signed order WITHOUT the retry wrapper.

        Market (FOK) orders are not idempotent: a retry re-submits the same
        signed order (same salt) and the exchange rejects it as "invalid.
        Duplicated" — or, worse, a spurious retry after a silent success
        double-executes.  Single-shot posting is the safe path for
        market-order flows (unwind / emergency close / taker completion).
        """
        return await self._run(self._post, signed_order)

    async def _rederive_creds(self) -> None:
        """Rebuild the SecureClient in response to a 401.  Thread-safe."""
        logger.warning("CLOB 401 — rebuilding SecureClient / credentials")
        new_client = await self._run(self._build_client, self._pk, self._funder)
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
                status = _classify_sdk_error(exc) or _extract_http_status(exc)

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

    async def get_orderbook(self, token_id: str) -> dict:
        """
        Return the current L2 orderbook for a token as a plain dict:
            {"bids": [{"price": float, "size": float}, …], "asks": […]}

        Modern py-clob-client returns an OrderBookSummary dataclass (with
        str-typed price levels); older SDKs returned a dict.  Normalise both
        so callers (post_maker_order, MakerPairGuard) can rely on dict access.
        """
        book = await self._run_with_retry(
            lambda: self._client.get_order_book(token_id=token_id)
        )
        return _normalise_book(book)

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — order management
    # ──────────────────────────────────────────────────────────────────────────

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order by ID."""
        resp = await self._run_with_retry(
            lambda: self._client.cancel_order(order_id=order_id)
        )
        return {
            "canceled":     list(_field(resp, "canceled", []) or []),
            "not_canceled": _field(resp, "not_canceled", {}) or {},
        }

    async def get_open_orders(self) -> list[dict]:
        """Return all open orders for the authenticated account."""
        orders = await self._run_with_retry(
            lambda: list(self._client.list_open_orders())
        )
        return [_open_order_to_dict(o) for o in orders]

    async def get_order_status(self, order_id: str) -> dict | None:
        """
        Fetch the current status of a single order.

        Returns the order dict if found, or None if the order is no longer
        tracked (fully filled or cancelled / unknown to the CLOB).
        """
        def _fetch():
            try:
                return self._client.get_order(order_id=order_id)
            except UnexpectedResponseError as exc:
                # Null body: the CLOB no longer tracks this order. Return None
                # rather than burning five retries — but None means "filled OR
                # cancelled", so the caller must check the chain to tell which.
                if _is_untracked_order(exc):
                    return None
                raise

        try:
            order = await self._run_with_retry(_fetch)
        except ClobApiError as exc:
            if exc.status_code == 404:
                return None   # order fully consumed or never existed
            raise
        if order is None:
            return None
        return _open_order_to_dict(order)

    async def cancel_all_orders(self) -> dict:
        """Cancel all open orders for the authenticated account."""
        if _PAPER_TRADE:
            logger.info("PAPER CANCEL ALL | no real orders to cancel")
            return {"status": "paper", "cancelled": 0}

        result = await self._run_with_retry(self._client.cancel_all)
        logger.info("cancel_all_orders result: %s", result)
        return {
            "canceled":     list(_field(result, "canceled", []) or []),
            "not_canceled": _field(result, "not_canceled", {}) or {},
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — single-leg taker order (FOK)
    # ──────────────────────────────────────────────────────────────────────────

    async def post_order(
        self,
        token_id:    str,
        side:        str,
        price:       float,
        size:        float,
        _order_type: str = "FOK",   # retained for API compatibility; FOK only
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

        signed_order = await self._run_with_retry(
            self._create_market,
            token_id=token_id, side=side, shares=size, price=price,
        )
        # Single-shot: market FOK orders are not idempotent (retry → duplicate).
        response = await self._post_once(signed_order)
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
        signed = await self._run_with_retry(
            self._create_limit,
            token_id=token_id, price=safe_price, size=size, side=side,
        )
        response = await self._run_with_retry(self._post, signed)
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
        # Step 1 — sign both concurrently (CPU-bound ECDSA in executor)
        _sign_t0 = time.monotonic()
        yes_signed, no_signed = await asyncio.gather(
            self._run_with_retry(
                self._create_market,
                token_id=yes_token_id, side="BUY", shares=yes_size, price=yes_price,
            ),
            self._run_with_retry(
                self._create_market,
                token_id=no_token_id, side="BUY", shares=no_size, price=no_price,
            ),
        )
        SIGN_SECONDS.observe(time.monotonic() - _sign_t0)

        # Step 2 — submit both FOK orders simultaneously.  _submit_pair keeps
        # the surviving leg when one submission errors (orphan prevention).
        _submit_t0 = time.monotonic()
        yes_resp, no_resp = await self._submit_pair(
            self._run_with_retry(self._post, yes_signed),
            self._run_with_retry(self._post, no_signed),
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

        # Floor to 2 d.p. — never round UP, or the sell exceeds the on-chain
        # balance ("not enough balance / allowance").
        sell_size = math.floor(size * 100) / 100.0
        if sell_size < 0.01:
            logger.warning("UNWIND skipped | size %.4f below minimum", size)
            return {"status": "error", "error": "size below minimum"}
        # A market order here is FOK: fully filled or KILLED. Dumping the whole
        # naked leg into a thin book therefore sells NOTHING and leaves the
        # shares stranded — observed live 2026-09-05, four consecutive failures
        # ("order couldn't be fully filled") that stranded ~35 pUSD.
        #
        # So: cap the first slice to the visible bid depth, and on a kill, halve
        # and retry. Selling SOME of the naked leg always beats selling none.
        try:
            book  = await self.get_orderbook(token_id)
            depth = 0.0
            for lvl in book.get("bids", []):
                try:
                    depth += float(lvl["size"] if isinstance(lvl, dict) else lvl)
                except (KeyError, TypeError, ValueError):
                    continue
            if depth > 0:
                capped = math.floor(min(sell_size, depth) * 100) / 100.0
                if capped < sell_size:
                    logger.warning(
                        "UNWIND | book depth %.2f < naked size %.2f — slicing",
                        depth, sell_size,
                    )
                sell_size = max(0.01, capped)
        except Exception as exc:  # noqa: BLE001
            logger.warning("UNWIND | depth probe failed (%s) — using full size", exc)

        # `outstanding` is what still needs selling; `slice_size` is how much we
        # dare offer in one FOK. A kill halves the slice; a fill keeps it (that
        # size demonstrably clears) and we go round again for the remainder.
        outstanding = sell_size
        slice_size  = sell_size
        sold_total  = 0.0
        proceeds    = 0.0
        last_resp: dict = {}
        for _ in range(_UNWIND_MAX_SLICES):
            if outstanding < 0.01 or slice_size < 0.01:
                break
            attempt = math.floor(min(slice_size, outstanding) * 100) / 100.0
            if attempt < 0.01:
                break
            try:
                signed = await self._run_with_retry(
                    self._create_market,
                    token_id=token_id, side="SELL", shares=attempt,
                    price=price if price > 0 else None,
                )
                # Single-shot post: a retried market SELL duplicates or double-sells.
                resp = last_resp = await self._post_once(signed)
            except Exception as exc:  # noqa: BLE001
                if "fully filled" in str(exc).lower() or "killed" in str(exc).lower():
                    slice_size = math.floor(attempt * 50) / 100.0   # halve, 2 d.p.
                    logger.warning(
                        "UNWIND | FOK killed at %.2f — retrying with %.2f shares",
                        attempt, slice_size,
                    )
                    continue
                raise
            if str(resp.get("status", "")).strip().lower() in _FILLED_STATUSES:
                got = float(resp.get("making_amount") or attempt)
                proceeds    += float(resp.get("taking_amount") or 0.0)
                sold_total  += got
                outstanding  = math.floor((outstanding - got) * 100) / 100.0
                logger.warning(
                    "UNWIND submitted | SELL %.2f of %s (%.2f left) | resp=%s",
                    got, token_id[:16], outstanding, resp,
                )
            else:
                slice_size = math.floor(attempt * 50) / 100.0
                logger.warning(
                    "UNWIND | not filled (%s) — retrying with %.2f shares",
                    resp.get("status"), slice_size,
                )

        if sold_total <= 0.0:
            logger.error(
                "UNWIND FAILED | could not sell any of %.2f shares of %s",
                size, token_id[:16],
            )
            return last_resp or {"status": "error", "error": "unwind sold nothing"}

        if sold_total + 0.005 < size:
            logger.warning(
                "UNWIND PARTIAL | sold %.2f of %.2f shares — %.2f still naked",
                sold_total, size, size - sold_total,
            )
        return {
            "status":        "matched",
            "order_id":      last_resp.get("order_id", ""),
            "making_amount": sold_total,
            "taking_amount": proceeds,
            "partial":       sold_total + 0.005 < size,
        }

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
        The caller MUST register the pair with MakerPairGuard
        (execution/pair_guard.py) unless both legs confirmed filled at ack —
        the guard cancels stale orders and hedges/unwinds any naked leg that
        remains one-sided for > HEDGE_TIMEOUT_S seconds.

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

        # Concurrent ECDSA signing — eliminates the ~1 s temporal drag
        yes_signed, no_signed = await asyncio.gather(
            self._run_with_retry(
                self._create_limit,
                token_id=yes_token_id, price=yes_bid, size=yes_size, side="BUY",
            ),
            self._run_with_retry(
                self._create_limit,
                token_id=no_token_id, price=no_bid, size=no_size, side="BUY",
            ),
        )

        # Concurrent GTC submission.  _submit_pair keeps the surviving leg
        # when one submission errors — the PairGuard then owns its lifecycle.
        yes_resp, no_resp = await self._submit_pair(
            self._run_with_retry(self._post, yes_signed),
            self._run_with_retry(self._post, no_signed),
        )

        logger.info(
            "MAKER ARB PAIR placed | YES order=%s | NO order=%s",
            yes_resp.get("order_id", "?")[:16],
            no_resp.get("order_id",  "?")[:16],
        )
        return yes_resp, no_resp

    # ──────────────────────────────────────────────────────────────────────────
    # Public async API — N-leg NegRisk bundle via CLOB limit orders
    # ──────────────────────────────────────────────────────────────────────────

    async def execute_negrisk_clob_bundle(
        self,
        legs: list[BundleLeg],
    ) -> list[dict]:
        """
        Submit an N-leg NegRisk bundle as N concurrent GTC limit orders.

        This is the supported replacement for `execute_arb_maker_bundle`, whose
        on-chain matchOrders path died with the April 2026 pUSD migration and
        required exchange-operator registration even before that.  Ordinary CLOB
        limit orders need no special wallet role.

        What this method does NOT give you
        ──────────────────────────────────
        Atomicity.  matchOrders either minted the whole bundle or reverted; N
        independent limit orders can fill in any subset.  A partly filled bundle
        is NOT a smaller arbitrage — holding NO on M' of the M selected outcomes
        guarantees only `M'−1` at expiry against a cost of `Σ no_bid`, which goes
        negative as M' shrinks.  arXiv:2508.03474 §6 names this directly: "since
        placing multiple orders in an order book is non-atomic (only a subset of
        the attempts may succeed), there is some inherent risk to attempting
        arbitrage."

        The caller MUST therefore hand every response to NegRiskBundleGuard
        (execution/negrisk_guard.py), which cancels the stragglers and unwinds
        whatever filled if the bundle does not complete inside its timeout.

        `leg.bid` values must already be clamped below their best asks by
        NegRiskArbDetector; this method trusts them as valid post-only bids.

        Returns one response dict per leg, in the order given.  A leg whose
        submission raised is normalised to {"status": "error", ...} rather than
        aborting the batch — a surviving sibling that reached the book must stay
        visible to the guard, never be orphaned.
        """
        if not legs:
            raise ValueError("execute_negrisk_clob_bundle: legs list is empty")

        bundle_cost = sum(leg.bid for leg in legs)          # cost of 1 bundle
        payout      = float(len(legs) - 1)                  # NegRisk floor payout
        if bundle_cost >= payout:
            raise ClobApiError(
                0,
                f"bundle cost {bundle_cost:.6f} USDC ≥ payout {payout:.1f} USDC "
                f"({len(legs)} legs) — not profitable; refusing to submit",
            )

        total_cost = sum(leg.bid * leg.size for leg in legs)
        if total_cost > _MAX_BUNDLE_USDC:
            raise ClobApiError(
                0,
                f"Bundle cost {total_cost:.4f} USDC exceeds "
                f"${_MAX_BUNDLE_USDC:.0f} cap ({len(legs)} legs)",
            )

        if _PAPER_TRADE:
            logger.info(
                "PAPER NEGRISK CLOB BUNDLE | %d legs | total_cost=%.4f USDC "
                "payout=%.1f",
                len(legs), total_cost, payout,
            )
            responses: list[dict] = []
            for i, leg in enumerate(legs):
                responses.append({
                    "status":   "paper",
                    "order_id": f"paper-negrisk-{i}-{uuid.uuid4()}",
                    "token_id": leg.token_id,
                    "side":     "BUY",
                    "price":    leg.bid,
                    "size":     leg.size,
                    "maker":    True,
                    "leg_idx":  i,
                })
                logger.info(
                    "  leg[%d] token=%s bid=%.4f size=%.2f",
                    i, leg.token_id[:12], leg.bid, leg.size,
                )
            return responses

        # ── Concurrent ECDSA signing ──────────────────────────────────────────
        signed = await asyncio.gather(*(
            self._run_with_retry(
                self._create_limit,
                token_id=leg.token_id, price=leg.bid, size=leg.size, side="BUY",
            )
            for leg in legs
        ), return_exceptions=True)

        # ── Concurrent submission ─────────────────────────────────────────────
        # Legs that failed to sign never reach the book; submit only the rest,
        # then stitch the results back into leg order.
        submit_idx = [i for i, s in enumerate(signed)
                      if not isinstance(s, BaseException)]
        posted = await asyncio.gather(*(
            self._run_with_retry(self._post, signed[i]) for i in submit_idx
        ), return_exceptions=True)

        results: list[dict] = [
            {"status": "error", "error": str(s)}
            for s in signed
        ]
        for slot, resp in zip(submit_idx, posted):
            if isinstance(resp, BaseException):
                logger.error(
                    "NegRisk leg[%d] token=%s submission FAILED (siblings may "
                    "have reached the book — guard takes over): %s",
                    slot, legs[slot].token_id[:12], resp,
                )
                results[slot] = {"status": "error", "error": str(resp)}
            else:
                results[slot] = resp

        for i, (leg, resp) in enumerate(zip(legs, results)):
            resp.setdefault("token_id", leg.token_id)
            resp.setdefault("price", leg.bid)
            resp.setdefault("size", leg.size)
            resp["leg_idx"] = i

        ok = sum(1 for r in results if str(r.get("status", "")).lower() != "error")
        logger.info(
            "NEGRISK CLOB BUNDLE placed | %d/%d legs accepted | "
            "total_cost=%.4f USDC payout=%.1f",
            ok, len(legs), total_cost, payout,
        )
        if ok == 0:
            raise ClobApiError(0, "every NegRisk leg failed to submit")
        return results

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
        # Polygon enforces a 25 gwei minimum priority fee — anything lower is
        # rejected by the mempool ("gas tip cap below minimum").
        priority_tip = Web3.to_wei(30, "gwei")
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
        # OBSOLETE since the April 2026 protocol migration (pUSD / new exchange
        # contracts): the V1-era matchOrders call would target a defunct
        # exchange and requires operator registration regardless.  NegRisk
        # execution is gated off by NEGRISK_EXEC_MODE=off; fail loudly if it
        # is ever forced on.
        raise ClobApiError(
            0,
            "execute_arb_maker_bundle live path is disabled: the pre-migration "
            "matchOrders flow is obsolete post-pUSD-upgrade. Route NegRisk "
            "bundles through CLOB limit orders instead.",
        )

        # Step 1 — ECDSA-sign all N order structs concurrently
        signed_orders: list[Any] = list(await asyncio.gather(*[
            self._run_with_retry(
                self._create_limit,
                token_id=leg.token_id, price=leg.bid, size=leg.size, side="BUY",
            )
            for leg in legs
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
    # Public async API — on-chain settlement via the V2 SDK
    # ──────────────────────────────────────────────────────────────────────────

    async def merge_positions(self, condition_id: str, amount_units: int) -> str:
        """
        Burn a complementary YES+NO set and receive collateral (pUSD) back.

        Post-migration the SDK routes this through the correct contracts
        (protocol V2 router / adapters) and, for EOA wallets, submits the
        transaction directly.  Blocking `handle.wait()` runs in the executor.

        `amount_units` is in base units (1e6 per share).  Returns a tx-hash
        string when available ("" otherwise).
        """
        if _PAPER_TRADE:
            logger.info(
                "PAPER SDK merge | condition=%s amount=%d",
                condition_id[:16], amount_units,
            )
            return "paper-merge"

        def _merge() -> str:
            handle = self._client.merge_positions(
                condition_id=condition_id, amount=amount_units,
            )
            outcome = handle.wait()
            tx = (
                _field(outcome, "transaction_hash", None)
                or _field(outcome, "tx_hash", None) or ""
            )
            logger.info(
                "SDK merge confirmed | condition=%s amount=%d tx=%s",
                condition_id[:16], amount_units, str(tx)[:20],
            )
            return str(tx)

        return await self._run_with_retry(_merge)

    async def redeem_positions(self, condition_id: str) -> str:
        """
        Redeem winnings on a resolved market via the V2 SDK.

        Returns a tx-hash string when available ("" otherwise).
        """
        if _PAPER_TRADE:
            logger.info("PAPER SDK redeem | condition=%s", condition_id[:16])
            return "paper-redeem"

        def _redeem() -> str:
            handle = self._client.redeem_positions(condition_id=condition_id)
            outcome = handle.wait()
            tx = (
                _field(outcome, "transaction_hash", None)
                or _field(outcome, "tx_hash", None) or ""
            )
            logger.info(
                "SDK redeem confirmed | condition=%s tx=%s",
                condition_id[:16], str(tx)[:20],
            )
            return str(tx)

        return await self._run_with_retry(_redeem)

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

        signed = await self._run_with_retry(
            self._create_market,
            token_id=token_id, side="SELL", shares=size, price=min_price,
        )
        # Single-shot: market FOK orders are not idempotent (retry → duplicate).
        response = await self._post_once(signed)
        logger.info(
            "Emergency leg close | SELL %s size=%.2f → %s",
            token_id[:12], size, response,
        )
        return response
