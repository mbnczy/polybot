"""
strategy/arbitrage.py
─────────────────────
Low-latency pricing engine — MakerArbitrageMachine 2026 Edition.

Mathematical model — binary markets
─────────────────────────────────────
At expiry, exactly one outcome token pays $1.00 USDC per share; the other
pays $0.00.  Posting MAKER bids on both legs below their current best asks
creates a Dutch Book position: if both fills execute, we are guaranteed to
collect $1.00 while having paid strictly less.

Dutch Book spread (gross, pre-rebate)
──────────────────────────────────────
    yes_bid = best_ask_yes − TICK_SIZE      ← synthetic post-only clamp
    no_bid  = best_ask_no  − TICK_SIZE

    gross_spread = 1.0 − (yes_bid + no_bid)
                 = 1.0 − (best_ask_yes + best_ask_no − 2 × TICK_SIZE)

Mathematical model — NegRisk multi-outcome markets
───────────────────────────────────────────────────
In a market with N mutually exclusive outcomes (O₁ … Oₙ), each outcome i has
a NO token.  At expiry, exactly one outcome wins; N−1 NO tokens pay $1 each.

Strategy: buy 1 NO share per outcome simultaneously (a "bundle").

    no_bid_i      = no_ask_i − TICK_SIZE            ← synthetic post-only
    combined_bid  = Σ no_bid_i                       (sum over all N legs)
    payout        = N − 1                            (USDC at expiry)
    effective_cost = combined_bid × (1 − maker_rebate)
    net_edge      = payout − effective_cost
    relative_edge = net_edge / payout                ← comparable across N

Signal condition (NegRisk maker arb):
    relative_edge > DESIRED_NET_MARGIN
    ⟺ combined_bid < (N−1) × (1 − DESIRED_NET_MARGIN) / (1 − maker_rebate)

2026 Dynamic Maker Rebate System
─────────────────────────────────
Polymarket pays a MAKER REBATE to liquidity providers whose resting orders
get lifted by takers.  The rebate is a percentage of the filled notional and
acts as a COST DEDUCTION, making our effective combined purchase price lower:

    effective_cost = (yes_bid + no_bid) × (1 − maker_rebate_rate)

    net_edge = 1.0 − effective_cost
             = 1.0 − (yes_bid + no_bid) × (1 − maker_rebate_rate)

2026 category-level rebate schedule
─────────────────────────────────────
  politics       0.0100  (1.00 %)   high-volume, lowest spread
  crypto         0.0144  (1.44 %)   highest volatility → highest rebate
  sports         0.0075  (0.75 %)
  entertainment  0.0075  (0.75 %)
  economy        0.0080  (0.80 %)
  science        0.0060  (0.60 %)
  other/unknown  0.0050  (0.50 %)   conservative default

Signal condition (maker arb)
─────────────────────────────
    net_edge > DESIRED_NET_MARGIN
    ⟺ (yes_bid + no_bid) < (1 − DESIRED_NET_MARGIN) / (1 − maker_rebate)

Taker fallback (FeeEngine)
───────────────────────────
For FOK taker arb (used when maker queues are empty or fills are too slow),
the taker fee is a COST ADDITION:

    taker_net_edge = 1.0 − (yes_ask + no_ask) × (1 + taker_fee_rate)

The FeeEngine fetches each market's fee from:
  1. Gamma API  GET /markets?conditionId={id}  →  market.feeRate
  2. CLOB API   GET /markets/{id}              →  takerBaseFee
  3. DEFAULT_TAKER_FEE (conservative 2.0 %)

Both engines cache their rates per condition_id with a 5-minute TTL.

Public contract
───────────────
  MakerRebateEngine
    async get_maker_rebate(condition_id: str) → float
    prime_cache(condition_id, category_slug, fee_rate)

  DutchBookPricer(desired_net_margin, default_rebate_rate)
    evaluate_maker(condition_id, yes_token_id, no_token_id,
                   yes_ask, no_ask, max_position_usdc,
                   maker_rebate=None) → Optional[ArbSignal]

  FeeEngine(default_fee)
    async get_taker_fee(condition_id: str) → float
    prime_cache(condition_id, fee_rate)

  ArbDetector — backward-compatible taker-arb evaluator (unchanged API)
    evaluate(condition_id, yes_token_id, no_token_id,
             yes_ask, no_ask, max_position_usdc, fee_rate=None) → Optional[ArbSignal]

  NegRiskArbDetector — N-outcome NegRisk maker arb (Phase 9)
    evaluate_neg_risk(condition_id, outcome_token_ids, no_asks,
                      max_position_usdc, maker_rebate=None) → Optional[NegRiskSignal]

  ArbSignal fields (frozen dataclass):
    condition_id    yes_token_id    no_token_id
    yes_ask         no_ask          combined_cost   fee_rate    fee_cost    net_edge
    yes_size        no_size
    yes_bid*        no_bid*         maker_rebate*   maker_net_edge*
    (* = 0.0 for taker-path signals; populated for maker-path signals)

  ArbLeg fields (frozen dataclass, NegRisk per-outcome leg):
    token_id    no_ask    no_bid    size

  NegRiskSignal fields (frozen dataclass):
    condition_id    n_outcomes      legs            combined_bid
    payout          maker_rebate    effective_cost  net_edge
    relative_edge   n_bundles
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from core import market_titles          # dependency-free registry

logger = logging.getLogger(__name__)

# ── Maker rebate schedule — 2026 Polymarket category rates ──────────────────
MAKER_REBATES: dict[str, float] = {
    "politics":     0.0100,   # 1.00 %
    "crypto":       0.0144,   # 1.44 %
    "sports":       0.0075,   # 0.75 %
    "entertainment":0.0075,   # 0.75 %
    "economy":      0.0080,   # 0.80 %
    "science":      0.0060,   # 0.60 %
    # Any category not in this map falls back to DEFAULT_MAKER_REBATE
}
DEFAULT_MAKER_REBATE: float = 0.0050   # 0.50 % — conservative unknown-category default
MAX_MAKER_REBATE:     float = 0.0200   # sanity clamp

# ── Taker fee constants (used by FeeEngine / ArbDetector) ───────────────────
DEFAULT_TAKER_FEE:  float = 0.0200   # 2.00 % — conservative fall-back
MAX_TAKER_FEE:      float = 0.0400   # sanity clamp for implausibly high API values

# ── Common constants ─────────────────────────────────────────────────────────
DESIRED_NET_MARGIN: float = 0.0050   # 0.50 % minimum guaranteed profit per pair
TICK_SIZE:          float = 0.001    # legacy default price increment (0.1 cent)
# Markets have per-market tick sizes (0.01 OR 0.001). Posting a bid that is not
# on the market's tick grid is rejected: "price must conform to tick size 0.01
# with at most 2 decimal places". When the market's tick is unknown we default
# to the COARSER 0.01 grid, which is valid on every market (a 0.01-multiple also
# conforms to a 0.001 grid) — never produces an invalid price.
DEFAULT_TICK_SIZE:  float = 0.01

# Only enter a NegRisk bundle that still clears when the legs we cannot quote
# competitively are paid at the ask. See the realistic-completion filter in
# evaluate_neg_risk. Set false to score bundles at the all-maker price, which
# is what produced a full night of half-fills.
NEGRISK_REQUIRE_COMPLETABLE: bool = os.environ.get(
    "NEGRISK_REQUIRE_COMPLETABLE", "true"
).strip().lower() in ("1", "true", "yes", "on")

# Absolute USDC edge per bundle required at those completion prices.
NEGRISK_MIN_COMPLETABLE_EDGE: float = float(
    os.environ.get("NEGRISK_MIN_COMPLETABLE_EDGE", 0.002)
)


def snap_post_only_bid(ask: float, tick: float | None) -> float:
    """
    One market-tick below `ask`, snapped to the market's tick grid.

    tick 0.01 → 2-decimal price (e.g. ask 0.49 → bid 0.48)
    tick 0.001 → 3-decimal price (e.g. ask 0.49 → bid 0.489)
    Unknown tick → 0.01 grid (safe on every market).
    Result is clamped to the valid [tick, 1 − tick] price range.
    """
    t = tick if (tick and tick > 0) else DEFAULT_TICK_SIZE
    decimals = 2 if t >= 0.01 else 3
    bid = round(ask - t, decimals)
    lo, hi = t, round(1.0 - t, decimals)
    return max(lo, min(bid, hi))

# ── Signal-quality guards (efficiency-and-reliability) ───────────────────────
# Near-resolved markets (one outcome ~certain) produce "signals" whose edge is
# dominated by the maker-rebate subsidy on thin, rarely-filled books — not a
# durable, capturable price gap.  Skip a market when either leg's ask sits
# outside [EXTREME_PRICE_LO, EXTREME_PRICE_HI].  Default band is wide enough to
# leave all genuinely contested markets untouched.
EXTREME_PRICE_LO: float = 0.05   # skip if yes_ask or no_ask < this (≈ resolved)
EXTREME_PRICE_HI: float = 0.95   # skip if yes_ask or no_ask > this (≈ resolved)

# Minimum *real* (pre-rebate) Dutch-book gap on the maker path, measured on the
# observed asks: real_edge = 1 − (yes_ask + no_ask).  0.0 = OFF (legacy: allow
# rebate-funded entries where combined asks may exceed $1).  Set > 0 (e.g.
# 0.002 = 20 bps) to require a genuine sub-$1 combined cost so the rebate is
# upside, not the sole justification.
MIN_REAL_EDGE: float = 0.0

# ── NegRisk selection heuristics ─────────────────────────────────────────────
# Empirical defaults from Saguillo, Ghafouri, Kiffer & Suarez-Tangil,
# "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
# (arXiv:2508.03474), which measured a year of Polymarket order-book history.
#
# Why each one matters for the buy-NO-on-every-outcome bundle:
#
#   NEGRISK_MIN_OUTCOME_PROB — §6.2 sizes bundles from "the minimum volume
#     across all conditions that have a probability more than 2%".  Dropping a
#     near-zero-probability outcome is *risk-free* here (see the subset lemma in
#     evaluate_neg_risk) but frees ~$1 of capital per dropped leg, because a
#     3%-probability outcome sells NO at ~$0.97 while contributing only ~$0.03
#     of edge.  Pure capital-efficiency win.
#
#   NEGRISK_MAX_LEGS — §5.1 reduces every market to its top-4 conditions,
#     showing "over 90% of all liquidity in a market resides in the top 4
#     conditions".  It also bounds the paper's own caveat that "placing multiple
#     orders in an order book is non-atomic (only a subset of the attempts may
#     succeed)": leg count is exactly the number of ways a bundle can go
#     half-filled, and live NegRisk groups run to 51 outcomes.
#
#   NEGRISK_MIN_RELATIVE_EDGE — §6 restricts the study "to opportunities with a
#     profit of at least $0.05 on the dollar to focus on the higher-reward
#     opportunities given the risk".  Their profit-on-the-dollar is
#     net_edge / payout, i.e. exactly our `relative_edge`.  This is a NegRisk-
#     only floor and deliberately 10× the binary DESIRED_NET_MARGIN — the
#     binary path is atomic per pair, an N-leg bundle is not.
#
#   NEGRISK_MIN_LEG_SHARES — not from the paper: Polymarket's Gamma metadata
#     reports orderMinSize = 5 on live NegRisk outcomes, so a bundle sized below
#     5 shares has every leg rejected at submission.
NEGRISK_MIN_OUTCOME_PROB:  float = 0.02   # paper §6.2 — ignore <2 % outcomes
NEGRISK_MAX_LEGS:          int   = 4      # paper §5.1 — top-4 hold >90 % liquidity
NEGRISK_MIN_RELATIVE_EDGE: float = 0.05   # paper §6   — $0.05 on the dollar
NEGRISK_MIN_LEG_SHARES:    float = 5.0    # Gamma orderMinSize on live markets


def _within_quality_band(
    yes_ask: float,
    no_ask:  float,
    lo:      float,
    hi:      float,
) -> bool:
    """
    True iff BOTH legs' asks sit inside [lo, hi] — i.e. the market is genuinely
    contested rather than near-resolved.  Shared by the taker (ArbDetector) and
    maker (DutchBookPricer) paths so the quality rule lives in exactly one place.
    """
    return lo <= yes_ask <= hi and lo <= no_ask <= hi


# ── API endpoints ─────────────────────────────────────────────────────────────
_CLOB_HOST   = "https://clob.polymarket.com"
_GAMMA_HOST  = "https://gamma-api.polymarket.com"
_CACHE_TTL   = 300.0   # 5-minute cache TTL for both engines


# ═══════════════════════════════════════════════════════════════════════════════
# MakerRebateEngine
# ═══════════════════════════════════════════════════════════════════════════════

class MakerRebateEngine:
    """
    Fetches and caches per-market maker rebate rates.

    Resolution order for a cache miss:
      1. Gamma API  /markets?conditionId={id}  →  market.category (slug)
                                                   then MAKER_REBATES[category]
      2. DEFAULT_MAKER_REBATE

    Cache policy:  each condition_id is cached for CACHE_TTL seconds.

    Usage::

        engine   = MakerRebateEngine()
        rebate   = await engine.get_maker_rebate("0xabc…")
        # e.g. 0.0144 for a crypto market
    """

    def __init__(self, default_rebate: float = DEFAULT_MAKER_REBATE) -> None:
        self._default = max(0.0, min(default_rebate, MAX_MAKER_REBATE))
        # condition_id → (rebate_rate, cached_at_monotonic)
        self._cache: dict[str, tuple[float, float]] = {}
        # Persistent keep-alive session — reused across cold-miss fetches (e.g.
        # cache TTL expiry) to avoid a fresh TCP/TLS handshake each time.
        self._session: "aiohttp.ClientSession | None" = None

    def _get_session(self) -> "aiohttp.ClientSession":
        if self._session is None or getattr(self._session, "closed", False):
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the keep-alive session (call on shutdown)."""
        if self._session is not None and not getattr(self._session, "closed", False):
            await self._session.close()

    async def get_maker_rebate(self, condition_id: str) -> float:
        """
        Return the maker rebate rate for a market as a fraction (0.0144 = 1.44 %).
        Returns cached value if fresh; otherwise fetches from Gamma API.
        """
        cached = self._cache.get(condition_id)
        if cached is not None:
            rate, ts = cached
            if time.monotonic() - ts < _CACHE_TTL:
                return rate

        rate = await self._fetch_rebate(condition_id)
        self._cache[condition_id] = (rate, time.monotonic())
        return rate

    def peek_maker_rebate(self, condition_id: str) -> float | None:
        """
        Synchronous cache peek — returns the fresh cached rebate or None.

        Never fetches.  Lets the hot path avoid an event-loop bounce
        (`await`) when the cache is warm (the common case after pre-warming).
        """
        cached = self._cache.get(condition_id)
        if cached is not None:
            rate, ts = cached
            if time.monotonic() - ts < _CACHE_TTL:
                return rate
        return None

    def prime_cache(
        self,
        condition_id:  str,
        category_slug: str | None = None,
        rebate_rate:   float | None = None,
    ) -> None:
        """
        Pre-populate cache (e.g. from config file or test fixture).
        If rebate_rate is given it is used directly; otherwise the category slug
        is resolved against MAKER_REBATES.
        """
        if rebate_rate is not None:
            rate = max(0.0, min(rebate_rate, MAX_MAKER_REBATE))
        elif category_slug is not None:
            rate = _resolve_rebate(category_slug)
        else:
            rate = self._default
        self._cache[condition_id] = (rate, time.monotonic())

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    async def _fetch_rebate(self, condition_id: str) -> float:
        """Query Gamma API for market category, then look up the rebate."""
        try:
            session = self._get_session()
            async with session.get(
                f"{_GAMMA_HOST}/markets",
                params={"conditionId": condition_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return self._default
                data = await resp.json(content_type=None)

            markets = data if isinstance(data, list) else data.get("data", [])
            if not markets:
                return self._default

            market   = markets[0]
            category = (
                market.get("category")
                or market.get("market_type")
                or market.get("type")
                or ""
            )
            rate = _resolve_rebate(str(category))
            logger.debug(
                "MakerRebateEngine | %s → category=%r rebate=%.4f (%.2f%%)",
                condition_id[:16], category, rate, rate * 100,
            )
            return rate

        except Exception as exc:
            logger.debug(
                "MakerRebateEngine | Gamma API error for %s: %s",
                condition_id[:16], exc,
            )
            return self._default


def _resolve_rebate(category: str) -> float:
    """Map a category string to the closest MAKER_REBATES entry."""
    slug = category.strip().lower()
    # Direct match
    if slug in MAKER_REBATES:
        return MAKER_REBATES[slug]
    # Partial match (e.g. "us-politics" → "politics")
    for key, rate in MAKER_REBATES.items():
        if key in slug or slug in key:
            return rate
    return DEFAULT_MAKER_REBATE


# ═══════════════════════════════════════════════════════════════════════════════
# ArbSignal (extended for both taker and maker paths)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ArbSignal:
    """
    Describes a viable arbitrage opportunity on a Polymarket binary market.

    Populated by either:
      • ArbDetector.evaluate()     — taker path  (yes_bid/no_bid = 0.0)
      • DutchBookPricer.evaluate_maker() — maker path  (yes_bid/no_bid set)

    All prices are USDC per share (range 0.01 – 0.99).
    Sizes are number of shares (Polymarket minimum increment 0.01).
    """
    condition_id:   str
    yes_token_id:   str
    no_token_id:    str
    yes_ask:        float   # best ask for YES leg  (taker entry point)
    no_ask:         float   # best ask for NO leg
    combined_cost:  float   # yes_ask + no_ask  (gross cost per share-pair)
    fee_rate:       float   # taker fee rate applied  (fraction, e.g. 0.02 = 2 %)
    fee_cost:       float   # combined_cost × fee_rate  (taker fee USDC per pair)
    net_edge:       float   # 1 − combined_cost × (1 + fee_rate)  — taker net profit
    yes_size:       float   # shares to buy / bid on YES leg
    no_size:        float   # shares to buy / bid on NO leg

    # ── Maker-path fields (populated by DutchBookPricer; 0.0 for taker signals)
    yes_bid:        float = 0.0   # synthetic post-only bid for YES (ask − TICK)
    no_bid:         float = 0.0   # synthetic post-only bid for NO  (ask − TICK)
    maker_rebate:   float = 0.0   # maker rebate rate for this market
    maker_net_edge: float = 0.0   # 1 − (yes_bid+no_bid)×(1−maker_rebate)

    @property
    def is_maker_signal(self) -> bool:
        return self.yes_bid > 0.0 and self.no_bid > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# NegRisk data structures (Phase 9)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ArbLeg:
    """
    A single NO-side leg in a NegRisk multi-outcome arbitrage bundle.

    Produced by `NegRiskArbDetector.evaluate_neg_risk()`.  Pass token_id,
    no_bid, and size to `BundleLeg` (core/clob_client.py) when submitting.
    """
    token_id: str
    no_ask:   float   # best ask from WS feed
    no_bid:   float   # synthetic post-only bid (no_ask − TICK_SIZE)
    size:     float   # shares per bundle (= n_bundles)


@dataclass(frozen=True, slots=True)
class NegRiskSignal:
    """
    Describes a viable NegRisk arbitrage opportunity across N mutually
    exclusive outcomes.

    Produced by `NegRiskArbDetector.evaluate_neg_risk()`.

    All costs/profits are USDC per bundle (one NO share per outcome).

    Fields
    ------
    condition_id   : Polymarket condition ID of the parent market.
    n_outcomes     : Number of outcomes (= len(legs)).
    legs           : Per-outcome ArbLeg tuple, order preserved from input.
    combined_bid   : Σ leg.no_bid — total USDC spent per bundle.
    payout         : N − 1 — USDC collected at expiry per bundle.
    maker_rebate   : Rebate rate applied (fraction, e.g. 0.0100 = 1 %).
    effective_cost : combined_bid × (1 − maker_rebate).
    net_edge       : payout − effective_cost  (absolute USDC profit / bundle).
    relative_edge  : net_edge / payout  (comparable to binary DESIRED_NET_MARGIN).
    n_bundles      : Bundles fitting within max_position_usdc (floor to 2 d.p.).
    """
    condition_id:   str
    n_outcomes:     int
    legs:           tuple[ArbLeg, ...]
    combined_bid:   float
    payout:         float
    maker_rebate:   float
    effective_cost: float
    net_edge:       float
    relative_edge:  float
    n_bundles:      float


# ═══════════════════════════════════════════════════════════════════════════════
# DutchBookPricer — maker-arb pricing engine
# ═══════════════════════════════════════════════════════════════════════════════

class DutchBookPricer:
    """
    2026 maker-rebate-aware Dutch Book pricing engine.

    Signal condition:
        net_edge = 1.0 − (yes_bid + no_bid) × (1 − maker_rebate) > desired_net_margin

    Equivalently:
        (yes_bid + no_bid) < (1 − desired_net_margin) / (1 − maker_rebate)

    where:
        yes_bid = best_ask_yes − TICK_SIZE   ← synthetic post-only clamp
        no_bid  = best_ask_no  − TICK_SIZE

    Usage::

        pricer       = DutchBookPricer()
        rebate_engine = MakerRebateEngine()

        # In the strategy loop (maker path):
        rebate = await rebate_engine.get_maker_rebate(condition_id)
        signal = pricer.evaluate_maker(
            condition_id, yes_token_id, no_token_id,
            yes_ask, no_ask,
            max_position_usdc=50.0,
            maker_rebate=rebate,
        )
        if signal:
            yes_resp, no_resp = await client.execute_arb_maker_pair(
                yes_token_id, signal.yes_bid, signal.yes_size,
                no_token_id,  signal.no_bid,  signal.no_size,
            )
            inventory_manager.register_maker_pair(signal, yes_resp, no_resp)
    """

    def __init__(
        self,
        desired_net_margin:  float = DESIRED_NET_MARGIN,
        default_rebate_rate: float = DEFAULT_MAKER_REBATE,
        extreme_lo:          float = EXTREME_PRICE_LO,
        extreme_hi:          float = EXTREME_PRICE_HI,
        min_real_edge:       float = MIN_REAL_EDGE,
    ) -> None:
        if not (0.0 < desired_net_margin < 1.0):
            raise ValueError(
                f"desired_net_margin must be in (0, 1); got {desired_net_margin}"
            )
        self._net_margin     = desired_net_margin
        self._default_rebate = max(0.0, min(default_rebate_rate, MAX_MAKER_REBATE))
        self._extreme_lo     = extreme_lo
        self._extreme_hi     = extreme_hi
        self._min_real_edge  = min_real_edge

    def evaluate_maker(
        self,
        condition_id:      str,
        yes_token_id:      str,
        no_token_id:       str,
        yes_ask:           float,   # best ask from WS feed
        no_ask:            float,   # best ask from WS feed
        max_position_usdc: float = 50.0,
        maker_rebate:      float | None = None,
        tick_size:         float | None = None,   # market tick (0.01 / 0.001)
    ) -> Optional[ArbSignal]:
        """
        Evaluate a Dutch Book opportunity using synthetic post-only maker bids.

        Steps:
          1. Clamp bids below respective best asks (post-only safety)
          2. Resolve maker rebate rate
          3. Calculate effective_cost = (yes_bid + no_bid) × (1 − rebate)
          4. Check net_edge = 1.0 − effective_cost > desired_net_margin
          5. Size the position: n_shares × (yes_bid + no_bid) ≤ max_position_usdc
          6. Return ArbSignal or None

        Parameters
        ----------
        yes_ask / no_ask   : Current best asks from the WS feed.
        max_position_usdc  : Hard capital ceiling ($50 from CircuitBreaker).
        maker_rebate       : Pre-fetched rebate from MakerRebateEngine.
                             Pass None to use the conservative default.
        """
        # ── 1. Validate ask prices ────────────────────────────────────────────
        if not (0.01 <= yes_ask <= 0.99 and 0.01 <= no_ask <= 0.99):
            logger.debug(
                "DutchBookPricer | invalid asks yes=%.4f no=%.4f — skip",
                yes_ask, no_ask,
            )
            return None

        # ── 1a. Signal-quality guards ─────────────────────────────────────────
        # Reject near-resolved / extreme markets: their edge is rebate-driven on
        # thin books that rarely fill, not a durable price gap.
        if not _within_quality_band(yes_ask, no_ask, self._extreme_lo, self._extreme_hi):
            logger.debug(
                "DutchBookPricer | extreme/near-resolved yes=%.4f no=%.4f "
                "(band [%.3f, %.3f]) — skip",
                yes_ask, no_ask, self._extreme_lo, self._extreme_hi,
            )
            return None

        # Require a genuine pre-rebate gap when MIN_REAL_EDGE is enabled (> 0):
        # real_edge = 1 − (yes_ask + no_ask). Keeps the rebate as upside, not the
        # sole justification for entering.
        real_edge = 1.0 - (yes_ask + no_ask)
        if self._min_real_edge > 0.0 and real_edge < self._min_real_edge:
            logger.debug(
                "DutchBookPricer | real_edge=%.6f < min_real_edge=%.6f "
                "(rebate-only) — skip", real_edge, self._min_real_edge,
            )
            return None

        # ── 2. Apply synthetic post-only clamp (stay below the ask) ──────────
        # Snap to the MARKET's tick grid — a bid off-grid is rejected by the
        # exchange ("price must conform to tick size …"). Unknown → 0.01 grid.
        yes_bid = snap_post_only_bid(yes_ask, tick_size)
        no_bid  = snap_post_only_bid(no_ask,  tick_size)

        # ── 3. Resolve maker rebate ───────────────────────────────────────────
        rebate = (
            max(0.0, min(maker_rebate, MAX_MAKER_REBATE))
            if maker_rebate is not None
            else self._default_rebate
        )

        # ── 4. Dutch Book spread with rebate as net-cost deduction ───────────
        combined_bid    = yes_bid + no_bid
        effective_cost  = combined_bid * (1.0 - rebate)   # rebate lowers our cost
        maker_net_edge  = round(1.0 - effective_cost, 6)

        # Dynamic threshold: edge must exceed required net margin
        if maker_net_edge <= self._net_margin:
            logger.debug(
                "DutchBookPricer | no edge — yes_bid=%.4f no_bid=%.4f "
                "combined=%.6f rebate=%.4f(%.2f%%) effective=%.6f edge=%.6f < margin=%.4f",
                yes_bid, no_bid, combined_bid,
                rebate, rebate * 100, effective_cost,
                maker_net_edge, self._net_margin,
            )
            return None

        # ── 5. Size within the hard capital ceiling ───────────────────────────
        if combined_bid <= 0.0:
            return None

        raw_shares = max_position_usdc / combined_bid
        n_shares   = math.floor(raw_shares * 100) / 100.0   # floor to 2 d.p.

        if n_shares < 0.01:
            logger.warning(
                "DutchBookPricer | n_shares %.4f below minimum "
                "(yes_bid=%.4f no_bid=%.4f combined=%.4f cap=%.2f)",
                n_shares, yes_bid, no_bid, combined_bid, max_position_usdc,
            )
            return None

        # ── 6. Populate full ArbSignal (maker fields + taker fields for compat)
        spread_bps = round(maker_net_edge * 10_000, 1)
        signal = ArbSignal(
            condition_id=condition_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_ask=yes_ask,
            no_ask=no_ask,
            combined_cost=round(combined_bid, 6),   # bid-based combined cost
            fee_rate=0.0,                            # no taker fee on maker fills
            fee_cost=0.0,
            net_edge=round(1.0 - combined_bid, 6),  # gross Dutch Book spread
            yes_size=n_shares,
            no_size=n_shares,
            # Maker-path fields
            yes_bid=yes_bid,
            no_bid=no_bid,
            maker_rebate=round(rebate, 6),
            maker_net_edge=maker_net_edge,
        )

        logger.info(
            "DUTCH BOOK SIGNAL | %s "
            "yes_ask=%.4f→bid=%.4f no_ask=%.4f→bid=%.4f "
            "rebate=%.2f%% effective_cost=%.6f maker_net_edge=%.6f(%+.1f bps) "
            "shares=%.2f",
            market_titles.label(condition_id),
            yes_ask, yes_bid, no_ask, no_bid,
            rebate * 100, effective_cost, maker_net_edge, spread_bps,
            n_shares,
        )
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# FeeEngine — taker fee cache (unchanged from Phase 7; kept for taker fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class FeeEngine:
    """
    Fetches and caches per-market Polymarket taker fee rates.

    Resolution order for a cache miss:
      1. Gamma API  /markets?conditionId={id}  →  feeRate  (fraction or bps)
      2. CLOB API   /markets/{id}              →  takerBaseFee
      3. DEFAULT_TAKER_FEE (conservative 2.0 %)

    Usage::

        fee_engine = FeeEngine()
        fee = await fee_engine.get_taker_fee("0xabc…")   # e.g. 0.02
    """

    def __init__(self, default_fee: float = DEFAULT_TAKER_FEE) -> None:
        self._default = default_fee
        self._calibrated = False
        self._fallbacks = 0
        self._cache: dict[str, tuple[float, float]] = {}
        # Persistent keep-alive session — reused across cold-miss fee fetches.
        self._session: "aiohttp.ClientSession | None" = None

    def _get_session(self) -> "aiohttp.ClientSession":
        if self._session is None or getattr(self._session, "closed", False):
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the keep-alive session (call on shutdown)."""
        if self._session is not None and not getattr(self._session, "closed", False):
            await self._session.close()

    def calibrate(self, observed: float) -> None:
        """
        Replace the assumed fee with one measured from our own settled trades.

        Neither Gamma nor the CLOB market record exposes a usable fee for these
        markets, so every lookup fell through to the hard-coded default and no
        log above DEBUG said so. Measuring what we are actually charged is the
        only honest source. Clamped to the sane band, and the cache is dropped
        so nothing keeps serving the old assumption.
        """
        rate = max(0.0, min(float(observed), MAX_TAKER_FEE))
        if rate != self._default:
            logger.warning(
                "FeeEngine | calibrated taker fee %.4f -> %.4f (%.2f%%) from "
                "settled trades; the assumed default was never charged",
                self._default, rate, rate * 100,
            )
        self._default = rate
        self._calibrated = True
        self._cache.clear()

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def fallback_count(self) -> int:
        """How many markets fell through to the default. Nonzero is expected;
        growing without a calibration behind it means we are guessing."""
        return self._fallbacks

    async def get_taker_fee(self, condition_id: str) -> float:
        """Return the taker fee rate as a fraction (0.02 = 2.0 %)."""
        cached = self._cache.get(condition_id)
        if cached is not None:
            fee, ts = cached
            if time.monotonic() - ts < _CACHE_TTL:
                return fee

        fee = await self._fetch_fee(condition_id)
        self._cache[condition_id] = (fee, time.monotonic())
        return fee

    def peek_taker_fee(self, condition_id: str) -> float | None:
        """
        Synchronous cache peek — returns the fresh cached fee or None (no fetch).
        Lets the hot path skip an `await` when the cache is warm.
        """
        cached = self._cache.get(condition_id)
        if cached is not None:
            fee, ts = cached
            if time.monotonic() - ts < _CACHE_TTL:
                return fee
        return None

    def prime_cache(self, condition_id: str, fee_rate: float) -> None:
        self._cache[condition_id] = (fee_rate, time.monotonic())

    async def _fetch_fee(self, condition_id: str) -> float:
        fee = await self._try_gamma(condition_id)
        if fee is not None:
            logger.debug(
                "FeeEngine | Gamma fee for %s → %.4f (%.2f%%)",
                condition_id[:16], fee, fee * 100,
            )
            return fee
        fee = await self._try_clob(condition_id)
        if fee is not None:
            logger.debug(
                "FeeEngine | CLOB fee for %s → %.4f (%.2f%%)",
                condition_id[:16], fee, fee * 100,
            )
            return fee
        self._fallbacks += 1
        if self._fallbacks == 1 and not self._calibrated:
            logger.warning(
                "FeeEngine | no market exposes a fee field; falling back to the "
                "assumed %.4f (%.2f%%) and it is UNVERIFIED. Calibrate against "
                "settled trades or this number silently gates every signal.",
                self._default, self._default * 100,
            )
        else:
            logger.debug(
                "FeeEngine | using default %.4f for %s",
                self._default, condition_id[:16],
            )
        return self._default

    async def _try_schedule(self, condition_id: str) -> "tuple[float, float] | None":
        """Read Gamma's feeSchedule -> (rate, exponent), or None."""
        try:
            session = self._get_session()
            async with session.get(
                f"{_GAMMA_HOST}/markets",
                params={"conditionId": condition_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
            markets = data if isinstance(data, list) else data.get("data", [])
            if not markets:
                return None
            market = markets[0]
            if market.get("feesEnabled") is False:
                return (0.0, 1.0)
            sched = market.get("feeSchedule")
            if not isinstance(sched, dict):
                return None
            rate = float(sched.get("rate", _FEE_SCHEDULE_RATE))
            exp  = float(sched.get("exponent", _FEE_SCHEDULE_EXPONENT))
            if not (0.0 <= rate <= 1.0) or not (0.0 < exp <= 4.0):
                return None
            return (rate, exp)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "FeeEngine | schedule lookup failed for %s: %s",
                condition_id[:16], exc,
            )
            return None

    async def _try_gamma(self, condition_id: str) -> float | None:
        try:
            session = self._get_session()
            async with session.get(
                f"{_GAMMA_HOST}/markets",
                params={"conditionId": condition_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
            markets = data if isinstance(data, list) else data.get("data", [])
            if not markets:
                return None
            market = markets[0]
            for fld in ("feeRate", "fee_rate", "fee"):
                raw = market.get(fld)
                if raw is not None:
                    return _normalise_fee(raw)
        except Exception as exc:
            logger.debug("FeeEngine | Gamma API error for %s: %s", condition_id[:16], exc)
        return None

    async def _try_clob(self, condition_id: str) -> float | None:
        try:
            session = self._get_session()
            async with session.get(
                f"{_CLOB_HOST}/markets/{condition_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
            for fld in ("takerBaseFee", "taker_base_fee"):
                raw = data.get(fld)
                if raw is not None:
                    return _normalise_fee(raw)
        except Exception as exc:
            logger.debug("FeeEngine | CLOB API error for %s: %s", condition_id[:16], exc)
        return None


# Polymarket's published fee schedule, from Gamma's `feeSchedule` field:
#     {"exponent": 1, "rate": 0.04, "takerOnly": true, "rebateRate": 0.25}
#
# The charge is NOT a flat percentage of notional. It is
#
#     fee = rate x min(p, 1-p)^exponent x size          (takers only)
#
# so it vanishes at the extremes and peaks in the middle:
#
#     p = 0.50  ->  0.04 x 0.50 = 2.00%     <- the maximum
#     p = 0.83  ->  0.04 x 0.17 = 0.68%
#     p = 0.96  ->  0.04 x 0.04 = 0.16%
#
# That is why the old DEFAULT_TAKER_FEE = 0.02 constant was not arbitrary: it is
# this formula's WORST case. Charging it everywhere over-stated the cost on
# every extreme-priced leg this bot actually trades, but it was not invented.
#
# Two things hid this. Gamma does not expose `feeRate`/`fee_rate`/`fee`, the
# names the lookup asked for, so it always fell through. And ClobTrade's
# fee_rate_bps reads 0 on every settled trade even when a fee was charged —
# measuring that field alone says "no fee" and is wrong. The charge shows up in
# the cash balance: the 65.91-share flatten received 63.04996 in the trade while
# cash rose only 62.9405, a deduction of 0.10946 against a predicted 0.11442.
_FEE_SCHEDULE_RATE:     float = 0.04
_FEE_SCHEDULE_EXPONENT: float = 1.0


def effective_taker_fee(
    price:    float,
    rate:     float = _FEE_SCHEDULE_RATE,
    exponent: float = _FEE_SCHEDULE_EXPONENT,
) -> float:
    """
    Taker fee as a fraction of notional at `price`, per the published schedule.

    Returns 0 for a price outside (0, 1) rather than guessing.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    return rate * (min(price, 1.0 - price) ** exponent)


def _normalise_fee(raw: object) -> float | None:
    """
    Normalise a raw fee value to a fraction in [0, MAX_TAKER_FEE].
    Handles basis-point integers (100 → 0.0100) and fractions (0.02 → 0.0200).
    """
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val < 0:
        return None
    fraction = val / 10_000 if val > 1.0 else val
    return fraction if fraction <= MAX_TAKER_FEE else None


# ═══════════════════════════════════════════════════════════════════════════════
# ArbDetector — backward-compatible taker arb evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class ArbDetector:
    """
    Fee-aware taker arbitrage signal detector (FOK market orders).

    Trigger condition:
        (P_yes + P_no) × (1 + fee_rate) + desired_net_margin < 1.0

    API is unchanged from Phase 7 — the `yes_bid` / `no_bid` fields on the
    returned ArbSignal are 0.0 (taker path does not clamp bids).

    Usage::

        detector   = ArbDetector()
        fee_engine = FeeEngine()
        fee_rate   = await fee_engine.get_taker_fee(condition_id)
        signal     = detector.evaluate(
                         condition_id, yes_token_id, no_token_id,
                         yes_ask, no_ask,
                         max_position_usdc=50.0,
                         fee_rate=fee_rate,
                     )
        if signal:
            await client.execute_arb_pair(...)
    """

    def __init__(
        self,
        desired_net_margin: float = DESIRED_NET_MARGIN,
        default_fee_rate:   float = DEFAULT_TAKER_FEE,
        extreme_lo:         float = EXTREME_PRICE_LO,
        extreme_hi:         float = EXTREME_PRICE_HI,
    ) -> None:
        if not (0.0 < desired_net_margin < 1.0):
            raise ValueError(
                f"desired_net_margin must be in (0, 1); got {desired_net_margin}"
            )
        if not (0.0 <= default_fee_rate <= MAX_TAKER_FEE):
            raise ValueError(
                f"default_fee_rate must be in [0, {MAX_TAKER_FEE}]; got {default_fee_rate}"
            )
        self._net_margin  = desired_net_margin
        self._default_fee = default_fee_rate
        self._extreme_lo  = extreme_lo
        self._extreme_hi  = extreme_hi

    def evaluate(
        self,
        condition_id:      str,
        yes_token_id:      str,
        no_token_id:       str,
        yes_ask:           float,
        no_ask:            float,
        max_position_usdc: float = 50.0,
        fee_rate:          float | None = None,
    ) -> Optional[ArbSignal]:
        """
        Evaluate whether a YES/NO best-ask pair offers a taker fee-adjusted edge.

        Returns ArbSignal when (yes_ask + no_ask) × (1 + fee_rate) < 1 − margin.
        Returns None otherwise.
        """
        if not (0.01 <= yes_ask <= 0.99 and 0.01 <= no_ask <= 0.99):
            logger.debug(
                "ArbDetector | invalid prices yes=%.4f no=%.4f — skip",
                yes_ask, no_ask,
            )
            return None

        # Signal-quality guard: skip near-resolved / extreme markets.
        if not _within_quality_band(yes_ask, no_ask, self._extreme_lo, self._extreme_hi):
            logger.debug(
                "ArbDetector | extreme/near-resolved yes=%.4f no=%.4f — skip",
                yes_ask, no_ask,
            )
            return None

        effective_fee  = max(0.0, min(fee_rate if fee_rate is not None else self._default_fee, MAX_TAKER_FEE))
        combined_cost  = yes_ask + no_ask
        fee_cost       = combined_cost * effective_fee
        adjusted_cost  = combined_cost * (1.0 + effective_fee)

        if adjusted_cost >= 1.0 - self._net_margin:
            logger.debug(
                "ArbDetector | no edge — adjusted=%.6f fee=%.2f%% threshold=%.6f",
                adjusted_cost, effective_fee * 100, 1.0 - self._net_margin,
            )
            return None

        net_edge = round(1.0 - adjusted_cost, 6)

        if combined_cost <= 0.0:
            return None
        raw_shares = max_position_usdc / combined_cost
        n_shares   = math.floor(raw_shares * 100) / 100.0

        if n_shares < 0.01:
            logger.warning(
                "ArbDetector | n_shares=%.4f below minimum "
                "(yes=%.4f no=%.4f combined=%.4f cap=%.2f)",
                n_shares, yes_ask, no_ask, combined_cost, max_position_usdc,
            )
            return None

        spread_bps = round(net_edge * 10_000, 1)
        signal = ArbSignal(
            condition_id=condition_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_ask=yes_ask,
            no_ask=no_ask,
            combined_cost=round(combined_cost, 6),
            fee_rate=round(effective_fee, 6),
            fee_cost=round(fee_cost, 6),
            net_edge=net_edge,
            yes_size=n_shares,
            no_size=n_shares,
        )

        logger.info(
            "ARB SIGNAL | condition=%s yes=%.4f no=%.4f combined=%.6f "
            "fee=%.2f%% net_edge=%.6f(%+.1f bps) shares=%.2f",
            condition_id[:16], yes_ask, no_ask, combined_cost,
            effective_fee * 100, net_edge, spread_bps, n_shares,
        )
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# NegRiskArbDetector — N-outcome NegRisk maker arb (Phase 9)
# ═══════════════════════════════════════════════════════════════════════════════

class NegRiskArbDetector:
    """
    Negative Risk multi-outcome arbitrage detector (maker path).

    Strategy: simultaneously post synthetic post-only BUY bids on the NO token
    of every mutually exclusive outcome.  At expiry N−1 of the N NO tokens pay
    $1, guaranteeing a positive return when the bundle cost is low enough.

    NegRisk formula (per bundle of 1 NO share per SELECTED outcome):
    ─────────────────────────────────────────────────────────
        no_bid_i      = no_ask_i − TICK_SIZE
        combined_bid  = Σ no_bid_i          over the M selected outcomes
        payout        = M − 1
        effective_cost = combined_bid × (1 − maker_rebate)
        net_edge      = payout − effective_cost
        relative_edge = net_edge / payout       ← normalised; directly comparable
                                                   to DESIRED_NET_MARGIN (binary)

    Signal condition:
        relative_edge > DESIRED_NET_MARGIN  AND  relative_edge ≥ min_relative_edge

    Outcome selection (arXiv:2508.03474 heuristics)
    ───────────────────────────────────────────────
    M is a *subset* of the group's N outcomes: outcomes below
    `min_outcome_prob` are dropped and at most `max_legs` are kept, ranked by
    implied probability.  See evaluate_neg_risk for why subsetting preserves the
    profit guarantee.

    Usage::

        detector      = NegRiskArbDetector()
        rebate_engine = MakerRebateEngine()

        rebate = await rebate_engine.get_maker_rebate(condition_id)
        signal = detector.evaluate_neg_risk(
            condition_id       = condition_id,
            outcome_token_ids  = [no_token_A, no_token_B, no_token_C],
            no_asks            = [ask_A, ask_B, ask_C],
            max_position_usdc  = 50.0,
            maker_rebate       = rebate,
        )
        if signal:
            bundle_legs = [
                BundleLeg(token_id=leg.token_id, bid=leg.no_bid, size=leg.size)
                for leg in signal.legs
            ]
            responses = await client.execute_arb_maker_bundle(bundle_legs)
    """

    def __init__(
        self,
        desired_net_margin:  float = DESIRED_NET_MARGIN,
        default_rebate_rate: float = DEFAULT_MAKER_REBATE,
        *,
        min_outcome_prob:    float = NEGRISK_MIN_OUTCOME_PROB,
        max_legs:            int   = NEGRISK_MAX_LEGS,
        min_relative_edge:   float = NEGRISK_MIN_RELATIVE_EDGE,
        min_leg_shares:      float = NEGRISK_MIN_LEG_SHARES,
        extreme_hi:          float = EXTREME_PRICE_HI,
    ) -> None:
        if not (0.0 < desired_net_margin < 1.0):
            raise ValueError(
                f"desired_net_margin must be in (0, 1); got {desired_net_margin}"
            )
        if not (0.0 <= min_outcome_prob < 1.0):
            raise ValueError(
                f"min_outcome_prob must be in [0, 1); got {min_outcome_prob}"
            )
        if max_legs < 2:
            raise ValueError(f"max_legs must be ≥ 2; got {max_legs}")
        if not (0.0 <= min_relative_edge < 1.0):
            raise ValueError(
                f"min_relative_edge must be in [0, 1); got {min_relative_edge}"
            )
        if min_leg_shares < 0.0:
            raise ValueError(f"min_leg_shares must be ≥ 0; got {min_leg_shares}")
        self._net_margin       = desired_net_margin
        self._default_rebate   = max(0.0, min(default_rebate_rate, MAX_MAKER_REBATE))
        self._min_outcome_prob = min_outcome_prob
        self._max_legs         = max_legs
        self._require_completable  = NEGRISK_REQUIRE_COMPLETABLE
        self._min_completable_edge = NEGRISK_MIN_COMPLETABLE_EDGE
        # Counters so the filter's effect is measurable rather than assumed.
        self.completable_checks  = 0
        self.completable_rejects = 0
        self._min_rel_edge     = min_relative_edge
        self._min_leg_shares   = min_leg_shares
        self._extreme_hi       = extreme_hi

    def evaluate_neg_risk(
        self,
        condition_id:       str,
        outcome_token_ids:  list[str],    # NO token ID for each of the N outcomes
        no_asks:            list[float],  # corresponding best NO asks from WS feed
        max_position_usdc:  float = 50.0,
        maker_rebate:       float | None = None,
        tick_size:          float | None = None,
        no_ask_sizes:       list[float] | None = None,
        no_best_bids:       list[float | None] | None = None,
        leg_tick_sizes:     list[float | None] | None = None,
    ) -> Optional[NegRiskSignal]:
        """
        Evaluate whether buying NO on a subset of the group's outcomes pays.

        Subset lemma (why dropping outcomes is safe)
        ────────────────────────────────────────────
        Take any M outcomes out of a mutually exclusive set and hold one NO
        share on each.  At most one outcome in the whole group resolves YES, so:

          • the winner is one of our M  → the other M−1 NO shares pay $1 → M−1
          • the winner is outside our M → all M NO shares pay $1        → M

        The floor is M−1 either way, so `payout = M − 1` stays a hard guarantee
        for *any* subset.  Exhaustiveness is not required — only mutual
        exclusivity, which the NegRisk contract enforces by construction.  That
        is what makes the paper's <2 % filter a free capital saving rather than
        a risk trade: an outcome priced at 1 % costs ~$0.99 of NO and removes
        only $0.01 from `Σ implied_yes`, the quantity the edge is made of.

        Parameters
        ----------
        condition_id      : NegRisk group ID (negRiskMarketID); used for logging.
        outcome_token_ids : NO token IDs — one per outcome, same order as no_asks.
        no_asks           : Current best NO asks from the WS feed.
        max_position_usdc : Hard capital ceiling (default $50).
        maker_rebate      : Pre-fetched rate from MakerRebateEngine.  Pass None
                            to use the conservative default.
        tick_size         : Market tick grid for the synthetic post-only bids.
        no_ask_sizes      : Displayed ask depth (shares) per outcome, same order
                            as `no_asks`.  When given, the bundle is capped at
                            the *minimum* depth across the selected legs — the
                            paper's §6.2 sizing rule — so no leg is sized beyond
                            what the book can actually fill.  None → capital-only
                            sizing (legacy behaviour).

        Returns
        -------
        NegRiskSignal when every gate passes, else None.
        """
        n = len(outcome_token_ids)

        # ── 1. Basic validation ───────────────────────────────────────────────
        if n != len(no_asks):
            logger.error(
                "NegRiskArbDetector | token_ids length %d ≠ no_asks length %d",
                n, len(no_asks),
            )
            return None

        if no_ask_sizes is not None and len(no_ask_sizes) != n:
            logger.error(
                "NegRiskArbDetector | no_ask_sizes length %d ≠ outcomes %d",
                len(no_ask_sizes), n,
            )
            return None

        if n < 2:
            logger.debug(
                "NegRiskArbDetector | need ≥ 2 outcomes; got %d — skip", n
            )
            return None

        # ── 2. Candidate legs — drop unusable quotes ──────────────────────────
        # A malformed/absent quote only costs us that outcome's contribution to
        # the edge (subset lemma), so drop the leg instead of the whole group.
        # id, ask, implied prob, ask depth, best bid, tick
        candidates: list[tuple[str, float, float, float, "float|None", "float|None"]] = []
        for i, (token_id, ask) in enumerate(zip(outcome_token_ids, no_asks)):
            if not (0.01 <= ask <= 0.99):
                logger.debug(
                    "NegRiskArbDetector | outcome %d invalid no_ask=%.4f — drop leg",
                    i, ask,
                )
                continue
            # None = depth unknown (a batched price_change states the price but
            # not the size at it) → uncapped, the guard absorbs a short fill.
            # 0 = the level is genuinely empty → the leg is unfillable, drop it.
            raw_depth = no_ask_sizes[i] if no_ask_sizes is not None else None
            if raw_depth is None:
                depth = math.inf
            else:
                depth = float(raw_depth)
                if depth <= 0.0:
                    logger.debug(
                        "NegRiskArbDetector | outcome %d has no ask depth — drop leg",
                        i,
                    )
                    continue
            leg_bid  = no_best_bids[i] if no_best_bids else None
            leg_tick = leg_tick_sizes[i] if leg_tick_sizes else None
            candidates.append((token_id, ask, 1.0 - ask, depth, leg_bid, leg_tick))

        if len(candidates) < 2:
            return None

        # ── 3. Near-resolved guard (paper §6) ─────────────────────────────────
        # The study only looks at times when no position is worth more than
        # $0.95, i.e. the group is still genuinely contested.  An implied YES
        # above the band means the group is decided and the remaining "edge" is
        # rebate noise on a book nobody fills.
        top_prob = max(c[2] for c in candidates)
        if top_prob > self._extreme_hi:
            logger.debug(
                "NegRiskArbDetector | condition=%s top outcome implied %.4f > %.2f "
                "— group effectively resolved, skip",
                condition_id[:16], top_prob, self._extreme_hi,
            )
            return None

        # ── 4. Outcome selection (paper §6.2 + §5.1) ──────────────────────────
        selected = [c for c in candidates if c[2] >= self._min_outcome_prob]
        dropped_lowprob = len(candidates) - len(selected)
        if len(selected) < 2:
            logger.debug(
                "NegRiskArbDetector | condition=%s only %d outcome(s) above the "
                "%.0f%% probability floor — skip",
                condition_id[:16], len(selected), self._min_outcome_prob * 100,
            )
            return None

        selected.sort(key=lambda c: c[2], reverse=True)
        dropped_tail = max(0, len(selected) - self._max_legs)
        selected = selected[:self._max_legs]

        m = len(selected)
        sel_ids   = [c[0] for c in selected]
        sel_asks  = [c[1] for c in selected]
        sel_depth = [c[3] for c in selected]
        sel_bids  = [c[4] for c in selected]
        sel_ticks = [c[5] if c[5] is not None else tick_size for c in selected]

        # ── 5. Synthetic post-only NO bids ────────────────────────────────────
        # One tick under each leg's OWN ask. Previously every leg was snapped to
        # the group's coarsest grid (max of the member ticks), so a leg whose
        # real tick is 0.001 was quoted a full cent below the touch — invisible
        # to the book while the coarse legs quoted normally.
        #
        # Note what is NOT here: there is no way to out-bid the touch post-only.
        # naive = ask - tick and best_bid = ask - spread, so the quote only sits
        # at or under the best bid when the spread IS one tick, and then the
        # next price up is the ask itself. On a one-tick book the maker path can
        # only ever join a queue; the lever that actually fills is crossing, and
        # that lives in NegRiskBundleGuard._try_complete.
        no_bids = [
            snap_post_only_bid(ask, t) for ask, t in zip(sel_asks, sel_ticks)
        ]

        # ── 6. Resolve maker rebate ───────────────────────────────────────────
        rebate = (
            max(0.0, min(maker_rebate, MAX_MAKER_REBATE))
            if maker_rebate is not None
            else self._default_rebate
        )

        # ── 7. NegRisk math over the SELECTED subset ──────────────────────────
        payout         = float(m - 1)
        combined_bid   = sum(no_bids)
        effective_cost = combined_bid * (1.0 - rebate)
        net_edge       = round(payout - effective_cost, 6)
        relative_edge  = round(net_edge / payout, 6)

        if relative_edge <= self._net_margin:
            logger.debug(
                "NegRiskArbDetector | no edge — legs=%d/%d combined=%.6f "
                "payout=%.1f rebate=%.4f(%.2f%%) effective=%.6f "
                "relative_edge=%.6f < margin=%.4f",
                m, n, combined_bid, payout, rebate, rebate * 100,
                effective_cost, relative_edge, self._net_margin,
            )
            return None

        # Paper §6: only opportunities worth ≥ $0.05 on the dollar are worth the
        # non-atomic multi-leg risk.
        if relative_edge < self._min_rel_edge:
            logger.debug(
                "NegRiskArbDetector | condition=%s relative_edge=%.6f below the "
                "%.4f on-the-dollar floor — skip",
                condition_id[:16], relative_edge, self._min_rel_edge,
            )
            return None

        # ── 7b. Realistic-completion filter ───────────────────────────────────
        # Runs AFTER the edge gates, so it only ever refuses a bundle that would
        # otherwise have been a signal. Placed before them it fired 214 times in
        # two minutes on bundles whose all-maker edge was already negative —
        # pure noise, and it made the counters meaningless.
        #
        # An all-maker price is an assumption, not a plan. A leg whose spread IS
        # one tick cannot be quoted above the touch post-only, so that order
        # joins a queue and usually does not fill; the bundle then finishes as a
        # taker on that leg or not at all. Scoring the all-maker outcome prices
        # something that does not happen — which is how bundles cleared on entry
        # all night and lost money on resolution.
        #
        # So score it at what we will realistically PAY: maker where our quote
        # leads the book, the ask where it cannot.
        if self._require_completable:
            realistic = 0.0
            for ask, tick, bid in zip(sel_asks, sel_ticks, sel_bids):
                quote = snap_post_only_bid(ask, tick)
                leads = bid is None or quote > bid + 1e-12
                realistic += quote if leads else ask
            realistic_edge = float(m - 1) - realistic
            self.completable_checks += 1
            if realistic_edge < self._min_completable_edge:
                self.completable_rejects += 1
                logger.info(
                    "NegRiskArbDetector | condition=%s clears all-maker "
                    "(%.4f) but not at completion prices (%.4f < %.4f) — skip "
                    "| bids_known=%d/%d",
                    condition_id[:16], net_edge, realistic_edge,
                    self._min_completable_edge,
                    sum(1 for b in sel_bids if b is not None), m,
                )
                return None

        # ── 8. Size: capital ceiling ∧ shallowest leg (paper §6.2) ────────────
        if combined_bid <= 0.0:
            return None

        raw_bundles = max_position_usdc / combined_bid
        depth_cap   = min(sel_depth)          # inf when no depth was supplied
        raw_bundles = min(raw_bundles, depth_cap)
        n_bundles   = math.floor(raw_bundles * 100) / 100.0

        if n_bundles < self._min_leg_shares:
            logger.debug(
                "NegRiskArbDetector | condition=%s n_bundles %.2f below the "
                "%.2f-share exchange minimum (combined_bid=%.4f cap=%.2f "
                "depth_cap=%.2f) — skip",
                condition_id[:16], n_bundles, self._min_leg_shares,
                combined_bid, max_position_usdc, depth_cap,
            )
            return None

        # ── 9. Build per-outcome leg tuple ────────────────────────────────────
        legs = tuple(
            ArbLeg(
                token_id=token_id,
                no_ask=ask,
                no_bid=bid,
                size=n_bundles,
            )
            for token_id, ask, bid in zip(sel_ids, sel_asks, no_bids)
        )

        spread_bps = round(relative_edge * 10_000, 1)
        signal = NegRiskSignal(
            condition_id=condition_id,
            n_outcomes=m,
            legs=legs,
            combined_bid=round(combined_bid, 6),
            payout=payout,
            maker_rebate=round(rebate, 6),
            effective_cost=round(effective_cost, 6),
            net_edge=net_edge,
            relative_edge=relative_edge,
            n_bundles=n_bundles,
        )

        logger.info(
            "NEG RISK SIGNAL | condition=%s legs=%d/%d "
            "(dropped %d <%.0f%%, %d beyond top-%d) "
            "combined_bid=%.6f payout=%.1f rebate=%.2f%% "
            "effective=%.6f net_edge=%.6f relative=%.6f(%+.1f bps) bundles=%.2f",
            condition_id[:16], m, n,
            dropped_lowprob, self._min_outcome_prob * 100,
            dropped_tail, self._max_legs,
            combined_bid, payout, rebate * 100,
            effective_cost, net_edge, relative_edge, spread_bps, n_bundles,
        )
        return signal
