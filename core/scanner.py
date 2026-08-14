"""
core/scanner.py
───────────────
Dynamic market discovery — Polymarket Gamma API scanner.

Background asyncio task that periodically polls the Gamma API to find active
binary markets, extracts their YES/NO CLOB token IDs, and registers each
newly discovered market with a FeedRegistry, which spins up a MarketFeed
task for it without requiring a bot restart.

Gamma API
─────────
  GET https://gamma-api.polymarket.com/markets
      ?active=true&closed=false&archived=false
      &limit=100&next_cursor=<cursor>

Each market response item has at minimum:
  {
    "conditionId":   "0x…",          # bytes32 hex condition identifier
    "clobTokenIds":  ["0xYES…","0xNO…"],  # [YES token, NO token]
    "active":        true,
    "closed":        false,
  }

The scanner pages through all results using next_cursor, deduplicates via an
internal set of known condition IDs, and fires on_market_added only once per
market per process lifetime.

FeedRegistry
────────────
Wraps a live dict of condition_id → asyncio.Task (each running a MarketFeed).
New feeds are created by calling add_market(); existing feeds survive
reconnects inside MarketFeed.run().  stop_all() cancels every feed task
gracefully for clean shutdown.

Usage in main.py::

    queue         = asyncio.Queue(maxsize=2048)
    feed_registry = FeedRegistry(queue=queue)

    # Seed from ENV if present
    if YES_TOKEN_ID and NO_TOKEN_ID:
        await feed_registry.add_market(CONDITION_ID, YES_TOKEN_ID, NO_TOKEN_ID)

    scanner = MarketScanner(on_market_added=feed_registry.add_market)
    asyncio.create_task(scanner.run(), name="scanner")
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import time
import warnings
from datetime import datetime, timezone
from typing import Awaitable, Callable

import aiohttp

from core.ws_feed import MarketFeed
from strategy.arbitrage import _resolve_rebate
from telemetry.metrics import (
    FEEDS_PRUNED,
    SCANNER_ADMITTED,
    SCANNER_CANDIDATES,
    SCANNER_FEEDS_FETCHED,
    SCANNER_SCAN_DURATION,
    SCANNER_SCORE_AVG,
    SCANNER_SCORE_MAX,
    SCANNER_SCORE_MIN,
)

logger = logging.getLogger(__name__)

# ── Gamma API endpoint ──────────────────────────────────────────────────────
_GAMMA_URL   = "https://gamma-api.polymarket.com/markets"
_PAGE_LIMIT  = 100
# Polymarket uses "LTE=" as the terminal cursor value for paginated endpoints.
_END_CURSOR  = "LTE="
# Safety cap on offset pagination (the active universe is ~2k markets; this
# bounds a runaway loop if the API ever stops returning short final pages).
_MAX_SCAN_MARKETS: int = 8000

# ── Scanner defaults ────────────────────────────────────────────────────────
SCAN_INTERVAL: float = 300.0   # seconds between full market-list polls (5 min)

# ── V2 market-scoring parameters (docs/Scorer machine spec) ──────────────────
# Target markets liquid enough for $100–500 positions but NOT yet fully
# efficient — NOT simply the biggest markets. Score = Liquidity × Inefficiency ×
# Penalty / sqrt(days_to_close), computed from a single Gamma /markets response
# (no per-market CLOB /book calls — the scanner layer must stay fast).
V2_VOL_EXP:        float = 0.30     # volume24h exponent (dampened)
V2_LIQ_EXP:        float = 0.20     # liquidity exponent (dampened)
V2_MIN_LIQUIDITY:  float = 500.0    # exclude markets below this liquidity ($)
V2_MIN_VOLUME_24H: float = 100.0    # exclude markets below this 24h volume ($)
V2_SPREAD_WEIGHT:  float = 100.0    # weight on relative spread in I_factor
V2_INEFF_WEIGHT:   float = 1000.0   # weight on YES+NO arb edge in I_factor
V2_PENALTY_PIVOT:  float = 100_000.0  # 24h volume at which P_eff halves-ish

# Callable signature for the market-added callback.
# Async: async def cb(condition_id, yes_token_id, no_token_id) -> bool | None
# Returns False when the market was NOT registered (e.g. global feed cap hit);
# any other value (None/True) is treated as success by the scanner.
MarketCallback = Callable[[str, str, str], Awaitable[object]]


# ═══════════════════════════════════════════════════════════════════════════
# FeedRegistry
# ═══════════════════════════════════════════════════════════════════════════

class FeedRegistry:
    """
    Manages one MarketFeed asyncio.Task per active Polymarket binary market.

    Each feed runs independently; they all push arb_tick dicts to the shared
    queue supplied at construction.  add_market() is idempotent — calling it
    for an already-tracked condition_id is a no-op.

    Attributes
    ----------
    condition_ids : frozenset[str]
        Snapshot of all condition IDs currently being fed.
    active_count : int
        Number of live feed tasks.
    """

    def __init__(self, queue: asyncio.Queue, max_feeds: int = 0) -> None:
        self._queue = queue
        # condition_id → (MarketFeed, asyncio.Task)
        self._feeds: dict[str, tuple[MarketFeed, asyncio.Task]] = {}
        # Global hard cap on concurrently-active feeds (0 = unlimited). This is
        # the real ceiling on WebSocket connections; the scanner's per-scan
        # admission limit is separate and must not be allowed to grow feeds
        # without bound across re-scans.
        self._max_feeds = max(0, max_feeds)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def add_market(
        self,
        condition_id:  str,
        yes_token_id:  str,
        no_token_id:   str,
    ) -> bool:
        """
        Register a market and start its WebSocket feed task.

        Idempotent — returns True if already tracked.  Returns False (without
        starting a feed) when the global max_feeds cap is reached, so the caller
        can avoid marking the market as permanently seen and retry it later.
        Returns False and logs (does not crash) if the feed task cannot start.
        """
        if condition_id in self._feeds:
            return True

        if self._max_feeds and len(self._feeds) >= self._max_feeds:
            logger.debug(
                "FeedRegistry | at cap (%d) — rejecting condition=%s",
                self._max_feeds, condition_id[:16],
            )
            return False

        try:
            feed = MarketFeed(
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                condition_id=condition_id,
                queue=self._queue,
            )
            task = asyncio.create_task(
                feed.run(),
                name=f"feed-{condition_id[:16]}",
            )
            self._feeds[condition_id] = (feed, task)
            logger.info(
                "FeedRegistry | feed started condition=%s yes=%s no=%s "
                "(total_feeds=%d)",
                condition_id[:16], yes_token_id[:12], no_token_id[:12],
                len(self._feeds),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "FeedRegistry | failed to start feed for condition=%s: %s",
                condition_id[:16], exc,
            )
            return False

    async def prune_stale(self, max_idle_s: float) -> list[str]:
        """
        Remove feeds that have not produced a two-sided arb_tick within
        `max_idle_s` seconds (dead / illiquid / one-sided markets).

        Returns the list of pruned condition_ids so the caller can also forget
        them and free the slot for a fresher candidate.
        """
        if max_idle_s <= 0 or not self._feeds:
            return []
        now   = time.monotonic()
        stale = [
            cid for cid, (feed, _task) in self._feeds.items()
            if feed.idle_seconds(now) > max_idle_s
        ]
        for cid in stale:
            await self.remove_market(cid)
        if stale:
            FEEDS_PRUNED.inc(len(stale))
            logger.info(
                "FeedRegistry | pruned %d stale feed(s) (idle > %.0fs) | remaining=%d",
                len(stale), max_idle_s, len(self._feeds),
            )
        return stale

    async def remove_market(self, condition_id: str) -> None:
        """
        Cancel and remove the feed for a specific condition ID.

        Safe to call even if the market is not tracked (no-op in that case).
        """
        entry = self._feeds.pop(condition_id, None)
        if entry is None:
            return
        feed, task = entry
        feed.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        logger.info(
            "FeedRegistry | feed removed condition=%s (remaining=%d)",
            condition_id[:16], len(self._feeds),
        )

    async def stop_all(self) -> None:
        """Cancel all feed tasks for clean shutdown."""
        if not self._feeds:
            return
        logger.info("FeedRegistry | stopping %d feed(s) …", len(self._feeds))
        for _cond_id, (feed, task) in list(self._feeds.items()):
            feed.stop()
            task.cancel()
        await asyncio.gather(
            *(task for _, task in self._feeds.values()),
            return_exceptions=True,
        )
        self._feeds.clear()
        logger.info("FeedRegistry | all feeds stopped")

    # ──────────────────────────────────────────────────────────────────────────
    # Observability
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def condition_ids(self) -> frozenset[str]:
        """Snapshot of all currently tracked condition IDs."""
        return frozenset(self._feeds.keys())

    @property
    def active_count(self) -> int:
        """Number of live feed tasks."""
        return len(self._feeds)


# ═══════════════════════════════════════════════════════════════════════════
# MarketScorer
# ═══════════════════════════════════════════════════════════════════════════

class MarketScorer:
    """
    V2 market scorer — ranks Gamma API market dicts by *capturable arbitrage
    potential* rather than raw size (docs/Scorer machine spec).

    The biggest markets (e.g. the US presidential election) carry huge capital
    but are extremely efficient — tight spreads, pro market makers, low-latency
    algos — leaving almost no real arbitrage. The goal is markets liquid enough
    for $100–500 positions yet not yet fully efficient.

    Three independent components, multiplied::

        SCORE = (L_factor × I_factor × P_eff) / sqrt(days_to_close)

      L_factor (liquidity)    = volume24h^0.3 × liquidity^0.2   (dampened so giant
                                markets don't dominate); excluded entirely below
                                the liquidity / volume floors.
      I_factor (inefficiency) = rel_spread×100 + ineff_edge×1000
                                rel_spread = (ask−bid)/mid
                                ineff_edge = |yes_price + no_price − 1.0|
      P_eff    (penalty)      = 1 / (1 + (volume24h / 100000)^2)  — suppresses
                                too-big / too-efficient markets.

    Crucially, **every input comes from a single Gamma /markets response**
    (volume24hr, liquidityNum, bestBid/bestAsk/spread, outcomePrices). No
    per-market CLOB /book calls are made here — order-book depth and live arb
    analysis happen only for the admitted top-N feeds, over WebSocket.
    """

    def score(self, market: dict) -> float:
        """Return the V2 priority score for a single Gamma API market object."""
        volume_24h = self._volume_24h(market)
        liquidity  = self._liquidity(market)

        # Hard exclusions — not tradeable for $100–500 positions.
        if liquidity < V2_MIN_LIQUIDITY or volume_24h < V2_MIN_VOLUME_24H:
            return 0.0

        l_factor = (volume_24h ** V2_VOL_EXP) * (liquidity ** V2_LIQ_EXP)
        i_factor = (
            self._relative_spread(market) * V2_SPREAD_WEIGHT
            + self._inefficiency_edge(market) * V2_INEFF_WEIGHT
        )
        p_eff    = 1.0 / (1.0 + (volume_24h / V2_PENALTY_PIVOT) ** 2)

        # _days_to_close is floored at 1.0; sqrt keeps near-expiry markets
        # favoured without letting them dominate the ranking.
        days = self._days_to_close(market)
        score = (l_factor * i_factor * p_eff) / math.sqrt(days)
        return score if math.isfinite(score) else 0.0

    # ── Field extraction (robust to Gamma string/number variants) ─────────────

    @staticmethod
    def _num(value: object) -> float:
        """Coerce a Gamma field (float or numeric string) to a finite float."""
        try:
            f = float(value)  # type: ignore[arg-type]
            return f if math.isfinite(f) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _volume_24h(cls, m: dict) -> float:
        return cls._num(
            m.get("volume24hr")
            if m.get("volume24hr") is not None
            else (m.get("volume_24h") or m.get("volume") or 0.0)
        )

    @classmethod
    def _liquidity(cls, m: dict) -> float:
        return cls._num(
            m.get("liquidityNum")
            if m.get("liquidityNum") is not None
            else (m.get("liquidity") or m.get("liquidityClob") or 0.0)
        )

    @classmethod
    def _relative_spread(cls, m: dict) -> float:
        """(best_ask − best_bid) / mid_price — 0.0 when unavailable."""
        bid = cls._num(m.get("bestBid"))
        ask = cls._num(m.get("bestAsk"))
        if bid > 0.0 and ask > 0.0 and ask >= bid:
            mid = (ask + bid) / 2.0
            return (ask - bid) / mid if mid > 0.0 else 0.0
        # Fallback: raw `spread` field over the last/mid price.
        spread = cls._num(m.get("spread"))
        ref    = cls._num(m.get("lastTradePrice")) or 0.5
        return spread / ref if (spread > 0.0 and ref > 0.0) else 0.0

    @classmethod
    def _inefficiency_edge(cls, m: dict) -> float:
        """
        |yes_price + no_price − 1.0| from outcomePrices — how far the combined
        binary price strays from the theoretical $1.00 (larger = more arb-prone).
        """
        raw = m.get("outcomePrices")
        prices: list[float] = []
        if isinstance(raw, str):
            try:
                prices = [float(x) for x in json.loads(raw)]
            except (json.JSONDecodeError, TypeError, ValueError):
                prices = []
        elif isinstance(raw, (list, tuple)):
            prices = [cls._num(x) for x in raw]
        if len(prices) >= 2:
            return abs((prices[0] + prices[1]) - 1.0)
        return 0.0

    @staticmethod
    def _days_to_close(market: dict) -> float:
        """
        Calendar days until market expiry, **floored at 1.0**.

        The floor makes this a safe scoring denominator (avoids dividing by a
        sub-1 value, which would explode the score in the final hours) and gives
        a neutral 1.0 for missing / unparseable / already-expired dates.

        Expiry *gating* (dropping dead markets) is a separate concern handled by
        `_is_expired()`; this function therefore never returns < 1.0.
        """
        raw = market.get("endDate") or market.get("end_date")
        if not raw:
            return 1.0
        try:
            end_dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            delta  = (end_dt - datetime.now(timezone.utc)).total_seconds()
            return max(delta / 86_400.0, 1.0)
        except (ValueError, TypeError):
            return 1.0

    @staticmethod
    def _is_expired(market: dict) -> bool:
        """
        True only if the market has a parseable endDate in the past.

        Missing / unparseable dates are treated as NOT expired — Gamma is already
        queried with active=true&closed=false, so we never drop a market merely
        because its date could not be read.
        """
        raw = market.get("endDate") or market.get("end_date")
        if not raw:
            return False
        try:
            end_dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return (end_dt - datetime.now(timezone.utc)).total_seconds() <= 0
        except (ValueError, TypeError):
            return False


# ═══════════════════════════════════════════════════════════════════════════
# MarketScanner
# ═══════════════════════════════════════════════════════════════════════════

class MarketScanner:
    """
    Polls the Polymarket Gamma API for active binary markets and calls
    on_market_added for each newly discovered market.

    Deduplication
    ─────────────
    self._known tracks every condition_id that has already been dispatched so
    the callback fires exactly once per market per process lifetime.  Pass
    seed_condition_ids to pre-populate this set with markets that were seeded
    via ENV or a config file so they are not re-added by the scanner.

    Usage::

        registry = FeedRegistry(queue)
        scanner  = MarketScanner(on_market_added=registry.add_market)
        asyncio.create_task(scanner.run(), name="scanner")
    """

    def __init__(
        self,
        on_market_added:    MarketCallback,
        *,
        scan_interval:      float = SCAN_INTERVAL,
        seed_condition_ids: set[str] | None = None,
        max_feeds:          int = 50,
        feed_registry:      "FeedRegistry | None" = None,
        prune_idle_s:       float = 0.0,
        min_volume_24h:     float = 0.0,
        on_admit:           "Callable[[str, dict], None] | None" = None,
    ) -> None:
        self._on_market_added = on_market_added
        self._scan_interval   = scan_interval
        self._known: set[str] = set(seed_condition_ids or [])
        self._running         = False
        self._max_feeds       = max(0, max_feeds)
        self._scorer          = MarketScorer()
        # Optional registry reference enables a GLOBAL active-feed cap and
        # stale-feed pruning (prevents unbounded feed growth across re-scans).
        self._registry        = feed_registry
        self._prune_idle_s    = max(0.0, prune_idle_s)
        # Liquidity floor: skip candidates whose 24h volume is below this (dead
        # books waste a feed slot and rarely offer capturable arbs). 0 = off.
        self._min_volume_24h  = max(0.0, min_volume_24h)
        # Latency: fired (condition_id, market_dict) on each admitted market so
        # callers can pre-warm fee/rebate caches BEFORE the first tick arrives,
        # removing a network round-trip from the hot path. Synchronous + cheap.
        self._on_admit        = on_admit

    # ──────────────────────────────────────────────────────────────────────────
    # Public control
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Long-running scanner loop.  Performs an initial scan immediately on
        startup, then sleeps scan_interval seconds between subsequent polls.
        Cancel the task to stop cleanly.
        """
        self._running = True
        logger.info(
            "MarketScanner started | interval=%.0fs seed_count=%d max_feeds=%d",
            self._scan_interval, len(self._known), self._max_feeds,
        )

        try:
            while self._running:
                try:
                    await self._scan_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("MarketScanner scan error: %s", exc)

                await asyncio.sleep(self._scan_interval)

        except asyncio.CancelledError:
            self._running = False
            logger.info("MarketScanner stopped")
            raise

    def stop(self) -> None:
        """Signal the scanner to stop after the current sleep expires."""
        self._running = False

    # ──────────────────────────────────────────────────────────────────────────
    # Internal scan logic
    # ──────────────────────────────────────────────────────────────────────────

    async def _scan_once(self) -> None:
        """
        Fetch every active binary market from the Gamma API, call
        on_market_added for any not yet in self._known.

        When max_feeds > 0 all new candidates are scored and sorted by
        MarketScorer before admission; only the top max_feeds are registered.
        When max_feeds == 0 the original sequential behaviour is preserved.
        """
        _t0 = time.perf_counter()
        try:
            await self._scan_once_inner()
        finally:
            SCANNER_SCAN_DURATION.observe(time.perf_counter() - _t0)

    async def _scan_once_inner(self) -> None:
        # Prune dead/illiquid feeds before admitting new ones so freed slots can
        # be reused and forgotten markets become eligible again.
        if self._registry is not None and self._prune_idle_s > 0:
            pruned = await self._registry.prune_stale(self._prune_idle_s)
            for cid in pruned:
                self._known.discard(cid)

        markets = await self._fetch_all_active()
        SCANNER_FEEDS_FETCHED.inc(len(markets))

        if self._max_feeds == 0:
            # ── original path (unchanged) ───────────────────────────────────
            new_count = 0
            for m in markets:
                condition_id = _extract_condition_id(m)
                if not condition_id or condition_id in self._known:
                    continue

                yes_id, no_id = _extract_token_ids(m)
                if not yes_id or not no_id:
                    logger.debug(
                        "MarketScanner | %s skipped — could not parse YES/NO token IDs",
                        condition_id[:16],
                    )
                    continue

                self._known.add(condition_id)
                new_count += 1
                if self._on_admit is not None:
                    self._on_admit(condition_id, m)   # pre-warm caches
                logger.info(
                    "MarketScanner | new market condition=%s yes=%s no=%s",
                    condition_id[:16], yes_id[:12], no_id[:12],
                )
                await self._on_market_added(condition_id, yes_id, no_id)

            SCANNER_CANDIDATES.set(new_count)
            SCANNER_ADMITTED.set(new_count)
            logger.debug(
                "MarketScanner scan done | new=%d total_known=%d",
                new_count, len(self._known),
            )
            if new_count:
                logger.info(
                    "MarketScanner | %d new market(s) added | %d total tracked",
                    new_count, len(self._known),
                )
            return

        # ── priority scoring path ───────────────────────────────────────────
        # Collect all unseen, parseable candidates then score once upfront.
        # Tuple layout: (score, market_dict, cond_id, yes_id, no_id)
        ScoredCandidate = tuple[float, dict, str, str, str]
        scored_pairs: list[ScoredCandidate] = []
        for m in markets:
            condition_id = _extract_condition_id(m)
            if not condition_id or condition_id in self._known:
                continue
            # Skip expired markets before scoring.
            if MarketScorer._is_expired(m):
                logger.debug(
                    "MarketScanner | %s skipped — market expired",
                    condition_id[:16],
                )
                continue
            yes_id, no_id = _extract_token_ids(m)
            if not yes_id or not no_id:
                logger.debug(
                    "MarketScanner | %s skipped — could not parse YES/NO token IDs",
                    condition_id[:16],
                )
                continue
            # Liquidity floor — skip dead books that would waste a feed slot.
            if self._min_volume_24h > 0:
                vol = float(
                    m.get("volume24hr") or m.get("volume_24h") or m.get("volume") or 0.0
                )
                if not math.isfinite(vol) or vol < self._min_volume_24h:
                    logger.debug(
                        "MarketScanner | %s skipped — volume %.0f < floor %.0f",
                        condition_id[:16], vol, self._min_volume_24h,
                    )
                    continue
            score = self._scorer.score(m)
            # V2: a 0 score means excluded (below liquidity/volume floors or
            # perfectly efficient) — never admit it, even if feed slots are free.
            if score <= 0.0:
                continue
            scored_pairs.append((score, m, condition_id, yes_id, no_id))

        scored_pairs.sort(key=lambda t: t[0], reverse=True)
        total_candidates = len(scored_pairs)

        # Score distribution metrics — emit 0.0 sentinels when list is empty.
        if scored_pairs:
            all_scores = [s for s, *_ in scored_pairs]
            SCANNER_SCORE_MIN.set(min(all_scores))
            SCANNER_SCORE_MAX.set(max(all_scores))
            SCANNER_SCORE_AVG.set(sum(all_scores) / len(all_scores))
        else:
            SCANNER_SCORE_MIN.set(0.0)
            SCANNER_SCORE_MAX.set(0.0)
            SCANNER_SCORE_AVG.set(0.0)

        SCANNER_CANDIDATES.set(total_candidates)

        # Log top-5 for observability.
        for score, m, cond_id, _yes, _no in scored_pairs[:5]:
            vol    = float(
                m.get("volume24hr") or m.get("volume_24h") or m.get("volume") or 0.0
            )
            if not math.isfinite(vol):
                vol = 0.0
            days          = MarketScorer._days_to_close(m)
            rebate        = _resolve_rebate(str(m.get("category") or "")) or 0.0
            daily_rebate  = vol * rebate
            logger.info(
                "MarketScorer | top-5 condition=%s category=%s score=%.4f "
                "volume_24h=%.2f days_to_close=%.1f rebate=%.2f%% "
                "expected_daily_rebate=%.4f",
                cond_id[:16], m.get("category", ""), score,
                vol, days, rebate * 100, daily_rebate,
            )

        # Respect the GLOBAL active-feed cap when a registry is wired: only fill
        # the slots actually free right now, instead of admitting max_feeds NEW
        # markets every scan (which previously grew feeds without bound).
        if self._registry is not None:
            free_slots = max(0, self._max_feeds - self._registry.active_count)
        else:
            free_slots = self._max_feeds
        admitted = scored_pairs[:free_slots]
        SCANNER_ADMITTED.set(len(admitted))
        logger.info(
            "MarketScorer | total_candidates=%d admitted=%d "
            "(max_feeds=%d, free_slots=%d)",
            total_candidates, len(admitted), self._max_feeds, free_slots,
        )

        new_count = 0
        for score, m, condition_id, yes_id, no_id in admitted:
            # Mark known only on a successful add so cap-rejected markets remain
            # eligible on a later scan (after pruning frees a slot).
            result = await self._on_market_added(condition_id, yes_id, no_id)
            if result is False:
                continue
            self._known.add(condition_id)
            new_count += 1
            if self._on_admit is not None:
                self._on_admit(condition_id, m)   # pre-warm caches off the hot path
            logger.info(
                "MarketScanner | new market condition=%s yes=%s no=%s score=%.4f",
                condition_id[:16], yes_id[:12], no_id[:12], score,
            )

        logger.debug(
            "MarketScanner scan done | new=%d total_known=%d",
            new_count, len(self._known),
        )
        if new_count:
            logger.info(
                "MarketScanner | %d new market(s) added | %d total tracked",
                new_count, len(self._known),
            )

    async def _fetch_all_active(self) -> list[dict]:
        """
        Page through the Gamma API and collect all active, non-closed,
        non-archived binary markets (clobTokenIds has exactly 2 entries).

        Gamma `/markets` returns a PLAIN LIST but is paginated by `offset` —
        it is NOT a single complete response.  Earlier this method stopped
        after the first page ("plain list is never paginated"), so the scanner
        only ever saw the first ~100 markets (≈30 candidates after the volume
        /liquidity floors) instead of the full liquid universe (~600). We now
        page by offset until a short/empty page, an end-of-range 422, or the
        _MAX_SCAN_MARKETS safety cap.  The legacy dict/next_cursor shape is
        still handled for forward compatibility.
        """
        results: list[dict] = []
        cursor              = ""
        offset              = 0

        async with aiohttp.ClientSession() as session:
            while True:
                params: dict[str, object] = {
                    "active":   "true",
                    "closed":   "false",
                    "archived": "false",
                    "limit":    _PAGE_LIMIT,
                }
                if cursor:
                    params["next_cursor"] = cursor
                else:
                    params["offset"] = offset

                data = None
                for _attempt in range(3):
                    try:
                        async with session.get(
                            _GAMMA_URL,
                            params=params,
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            # 422 past the last page = clean end of pagination.
                            if resp.status == 422 and (offset > 0 or results):
                                return results
                            resp.raise_for_status()
                            data = await resp.json(content_type=None)
                        break  # success
                    except Exception as exc:  # noqa: BLE001
                        if _attempt == 2:
                            warnings.warn(
                                f"Gamma API failed after 3 attempts: {exc} — "
                                "returning partial results",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            logger.warning(
                                "Gamma API failed after 3 attempts: %s", exc
                            )
                            return results
                        await asyncio.sleep(0.5 * (2 ** _attempt))  # 0.5 s, 1.0 s
                if data is None:
                    return results

                # Plain-list response → paginate by offset until a short page.
                if isinstance(data, list):
                    for m in data:
                        if _is_binary(m):
                            results.append(m)
                    if len(data) < _PAGE_LIMIT:
                        break                      # last (short) page
                    offset += len(data)
                    if offset >= _MAX_SCAN_MARKETS:
                        logger.warning(
                            "MarketScanner | hit _MAX_SCAN_MARKETS=%d scan cap",
                            _MAX_SCAN_MARKETS,
                        )
                        break
                    continue

                # Paginated dict response: {"data": [...], "next_cursor": "…"}
                for m in data.get("data", []):
                    if _is_binary(m):
                        results.append(m)

                cursor = data.get("next_cursor", "")
                if not cursor or cursor == _END_CURSOR:
                    break

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — Gamma API response parsing
# ═══════════════════════════════════════════════════════════════════════════

def _is_binary(market: dict) -> bool:
    """Return True if market has exactly 2 non-empty, distinct CLOB token IDs."""
    ids = _raw_clob_ids(market)
    return len(ids) == 2 and bool(ids[0]) and bool(ids[1]) and ids[0] != ids[1]


def _raw_clob_ids(market: dict) -> list[str]:
    """
    Extract CLOB token IDs from a Gamma API market object.

    The API may return clobTokenIds as:
      - A JSON list:   ["0xYES", "0xNO"]
      - A JSON string: '["0xYES", "0xNO"]'   (double-encoded)
      - A nested list under "tokens": [{"token_id": "…"}, …]
    """
    raw = market.get("clobTokenIds") or market.get("clob_token_ids")
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str):
        # Unwrap double- or triple-encoded JSON strings.
        decoded: object = raw
        while isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except (json.JSONDecodeError, ValueError):
                # Try ast as a last resort, then give up.
                try:
                    decoded = ast.literal_eval(decoded)
                except Exception:  # noqa: BLE001
                    decoded = None
                break
        if isinstance(decoded, list):
            return [str(x) for x in decoded if x]

    # Fallback: parse from "tokens" array
    tokens: list[dict] = market.get("tokens", [])
    ids = [t.get("token_id", "") for t in tokens if isinstance(t, dict)]
    return [x for x in ids if x]


def _extract_condition_id(market: dict) -> str:
    """Return the normalised condition ID hex string, or '' if missing."""
    cid = (
        market.get("conditionId")
        or market.get("condition_id")
        or ""
    )
    return str(cid).strip()


def _extract_token_ids(market: dict) -> tuple[str, str]:
    """
    Return (yes_token_id, no_token_id).

    Polymarket convention: clobTokenIds[0] = YES outcome, [1] = NO outcome.
    Falls back to alphabetical sort when the "tokens" outcome labels are missing.
    """
    ids = _raw_clob_ids(market)
    if len(ids) >= 2:
        return ids[0], ids[1]

    # Fallback: find YES/NO tokens by outcome label, then alphabetical order.
    tokens: list[dict] = market.get("tokens", [])
    if len(tokens) >= 2:
        yes_tok = next(
            (t for t in tokens if "yes" in str(t.get("outcome", "")).lower()), None
        )
        no_tok  = next(
            (t for t in tokens if "no"  in str(t.get("outcome", "")).lower()), None
        )
        if yes_tok and no_tok and yes_tok is not no_tok:
            return str(yes_tok.get("token_id", "")), str(no_tok.get("token_id", ""))
        # Last resort: alphabetical sort by outcome label.
        tokens_sorted = sorted(tokens, key=lambda t: str(t.get("outcome", "")).lower())
        yes = str(tokens_sorted[0].get("token_id", ""))
        no  = str(tokens_sorted[1].get("token_id", ""))
        return yes, no

    return "", ""
