"""
backtest/data_handler.py
────────────────────────
HistoricalDataFeed — downloads the Polymarket Kaggle dataset programmatically
and streams it tick-by-tick into an asyncio.Queue to eliminate look-ahead bias.

Authentication
──────────────
Reads KAGGLE_API_TOKEN from .env before any download attempt.
Supported formats (pick one):

  KAGGLE_API_TOKEN=username:api_key     ← combined token (preferred)
  KAGGLE_USERNAME=user KAGGLE_KEY=key  ← separate env vars

kagglehub never opens a browser or prompts for OAuth when KAGGLE_KEY is set;
it authenticates headlessly via the Kaggle REST API.

Order-book reconstruction
─────────────────────────
The Kaggle CSV stores per-market YES-token prices only (bestBid / bestAsk).
In a Polymarket binary market the two outcome tokens are strict complements:

    YES_bid  +  NO_ask  ≈  1.0   (pre-fee mid-market identity)
    YES_ask  +  NO_bid  ≈  1.0

So we reconstruct the NO side as:

    no_ask  =  1.0 − bestBid
    no_bid  =  1.0 − bestAsk

These derived values are stored in OrderBookTick and fed to the ArbDetector /
DutchBookPricer without any modification.

Look-ahead prevention
─────────────────────
`stream()` is a coroutine that calls `await asyncio.sleep(0)` after every
enqueue.  This hands control back to the event loop between each tick, so
the strategy can only ever inspect the *current* tick — never a future one.
The consumer (run_backtest.py) processes each tick via `await queue.get()`
before the next tick is loaded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Kaggle dataset coordinates ────────────────────────────────────────────────
_KAGGLE_DATASET = "ismetsemedov/polymarket-prediction-markets"
_CSV_FILENAME   = "polymarket_markets.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# OrderBookTick — one historical snapshot for a single binary market
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class OrderBookTick:
    """
    A single point-in-time order-book snapshot for one Polymarket binary market.

    yes_ask / no_ask are the prices a taker must pay to buy each leg.
    yes_bid / no_bid are the passive bids (best resting buy orders).

    The DutchBookPricer receives yes_ask / no_ask and derives its own synthetic
    post-only bids internally (ask − TICK_SIZE).
    """
    timestamp:    str
    condition_id: str
    yes_token_id: str
    no_token_id:  str
    # YES leg (direct from CSV)
    yes_bid:      float   # bestBid column
    yes_ask:      float   # bestAsk column
    # NO leg (reconstructed via binary market complementarity)
    no_bid:       float   # = 1.0 − yes_ask
    no_ask:       float   # = 1.0 − yes_bid
    # Market metadata
    category:     str
    row_index:    int


# ═══════════════════════════════════════════════════════════════════════════════
# HistoricalDataFeed
# ═══════════════════════════════════════════════════════════════════════════════

class HistoricalDataFeed:
    """
    Downloads the Polymarket dataset from Kaggle and streams rows as
    OrderBookTick objects into an asyncio.Queue, tick by tick.

    Parameters
    ----------
    env_path  : Path to .env file (default ".env" relative to cwd).
    max_rows  : Optional row cap — useful for quick smoke-tests.
    csv_path  : Optional pre-downloaded CSV path; skips the Kaggle download.
    """

    def __init__(
        self,
        env_path: str | Path = ".env",
        max_rows: Optional[int] = None,
        csv_path: Optional[Path] = None,
    ) -> None:
        load_dotenv(env_path, override=False)
        self._configure_kaggle_auth()
        self._max_rows  = max_rows
        self._csv_path: Optional[Path] = csv_path

    # ── Kaggle authentication ─────────────────────────────────────────────────

    def _configure_kaggle_auth(self) -> None:
        """
        Map KAGGLE_API_TOKEN → KAGGLE_USERNAME + KAGGLE_KEY that kagglehub
        reads.  Runs before any import of kagglehub so the env vars are set
        before the library's module-level auth check fires.

        Supports:
          KAGGLE_API_TOKEN=username:api_key  → splits on first ':'
          KAGGLE_USERNAME + KAGGLE_KEY       → already in env, no-op
        """
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()

        if ":" in token:
            username, api_key = token.split(":", 1)
            # setdefault → never overwrite keys already set by the user
            os.environ.setdefault("KAGGLE_USERNAME", username.strip())
            os.environ.setdefault("KAGGLE_KEY", api_key.strip())
            logger.debug(
                "HistoricalDataFeed | Kaggle auth configured from KAGGLE_API_TOKEN"
                " (user=%s)", username.strip()
            )
        elif token:
            # Token is just the API key; KAGGLE_USERNAME must be set separately
            os.environ.setdefault("KAGGLE_KEY", token)
            logger.debug(
                "HistoricalDataFeed | Kaggle auth: bare key from KAGGLE_API_TOKEN"
            )
        else:
            # Fall through — may already have KAGGLE_USERNAME + KAGGLE_KEY set
            if not os.environ.get("KAGGLE_KEY"):
                logger.warning(
                    "HistoricalDataFeed | KAGGLE_API_TOKEN not set and KAGGLE_KEY "
                    "is missing.  Download will fail unless ~/.kaggle/kaggle.json "
                    "exists."
                )

    # ── Dataset download ──────────────────────────────────────────────────────

    def download(self) -> Path:
        """
        Programmatically download the dataset via kagglehub.

        kagglehub caches the dataset locally after the first download; subsequent
        calls return the cached path immediately with no network traffic.

        Returns
        -------
        Path to the polymarket_markets.csv file.
        """
        import kagglehub  # deferred — avoids mandatory install for unit tests

        logger.info(
            "HistoricalDataFeed | downloading dataset '%s' via kagglehub …",
            _KAGGLE_DATASET,
        )
        dataset_dir = kagglehub.dataset_download(_KAGGLE_DATASET)
        base = Path(dataset_dir)
        logger.info("HistoricalDataFeed | dataset downloaded to %s", base)

        # Locate the target CSV (handles sub-directory layouts)
        csv_candidates = sorted(base.rglob(_CSV_FILENAME))
        if not csv_candidates:
            # Fallback: any CSV in the download
            csv_candidates = sorted(base.rglob("*.csv"))

        if not csv_candidates:
            raise FileNotFoundError(
                f"No CSV found under {base}.  "
                f"Expected '{_CSV_FILENAME}' inside the Kaggle download.  "
                f"Check that the dataset still contains this file."
            )

        chosen = csv_candidates[0]
        if len(csv_candidates) > 1:
            logger.debug(
                "HistoricalDataFeed | multiple CSVs found; using %s", chosen
            )
        else:
            logger.info("HistoricalDataFeed | CSV located at %s", chosen)

        self._csv_path = chosen
        return chosen

    # ── Async tick generator ──────────────────────────────────────────────────

    async def stream(self, queue: "asyncio.Queue[Optional[OrderBookTick]]") -> None:
        """
        Parse the CSV row-by-row and enqueue one OrderBookTick per valid row.

        After every enqueue the coroutine yields control with
        `await asyncio.sleep(0)`.  This physically prevents the downstream
        strategy from seeing future ticks — each tick is consumed by the event
        loop before the next one is loaded.

        When all rows are exhausted, sends `None` as a sentinel to signal
        end-of-stream to the consumer.

        Parameters
        ----------
        queue : asyncio.Queue shared with the consumer (run_backtest.py).
        """
        if self._csv_path is None:
            self._csv_path = self.download()

        df = self._load_and_normalise_csv(self._csv_path)

        row_count  = len(df)
        tick_count = 0
        skipped    = 0

        logger.info(
            "HistoricalDataFeed | streaming %d rows into queue …", row_count
        )

        for idx, row in df.iterrows():
            tick = self._row_to_tick(row, int(idx))  # type: ignore[arg-type]
            if tick is None:
                skipped += 1
                continue

            await queue.put(tick)
            tick_count += 1

            # ── CRITICAL: yield after every enqueue ───────────────────────
            # Handing control back to the event loop here is what prevents
            # look-ahead bias.  The consumer's `await queue.get()` will fire
            # before we load the next row.
            await asyncio.sleep(0)

        await queue.put(None)   # sentinel — stream exhausted

        logger.info(
            "HistoricalDataFeed | stream complete: %d ticks queued, %d skipped",
            tick_count, skipped,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_and_normalise_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, nrows=self._max_rows, low_memory=False)
        logger.info(
            "HistoricalDataFeed | loaded %d rows — columns: %s",
            len(df), list(df.columns),
        )
        return _normalise_columns(df)

    def _row_to_tick(self, row: "pd.Series", idx: int) -> Optional[OrderBookTick]:
        """
        Convert one CSV row to an OrderBookTick.

        Handles multiple column naming conventions observed in Polymarket
        Kaggle exports (camelCase, snake_case, abbreviated).

        Returns None for rows with invalid or out-of-range price data so the
        strategy never sees malformed inputs.
        """
        # ── YES ask (primary price column) ───────────────────────────────────
        yes_ask = _col_float(row, (
            "bestask", "best_ask", "ask", "yes_ask", "close", "price",
        ))
        if yes_ask is None or not (0.01 <= yes_ask <= 0.99):
            return None   # skip: price out of valid Polymarket range

        # ── YES bid ───────────────────────────────────────────────────────────
        yes_bid = _col_float(row, (
            "bestbid", "best_bid", "bid", "yes_bid", "open",
        ))
        if yes_bid is None or not (0.01 <= yes_bid <= yes_ask):
            # Synthesise a plausible bid if missing (1 tick below ask)
            yes_bid = max(0.01, round(yes_ask - 0.001, 3))

        # ── NO leg — reconstructed via binary market identity ─────────────────
        #   YES_bid + NO_ask ≈ 1.0   →   no_ask = 1.0 − yes_bid
        #   YES_ask + NO_bid ≈ 1.0   →   no_bid = 1.0 − yes_ask
        no_ask = round(1.0 - yes_bid, 4)
        no_bid = round(1.0 - yes_ask, 4)

        if not (0.01 <= no_ask <= 0.99):
            return None   # market is effectively expired / trivial

        # ── Market identifiers ────────────────────────────────────────────────
        condition_id = str(
            _col_str(row, (
                "condition_id", "conditionid", "market_id", "id",
                "question_id", "questionid",
            )) or f"0x{idx:040x}"
        )
        yes_token_id = str(
            _col_str(row, (
                "yes_token_id", "yestokenid", "token_id_yes",
                "yes_token", "tokenid_yes",
            )) or f"{condition_id}_YES"
        )
        no_token_id = str(
            _col_str(row, (
                "no_token_id", "notokenid", "token_id_no",
                "no_token", "tokenid_no",
            )) or f"{condition_id}_NO"
        )

        # ── Metadata ──────────────────────────────────────────────────────────
        timestamp = str(
            _col_str(row, (
                "timestamp", "created_at", "date", "end_date_iso",
                "startdate", "start_date", "updated_at",
            )) or str(idx)
        )
        category = str(
            _col_str(row, (
                "category", "market_type", "type", "slug",
                "groupitemtitle", "group",
            )) or "other"
        ).lower().strip()

        return OrderBookTick(
            timestamp=timestamp,
            condition_id=condition_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_bid=round(yes_bid, 4),
            yes_ask=round(yes_ask, 4),
            no_bid=no_bid,
            no_ask=no_ask,
            category=category,
            row_index=idx,
        )


# ── Column accessor helpers ───────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, spaces/hyphens → underscores in column names."""
    return df.rename(columns={
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    })

def _col_float(
    row: "pd.Series",
    candidates: tuple[str, ...],
) -> Optional[float]:
    """Return the first parseable float from `candidates` column names."""
    for name in candidates:
        if name in row.index:
            try:
                val = float(row[name])
                if val == val:   # NaN check (NaN != NaN)
                    return val
            except (TypeError, ValueError):
                pass
    return None


def _col_str(
    row: "pd.Series",
    candidates: tuple[str, ...],
) -> Optional[str]:
    """Return the first non-empty string value from `candidates` column names."""
    for name in candidates:
        if name in row.index:
            val = row[name]
            if val is not None:
                s = str(val).strip()
                if s and s.lower() not in ("nan", "none", "nat", ""):
                    return s
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# OrderFilledDataFeed — warproxxx/poly_data adapter
# ═══════════════════════════════════════════════════════════════════════════════

_USDC_TOKEN_ID = "0"


class OrderFilledDataFeed:
    """
    Synthetic OrderBookTick stream built from warproxxx/poly_data:
      orderFilled.csv  — raw on-chain fill events; price derived from amounts
      markets.csv      — token-id → condition_id + YES/NO mapping

    For each fill, the complementary leg is resolved from the last seen trade
    within `align_window_s` seconds. Because YES and NO prices come from
    independent trades, their combined cost can fall below 1.0 — making genuine
    Dutch-book signals detectable.

    Limitations vs real orderbook data
    ────────────────────────────────────
    • Trade price ≈ last-executed price, not a live bid/ask quote
    • Only emits ticks when both legs have traded recently
    • Arb signals reflect prices at which trades cleared, not resting quotes
    """

    def __init__(
        self,
        orders_path: Path,
        markets_path: Path,
        align_window_s: float = 30.0,
        max_rows: Optional[int] = None,
    ) -> None:
        self._orders_path = Path(orders_path)
        self._markets_path = Path(markets_path)
        self._align_window_s = align_window_s
        self._max_rows = max_rows

    # ── Market / token map ────────────────────────────────────────────────────

    def _build_token_map(self, markets_df: pd.DataFrame) -> dict[str, dict]:
        """
        Returns: token_id → {condition_id, is_yes, category,
                              yes_token_id, no_token_id}
        """
        token_map: dict[str, dict] = {}

        for row in markets_df.itertuples(index=False):
            condition_id = str(
                getattr(row, "conditionid", None) or getattr(row, "condition_id", None) or ""
            ).strip().lower()

            category = str(
                getattr(row, "category", None) or getattr(row, "market_type", None) or "other"
            ).strip().lower()

            clob_raw = getattr(row, "clobtokenids", None) or getattr(row, "clob_token_ids", None) or ""
            try:
                token_ids = json.loads(str(clob_raw)) if clob_raw else []
            except (json.JSONDecodeError, TypeError):
                continue

            if len(token_ids) < 2:
                continue

            yes_id = str(token_ids[0]).strip()
            no_id  = str(token_ids[1]).strip()

            if not condition_id or not yes_id or not no_id:
                continue

            base = {
                "condition_id": condition_id,
                "category":     category,
                "yes_token_id": yes_id,
                "no_token_id":  no_id,
            }
            token_map[yes_id] = {**base, "is_yes": True}
            token_map[no_id]  = {**base, "is_yes": False}

        return token_map

    # ── Tick generation ───────────────────────────────────────────────────────

    def _generate_ticks(
        self,
        orders_df: pd.DataFrame,
        token_map: dict[str, dict],
    ):
        """
        Walk fills chronologically. Maintain last-seen price per token.
        Emit a tick whenever both legs of a market have traded within
        align_window_s seconds of each other.
        """
        ts_col = next(
            (c for c in ("timestamp", "created_at", "block_timestamp")
             if c in orders_df.columns),
            None,
        )
        if ts_col is None:
            raise ValueError("No timestamp column found in orderFilled CSV")

        orders_df = orders_df.sort_values(ts_col).reset_index(drop=True)

        # last_price[token_id] = (unix_ts_float, price)
        last_price: dict[str, tuple[float, float]] = {}

        for idx, row in orders_df.iterrows():
            maker_asset  = str(row.get("makerassetid")      or "").strip()
            taker_asset  = str(row.get("takerassetid")      or "").strip()
            try:
                maker_amt = float(row.get("makeramountfilled") or 0)
                taker_amt = float(row.get("takeramountfilled") or 0)
            except (TypeError, ValueError):
                continue

            if maker_amt <= 0 or taker_amt <= 0:
                continue

            # Identify which leg is USDC and derive price
            # (both USDC and conditional tokens use 6 decimals → scale cancels)
            if maker_asset == _USDC_TOKEN_ID:
                token_id = taker_asset
                price    = maker_amt / taker_amt
            elif taker_asset == _USDC_TOKEN_ID:
                token_id = maker_asset
                price    = taker_amt / maker_amt
            else:
                continue  # neither side is USDC

            if not (0.01 <= price <= 0.99):
                continue
            if token_id not in token_map:
                continue

            try:
                ts_val = float(row[ts_col])
            except (TypeError, ValueError):
                ts_val = float(idx)

            last_price[token_id] = (ts_val, price)

            info         = token_map[token_id]
            complement   = info["no_token_id"] if info["is_yes"] else info["yes_token_id"]

            if complement not in last_price:
                continue

            comp_ts, comp_price = last_price[complement]
            if abs(ts_val - comp_ts) > self._align_window_s:
                continue

            yes_ask = price      if info["is_yes"] else comp_price
            no_ask  = comp_price if info["is_yes"] else price

            if not (0.01 <= yes_ask <= 0.99 and 0.01 <= no_ask <= 0.99):
                continue

            yes_bid = round(max(0.01, yes_ask - 0.001), 4)
            no_bid  = round(max(0.01, no_ask  - 0.001), 4)

            yield OrderBookTick(
                timestamp    = str(int(ts_val)),
                condition_id = info["condition_id"],
                yes_token_id = info["yes_token_id"],
                no_token_id  = info["no_token_id"],
                yes_bid      = yes_bid,
                yes_ask      = round(yes_ask, 4),
                no_bid       = no_bid,
                no_ask       = round(no_ask, 4),
                category     = info["category"],
                row_index    = int(idx),  # type: ignore[arg-type]
            )

    # ── Async stream ──────────────────────────────────────────────────────────

    async def stream(
        self,
        queue: "asyncio.Queue[Optional[OrderBookTick]]",
    ) -> None:
        # on_bad_lines="skip": real Gamma exports embed nested JSON whose commas
        # /quotes occasionally break a row; a crash-truncated last row is also
        # possible. Skipping a handful of malformed rows out of millions is fine.
        markets_df = _normalise_columns(
            pd.read_csv(self._markets_path, low_memory=False, on_bad_lines="skip")
        )

        token_map = self._build_token_map(markets_df)
        logger.info(
            "OrderFilledDataFeed | token map: %d tokens across %d markets",
            len(token_map), len(markets_df),
        )

        # dtype=str preserves large token-ID integers that pandas would otherwise
        # convert to float (e.g. "1000...0" → 1e+76), breaking token map lookups.
        orders_df = _normalise_columns(
            pd.read_csv(
                self._orders_path,
                nrows=self._max_rows,
                dtype=str,
                on_bad_lines="skip",
            )
        )

        logger.info("OrderFilledDataFeed | loaded %d fills", len(orders_df))

        tick_count = 0
        for i, tick in enumerate(self._generate_ticks(orders_df, token_map)):
            await queue.put(tick)
            tick_count += 1
            if i % 100 == 0:
                await asyncio.sleep(0)

        logger.info(
            "OrderFilledDataFeed | %d ticks generated (window=%.0fs)",
            tick_count, self._align_window_s,
        )
        await queue.put(None)


# ═══════════════════════════════════════════════════════════════════════════════
# TradesParquetDataFeed — SII-WANGZJ/Polymarket_data adapter
# ═══════════════════════════════════════════════════════════════════════════════

_HF_DATASET = "SII-WANGZJ/Polymarket_data"

# Best-effort event_slug → category mapping
_SLUG_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("crypto", "bitcoin", "btc", "eth", "solana", "sol", "xrp", "doge", "matic"), "crypto"),
    (("politic", "election", "president", "congress", "senate", "vote", "democrat", "republican"), "politics"),
    (("nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "tennis", "golf",
      "esport", "valorant", "counter-strike", "dota", "league-of-legends", "sport"), "sports"),
    (("entertain", "movie", "film", "music", "award", "oscar", "grammy", "emmy"), "entertainment"),
    (("econ", "gdp", "inflation", "fed", "rate", "recession"), "economy"),
    (("science", "space", "nasa", "ai", "tech", "climate"), "science"),
]


def _slug_to_category(slug: str) -> str:
    s = slug.lower()
    for keywords, cat in _SLUG_KEYWORDS:
        if any(k in s for k in keywords):
            return cat
    return "other"


class TradesParquetDataFeed:
    """
    Synthetic OrderBookTick stream from SII-WANGZJ/Polymarket_data via
    HuggingFace Datasets streaming — no full 28 GB download required.

    Uses trades.parquet (price pre-computed, YES/NO explicit via nonusdc_side)
    and markets.parquet (85 MB, loaded fully) for token1/token2 IDs.

    trades.parquet key columns
    ───────────────────────────
      timestamp    uint64   Unix seconds
      condition_id string   CTF condition ID
      asset_id     string   Non-USDC token ID traded
      price        float64  Trade price (0–1)
      nonusdc_side string   "token1" = YES, "token2" = NO

    markets.parquet key columns
    ────────────────────────────
      condition_id string
      token1       string   YES token asset ID
      token2       string   NO  token asset ID
      event_slug   string   Used for category inference

    Limitations (same as OrderFilledDataFeed)
    ──────────────────────────────────────────
    Trade prices approximate the ask at execution time, not live quotes.
    Arb signals are sparse — most real-market combined costs stay ≥ 1.0.
    """

    def __init__(
        self,
        align_window_s: float = 30.0,
        max_rows: Optional[int] = None,
    ) -> None:
        self._align_window_s = align_window_s
        self._max_rows       = max_rows

    # ── Token map ─────────────────────────────────────────────────────────────

    def _build_token_map(self) -> dict[str, dict]:
        """Download markets.parquet (85 MB) and build token_id → market info."""
        from datasets import load_dataset  # deferred import

        logger.info("TradesParquetDataFeed | loading markets.parquet from HuggingFace …")
        ds = load_dataset(_HF_DATASET, data_files="markets.parquet", split="train")
        df = ds.to_pandas()  # Arrow → pandas: vectorised, much faster than row iteration
        df = _normalise_columns(df)

        # Vectorised category derivation
        df["category"] = df.get("event_slug", pd.Series("", index=df.index)) \
                           .fillna("").apply(_slug_to_category)

        token_map: dict[str, dict] = {}
        needed = ["condition_id", "token1", "token2", "category"]
        available = [c for c in needed if c in df.columns]
        if "token1" not in df.columns or "token2" not in df.columns:
            raise ValueError("markets.parquet missing token1/token2 columns")

        for rec in df[available].dropna(subset=["condition_id", "token1", "token2"]) \
                                 .to_dict("records"):
            cid    = str(rec["condition_id"]).strip().lower()
            token1 = str(rec["token1"]).strip()
            token2 = str(rec["token2"]).strip()
            if not cid or not token1 or not token2:
                continue
            cat  = rec.get("category", "other")
            base = {"condition_id": cid, "category": cat,
                    "yes_token_id": token1, "no_token_id": token2}
            token_map[token1] = {**base, "is_yes": True}
            token_map[token2] = {**base, "is_yes": False}

        logger.info("TradesParquetDataFeed | token map: %d tokens", len(token_map))
        return token_map

    # ── Tick generator ────────────────────────────────────────────────────────

    def _generate_ticks(
        self,
        trades_iter,
        token_map: dict[str, dict],
    ):
        last_price: dict[str, tuple[float, float]] = {}

        for idx, row in enumerate(trades_iter):
            if self._max_rows and idx >= self._max_rows:
                break

            asset_id     = str(row.get("asset_id")     or "").strip()
            nonusdc_side = str(row.get("nonusdc_side")  or "").strip()
            price_raw    = row.get("price")
            ts_raw       = row.get("timestamp")

            if price_raw is None:
                continue
            try:
                price  = float(price_raw)
                ts_val = float(ts_raw) if ts_raw is not None else float(idx)
            except (TypeError, ValueError):
                continue

            if not (0.01 <= price <= 0.99):
                continue
            if asset_id not in token_map:
                continue

            info = token_map[asset_id]

            # Sanity-check nonusdc_side against token_map
            expected_yes = nonusdc_side == "token1"
            if expected_yes != info["is_yes"]:
                continue  # token map disagrees with nonusdc_side — skip

            last_price[asset_id] = (ts_val, price)

            complement = info["no_token_id"] if info["is_yes"] else info["yes_token_id"]
            if complement not in last_price:
                continue

            comp_ts, comp_price = last_price[complement]
            if abs(ts_val - comp_ts) > self._align_window_s:
                continue

            yes_ask = price      if info["is_yes"] else comp_price
            no_ask  = comp_price if info["is_yes"] else price

            if not (0.01 <= yes_ask <= 0.99 and 0.01 <= no_ask <= 0.99):
                continue

            yield OrderBookTick(
                timestamp    = str(int(ts_val)),
                condition_id = info["condition_id"],
                yes_token_id = info["yes_token_id"],
                no_token_id  = info["no_token_id"],
                yes_bid      = round(max(0.01, yes_ask - 0.001), 4),
                yes_ask      = round(yes_ask, 4),
                no_bid       = round(max(0.01, no_ask  - 0.001), 4),
                no_ask       = round(no_ask,  4),
                category     = info["category"],
                row_index    = idx,
            )

    # ── Async stream ──────────────────────────────────────────────────────────

    async def stream(
        self,
        queue: "asyncio.Queue[Optional[OrderBookTick]]",
    ) -> None:
        from datasets import load_dataset  # deferred import

        token_map = self._build_token_map()

        logger.info(
            "TradesParquetDataFeed | streaming trades.parquet "
            "(max_rows=%s, window=%.0fs) …",
            self._max_rows or "all", self._align_window_s,
        )
        trades_ds = load_dataset(
            _HF_DATASET,
            data_files="trades.parquet",
            split="train",
            streaming=True,
        )

        tick_count = 0
        for i, tick in enumerate(self._generate_ticks(iter(trades_ds), token_map)):
            await queue.put(tick)
            tick_count += 1
            if i % 100 == 0:
                await asyncio.sleep(0)

        logger.info("TradesParquetDataFeed | %d ticks enqueued", tick_count)
        await queue.put(None)
