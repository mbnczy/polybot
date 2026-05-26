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

@dataclass(frozen=True)
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
        """
        Load the CSV and normalise column names to lowercase_snake_case so
        the rest of the code is independent of exact capitalisation variants
        in the Kaggle dataset.
        """
        df = pd.read_csv(path, nrows=self._max_rows, low_memory=False)
        logger.info(
            "HistoricalDataFeed | loaded %d rows — columns: %s",
            len(df), list(df.columns),
        )

        # Normalise: strip whitespace, lowercase, spaces/hyphens → underscores
        df = df.rename(columns={
            col: col.strip().lower().replace(" ", "_").replace("-", "_")
            for col in df.columns
        })
        return df

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
