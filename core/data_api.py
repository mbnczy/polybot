"""
core/data_api.py
────────────────
Polymarket Data API v2: the wallet's positions and activity, and market trades.

v1 was retired on 2026-11-02 and answers every route with HTTP 410. v2 wraps rows
in `data`, names fields in snake_case (`token_id`, `usdc_size`), and pages only by
cursor: `offset` is rejected, `limit` goes on the first page only, and the filters
are sent again on every page. https://docs.polymarket.com/migrate/data-api-v1-to-v2
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

BASE_URL = "https://data-api.polymarket.com/v2"

# A wallet holds a few dozen positions at most. Past this something is wrong, and
# half a wallet must not pass for all of it.
_MAX_PAGES = 20


def next_cursor(body: dict) -> "str | None":
    """The cursor of the next page, or None on the last one."""
    return (body.get("pagination") or {}).get("next_cursor") or None


async def fetch_page(http: Any, path: str, params: dict) -> "tuple[list[dict], str | None] | None":
    """One page as (rows, next cursor), or None if the API did not answer with one."""
    resp = await http.get(f"{BASE_URL}/{path}", params=params)
    if resp.status_code != 200:
        logger.warning("Data API /v2/%s returned HTTP %d", path, resp.status_code)
        return None
    body = resp.json()
    rows = body.get("data") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        logger.warning("Data API /v2/%s answered without a data list", path)
        return None
    return rows, next_cursor(body)


async def fetch_all(http: Any, path: str, params: dict, *, limit: int = 500) -> "list[dict] | None":
    """
    Every row across all pages, or None if any page failed.

    All or nothing: the reconciler reads a missing position as shares that moved,
    so a partial answer would raise a false alarm.
    """
    rows: list[dict] = []
    cursor = None
    for _ in range(_MAX_PAGES):
        query = {**params, "cursor": cursor} if cursor else {**params, "limit": limit}
        page = await fetch_page(http, path, query)
        if page is None:
            return None
        rows += page[0]
        cursor = page[1]
        if cursor is None:
            return rows
    logger.warning("Data API /v2/%s still had pages after %d — giving up", path, _MAX_PAGES)
    return None


def market_trades(
    get: "Callable[[str, dict], Any]",
    condition_id: str,
    *,
    since: float = 0.0,
    max_rows: int = 10_000,
    page_size: int = 500,
) -> list[dict]:
    """
    A market's trades, newest first, back to `since` or `max_rows`.

    For the research scripts. `get(url, params)` returns a response or None, and a
    failed page ends the walk with what came before it. `asset` is kept next to
    `token_id` because that is the name the scripts match on.
    """
    out: list[dict] = []
    cursor = None
    while len(out) < max_rows:
        query = {"condition": condition_id}
        query.update({"cursor": cursor} if cursor else {"limit": page_size})
        resp = get(f"{BASE_URL}/trades", query)
        if resp is None or resp.status_code != 200:
            break
        body = resp.json()
        page = body.get("data") or []
        out += [{**t, "asset": str(t.get("token_id") or "")} for t in page]
        cursor = next_cursor(body)
        if cursor is None or (page and page[-1]["timestamp"] < since):
            break
    return out
