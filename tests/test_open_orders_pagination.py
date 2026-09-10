"""
tests/test_open_orders_pagination.py
────────────────────────────────────
PolyClient.get_open_orders() drives a money decision indirectly.

InventoryManager._poll_fills reads it and treats "this order id is not in the
open set" as proof the order filled — booking a LegFill at price 0.0, promoting
the position to PAIRED and scheduling an on-chain mergePositions.

The SDK's list_open_orders() returns a Paginator, and iterating a Paginator
yields PAGES, not orders. `list(paginator)` therefore produced a single Page
object, which the row mapper turned into one order with an empty id — so the
open-order set was blank no matter what was actually resting on the book, and
every PENDING leg looked filled.

Checked on a live account on 2026-09-09: zero open orders, reported as one.
"""

import asyncio

import pytest

from core.clob_client import PolyClient


class _Page:
    def __init__(self, items, has_more=False, next_cursor=None):
        self.items = tuple(items)
        self.has_more = has_more
        self.next_cursor = next_cursor


class _Paginator:
    """Mirrors the SDK: iterating gives pages, iter_items() gives orders."""

    def __init__(self, pages):
        self._pages = pages

    def __iter__(self):
        return iter(self._pages)

    def iter_items(self):
        for p in self._pages:
            yield from p.items


class _Order:
    def __init__(self, oid):
        self.id = oid
        self.order_id = oid
        self.status = "LIVE"
        self.size_matched = 0.0
        self.original_size = 10.0
        self.price = 0.5
        self.side = "BUY"
        self.asset_id = "tok"
        self.token_id = "tok"


def _open_orders(pages) -> list[dict]:
    """Run get_open_orders() against an SDK whose paginator holds `pages`."""
    return asyncio.run(_client(pages).get_open_orders())


def _client(pages):
    c = PolyClient.__new__(PolyClient)

    class _SDK:
        def list_open_orders(self_inner):
            return _Paginator(pages)

    c._client = _SDK()

    async def _run_with_retry(fn, *a, **kw):
        return fn()
    c._run_with_retry = _run_with_retry
    return c


def test_orders_are_read_out_of_the_pages():
    rows = _open_orders([_Page([_Order("a"), _Order("b")])])
    assert {r.get("id") or r.get("order_id") for r in rows} == {"a", "b"}


def test_every_page_is_walked():
    rows = _open_orders([_Page([_Order("a")], has_more=True, next_cursor="n"),
                         _Page([_Order("b")])])
    assert len(rows) == 2


def test_an_empty_book_reports_nothing_rather_than_one_blank_order():
    """
    The regression itself. An account with no resting orders must return an
    empty list — a blank placeholder here makes _poll_fills believe every
    tracked leg has filled.
    """
    rows = _open_orders([_Page([])])
    assert rows == []


def test_a_resting_order_is_never_reported_as_a_blank_id():
    """A real order must carry its id, or _poll_fills cannot match it."""
    rows = _open_orders([_Page([_Order("real-order-1")])])
    assert len(rows) == 1
    assert (rows[0].get("id") or rows[0].get("order_id")) == "real-order-1"
