#!/usr/bin/env python3
"""
demo_cross_market.py — cross-market implication arbitrage, READ ONLY.

Runs the whole pipeline against the LIVE Polymarket universe:

    fetch markets → prefilter pairs → LLM classify → register implications
                 → poll live prices → detect ordering violations → Telegram

Two things get alerted, and they are different:

  IMPLICATION FOUND    the model asserts A ⊆ B above a confidence threshold.
                       Informational — a relationship, not yet an opportunity.

  ARBITRAGE DETECTED   a registered implication is VIOLATED by live prices:
                       P(narrow) > P(broad), which is logically impossible, so
                       buying broad-YES + narrow-NO costs < 1.00 for a payout
                       floor of 1.00. This is the tradeable event.

SAFETY — this process is structurally incapable of trading. It never constructs
a signing client, never imports an order-placement path, and holds no private
key. It reads public market data, calls the model, and sends Telegram messages.
The production bot is untouched: separate process, separate directory, separate
systemd unit, no shared state files.

Usage
─────
    python scripts/demo_cross_market.py --dry-run        # print, send nothing
    python scripts/demo_cross_market.py                  # one pass, alert
    python scripts/demo_cross_market.py --loop 900       # daemon: re-poll prices

Configuration comes from two .env files:
    ./.env                    IMPLICATION_PROVIDER / _BASE_URL / _API_KEY / _MODEL
    /home/ubuntu/polybot/.env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logger = logging.getLogger("demo")

_GAMMA = "https://gamma-api.polymarket.com/markets"
_BOOK  = "https://clob.polymarket.com/book"


# ── market data (public, read-only) ───────────────────────────────────────────

_GAMMA_MAX_OFFSET = 2000   # beyond this Gamma answers 422 "use /markets/keyset"


def _page(params: dict, limit: int) -> list[dict]:
    """Page one Gamma query, stopping cleanly at its offset ceiling."""
    import httpx

    out: list[dict] = []
    offset = 0
    with httpx.Client(timeout=30) as h:
        while len(out) < limit and offset < _GAMMA_MAX_OFFSET:
            r = h.get(_GAMMA, params={**params, "limit": 100, "offset": offset})
            if r.status_code != 200:
                logger.warning("gamma %s at offset %d: %s", r.status_code, offset,
                               r.text[:120])
                break
            page = r.json()
            if not page:
                break
            out += page
            offset += len(page)
    return out[:limit]


def fetch_markets(
    limit: int, fast_days: float = 0.0, fast_limit: int = 0,
    fast_min_hours: float = 1.0,
) -> list[dict]:
    """
    The active universe, plus the markets that resolve soonest.

    Unordered, Gamma's active universe is almost all long-dated: measured on
    800 live markets, none resolved within a week and the median was 112 days.
    So the prefilter's time priority had nothing to promote — every candidate
    pair it could see locked capital until year end.

    A second query ordered by end date and bounded by `fast_days` brings the
    fast ones in. What it finds is mostly match sub-markets — "Team X O/U 0.5"
    under "O/U 0.5", "1st Half O/U 0.5" under "O/U 0.5" — genuine implications
    that resolve in hours rather than months. Merged and de-duplicated with the
    general pool so the long-dated coverage is kept, not traded away.
    """
    base = {"active": "true", "closed": "false", "archived": "false"}
    out = _page(base, limit)
    if fast_days > 0 and fast_limit > 0:
        from datetime import timedelta, timezone        # noqa: PLC0415
        now = datetime.now(timezone.utc)
        iso = "%Y-%m-%dT%H:%M:%SZ"
        fast = _page({
            **base, "order": "endDate", "ascending": "true",
            # A full timestamp, not a date. Gamma reads a bare "2026-09-11" as
            # that day's midnight, so a date-only lower bound returned 100 of 100
            # markets whose end time had ALREADY passed — ended but not yet
            # resolved, the worst place there is to read a price. The same query
            # with the current time returned 0 of those and 100 genuinely ahead.
            #
            # `fast_min_hours` skips the last stretch before the end. A match
            # sub-market closing within the hour is usually in play, where the
            # price jumps on every event and a REST snapshot is stale before it
            # lands.
            "end_date_min": (now + timedelta(hours=fast_min_hours)).strftime(iso),
            "end_date_max": (now + timedelta(days=fast_days)).strftime(iso),
        }, fast_limit)
        seen = {str(m.get("conditionId")) for m in out}
        added = [m for m in fast if str(m.get("conditionId")) not in seen]
        out += added
        print(f"    +{len(added)} market(s) resolving within {fast_days:g} days")
    return out


def yes_price(market: dict) -> float | None:
    """
    Live P(YES) for a market, taken as the best ASK on the YES token.

    The ask is what it would cost to acquire the outcome now, which is the
    conservative read of the market's implied probability — using the mid would
    flatter every spread into looking like an opportunity.
    """
    import httpx

    try:
        toks = market.get("clobTokenIds")
        if isinstance(toks, str):
            toks = json.loads(toks)
        if not toks:
            return None
        r = httpx.get(_BOOK, params={"token_id": toks[0]}, timeout=20)
        if r.status_code != 200:
            return None
        asks = sorted(r.json().get("asks", []), key=lambda x: float(x["price"]))
        if not asks:
            return None
        p = float(asks[0]["price"])
        return p if 0.0 < p < 1.0 else None
    except Exception as exc:                        # noqa: BLE001 — demo, stay up
        logger.debug("price fetch failed: %s", exc)
        return None


# ── telegram ──────────────────────────────────────────────────────────────────

def fmt_implication(rel, titles: dict[str, str]) -> str:
    narrow = titles.get(rel.narrow, rel.narrow[:16])
    broad  = titles.get(rel.broad,  rel.broad[:16])
    reason = rel.evidence.split("—", 1)[-1].strip() if "—" in rel.evidence else ""
    model  = rel.evidence.split(":", 1)[0].strip() if ":" in rel.evidence else "?"
    return (
        f"<b>🔗 Implication found</b>\n"
        f"<i>confidence {rel.confidence:.2f} · {model}</i>\n\n"
        f"<b>IF</b> this resolves YES…\n  {narrow}\n\n"
        f"<b>THEN</b> this must too:\n  {broad}\n\n"
        f"<i>{reason}</i>"
    )


def fmt_arbitrage(sig) -> str:
    """The tradeable event: an implication contradicted by live prices."""
    return (
        f"<b>💰 CROSS-MARKET ARBITRAGE</b>\n"
        f"<i>edge {sig.edge * 10_000:+.0f} bps · confidence {sig.confidence:.2f}"
        + (f" · capital locked {sig.lockup_days:.0f}d → {sig.apr * 100:.0f}% APR"
           if sig.lockup_days > 0 else " · resolution date unknown")
        + "</i>\n\n"
        f"<b>Narrower market</b> — priced <b>{sig.narrow_price:.3f}</b>\n"
        f"  {sig.narrow_title}\n\n"
        f"<b>Broader market</b> — priced <b>{sig.broad_price:.3f}</b>\n"
        f"  {sig.broad_title}\n\n"
        f"The narrower outcome cannot be more likely than the broader one, "
        f"yet it is priced <b>{sig.violation:.3f}</b> higher.\n\n"
        f"<b>Trade</b>\n"
        f"  buy broad YES  @ {sig.broad_price:.3f}\n"
        f"  buy narrow NO  @ {1 - sig.narrow_price:.3f}\n"
        f"  cost <b>{sig.cost:.4f}</b> → payout floor <b>{sig.min_payout:.2f}</b>\n"
        f"  edge <b>{sig.edge:+.4f}</b> per pair\n\n"
        f"<i>READ ONLY — no order placed.</i>"
    )


async def send_all(messages: list[str]) -> int:
    from telemetry.telegram import TelegramNotifier

    notifier = TelegramNotifier()
    sent = 0
    try:
        for m in messages:
            if await notifier.notify(m, parse_mode="HTML"):
                sent += 1
    finally:
        await notifier.close()
    return sent


def deliver(messages: list[str], dry_run: bool) -> None:
    if not messages:
        return
    for m in messages:
        plain = (m.replace("<b>", "").replace("</b>", "")
                  .replace("<i>", "").replace("</i>", "")
                  .replace("<code>", "").replace("</code>", ""))
        print("  " + "─" * 68)
        for line in plain.splitlines():
            print(f"  {line}")
    print("  " + "─" * 68)
    if dry_run:
        print(f"  DRY RUN — {len(messages)} message(s) not sent.")
        return
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        print("  TELEGRAM_BOT_TOKEN not set — cannot send.")
        return
    sent = asyncio.run(send_all(messages))
    print(f"  sent {sent}/{len(messages)} Telegram message(s).")


# ── stages ────────────────────────────────────────────────────────────────────

def discover(markets: list[dict], args) -> list:
    """Stage 1 — expensive, infrequent. Ask the model for implications."""
    import strategy.implication_mapper as im

    provider = im.resolve_provider()
    model    = im.resolve_model(provider)
    # A fast match fields a dozen sub-markets, every pair of them a nest, so one
    # fixture filled all 30 slots when measured. The cap buys breadth across
    # matches — ten fixtures at three pairs each rather than one at thirty.
    cands    = im.build_candidates(markets, max_pairs=args.pairs,
                                   max_per_event=args.max_per_event)
    print(f"  prefilter  : {len(cands)} candidate pair(s)")
    if not cands:
        return []
    print(f"  classifying on {provider.name}/{model}…")
    # Build the client here and close it here. Left to classify_candidates it
    # would make a fresh one per discovery round, each with its own connection
    # pool that nobody closes: 12 hours of running left 30 sockets to the model
    # endpoint in CLOSE-WAIT. The reader's file-descriptor limit is 1024, so a
    # long-lived process eventually runs out.
    client = im.build_client(provider)
    try:
        rels = im.classify_candidates(
            cands, provider=provider, model=model, client=client,
            concurrency=args.concurrency,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    strong = [r for r in rels if r.confidence >= args.threshold]
    print(f"    {len(rels)} asserted · {len(strong)} at/above {args.threshold}")
    return strong


def _drop_backwards(rels: list, markets: list[dict]) -> list:
    """
    Refuse implications whose direction contradicts their own ladder family.

    The ten largest arbitrages this reader ever reported were one falling ladder
    read as a rising one: "Trump approval hit 35%" called narrower than "hit
    25%", when falling to 25% requires passing 35% first. The prices were right;
    the implication was backwards; a trade on it could pay nothing on either
    leg. Filtered here, straight after discovery, so a backwards relation is
    neither announced nor priced. See strategy/ladder_direction.py.
    """
    from strategy.ladder_direction import filter_implications  # noqa: PLC0415

    kept, backwards = filter_implications(rels, markets)
    if backwards:
        titles = {str(m.get("conditionId")): str(m.get("question") or "")
                  for m in markets}
        print(f"  {len(backwards)} implication(s) refused — direction contradicts "
              f"the ladder they belong to")
        for r in backwards[:5]:
            print(f"    ✗ {titles.get(r.narrow, r.narrow[:12])[:54]}\n"
                  f"      does not imply {titles.get(r.broad, r.broad[:12])[:48]}")
    return kept


def check_prices(rels: list, markets: list[dict], args) -> list:
    """
    Stage 2 — cheap, repeatable. Price both legs and look for violations.

    Only markets that appear in a registered implication are priced, so this
    costs a couple of book reads per relation rather than a universe sweep.
    """
    from strategy.cross_market import (
        CrossMarketDetector, RelationRegistry,
    )

    if not rels:
        return []
    reg = RelationRegistry()
    reg.extend(rels)
    det = CrossMarketDetector(
        reg, min_edge=args.min_edge, min_confidence=args.threshold,
    )

    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    needed = {r.narrow for r in rels} | {r.broad for r in rels}
    print(f"  pricing {len(needed)} leg(s) from live books…")

    # Resolution dates, so the detector can price how long capital would be
    # locked. A cross-market pair has no complete set to merge, so the money
    # sits until the LATER leg resolves — the difference between a 2% edge over
    # two days and the same 2% over eight months.
    for cid in needed:
        m = by_id.get(cid)
        if not m:
            continue
        end = m.get("endDate") or m.get("end_date")
        if not end:
            continue
        try:
            ts = datetime.fromisoformat(str(end).replace("Z", "+00:00")).timestamp()
            det.set_resolution(cid, ts)
        except (ValueError, TypeError):
            pass

    signals = []
    now = time.time()
    for cid in needed:
        m = by_id.get(cid)
        if not m:
            continue
        p = yes_price(m)
        if p is None:
            continue
        # One REST snapshot prices every leg, so they share an observation time
        # by construction. The freshness checks still matter under a live feed,
        # where each book updates on its own clock — measured median skew on a
        # related pair was 5.6 s.
        signals += det.update_price(cid, p, str(m.get("question") or ""), ts=now)
    print(f"    {len(signals)} violation(s) detected")
    if det.rejected_stale or det.rejected_skew or det.rejected_lockup:
        print(f"    filtered: {det.rejected_stale} stale, "
              f"{det.rejected_skew} skewed, {det.rejected_lockup} lockup")
    return signals


# ── positions: following the arbitrages the reader finds ─────────────────────
#
# A cross-market arbitrage pays at least 1.00 per pair, but only at resolution —
# and the pairs this reader finds resolve months out. Completing a full set on
# each market (buy YES on narrow, NO on broad) merges to 2.00 immediately, and
# once the market has corrected that returns the guaranteed edge plus the price
# of the middle outcome, in weeks instead of months. See strategy/cross_exit.py.
#
# These are PAPER positions: nothing is bought. The point is to find out, on
# live books, how often and how fast the corrections actually come, before any
# money depends on the answer.

def book_top(token_id: str) -> tuple[float | None, float | None]:
    """
    (best bid, best ask) for one token, or Nones.

    Scanned for extremes rather than read off the ends: the exchange sends bids
    ascending and asks descending, so bids[0] and asks[0] are the WORST prices.
    """
    import httpx

    try:
        r = httpx.get(_BOOK, params={"token_id": token_id}, timeout=20)
        if r.status_code != 200:
            return None, None
        body = r.json()
        bids = [float(x["price"]) for x in body.get("bids", []) if float(x["price"]) > 0]
        asks = [float(x["price"]) for x in body.get("asks", []) if float(x["price"]) > 0]
        return (max(bids) if bids else None, min(asks) if asks else None)
    except Exception as exc:                        # noqa: BLE001 — stay up
        logger.debug("book fetch failed for %s: %s", token_id[:12], exc)
        return None, None


def _tokens_of(market: dict | None) -> tuple[str, str] | None:
    """(YES token, NO token) for a binary market."""
    if not market:
        return None
    toks = market.get("clobTokenIds")
    if isinstance(toks, str):
        try:
            toks = json.loads(toks)
        except ValueError:
            return None
    if not toks or len(toks) != 2:
        return None
    return str(toks[0]), str(toks[1])


def _market_end_ts(market: dict | None) -> float | None:
    if not market:
        return None
    end = market.get("endDate") or market.get("end_date")
    if not end:
        return None
    try:
        return datetime.fromisoformat(str(end).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _resolved_outcome(condition_id: str) -> bool | None:
    """
    True if YES won, False if NO won, None while unresolved or unclear.

    Gamma filters on `condition_ids`. The camel-case `conditionId` is silently
    ignored and returns an unfiltered page — which is how a query for one
    market's fee once returned a different market's every time.
    """
    import httpx

    try:
        r = httpx.get(_GAMMA, params={"condition_ids": condition_id}, timeout=20)
        rows = r.json() if r.status_code == 200 else []
    except Exception:                               # noqa: BLE001
        return None
    for m in rows:
        if str(m.get("conditionId")) != condition_id:
            continue
        try:
            yes, no = (float(x) for x in json.loads(m.get("outcomePrices") or "[]"))
        except (TypeError, ValueError):
            return None
        if {yes, no} != {0.0, 1.0}:
            return None
        return yes == 1.0
    return None


def fmt_position_opened(pos, sig) -> str:
    days = pos.days_to_resolution()
    when = f"resolves in {days:.0f} days" if days is not None else "resolution date unknown"
    shortfall = sig.edge - pos.guaranteed_edge
    return (
        "📒 CROSS POSITION OPENED — paper, nothing bought\n"
        f"  NO  {pos.narrow_title[:56]}  @ {pos.narrow_no_paid:.4f}\n"
        f"  YES {pos.broad_title[:56]}  @ {pos.broad_yes_paid:.4f}\n"
        f"  entry {pos.entry_cost:.4f}/pair → guaranteed {pos.guaranteed_edge:+.4f} "
        f"({when})\n"
        f"  the signal said {sig.edge:+.4f}; the real NO ask and fees cost "
        f"{shortfall:.4f} of it\n"
        "  Watching for the market to correct."
    )


def fmt_position_closed(pos) -> str:
    from strategy.cross_exit import annualised

    held = pos.days_held()
    apr = annualised(pos.exit_profit, pos.entry_cost, held)
    head = {
        "exited": "🔓 CROSS POSITION EXITED EARLY — paper",
        "resolved": "🏁 CROSS POSITION RESOLVED — paper",
    }.get(pos.status, "CROSS POSITION CLOSED")
    lines = [
        head,
        f"  {pos.narrow_title[:56]}",
        f"  {pos.broad_title[:56]}",
        f"  route {pos.exit_route or '—'} · held {held:.1f} days",
        f"  {pos.exit_profit:+.4f}/pair against {pos.guaranteed_edge:+.4f} guaranteed "
        f"· {pos.exit_profit * pos.size:+.2f} USDC on {pos.size:g} pairs",
    ]
    if apr is not None:
        lines.append(f"  {apr * 100:.0f}% annualised")
    return "\n".join(lines)


def confirm_at_asks(sigs: list, markets: list[dict]) -> tuple[list, list]:
    """
    Split violations into (still an arbitrage at the real asks, not).

    The detector prices NO on narrow as 1 − the YES ask. On a thin book that is
    badly wrong, and it reached the operator: the first live pass after this
    branch went up alerted "+200 bps" on "Moik Baku O/U 0.5" under "O/U 0.5" —
    narrow YES at 0.98, so "NO @ 0.02" — when the only NO ask on the book was
    0.83 and the real entry 1.79. The paper tracker refused it in the same pass,
    but the Telegram alert had already gone. A violation the system can see is
    not real should not reach anyone as an opportunity.

    A leg with no ask at all is not an opportunity either: there is nobody to buy
    it from. A violation whose markets cannot be found keeps the old behaviour
    and is alerted — suppressing on missing data would hide more than it saves.
    """
    from strategy.cross_exit import buy_cost        # noqa: PLC0415

    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    real, illusory = [], []
    for s in sigs:
        tn, tb = _tokens_of(by_id.get(s.narrow)), _tokens_of(by_id.get(s.broad))
        if not tn or not tb:
            real.append(s)
            continue
        _, narrow_no_ask = book_top(tn[1])
        _, broad_yes_ask = book_top(tb[0])
        if narrow_no_ask is None or broad_yes_ask is None:
            illusory.append(s)
            logger.info("cross-market | %s/%s signalled %+.4f but a leg has no ask "
                        "— nothing to buy, not alerted", s.narrow[:10], s.broad[:10],
                        s.edge)
            continue
        entry = buy_cost(narrow_no_ask) + buy_cost(broad_yes_ask)
        if entry >= 1.0:
            illusory.append(s)
            logger.info("cross-market | %s/%s signalled %+.4f but costs %.4f at the "
                        "real asks — not alerted", s.narrow[:10], s.broad[:10],
                        s.edge, entry)
        else:
            real.append(s)
    return real, illusory


def track_positions(sigs: list, markets: list[dict], args, cache: dict) -> list[str]:
    """
    Open a paper position for each new violation, then price every open one's
    exits and close it when leaving beats holding.
    """
    from strategy.cross_exit import (
        CrossPosition, PositionBook, best_exit, buy_cost, resolution_profit,
        should_exit,
    )

    book = cache.get("positions")
    if book is None:
        book = PositionBook(args.positions_file).load()
        cache["positions"] = book
    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    now = time.time()
    msgs: list[str] = []
    changed = False

    # ── open ──────────────────────────────────────────────────────────────────
    for s in sigs:
        if book.has_open(s.narrow, s.broad):
            continue
        mn, mb = by_id.get(s.narrow), by_id.get(s.broad)
        tn, tb = _tokens_of(mn), _tokens_of(mb)
        if not tn or not tb:
            continue
        # Priced at what the legs actually cost. The signal prices NO on narrow
        # as 1 − the YES ask; the NO ask is nearer 1 − the YES bid, which puts
        # the real entry about one spread higher — a median of 100 bps on the
        # live book. Opening at the signal's price would book an edge that was
        # never available.
        _, narrow_no_ask = book_top(tn[1])
        _, broad_yes_ask = book_top(tb[0])
        if narrow_no_ask is None or broad_yes_ask is None:
            continue
        no_paid, yes_paid = buy_cost(narrow_no_ask), buy_cost(broad_yes_ask)
        if no_paid + yes_paid >= 1.0:
            cache["illusory"] = cache.get("illusory", 0) + 1
            logger.info(
                "cross positions | %s/%s signalled %+.4f but costs %.4f at the "
                "real asks — not an arbitrage, not opened",
                s.narrow[:10], s.broad[:10], s.edge, no_paid + yes_paid,
            )
            continue
        ends = [_market_end_ts(mn), _market_end_ts(mb)]
        pos = CrossPosition(
            narrow=s.narrow, broad=s.broad,
            narrow_title=s.narrow_title, broad_title=s.broad_title,
            narrow_yes_token=tn[0], narrow_no_token=tn[1],
            broad_yes_token=tb[0], broad_no_token=tb[1],
            size=args.paper_size,
            narrow_no_paid=no_paid, broad_yes_paid=yes_paid,
            opened_ts=now,
            resolves_ts=max(ends) if None not in ends else None,
            paper=True,
        )
        if book.open(pos):
            changed = True
            msgs.append(fmt_position_opened(pos, s))

    # ── watch ─────────────────────────────────────────────────────────────────
    for pos in book.open_positions():
        if pos.resolves_ts is not None and now >= pos.resolves_ts:
            ny, by = _resolved_outcome(pos.narrow), _resolved_outcome(pos.broad)
            if ny is not None and by is not None:
                book.close(pos, route="resolution",
                           profit=resolution_profit(pos, ny, by),
                           status="resolved", now=now)
                changed = True
                msgs.append(fmt_position_closed(pos))
                continue

        narrow_no_bid, _ = book_top(pos.narrow_no_token)
        _, narrow_yes_ask = book_top(pos.narrow_yes_token)
        broad_yes_bid, _ = book_top(pos.broad_yes_token)
        _, broad_no_ask = book_top(pos.broad_no_token)
        q = best_exit(
            pos,
            narrow_yes_ask=narrow_yes_ask, broad_no_ask=broad_no_ask,
            narrow_no_bid=narrow_no_bid, broad_yes_bid=broad_yes_bid,
        )
        if should_exit(pos, q, min_capture=args.exit_min_capture):
            book.close(pos, route=q.route, profit=q.profit, status="exited", now=now)
            changed = True
            msgs.append(fmt_position_closed(pos))
        else:
            logger.info(
                "cross positions | %s/%s holding — best exit %s %+.4f against "
                "%+.4f guaranteed",
                pos.narrow[:10], pos.broad[:10], q.route or "none", q.profit,
                pos.guaranteed_edge,
            )

    if changed:
        book.save()
    sm = book.summary()
    print(f"  positions (paper): {sm['open']} open · {sm['exited_early']} exited "
          f"early · {sm['resolved']} resolved · realised "
          f"{sm['realised_usdc']:+.2f} USDC"
          + (f" · {cache['illusory']} signal(s) not real at the asks"
             if cache.get("illusory") else ""))
    return msgs


# ── main ──────────────────────────────────────────────────────────────────────

def _load_announced(args) -> set:
    """
    Implications already announced, across restarts.

    In-memory dedup is not enough: the process restarts on deploys and on
    failure, and each restart re-announced the whole set — ten identical
    messages, which is exactly the flood the dedup exists to stop.
    """
    path = Path(args.state_file)
    try:
        rows = json.loads(path.read_text())
        return {tuple(r) for r in rows if isinstance(r, list) and len(r) == 2}
    except FileNotFoundError:
        return set()
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN could not read {path}: {exc} — starting empty")
        return set()


def _save_announced(args, announced: set) -> None:
    path = Path(args.state_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(list(p) for p in announced)))
        tmp.replace(path)          # atomic, so a crash mid-write cannot corrupt it
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN could not persist {path}: {exc}")


def one_pass(args, markets_cache: dict) -> int:
    markets = markets_cache.get("markets")
    if markets is None:
        print(f"  fetching up to {args.markets} live markets…")
        markets = fetch_markets(args.markets, args.fast_days, args.fast_markets,
                                args.fast_min_hours)
        markets_cache["markets"] = markets
        print(f"    got {len(markets)}")

    titles = {
        str(m.get("conditionId")): str(m.get("question") or "")
        for m in markets if m.get("conditionId")
    }

    # Implications are stable; discover once and reuse across price polls.
    rels = markets_cache.get("rels")
    if rels is None:
        rels = _drop_backwards(discover(markets, args), markets)
        markets_cache["rels"] = rels

        # Only announce relations we have not announced before.
        #
        # Rediscovery re-runs the model over the same universe and gets the same
        # answers, so without this every cycle re-sent an identical batch — the
        # Fed rate-cut ladders arrived again every 90 minutes, unchanged. A
        # relation is news exactly once; after that it is a fact about the world
        # that has not moved.
        announced = markets_cache.setdefault("announced", _load_announced(args))
        fresh = [r for r in rels if (r.narrow, r.broad) not in announced]
        if fresh and not args.no_implication_alerts:
            deliver([fmt_implication(r, titles) for r in fresh[:args.max_alerts]],
                    args.dry_run)
        announced.update((r.narrow, r.broad) for r in rels)
        _save_announced(args, announced)
        if rels and not fresh:
            print(f"  {len(rels)} implication(s), all already announced — quiet.")

    sigs = check_prices(rels, markets, args)
    sigs, illusory = confirm_at_asks(sigs, markets)
    if illusory:
        markets_cache["illusory"] = markets_cache.get("illusory", 0) + len(illusory)
        print(f"  {len(illusory)} violation(s) NOT alerted — not an arbitrage at the "
              f"asks actually payable")
    if sigs:
        # An arbitrage is a price condition, not a fact: it can open, close and
        # reopen on the same pair, and each opening is worth saying. Re-alert
        # only after the pair has been quiet for a cooldown, so a violation that
        # simply persists across polls does not repeat every cycle.
        now = time.time()
        last = markets_cache.setdefault("last_arb", {})
        due = [
            s for s in sigs
            if now - last.get((s.narrow, s.broad), 0.0) >= args.arb_cooldown
        ]
        if due:
            deliver([fmt_arbitrage(s) for s in due[:args.max_alerts]], args.dry_run)
            for s in due:
                last[(s.narrow, s.broad)] = now
        else:
            print(f"  {len(sigs)} violation(s), all within the alert cooldown.")
    else:
        print("  no price violation this pass.")

    if not args.no_paper:
        msgs = track_positions(sigs, markets, args, markets_cache)
        if msgs:
            deliver(msgs[:args.max_alerts], args.dry_run)
    return len(sigs)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets",     type=int,   default=800)
    ap.add_argument("--pairs",       type=int,   default=30,
                    help="candidate pairs sent to the model")
    ap.add_argument("--threshold",   type=float, default=0.90,
                    help="minimum implication confidence (default 0.90)")
    ap.add_argument("--min-edge",    type=float, default=0.02,
                    help="minimum price violation to alert on (default 0.02)")
    ap.add_argument("--max-alerts",  type=int,   default=10,
                    help="cap per pass, so a demo cannot spam")
    ap.add_argument("--concurrency", type=int,   default=4)
    ap.add_argument("--loop",        type=int,   default=0, metavar="SECONDS",
                    help="re-poll prices every N seconds (0 = single pass)")
    ap.add_argument("--rediscover",  type=int,   default=6, metavar="PASSES",
                    help="re-run the model every N passes when looping")
    ap.add_argument("--state-file", default=str(REPO / "announced.json"),
                    help="where announced implications are remembered across "
                         "restarts (default: announced.json in the worktree)")
    ap.add_argument("--arb-cooldown", type=float, default=3600.0, metavar="SEC",
                    help="do not re-alert the same violated pair within this "
                         "window (default 3600)")
    ap.add_argument("--fast-days", type=float, default=30.0, metavar="DAYS",
                    help="also fetch markets resolving within this many days "
                         "(default 30; 0 disables)")
    ap.add_argument("--fast-min-hours", type=float, default=1.0, metavar="HOURS",
                    help="skip markets ending sooner than this — usually in play "
                         "(default 1)")
    ap.add_argument("--fast-markets", type=int, default=400,
                    help="how many soonest-resolving markets to add (default 400)")
    ap.add_argument("--max-per-event", type=int, default=3,
                    help="candidate pairs allowed from any one event (default 3)")
    ap.add_argument("--positions-file",
                    default=str(REPO / "cross_positions.json"),
                    help="where paper positions are kept across restarts")
    ap.add_argument("--paper-size", type=float, default=10.0, metavar="PAIRS",
                    help="pairs per paper position (default 10)")
    ap.add_argument("--exit-min-capture", type=float, default=None,
                    metavar="FRACTION",
                    help="fraction of the guaranteed edge an early exit must "
                         "capture (default CROSS_EXIT_MIN_CAPTURE, 1.0)")
    ap.add_argument("--no-paper", action="store_true",
                    help="do not open or track paper positions")
    ap.add_argument("--no-implication-alerts", action="store_true",
                    help="only alert on price violations, not on discoveries")
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--env-file",    default=str(REPO / ".env"))
    ap.add_argument("--bot-env",     default="/home/ubuntu/polybot/.env")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from dotenv import load_dotenv
    load_dotenv(args.env_file)                      # LLM settings win
    load_dotenv(args.bot_env, override=False)       # bot's .env fills in Telegram

    import strategy.implication_mapper as im
    provider = im.resolve_provider()
    print(f"\n  provider   : {provider.name} · {im.resolve_model(provider)}")
    print(f"  endpoint   : {provider.base_url or '(default)'}")
    print(f"  threshold  : implication {args.threshold} · edge {args.min_edge}")
    print(f"  telegram   : {'DRY RUN' if args.dry_run else 'live'}")
    print(f"  mode       : READ ONLY — no order is ever placed\n")

    cache: dict = {}
    if not args.loop:
        one_pass(args, cache)
        return 0

    print(f"  looping every {args.loop}s; re-discovering every "
          f"{args.rediscover} passes\n")
    n = 0
    while True:
        n += 1
        print(f"  ── pass {n} ── {time.strftime('%H:%M:%S')}")
        try:
            if n > 1 and args.rediscover and n % args.rediscover == 1:
                cache.pop("markets", None)
                cache.pop("rels", None)
            one_pass(args, cache)
        except KeyboardInterrupt:
            print("\n  stopped.")
            return 0
        except Exception as exc:                    # noqa: BLE001 — daemon stays up
            logger.error("pass failed: %s", exc)
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
