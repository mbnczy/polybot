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
import re
import threading
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


# The fields the reader and the prefilter read. A Gamma market is ~7 KB of JSON;
# across the whole one-week window (~40,000 markets) that is 280 MB raw and well
# over the reader's 1 GB cap as Python objects. Slimmed, it is a few hundred bytes.
_KEEP_FIELDS = ("conditionId", "question", "endDate", "outcomes", "clobTokenIds",
                "negRiskMarketID", "outcomePrices", "bestBid", "bestAsk", "updatedAt")


def _slim(m: dict) -> dict:
    out = {k: m[k] for k in _KEEP_FIELDS if m.get(k) is not None}
    ev = m.get("events")
    if isinstance(ev, list) and ev and isinstance(ev[0], dict) and ev[0].get("id") is not None:
        out["events"] = [{"id": str(ev[0]["id"])}]
    return out


_SLICE_HOURS = 6.0
# Halving stops at a minute. Below that the markets share an end time — every
# sub-market of one match closes together — and no time slice can separate them;
# what still hits the page ceiling is reported, not silently dropped.
_MIN_SLICE_S = 60.0
_FETCH_LOCAL = threading.local()


def _thread_client():
    """One Gamma client per fetch thread, not per request (see _http)."""
    client = getattr(_FETCH_LOCAL, "client", None)
    if client is None:
        import httpx  # noqa: PLC0415
        client = _FETCH_LOCAL.client = httpx.Client(timeout=30)
    return client


def _gamma_get(params: dict) -> "list[dict] | None":
    for attempt in range(4):
        try:
            r = _thread_client().get(_GAMMA, params=params)
        except Exception as exc:  # noqa: BLE001 — retried, then given up
            logger.warning("gamma request failed: %s", exc)
            time.sleep(1.0 + attempt)
            continue
        # A rate limit or a server error is transient. Giving up on it mid-slice
        # would leave that slice's later pages silently unfetched.
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code != 200:
            logger.warning("gamma %s: %s", r.status_code, r.text[:120])
            return None
        return r.json()
    return None


def _fetch_slice(lo, hi) -> "tuple[list[dict] | None, bool]":
    """
    Every market ending in [lo, hi]: (markets, still capped). (None, True) when
    the slice holds more than one query can page, so the caller splits it.
    """
    iso = "%Y-%m-%dT%H:%M:%SZ"
    params = {"active": "true", "closed": "false", "archived": "false", "limit": 100,
              "order": "endDate", "ascending": "true",
              "end_date_min": lo.strftime(iso), "end_date_max": hi.strftime(iso)}
    if (hi - lo).total_seconds() > _MIN_SLICE_S:
        probe = _gamma_get({**params, "offset": _GAMMA_MAX_OFFSET - 100})
        if probe is not None and len(probe) >= 100:
            return None, True
    out: list[dict] = []
    offset = 0
    while offset < _GAMMA_MAX_OFFSET:
        page = _gamma_get({**params, "offset": offset})
        if not page:
            break
        out.extend(_slim(m) for m in page)
        offset += len(page)
        if len(page) < 100:
            break
    return out, offset >= _GAMMA_MAX_OFFSET


def fetch_window(days: float, min_hours: float = 1.0, concurrency: int = 4,
                 exclude: "str | None" = None) -> list[dict]:
    """
    Every market resolving within `days` — not just the first page of them.

    The first version asked for the 400 soonest-ending markets. On 2026-09-15
    those 400 covered 90 minutes: 189 temperature markets and ~180 five-minute
    crypto "Up or Down" markets. The first football market sat at position
    1,390, and for hours the reader exported nothing at all.

    The window holds about 40,000 markets and one Gamma query pages at most
    2,000. So it is cut into slices, a slice too full for one query is halved
    until it fits, slices are fetched in parallel, and each market is slimmed to
    the fields that are read. `exclude` drops title series that cannot hold an
    implication and would only crowd the prefilter.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from datetime import timedelta, timezone          # noqa: PLC0415

    now = datetime.now(timezone.utc).replace(microsecond=0)
    start, end = now + timedelta(hours=min_hours), now + timedelta(days=days)
    pending, t = [], start
    while t < end:
        nxt = min(t + timedelta(hours=_SLICE_HOURS), end)
        pending.append((t, nxt))
        t = nxt
    seen: set[str] = set()
    out: list[dict] = []
    splits = capped = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while pending:
            # One second of overlap between neighbours: a market ending exactly
            # on a boundary is fetched twice and de-duplicated, never missed.
            results = list(pool.map(
                lambda sl: _fetch_slice(sl[0], sl[1] + timedelta(seconds=1)), pending))
            next_round = []
            for (lo, hi), (rows, full) in zip(pending, results):
                if rows is None:
                    mid = lo + timedelta(seconds=int((hi - lo).total_seconds() // 2))
                    next_round += [(lo, mid), (mid, hi)]
                    splits += 1
                    continue
                capped += bool(full)
                for m in rows:
                    cid = m.get("conditionId")
                    if cid and cid not in seen:
                        seen.add(cid)
                        out.append(m)
            pending = next_round
    rx = re.compile(exclude) if exclude else None
    kept = [m for m in out if not (rx and rx.search(str(m.get("question") or "")))]
    print(f"    {len(kept)} market(s) resolving within {days:g} days "
          f"({len(out) - len(kept)} excluded by title, {splits} slice split(s)"
          f"{f', {capped} slice(s) still at the page ceiling' if capped else ''})")
    return kept


_RULES: "dict[str, str]" = {}
_RULES_MAX = 20_000


def fetch_rules(cids, concurrency: int = 4) -> "dict[str, str]":
    """
    Resolution-rule digests (strategy/market_rules.py) for the markets about to be
    classified.

    Only for those: a discovery sends a few hundred markets to the model, and
    carrying a digest for all ~70,000 in the window would add tens of megabytes
    to a reader already peaking at 735 MB. Gamma answers up to 100 markets per
    request, so this is a handful of requests; digests are kept between passes.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from strategy.market_rules import rules_digest    # noqa: PLC0415

    want = [c for c in dict.fromkeys(cids) if c and c not in _RULES]
    batches = [want[i:i + 50] for i in range(0, len(want), 50)]

    def one(batch):
        return _gamma_get([("condition_ids", c) for c in batch] + [("limit", "100")]) or []

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for rows in pool.map(one, batches):
            for m in rows:
                if m.get("conditionId"):
                    _RULES[str(m["conditionId"])] = rules_digest(m.get("description"))
    while len(_RULES) > _RULES_MAX:
        _RULES.pop(next(iter(_RULES)))
    return {c: _RULES.get(c, "") for c in cids}


def fetch_resolved(cids) -> list[dict]:
    """Markets by condition id, closed ones included, for the resolution audit."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    ids = list(dict.fromkeys(c for c in cids if c))
    batches = [ids[i:i + 50] for i in range(0, len(ids), 50)]

    def one(batch):
        # Only a closed market can be judged, and Gamma returns closed markets
        # only when asked for them.
        return _gamma_get([("condition_ids", c) for c in batch]
                          + [("closed", "true"), ("limit", "100")]) or []

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(one, batches):
            out += rows
    return out


def run_resolution_audit(args, cache: dict) -> "object":
    """Check resolved pairs; forbid the violated ones and alert on newly blocked templates."""
    from strategy.resolution_audit import ResolutionAudit  # noqa: PLC0415
    from strategy.verdict_cache import VerdictCache        # noqa: PLC0415

    audit = cache.get("audit")
    if audit is None:
        audit = cache["audit"] = ResolutionAudit(args.audit_file).load()
    summary = audit.run(fetch_resolved)
    if summary["violated"]:
        verdicts = VerdictCache(args.cache_file).load()
        for p in summary["violated"]:
            verdicts.forbid(p["narrow"], p["broad"], "violated by its resolution")
        verdicts.save()
        bad = {(p["narrow"], p["broad"]) for p in summary["violated"]}
        if cache.get("rels"):
            cache["rels"] = [r for r in cache["rels"] if (r.narrow, r.broad) not in bad]
    audit.save()
    print(f"  audit      : {summary['checked']} resolved pair(s) checked · {summary['held']} held · "
          f"{len(summary['violated'])} violated · {summary['pending']} awaiting resolution · "
          f"blocked templates: {', '.join(audit.blocked_templates()) or 'none'}")
    if summary["newly_blocked"]:
        lines = []
        for t in summary["newly_blocked"]:
            st = audit.templates[t]
            lines.append(f"<b>🚫 IMPLICATION TEMPLATE BLOCKED</b>\n{t}\n"
                         f"{st['violated']} of {st['held'] + st['violated']} resolved pairs paid "
                         f"less than 1\n" + "\n".join(f"  • {e}" for e in st["examples"]))
        deliver(lines, args.dry_run)
    return audit


def snapshot_yes_ask(market: dict) -> float | None:
    """
    P(YES) as the best YES ask in the fetched snapshot — no book read.

    Reading a book per leg was a request per leg per pass; over the whole window
    that is thousands every five minutes. The snapshot is up to one discovery
    old, which is fine for flagging: anything flagged is confirmed against the
    live book before it reaches anyone (confirm_at_asks).
    """
    try:
        p = float(market.get("bestAsk"))
    except (TypeError, ValueError):
        return None
    return p if 0.0 < p < 1.0 else None


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
        r = _http().get(_BOOK, params={"token_id": toks[0]})
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
    """
    Stage 1 — ask the model for implications, but only about pairs it has not
    seen before.

    A verdict never expires: both questions are immutable once listed. The cache
    therefore turns the model's cost from per-pass into one-time, which is what
    lets the net be thousands of pairs wide instead of thirty. Each pass spends
    its budget (--new-per-pass) on pairs nobody has judged yet, so coverage
    grows every cycle while the pass stays short.
    """
    import strategy.implication_mapper as im
    from strategy.verdict_cache import VerdictCache

    provider = im.resolve_provider()
    model    = im.resolve_model(provider)
    # A fast match fields a dozen sub-markets, every pair of them a nest, so one
    # fixture filled all 30 slots when measured. The cap buys breadth across
    # matches — ten fixtures at three pairs each rather than one at thirty.
    cands    = im.build_candidates(markets, max_pairs=args.pairs,
                                   max_per_event=args.max_per_event)
    cache = VerdictCache(args.cache_file).load()
    known, unknown = cache.split(cands)
    fresh = unknown[:args.new_per_pass]
    print(f"  prefilter  : {len(cands)} candidate pair(s) | "
          f"{len(cands) - len(unknown)} already judged ({len(cache)} in cache), "
          f"{len(unknown)} new → classifying {len(fresh)}")
    if not fresh:
        strong = [r for r in known if r.confidence >= args.threshold]
        print(f"    {len(known)} from cache · {len(strong)} at/above {args.threshold}")
        return strong
    cands = fresh
    # The rules decide an implication, not the titles (strategy/market_rules.py).
    from strategy.market_rules import rules_digest  # noqa: PLC0415
    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    needed = {c.a_id for c in cands} | {c.b_id for c in cands}
    rules = {cid: rules_digest(by_id[cid]["description"]) for cid in needed
             if by_id.get(cid, {}).get("description")}
    rules.update(fetch_rules([cid for cid in needed if cid not in rules],
                             getattr(args, "fetch_concurrency", 4)))
    print(f"  rules      : {sum(1 for cid in needed if rules.get(cid))}/{len(needed)} market(s)")
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
            concurrency=args.concurrency, rules=rules,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    cache.remember(cands, rels)
    cache.save()
    rels = known + rels
    strong = [r for r in rels if r.confidence >= args.threshold]
    print(f"    {len(rels)} asserted ({len(known)} from cache) · "
          f"{len(strong)} at/above {args.threshold}")
    return strong


def export_implications(rels: list, markets: list[dict], path: str, audit=None) -> int:
    """
    Hand the verified implications to the trading bot.

    The reader stays read-only by construction — its systemd unit cannot even
    see the wallet's .env — so it does not trade. It writes what it has found,
    after the direction guard, to one JSON file the bot's cross guard reads and
    prices live. Everything the bot needs to act is in the file: both markets'
    token pairs and end dates, so it never has to trust a title.

    Written atomically: the bot must never read half a file.
    """
    from strategy.outcomes import market_outcomes  # noqa: PLC0415

    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    now = time.time()
    rows, expired, blocked = [], 0, 0
    for r in rels:
        mn, mb = by_id.get(r.narrow), by_id.get(r.broad)
        tn, tb = _tokens_of(mn), _tokens_of(mb)
        if not tn or not tb:
            continue
        # A pair whose later market has ended pays nothing and cannot be
        # entered; the bot would only read it to refuse it.
        ends = [e for e in (_market_end_ts(mn), _market_end_ts(mb)) if e is not None]
        if ends and max(ends) <= now:
            expired += 1
            continue
        # A pair proven wrong by its own resolution, or of a template that keeps
        # being wrong, never reaches the bot (strategy/resolution_audit.py).
        if audit is not None and (audit.violated(r.narrow, r.broad) or audit.is_blocked(
                str(mn.get("question") or ""), str(mb.get("question") or ""))):
            blocked += 1
            continue
        rows.append({
            "narrow": r.narrow, "broad": r.broad,
            "narrow_title": str(mn.get("question") or ""),
            "broad_title": str(mb.get("question") or ""),
            "narrow_yes_token": tn[0], "narrow_no_token": tn[1],
            "broad_yes_token": tb[0], "broad_no_token": tb[1],
            "narrow_end_ts": _market_end_ts(mn), "broad_end_ts": _market_end_ts(mb),
            # Which outcome each YES token is. The bot refuses a leg whose first
            # token is not provably "Yes" (strategy/outcomes.py).
            "narrow_outcomes": market_outcomes(mn), "broad_outcomes": market_outcomes(mb),
            "confidence": float(getattr(r, "confidence", 0.0)),
            "evidence": str(getattr(r, "evidence", ""))[:300],
        })
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps({"generated_at": time.time(), "implications": rows},
                                  indent=1))
        tmp.replace(p)
    except OSError as exc:
        logger.error("cannot write implications to %s: %s", p, exc)
        return 0
    if audit is not None:
        audit.record(rows)
        audit.save()
    print(f"  exported {len(rows)} implication(s) for the bot → {p.name}"
          + (f" ({expired} expired dropped)" if expired else "")
          + (f" ({blocked} refused by the resolution audit)" if blocked else ""))
    return len(rows)


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
    window = getattr(args, "window_only", False)
    print(f"  pricing {len(needed)} leg(s) from "
          f"{'the market snapshot' if window else 'live books'}…")

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
        p = snapshot_yes_ask(m) if window else yes_price(m)
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

_HTTP = None


def _http():
    """
    One HTTP client for every book and resolution read, for the life of the process.

    httpx.get() builds a whole client per call, and a client builds an SSL
    context, loading the CA bundle into native OpenSSL memory. The reader made
    over 80 of those a pass — a YES price and a book per leg — and on
    2026-09-14 grew 213 MB an hour: the contexts sit in reference cycles, so
    their native memory is only returned when a full collection happens to run,
    and tracemalloc cannot see it at all. One client means one context.
    """
    global _HTTP
    if _HTTP is None:
        import httpx  # noqa: PLC0415
        _HTTP = httpx.Client(timeout=20)
    return _HTTP


def book_top(token_id: str) -> tuple[float | None, float | None]:
    """
    (best bid, best ask) for one token, or Nones.

    Scanned for extremes rather than read off the ends: the exchange sends bids
    ascending and asks descending, so bids[0] and asks[0] are the WORST prices.
    """
    import httpx

    try:
        r = _http().get(_BOOK, params={"token_id": token_id})
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
        r = _http().get(_GAMMA, params={"condition_ids": condition_id})
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


def confirm_at_asks(sigs: list, markets: list[dict],
                    min_edge: float = 0.0) -> tuple[list, list]:
    """
    Split violations into (an arbitrage at the real asks, not).

    Only an edge the live book confirms is alerted, and the alert carries the
    confirmed prices and edge, not the snapshot's.

    The first version only asked whether the entry was under 1.00. On
    2026-09-15 "1st Half Spread: CA Osasuna (-2.5)" under "Spread" went out
    twelve times as "+5020 bps" while the real entry was 0.9999 — an edge of
    one hundredth of a cent, reported from the snapshot. Before that, "+200 bps"
    on "Moik Baku O/U 0.5" went out with a real entry of 1.79.

    A violation whose markets cannot be found, or with a leg nobody sells, cannot
    be confirmed and is not alerted.
    """
    import dataclasses                              # noqa: PLC0415
    from strategy.cross_exit import buy_cost        # noqa: PLC0415

    by_id = {str(m.get("conditionId")): m for m in markets if m.get("conditionId")}
    real, illusory = [], []
    for s in sigs:
        tn, tb = _tokens_of(by_id.get(s.narrow)), _tokens_of(by_id.get(s.broad))
        if not tn or not tb:
            illusory.append(s)
            logger.info("cross-market | %s/%s signalled %+.4f but its markets are not in "
                        "the snapshot — cannot confirm, not alerted",
                        s.narrow[:10], s.broad[:10], s.edge)
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
        edge = 1.0 - entry
        if edge < min_edge:
            illusory.append(s)
            logger.info("cross-market | %s/%s signalled %+.4f but %+.4f at the real asks "
                        "(entry %.4f) — not alerted", s.narrow[:10], s.broad[:10],
                        s.edge, edge, entry)
            continue
        apr = (edge / entry) * 365.0 / s.lockup_days if s.lockup_days > 0 and entry > 0 else 0.0
        real.append(dataclasses.replace(
            s, narrow_price=1.0 - narrow_no_ask, broad_price=broad_yes_ask,
            violation=(1.0 - narrow_no_ask) - broad_yes_ask, cost=entry, edge=edge, apr=apr))
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
        # The same bar as an alert: a paper position is a claim that the edge was
        # real. The Osasuna spread opened one at an entry of 0.9999.
        if 1.0 - (no_paid + yes_paid) < args.min_edge:
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


def _release_freed_memory() -> bool:
    """
    Hand memory Python has already freed back to the operating system.

    Collect first. The measurement that justified trimming alone forced a
    gc.collect() every pass, and production does not: with trim but no collect
    the reader still grew 213 MB an hour on 2026-09-14, because garbage in
    reference cycles — HTTP clients holding native SSL contexts — stays
    allocated until a full collection, and nothing can be trimmed while it is.

    glibc only for the trim; the collection runs everywhere.
    """
    import gc  # noqa: PLC0415
    gc.collect()
    try:
        import ctypes  # noqa: PLC0415
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        return False


def one_pass(args, markets_cache: dict) -> int:
    markets = markets_cache.get("markets")
    if markets is None:
        print(f"  fetching up to {args.markets} live markets…")
        if args.window_only:
            markets = fetch_window(args.fast_days, args.fast_min_hours,
                                   args.fetch_concurrency, args.exclude_title)
        else:
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
    fresh_discovery = rels is None
    audit = markets_cache.get("audit")
    if fresh_discovery and args.audit_file:
        try:
            audit = run_resolution_audit(args, markets_cache)
        except Exception as exc:                    # noqa: BLE001 — never stops a pass
            logger.error("resolution audit failed: %s", exc)
    if fresh_discovery:
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

    # Every pass, not only on rediscovery. The bot prices what is in this file,
    # and inside a one-week window the match sub-markets in it expire hourly: on
    # 2026-09-12 whole passes read "past_end 25" because the file was 90 minutes
    # old. Expired pairs are dropped as it is written.
    if audit is not None:
        rels = [r for r in rels if not audit.violated(r.narrow, r.broad)
                and not audit.is_blocked(titles.get(r.narrow, ""), titles.get(r.broad, ""))]
    export_implications(rels, markets, args.implications_file, audit)

    sigs = check_prices(rels, markets, args)
    sigs, illusory = confirm_at_asks(sigs, markets, args.min_edge)
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
    ap.add_argument("--audit-file",  default="cross_audit.json", metavar="PATH",
                    help="resolution audit ledger; empty to disable")
    ap.add_argument("--cache-file",  default="cross_verdicts.json", metavar="PATH",
                    help="where model verdicts are remembered between passes")
    ap.add_argument("--new-per-pass", type=int,  default=60, metavar="N",
                    help="how many unseen pairs each discovery sends to the model")
    ap.add_argument("--pairs",       type=int,   default=800,
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
    ap.add_argument("--rediscover",  type=int,   default=2, metavar="PASSES",
                    help="re-run the model every N passes when looping")
    ap.add_argument("--state-file", default=str(REPO / "announced.json"),
                    help="where announced implications are remembered across "
                         "restarts (default: announced.json in the worktree)")
    ap.add_argument("--arb-cooldown", type=float, default=3600.0, metavar="SEC",
                    help="do not re-alert the same violated pair within this "
                         "window (default 3600)")
    ap.add_argument("--fast-days", type=float, default=7.0, metavar="DAYS",
                    help="also fetch markets resolving within this many days "
                         "(default 7 — the execution window; 0 disables)")
    ap.add_argument("--fast-min-hours", type=float, default=1.0, metavar="HOURS",
                    help="skip markets ending sooner than this — usually in play "
                         "(default 1)")
    ap.add_argument("--window-only", action="store_true",
                    help="fetch every market resolving within --fast-days, in slices, "
                         "instead of the general pool plus the first --fast-markets")
    ap.add_argument("--exclude-title", default=None, metavar="REGEX",
                    help="drop markets whose title matches, e.g. 'Up or Down'")
    ap.add_argument("--fetch-concurrency", type=int, default=4, metavar="N")
    ap.add_argument("--fast-markets", type=int, default=400,
                    help="how many soonest-resolving markets to add (default 400)")
    ap.add_argument("--max-per-event", type=int, default=8,
                    help="candidate pairs allowed from any one event (default 3)")
    ap.add_argument("--implications-file",
                    default=str(REPO / "cross_implications.json"),
                    help="where verified implications are written for the bot")
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
    # httpx logs every request at INFO. Fetching the whole window is ~500 of them
    # per discovery — 70,000 journal lines a day saying "200 OK".
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
        _release_freed_memory()
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
