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
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logger = logging.getLogger("demo")

_GAMMA = "https://gamma-api.polymarket.com/markets"
_BOOK  = "https://clob.polymarket.com/book"


# ── market data (public, read-only) ───────────────────────────────────────────

def fetch_markets(limit: int) -> list[dict]:
    """Page the Gamma active universe, the same way core.scanner does."""
    import httpx

    out: list[dict] = []
    offset = 0
    with httpx.Client(timeout=30) as h:
        while len(out) < limit:
            r = h.get(_GAMMA, params={
                "active": "true", "closed": "false", "archived": "false",
                "limit": 100, "offset": offset,
            })
            if r.status_code != 200:
                break
            page = r.json()
            if not page:
                break
            out += page
            offset += 100
    return out[:limit]


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
        f"<i>edge {sig.edge * 10_000:+.0f} bps · confidence {sig.confidence:.2f}</i>\n\n"
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
    cands    = im.build_candidates(markets, max_pairs=args.pairs)
    print(f"  prefilter  : {len(cands)} candidate pair(s)")
    if not cands:
        return []
    print(f"  classifying on {provider.name}/{model}…")
    rels = im.classify_candidates(
        cands, provider=provider, model=model, concurrency=args.concurrency,
    )
    strong = [r for r in rels if r.confidence >= args.threshold]
    print(f"    {len(rels)} asserted · {len(strong)} at/above {args.threshold}")
    return strong


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

    signals = []
    for cid in needed:
        m = by_id.get(cid)
        if not m:
            continue
        p = yes_price(m)
        if p is None:
            continue
        signals += det.update_price(cid, p, str(m.get("question") or ""))
    print(f"    {len(signals)} violation(s) detected")
    return signals


# ── main ──────────────────────────────────────────────────────────────────────

def one_pass(args, markets_cache: dict) -> int:
    markets = markets_cache.get("markets")
    if markets is None:
        print(f"  fetching up to {args.markets} live markets…")
        markets = fetch_markets(args.markets)
        markets_cache["markets"] = markets
        print(f"    got {len(markets)}")

    titles = {
        str(m.get("conditionId")): str(m.get("question") or "")
        for m in markets if m.get("conditionId")
    }

    # Implications are stable; discover once and reuse across price polls.
    rels = markets_cache.get("rels")
    if rels is None:
        rels = discover(markets, args)
        markets_cache["rels"] = rels
        if rels and not args.no_implication_alerts:
            deliver([fmt_implication(r, titles) for r in rels[:args.max_alerts]],
                    args.dry_run)

    sigs = check_prices(rels, markets, args)
    if sigs:
        deliver([fmt_arbitrage(s) for s in sigs[:args.max_alerts]], args.dry_run)
    else:
        print("  no price violation this pass.")
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
