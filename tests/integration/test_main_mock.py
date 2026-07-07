"""Mirror of main.py — same startup sequence, same 8 tasks, same gather/halt,
but every external dependency replaced by a fake.

Structure mirrors main() precisely:
  1.  Build components (PolyClient, CircuitBreaker, FakeTelegramNotifier, …)
  2.  Initialise SignalLogger
  3.  Create FeedRegistry + market_queue
  4.  Seed feed from fake scanner
  5.  Build MarketScanner with mocked _fetch_all_active
  6.  Build AutoRedeemer (patched Web3)
  7.  Wire _do_halt closure (same logic as main._do_halt)
  8.  asyncio.gather all 8 tasks
  9.  Assert post-conditions, then trigger /halt
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import aiosqlite
import pytest

# Import the real coroutines from main.py — we run them unchanged.
from main import (
    auto_redeem_loop,
    heartbeat_loop,
    scanner_loop,
    strategy_loop,
    telegram_loop,
)

from core.clob_client      import BundleLeg, PolyClient
from core.scanner          import FeedRegistry, MarketScanner
from execution.auto_redeem import AutoRedeemer
from risk.circuit_breaker  import (
    ArbOrderIntent, CircuitBreaker, CircuitBreakerTripped, MAX_ARB_PAIR_USDC,
)
from strategy.arbitrage    import (
    DEFAULT_TAKER_FEE, DESIRED_NET_MARGIN,
    ArbDetector, DutchBookPricer, FeeEngine, MakerRebateEngine, NegRiskArbDetector,
)
from strategy.tuner        import tuner_loop
from telemetry.db_logger   import SignalLogger
from telemetry.metrics     import metrics_server
from tests.mocks.fake_clob_ws  import FakeAiohttpSession, FakeWebSocket, make_book_message
from tests.mocks.fake_gamma    import FakeGamma, make_market
from tests.mocks.fake_telegram import FakeTelegramNotifier

logger = logging.getLogger("test.main_mock")


# ── shared fake markets ──────────────────────────────────────────────────────

CID     = "0x" + "ab" * 32
YES_ID  = "YES_TOK"
NO_ID   = "NO_TOK"

# yes_ask + no_ask = 0.97  →  net_edge=0.03, well above 0.5% margin threshold
YES_ASK = 0.47
NO_ASK  = 0.50


# ── pytest fixture: build the whole component tree ───────────────────────────

@pytest.fixture
def components(monkeypatch, patched_clob, patched_web3, tmp_path):
    """Initialise every component exactly as main() does, with fakes substituted."""

    # ── daily state isolation ────────────────────────────────────────────────
    import risk.circuit_breaker as cb_mod
    monkeypatch.setattr(cb_mod, "_DAILY_STATE_PATH", str(tmp_path / "daily.json"))

    # ── fake WS feed: YES book → NO book → hold ──────────────────────────────
    ws = FakeWebSocket([
        make_book_message(YES_ID, YES_ASK),
        make_book_message(NO_ID,  NO_ASK),
    ])

    # ── unified session: both ws_connect (MarketFeed) and get (AutoRedeemer) ─
    # Both modules import aiohttp globally so they share the same object;
    # a single patch must handle all call sites.
    class _UnifiedFakeSession:
        """Supports ws_connect (ws_feed) + GET (auto_redeem scanner)."""

        def __init__(self, *_a, **_kw) -> None: pass

        async def __aenter__(self): return self
        async def __aexit__(self, *_e): return None
        async def close(self): pass

        def ws_connect(self, *_a, **_kw):
            return FakeAiohttpSession(ws).ws_connect()

        def get(self, *_a, **_kw):
            class _Resp:
                status = 200
                async def __aenter__(self): return self
                async def __aexit__(self, *_e): return None
                def raise_for_status(self): pass
                async def json(self, **_kw): return []
            return _Resp()

    import aiohttp as _aiohttp
    monkeypatch.setattr(_aiohttp, "ClientSession", _UnifiedFakeSession)

    # ── fake scanner: one arb market returned ───────────────────────────────
    gamma = FakeGamma()
    gamma.add(make_market(CID, YES_ID, NO_ID, volume_24h=100_000))

    async def _fake_fetch(self):          # noqa: ANN001
        return gamma.get_all()

    monkeypatch.setattr("core.scanner.MarketScanner._fetch_all_active", _fake_fetch, raising=True)

    # ── step 1: components (same order as main()) ────────────────────────────
    client         = PolyClient()
    breaker        = CircuitBreaker(starting_balance=500.0)
    notifier       = FakeTelegramNotifier(on_status=breaker.status_dict)
    fee_engine     = FeeEngine(default_fee=0.0)
    fee_engine.prime_cache(CID, 0.0)
    rebate_engine  = MakerRebateEngine()
    rebate_engine.prime_cache(CID, rebate_rate=0.0)
    detector       = ArbDetector(desired_net_margin=0.005, default_fee_rate=0.0)
    dutch_pricer   = DutchBookPricer(desired_net_margin=0.005)
    neg_risk_det   = NegRiskArbDetector(desired_net_margin=0.005)

    # ── step 2: signal logger ────────────────────────────────────────────────
    db_path  = str(tmp_path / "signals.db")
    sig_log  = SignalLogger(db_path=db_path)

    # ── step 3: queue + registry ─────────────────────────────────────────────
    market_queue  = asyncio.Queue(maxsize=2048)
    feed_registry = FeedRegistry(queue=market_queue)

    # ── step 5: scanner ──────────────────────────────────────────────────────
    scanner = MarketScanner(
        on_market_added=feed_registry.add_market,
        scan_interval=0.05,
        max_feeds=10,
    )

    # ── step 6: auto redeemer ────────────────────────────────────────────────
    redeemer = AutoRedeemer(feed_registry=feed_registry, notifier=notifier)

    return dict(
        client=client, breaker=breaker, notifier=notifier,
        fee_engine=fee_engine, detector=detector,
        rebate_engine=rebate_engine, dutch_pricer=dutch_pricer,
        neg_risk_det=neg_risk_det,
        sig_log=sig_log, market_queue=market_queue,
        feed_registry=feed_registry, scanner=scanner,
        redeemer=redeemer, db_path=db_path,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main test
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_main_mock_full_pipeline(components):
    """
    Runs the exact same 8 tasks as main() under asyncio.gather.
    Asserts the full arb happy-path, then sends /halt and verifies clean shutdown.
    """
    c = components

    await c["sig_log"].init()

    # ── _do_halt closure — identical logic to main._do_halt ──────────────────
    _halt_tasks: list[asyncio.Task] = []

    async def _do_halt() -> None:
        logger.info("_do_halt triggered")
        await c["notifier"].send_critical_error(
            "MANUAL HALT — /halt command received via Telegram\n"
            "Cancelling all open orders..."
        )
        try:
            await c["client"].cancel_all_orders()
            await c["notifier"].notify("All orders cancelled. Bot halted.")
        except Exception as exc:           # noqa: BLE001
            await c["notifier"].notify(f"WARNING: cancel_all_orders failed: {exc}")
        for t in _halt_tasks:
            t.cancel()
        await c["feed_registry"].stop_all()

    c["notifier"].set_halt_callback(_do_halt)

    # ── launch all 8 tasks (same list as main()) ─────────────────────────────
    tasks = [
        asyncio.create_task(scanner_loop(c["scanner"]),                         name="scanner"),
        asyncio.create_task(
            strategy_loop(
                c["market_queue"], c["detector"], c["dutch_pricer"],
                c["neg_risk_det"], c["rebate_engine"],
                c["fee_engine"], c["breaker"],
                c["client"], c["notifier"], c["sig_log"],
            ),
            name="strategy",
        ),
        asyncio.create_task(auto_redeem_loop(c["redeemer"]),                    name="auto_redeem"),
        asyncio.create_task(
            heartbeat_loop(c["breaker"], c["notifier"], c["feed_registry"]),    name="heartbeat",
        ),
        asyncio.create_task(telegram_loop(c["notifier"]),                       name="telegram"),
        asyncio.create_task(c["sig_log"].run(),                                 name="sig_logger"),
        asyncio.create_task(metrics_server(),                                   name="metrics"),
        asyncio.create_task(tuner_loop(c["detector"], db_path=c["db_path"], interval=9999.0), name="tuner"),
    ]
    _halt_tasks.extend(tasks)

    # ── wait for strategy_loop to process one arb trade ──────────────────────
    # Poll breaker until one order passes, or give up after 3 s.
    async def _wait_for_trade(timeout: float = 3.0) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if c["breaker"].status_dict()["orders_passed"] >= 1:
                return True
            await asyncio.sleep(0.05)
        return False

    trade_seen = await _wait_for_trade()

    # ── trigger /halt (same path as TelegramNotifier's /halt command) ────────
    await c["notifier"].trigger_halt()

    # wait for all tasks to finish (they were cancelled by _do_halt)
    await asyncio.gather(*tasks, return_exceptions=True)
    await c["feed_registry"].stop_all()

    # ── assert: arb was traded ───────────────────────────────────────────────
    assert trade_seen, "strategy_loop never processed an arb opportunity within 3 s"
    assert c["breaker"].status_dict()["orders_passed"] == 1
    assert c["breaker"]._state.session_pnl > 0.0

    # ── assert: Telegram notified about the trade ────────────────────────────
    assert len(c["notifier"].trade_executions) == 1
    trade = c["notifier"].trade_executions[0]
    assert trade["yes_ask"] == pytest.approx(YES_ASK)
    assert trade["no_ask"]  == pytest.approx(NO_ASK)

    # ── assert: startup notification was sent ────────────────────────────────
    # (main() calls notifier.notify(…) right after wiring — we replicate below)
    # The test does NOT call notifier.notify() itself, so messages list reflects
    # only what the coroutines sent.  /halt itself sends at least one message.
    assert any("HALT" in m or "cancelled" in m.lower() for m in c["notifier"].messages)
    assert len(c["notifier"].critical_errors) == 1

    # ── assert: signal was persisted to SQLite ───────────────────────────────
    await c["sig_log"].close()
    async with aiosqlite.connect(c["db_path"]) as db:
        cur = await db.execute("SELECT COUNT(*) FROM arb_signals")
        (n_rows,) = await cur.fetchone()
    assert n_rows == 1


# ════════════════════════════════════════════════════════════════════════════
# circuit_breaker trip → emergency halt path
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_main_mock_circuit_breaker_tripped(components, monkeypatch):
    """
    Pre-trip the daily loss limit so strategy_loop raises CircuitBreakerTripped.
    The gather() block should catch it, send critical_error, cancel all tasks,
    and exit cleanly — mirroring the except CircuitBreakerTripped branch in main().
    """
    import risk.circuit_breaker as cb_mod

    c = components
    await c["sig_log"].init()

    # Force the daily loss limit to already be breached.
    c["breaker"]._daily.daily_pnl = cb_mod.DAILY_LOSS_LIMIT - 0.01

    _halt_tasks: list[asyncio.Task] = []

    async def _do_halt() -> None:
        for t in _halt_tasks:
            t.cancel()
        await c["feed_registry"].stop_all()

    c["notifier"].set_halt_callback(_do_halt)

    tasks = [
        asyncio.create_task(scanner_loop(c["scanner"]),                         name="scanner"),
        asyncio.create_task(
            strategy_loop(
                c["market_queue"], c["detector"], c["dutch_pricer"],
                c["neg_risk_det"], c["rebate_engine"],
                c["fee_engine"], c["breaker"],
                c["client"], c["notifier"], c["sig_log"],
            ),
            name="strategy",
        ),
        asyncio.create_task(auto_redeem_loop(c["redeemer"]),                    name="auto_redeem"),
        asyncio.create_task(
            heartbeat_loop(c["breaker"], c["notifier"], c["feed_registry"]),    name="heartbeat",
        ),
        asyncio.create_task(telegram_loop(c["notifier"]),                       name="telegram"),
        asyncio.create_task(c["sig_log"].run(),                                 name="sig_logger"),
        asyncio.create_task(metrics_server(),                                   name="metrics"),
        asyncio.create_task(tuner_loop(c["detector"], db_path=c["db_path"], interval=9999.0), name="tuner"),
    ]
    _halt_tasks.extend(tasks)

    # ── mirror main()'s gather + CircuitBreakerTripped handler ───────────────
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=3.0)
    except CircuitBreakerTripped:
        await c["notifier"].send_critical_error(
            "EMERGENCY HALT — daily loss limit breached"
        )
        await c["client"].cancel_all_orders()
        await c["notifier"].notify("All orders cancelled. Bot halted.")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await c["feed_registry"].stop_all()
        await c["sig_log"].close()

    assert len(c["notifier"].critical_errors) == 1
    assert "EMERGENCY HALT" in c["notifier"].critical_errors[0]
    assert any("cancelled" in m.lower() for m in c["notifier"].messages)


# ════════════════════════════════════════════════════════════════════════════
# DutchBookPricer maker path — taker arb is unavailable (margin too tight)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_main_mock_dutch_book_maker_path(monkeypatch, patched_clob, patched_web3, tmp_path):
    """
    When the taker arb margin is below the threshold but the maker (DutchBook)
    margin is above it, strategy_loop should execute via execute_arb_maker_pair.

    We raise the ArbDetector margin to 0.99 so no taker signal fires,
    but keep DutchBookPricer at 0.005 so the maker path triggers on YES_ASK=0.47/NO_ASK=0.50.
    """
    import risk.circuit_breaker as cb_mod
    monkeypatch.setattr(cb_mod, "_DAILY_STATE_PATH", str(tmp_path / "daily.json"))

    # Prices that produce maker edge but NOT taker edge (margin too tight for taker)
    MAKER_YES_ASK = 0.47
    MAKER_NO_ASK  = 0.50
    MKR_YES_ID    = "MAKER_YES"
    MKR_NO_ID     = "MAKER_NO"
    MKR_CID       = "0x" + "cd" * 32

    ws = FakeWebSocket([
        make_book_message(MKR_YES_ID, MAKER_YES_ASK),
        make_book_message(MKR_NO_ID,  MAKER_NO_ASK),
    ])

    class _Session:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_e): return None
        async def close(self): pass
        def ws_connect(self, *_a, **_kw):
            return FakeAiohttpSession(ws).ws_connect()
        def get(self, *_a, **_kw):
            class _R:
                status = 200
                async def __aenter__(self): return self
                async def __aexit__(self, *_e): return None
                def raise_for_status(self): pass
                async def json(self, **_kw): return []
            return _R()

    import aiohttp as _aiohttp
    monkeypatch.setattr(_aiohttp, "ClientSession", _Session)

    gamma = FakeGamma()
    gamma.add(make_market(MKR_CID, MKR_YES_ID, MKR_NO_ID, volume_24h=100_000))

    async def _fake_fetch(self):
        return gamma.get_all()

    monkeypatch.setattr("core.scanner.MarketScanner._fetch_all_active", _fake_fetch, raising=True)

    client         = PolyClient()
    breaker        = CircuitBreaker(starting_balance=500.0)
    notifier       = FakeTelegramNotifier(on_status=breaker.status_dict)
    fee_engine     = FeeEngine(default_fee=0.0)
    fee_engine.prime_cache(MKR_CID, 0.0)
    rebate_engine  = MakerRebateEngine()
    rebate_engine.prime_cache(MKR_CID, rebate_rate=0.0)
    # Taker detector: margin so high no taker signal fires
    detector       = ArbDetector(desired_net_margin=0.99, default_fee_rate=0.0)
    dutch_pricer   = DutchBookPricer(desired_net_margin=0.005)
    neg_risk_det   = NegRiskArbDetector(desired_net_margin=0.005)

    import risk.circuit_breaker as cb_mod2
    monkeypatch.setattr(cb_mod2, "_DAILY_STATE_PATH", str(tmp_path / "daily2.json"))

    db_path       = str(tmp_path / "signals.db")
    sig_log       = SignalLogger(db_path=db_path)
    await sig_log.init()

    market_queue  = asyncio.Queue(maxsize=2048)
    feed_registry = FeedRegistry(queue=market_queue)

    scanner = MarketScanner(
        on_market_added=feed_registry.add_market,
        scan_interval=0.05,
        max_feeds=10,
    )
    redeemer = AutoRedeemer(feed_registry=feed_registry, notifier=notifier)

    tasks: list[asyncio.Task] = []

    async def _do_halt():
        for t in tasks:
            t.cancel()
        await feed_registry.stop_all()

    notifier.set_halt_callback(_do_halt)

    tasks.extend([
        asyncio.create_task(scanner_loop(scanner),                       name="scanner"),
        asyncio.create_task(
            strategy_loop(
                market_queue, detector, dutch_pricer, neg_risk_det, rebate_engine,
                fee_engine, breaker, client, notifier, sig_log,
            ),
            name="strategy",
        ),
        asyncio.create_task(auto_redeem_loop(redeemer),                  name="auto_redeem"),
        asyncio.create_task(heartbeat_loop(breaker, notifier, feed_registry), name="heartbeat"),
        asyncio.create_task(telegram_loop(notifier),                     name="telegram"),
        asyncio.create_task(sig_log.run(),                               name="sig_logger"),
        asyncio.create_task(metrics_server(),                            name="metrics"),
        asyncio.create_task(tuner_loop(detector, db_path=db_path, interval=9999.0), name="tuner"),
    ])

    async def _wait_for_maker_trade(timeout: float = 3.0) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if breaker.status_dict()["orders_passed"] >= 1:
                return True
            await asyncio.sleep(0.05)
        return False

    trade_seen = await _wait_for_maker_trade()
    await notifier.trigger_halt()
    await asyncio.gather(*tasks, return_exceptions=True)
    await feed_registry.stop_all()
    await sig_log.close()

    assert trade_seen, "DutchBookPricer maker path never fired within 3 s"
    assert breaker.status_dict()["orders_passed"] == 1
    assert breaker._state.session_pnl > 0.0


# ════════════════════════════════════════════════════════════════════════════
# NegRisk multi-outcome path — direct queue injection
# ════════════════════════════════════════════════════════════════════════════

NR_CID       = "0x" + "ef" * 32
NR_TOKEN_A   = "NR_NO_A"
NR_TOKEN_B   = "NR_NO_B"
NR_TOKEN_C   = "NR_NO_C"
# 3-outcome NegRisk: combined_bid = 3 × (0.30 − 0.001) = 0.897
# payout = 2.0 → net_edge = 2.0 − 0.897 = 1.103 → relative_edge ≈ 0.55 >> 0.005
NR_NO_ASKS   = [0.30, 0.30, 0.30]


@pytest.mark.asyncio
async def test_main_mock_negrisk_path(monkeypatch, patched_clob, patched_web3, tmp_path):
    """
    Inject a neg_risk_tick directly into the market queue.
    Asserts NegRiskArbDetector fires, execute_arb_maker_bundle is called,
    circuit breaker registers the trade, and the notifier logs the outcome.
    """
    import risk.circuit_breaker as cb_mod
    monkeypatch.setattr(cb_mod, "_DAILY_STATE_PATH", str(tmp_path / "daily.json"))

    # No WS feed needed — inject the tick manually
    class _EmptySession:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_e): return None
        async def close(self): pass
        def ws_connect(self, *_a, **_kw):
            class _CM:
                async def __aenter__(self_): return _NeverWS()
                async def __aexit__(self_, *_e): return None
            return _CM()
        def get(self, *_a, **_kw):
            class _R:
                status = 200
                async def __aenter__(self): return self
                async def __aexit__(self, *_e): return None
                def raise_for_status(self): pass
                async def json(self, **_kw): return []
            return _R()

    class _NeverWS:
        def __aiter__(self): return self
        async def __anext__(self):
            await asyncio.sleep(9999)
            raise StopAsyncIteration

    import aiohttp as _aiohttp
    monkeypatch.setattr(_aiohttp, "ClientSession", _EmptySession)

    # Scanner returns no markets (we inject the tick ourselves)
    async def _empty_fetch(self):
        return []

    monkeypatch.setattr(
        "core.scanner.MarketScanner._fetch_all_active",
        _empty_fetch,
        raising=True,
    )

    client         = PolyClient()
    breaker        = CircuitBreaker(starting_balance=500.0)
    notifier       = FakeTelegramNotifier(on_status=breaker.status_dict)
    fee_engine     = FeeEngine(default_fee=0.0)
    rebate_engine  = MakerRebateEngine()
    rebate_engine.prime_cache(NR_CID, rebate_rate=0.0)
    detector       = ArbDetector(desired_net_margin=0.005, default_fee_rate=0.0)
    dutch_pricer   = DutchBookPricer(desired_net_margin=0.005)
    neg_risk_det   = NegRiskArbDetector(desired_net_margin=0.005)

    db_path        = str(tmp_path / "signals.db")
    sig_log        = SignalLogger(db_path=db_path)
    await sig_log.init()

    market_queue   = asyncio.Queue(maxsize=2048)
    feed_registry  = FeedRegistry(queue=market_queue)

    scanner = MarketScanner(
        on_market_added=feed_registry.add_market,
        scan_interval=9999.0,
        max_feeds=10,
    )
    redeemer = AutoRedeemer(feed_registry=feed_registry, notifier=notifier)

    tasks: list[asyncio.Task] = []

    async def _do_halt():
        for t in tasks:
            t.cancel()
        await feed_registry.stop_all()

    notifier.set_halt_callback(_do_halt)

    tasks.extend([
        asyncio.create_task(scanner_loop(scanner),                       name="scanner"),
        asyncio.create_task(
            strategy_loop(
                market_queue, detector, dutch_pricer, neg_risk_det, rebate_engine,
                fee_engine, breaker, client, notifier, sig_log,
            ),
            name="strategy",
        ),
        asyncio.create_task(auto_redeem_loop(redeemer),                  name="auto_redeem"),
        asyncio.create_task(heartbeat_loop(breaker, notifier, feed_registry), name="heartbeat"),
        asyncio.create_task(telegram_loop(notifier),                     name="telegram"),
        asyncio.create_task(sig_log.run(),                               name="sig_logger"),
        asyncio.create_task(metrics_server(),                            name="metrics"),
        asyncio.create_task(tuner_loop(detector, db_path=db_path, interval=9999.0), name="tuner"),
    ])

    # Inject a NegRisk tick directly into the queue
    import time
    await market_queue.put({
        "type":              "neg_risk_tick",
        "condition_id":      NR_CID,
        "outcome_token_ids": [NR_TOKEN_A, NR_TOKEN_B, NR_TOKEN_C],
        "no_asks":           NR_NO_ASKS,
        "ts":                asyncio.get_event_loop().time(),
    })

    async def _wait_for_trade(timeout: float = 3.0) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if breaker.status_dict()["orders_passed"] >= 1:
                return True
            await asyncio.sleep(0.05)
        return False

    trade_seen = await _wait_for_trade()
    await notifier.trigger_halt()
    await asyncio.gather(*tasks, return_exceptions=True)
    await feed_registry.stop_all()
    await sig_log.close()

    assert trade_seen, "NegRiskArbDetector never fired within 3 s"
    assert breaker.status_dict()["orders_passed"] == 1
    assert breaker._state.session_pnl > 0.0
    assert any("NEG RISK" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_main_mock_negrisk_exec_off_alerts_without_trading(monkeypatch, tmp_path):
    """
    NEGRISK_EXEC_MODE=off (the live default): a NegRisk signal must be
    detected and alerted, but NOTHING may be submitted — matchOrders would
    revert for a non-operator wallet.
    """
    import risk.circuit_breaker as cb_mod
    monkeypatch.setattr(cb_mod, "_DAILY_STATE_PATH", str(tmp_path / "daily.json"))
    monkeypatch.setattr("main._negrisk_exec_mode", "off", raising=True)

    breaker       = CircuitBreaker(starting_balance=500.0)
    notifier      = FakeTelegramNotifier(on_status=breaker.status_dict)
    fee_engine    = FeeEngine(default_fee=0.0)
    rebate_engine = MakerRebateEngine()
    rebate_engine.prime_cache(NR_CID, rebate_rate=0.0)

    class _NoTradeClient:
        """Any order/tx attempt fails the test."""
        def __getattr__(self, name):
            raise AssertionError(
                f"client.{name} must not be touched when NegRisk exec is off"
            )

    class _NoopSigLogger:
        def log_arb(self, *_a, **_kw):
            pass

    queue: asyncio.Queue = asyncio.Queue()
    strat = asyncio.create_task(strategy_loop(
        queue,
        ArbDetector(desired_net_margin=0.005, default_fee_rate=0.0),
        DutchBookPricer(desired_net_margin=0.005),
        NegRiskArbDetector(desired_net_margin=0.005),
        rebate_engine, fee_engine, breaker,
        _NoTradeClient(), notifier, _NoopSigLogger(),
    ))

    await queue.put({
        "type":              "neg_risk_tick",
        "condition_id":      NR_CID,
        "outcome_token_ids": [NR_TOKEN_A, NR_TOKEN_B, NR_TOKEN_C],
        "no_asks":           NR_NO_ASKS,
        "ts":                asyncio.get_event_loop().time(),
    })
    await asyncio.sleep(0.3)

    strat.cancel()
    await asyncio.gather(strat, return_exceptions=True)

    detections = getattr(notifier, "arb_detections", [])
    assert detections, "NegRisk signal was not alerted in off mode"
    assert detections[0]["condition_id"] == NR_CID
    assert "negrisk" in detections[0].get("category", "")
    status = breaker.status_dict()
    assert status["orders_passed"] == 0, "no order may pass the breaker in off mode"
    assert status["open_positions"] == 0
    assert status["session_pnl"] == 0.0
