"""
tests/test_config.py
────────────────────
BotConfig.from_env — reads + validates all env-driven settings, fail-fast on
any invalid value (Phase 5).
"""
from __future__ import annotations

import pytest

from config import BotConfig


def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_defaults_are_valid(monkeypatch):
    _clear(monkeypatch, "DESIRED_NET_MARGIN", "DEFAULT_TAKER_FEE", "MAX_FEEDS",
           "STARTING_BALANCE", "SCAN_INTERVAL", "EXTREME_PRICE_LO",
           "EXTREME_PRICE_HI", "MIN_REAL_EDGE", "MAX_TICK_AGE_S",
           "FEED_PRUNE_IDLE_S", "MIN_VOLUME_24H")
    cfg = BotConfig.from_env()
    assert 0.0 < cfg.desired_net_margin < 1.0
    assert cfg.extreme_lo < cfg.extreme_hi
    assert cfg.max_feeds >= 0
    assert cfg.starting_balance > 0


def test_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("DESIRED_NET_MARGIN", "0.012")
    monkeypatch.setenv("MAX_FEEDS", "80")
    monkeypatch.setenv("MIN_VOLUME_24H", "5000")
    cfg = BotConfig.from_env()
    assert cfg.desired_net_margin == 0.012
    assert cfg.max_feeds == 80
    assert cfg.min_volume_24h == 5000.0


@pytest.mark.parametrize("name,value", [
    ("DESIRED_NET_MARGIN", "1.5"),     # > 1
    ("DESIRED_NET_MARGIN", "0"),       # not > 0
    ("DEFAULT_TAKER_FEE", "0.5"),      # > MAX_TAKER_FEE
    ("MIN_REAL_EDGE", "-0.1"),         # < 0
    ("STARTING_BALANCE", "0"),         # not > 0
    ("MAX_FEEDS", "-1"),               # < 0
    ("SCAN_INTERVAL", "0"),            # not > 0
])
def test_invalid_values_fail_fast(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        BotConfig.from_env()


def test_invalid_extreme_band_fails(monkeypatch):
    monkeypatch.setenv("EXTREME_PRICE_LO", "0.9")
    monkeypatch.setenv("EXTREME_PRICE_HI", "0.1")   # lo >= hi
    with pytest.raises(ValueError):
        BotConfig.from_env()


def test_non_numeric_fails(monkeypatch):
    monkeypatch.setenv("MAX_FEEDS", "lots")
    with pytest.raises(ValueError):
        BotConfig.from_env()


def test_negrisk_exec_mode_defaults_off(monkeypatch):
    # conftest sets NEGRISK_EXEC_MODE=onchain for the integration suite;
    # remove it to observe the true production default.
    _clear(monkeypatch, "NEGRISK_EXEC_MODE")
    assert BotConfig.from_env().negrisk_exec_mode == "off"


def test_negrisk_exec_mode_accepts_onchain(monkeypatch):
    monkeypatch.setenv("NEGRISK_EXEC_MODE", "  OnChain ")
    assert BotConfig.from_env().negrisk_exec_mode == "onchain"


def test_negrisk_exec_mode_rejects_unknown(monkeypatch):
    monkeypatch.setenv("NEGRISK_EXEC_MODE", "clob")
    with pytest.raises(ValueError):
        BotConfig.from_env()
