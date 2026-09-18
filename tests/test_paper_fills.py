"""The paper-fill replay: which trades would have met our bids, and what it made."""

from __future__ import annotations

import json

import scripts.paper_fills as pf

T0 = 1_000_000.0
REST = {"ts": T0, "narrow": "N", "broad": "B", "narrow_title": "n", "broad_title": "b",
        "narrow_yes_token": "ny", "narrow_no_token": "nn",
        "broad_yes_token": "by", "broad_no_token": "bn",
        "no_price": 0.94, "yes_price": 0.02, "shares": 5.0, "entry": 0.96, "edge": 0.04,
        "resolves_ts": T0 + 86_400}


def _t(ts, asset, price, size):
    return {"timestamp": ts, "asset": asset, "price": price, "size": size}


def test_a_yes_buy_at_our_implied_offer_fills_the_no_bid():
    """NO bid 0.94 is a YES offer at 0.06: a YES trade at 0.06 or dearer met it."""
    hits = lambda t: pf.yes_price(t, "ny") >= 1 - 0.94 - 1e-9
    trades = [_t(T0 + 10, "ny", 0.05, 100), _t(T0 + 20, "ny", 0.06, 3), _t(T0 + 30, "nn", 0.93, 4)]
    assert pf.filled(trades, T0, T0 + 60, 5.0, hits) == 5.0       # 3 at 0.06, then NO sold at 0.93
    assert pf.filled(trades, T0, T0 + 25, 5.0, hits) == 3.0       # partial inside the window


def test_trades_before_the_rest_do_not_count():
    hits = lambda t: True
    assert pf.filled([_t(T0 - 1, "ny", 0.5, 99)], T0, T0 + 60, 5.0, hits) == 0.0


def _run(tmp_path, monkeypatch, trades, resolved):
    p = tmp_path / "rests.jsonl"
    rests = [REST, {**REST, "ts": T0 + 900}]                       # re-rested 15 min later
    p.write_text("\n".join(json.dumps(r) for r in rests))
    monkeypatch.setattr(pf, "trades_since", lambda cid, since: trades.get(cid, []))
    monkeypatch.setattr(pf, "resolutions", lambda cids: resolved)
    monkeypatch.setattr(pf.time, "time", lambda: T0 + 200_000)
    return pf.main(type("A", (), {"rests": str(p), "show": 5})())


def test_both_legs_filled_pay_the_locked_edge_and_block_the_re_rest(tmp_path, monkeypatch, capsys):
    trades = {"N": [_t(T0 + 5, "ny", 0.07, 10)], "B": [_t(T0 + 50, "by", 0.02, 10)]}
    # nine goals did not happen, eight did: NO on 8.5 pays, YES on 7.5 pays
    _run(tmp_path, monkeypatch, trades, {"N": (0.0, 1.0), "B": (1.0, 0.0)})
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("── rest 3 min"))
    assert "1 rest(s) — both legs 1" in line                      # the re-rest was not counted
    # 5 × ((1 − 0) − 0.94) + 5 × (1 − 0.02) = 0.30 + 4.90
    assert "realised +5.20 USDC" in out.split("── rest 3 min")[1].splitlines()[1]


def test_a_lone_leg_is_held_to_resolution(tmp_path, monkeypatch, capsys):
    trades = {"N": [_t(T0 + 5, "ny", 0.07, 10)], "B": []}
    _run(tmp_path, monkeypatch, trades, {"N": (1.0, 0.0), "B": (1.0, 0.0)})   # nine goals
    out = capsys.readouterr().out
    section = out.split("── rest 3 min")[1]
    assert "narrow only 1" in section.splitlines()[0]
    assert "realised -4.70 USDC" in section.splitlines()[1]       # 5 × (0 − 0.94)


def test_an_unresolved_fill_is_left_open(tmp_path, monkeypatch, capsys):
    trades = {"N": [_t(T0 + 5, "ny", 0.07, 10)], "B": [_t(T0 + 50, "by", 0.02, 10)]}
    _run(tmp_path, monkeypatch, trades, {})
    assert "1 filled rest(s) still waiting" in capsys.readouterr().out


def test_a_rest_in_a_one_sided_book_is_not_replayed(tmp_path):
    p = tmp_path / "rests.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (
        {**REST, "no_ask": None}, {**REST, "ts": T0 + 1, "no_ask": [0.98, 5]})))
    assert [r["ts"] for r in pf.load_rests(p)] == [T0 + 1]
