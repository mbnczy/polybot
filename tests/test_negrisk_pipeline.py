"""
Tests for the NegRisk multi-outcome pipeline.

Covers the four stages that turn a NegRisk group into a guarded bundle:

  scanner  → group discovery + volume-ranked outcome reduction
  ws_feed  → N-leg group state, depth capture, one-to-many asset routing
  strategy → NegRiskArbDetector selection heuristics (arXiv:2508.03474)
  execution→ CLOB bundle submission + NegRiskBundleGuard partial-fill handling

The detector cases are written against the paper's stated parameters so a
regression in the heuristics fails loudly rather than silently widening risk.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.ws_feed import _MarketState, _NegRiskGroupState, MarketShard
from strategy.arbitrage import (
    NEGRISK_MAX_LEGS,
    NEGRISK_MIN_OUTCOME_PROB,
    NEGRISK_MIN_RELATIVE_EDGE,
    NegRiskArbDetector,
)


def _det(**kw) -> NegRiskArbDetector:
    """Detector with the paper defaults unless a case overrides them."""
    params = dict(desired_net_margin=0.005, min_leg_shares=0.0)
    params.update(kw)
    return NegRiskArbDetector(**params)


# ═══════════════════════════════════════════════════════════════════════════
# Detector — selection heuristics
# ═══════════════════════════════════════════════════════════════════════════

class TestNegRiskSelection:

    def test_low_probability_outcomes_are_dropped(self):
        """
        Paper §6.2 sizes from conditions with probability > 2 %.

        Three real contenders plus one 1 %-probability tail outcome. The tail
        leg costs ~$0.99 of capital and adds ~$0.01 of edge, so it must not
        appear in the bundle.
        """
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp",
            ["A", "B", "C", "TAIL"],
            [0.58, 0.63, 0.68, 0.99],   # implied YES: .42 .37 .32 .01
            maker_rebate=0.0,
            tick_size=0.01,
        )
        assert sig is not None
        assert sig.n_outcomes == 3
        assert {leg.token_id for leg in sig.legs} == {"A", "B", "C"}
        # payout follows the SELECTED subset, not the group size.
        assert sig.payout == 2.0

    def test_leg_count_capped_at_top_k(self):
        """Paper §5.1 — keep the top-4; ranking is by implied probability."""
        d = _det()
        # Six live outcomes, all above the probability floor.
        sig = d.evaluate_neg_risk(
            "grp",
            ["p10", "p32", "p05", "p22", "p26", "p40"],
            [0.90, 0.68, 0.95, 0.78, 0.74, 0.60],
            maker_rebate=0.0,
            tick_size=0.01,
        )
        assert sig is not None
        assert sig.n_outcomes == NEGRISK_MAX_LEGS == 4
        # The four highest-probability outcomes, not the first four given.
        assert [leg.token_id for leg in sig.legs] == ["p40", "p32", "p26", "p22"]

    def test_max_legs_is_configurable(self):
        d = _det(max_legs=2)
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.35, 0.40],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is not None and sig.n_outcomes == 2
        assert [leg.token_id for leg in sig.legs] == ["A", "B"]

    def test_near_resolved_group_is_skipped(self):
        """
        Paper §6 only measures while no position is worth more than $0.95.
        A NO ask of 0.02 means that outcome's YES trades at 0.98 — decided.
        """
        d = _det()
        assert d.evaluate_neg_risk(
            "grp", ["WIN", "LOSE"], [0.02, 0.985],
            maker_rebate=0.0, tick_size=0.01,
        ) is None

    def test_unusable_quote_drops_only_its_leg(self):
        """A malformed ask costs that outcome's edge, not the whole group."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "BAD"], [0.30, 0.35, 0.0],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is not None
        # A (implied .70) outranks B (implied .65); BAD never enters.
        assert [leg.token_id for leg in sig.legs] == ["A", "B"]

    def test_needs_at_least_two_surviving_legs(self):
        d = _det()
        assert d.evaluate_neg_risk(
            "grp", ["A", "B"], [0.60, 0.0],
            maker_rebate=0.0, tick_size=0.01,
        ) is None


class TestNegRiskEdgeGates:

    def test_relative_edge_floor_rejects_thin_opportunity(self):
        """
        Paper §6 restricts to ≥ $0.05 on the dollar. Construct a bundle that
        clears the 0.5 % binary margin but sits under the 5 % NegRisk floor.
        """
        # 3 legs, bids 0.49 each → combined 1.47, payout 2.0
        # relative_edge = (2.0 - 1.47) / 2.0 = 0.265 → too generous; tighten:
        # bids 0.63 each → combined 1.89 → (2.0-1.89)/2 = 0.055  (passes)
        # bids 0.65 each → combined 1.95 → (2.0-1.95)/2 = 0.025  (fails)
        d = _det()
        passing = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.64, 0.64, 0.64],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert passing is not None
        assert passing.relative_edge >= NEGRISK_MIN_RELATIVE_EDGE

        failing = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.66, 0.66, 0.66],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert failing is None

    def test_floor_is_configurable_off(self):
        """min_relative_edge=0 restores pure DESIRED_NET_MARGIN behaviour."""
        d = _det(min_relative_edge=0.0)
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.66, 0.66, 0.66],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is not None
        assert 0.005 < sig.relative_edge < NEGRISK_MIN_RELATIVE_EDGE

    def test_payout_matches_selected_subset_not_group_size(self):
        """
        The subset lemma: M legs guarantee M−1, whatever the group's true N.
        Booking N−1 on an M-leg bundle would overstate profit.
        """
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp",
            ["A", "B", "C", "D", "E", "F"],
            [0.60, 0.64, 0.68, 0.72, 0.99, 0.99],
            maker_rebate=0.0, tick_size=0.01,
        )
        assert sig is not None
        assert sig.n_outcomes == 4
        assert sig.payout == 3.0                       # M − 1, not 6 − 1
        assert sig.net_edge == pytest.approx(
            sig.payout - sig.combined_bid, abs=1e-6
        )


class TestNegRiskSizing:

    def test_bundle_capped_by_shallowest_leg(self):
        """Paper §6.2 — minimum volume across the selected conditions."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.30, 0.30],
            max_position_usdc=1_000.0,       # capital is not the binding limit
            maker_rebate=0.0, tick_size=0.01,
            no_ask_sizes=[500.0, 40.0, 900.0],
        )
        assert sig is not None
        assert sig.n_bundles == 40.0
        assert all(leg.size == 40.0 for leg in sig.legs)

    def test_capital_ceiling_still_binds(self):
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.30, 0.30],
            max_position_usdc=20.0,
            maker_rebate=0.0, tick_size=0.01,
            no_ask_sizes=[9_000.0, 9_000.0, 9_000.0],
        )
        assert sig is not None
        # combined_bid = 3 × 0.29 = 0.87 → 20 / 0.87 = 22.98
        assert sig.n_bundles == pytest.approx(22.98, abs=0.01)
        assert sig.combined_bid * sig.n_bundles <= 20.0

    def test_depth_of_zero_drops_the_leg(self):
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.30, 0.30],
            maker_rebate=0.0, tick_size=0.01,
            no_ask_sizes=[100.0, 0.0, 100.0],
        )
        assert sig is not None
        assert [leg.token_id for leg in sig.legs] == ["A", "C"]

    def test_below_exchange_minimum_is_rejected(self):
        """Gamma reports orderMinSize=5; a 3-share bundle would be rejected."""
        d = NegRiskArbDetector(desired_net_margin=0.005)   # default min 5 shares
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.30, 0.30],
            max_position_usdc=1_000.0,
            maker_rebate=0.0, tick_size=0.01,
            no_ask_sizes=[3.0, 100.0, 100.0],
        )
        assert sig is None

    def test_legacy_call_without_depth_still_works(self):
        """no_ask_sizes=None keeps the pre-existing capital-only sizing."""
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.30, 0.30],
            max_position_usdc=50.0, maker_rebate=0.01, tick_size=0.01,
        )
        assert sig is not None
        assert sig.n_bundles == pytest.approx(57.47, abs=0.01)

    def test_mismatched_depth_length_is_rejected(self):
        d = _det()
        assert d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.30, 0.30],
            maker_rebate=0.0, tick_size=0.01,
            no_ask_sizes=[100.0, 100.0],
        ) is None


class TestNegRiskDetectorConstruction:

    @pytest.mark.parametrize("kw", [
        {"min_outcome_prob": 1.0},
        {"min_outcome_prob": -0.01},
        {"max_legs": 1},
        {"min_relative_edge": 1.0},
        {"min_leg_shares": -1.0},
    ])
    def test_invalid_params_rejected(self, kw):
        with pytest.raises(ValueError):
            NegRiskArbDetector(**kw)

    def test_defaults_match_published_values(self):
        assert NEGRISK_MIN_OUTCOME_PROB  == 0.02
        assert NEGRISK_MAX_LEGS          == 4
        assert NEGRISK_MIN_RELATIVE_EDGE == 0.05


# ═══════════════════════════════════════════════════════════════════════════
# ws_feed — NegRisk group state
# ═══════════════════════════════════════════════════════════════════════════

def _book(asset_id: str, price: float, size: float, tick: float = 0.01) -> dict:
    return {
        "event_type": "book",
        "asset_id":   asset_id,
        "tick_size":  str(tick),
        "asks":       [{"price": str(price), "size": str(size)}],
    }


class TestNegRiskGroupState:

    def test_no_tick_until_two_legs_quoted(self):
        st = _NegRiskGroupState("grp", ["A", "B", "C"])
        assert st.update_leg("A", _book("A", 0.30, 100))
        assert st.build_tick() is None

        assert st.update_leg("B", _book("B", 0.31, 200))
        tick = st.build_tick()
        assert tick is not None
        assert tick["type"] == "neg_risk_tick"
        assert tick["condition_id"] == "grp"
        assert tick["outcome_token_ids"] == ["A", "B"]
        assert tick["no_asks"] == [0.30, 0.31]
        assert tick["no_ask_sizes"] == [100.0, 200.0]
        assert tick["n_group_outcomes"] == 3

    def test_partial_group_is_emitted(self):
        """
        A group where only some legs are quoted still ticks: the detector's
        subset lemma makes any quoted subset a valid bundle candidate.
        """
        st = _NegRiskGroupState("grp", ["A", "B", "C", "D"])
        st.update_leg("A", _book("A", 0.30, 10))
        st.update_leg("D", _book("D", 0.40, 10))
        tick = st.build_tick()
        assert tick is not None
        assert tick["outcome_token_ids"] == ["A", "D"]
        assert tick["n_group_outcomes"] == 4

    def test_identical_snapshot_is_deduped(self):
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        st.update_leg("B", _book("B", 0.31, 200))
        assert st.build_tick() is not None
        assert st.build_tick() is None

    def test_depth_change_alone_produces_a_tick(self):
        """Bundle size depends on depth, so depth moves must not be deduped."""
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        st.update_leg("B", _book("B", 0.31, 200))
        st.build_tick()

        assert st.update_leg("A", _book("A", 0.30, 25))
        tick = st.build_tick()
        assert tick is not None
        assert tick["no_ask_sizes"] == [25.0, 200.0]

    def test_price_change_improves_best_ask(self):
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        st.update_leg("B", _book("B", 0.31, 100))
        st.build_tick()

        assert st.update_leg("A", {
            "event_type": "price_change", "asset_id": "A",
            "price": "0.28", "side": "SELL", "size": "50",
        })
        tick = st.build_tick()
        assert tick["no_asks"][0] == 0.28
        assert tick["no_ask_sizes"][0] == 50.0

    def test_worse_price_does_not_replace_best_ask(self):
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        assert not st.update_leg("A", {
            "event_type": "price_change", "asset_id": "A",
            "price": "0.35", "side": "SELL", "size": "50",
        })

    def test_pulling_the_best_level_invalidates_the_leg(self):
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        st.update_leg("B", _book("B", 0.31, 100))
        st.build_tick()

        assert st.update_leg("A", {
            "event_type": "price_change", "asset_id": "A",
            "price": "0.30", "side": "SELL", "size": "0",
        })
        # Only one leg quoted again → nothing actionable.
        assert st.build_tick() is None

    def test_pulling_a_deeper_level_is_ignored(self):
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        assert not st.update_leg("A", {
            "event_type": "price_change", "asset_id": "A",
            "price": "0.44", "side": "SELL", "size": "0",
        })

    def test_reset_clears_prices_on_reconnect(self):
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        st.update_leg("B", _book("B", 0.31, 100))
        assert st.build_tick() is not None
        st.reset()
        assert st.build_tick() is None

    def test_bids_only_event_is_ignored(self):
        st = _NegRiskGroupState("grp", ["A", "B"])
        assert not st.update_leg("A", {
            "event_type": "book", "asset_id": "A",
            "bids": [{"price": "0.29", "size": "10"}], "asks": [],
        })


class TestShardRouting:

    def test_one_token_feeds_both_binary_and_group(self):
        """
        A NegRisk member's NO token belongs to a binary market AND its group.
        Routing must be one-to-many or one detector goes blind.
        """
        queue: asyncio.Queue = asyncio.Queue()
        shard = MarketShard(queue)
        shard.add("cond-A", "YES_A", "NO_A")
        shard.add_neg_risk_group("grp", ["NO_A", "NO_B"])

        shard._dispatch(json.dumps([
            _book("YES_A", 0.60, 100),
            _book("NO_A",  0.30, 100),
            _book("NO_B",  0.31, 100),
        ]))

        ticks = []
        while not queue.empty():
            ticks.append(queue.get_nowait())
        kinds = {t["type"] for t in ticks}
        assert kinds == {"arb_tick", "neg_risk_tick"}

    def test_group_removal_unroutes_only_its_own_legs(self):
        queue: asyncio.Queue = asyncio.Queue()
        shard = MarketShard(queue)
        shard.add("cond-A", "YES_A", "NO_A")
        shard.add_neg_risk_group("grp", ["NO_A", "NO_B"])
        shard.remove("grp")

        shard._dispatch(json.dumps([
            _book("YES_A", 0.60, 100),
            _book("NO_A",  0.30, 100),
        ]))
        ticks = []
        while not queue.empty():
            ticks.append(queue.get_nowait())
        assert [t["type"] for t in ticks] == ["arb_tick"]

    def test_group_needs_two_outcomes(self):
        queue: asyncio.Queue = asyncio.Queue()
        shard = MarketShard(queue)
        shard.add_neg_risk_group("grp", ["NO_A"])
        assert shard.count == 0

    def test_binary_market_unaffected_by_group_state(self):
        """_MarketState still emits the legacy arb_tick shape."""
        st = _MarketState("cond", "YES", "NO")
        st.update_leg("YES", _book("YES", 0.60, 10))
        st.update_leg("NO",  _book("NO",  0.39, 10))
        tick = st.build_tick()
        assert tick["type"] == "arb_tick"
        assert tick["yes_ask"] == 0.60 and tick["no_ask"] == 0.39


# ═══════════════════════════════════════════════════════════════════════════
# Scanner — NegRisk group discovery
# ═══════════════════════════════════════════════════════════════════════════

def _gamma_market(
    group: str, qid: str, *, vol: float, neg_risk: bool = True,
) -> dict:
    """Minimal Gamma /markets row shaped like the live API."""
    return {
        "conditionId":      f"0xcond{qid}",
        "question":         f"Will {qid} win?",
        "negRisk":          neg_risk,
        "negRiskMarketID":  group,
        "clobTokenIds":     json.dumps([f"YES_{qid}", f"NO_{qid}"]),
        "volume24hr":       vol,
        "endDate":          "2099-01-01T00:00:00Z",
        "outcomes":         json.dumps(["Yes", "No"]),
    }


class TestScannerGroupDiscovery:

    @pytest.mark.asyncio
    async def test_groups_by_negrisk_market_id(self):
        from core.scanner import MarketScanner

        seen: list[tuple[str, list[str]]] = []

        async def _capture(group_id: str, no_ids: list[str]) -> bool:
            seen.append((group_id, no_ids))
            return True

        scanner = MarketScanner(
            on_market_added=_noop_market,
            on_neg_risk_group=_capture,
        )
        await scanner._register_neg_risk_groups([
            _gamma_market("G1", "a", vol=10),
            _gamma_market("G1", "b", vol=20),
            _gamma_market("G2", "c", vol=30),
            _gamma_market("G2", "d", vol=40),
            _gamma_market("none", "e", vol=50, neg_risk=False),
        ])

        assert {g for g, _ in seen} == {"G1", "G2"}
        assert dict(seen)["G1"] == ["NO_b", "NO_a"]   # volume-ranked

    @pytest.mark.asyncio
    async def test_outcomes_capped_and_volume_ranked(self):
        """Paper §5.1 reduction — keep the highest-volume slice of a big group."""
        from core.scanner import MarketScanner

        seen: dict[str, list[str]] = {}

        async def _capture(group_id: str, no_ids: list[str]) -> bool:
            seen[group_id] = no_ids
            return True

        scanner = MarketScanner(
            on_market_added=_noop_market,
            on_neg_risk_group=_capture,
            negrisk_feed_outcomes=3,
        )
        await scanner._register_neg_risk_groups([
            _gamma_market("BIG", f"m{i}", vol=float(i)) for i in range(10)
        ])
        assert seen["BIG"] == ["NO_m9", "NO_m8", "NO_m7"]

    @pytest.mark.asyncio
    async def test_group_registered_once(self):
        from core.scanner import MarketScanner

        calls: list[str] = []

        async def _capture(group_id: str, no_ids: list[str]) -> bool:
            calls.append(group_id)
            return True

        scanner = MarketScanner(
            on_market_added=_noop_market, on_neg_risk_group=_capture,
        )
        markets = [_gamma_market("G1", "a", vol=1), _gamma_market("G1", "b", vol=2)]
        await scanner._register_neg_risk_groups(markets)
        await scanner._register_neg_risk_groups(markets)
        assert calls == ["G1"]

    @pytest.mark.asyncio
    async def test_rejected_group_stays_eligible(self):
        """A cap rejection must not mark the group known."""
        from core.scanner import MarketScanner

        calls: list[str] = []

        async def _reject(group_id: str, no_ids: list[str]) -> bool:
            calls.append(group_id)
            return False

        scanner = MarketScanner(
            on_market_added=_noop_market, on_neg_risk_group=_reject,
        )
        markets = [_gamma_market("G1", "a", vol=1), _gamma_market("G1", "b", vol=2)]
        await scanner._register_neg_risk_groups(markets)
        await scanner._register_neg_risk_groups(markets)
        assert calls == ["G1", "G1"]

    @pytest.mark.asyncio
    async def test_discovery_disabled_when_callback_absent(self):
        from core.scanner import MarketScanner

        scanner = MarketScanner(on_market_added=_noop_market)
        # Must not raise, must not track anything.
        await scanner._register_neg_risk_groups([_gamma_market("G1", "a", vol=1)])
        assert scanner._known_groups == set()

    @pytest.mark.asyncio
    async def test_singleton_group_skipped(self):
        from core.scanner import MarketScanner

        calls: list[str] = []

        async def _capture(group_id: str, no_ids: list[str]) -> bool:
            calls.append(group_id)
            return True

        scanner = MarketScanner(
            on_market_added=_noop_market, on_neg_risk_group=_capture,
        )
        await scanner._register_neg_risk_groups([_gamma_market("G1", "a", vol=1)])
        assert calls == []


async def _noop_market(*_a, **_kw) -> bool:
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Execution — CLOB bundle submission
# ═══════════════════════════════════════════════════════════════════════════

class TestClobBundleSubmission:

    @pytest.mark.asyncio
    async def test_paper_mode_returns_one_response_per_leg(self, patched_clob):
        from core.clob_client import BundleLeg, PolyClient

        client = PolyClient()
        legs = [
            BundleLeg(token_id="A", bid=0.30, size=10.0),
            BundleLeg(token_id="B", bid=0.31, size=10.0),
            BundleLeg(token_id="C", bid=0.32, size=10.0),
        ]
        resps = await client.execute_negrisk_clob_bundle(legs)
        assert len(resps) == 3
        assert [r["token_id"] for r in resps] == ["A", "B", "C"]
        assert [r["leg_idx"] for r in resps] == [0, 1, 2]
        assert all(r["status"] == "paper" for r in resps)

    @pytest.mark.asyncio
    async def test_unprofitable_bundle_is_refused(self, patched_clob):
        """Σ bid ≥ payout (N−1) can never be an arb — never submit it."""
        from core.clob_client import BundleLeg, ClobApiError, PolyClient

        client = PolyClient()
        legs = [
            BundleLeg(token_id="A", bid=0.60, size=1.0),
            BundleLeg(token_id="B", bid=0.60, size=1.0),
        ]   # combined 1.20 ≥ payout 1.0
        with pytest.raises(ClobApiError):
            await client.execute_negrisk_clob_bundle(legs)

    @pytest.mark.asyncio
    async def test_position_cap_enforced(self, patched_clob):
        from core.clob_client import BundleLeg, ClobApiError, PolyClient

        client = PolyClient()
        legs = [
            BundleLeg(token_id="A", bid=0.30, size=5_000.0),
            BundleLeg(token_id="B", bid=0.31, size=5_000.0),
            BundleLeg(token_id="C", bid=0.32, size=5_000.0),
        ]
        with pytest.raises(ClobApiError):
            await client.execute_negrisk_clob_bundle(legs)

    @pytest.mark.asyncio
    async def test_empty_legs_rejected(self, patched_clob):
        from core.clob_client import PolyClient

        with pytest.raises(ValueError):
            await PolyClient().execute_negrisk_clob_bundle([])


# ═══════════════════════════════════════════════════════════════════════════
# Execution — NegRiskBundleGuard
# ═══════════════════════════════════════════════════════════════════════════

class _StubBreaker:
    def __init__(self) -> None:
        self.filled: list[float] = []
        self.released = 0

    def on_fill(self, pnl: float) -> None:
        self.filled.append(pnl)

    def release_open(self) -> None:
        self.released += 1


class _StubNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.critical: list[str] = []

    async def notify(self, msg: str) -> None:
        self.messages.append(msg)

    async def send_critical_error(self, msg: str) -> None:
        self.critical.append(msg)


class _StubClient:
    """Order-status/cancel/unwind stub driven by a per-order script."""

    def __init__(self, statuses: dict[str, dict | None]) -> None:
        self._statuses = statuses
        self.cancelled: list[str] = []
        self.unwound:   list[tuple[str, float]] = []
        self.unwind_fails = False
        # order_id -> shares the TRADE FEED attributes to that order. Per-order,
        # unlike a wallet balance — the distinction the guard depends on.
        self.order_fills: dict[str, float] = {}
        # token_id -> wallet balance. Still here, deliberately NOT consulted for
        # fill decisions: reading it as a per-order fill caused the incident.
        self.share_balances: dict[str, float] = {}

    async def share_balance(self, token_id: str):
        return self.share_balances.get(token_id, 0.0)

    async def order_filled_size(self, order_id: str, token_id=None):
        return self.order_fills.get(order_id, 0.0)

    async def get_order_status(self, order_id: str) -> dict | None:
        return self._statuses.get(order_id)

    async def cancel_order(self, order_id: str) -> dict:
        self.cancelled.append(order_id)
        return {"status": "cancelled"}

    async def unwind_leg(self, token_id: str, size: float, price: float = 0.0) -> dict:
        self.unwound.append((token_id, size))
        if self.unwind_fails:
            raise RuntimeError("no liquidity")
        return {
            "status": "matched",
            "taking_amount": size * 0.25,   # sold below the 0.30 entry
            "making_amount": size,
        }


def _signal(n_legs: int = 3, size: float = 10.0):
    from strategy.arbitrage import ArbLeg, NegRiskSignal

    legs = tuple(
        ArbLeg(token_id=f"T{i}", no_ask=0.31, no_bid=0.30, size=size)
        for i in range(n_legs)
    )
    combined = 0.30 * n_legs
    payout   = float(n_legs - 1)
    return NegRiskSignal(
        condition_id="0xgroup", n_outcomes=n_legs, legs=legs,
        combined_bid=combined, payout=payout, maker_rebate=0.0,
        effective_cost=combined, net_edge=payout - combined,
        relative_edge=(payout - combined) / payout, n_bundles=size,
    )


def _acks(n_legs: int = 3) -> list[dict]:
    return [{"status": "live", "order_id": f"o{i}"} for i in range(n_legs)]


class TestNegRiskBundleGuard:

    @pytest.mark.asyncio
    async def test_complete_bundle_books_guaranteed_profit(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({
            f"o{i}": {"status": "matched", "size_matched": 10.0} for i in range(3)
        })
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert guard.watched_count == 0
        # net_edge = 2.0 − 0.90 = 1.10 per bundle × 10 bundles
        assert breaker.filled == [pytest.approx(11.0)]
        assert breaker.released == 0
        assert "✅" in notifier.messages[0]

    @pytest.mark.asyncio
    async def test_unfilled_bundle_releases_without_booking(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({
            f"o{i}": {"status": "live", "size_matched": 0.0} for i in range(3)
        })
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, order_ttl=0.0)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert breaker.filled == []
        assert breaker.released == 1
        assert sorted(client.cancelled) == ["o0", "o1", "o2"]

    @pytest.mark.asyncio
    async def test_partial_bundle_is_unwound_not_held(self):
        """
        Two of three legs filled is a directional bet, not a smaller arb.
        The guard must cancel the straggler and flatten the filled legs.
        """
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({
            "o0": {"status": "matched", "size_matched": 10.0},
            "o1": {"status": "matched", "size_matched": 10.0},
            "o2": {"status": "live",    "size_matched": 0.0},
        })
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(
            client, breaker, notifier, bundle_timeout=0.0, index_grace=0.0
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert client.cancelled == ["o2"]
        assert sorted(client.unwound) == [("T0", 10.0), ("T1", 10.0)]
        # Realised loss: sold at 0.25, bought at 0.30 → −0.05 × 10 × 2 legs
        assert breaker.filled == [pytest.approx(-1.0)]
        assert breaker.released == 0
        assert "🔻" in notifier.messages[0]

    @pytest.mark.asyncio
    async def test_failed_unwind_raises_critical_alert(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal(n_legs=2)
        client = _StubClient({
            "o0": {"status": "matched", "size_matched": 10.0},
            "o1": {"status": "live",    "size_matched": 0.0},
        })
        client.unwind_fails = True
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(
            client, breaker, notifier, bundle_timeout=0.0, index_grace=0.0
        )

        guard.watch_bundle(sig, _acks(2))
        await guard.poll_once()

        assert notifier.critical and "MANUAL INTERVENTION" in notifier.critical[0]
        assert breaker.filled == [0.0]      # nothing realised, slot still released
        assert breaker.released == 0

    @pytest.mark.asyncio
    async def test_vanished_order_filled_when_trade_feed_attributes_shares(self):
        """Untracked order + attributed fills => genuinely filled."""
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        # The trade feed is the arbiter: all three legs really did fill.
        client.order_fills = {"o0": 10.0, "o1": 10.0, "o2": 10.0}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, index_grace=0.0)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert breaker.filled == [pytest.approx(11.0)]

    @pytest.mark.asyncio
    async def test_vanished_order_not_filled_when_wallet_is_empty(self):
        """
        Untracked order + NO shares on-chain => cancelled, not filled.

        Regression for 2026-09-04: treating "untracked" as "filled" fabricated a
        +0.3040 profit, booked it to daily_state.json, and then tried to unwind
        shares that did not exist ("balance: 0, order amount: 10190000").
        """
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {}             # no fills attributed to any leg
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, index_grace=0.0)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        # No phantom fill, so no fabricated P&L and no bogus unwind.
        assert breaker.filled == []
        assert client.unwound == []

    @pytest.mark.asyncio
    async def test_preexisting_inventory_is_not_mistaken_for_a_fill(self):
        """
        Regression for 2026-09-05 23:12 UTC, live money.

        The guard resolved an untracked order by reading the WALLET balance of
        the leg's token. A wallet balance is not a per-order quantity: it also
        contains earlier positions and the fills of other bundles resting on the
        same token at the same time. Three bundles were in flight on NegRisk
        group 0x905a88afbd9a5f simultaneously, so each guard credited itself
        with the others' fills plus inventory that was already there.

        Concretely: orders 0xdcb4af46f1 and 0x15fc573dd4 filled 0.00, but the
        wallet held 5.07 and 4.00 of their tokens from earlier trades, so both
        were logged "chain confirms N share(s) held, treating as filled". The
        bundle was then declared incomplete and "unwound" — selling a 4.00
        Stefany Shaheen position and 15.21 of a Maura Sullivan position that
        those bundles had never opened, and rebooking the proceeds into 15.21
        more shares of a single outcome. A market-neutral bundle strategy turned
        47% of the account into one directional position.

        The wallet may hold any amount of the token. Only fills the trade feed
        attributes to THIS order id may count.
        """
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        # Inventory from earlier, unrelated trades — the exact trap.
        client.share_balances = {"T0": 5.07, "T1": 5.07, "T2": 4.00}
        client.order_fills    = {}          # none of it belongs to these orders
        breaker, notifier = _StubBreaker(), _StubNotifier()
        # bundle_timeout=0 so a flatten decision is reachable on this pass —
        # without it the test would pass merely because the imbalance clock had
        # not expired, which is not what it is meant to prove.
        guard = NegRiskBundleGuard(
            client, breaker, notifier, bundle_timeout=0.0, index_grace=0.0
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        # Nothing filled -> nothing booked, and above all nothing sold.
        assert breaker.filled == []
        assert client.unwound == [], (
            "unwound positions that this bundle never opened"
        )

    @pytest.mark.asyncio
    async def test_only_this_orders_fill_counts_when_wallet_holds_more(self):
        """
        The partial-credit case: the order really did fill, but the wallet holds
        far more than it bought. The leg must be credited with its own fill
        only, never the surrounding inventory.
        """
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.share_balances = {"T0": 99.0, "T1": 99.0, "T2": 99.0}
        client.order_fills    = {"o0": 10.0, "o1": 10.0, "o2": 4.0}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        # bundle_timeout=0: tolerate no imbalance, so the flatten happens on the
        # same pass that detects it.
        guard = NegRiskBundleGuard(
            client, breaker, notifier, bundle_timeout=0.0, index_grace=0.0
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        # o2 is short of its 10.0 size, so the bundle is incomplete and the two
        # legs that DID fill are flattened — at their own sizes, not the
        # wallet's 99.0.
        assert client.unwound, "an incomplete bundle must be flattened"
        for _tok, size in client.unwound:
            assert size <= 10.0 + 1e-9, f"unwound {size}, more than this bundle bought"

    @pytest.mark.asyncio
    async def test_group_is_off_limits_after_a_flatten(self):
        """
        A group that just had to be flattened must not be re-entered instantly.

        On 2026-09-05 the loop placed a new bundle on the same NegRisk group
        ~200 ms after unwinding the last one, 9 times in 4 minutes, paying the
        spread on every round trip.
        """
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {"o0": 10.0, "o1": 10.0}     # o2 never filled
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(
            client, breaker, notifier, bundle_timeout=0.0, group_cooldown=60.0,
            index_grace=0.0,
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert not guard.is_watching(sig.condition_id), "bundle should be resolved"
        assert guard.is_busy(sig.condition_id), "group must be cooling off"

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {"o0": 10.0, "o1": 10.0}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(
            client, breaker, notifier, bundle_timeout=0.0, group_cooldown=0.0,
            index_grace=0.0,
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        # A zero cooldown must not latch the group off permanently.
        assert not guard.is_busy(sig.condition_id)

    @pytest.mark.asyncio
    async def test_failed_submission_leg_is_not_polled(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({
            "o0": {"status": "live", "size_matched": 0.0},
            "o1": {"status": "live", "size_matched": 0.0},
        })
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, order_ttl=0.0)

        acks = _acks()
        acks[2] = {"status": "error", "error": "rejected"}
        guard.watch_bundle(sig, acks)
        await guard.poll_once()

        assert sorted(client.cancelled) == ["o0", "o1"]
        assert breaker.released == 1

    @pytest.mark.asyncio
    async def test_is_watching_blocks_reentry(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({
            f"o{i}": {"status": "live", "size_matched": 0.0} for i in range(3)
        })
        guard = NegRiskBundleGuard(client, _StubBreaker(), _StubNotifier())

        assert not guard.is_watching("0xgroup")
        guard.watch_bundle(sig, _acks())
        assert guard.is_watching("0xgroup")

    @pytest.mark.asyncio
    async def test_incomplete_bundle_waits_out_the_timeout(self):
        """A leg lagging by less than the timeout must not trigger an unwind."""
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({
            "o0": {"status": "matched", "size_matched": 10.0},
            "o1": {"status": "live",    "size_matched": 0.0},
            "o2": {"status": "live",    "size_matched": 0.0},
        })
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, bundle_timeout=60.0)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert guard.watched_count == 1
        assert client.unwound == []
        assert breaker.filled == []


# ═══════════════════════════════════════════════════════════════════════════
# ws_feed — live batched price_change frames
# ═══════════════════════════════════════════════════════════════════════════

def _live_price_change(*entries: tuple[str, str, str]) -> str:
    """
    The frame shape Polymarket actually sends: no top-level asset_id, every
    affected asset batched under `price_changes`, each with its own best_ask.
    """
    return json.dumps({
        "market": "0xmarket",
        "price_changes": [
            {
                "asset_id": asset, "price": "0.5", "size": "2430",
                "side": side, "hash": "abc",
                "best_bid": "0.001", "best_ask": best_ask,
            }
            for asset, side, best_ask in entries
        ],
        "timestamp": "1786905468000",
    })


class TestLivePriceChangeFrames:

    def test_batched_frame_is_expanded_per_asset(self):
        from core.ws_feed import _parse_events

        events = _parse_events(_live_price_change(
            ("A", "BUY", "0.31"), ("B", "SELL", "0.44"),
        ))
        assert [e["asset_id"] for e in events] == ["A", "B"]
        assert all(e["event_type"] == "price_change" for e in events)
        assert [e["best_ask"] for e in events] == ["0.31", "0.44"]

    def test_batched_frame_updates_a_group(self):
        """Regression: unexpanded frames routed to "" and were dropped."""
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        st.update_leg("B", _book("B", 0.31, 100))
        st.build_tick()

        for ev in __import__("core.ws_feed", fromlist=["_parse_events"])._parse_events(
            _live_price_change(("A", "BUY", "0.26"))
        ):
            assert st.update_leg(ev["asset_id"], ev)

        tick = st.build_tick()
        assert tick is not None
        assert tick["no_asks"][0] == 0.26

    def test_best_ask_wins_over_level_inference(self):
        """
        A batched frame reporting a WORSE best_ask must still move our view:
        the old level-delta logic only accepted improvements and would have
        left a stale, too-cheap ask driving phantom signals.
        """
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        events = __import__("core.ws_feed", fromlist=["x"])._parse_events(
            _live_price_change(("A", "SELL", "0.48"))
        )
        assert st.update_leg("A", events[0])
        assert st._best_ask["A"] == 0.48

    def test_unknown_depth_is_none_not_zero(self):
        """
        A batched frame states price but not depth-at-price. Reporting 0 would
        make the detector drop the leg as unfillable; None means "uncapped".
        """
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        st.update_leg("B", _book("B", 0.31, 100))
        st.build_tick()

        events = __import__("core.ws_feed", fromlist=["x"])._parse_events(
            _live_price_change(("A", "SELL", "0.26"))
        )
        st.update_leg("A", events[0])
        tick = st.build_tick()
        assert tick["no_ask_sizes"] == [None, 100.0]

    def test_detector_treats_unknown_depth_as_uncapped(self):
        d = _det()
        sig = d.evaluate_neg_risk(
            "grp", ["A", "B", "C"], [0.30, 0.30, 0.30],
            max_position_usdc=20.0,
            maker_rebate=0.0, tick_size=0.01,
            no_ask_sizes=[None, 50.0, None],
        )
        assert sig is not None
        # Only leg B caps depth; capital is the tighter constraint here.
        assert sig.n_bundles == pytest.approx(22.98, abs=0.01)

    def test_binary_market_also_consumes_batched_frames(self):
        """The same fix restores quote updates on the live binary path."""
        queue: asyncio.Queue = asyncio.Queue()
        shard = MarketShard(queue)
        shard.add("cond", "YES", "NO")
        shard._dispatch(json.dumps([_book("YES", 0.60, 10), _book("NO", 0.39, 10)]))
        assert queue.get_nowait()["type"] == "arb_tick"

        shard._dispatch(_live_price_change(("YES", "SELL", "0.58")))
        tick = queue.get_nowait()
        assert tick["yes_ask"] == 0.58

    def test_malformed_entries_are_skipped(self):
        from core.ws_feed import _parse_events

        events = _parse_events(json.dumps({
            "market": "0x", "price_changes": [
                {"price": "0.5"},                       # no asset_id
                "not-a-dict",
                {"asset_id": "A", "best_ask": "0.30"},
            ],
        }))
        assert [e["asset_id"] for e in events] == ["A"]

    def test_out_of_range_best_ask_falls_back(self):
        """A nonsense best_ask must not overwrite a good quote."""
        st = _NegRiskGroupState("grp", ["A", "B"])
        st.update_leg("A", _book("A", 0.30, 100))
        assert not st.update_leg("A", {
            "event_type": "price_change", "asset_id": "A",
            "best_ask": "0", "side": "BUY", "price": "0.5", "size": "10",
        })
        assert st._best_ask["A"] == 0.30

    def test_legacy_flat_frame_still_parses(self):
        from core.ws_feed import _parse_events

        events = _parse_events(json.dumps({
            "event_type": "price_change", "asset_id": "A",
            "price": "0.28", "side": "SELL", "size": "50",
        }))
        assert len(events) == 1 and events[0]["asset_id"] == "A"


# ═══════════════════════════════════════════════════════════════════════════════
# Duplicate-order rejection is terminal, not transient
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuplicateOrderIsTerminal:
    """
    Regression for 2026-09-05 23:34-23:37. "order 0x… is invalid. Duplicated."
    carries no HTTP status, so the retry taxonomy filed it under HTTP-unknown
    and retried five times with backoff — ~8 s of certain-to-fail requests
    before failing the leg anyway, repeating roughly once a minute on the same
    NegRisk group.
    """

    def test_duplicate_message_is_recognised(self):
        from core.clob_client import _is_duplicate_order
        exc = RuntimeError(
            "order 0xad8e501bc907b39339fbddf99be3baf44f895a8402ebab19ed12a382"
            "f98828c1 is invalid. Duplicated."
        )
        assert _is_duplicate_order(exc) is True

    @pytest.mark.parametrize("msg", [
        "429 Too Many Requests",
        "connection reset by peer",
        "",
    ])
    def test_transport_failures_stay_retryable(self, msg):
        """Do not swallow the retries that actually help."""
        from core.clob_client import _terminal_order_error
        assert _terminal_order_error(RuntimeError(msg)) is None

    def test_insufficient_balance_is_terminal(self):
        """
        Nothing about the wallet changes while we back off, so retrying a
        rejection the exchange derived from our balance just re-sends it.
        """
        from core.clob_client import _terminal_order_error
        exc = RuntimeError(
            "not enough balance / allowance: the balance is not enough -> "
            "balance: 4885275, order amount: 4997830"
        )
        assert _terminal_order_error(exc) == "insufficient balance/allowance"

    def test_duplicate_is_named_distinctly(self):
        from core.clob_client import _terminal_order_error
        exc = RuntimeError("order 0xabc is invalid. Duplicated.")
        assert _terminal_order_error(exc) == "duplicate order"


class TestGroupCooldownOnSubmissionFailure:
    """
    A leg the exchange refuses keeps being refused, so the group must back off
    rather than re-signal ~50 s later and reproduce the same rejection.
    """

    def test_cool_down_marks_the_group_busy(self):
        from execution.negrisk_guard import NegRiskBundleGuard
        client = _StubClient({})
        guard = NegRiskBundleGuard(
            client, _StubBreaker(), _StubNotifier(), group_cooldown=60.0
        )
        assert not guard.is_busy("0xgroup")
        guard.cool_down("0xgroup")
        assert guard.is_busy("0xgroup")

    def test_cool_down_respects_a_disabled_cooldown(self):
        from execution.negrisk_guard import NegRiskBundleGuard
        client = _StubClient({})
        guard = NegRiskBundleGuard(
            client, _StubBreaker(), _StubNotifier(), group_cooldown=0.0
        )
        guard.cool_down("0xgroup")
        assert not guard.is_busy("0xgroup")


class TestFreshOrdersAreNotJudged:
    """
    Regression for 2026-09-05 23:33-23:41, the costliest fault of the night.

    The CLOB returns a null body for an order it does not track — and it returns
    the SAME null body for an order it does not track YET. A limit order is not
    queryable for a moment after it is accepted.

    The guard judged inside that window: eight bundles were released ~200 ms
    after being placed, every leg logged "untracked with no attributed fills —
    cancelled, not filled", and the slot was handed back. But the orders were
    live on the book. Six of those 24 abandoned legs filled with nobody
    watching, turning 29 pUSD of cash into 30.42 shares of a single outcome —
    an unmanaged naked directional position, which is precisely what this guard
    exists to prevent. The guard's own log said nothing had filled.
    """

    @pytest.mark.asyncio
    async def test_a_just_placed_order_is_left_alone(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        # every leg untracked, as a freshly accepted order looks
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(
            client, breaker, notifier, index_grace=5.0, bundle_timeout=0.0
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert guard.is_watching(sig.condition_id), (
            "released a bundle whose orders were still being indexed"
        )
        assert breaker.released == 0, "handed back the slot for live orders"
        assert client.cancelled == [], "cancelled orders that were still resting"

    @pytest.mark.asyncio
    async def test_after_the_grace_window_it_resolves_normally(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        # grace=0 -> judge immediately, the pre-existing behaviour
        guard = NegRiskBundleGuard(client, breaker, notifier, index_grace=0.0)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert not guard.is_watching(sig.condition_id)
        assert breaker.released == 1

    @pytest.mark.asyncio
    async def test_grace_does_not_delay_a_leg_we_cancelled(self):
        """An order we asked to cancel is ours to judge immediately."""
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, index_grace=999.0)

        guard.watch_bundle(sig, _acks())
        for b in guard._bundles.values():
            for leg in b.legs:
                leg.cancel_requested = True
        await guard.poll_once()

        assert not guard.is_watching(sig.condition_id)


class TestBlindPollingDoesNotLeakAFill:
    """
    Regression for the residual hole found on 2026-09-06.

    `matched` is only as good as the status polls that produced it. Cloudflare
    started returning 400 on /data/order, and get_order_status failures leave a
    leg's matched at 0 by design (the conservative choice — do not guess). But
    if that lasts a leg's whole life, _finalize sees matched == 0 everywhere,
    concludes "expired unfilled", releases the slot and walks away from a leg
    that DID fill — the same naked position that cost 29 pUSD earlier that
    night, reached by a different route.

    So the "nothing filled" conclusion must be confirmed against the trade feed
    before the slot is released.
    """

    @pytest.mark.asyncio
    async def test_a_fill_polling_never_saw_is_still_handled(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        # Every status poll fails, exactly as a Cloudflare block behaves.
        class _BlindClient(_StubClient):
            async def get_order_status(self, order_id):
                raise RuntimeError("blocked by Cloudflare with status 400")

        client = _BlindClient({"o0": None, "o1": None, "o2": None})
        # ...but one leg really did fill.
        client.order_fills = {"o1": 10.0}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(
            client, breaker, notifier, order_ttl=0.0, index_grace=0.0
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert breaker.released == 0, (
            "released the slot while a filled leg was still held"
        )
        assert client.unwound, "abandoned a filled leg instead of flattening it"

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_bundle_still_releases(self):
        """The recheck must not turn every dissolved bundle into a flatten."""
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(
            client, breaker, notifier, order_ttl=0.0, index_grace=0.0
        )

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert breaker.released == 1
        assert client.unwound == []


class TestPresumedGoneOrdersAreActuallyCancelled:
    """
    Regression for 2026-09-06 00:31, the leak that survived the first two fixes.

    An untracked order with no attributed fills was logged "cancelled, not
    filled" and marked closed — but nothing ever cancelled it. _finalize only
    cancels legs still flagged open, so it skipped them, and the orders stayed
    live on the book.

    Between 23:49 and 00:31 the guards logged 35 bundles "expired unfilled" and
    resolved 24 untracked orders as not-filled, with only 2 status-poll
    failures. Over the same window the wallet spent 27 pUSD acquiring 25.36
    Beriont and 15.21 Sullivan shares. The guard was announcing that nothing
    had happened while the orders it had walked away from were filling.

    "Untracked" is not proof an order is gone. Cancelling one that genuinely is
    gone costs nothing; assuming it is gone costs the position.
    """

    @pytest.mark.asyncio
    async def test_an_unfilled_untracked_order_is_cancelled_for_real(self):
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {}          # nothing attributed to any leg
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, index_grace=0.0)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert sorted(client.cancelled) == ["o0", "o1", "o2"], (
            "walked away from orders without cancelling them"
        )

    @pytest.mark.asyncio
    async def test_a_filled_leg_is_not_cancelled(self):
        """A leg the feed confirms filled must be flattened, not cancelled."""
        from execution.negrisk_guard import NegRiskBundleGuard

        sig = _signal()
        client = _StubClient({"o0": None, "o1": None, "o2": None})
        client.order_fills = {"o0": 10.0, "o1": 10.0, "o2": 10.0}
        breaker, notifier = _StubBreaker(), _StubNotifier()
        guard = NegRiskBundleGuard(client, breaker, notifier, index_grace=0.0)

        guard.watch_bundle(sig, _acks())
        await guard.poll_once()

        assert client.cancelled == []
        assert breaker.filled, "a complete bundle must be booked"
