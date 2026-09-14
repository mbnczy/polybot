"""
The reader was OOM-killed at its 1 GB cap on 2026-09-13. The Python heap was
flat; glibc was keeping memory Python had freed. Two changes, pinned here.
"""

from __future__ import annotations

from scripts.demo_cross_market import _release_freed_memory
from strategy.implication_mapper import build_candidates


def test_releasing_memory_never_raises():
    assert _release_freed_memory() in (True, False)


def _mkt(cid, q):
    return {"conditionId": cid, "question": q}


def test_disjoint_pairs_are_skipped_and_overlap_is_unchanged():
    """The allocation-free overlap must equal |A∩B| / |A∪B| exactly."""
    markets = [
        _mkt("0x1", "Cesena FC vs. US Cremonese: O/U 5.5"),
        _mkt("0x2", "Cesena FC vs. US Cremonese: O/U 4.5"),
        _mkt("0x3", "Will Bitcoin reach 200k in December?"),
    ]
    cands = build_candidates(markets, max_pairs=10)
    ids = {frozenset((c.a_id, c.b_id)) for c in cands}
    assert frozenset(("0x1", "0x2")) in ids
    assert not any("0x3" in pair for pair in ids)            # shares no word
    from strategy.implication_mapper import _tokens            # noqa: PLC0415
    a, b = _tokens(markets[0]["question"]), _tokens(markets[1]["question"])
    (c,) = [c for c in cands if {c.a_id, c.b_id} == {"0x1", "0x2"}]
    assert c.overlap == len(a & b) / len(a | b)
