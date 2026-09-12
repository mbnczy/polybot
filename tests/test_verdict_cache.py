"""
The model should see a pair once, ever.

The prefilter builds thousands of candidate pairs a pass and the model costs
three minutes per thirty, so breadth was capped at 30 — one percent of what was
found. A verdict is a fact about two immutable questions, so it is cached, and
the cost of coverage becomes one-time instead of per pass.
"""

from __future__ import annotations

import json
import time

from strategy.cross_market import Implication
from strategy.verdict_cache import VerdictCache


class _Cand:
    def __init__(self, a, b):
        self.a_id, self.b_id = a, b


def test_a_known_pair_is_not_sent_to_the_model_again(tmp_path):
    c = VerdictCache(tmp_path / "v.json")
    cands = [_Cand("0xa", "0xb"), _Cand("0xc", "0xd")]
    c.remember(cands, [Implication("0xa", "0xb", 0.97, "first half ⊆ match")])

    known, unknown = c.split(cands)
    assert unknown == []
    assert [(i.narrow, i.broad, i.confidence) for i in known] == [("0xa", "0xb", 0.97)]


def test_a_refusal_is_remembered_too():
    """Most candidates are not implications; re-asking about them is the cost."""
    c = VerdictCache("/dev/null")
    pair = _Cand("0xc", "0xd")
    c.remember([pair], [])
    known, unknown = c.split([pair])
    assert known == [] and unknown == []


def test_the_pair_is_unordered():
    c = VerdictCache("/dev/null")
    c.remember([_Cand("0xa", "0xb")], [Implication("0xa", "0xb", 0.97, "e")])
    known, unknown = c.split([_Cand("0xb", "0xa")])
    assert len(known) == 1 and unknown == []


def test_an_unseen_pair_still_goes_to_the_model():
    c = VerdictCache("/dev/null")
    c.remember([_Cand("0xa", "0xb")], [])
    known, unknown = c.split([_Cand("0xa", "0xb"), _Cand("0xe", "0xf")])
    assert [(u.a_id, u.b_id) for u in unknown] == [("0xe", "0xf")]


def test_it_survives_a_restart(tmp_path):
    path = tmp_path / "v.json"
    c = VerdictCache(path)
    c.remember([_Cand("0xa", "0xb")], [Implication("0xa", "0xb", 0.95, "e")])
    c.save()
    known, unknown = VerdictCache(path).load().split([_Cand("0xa", "0xb")])
    assert len(known) == 1 and unknown == []


def test_stale_entries_are_pruned(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"verdicts": {
        "0xa|0xb": {"narrow": None, "ts": time.time() - 40 * 86_400},
        "0xc|0xd": {"narrow": None, "ts": time.time()},
    }}))
    c = VerdictCache(path).load()
    assert c.prune() == 1 and len(c) == 1


def test_a_corrupt_cache_does_not_stop_the_reader(tmp_path):
    path = tmp_path / "v.json"
    path.write_text("{not json")
    c = VerdictCache(path).load()
    assert len(c) == 0
    assert c.split([_Cand("0xa", "0xb")])[1] != []
