"""
The prefilter exactly as it was before it was rewritten for a 40,000-market
universe: a full cross product. Kept verbatim as the reference the rewrite must
reproduce pair for pair (tests/test_prefilter_scale.py). Not used in production.
"""

from __future__ import annotations

import itertools
import time

from strategy.implication_mapper import (  # noqa: F401
    Candidate, SHAPE_DUPLICATE, _EVENT_WEIGHT, _SAME_EVENT_MIN_OVERLAP, _SHAPE_RANK,
    _TIME_WEIGHT, _end_ts, _event_id, _exclusion_group, _tokens, is_over_under,
    logger, market_outcomes, pair_shape, statistics, time_score,
)
import strategy.implication_mapper as _im


def build_candidates_reference(
    markets:       list[dict],
    min_overlap:   float = 0.5,
    max_pairs:     int   = 400,
    max_per_event: int   = 20,
) -> list[Candidate]:
    """
    Reduce a market universe to plausible pairs.

    A full cross-product of 2,100 markets is ~2.2M pairs. Two markets can only be
    logically related if they discuss the same subject, so require token overlap,
    and treat a shared event as the stronger signal.

    `max_per_event` matters more than it looks. Measured on the live universe,
    39,595 pairs clear the 0.5 threshold and ~19,000 of those have IDENTICAL
    token sets — Polymarket generates dozens of near-identical titles per event
    ("Balance of Power: D Senate, D House", "...D Senate, R House", ...). Without
    a per-event cap one 40-outcome election fills every slot with mutually
    exclusive pairs that all come back NONE, and the rest of the universe is
    never examined.
    """
    rows = []
    for m in markets:
        title = str(m.get("question") or m.get("title") or "").strip()
        cid   = str(m.get("conditionId") or m.get("condition_id") or "").strip()
        if not title or not cid:
            continue
        # What an over/under market counts, computed once per market: comparing
        # it per PAIR would run the vocabulary against ~720,000 pairs.
        quantity = statistics(title) if is_over_under(title) else None
        rows.append((cid, title, _tokens(title), _event_id(m), _exclusion_group(m),
                     _end_ts(m), quantity, market_outcomes(m)))

    out: list[Candidate] = []
    excluded_siblings = 0
    now = time.time()
    too_slow = 0
    cross_statistic = 0
    for (a_id, a_t, a_tok, a_ev, a_ng, a_end, a_q, a_o), (b_id, b_t, b_tok, b_ev, b_ng, b_end, b_q, b_o) \
            in itertools.combinations(rows, 2):
        if a_id == b_id or not a_tok or not b_tok:
            continue
        # Alternative outcomes of one NegRisk question. Mutually exclusive by
        # contract, so there is nothing here for an implication mapper to find —
        # and where they ARE mispriced (Σ yes > 1) the main bot trades them
        # directly through the NegRisk path. Skipped before scoring, because
        # their titles are near-identical and they would otherwise dominate.
        if a_ng and a_ng == b_ng:
            excluded_siblings += 1
            continue
        # Two over/under markets counting different things — a team's corners
        # and its goals — share a template and look like a nest. They are not
        # related at all, and the one-week window is full of them. See
        # strategy/quantity_guard.py.
        if a_q is not None and b_q is not None and a_q != b_q:
            cross_statistic += 1
            continue
        # Both floors are positive, so a pair with no word in common can never
        # pass — and that is most of the ~2.9M pairs 2,400 markets make. Building
        # a union and an intersection set for each of them was millions of
        # short-lived allocations per discovery, and glibc kept the freed memory:
        # the reader crept to its 1 GB cap and was OOM-killed on 2026-09-13.
        if a_tok.isdisjoint(b_tok):
            continue
        common = len(a_tok & b_tok)
        overlap = common / (len(a_tok) + len(b_tok) - common)
        same_event = bool(a_ev and b_ev and a_ev == b_ev)
        floor = _SAME_EVENT_MIN_OVERLAP if same_event else min_overlap
        if overlap < floor:
            continue
        lockup = (
            max(0.0, (max(a_end, b_end) - now) / 86_400.0)
            if a_end is not None and b_end is not None else None
        )
        # With a cap set, an unknown end date cannot be shown to meet it, and a
        # pair whose lockup cannot be priced cannot be traded inside the window.
        if _im._MAX_LOCKUP_DAYS > 0 and (lockup is None or lockup > _im._MAX_LOCKUP_DAYS):
            too_slow += 1
            continue
        out.append(Candidate(
            a_id, b_id, a_t, b_t, overlap, same_event,
            a_ev if same_event else "", pair_shape(a_t, b_t), lockup, a_o, b_o,
        ))

    if cross_statistic:
        logger.info(
            "implication mapper | skipped %d over/under pair(s) counting different "
            "statistics — corners do not imply goals", cross_statistic,
        )

    if excluded_siblings:
        logger.info(
            "implication mapper | skipped %d mutually exclusive NegRisk sibling "
            "pair(s) — no implication can hold between them", excluded_siblings,
        )

    # Rank by SHAPE, not by similarity.
    #
    # Sorting on raw overlap sent the model 90% numeric ladders; sorting on
    # proximity to a "typical" overlap replaced them with name-swap siblings.
    # Both were guesses about what a promising pair looks like. The shape
    # classes above are not a guess — they are measured against how the markets
    # actually resolved, and they separate the ~1.6% of pairs whose shape can
    # carry an implication from the 98% whose shape cannot.
    #
    # A shared event breaks ties within a shape: Polymarket groups genuinely
    # related markets, so it is real evidence, but it is weaker than shape —
    # the siblings we just demoted were overwhelmingly same-event.
    # Secondary terms are capped below the 1.0 gap between shape ranks, so a
    # misconfigured pair of weights cannot let speed outrank shape.
    secondary = _TIME_WEIGHT + _EVENT_WEIGHT
    scale = 0.99 / secondary if secondary >= 0.99 else 1.0

    def _score(c: Candidate) -> float:
        return (
            _SHAPE_RANK.get(c.shape, 0.0)
            + scale * _EVENT_WEIGHT * (1.0 if c.same_event else 0.0)
            + scale * _TIME_WEIGHT * time_score(c.lockup_days)
        )

    if too_slow:
        logger.info(
            "implication mapper | %d pair(s) dropped for locking capital longer "
            "than %.0f days", too_slow, _im._MAX_LOCKUP_DAYS,
        )

    dupes = [c for c in out if c.shape == SHAPE_DUPLICATE]
    if dupes:
        out = [c for c in out if c.shape != SHAPE_DUPLICATE]
        logger.info(
            "implication mapper | %d identically worded pair(s) excluded — the "
            "same question listed twice needs no model to recognise; example: %s",
            len(dupes), dupes[0].a_title[:70],
        )

    out.sort(key=_score, reverse=True)

    # There was a quota here capping how much of the budget threshold ladders
    # could take. It was written when ladders looked like the problem; once the
    # shape ranking put name-swap siblings last, the quota was handing 320 of
    # 400 model calls to the one class that cannot contain an implication.
    #
    # It was also ordered wrongly: it truncated the ladder pool BEFORE the
    # per-event cap, so the ladders it kept were concentrated in a few crypto
    # events and the cap then culled them anyway — a 60% quota admitted 12
    # ladders. Removing it moved the selected set from 22.9% to 37.3% expected
    # both-YES. max_per_event already prevents any one event monopolising, and
    # it does so after ranking, where it works.

    if max_per_event > 0:
        seen: dict[str, int] = {}
        capped: list[Candidate] = []
        for c in out:
            if c.event_id:
                n = seen.get(c.event_id, 0)
                if n >= max_per_event:
                    continue
                seen[c.event_id] = n + 1
            capped.append(c)
        if len(capped) < len(out):
            logger.info(
                "implication mapper | per-event cap dropped %d pair(s) so no "
                "single event monopolises the budget", len(out) - len(capped),
            )
        out = capped

    if len(out) > max_pairs:
        logger.info(
            "implication mapper | %d candidates over the %d cap — keeping the "
            "strongest; raise max_pairs to widen coverage", len(out), max_pairs,
        )
    return out[:max_pairs]
