"""
strategy/arb_duration.py
────────────────────────
ArbDurationTracker — measures how long each market stays in an arbitrage state.

A market is "in arb" on any tick where the detector/pricer produces a signal.
The *arbitrage duration* is the wall-clock time from the first such tick until
the market transitions back to no-arb (the arb window closes). Reporting this on
Telegram tells you how long opportunities actually persist — a key signal for
whether a market is worth chasing (long windows = capturable; sub-second flickers
= already arbitraged away by faster players).

Pure and synchronous (no I/O) so it is trivially unit-testable; the caller is
responsible for sending the returned ArbWindow to Telegram.

Usage::

    tracker = ArbDurationTracker(min_duration_s=0.0)
    # for every evaluated arb_tick:
    window = tracker.update(cid, in_arb=signal is not None,
                            edge_bps=edge, is_maker=is_maker, ts=tick_ts)
    if window is not None:          # the arb just closed
        notifier.send_arb_duration(window.condition_id, window.duration_s,
                                   window.peak_edge_bps, window.ticks,
                                   window.is_maker_peak)

Timestamps must come from a monotonic clock (e.g. time.monotonic() — the same
clock ws_feed stamps arb_tick["ts"] with) so durations are correct.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArbWindow:
    """A closed arbitrage window for one market."""
    condition_id:  str
    duration_s:    float   # first-arb-tick → close-tick, seconds
    peak_edge_bps: float   # largest edge observed during the window
    ticks:         int     # number of in-arb ticks observed
    is_maker_peak: bool    # whether the peak edge came from the maker path
    still_open:    bool = False   # True only for windows flushed at shutdown


@dataclass(slots=True)
class _Open:
    start_ts:      float
    peak_edge_bps: float
    ticks:         int
    is_maker_peak: bool


class ArbDurationTracker:
    """
    Tracks open arbitrage windows per market and emits an ArbWindow on close.

    Feed every evaluated tick to update(). It returns:
      • None while a window stays open (or for a no-arb tick on a market with no
        open window);
      • an ArbWindow the moment a market goes arb → no-arb (window ≥ min_duration_s).
    """

    def __init__(self, min_duration_s: float = 0.0) -> None:
        self._min = max(0.0, min_duration_s)
        self._open: dict[str, _Open] = {}

    def update(
        self,
        condition_id: str,
        *,
        in_arb:   bool,
        ts:       float,
        edge_bps: float = 0.0,
        is_maker: bool = False,
    ) -> ArbWindow | None:
        w = self._open.get(condition_id)

        if in_arb:
            if w is None:
                self._open[condition_id] = _Open(
                    start_ts=ts, peak_edge_bps=edge_bps, ticks=1, is_maker_peak=is_maker,
                )
            else:
                w.ticks += 1
                if edge_bps > w.peak_edge_bps:
                    w.peak_edge_bps = edge_bps
                    w.is_maker_peak = is_maker
            return None

        # not in arb → close any open window for this market
        if w is None:
            return None
        del self._open[condition_id]
        duration = max(0.0, ts - w.start_ts)
        if duration < self._min:
            return None
        return ArbWindow(
            condition_id=condition_id,
            duration_s=duration,
            peak_edge_bps=w.peak_edge_bps,
            ticks=w.ticks,
            is_maker_peak=w.is_maker_peak,
        )

    @property
    def open_count(self) -> int:
        return len(self._open)

    def flush(self, ts: float) -> list[ArbWindow]:
        """
        Close every still-open window (e.g. on shutdown), returning their
        ArbWindows with still_open=True. The reported duration is time-so-far,
        not a true close.
        """
        out: list[ArbWindow] = []
        for cid, w in self._open.items():
            out.append(ArbWindow(
                condition_id=cid,
                duration_s=max(0.0, ts - w.start_ts),
                peak_edge_bps=w.peak_edge_bps,
                ticks=w.ticks,
                is_maker_peak=w.is_maker_peak,
                still_open=True,
            ))
        self._open.clear()
        return out
