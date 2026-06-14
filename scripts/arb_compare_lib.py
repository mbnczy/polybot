"""
scripts/arb_compare_lib.py
──────────────────────────
Shared stats backing the main-vs-refactored A/B comparison.

Both `compare_monitor.py` instances (one per branch/pipeline) append every
closed arbitrage window to a single shared JSONL file, tagged with their
BOT_LABEL. A designated "summary leader" reads that file at end-of-day and posts
a side-by-side comparison of each branch's **average arb duration** and
**frequency** (and average magnitude) to Telegram.

Pure + dependency-free (stdlib only) so it works identically when copied into the
`main` worktree and is trivially unit-testable.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


def append_window(
    path: str,
    label: str,
    *,
    ts: float,
    duration_s: float,
    edge_bps: float,
) -> None:
    """Append one closed arb window record (one JSON object per line)."""
    rec = {
        "label":      label,
        "ts":         round(float(ts), 3),     # epoch seconds (wall clock)
        "duration_s": round(float(duration_s), 3),
        "edge_bps":   round(float(edge_bps), 1),
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_windows(path: str, *, since_ts: float = 0.0) -> list[dict]:
    """Read all window records with ts >= since_ts (missing file → [])."""
    out: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if float(r.get("ts", 0.0)) >= since_ts:
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def aggregate(windows: list[dict]) -> dict[str, dict]:
    """
    Group window records by label → per-label stats:
      {label: {count, avg_duration_s, max_duration_s, avg_edge_bps}}
    """
    agg: dict[str, dict] = {}
    for r in windows:
        lab = str(r.get("label", "?"))
        a = agg.setdefault(
            lab, {"count": 0, "_dsum": 0.0, "_esum": 0.0, "max_duration_s": 0.0}
        )
        d = float(r.get("duration_s", 0.0))
        e = float(r.get("edge_bps", 0.0))
        a["count"] += 1
        a["_dsum"] += d
        a["_esum"] += e
        a["max_duration_s"] = max(a["max_duration_s"], d)
    for a in agg.values():
        n = a["count"] or 1
        a["avg_duration_s"] = round(a["_dsum"] / n, 3)
        a["avg_edge_bps"] = round(a["_esum"] / n, 1)
        a["max_duration_s"] = round(a["max_duration_s"], 3)
        del a["_dsum"]
        del a["_esum"]
    return agg


def format_summary(agg: dict[str, dict], *, day_label: str, window_hours: float = 24.0) -> str:
    """Human-readable Telegram summary comparing each branch."""
    lines = [
        f"📊 <b>DAILY ARB SUMMARY</b> — {day_label} (last {window_hours:.0f}h)",
        "Avg duration &amp; frequency of arbitrage windows per branch:",
    ]
    if not agg:
        lines.append("  (no arbitrage windows recorded)")
        return "\n".join(lines)
    for lab in sorted(agg):
        a = agg[lab]
        freq_per_h = round(a["count"] / window_hours, 2) if window_hours else 0.0
        lines.append(
            f"<b>[{lab}]</b> windows=<code>{a['count']}</code> "
            f"freq=<code>{freq_per_h}/h</code> "
            f"avg_dur=<code>{a['avg_duration_s']:.2f}s</code> "
            f"max_dur=<code>{a['max_duration_s']:.2f}s</code> "
            f"avg_edge=<code>{a['avg_edge_bps']:.0f}bps</code>"
        )
    return "\n".join(lines)


def seconds_until_next_utc_midnight(now: datetime | None = None) -> float:
    """Seconds from `now` until the next 00:00 UTC (the end-of-day boundary)."""
    now = now or datetime.now(timezone.utc)
    nxt = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return max(1.0, (nxt - now).total_seconds())


def prune(path: str, *, keep_days: float = 7.0) -> None:
    """Drop records older than keep_days so the stats file stays bounded."""
    cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86_400.0
    recent = read_windows(path, since_ts=cutoff)
    with open(path, "w") as f:
        for r in recent:
            f.write(json.dumps(r) + "\n")
