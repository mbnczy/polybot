# Deploying the cross-market reader

The reader runs as the system unit `polybot-crossmarket.service`, installed
earlier with the original worktree (`/home/ubuntu/polybot-dev/cross-market`).
`Restart=always` keeps exactly one instance alive — which is also why launching
it by hand alongside the unit produces a SECOND reader: the unit restarts its own
30 seconds after any kill. Change what it runs through the unit, never beside it.

`polybot-crossmarket.service.d/exit-tracking.conf` is a drop-in that points the
unit at this branch:

    sudo cp -r deploy/polybot-crossmarket.service.d /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl restart polybot-crossmarket

To go back to the original unit:

    sudo rm /etc/systemd/system/polybot-crossmarket.service.d/exit-tracking.conf
    sudo systemctl daemon-reload
    sudo systemctl restart polybot-crossmarket

Logs go to the journal: `journalctl -u polybot-crossmarket -f`.

The hardening is unchanged except for one path: paper positions are written
beside this branch's code, so the drop-in adds it to `ReadWritePaths`. The
trading bot's directory stays read-only and its `.env` unreachable.

## Execution

The trading bot trades what the reader finds through `execution/cross_guard.py`.
The reader writes verified implications to
`/home/ubuntu/polybot-dev/cross-exec/cross_implications.json`; the bot reads it
(its sandbox can read `/home/ubuntu`, only write its own directory).

Execution is OFF unless the bot's `.env` sets `CROSS_EXECUTION_ENABLED=true`.
While off it evaluates every implication against the live book and logs
`WOULD ENTER` / `WOULD EXIT`, so the gates can be checked before money moves.

    CROSS_EXECUTION_ENABLED=false   # the switch
    CROSS_MAX_LOCKUP_DAYS=7         # both markets must resolve inside this
    CROSS_MIN_EDGE=0.02             # per pair, at the real asks, fees included
    CROSS_MAX_POSITION_USDC=5       # per position; NOT lower — the exchange's
                                    # 5-share minimum costs ~4.25 USDC a pair
    CROSS_MAX_POSITIONS=1           # breaker cross slots, separate from bundles
    CROSS_MAX_COMMITTED_USDC=5      # breaker cross capital ceiling
    CROSS_MIN_TIME_TO_END_S=1800    # closer than this the book is stale quotes
    CROSS_SUSPICIOUS_EDGE=0.15      # above this, a second poll must confirm it
    CROSS_SPIKE_CONFIRM_S=300       # how long that confirmation stays valid
