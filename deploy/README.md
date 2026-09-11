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
