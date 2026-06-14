"""Fake TelegramNotifier — captures messages and halt callback."""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Optional


class FakeTelegramNotifier:
    def __init__(self, on_status: Optional[Callable[[], dict]] = None) -> None:
        self.messages: list[str] = []
        self.alerts:   list[tuple[str, str]] = []
        self.trade_executions: list[dict] = []
        self.critical_errors:  list[str] = []
        self._on_status = on_status
        self._on_halt: Optional[Callable[[], Coroutine]] = None
        self.closed = False

    def set_halt_callback(self, cb: Callable[[], Coroutine]) -> None:
        self._on_halt = cb

    async def notify(self, text: str, *, parse_mode: Optional[str] = None) -> bool:  # noqa: ARG002
        self.messages.append(text)
        return True

    async def alert(self, title: str, body: str) -> bool:
        self.alerts.append((title, body))
        return True

    async def heartbeat(self, status: dict) -> bool:
        self.messages.append(f"heartbeat: {status}")
        return True

    async def send_trade_execution(self, **kwargs: Any) -> None:
        self.trade_executions.append(kwargs)

    def send_arb_detected(self, **kwargs: Any) -> None:
        # Mirror the real notifier: synchronous, fire-and-forget detection alert.
        if not hasattr(self, "arb_detections"):
            self.arb_detections = []
        self.arb_detections.append(kwargs)

    def send_arb_duration(self, *args: Any, **kwargs: Any) -> None:
        # Synchronous, fire-and-forget arb-window-closed alert.
        if not hasattr(self, "arb_durations"):
            self.arb_durations = []
        self.arb_durations.append((args, kwargs))

    async def send_critical_error(self, error_message: str) -> None:
        self.critical_errors.append(error_message)

    async def run_listener(self) -> None:
        import asyncio
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    async def close(self) -> None:
        self.closed = True

    async def trigger_halt(self) -> None:
        if self._on_halt:
            await self._on_halt()
