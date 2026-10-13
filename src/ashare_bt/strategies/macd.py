"""MACD 策略：DIF 在 DEA 上方满仓，下方空仓。

比双均线平滑一些，信号更少、更滞后。
"""

from __future__ import annotations

from typing import Any

from ..indicators import macd
from .base import History, Strategy


class MacdCross(Strategy):
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if not 0 < fast < slow:
            raise ValueError("必须满足 0 < fast < slow")
        if signal < 1:
            raise ValueError("signal 必须是正整数")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def params(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal}

    def target(self, h: History) -> float:
        values = macd(h.close, self.fast, self.slow, self.signal)
        if values is None:
            return 0.0
        dif, dea, _hist = values
        return 1.0 if dif > dea else 0.0
