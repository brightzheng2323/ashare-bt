"""RSI 超买超卖策略：RSI 低于阈值买入，高于阈值卖出。

带内部状态（现在几成仓），所以必须实现 `reset()` ——
否则同一个策略对象跑第二次回测会带着上一次的仓位开局。
"""

from __future__ import annotations

from typing import Any

from ..indicators import rsi
from .base import History, Strategy


class RsiReversion(Strategy):
    name = "rsi"

    def __init__(
        self,
        period: int = 14,
        buy_below: float = 30.0,
        sell_above: float = 70.0,
    ) -> None:
        if period < 2:
            raise ValueError("period 至少为 2")
        if not 0 < buy_below < sell_above < 100:
            raise ValueError(
                f"需要满足 0 < buy_below < sell_above < 100，"
                f"当前是 buy_below={buy_below}, sell_above={sell_above}"
            )
        self.period = period
        self.buy_below = buy_below
        self.sell_above = sell_above
        self._position = 0.0

    def reset(self) -> None:
        self._position = 0.0

    @property
    def params(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "buy_below": self.buy_below,
            "sell_above": self.sell_above,
        }

    def target(self, h: History) -> float:
        value = rsi(h.close, self.period)
        if value is None:
            return self._position

        if value < self.buy_below:
            self._position = 1.0
        elif value > self.sell_above:
            self._position = 0.0

        return self._position
