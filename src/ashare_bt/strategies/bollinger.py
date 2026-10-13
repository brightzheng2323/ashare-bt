"""布林带策略：跌破下轨买入，突破上轨卖出。

属于「均值回归」思路 —— 假设价格偏离太多会弹回来。
和趋势类策略（双均线、MACD）正好相反，放在一起跑能看出
同一个市场里两类策略的差异有多大。
"""

from __future__ import annotations

from typing import Any

from ..indicators import bollinger
from .base import History, Strategy


class BollingerReversion(Strategy):
    name = "bollinger"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        if window < 2:
            raise ValueError("window 至少为 2")
        if num_std <= 0:
            raise ValueError("num_std 必须为正数")
        self.window = window
        self.num_std = num_std
        self._position = 0.0

    def reset(self) -> None:
        self._position = 0.0

    @property
    def params(self) -> dict[str, Any]:
        return {"window": self.window, "num_std": self.num_std}

    def target(self, h: History) -> float:
        bands = bollinger(h.close, self.window, self.num_std)
        if bands is None:
            return self._position

        lower, _middle, upper = bands
        price = h.close[-1]

        if price < lower:
            self._position = 1.0  # 跌过头了，抄底
        elif price > upper:
            self._position = 0.0  # 涨过头了，离场
        # 价格在带内就维持现状，不折腾

        return self._position
