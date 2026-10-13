"""筹码峰策略。

思路和均线类策略**完全不同**：均线看的是价格走势，筹码看的是**持仓成本结构**。

- 获利盘很少（绝大多数人都在亏）→ 底部区域，往下抛压小，买入
- 获利盘很多（绝大多数人都赚了）→ 高位派发区，随时可能有人兑现，卖出

再加一道**集中度**过滤：筹码太分散（成本区间很宽）说明多空分歧大、没有形成
有效底部，这时候不买。
"""

from __future__ import annotations

from typing import Any

from ..chips import ChipDistribution
from .base import History, Strategy


class ChipPeak(Strategy):
    name = "chip_peak"

    def __init__(
        self,
        bins: int = 400,
        buy_below: float = 20.0,
        sell_above: float = 80.0,
        max_concentration: float = 0.40,
        decay: float = 1.0,
    ) -> None:
        """
        Args:
            bins: 价格网格的格子数，越大越精细
            buy_below: 获利盘低于这个百分比（%）时买入
            sell_above: 获利盘高于这个百分比（%）时卖出
            max_concentration: 集中度上限，超过就不买（筹码太散）
            decay: 换手衰减系数，1.0 表示完全按真实换手率衰减
        """
        if not 0 < buy_below < sell_above < 100:
            raise ValueError(
                f"需要满足 0 < buy_below < sell_above < 100，"
                f"当前是 buy_below={buy_below}, sell_above={sell_above}"
            )
        if max_concentration <= 0:
            raise ValueError("max_concentration 必须为正数")

        self.bins = bins
        self.buy_below = buy_below
        self.sell_above = sell_above
        self.max_concentration = max_concentration
        self.decay = decay

        self._chips = ChipDistribution(bins, decay)
        self._last_index = -1
        self._position = 0.0

    def reset(self) -> None:
        self._chips = ChipDistribution(self.bins, self.decay)
        self._last_index = -1
        self._position = 0.0

    @property
    def params(self) -> dict[str, Any]:
        return {
            "bins": self.bins,
            "buy_below": self.buy_below,
            "sell_above": self.sell_above,
            "max_concentration": self.max_concentration,
        }

    def target(self, h: History) -> float:
        self._advance(h)

        # 换手率缺失（数据源没给）时分布是空的，此时不交易，避免凭空满仓
        if self._chips.total_weight <= 0:
            return self._position

        state = self._chips.state(h.close[-1])
        profit_pct = state.profit_ratio * 100.0

        if profit_pct < self.buy_below and state.concentration <= self.max_concentration:
            self._position = 1.0
        elif profit_pct > self.sell_above:
            self._position = 0.0

        return self._position

    def _advance(self, h: History) -> None:
        """把筹码分布推进到 h.index 这一天。

        引擎是逐日调用 target() 的，所以这里只需要处理新增的那几天 ——
        每天的工作量只和当天的价格区间有关，与历史长度无关。
        """
        if h.index <= self._last_index:
            # 索引倒退了（正常回测不会发生），保险起见从头重建
            self._chips = ChipDistribution(self.bins, self.decay)
            self._last_index = -1

        for j in range(self._last_index + 1, h.index + 1):
            high = h.high[j]
            low = h.low[j]
            close = h.close[j]
            typical_price = (high + low + close) / 3.0
            self._chips.update(high, low, typical_price, h.turnover[j])

        self._last_index = h.index
