"""交易费用：佣金、印花税、过户费。

三项都是真金白银的成本，长期回测里能吃掉相当一部分收益。
尤其是**单笔最低 5 元佣金** —— 小资金回测里它是最大的坑，
很多网上的示例代码直接按比例算，结果高估收益。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .models import Side


@dataclass(frozen=True)
class FeeConfig:
    """费率。默认值对应 2026 年一个普通散户的账户。"""

    commission_rate: float = 0.00025
    """佣金费率，万 2.5。券商可协商，常见区间万 1 ~ 万 3。"""

    commission_min: float = 5.0
    """单笔最低佣金 5 元。沪深两市都收；北交所没有最低佣金。"""

    stamp_duty_rate: float = 0.0005
    """印花税 0.05%，**只在卖出时收**。2023-08-28 从 0.1% 减半到 0.05%。"""

    transfer_fee_rate: float = 0.00001
    """过户费 0.001%，买卖双向都收。"""

    def buy_fee(self, amount: float) -> float:
        commission = max(amount * self.commission_rate, self.commission_min)
        return commission + amount * self.transfer_fee_rate

    def sell_fee(self, amount: float) -> float:
        commission = max(amount * self.commission_rate, self.commission_min)
        return commission + amount * self.transfer_fee_rate + amount * self.stamp_duty_rate

    def fee(self, side: Side, amount: float) -> float:
        return self.buy_fee(amount) if side is Side.BUY else self.sell_fee(amount)

    def breakdown(self, side: Side, amount: float) -> "FeeBreakdown":
        """拆开算，报告里要分别显示三项各花了多少。"""
        commission = max(amount * self.commission_rate, self.commission_min)
        transfer = amount * self.transfer_fee_rate
        stamp = amount * self.stamp_duty_rate if side is Side.SELL else 0.0
        return FeeBreakdown(commission=commission, stamp=stamp, transfer=transfer)


class FeeBreakdown(NamedTuple):
    commission: float
    stamp: float
    transfer: float

    @property
    def total(self) -> float:
        return self.commission + self.stamp + self.transfer


# 常用预设，方便对比不同券商
DEFAULT_FEES = FeeConfig()

# 万一遇到「免 5」的账户
NO_MIN_FEES = FeeConfig(commission_min=0.0)
