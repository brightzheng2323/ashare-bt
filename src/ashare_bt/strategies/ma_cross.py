"""双均线策略：金叉满仓，死叉空仓。

最经典的入门策略，也最适合用来验证引擎本身对不对 ——
它的买卖点少、好手工核对，跑出来的结果一眼能看出是不是反了。
"""

from __future__ import annotations

from typing import Any

from ..indicators import sma
from .base import History, Strategy


class MaCross(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("均线周期必须是正整数")
        if fast >= slow:
            raise ValueError(f"fast({fast}) 必须小于 slow({slow})")
        self.fast = fast
        self.slow = slow

    @property
    def params(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow}

    def target(self, h: History) -> float:
        fast = sma(h.close, self.fast)
        slow = sma(h.close, self.slow)

        # 注意用 `is None` 而不是 `if not fast` —— 价格可以是 0.0，
        # 而 0.0 在 Python 里是假值，用真值判断会把合法数据当成缺失。
        if fast is None or slow is None:
            return 0.0

        return 1.0 if fast > slow else 0.0
