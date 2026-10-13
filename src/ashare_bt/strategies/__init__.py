"""策略注册表。

加自己的策略只要两步：
1. 在 strategies/ 下新建一个文件，写一个 `Strategy` 子类
2. 把它加进下面的 `REGISTRY`

之后就能用 `--strategy 你的名字` 跑了。
"""

from __future__ import annotations

import inspect
from typing import Any

from .base import History, LookAheadError, Strategy
from .bollinger import BollingerReversion
from .chip_peak import ChipPeak
from .ma_cross import MaCross
from .macd import MacdCross
from .rsi_reversion import RsiReversion

REGISTRY: dict[str, type[Strategy]] = {
    "ma_cross": MaCross,
    "macd": MacdCross,
    "bollinger": BollingerReversion,
    "rsi": RsiReversion,
    "chip_peak": ChipPeak,
}

__all__ = [
    "History",
    "LookAheadError",
    "Strategy",
    "REGISTRY",
    "available",
    "build",
    "describe_all",
]


def available() -> list[str]:
    return sorted(REGISTRY)


def accepted_params(name: str) -> set[str]:
    cls = REGISTRY[name]
    return set(inspect.signature(cls.__init__).parameters) - {"self"}


def build(name: str, **params: Any) -> Strategy:
    """按名字造一个策略。

    会检查参数名拼写 —— 命令行里把 `--fast` 打成 `--fats` 时，
    报错应该说清楚，而不是默默用默认值跑出一个看起来正常的结果。
    """
    if name not in REGISTRY:
        raise ValueError(f"没有叫 {name!r} 的策略。可用的有: {', '.join(available())}")

    accepted = accepted_params(name)
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"策略 {name} 不接受这些参数: {', '.join(sorted(unknown))}。"
            f"它接受的是: {', '.join(sorted(accepted)) or '（无参数）'}"
        )
    return REGISTRY[name](**params)


def describe_all() -> str:
    lines = []
    for name in available():
        cls = REGISTRY[name]
        doc = (cls.__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else ""
        params = ", ".join(sorted(accepted_params(name))) or "无参数"
        lines.append(f"  {name:<12} {summary}\n              参数: {params}")
    return "\n".join(lines)
