"""测试用的公共工具。

核心是 `build_series`：手工造一段行情，完全不联网、完全确定。
用真实数据做断言很脆弱（数据会变、网络会断），
用造出来的数据才能精确地卡住边界条件。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pytest

from ashare_bt.fetch import DEFAULT_FQT
from ashare_bt.models import Bar, Series
from ashare_bt.strategies.base import History, Strategy


def weekdays(n: int, start: dt.date = dt.date(2024, 1, 2)) -> list[dt.date]:
    """生成 n 个连续工作日。

    只跳周末，不跳节假日 —— 测试里"哪天是交易日"是我们自己说了算，
    引入真实节假日反而让断言变复杂。
    """
    days: list[dt.date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def build_series(
    closes: Sequence[float],
    *,
    opens: Sequence[float] | None = None,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    volumes: Sequence[float] | None = None,
    turnovers: Sequence[float] | None = None,
    code: str = "600519",
    name: str = "测试股",
    fqt: int = DEFAULT_FQT,
    start: dt.date = dt.date(2024, 1, 2),
) -> Series:
    """用一串收盘价造一个 Series。其余字段不指定就自动推导。

    默认让 open = 昨收（或首日 = 当日收盘），这样"按开盘价成交"的断言
    可以直接拿收盘价序列来对照，写测试时少一层心智负担。
    """
    n = len(closes)
    if n == 0:
        raise ValueError("closes 不能为空")

    def pick(seq: Sequence[float] | None, i: int, fallback: float) -> float:
        return fallback if seq is None else seq[i]

    dates = weekdays(n, start)
    bars: list[Bar] = []
    for i, close in enumerate(closes):
        default_open = closes[i - 1] if i > 0 else close
        open_ = pick(opens, i, default_open)
        high = pick(highs, i, max(open_, close))
        low = pick(lows, i, min(open_, close))
        bars.append(
            Bar(
                date=dates[i],
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=pick(volumes, i, 10000.0),
                amount=pick(volumes, i, 10000.0) * close * 100,
                turnover=pick(turnovers, i, 1.0),
            )
        )

    return Series(
        code=code,
        secid=f"1.{code}",
        name=name,
        bars=tuple(bars),
        fqt=fqt,
        fetched_at=dt.datetime(2024, 1, 1, 12, 0, 0),
    )


def flat_series(price: float, n: int, **kwargs) -> Series:
    """一条横盘直线。用来隔离费用、T+1 这类和价格波动无关的行为。"""
    return build_series([price] * n, **kwargs)


class ScriptedStrategy(Strategy):
    """按剧本给出目标仓位。

    用真实策略（双均线之类）测引擎很难精确控制"哪天买、哪天卖"——
    要反推出一段能让均线恰好交叉的价格序列，测试会变得又长又脆。
    直接把买卖时机写死，测的就纯粹是引擎行为。
    """

    name = "scripted"

    def __init__(self, targets: Sequence[float]) -> None:
        if not targets:
            raise ValueError("targets 不能为空")
        self._script = list(targets)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    @property
    def params(self):
        return {"script": self._script}

    def target(self, h: History) -> float:
        if self._step >= len(self._script):
            return self._script[-1]  # 剧本演完了就一直保持最后一幕
        value = self._script[self._step]
        self._step += 1
        return value


@pytest.fixture
def make_series():
    return build_series


@pytest.fixture
def make_flat():
    return flat_series
