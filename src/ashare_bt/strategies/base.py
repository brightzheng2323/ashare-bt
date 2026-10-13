"""策略基类，以及防未来函数的 `History` 视图。

这是整个项目最重要的一个设计。

**问题：** 回测里最常见、最难发现的 bug 是「未来函数」—— 策略在算今天该不该
买的时候，不小心用了明天的价格。程序不报错，收益曲线还特别漂亮，一上实盘就崩。

**办法：** 不给策略传行情数组，只传一个 `History` 对象。这个对象
- 只暴露到「今天」为止的数据；
- 访问越界直接抛 `LookAheadError`；
- 没有任何公开途径能拿到完整序列。

诚实地说：`History` 内部仍然持有 `Series` 的引用（叫 `_series`），
一个铁了心的策略可以写 `h._series.bars` 绕过去。要做到物理上不可能，
得把数据复制一份，代价不值得。这里的定位是：
**误用会被拦住，主动越狱得写得很刻意，而刻意的越狱在 code review 里一眼就能看见。**
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from typing import Any

from ..models import Bar, Series


class LookAheadError(RuntimeError):
    """策略试图读取尚未发生的行情。

    看到这个异常说明策略里有未来函数，必须修掉 —— 不要改成 try/except 绕过去。
    """


class _FieldView(Sequence):
    """某一行情字段（如收盘价）截至今日的视图。

    支持下标和切片，行为跟普通列表一样，唯一的区别是**够不到未来**：

        h.close[-1]    # 今天收盘价
        h.close[0]     # 回测可见的第一天
        h.close[999]   # 如果 999 还没到 → LookAheadError
    """

    __slots__ = ("_series", "_i", "_field")

    def __init__(self, series: Series, i: int, field: str) -> None:
        self._series = series
        self._i = i
        self._field = field

    def __len__(self) -> int:
        # 就是这一行：长度等于「到今天为止」的条数
        return self._i + 1

    def __getitem__(self, key: int | slice) -> Any:
        n = len(self)
        if isinstance(key, slice):
            # 切片越界会被 clamp 到 n，所以不可能借此摸到未来
            start, stop, step = key.indices(n)
            return [self[k] for k in range(start, stop, step or 1)]

        idx = key + n if key < 0 else key
        if not 0 <= idx < n:
            raise LookAheadError(
                f"策略试图读取第 {key} 个 {self._field}（共可见 {n} 个）。"
                f"这天还没到 —— 这就是未来函数。"
            )
        return getattr(self._series.bars[idx], self._field)

    def __iter__(self):
        for k in range(len(self)):
            yield getattr(self._series.bars[k], self._field)

    def tolist(self) -> list[float]:
        return list(self)

    def __repr__(self) -> str:
        return f"<{self._field} 到 {self._series.bars[self._i].date}，共 {len(self)} 个>"


class History:
    """策略能看到的全部数据：当前交易日及之前。

    刻意**不提供** `.bars` 属性，也不提供底层 `Series` —— 未来数据在这个对象上
    不可达。想要完整序列请用切片 + 边界检查，或者承认自己需要未来数据。
    """

    __slots__ = ("_series", "_i")

    def __init__(self, series: Series, i: int) -> None:
        if not 0 <= i < len(series.bars):
            raise IndexError(f"下标 {i} 超出范围（共 {len(series.bars)} 根 K 线）")
        self._series = series
        self._i = i

    # ── 长度与位置 ──────────────────────────────────────────────
    def __len__(self) -> int:
        """可见的 K 线根数 —— 永远等于「今天」的序号 + 1。"""
        return self._i + 1

    @property
    def index(self) -> int:
        """今天在整个序列里的下标。"""
        return self._i

    @property
    def date(self) -> date:
        """今天的日期。"""
        return self._series.bars[self._i].date

    @property
    def code(self) -> str:
        return self._series.code

    @property
    def name(self) -> str:
        return self._series.name

    @property
    def bars_since_listing(self) -> int:
        """距上市首日有多少根 K 线。用于判断新股不限价期。"""
        return self._i

    def bar(self, offset: int = 0) -> Bar:
        """取某一根 K 线。offset=0 是今天，-1 是昨天，以此类推。

        注意 offset 是「相对今天往回数」，所以负数才是历史；
        正数指向未来，直接报错。
        """
        idx = self._i + offset
        if not 0 <= idx <= self._i:
            raise LookAheadError(
                f"策略想取 offset={offset} 的 K 线，但今天才是第 {self._i} 根 —— "
                f"正数 offset 指向未来，不允许。"
            )
        return self._series.bars[idx]

    # ── 字段视图 ────────────────────────────────────────────────
    @property
    def open(self) -> _FieldView:
        return _FieldView(self._series, self._i, "open")

    @property
    def high(self) -> _FieldView:
        return _FieldView(self._series, self._i, "high")

    @property
    def low(self) -> _FieldView:
        return _FieldView(self._series, self._i, "low")

    @property
    def close(self) -> _FieldView:
        return _FieldView(self._series, self._i, "close")

    @property
    def volume(self) -> _FieldView:
        return _FieldView(self._series, self._i, "volume")

    @property
    def amount(self) -> _FieldView:
        return _FieldView(self._series, self._i, "amount")

    @property
    def turnover(self) -> _FieldView:
        return _FieldView(self._series, self._i, "turnover")

    def __repr__(self) -> str:
        return f"<History {self._series.code} 截至 {self.date}，共 {len(self)} 根>"


class Strategy(ABC):
    """策略基类。

    子类只需要实现 `target()`：看一眼历史，说出想拿多少仓位。

    用**目标仓位**而不是买卖信号，是为了处理「涨停买不进」这种情况：
    目标还是 1.0，明天引擎会接着试，不会因为一次失败就把状态丢了。
    """

    name: str = "base"

    def reset(self) -> None:
        """引擎在每次回测开始前调用。

        有内部状态的策略（比如"现在满仓吗"）必须重写这个方法，
        否则同一个策略对象跑第二遍会得到不同的结果。
        """
        return None

    @abstractmethod
    def target(self, h: History) -> float:
        """返回目标仓位。

        Args:
            h: 只包含今天及之前数据的视图。

        Returns:
            0.0 = 空仓，1.0 = 满仓。（当前版本只支持这两个值。）
        """

    @property
    def params(self) -> dict[str, Any]:
        """策略参数，报告里会打印出来。子类可重写。"""
        return {}

    def describe(self) -> str:
        if not self.params:
            return self.name
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({inner})"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"
