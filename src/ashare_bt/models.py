"""数据模型。

整个项目传来传去的都是这里的小对象。全部用不可变类型，
避免"某处偷偷改了价格"这类难查的 bug。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import NamedTuple


class Bar(NamedTuple):
    """一根日 K 线。"""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float  # 成交量，单位：手（1 手 = 100 股）
    amount: float  # 成交额，单位：元
    turnover: float = 0.0  # 换手率 %。筹码分布要用它，所以必须存


class Series(NamedTuple):
    """一只股票的完整历史日线。

    故意不可变：引擎在回测过程中绝不会修改行情数据。
    """

    code: str  # 纯代码，如 "600519"
    secid: str  # 带市场前缀，如 "1.600519"
    name: str  # 股票名称。ST 判定必须要它，光看代码分不出来
    bars: tuple[Bar, ...]  # 按日期升序，含上市首日
    fqt: int  # 复权方式：0 不复权 / 1 前复权 / 2 后复权
    fetched_at: datetime
    dropped_bars: int = 0  # 因前复权算出负价而被丢弃的前导 K 线数

    def index_of(self, d: date) -> int | None:
        """返回该日期在 bars 中的下标；不是交易日则返回 None。

        二分查找。用 None 而不是抛异常，是因为"这天没交易"
        （停牌、节假日）在回测里是正常情况，不是错误。
        """
        lo, hi = 0, len(self.bars) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            bar_date = self.bars[mid].date
            if bar_date < d:
                lo = mid + 1
            elif bar_date > d:
                hi = mid - 1
            else:
                return mid
        return None

    def index_or_after(self, d: date) -> int | None:
        """返回第一个 >= d 的交易日下标。用于把区间起点对齐到交易日。"""
        lo, hi = 0, len(self.bars)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.bars[mid].date < d:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < len(self.bars) else None

    def index_or_before(self, d: date) -> int | None:
        """返回最后一个 <= d 的交易日下标。"""
        idx = self.index_or_after(d)
        if idx is not None and self.bars[idx].date == d:
            return idx
        if idx is None:
            return len(self.bars) - 1
        return idx - 1 if idx > 0 else None

    @property
    def first_date(self) -> date:
        return self.bars[0].date

    @property
    def last_date(self) -> date:
        return self.bars[-1].date


@dataclass
class Position:
    """持仓。

    total 和 sellable 分开存，就是为了实现 T+1：
    今天买的股票进 total，但不进 sellable，所以今天卖不掉。
    """

    total: int = 0
    sellable: int = 0

    def settle(self) -> None:
        """每个交易日开盘前调用：昨天买的今天变成可卖。"""
        self.sellable = self.total

    def add_buy(self, qty: int) -> None:
        """买入。只加 total —— sellable 要等明天的 settle()。"""
        self.total += qty

    def apply_sell(self, qty: int) -> None:
        self.total -= qty
        self.sellable -= qty


class Side(Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def label(self) -> str:
        return "买入" if self is Side.BUY else "卖出"


class RejectReason(Enum):
    """订单被拒的原因。全部记录下来，报告里能看到。"""

    LIMIT_UP = "limit_up"  # 涨停，买不进
    LIMIT_DOWN = "limit_down"  # 跌停，卖不出
    T1 = "t1"  # 当日买入的还没结算
    CASH = "cash"  # 资金不够
    LOT = "lot"  # 不够一个整手

    @property
    def label(self) -> str:
        return {
            RejectReason.LIMIT_UP: "涨停无法买入",
            RejectReason.LIMIT_DOWN: "跌停无法卖出",
            RejectReason.T1: "T+1 当日不可卖",
            RejectReason.CASH: "资金不足",
            RejectReason.LOT: "不足一手",
        }[self]


@dataclass(frozen=True)
class Order:
    """挂在"明天开盘"执行的订单。"""

    side: Side
    qty: int
    created_date: date


@dataclass(frozen=True)
class Fill:
    """一笔成交。"""

    date: date
    side: Side
    qty: int
    price: float
    amount: float  # 成交金额，不含费用
    fee: float  # 总费用


@dataclass(frozen=True)
class Rejection:
    """一笔被拒的订单。"""

    date: date
    side: Side
    qty: int
    reason: RejectReason


@dataclass(frozen=True)
class Trade:
    """一次完整的「买入 → 卖出」往返。"""

    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    qty: int
    fee: float  # 买卖两边的总费用
    pnl: float  # 净盈亏（已扣费用）
    return_pct: float  # 净收益率
    holding_days: int


@dataclass
class BacktestResult:
    """回测的完整结果。图表、指标、报告都从这里取数。"""

    code: str
    name: str
    strategy_name: str
    dates: list[date] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    cash: list[float] = field(default_factory=list)
    position_value: list[float] = field(default_factory=list)
    position_pct: list[float] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    initial_cash: float = 0.0
    open_qty: int = 0  # 期末仍未平仓的股数
    open_cost: float = 0.0  # 期末未平仓部分的成本
    warnings: list[str] = field(default_factory=list)  # 值得提醒用户的异常情况
