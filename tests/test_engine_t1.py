"""T+1 规则测试。

T+1 是 A 股和美股最大的差别之一：**今天买的股票今天不能卖**。
回测里忽略它，会让换手频繁一点的策略收益虚高 —— 因为现实中那些
"当天冲高就跑"的操作根本做不出来。
"""

from __future__ import annotations

from ashare_bt.engine import EngineConfig, _Account, execute_order, run_backtest
from ashare_bt.models import Fill, Order, Position, RejectReason, Rejection, Side
from conftest import ScriptedStrategy


def _buy(series, account, i: int, qty: int, config: EngineConfig | None = None):
    bar = series.bars[i]
    order = Order(Side.BUY, qty, series.bars[i - 1].date)
    return execute_order(order, bar, series.bars[i - 1].close, series, account, config or EngineConfig(), i)


def _sell(series, account, i: int, qty: int, config: EngineConfig | None = None):
    bar = series.bars[i]
    order = Order(Side.SELL, qty, series.bars[i - 1].date)
    return execute_order(order, bar, series.bars[i - 1].close, series, account, config or EngineConfig(), i)


class TestSettleMechanics:
    def test_买入后总数增加但可卖仍为零(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0)
        outcome = _buy(series, account, 1, 1000)

        assert isinstance(outcome, Fill)
        assert account.position.total == 1000
        assert account.position.sellable == 0  # ← T+1 的全部含义

    def test_当日买入当日卖出被拒(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0)
        _buy(series, account, 1, 1000)

        outcome = _sell(series, account, 1, 1000)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.T1

    def test_结算之后就能卖(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0)
        _buy(series, account, 1, 1000)

        account.position.settle()  # 第二天开盘前
        outcome = _sell(series, account, 2, 1000)

        assert isinstance(outcome, Fill)
        assert account.position.total == 0
        assert account.position.sellable == 0

    def test_结算之后可卖等于总持仓(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0)
        _buy(series, account, 1, 1000)

        account.position.settle()

        assert account.position.sellable == account.position.total == 1000

    def test_卖出超过可卖数量会被拒(self, make_flat):
        """手里有 1000 股但只有 600 股可卖时，想卖 1000 股要被拒。

        这和真实券商一致：可卖不足直接拒单，不会"能卖多少卖多少"。
        否则回测会凭空多出一些现实中根本下不出去的单子。
        """
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0, position=Position(total=1000, sellable=600))

        outcome = _sell(series, account, 1, 1000)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.T1
        assert account.position.total == 1000  # 一股没动

    def test_卖可卖数量以内可以成交(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0, position=Position(total=1000, sellable=600))

        outcome = _sell(series, account, 1, 600)

        assert isinstance(outcome, Fill)
        assert account.position.total == 400
        assert account.position.sellable == 0


class TestT1InFullLoop:
    def test_收盘出信号次日开盘才成交(self, make_flat):
        """信号和成交必须差一天 —— 这条是整个引擎的核心承诺。"""
        series = make_flat(10.0, 6)
        # 第 0 天就想满仓；引擎只会把单子挂到第 1 天开盘
        strategy = ScriptedStrategy([1.0])
        result = run_backtest(series, strategy, EngineConfig(initial_cash=100_000.0))

        assert len(result.fills) == 1
        buy = result.fills[0]
        assert buy.side is Side.BUY
        assert buy.date == series.bars[1].date  # 不是第 0 天
        assert buy.price == series.bars[1].open

    def test_当天买当天卖在完整循环里不会发生(self, make_flat):
        """剧本是"立刻满仓然后立刻空仓"，但引擎的节奏决定了最早也要隔一天才能卖。"""
        series = make_flat(10.0, 6)
        strategy = ScriptedStrategy([1.0, 0.0])
        result = run_backtest(series, strategy, EngineConfig(initial_cash=100_000.0))

        buys = [f for f in result.fills if f.side is Side.BUY]
        sells = [f for f in result.fills if f.side is Side.SELL]
        assert len(buys) == 1
        if sells:
            assert sells[0].date > buys[0].date

    def test_买入那天持仓不可能是负的(self, make_flat):
        series = make_flat(10.0, 10)
        strategy = ScriptedStrategy([1.0, 0.0, 1.0, 0.0])
        result = run_backtest(series, strategy, EngineConfig(initial_cash=100_000.0))

        assert all(v >= 0 for v in result.position_value)
