"""绩效指标测试。

这些指标每一个都有"写错了也跑得通"的坑，所以逐个钉死。
"""

from __future__ import annotations

import math

import pytest

from ashare_bt.metrics import (
    annualized_return,
    daily_returns,
    max_drawdown,
    sharpe,
    total_return,
    trade_stats,
)
from ashare_bt.models import Trade
from datetime import date


def _trade(pnl: float, ret: float, days: int = 10) -> Trade:
    return Trade(
        entry_date=date(2024, 1, 2),
        entry_price=10.0,
        exit_date=date(2024, 1, 12),
        exit_price=10.0 + pnl / 100,
        qty=100,
        fee=10.0,
        pnl=pnl,
        return_pct=ret,
        holding_days=days,
    )


class TestTotalReturn:
    def test_简单翻倍(self):
        assert total_return([100.0, 200.0]) == pytest.approx(1.0)

    def test_腰斩(self):
        assert total_return([100.0, 50.0]) == pytest.approx(-0.5)

    def test_数据不足返回0(self):
        assert total_return([100.0]) == 0.0
        assert total_return([]) == 0.0


class TestAnnualizedReturn:
    def test_两年翻倍约等于41点4(self):
        """252 个交易日算一年。两年翻倍 → 2^(1/2)-1 = 41.42%。"""
        equity = [100.0] + [150.0] * 503 + [200.0]  # 共 505 个点 = 504 段 = 2 年
        assert len(equity) - 1 == 504
        assert annualized_return(equity) == pytest.approx(0.4142, abs=0.001)

    def test_用交易日不用自然日(self):
        """252 个交易日应当算作整一年。

        用 252 折算 → 年化正好 20%；用 365 折算 → 同一段时间只有 0.69 年，
        年化被算成 30% —— 高估。这就是为什么必须用交易日。
        """
        equity = [100.0 * (1.2 ** (i / 252)) for i in range(253)]
        by_trading_days = annualized_return(equity, periods=252)
        by_calendar_days = annualized_return(equity, periods=365)

        assert by_trading_days == pytest.approx(0.20, abs=0.01)
        assert by_calendar_days > by_trading_days  # 用错基准会虚高

    def test_数据不足返回None(self):
        assert annualized_return([100.0]) is None

    def test_权益归零返回None(self):
        assert annualized_return([100.0, 0.0]) is None


class TestMaxDrawdown:
    def test_经典序列正好是25个点(self):
        """[100, 120, 90, 110]：从 120 跌到 90，回撤 = -25%。"""
        dd, peak, trough = max_drawdown([100.0, 120.0, 90.0, 110.0])
        assert dd == pytest.approx(-0.25)
        assert peak == 1
        assert trough == 2

    def test_单边上涨没有回撤(self):
        assert max_drawdown([100.0, 110.0, 120.0])[0] == 0.0

    def test_峰值会在创新高时重置(self):
        """先跌 20%，再涨过前高，之后跌 10% —— 最大回撤应该是 20% 不是 30%。"""
        equity = [100.0, 80.0, 130.0, 117.0]
        dd, _, _ = max_drawdown(equity)
        assert dd == pytest.approx(-0.20)

    def test_空序列不崩(self):
        assert max_drawdown([])[0] == 0.0


class TestSharpe:
    def test_没有波动返回None而不是除零(self):
        assert sharpe([100.0] * 50) is None

    def test_数据不足返回None(self):
        assert sharpe([100.0]) is None

    def test_稳定上涨为正(self):
        equity = [100.0 * (1.001**i) for i in range(200)]
        value = sharpe(equity)
        assert value is not None and value > 1.0

    def test_稳定下跌为负(self):
        equity = [100.0 * (0.999**i) for i in range(200)]
        value = sharpe(equity)
        assert value is not None and value < 0

    def test_年化因子是根号252(self):
        equity = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0]
        returns = daily_returns(equity)
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        expected = mean / math.sqrt(var) * math.sqrt(252)
        assert sharpe(equity) == pytest.approx(expected)


class TestTradeStats:
    def test_空交易列表不除零(self):
        stats = trade_stats([])
        assert stats["trade_count"] == 0
        assert stats["win_rate"] is None
        assert stats["profit_factor"] is None

    def test_胜率与盈亏比(self):
        trades = [_trade(100.0, 0.10), _trade(200.0, 0.20), _trade(-50.0, -0.05)]
        stats = trade_stats(trades)
        assert stats["trade_count"] == 3
        assert stats["win_count"] == 2
        assert stats["loss_count"] == 1
        assert stats["win_rate"] == pytest.approx(2 / 3)
        assert stats["profit_factor"] == pytest.approx(300.0 / 50.0)

    def test_一次都没亏时盈亏比是无穷(self):
        stats = trade_stats([_trade(100.0, 0.1), _trade(50.0, 0.05)])
        assert stats["profit_factor"] == math.inf

    def test_平局算亏不算赢(self):
        stats = trade_stats([_trade(0.0, 0.0), _trade(10.0, 0.01)])
        assert stats["win_count"] == 1
        assert stats["loss_count"] == 1

    def test_平均持仓天数(self):
        stats = trade_stats([_trade(10.0, 0.01, 5), _trade(10.0, 0.01, 15)])
        assert stats["avg_holding_days"] == pytest.approx(10.0)
