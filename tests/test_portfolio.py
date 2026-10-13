"""组合回测测试。

组合的核心承诺只有一条：**合起来的权益等于各子账户权益之和**。
其余都是这条的推论。另外重点测日期轴对齐 —— 两只股票的交易日不一样长是常态
（上市晚、停牌），对齐写错了总额就会凭空多出来或漏掉。
"""

from __future__ import annotations

import datetime as dt

import pytest

from ashare_bt.engine import EngineConfig
from ashare_bt.portfolio import normalize_weights, run_portfolio
from conftest import build_series


def _loader(mapping):
    return lambda code: mapping[code]


class TestWeights:
    def test_默认等权(self):
        assert normalize_weights(["a", "b", "c"], None) == pytest.approx(
            {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        )

    def test_自定义权重要归一化(self):
        assert normalize_weights(["a", "b"], [3, 1]) == pytest.approx({"a": 0.75, "b": 0.25})

    def test_权重个数对不上要报错(self):
        with pytest.raises(ValueError, match="对不上"):
            normalize_weights(["a", "b"], [0.5])

    def test_负权重要报错(self):
        with pytest.raises(ValueError, match="不能为负"):
            normalize_weights(["a", "b"], [1.0, -0.5])

    def test_权重全零要报错(self):
        with pytest.raises(ValueError, match="必须为正"):
            normalize_weights(["a", "b"], [0.0, 0.0])


class TestValidation:
    def test_空列表要报错(self):
        with pytest.raises(ValueError, match="至少要给一只"):
            run_portfolio([], "ma_cross", loader=_loader({}))

    def test_代码重复要报错(self):
        with pytest.raises(ValueError, match="重复"):
            run_portfolio(["600519", "600519"], "ma_cross", loader=_loader({}))


class TestEquitySum:
    def test_组合权益每一天都等于各子账户之和(self):
        """这是组合唯一的核心承诺，必须逐日验证，不能只看首尾。"""
        closes = [10.0 + i * 0.05 for i in range(40)]
        a = build_series(closes)
        b = build_series(closes)

        portfolio = run_portfolio(
            ["X", "Y"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"X": a, "Y": b}),
        )

        assert portfolio.initial_cash == pytest.approx(100_000.0)
        assert len(portfolio.equity) == len(portfolio.dates)

        for i, day in enumerate(portfolio.dates):
            expected = 0.0
            for code in portfolio.codes:
                result = portfolio.per_stock[code]
                last = None
                for d, value in zip(result.dates, result.equity):
                    if d > day:
                        break
                    last = value
                expected += last if last is not None else portfolio.initial_cash * portfolio.weights[code]
            assert portfolio.equity[i] == pytest.approx(expected), f"{day} 对不上"

    def test_单只股票的股组合等于它自己(self):
        closes = [10.0 + i * 0.08 for i in range(50)]
        series = build_series(closes)

        portfolio = run_portfolio(
            ["600519"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"600519": series}),
        )

        sub = portfolio.per_stock["600519"]
        assert portfolio.equity == pytest.approx(sub.equity)


class TestAlignment:
    def test_两只股票起始日期不同也能对齐(self):
        """一只从第 1 天就有数据，另一只晚 20 天上市。"""
        long_series = build_series([10.0] * 60, code="600519")
        short_series = build_series(
            [20.0] * 40, code="600519", start=dt.date(2024, 1, 30)
        )

        portfolio = run_portfolio(
            ["A", "B"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"A": long_series, "B": short_series}),
        )

        # 日期轴是两者并集
        assert len(portfolio.dates) >= len(long_series.bars)
        # B 还没上市的那些天，它贡献的是"还没动过的现金"
        assert all(v > 0 for v in portfolio.equity)
        assert len(portfolio.cash) == len(portfolio.dates)

    def test_停牌日沿用上一个权益(self):
        from ashare_bt.portfolio import _align

        d1, d2, d3 = dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)
        aligned = _align([d1, d3], [100.0, 120.0], [d1, d2, d3], fallback=50.0)
        assert aligned == [100.0, 100.0, 120.0]  # d2 停牌，沿用 d1 的值

    def test_数据开始之前用备选值(self):
        from ashare_bt.portfolio import _align

        d1, d2 = dt.date(2024, 1, 2), dt.date(2024, 1, 3)
        aligned = _align([d2], [200.0], [d1, d2], fallback=99.0)
        assert aligned == [99.0, 200.0]


class TestDiagnostics:
    def test_子账户的警告会带上来(self):
        """分到的钱买不起一手的股票，警告不能吞掉。

        否则用户只会看到"某只股票 0 次交易、0% 收益"，完全不知道为什么。
        """
        # 必须是上涨行情：横盘时均线不交叉，策略根本不会发出买入信号，
        # 也就测不到"想买但买不起"
        pricey = build_series([1000.0 + i * 20 for i in range(40)], code="600519")
        cheap = build_series([10.0 + i * 0.2 for i in range(40)], code="600519")

        portfolio = run_portfolio(
            ["PRICEY", "CHEAP"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"PRICEY": pricey, "CHEAP": cheap}),
        )

        joined = " ".join(portfolio.warnings)
        assert "买不起" in joined or "资金不足" in joined

    def test_小资金均分会被提示最低佣金(self):
        series = build_series([10.0] * 40)
        portfolio = run_portfolio(
            ["A", "B", "C"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=30_000.0),
            loader=_loader({"A": series, "B": series, "C": series}),
        )
        assert any("最低佣金" in w for w in portfolio.warnings)


class TestCombine:
    def test_合并结果能直接喂给指标模块(self):
        series = build_series([10.0 + i * 0.06 for i in range(50)])
        portfolio = run_portfolio(
            ["A", "B"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"A": series, "B": series}),
        )
        combined = portfolio.as_backtest_result("ma_cross")
        assert combined.initial_cash == pytest.approx(100_000.0)
        assert len(combined.equity) == len(portfolio.dates)
        # 成交记录应该合并了所有子账户的
        assert len(combined.fills) == sum(len(r.fills) for r in portfolio.per_stock.values())
