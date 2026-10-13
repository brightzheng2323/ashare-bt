"""组合回测测试。

组合的核心承诺只有一条：**合起来的权益等于各子账户权益之和**。
其余都是这条的推论。另外重点测日期轴对齐 —— 两只股票的交易日不一样长是常态
（上市晚、停牌），对齐写错了总额就会凭空多出来或漏掉。
"""

from __future__ import annotations

import datetime as dt

import pytest

from ashare_bt.engine import EngineConfig
from ashare_bt.portfolio import (
    equal_weight_buy_and_hold,
    normalize_weights,
    run_portfolio,
)
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


class TestSkippedStocks:
    """成分股里混进几只这段时间还没上市的次新股是常态。

    实测跑沪深300时，第 92 只就是这种情况 —— 单只跑不起来不能让
    整个 300 只的组合失败。
    """

    def _loader_for(self, good, late):
        return _loader({"GOOD": good, "LATE": late})

    def test_没有数据的股票被跳过而不是让组合失败(self):
        good = build_series([10.0 + i * 0.1 for i in range(40)], code="600519")
        late = build_series([20.0] * 5, code="600519", start=dt.date(2030, 1, 1))

        portfolio = run_portfolio(
            ["GOOD", "LATE"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 12, 31),
            loader=self._loader_for(good, late),
        )

        assert portfolio.codes == ["GOOD"]
        assert any("跳过" in w for w in portfolio.warnings)
        assert "LATE" in " ".join(portfolio.warnings)

    def test_跳过的股票那份钱仍算作现金(self):
        """漏掉这笔钱的话，组合本金会凭空少一块，收益率就失真了。"""
        good = build_series([10.0 + i * 0.1 for i in range(40)], code="600519")
        late = build_series([20.0] * 5, code="600519", start=dt.date(2030, 1, 1))

        portfolio = run_portfolio(
            ["GOOD", "LATE"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 12, 31),
            loader=self._loader_for(good, late),
        )

        assert portfolio.initial_cash == pytest.approx(100_000.0)
        # 全都没开始交易之前，权益就等于本金
        assert portfolio.equity[0] == pytest.approx(100_000.0)

    def test_全部跑不起来时才报错(self):
        late = build_series([20.0] * 5, code="600519", start=dt.date(2030, 1, 1))
        with pytest.raises(ValueError, match="都没能跑起来"):
            run_portfolio(
                ["A", "B"],
                "ma_cross",
                params={"fast": 3, "slow": 10},
                config=EngineConfig(initial_cash=100_000.0),
                start=dt.date(2024, 1, 1),
                end=dt.date(2024, 12, 31),
                loader=_loader({"A": late, "B": late}),
            )

    def test_取数失败的股票也被跳过(self):
        good = build_series([10.0 + i * 0.1 for i in range(40)], code="600519")

        def loader(code):
            if code == "BROKEN":
                raise RuntimeError("网络断了")
            return good

        portfolio = run_portfolio(
            ["GOOD", "BROKEN"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=loader,
        )
        assert portfolio.codes == ["GOOD"]
        assert any("取不到行情" in w for w in portfolio.warnings)


class TestEqualWeightBuyAndHold:
    """组合回测最该对比的基准。只看「策略 vs 指数」会把选股偏差的功劳
    错记在策略头上 —— 实测沪深300成分股等权买入持有 +82%，而同一批股票
    跑双均线只有 +32%。"""

    def _portfolio(self, stocks):
        good = build_series([10.0 + i * 0.1 for i in range(40)], code="600519")
        loader = _loader({code: good for code in stocks})
        return run_portfolio(
            list(stocks),
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=loader,
        )

    def test_横盘时等于本金(self):
        flat = build_series([10.0] * 40, code="600519")
        portfolio = run_portfolio(
            ["A", "B"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"A": flat, "B": flat}),
        )
        curve = equal_weight_buy_and_hold(portfolio, _loader({"A": flat, "B": flat}))
        assert curve[-1] == pytest.approx(100_000.0)

    def test_上涨时赚钱(self):
        up = build_series([10.0 * (1.01**i) for i in range(40)], code="600519")
        portfolio = run_portfolio(
            ["A"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"A": up}),
        )
        curve = equal_weight_buy_and_hold(portfolio, _loader({"A": up}))
        assert curve[-1] > 100_000.0

    def test_曲线长度与组合日期轴一致(self):
        up = build_series([10.0 + i * 0.5 for i in range(40)], code="600519")
        portfolio = run_portfolio(
            ["A"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=_loader({"A": up}),
        )
        curve = equal_weight_buy_and_hold(portfolio, _loader({"A": up}))
        assert len(curve) == len(portfolio.dates)

    def test_取不到数据的股票被跳过(self):
        up = build_series([10.0 + i * 0.5 for i in range(40)], code="600519")

        def loader(code):
            if code == "BROKEN":
                raise RuntimeError("网络断了")
            return up

        portfolio = run_portfolio(
            ["GOOD", "BROKEN"],
            "ma_cross",
            params={"fast": 3, "slow": 10},
            config=EngineConfig(initial_cash=100_000.0),
            loader=loader,
        )
        curve = equal_weight_buy_and_hold(portfolio, loader)
        assert len(curve) == len(portfolio.dates)

    def test_买入持有在单边上涨时应跑赢双均线(self):
        """这是加了这条基准之后立刻能看出来的事。

        一路上涨的行情里，任何趋势策略都会被来回止损磨掉一部分收益，
        买入持有才是最优。如果这条测试红了，说明策略或基准有一个算错了。
        """
        up = build_series([10.0 * (1.008**i) for i in range(300)], code="600519")
        loader = _loader({"A": up, "B": up})
        portfolio = run_portfolio(
            ["A", "B"],
            "ma_cross",
            params={"fast": 5, "slow": 20},
            config=EngineConfig(initial_cash=200_000.0),
            loader=loader,
        )
        hold = equal_weight_buy_and_hold(portfolio, loader)
        strategy_return = portfolio.equity[-1] / portfolio.initial_cash - 1
        hold_return = hold[-1] / portfolio.initial_cash - 1
        assert hold_return > strategy_return


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
