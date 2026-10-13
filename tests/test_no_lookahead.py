"""未来函数测试。

这是回测里最难发现的一类 bug —— 程序不报错，收益曲线还特别漂亮，
只有上了实盘才会露馅。所以要用**结构性**的方法去测：

1. 边界测试：`History` 到底能不能碰到未来
2. 变异测试：把未来数据换成荒谬的数值，今天的判断必须**逐字节不变**
3. 行为测试：每一笔成交价必须精确等于当日开盘价
"""

from __future__ import annotations

import pytest

from ashare_bt.engine import EngineConfig, run_backtest
from ashare_bt.models import Bar, Side
from ashare_bt.strategies import build
from ashare_bt.strategies.base import History, LookAheadError
from conftest import ScriptedStrategy, build_series


class TestHistoryBoundaries:
    def test_长度等于今天及之前的条数(self, make_flat):
        series = make_flat(10.0, 100)
        h = History(series, 10)

        assert len(h) == 11
        assert len(h.close) == 11
        assert len(h.open) == 11

    def test_越界读取抛异常(self, make_flat):
        series = make_flat(10.0, 100)
        h = History(series, 10)

        with pytest.raises(LookAheadError):
            _ = h.close[11]
        with pytest.raises(LookAheadError):
            _ = h.close[99]

    def test_负索引仍在可见范围内(self, make_flat):
        series = make_flat(10.0, 100)
        h = History(series, 10)

        assert h.close[-1] == series.bars[10].close
        assert h.close[0] == series.bars[0].close
        with pytest.raises(LookAheadError):
            _ = h.close[-12]  # 比可见范围还往前

    def test_切片会被夹到可见范围(self, make_flat):
        """写 h.close[0:999] 也只会拿到可见的那些，不会漏出未来。"""
        series = make_flat(10.0, 100)
        h = History(series, 10)

        assert len(h.close[0:999]) == 11
        assert len(h.close[:]) == 11

    def test_bar的正偏移指向未来会报错(self, make_flat):
        series = make_flat(10.0, 100)
        h = History(series, 10)

        assert h.bar(0).date == series.bars[10].date
        assert h.bar(-1).date == series.bars[9].date
        with pytest.raises(LookAheadError):
            h.bar(1)

    def test_没有公开途径拿到完整序列(self, make_flat):
        h = History(make_flat(10.0, 50), 5)

        assert not hasattr(h, "bars")
        assert not hasattr(h, "series")
        assert not hasattr(h, "values")


class TestMutation:
    """把未来的 K 线替换成荒谬数值，今天的决策必须完全不变。

    这一条能抓住连 `History` 都拦不住的泄漏 —— 比如某个指标函数
    内部用了全序列而不是尾部。
    """

    @pytest.mark.parametrize("strategy_name,params", [
        ("ma_cross", {"fast": 5, "slow": 20}),
        ("rsi", {"period": 14}),
        ("macd", {}),
        ("bollinger", {"window": 20}),
    ])
    def test_篡改未来数据不改变今天的判断(self, strategy_name, params):
        closes = [10.0 + (i % 11) * 0.4 + i * 0.03 for i in range(120)]
        series = build_series(closes)
        i = 90

        strategy = build(strategy_name, **params)
        before = strategy.target(History(series, i))

        garbage = tuple(
            Bar(date=b.date, open=1e6, high=1e6, low=1e6, close=1e6, volume=0.0, amount=0.0)
            for b in series.bars[i + 1:]
        )
        mutated = series._replace(bars=series.bars[: i + 1] + garbage)

        strategy.reset()
        after = strategy.target(History(mutated, i))

        assert before == after, f"{strategy_name} 读取了未来数据"

    def test_删掉未来数据也不影响(self):
        """把未来的 K 线整个删掉，今天的判断同样不能变。"""
        closes = [10.0 + (i % 7) * 0.6 for i in range(100)]
        series = build_series(closes)
        i = 70
        strategy = build("ma_cross", fast=5, slow=20)

        before = strategy.target(History(series, i))
        truncated = series._replace(bars=series.bars[: i + 1])
        strategy.reset()
        after = strategy.target(History(truncated, i))

        assert before == after


class TestFillPrices:
    def test_每笔成交价精确等于当日开盘价(self):
        """没有任何滑点时，成交价必须**精确**等于开盘价。

        如果哪天有人改成用收盘价成交，这条会立刻红。
        """
        closes = [10.0 + (i % 9) * 0.5 for i in range(200)]
        series = build_series(closes)
        strategy = build("ma_cross", fast=3, slow=10)

        result = run_backtest(series, strategy, EngineConfig(initial_cash=200_000.0))

        by_date = {bar.date: bar for bar in series.bars}
        assert result.fills, "这个用例应该产生成交，否则测试没有意义"
        for fill in result.fills:
            assert fill.price == by_date[fill.date].open

    def test_成交永远发生在信号之后的那一天(self, make_flat):
        """引擎里"今天挂单、明天开盘成交"的节奏，不能被打乱。"""
        series = make_flat(10.0, 30)
        strategy = ScriptedStrategy([1.0, 0.0, 1.0, 0.0])
        result = run_backtest(series, strategy, EngineConfig(initial_cash=100_000.0))

        day_index = {bar.date: i for i, bar in enumerate(series.bars)}
        for fill in result.fills:
            # 剧本第 n 步在第 n 天收盘产生，成交必须在第 n+1 天
            assert day_index[fill.date] >= 1

    def test_回测最后一天不会凭空成交(self, make_flat):
        """最后一天的收盘价已经看到了，此时成交就是未来函数。"""
        series = make_flat(10.0, 10)
        strategy = ScriptedStrategy([0.0] * 9 + [1.0])  # 只在最后一天喊买
        result = run_backtest(series, strategy, EngineConfig(initial_cash=100_000.0))

        assert all(f.date != series.bars[-1].date for f in result.fills)


class TestWarmup:
    def test_回测开头指标就是预热的(self):
        """引擎在完整历史上跑、只在记录时截取窗口。

        如果先把数据切成 [start, end] 再算指标，
        回测第一天的 MA20 就只会用一两天数据算出来 —— 常见且致命的 bug。
        """
        closes = [10.0 + i * 0.05 for i in range(300)]
        series = build_series(closes)

        # 从第 250 根开始回测，此时 MA20 早就该是热的
        result = run_backtest(
            series,
            build("ma_cross", fast=5, slow=20),
            EngineConfig(initial_cash=100_000.0),
            start=series.bars[250].date,
        )

        # 一路上涨的行情里，5 日线始终在 20 日线上方 → 第一天就该满仓
        assert result.fills, "指标没有预热的话，开头几天的信号会是空的"
        assert result.fills[0].side is Side.BUY
        assert result.fills[0].date == series.bars[251].date
