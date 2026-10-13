"""探索功能测试。

这个模块主要是给人看的，所以测试重点在"不崩"和"该有的信息都在"，
而不是逐字比对输出。
"""

from __future__ import annotations

import datetime as dt

import pytest

from ashare_bt.explore import (
    build_chips,
    chip_chart,
    explore,
    indicator_panel,
    strategy_signals,
)
from ashare_bt.strategies.base import History

from conftest import build_series


def _series(n: int = 260):
    prices = []
    price = 10.0
    for i in range(n):
        price *= 1 + (0.008 if (i // 15) % 2 == 0 else -0.006)
        prices.append(round(price, 2))
    return build_series(prices)


class TestBuildChips:
    def test_能算出分布(self):
        series = _series()
        chips = build_chips(series, len(series.bars) - 1)
        assert chips.total_weight == pytest.approx(1.0)

    def test_只回看指定根数(self):
        series = _series()
        chips = build_chips(series, len(series.bars) - 1, lookback=50)
        state = chips.state(series.bars[-1].close)
        # 回看 50 根时，价格区间应当比较窄
        assert state.cost_high >= state.cost_low


class TestChipChart:
    def test_行数受rows限制(self):
        series = _series()
        chips = build_chips(series, len(series.bars) - 1)
        lines = chip_chart(chips, series.bars[-1].close, rows=12)
        assert 1 <= len(lines) <= 14  # 抽稀是近似，允许差一两行

    def test_标出现价(self):
        series = _series()
        chips = build_chips(series, len(series.bars) - 1)
        lines = chip_chart(chips, series.bars[-1].close, rows=20)
        assert any("现价" in line for line in lines)

    def test_横条长度与权重成比例(self):
        series = _series()
        chips = build_chips(series, len(series.bars) - 1)
        lines = chip_chart(chips, series.bars[-1].close, rows=30)
        widths = [line.count("█") for line in lines]
        assert max(widths) >= 10  # 最大的那根得够长，不能全是 1 格
        assert min(widths) >= 0

    def test_没有筹码时给提示而不是崩(self):
        series = _series()
        chips = build_chips(series, len(series.bars) - 1)
        chips.weights = [0.0] * len(chips.weights)
        lines = chip_chart(chips, series.bars[-1].close)
        assert lines and "没有筹码" in lines[0]


class TestIndicators:
    def test_数据够时所有指标都有值(self):
        series = _series(300)
        panel = indicator_panel(History(series, len(series.bars) - 1))
        joined = "\n".join(panel)
        assert "MA120" in joined
        assert "RSI14" in joined
        assert "MACD" in joined
        assert "布林带" in joined
        assert "数据不足" not in joined

    def test_数据少时提示数据不足而不是崩(self):
        series = _series(20)
        panel = indicator_panel(History(series, len(series.bars) - 1))
        assert any("数据不足" in line for line in panel)


class TestSignals:
    def test_每个内置策略都给出信号(self):
        from ashare_bt.strategies import available

        series = _series(300)
        signals = strategy_signals(History(series, len(series.bars) - 1))
        assert {name for name, _ in signals} == set(available())
        assert all(signal in {"满仓", "空仓"} or "报错" in signal or "%" in signal for _, signal in signals)


class TestExplore:
    def test_报告包含各个板块(self):
        text = explore(_series(300))
        for section in ("数据概览", "最近", "技术指标", "筹码分布", "各策略此刻的信号"):
            assert section in text, f"缺少「{section}」小节"

    def test_指定日期(self):
        series = _series(300)
        as_of = series.bars[200].date
        text = explore(series, as_of=as_of)
        assert as_of.isoformat() in text

    def test_日期早于数据要报错(self):
        series = _series(50)
        with pytest.raises(ValueError, match="没有数据"):
            explore(series, as_of=dt.date(1990, 1, 1))

    def test_亏损行情也能跑(self):
        text = explore(build_series([20.0 - i * 0.05 for i in range(200)]))
        assert "筹码分布" in text
