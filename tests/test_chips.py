"""筹码分布测试。

这块最容易错的是**网格管理**：价格区间随行情扩张时旧筹码要重采样，
重采样写错了筹码总量就会凭空变多或变少 —— 而总量本该恒为 1。
所以下面的测试大量在断言"总量守恒"。
"""

from __future__ import annotations

import pytest

from ashare_bt.chips import ChipDistribution


def feed(chips: ChipDistribution, bars) -> None:
    """bars: [(high, low, avg, turnover_pct), ...]"""
    for high, low, avg, turnover in bars:
        chips.update(high, low, avg, turnover)


class TestWeightConservation:
    def test_第一次更新后总量为一(self):
        chips = ChipDistribution(bins=100)
        chips.update(11.0, 9.0, 10.0, 5.0)
        assert chips.total_weight == pytest.approx(1.0)

    def test_连续更新总量始终为一(self):
        chips = ChipDistribution(bins=100)
        feed(chips, [(11.0, 9.0, 10.0, 5.0)] * 20)
        assert chips.total_weight == pytest.approx(1.0)

    def test_价格区间扩张后总量守恒(self):
        """行情一路创新高，网格要不断扩张并重采样 —— 这里是重采样写错的地方。"""
        chips = ChipDistribution(bins=100)
        price = 10.0
        for _ in range(50):
            chips.update(price * 1.1, price * 0.9, price, 3.0)
            price *= 1.05  # 一路上涨，网格要持续扩张
        assert chips.total_weight == pytest.approx(1.0, abs=1e-9)

    def test_价格区间向下扩张也守恒(self):
        chips = ChipDistribution(bins=100)
        price = 50.0
        for _ in range(50):
            chips.update(price * 1.05, price * 0.95, price, 3.0)
            price *= 0.95
        assert chips.total_weight == pytest.approx(1.0, abs=1e-9)

    def test_长时间来回震荡仍守恒(self):
        import random

        random.seed(7)
        chips = ChipDistribution(bins=200)
        price = 10.0
        for _ in range(500):
            price *= 1 + random.uniform(-0.06, 0.06)
            chips.update(price * 1.03, price * 0.97, price, random.uniform(0.5, 8.0))
        assert chips.total_weight == pytest.approx(1.0, abs=1e-9)


class TestDecay:
    def test_换手为零时分布完全不动(self):
        """停牌或一字板：没有换手，筹码不该有任何变化。"""
        chips = ChipDistribution(bins=100)
        chips.update(11.0, 9.0, 10.0, 5.0)
        before = list(chips.weights)
        chips.update(20.0, 5.0, 12.0, 0.0)
        assert chips.weights == before

    def test_高换手率会让旧筹码被大量替换(self):
        """一直在高位换手，平均成本应该被拉上去。"""
        chips = ChipDistribution(bins=200)
        feed(chips, [(10.2, 9.8, 10.0, 50.0)] * 10)  # 先在 10 元附近换手
        low_cost = chips.state(10.0).avg_cost

        feed(chips, [(20.4, 19.6, 20.0, 50.0)] * 10)  # 再到 20 元附近大量换手
        high_cost = chips.state(20.0).avg_cost

        assert high_cost > low_cost
        assert high_cost > 15.0  # 已经明显被拉到高位

    def test_低换手时旧筹码保留得住(self):
        chips = ChipDistribution(bins=200)
        feed(chips, [(10.2, 9.8, 10.0, 1.0)] * 100)  # 长期在 10 元附近低换手
        chips.update(20.4, 19.6, 20.0, 1.0)  # 突然到 20 元，但只换手 1%
        assert chips.state(20.0).avg_cost < 10.5  # 成本基本还在 10 元

    def test_换手率超过100会被夹住(self):
        """次新股上市初期换手率可能超过 100%，不能因此算出负权重。"""
        chips = ChipDistribution(bins=100)
        chips.update(11.0, 9.0, 10.0, 250.0)
        assert all(w >= 0 for w in chips.weights)
        assert chips.total_weight == pytest.approx(1.0)


class TestState:
    def test_全部筹码都在现价之下时获利盘是满的(self):
        chips = ChipDistribution(bins=200)
        feed(chips, [(10.2, 9.8, 10.0, 10.0)] * 30)
        assert chips.state(100.0).profit_ratio == pytest.approx(1.0, abs=0.01)

    def test_现价低于所有成本时获利盘为零(self):
        chips = ChipDistribution(bins=200)
        feed(chips, [(10.2, 9.8, 10.0, 10.0)] * 30)
        assert chips.state(1.0).profit_ratio == pytest.approx(0.0, abs=0.01)

    def test_平均成本贴近成交密集区(self):
        chips = ChipDistribution(bins=200)
        feed(chips, [(10.2, 9.8, 10.0, 10.0)] * 50)
        assert chips.state(10.0).avg_cost == pytest.approx(10.0, abs=0.2)

    def test_九十区间包含大部分筹码(self):
        chips = ChipDistribution(bins=200)
        feed(chips, [(10.5, 9.5, 10.0, 10.0)] * 50)
        state = chips.state(10.0)
        assert state.cost_low < 10.0 < state.cost_high
        assert state.cost_high >= state.cost_low

    def test_横盘比大幅震荡更集中(self):
        """集中度的意义就在这：窄幅震荡 = 筹码集中 = 底部扎实。"""
        tight = ChipDistribution(bins=200)
        feed(tight, [(10.05, 9.95, 10.0, 10.0)] * 60)

        loose = ChipDistribution(bins=200)
        feed(loose, [(15.0, 5.0, 10.0, 10.0)] * 60)

        assert tight.state(10.0).concentration < loose.state(10.0).concentration

    def test_空分布不会崩(self):
        state = ChipDistribution(bins=100).state(10.0)
        assert state.profit_ratio == 0.0
        assert state.avg_cost == 10.0


class TestValidation:
    def test_平均价不在区间内要报错(self):
        chips = ChipDistribution(bins=100)
        with pytest.raises(ValueError, match="不在当日区间"):
            chips.update(high=10.0, low=9.0, avg=12.0, turnover_pct=1.0)

    def test_格子太少要报错(self):
        with pytest.raises(ValueError):
            ChipDistribution(bins=5)

    def test_衰减系数越界要报错(self):
        with pytest.raises(ValueError):
            ChipDistribution(decay=0.0)
        with pytest.raises(ValueError):
            ChipDistribution(decay=1.5)


class TestOneWordBoard:
    def test_一字板不会除零(self):
        """一字跌停/涨停时 low == high，三角形会退化成一个点。"""
        chips = ChipDistribution(bins=100)
        chips.update(10.0, 9.0, 9.5, 5.0)
        chips.update(9.0, 9.0, 9.0, 1.0)  # 一字板

        assert chips.total_weight == pytest.approx(1.0)
        assert chips.state(9.0).profit_ratio >= 0.0
