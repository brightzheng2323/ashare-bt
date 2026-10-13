"""指标测试。

重点不在数值精度（那些是标准公式），而在**因果性**：
指标只能看序列末尾，不能看未来。这是未来函数的源头之一。
"""

from __future__ import annotations

import pytest

from ashare_bt.indicators import bollinger, ema, highest, lowest, macd, rsi, sma


class TestSma:
    def test_数据不够返回None(self):
        assert sma([1.0, 2.0], 5) is None

    def test_刚好够(self):
        assert sma([1.0, 2.0, 3.0], 3) == pytest.approx(2.0)

    def test_只用最后window个(self):
        assert sma([100.0, 1.0, 2.0, 3.0], 3) == pytest.approx(2.0)

    def test_窗口非法要报错(self):
        with pytest.raises(ValueError):
            sma([1.0, 2.0], 0)

    def test_只依赖到今天为止的数据(self):
        """因果性：截到今天算出来的值，和用全序列算法昨天那个位置的值，必须一致。"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # 只看到第 4 个元素时（下标 3），MA3 = mean(2,3,4)
        assert sma(values[:4], 3) == pytest.approx(3.0)
        # 再往后看到第 5 个元素时，值变成 mean(3,4,5)，这是新的"今天"，变了是对的
        assert sma(values, 3) == pytest.approx(4.0)


class TestEma:
    def test_数据不够返回None(self):
        assert ema([1.0], 5) is None

    def test_种子是前window个的简单平均(self):
        assert ema([1.0, 2.0, 3.0], 3) == pytest.approx(2.0)

    def test_指数平均比简单平均更贴近最新值(self):
        values = [1.0] * 10 + [100.0]
        assert ema(values, 5) > sma(values, 5)


class TestRsi:
    def test_数据不够返回None(self):
        assert rsi([1.0] * 10, 14) is None

    def test_一路上涨是100(self):
        assert rsi([float(i) for i in range(1, 30)], 14) == pytest.approx(100.0)

    def test_一路下跌接近0(self):
        assert rsi([float(i) for i in range(30, 1, -1)], 14) < 1.0

    def test_永远落在0到100之间(self):
        import random

        random.seed(42)
        values = [100.0]
        for _ in range(200):
            values.append(values[-1] * (1 + random.uniform(-0.05, 0.05)))
        value = rsi(values, 14)
        assert 0.0 <= value <= 100.0

    def test_横盘不除零(self):
        assert rsi([10.0] * 30, 14) is not None


class TestMacd:
    def test_数据不够返回None(self):
        assert macd([float(i) for i in range(20)]) is None

    def test_返回三个值且柱等于两倍差值(self):
        values = [10.0 + i * 0.1 for i in range(100)]
        dif, dea, hist = macd(values)
        assert hist == pytest.approx(2.0 * (dif - dea))

    def test_参数倒置要报错(self):
        with pytest.raises(ValueError):
            macd([float(i) for i in range(100)], fast=26, slow=12)


class TestBollinger:
    def test_数据不够返回None(self):
        assert bollinger([1.0, 2.0], 20) is None

    def test_三轨顺序正确(self):
        values = [10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.5]
        lower, middle, upper = bollinger(values, 10, 2.0)
        assert lower < middle < upper

    def test_横盘时标准差为零三轨重合(self):
        lower, middle, upper = bollinger([10.0] * 20, 20, 2.0)
        assert lower == middle == upper == pytest.approx(10.0)

    def test_波动越大轨道越宽(self):
        calm = [10.0, 10.1, 9.9, 10.05, 9.95] * 4
        wild = [10.0, 12.0, 8.0, 11.0, 9.0] * 4
        assert (bollinger(wild, 20, 2.0)[2] - bollinger(wild, 20, 2.0)[0]) > (
            bollinger(calm, 20, 2.0)[2] - bollinger(calm, 20, 2.0)[0]
        )


class TestExtremes:
    def test_最高最低(self):
        values = [5.0, 3.0, 9.0, 7.0]
        assert highest(values, 3) == 9.0
        assert lowest(values, 3) == 3.0

    def test_数据不够返回None(self):
        assert highest([1.0], 5) is None
        assert lowest([1.0], 5) is None
