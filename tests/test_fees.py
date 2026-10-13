"""费用计算测试。

重点是那个 **5 元最低佣金** —— 小资金回测里它能把收益吃掉一大截，
而绝大多数网上的示例代码都忽略了它。
"""

from __future__ import annotations

import pytest

from ashare_bt.fees import DEFAULT_FEES, NO_MIN_FEES, FeeConfig
from ashare_bt.models import Side


class TestBuyFee:
    def test_小额买入按5元最低佣金收(self):
        # 1000 × 0.025% = 0.25 元，低于 5 元，所以收 5 元
        # 过户费 1000 × 0.001% = 0.01 元
        assert DEFAULT_FEES.buy_fee(1000.0) == pytest.approx(5.01)

    def test_大额买入按费率收而不是最低值(self):
        # 100 万 × 0.025% = 250 元；过户费 100 万 × 0.001% = 10 元
        assert DEFAULT_FEES.buy_fee(1_000_000.0) == pytest.approx(260.0)

    def test_买入不收印花税(self):
        buy = DEFAULT_FEES.breakdown(Side.BUY, 100_000.0)
        assert buy.stamp == 0.0

    def test_佣金刚好等于5元的临界点(self):
        # 5 / 0.00025 = 20000，成交额 2 万时佣金恰好 5 元
        assert DEFAULT_FEES.commission_rate * 20_000.0 == pytest.approx(5.0)
        breakdown = DEFAULT_FEES.breakdown(Side.BUY, 20_000.0)
        assert breakdown.commission == pytest.approx(5.0)


class TestSellFee:
    def test_卖出多收印花税(self):
        # 佣金 5（最低）+ 过户费 0.01 + 印花税 1000 × 0.05% = 0.5
        assert DEFAULT_FEES.sell_fee(1000.0) == pytest.approx(5.51)

    def test_大额卖出(self):
        # 佣金 250 + 过户费 10 + 印花税 500
        assert DEFAULT_FEES.sell_fee(1_000_000.0) == pytest.approx(760.0)

    def test_卖出确实收印花税(self):
        sell = DEFAULT_FEES.breakdown(Side.SELL, 100_000.0)
        assert sell.stamp == pytest.approx(50.0)  # 10 万 × 0.05%


class TestSmallAccountDrag:
    def test_小资金来回一趟的成本超过1个百分点(self):
        """这是小资金回测最容易被高估收益的地方。

        1000 元买 1000 元卖，来回手续费 10.52 元 = 1.05%。
        策略必须跑赢这个数才谈得上赚钱。
        """
        buy = DEFAULT_FEES.buy_fee(1000.0)
        sell = DEFAULT_FEES.sell_fee(1000.0)
        round_trip_cost_pct = (buy + sell) / 1000.0
        assert round_trip_cost_pct == pytest.approx(0.01052)
        assert round_trip_cost_pct > 0.01


class TestFeeConfig:
    def test_免5账户(self):
        assert NO_MIN_FEES.buy_fee(1000.0) == pytest.approx(0.26)  # 0.25 + 0.01

    def test_自定义费率(self):
        fees = FeeConfig(commission_rate=0.0001, commission_min=0.0, stamp_duty_rate=0.001)
        assert fees.buy_fee(100_000.0) == pytest.approx(11.0)  # 10 + 1
        assert fees.sell_fee(100_000.0) == pytest.approx(111.0)  # 10 + 1 + 100

    def test_费用函数按方向分发(self):
        assert DEFAULT_FEES.fee(Side.BUY, 1000.0) == DEFAULT_FEES.buy_fee(1000.0)
        assert DEFAULT_FEES.fee(Side.SELL, 1000.0) == DEFAULT_FEES.sell_fee(1000.0)

    def test_拆分明细可以加总回总额(self):
        for side in (Side.BUY, Side.SELL):
            breakdown = DEFAULT_FEES.breakdown(side, 50_000.0)
            assert breakdown.total == pytest.approx(DEFAULT_FEES.fee(side, 50_000.0))
