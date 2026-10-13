"""整手与资金的边界。

这两件事最容易出"隐形杠杆"的 bug：算错了程序照跑，
但账户里会出现买不起却能买、或者现金变成负数的情况。
"""

from __future__ import annotations

from ashare_bt.engine import EngineConfig, _Account, execute_order, max_affordable_qty, run_backtest
from ashare_bt.fees import DEFAULT_FEES
from ashare_bt.models import Fill, Order, RejectReason, Rejection, Side
from conftest import ScriptedStrategy


def _buy(series, account, i, qty, config=None):
    bar = series.bars[i]
    order = Order(Side.BUY, qty, series.bars[i - 1].date)
    return execute_order(order, bar, series.bars[i - 1].close, series, account, config or EngineConfig(), i)


class TestLotRounding:
    def test_买入成交股数一定是整手(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0)
        outcome = _buy(series, account, 1, 1500)

        assert isinstance(outcome, Fill)
        assert outcome.qty % 100 == 0

    def test_非整手下单会向下补齐(self, make_flat):
        """下 150 股，实际只能成交 100 股。"""
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0)
        outcome = _buy(series, account, 1, 150)

        assert isinstance(outcome, Fill)
        assert outcome.qty == 100

    def test_不足一手整单作废(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=100_000.0)
        outcome = _buy(series, account, 1, 50)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.LOT

    def test_科创板200股起且可以1股递增(self, make_flat):
        series = make_flat(10.0, 5, code="688111")
        account = _Account(cash=100_000.0)

        assert isinstance(_buy(series, account, 1, 150), Rejection)  # 不足 200

        account2 = _Account(cash=100_000.0)
        outcome = _buy(series, account2, 1, 201)
        assert isinstance(outcome, Fill)
        assert outcome.qty == 201  # 科创板不清零到百股


class TestCashSafety:
    def test_账户现金永远不会变成负数(self):
        # 价格一路上涨，策略反复满仓 —— 最容易把现金算穿帮的情形
        from conftest import build_series

        series = build_series([10.0 + i * 0.37 for i in range(60)])
        strategy = ScriptedStrategy([1.0, 0.0] * 30)
        result = run_backtest(series, strategy, EngineConfig(initial_cash=100_000.0))

        assert all(c >= 0 for c in result.cash), "出现负现金 = 回测偷偷加了杠杆"

    def test_每笔成交都买得起(self, make_flat):
        series = make_flat(10.0, 20)
        strategy = ScriptedStrategy([1.0, 0.0] * 10)
        config = EngineConfig(initial_cash=50_000.0)
        result = run_backtest(series, strategy, config=config)

        for fill in result.fills:
            if fill.side is Side.BUY:
                assert fill.amount + fill.fee <= config.initial_cash + 1e-6

    def test_买满之后现金非负且几乎用尽(self, make_flat):
        series = make_flat(10.0, 5)
        account = _Account(cash=10_000.0)
        _buy(series, account, 1, 100_000)  # 想买天量

        assert account.cash >= 0
        assert account.cash < 1000  # 基本买满了

    def test_钱太少买不起一手(self, make_flat):
        series = make_flat(50.0, 5)
        account = _Account(cash=100.0)  # 连 1 手（5000 元）都买不起
        outcome = _buy(series, account, 1, 100)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.CASH


class TestAffordableQty:
    def test_手续费必须算进去(self):
        """只看 cash/price 会算出买不起的股数 —— 差的就是那 5 元最低佣金。"""
        qty = max_affordable_qty(cash=10_000.0, price=10.0, code="600519", fees=DEFAULT_FEES)
        assert qty == 900  # 900 股 = 9000 元 + 5.1 元费用；1000 股要 10000 + 5.1 就超了

    def test_价格为0时返回0(self):
        assert max_affordable_qty(cash=10_000.0, price=0.0, code="600519", fees=DEFAULT_FEES) == 0

    def test_资金为0时返回0(self):
        assert max_affordable_qty(cash=0.0, price=10.0, code="600519", fees=DEFAULT_FEES) == 0

    def test_大资金下佣金按费率走(self):
        qty = max_affordable_qty(cash=1_000_000.0, price=10.0, code="600519", fees=DEFAULT_FEES)
        amount = qty * 10.0
        assert amount + DEFAULT_FEES.buy_fee(amount) <= 1_000_000.0


class TestGapUp:
    def test_高开导致买入股数变少(self):
        """按昨天收盘价估的股数，遇到高开就买不起了 —— 必须按开盘价重算。"""
        from conftest import build_series

        series = build_series(closes=[10.0, 11.0], opens=[10.0, 11.0])
        account = _Account(cash=5_600.0)

        # 昨天收盘价 10 元时估算能买 500 股（5000 元 + 费用）
        estimate = max_affordable_qty(account.cash, 10.0, "600519", DEFAULT_FEES)
        assert estimate == 500

        # 今天开盘 11 元，只能买 500 股中的一部分 → 实际 500 股要 5500 + 5.05 也买得起
        outcome = _buy(series, account, 1, estimate)
        assert isinstance(outcome, Fill)
        assert outcome.qty * 11.0 + outcome.fee <= 5_600.0 + 1e-6
        assert account.cash >= 0
