"""涨跌停规则在引擎里的行为。

涨跌停的两面都要测：
- 触及了要**拒绝**（涨停买不进、跌停卖不出）
- 差一分钱要**能成交**（否则就是把拒绝条件写得太宽，凭空少掉一批交易）

注意：测试序列前面要垫足够多的平盘 K 线。新股上市前 5 个交易日不设涨跌幅限制，
如果直接在序列第 1 根上测涨跌停，测的其实是"新股豁免"而不是"涨跌停"。
"""

from __future__ import annotations

from ashare_bt.engine import EngineConfig, _Account, execute_order
from ashare_bt.models import Fill, Order, Position, RejectReason, Rejection, Side
from ashare_bt.rules import BSE, STAR, limit_prices, new_listing_exempt

from conftest import build_series

LEAD_IN = 6
"""垫在待测交易日前面的平盘 K 线根数 —— 要大于新股不限价的 5 天。"""
BASE = 10.0


def day_series(day_open: float, day_high: float, day_low: float, day_close: float,
               code: str = "600519", lead_in: int = LEAD_IN):
    """前面 lead_in 根平盘 10 元，最后一根是待测的一天。"""
    return build_series(
        closes=[BASE] * lead_in + [day_close],
        opens=[BASE] * lead_in + [day_open],
        highs=[BASE] * lead_in + [day_high],
        lows=[BASE] * lead_in + [day_low],
        code=code,
    )


def _account_for_sell(qty: int = 1000) -> _Account:
    """造一个"手里有票、随时可卖"的账户，用来测卖出方向的限制。"""
    return _Account(cash=0.0, position=Position(total=qty, sellable=qty))


def _run(series, account, side, qty, i, config=None):
    bar = series.bars[i]
    order = Order(side, qty, series.bars[i - 1].date)
    return execute_order(order, bar, series.bars[i - 1].close, series, account, config or EngineConfig(), i)


class TestLimitUp:
    def test_一字涨停买不进(self):
        """开盘就是涨停价并封死 —— 买不到。"""
        series = day_series(11.0, 11.0, 11.0, 11.0)
        account = _Account(cash=100_000.0)

        outcome = _run(series, account, Side.BUY, 1000, LEAD_IN)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.LIMIT_UP
        assert account.position.total == 0  # 一股都没买进来

    def test_差一分钱就能买(self):
        """10.99 没到涨停价 11.00，应该成交。"""
        series = day_series(10.99, 10.99, 10.99, 10.99)
        account = _Account(cash=100_000.0)

        outcome = _run(series, account, Side.BUY, 1000, LEAD_IN)

        assert isinstance(outcome, Fill)
        assert outcome.price == 10.99

    def test_创业板15个点不算涨停(self):
        """创业板 2020-08-24 起是 20%，涨 15% 不该被拒。"""
        series = day_series(11.5, 11.5, 11.5, 11.5, code="300750")
        account = _Account(cash=100_000.0)

        assert isinstance(_run(series, account, Side.BUY, 1000, LEAD_IN), Fill)

    def test_创业板涨20个点才是涨停(self):
        series = day_series(12.0, 12.0, 12.0, 12.0, code="300750")
        account = _Account(cash=100_000.0)

        outcome = _run(series, account, Side.BUY, 1000, LEAD_IN)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.LIMIT_UP

    def test_科创板是20个点不是10(self):
        """同样的涨幅，主板被拒、科创板能买 —— 板块判断错了这条会红。"""
        account_a = _Account(cash=100_000.0)
        assert isinstance(_run(day_series(11.5, 11.5, 11.5, 11.5), account_a, Side.BUY, 1000, LEAD_IN), Rejection)

        account_b = _Account(cash=100_000.0)
        assert isinstance(
            _run(day_series(11.5, 11.5, 11.5, 11.5, code="688111"), account_b, Side.BUY, 200, LEAD_IN), Fill
        )


class TestLimitDown:
    def test_跌停卖不出(self):
        series = day_series(9.0, 9.0, 9.0, 9.0)
        account = _account_for_sell()

        outcome = _run(series, account, Side.SELL, 1000, LEAD_IN)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.LIMIT_DOWN
        assert account.position.total == 1000  # 还拿着

    def test_差一分钱就能卖(self):
        series = day_series(9.01, 9.01, 9.01, 9.01)
        account = _account_for_sell()

        assert isinstance(_run(series, account, Side.SELL, 1000, LEAD_IN), Fill)

    def test_跌停那天买入不受影响(self):
        """跌停限制的是卖，不是买。"""
        series = day_series(9.0, 9.0, 9.0, 9.0)
        account = _Account(cash=100_000.0)

        assert isinstance(_run(series, account, Side.BUY, 1000, LEAD_IN), Fill)


class TestNewListing:
    def test_新股上市次日的暴涨不限价(self):
        """上市后第 1 个交易日仍在 5 日豁免期内，涨 44% 也能买。"""
        series = day_series(14.4, 14.4, 14.4, 14.4, lead_in=1)
        account = _Account(cash=100_000.0)

        # bar_index=1 → 距上市 1 个交易日
        outcome = _run(series, account, Side.BUY, 1000, 1)

        assert isinstance(outcome, Fill)

    def test_关掉新股规则后同样的行情会被拒(self):
        series = day_series(14.4, 14.4, 14.4, 14.4, lead_in=1)
        account = _Account(cash=100_000.0)

        outcome = _run(series, account, Side.BUY, 1000, 1, EngineConfig(apply_new_listing_rule=False))

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.LIMIT_UP

    def test_丢弃前导数据后不能误判成新股(self):
        """前复权负价被丢弃后，序列第 0 根并不是上市首日。

        茅台丢弃了 3526 根，它序列开头那一天是上市后第 3526 个交易日，
        绝不该享受"新股不限价"待遇。
        """
        series = day_series(11.0, 11.0, 11.0, 11.0)
        trimmed = series._replace(dropped_bars=3526)
        account = _Account(cash=100_000.0)

        outcome = _run(trimmed, account, Side.BUY, 1000, LEAD_IN)

        assert isinstance(outcome, Rejection)
        assert outcome.reason is RejectReason.LIMIT_UP


class TestRuleHelpers:
    def test_涨跌停价按四舍五入算(self):
        assert limit_prices(10.0, 0.10) == (11.0, 9.0)

    def test_科创板新股前5日不限价(self):
        assert new_listing_exempt(STAR, 4)
        assert not new_listing_exempt(STAR, 5)

    def test_北交所只有首日不限价(self):
        assert new_listing_exempt(BSE, 0)
        assert not new_listing_exempt(BSE, 1)
