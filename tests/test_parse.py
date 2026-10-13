"""接口解析测试。

用真实的接口响应做固定样本（data/fixtures/ 里那 11 根茅台的 K 线），
这样接口一改版测试就会红，而不是等到回测跑出怪结果才发现。
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest

from ashare_bt.errors import DataError
from ashare_bt.fetch import FQT_BACKWARD, parse_klines

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "eastmoney_600519.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestParseRealResponse:
    def test_基本信息(self, payload):
        series = parse_klines(payload, "1.600519", FQT_BACKWARD)
        assert series.code == "600519"
        assert series.secid == "1.600519"
        assert series.name == "贵州茅台"
        assert series.fqt == FQT_BACKWARD
        assert len(series.bars) == 11

    def test_首根k线的每个字段都要对得上(self, payload):
        """钉死字段顺序 —— 这条是整个数据层最容易出错的地方。

        接口返回的顺序是 **开、收、高、低**，不是通常的 OHLC。
        写成 open, high, low, close 的话，程序照跑不误，
        但"最高价"其实是收盘价，指标、涨跌停判断全都跟着错。
        """
        bar = parse_klines(payload, "1.600519", FQT_BACKWARD).bars[0]
        assert bar.date == date(2024, 1, 2)
        assert bar.open == pytest.approx(9548.13)
        assert bar.close == pytest.approx(9397.44)  # 第 3 个字段
        assert bar.high == pytest.approx(9564.16)  # 第 4 个字段
        assert bar.low == pytest.approx(9362.72)  # 第 5 个字段
        assert bar.volume == pytest.approx(32156)  # 手
        assert bar.amount == pytest.approx(5440082548.00)  # 元
        assert bar.turnover == pytest.approx(0.26)  # %

    def test_最高价确实大于等于最低价(self, payload):
        """如果字段读错了（把收盘当最高），这条会立刻失败。

        这是防"读错顺序"的第二道保险：真实性检验，而不是逐字段比对。
        """
        series = parse_klines(payload, "1.600519", FQT_BACKWARD)
        for bar in series.bars:
            assert bar.high >= bar.low
            assert bar.high >= bar.open >= bar.low
            assert bar.high >= bar.close >= bar.low

    def test_按日期升序(self, payload):
        series = parse_klines(payload, "1.600519", FQT_BACKWARD)
        dates = [bar.date for bar in series.bars]
        assert dates == sorted(dates)

    def test_index_of能定位到具体某天(self, payload):
        series = parse_klines(payload, "1.600519", FQT_BACKWARD)
        assert series.index_of(date(2024, 1, 2)) == 0
        assert series.index_of(date(2024, 1, 16)) == 10
        assert series.index_of(date(2024, 1, 7)) is None  # 周日，不是交易日


class TestParseErrorHandling:
    def test_data为null要报错(self):
        with pytest.raises(DataError, match="没有数据"):
            parse_klines({"rc": 0, "data": None}, "1.999999", FQT_BACKWARD)

    def test_klines为空要报错(self, payload):
        broken = copy.deepcopy(payload)
        broken["data"]["klines"] = []
        with pytest.raises(DataError, match="没有 K 线数据"):
            parse_klines(broken, "1.600519", FQT_BACKWARD)

    def test_字段不全要报错(self, payload):
        broken = copy.deepcopy(payload)
        broken["data"]["klines"] = ["2024-01-02,1,2,3"]
        with pytest.raises(DataError, match="字段少于"):
            parse_klines(broken, "1.600519", FQT_BACKWARD)

    def test_数据不自洽要报错(self, payload):
        """人为造一根 low > high 的 K 线，必须被挡下来。"""
        broken = copy.deepcopy(payload)
        broken["data"]["klines"] = ["2024-01-02,10.0,11.0,9.0,12.0,100,1000,0,0,0,0"]
        with pytest.raises(DataError, match="不自洽"):
            parse_klines(broken, "1.600519", FQT_BACKWARD)

    def test_整段价格都非正要报错(self, payload):
        """前复权的负价前导段会被丢弃，但整段都是负价就没救了。"""
        broken = copy.deepcopy(payload)
        broken["data"]["klines"] = ["2024-01-02,0.0,0.0,0.0,0.0,100,1000,0,0,0,0"]
        with pytest.raises(DataError, match="整段历史的价格都非正"):
            parse_klines(broken, "1.600519", FQT_BACKWARD)


class TestParseRobustness:
    def test_重复日期只保留一条(self, payload):
        dup = copy.deepcopy(payload)
        dup["data"]["klines"] = dup["data"]["klines"][:2] * 2
        series = parse_klines(dup, "1.600519", FQT_BACKWARD)
        assert len(series.bars) == 2

    def test_乱序会排好(self, payload):
        import copy as _copy

        shuffled = _copy.deepcopy(payload)
        shuffled["data"]["klines"] = list(reversed(shuffled["data"]["klines"]))
        series = parse_klines(shuffled, "1.600519", FQT_BACKWARD)
        dates = [bar.date for bar in series.bars]
        assert dates == sorted(dates)

    def test_缺少换手率字段也能解析(self, payload):
        """老接口偶尔少给字段。缺了只影响筹码分布，不该让整个回测挂掉。"""
        short = copy.deepcopy(payload)
        short["data"]["klines"] = ["2024-01-02,10.0,10.5,11.0,9.5,100,1000"]
        series = parse_klines(short, "1.600519", FQT_BACKWARD)
        assert series.bars[0].turnover == 0.0
