"""A 股规则的测试。

这些规则看着琐碎，但每一条写错都会让回测结果失真，
而且**错了程序照跑**，不报任何错 —— 这正是需要测试的原因。
"""

from __future__ import annotations

import datetime as dt

import pytest

from ashare_bt.rules import (
    BSE,
    CHINEXT,
    SSE_MAIN,
    STAR,
    SZSE_MAIN,
    board_of,
    floor_lot,
    is_st,
    limit_pct,
    limit_prices,
    lot_size,
    new_listing_exempt,
    normalize_code,
    round_half_up,
    to_secid,
)


class TestNormalizeCode:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("600519", "600519"),
            ("sh600519", "600519"),
            ("SH600519", "600519"),
            ("sz000001", "000001"),
            ("1.600519", "600519"),
            (" 600519 ", "600519"),
        ],
    )
    def test_常见写法都能认(self, raw, expected):
        assert normalize_code(raw) == expected

    @pytest.mark.parametrize("bad", ["", "60051", "abcdef", "1234567", "600-519"])
    def test_乱写的代码要报错(self, bad):
        with pytest.raises(ValueError):
            normalize_code(bad)


class TestSecid:
    def test_沪市加前缀1(self):
        assert to_secid("600519") == "1.600519"

    def test_深市加前缀0(self):
        assert to_secid("000001") == "0.000001"
        assert to_secid("300750") == "0.300750"

    def test_科创板属于沪市(self):
        assert to_secid("688111") == "1.688111"


class TestBoardOf:
    @pytest.mark.parametrize(
        "code,board",
        [
            ("600519", SSE_MAIN),
            ("601398", SSE_MAIN),
            ("000001", SZSE_MAIN),
            ("002594", SZSE_MAIN),
            ("300750", CHINEXT),
            ("301236", CHINEXT),
            ("688111", STAR),
            ("830799", BSE),
        ],
    )
    def test_板块识别(self, code, board):
        assert board_of(code) == board


class TestIsSt:
    @pytest.mark.parametrize("name", ["ST某某", "*ST某某", "S*ST某某", "某某退"])
    def test_是st(self, name):
        assert is_st(name)

    @pytest.mark.parametrize("name", ["贵州茅台", "宁德时代", "平安银行"])
    def test_不是st(self, name):
        assert not is_st(name)

    def test_st判定不区分大小写(self):
        assert is_st("st某某")


class TestLimitPct:
    def test_主板普通股10个点(self):
        assert limit_pct("600519", "贵州茅台", dt.date(2024, 1, 2)) == 0.10
        assert limit_pct("000001", "平安银行", dt.date(2024, 1, 2)) == 0.10

    def test_主板st在2026年7月6日前是5个点(self):
        assert limit_pct("600519", "ST某某", dt.date(2025, 1, 2)) == 0.05

    def test_主板st在2026年7月6日起改为10个点(self):
        """2026-07-06 沪深主板 ST 涨跌幅与普通股统一，从 5% 提到 10%。"""
        assert limit_pct("600519", "ST某某", dt.date(2026, 7, 6)) == 0.10
        assert limit_pct("600519", "ST某某", dt.date(2026, 7, 5)) == 0.05

    def test_创业板2020年8月24日前是10个点(self):
        assert limit_pct("300750", "宁德时代", dt.date(2020, 1, 2)) == 0.10

    def test_创业板2020年8月24日起是20个点(self):
        assert limit_pct("300750", "宁德时代", dt.date(2020, 8, 24)) == 0.20

    def test_创业板st也是20个点(self):
        """创业板 ST 在注册制改革后与普通股统一为 20%，不是 5%。"""
        assert limit_pct("300750", "ST某某", dt.date(2021, 1, 4)) == 0.20

    def test_科创板20个点(self):
        assert limit_pct("688111", "金山办公", dt.date(2024, 1, 2)) == 0.20

    def test_北交所30个点(self):
        assert limit_pct("830799", "未知", dt.date(2024, 1, 2)) == 0.30

    def test_认不出的板块没有涨跌停限制(self):
        assert limit_pct("900001", "未知", dt.date(2024, 1, 2)) is None


class TestRoundHalfUp:
    def test_四舍五入不是银行家舍入(self):
        """Python 内置 round(2.675, 2) == 2.67，交易所是 2.68。

        这一分钱直接决定涨跌停价，进而决定能不能成交。
        """
        assert round(2.675, 2) == 2.67  # Python 的行为
        assert round_half_up(2.675, 2) == 2.68  # 交易所的行为

    @pytest.mark.parametrize(
        "value,expected",
        [
            (10.0, 10.0),
            (10.005, 10.01),
            (10.004, 10.0),
            (10.014999, 10.01),
        ],
    )
    def test_各种边界(self, value, expected):
        assert round_half_up(value, 2) == expected

    def test_保留更多位(self):
        assert round_half_up(1.23456, 4) == 1.2346


class TestLimitPrices:
    def test_整数价的涨跌停(self):
        assert limit_prices(10.0, 0.10) == (11.0, 9.0)

    def test_非整数价的涨跌停要四舍五入到分(self):
        upper, lower = limit_prices(10.05, 0.10)
        assert upper == 11.06  # 11.055 四舍五入
        assert lower == 9.05  # 9.045 四舍五入

    def test_20个点(self):
        assert limit_prices(100.0, 0.20) == (120.0, 80.0)


class TestLotSize:
    def test_主板创业板是100股(self):
        assert lot_size("600519") == 100
        assert lot_size("300750") == 100

    def test_科创板是200股(self):
        assert lot_size("688111") == 200


class TestFloorLot:
    def test_主板向下取整到百股(self):
        assert floor_lot(150, "600519") == 100
        assert floor_lot(999, "600519") == 900
        assert floor_lot(100, "600519") == 100

    def test_主板不足100股买不了(self):
        assert floor_lot(99, "600519") == 0

    def test_科创板200股起然后1股递增(self):
        assert floor_lot(200, "688111") == 200
        assert floor_lot(201, "688111") == 201  # 1 股递增，不取整
        assert floor_lot(150, "688111") == 0  # 不够 200 买不了


class TestNewListing:
    def test_沪深新股前5个交易日不限价(self):
        assert new_listing_exempt(SSE_MAIN, 0)
        assert new_listing_exempt(SSE_MAIN, 4)
        assert not new_listing_exempt(SSE_MAIN, 5)

    def test_科创板新股也是前5日(self):
        assert new_listing_exempt(STAR, 3)
        assert not new_listing_exempt(STAR, 5)

    def test_北交所只有首日不限价(self):
        assert new_listing_exempt(BSE, 0)
        assert not new_listing_exempt(BSE, 1)
