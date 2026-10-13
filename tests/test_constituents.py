"""指数成分股获取测试。

`index_secid` 是这里最容易错的地方 —— **指数和个股的编码规则不一样**：
    000001 当个股是 0.000001（平安银行），当指数是 1.000001（上证指数）
所以指数不能复用个股的 to_secid。
"""

from __future__ import annotations

import pytest

from ashare_bt.errors import DataError
from ashare_bt.fetch import (
    CONSTITUENT_PAGE_SIZE,
    INDEX_BOARDS,
    fetch_index_constituents,
    index_secid,
)


class TestIndexSecid:
    def test_000开头是沪市指数(self):
        assert index_secid("000300") == "1.000300"
        assert index_secid("000016") == "1.000016"
        assert index_secid("000905") == "1.000905"

    def test_上证指数和深市个股同码但不同市场(self):
        """这是指数编码最容易踩的坑。"""
        from ashare_bt.rules import to_secid

        assert index_secid("000001") == "1.000001"  # 上证指数
        assert to_secid("000001") == "0.000001"  # 平安银行

    def test_399开头是深市指数(self):
        assert index_secid("399001") == "0.399001"
        assert index_secid("399006") == "0.399006"

    def test_已经带前缀就原样返回(self):
        assert index_secid("1.000300") == "1.000300"
        assert index_secid("0.399001") == "0.399001"

    def test_乱写会报错(self):
        with pytest.raises(ValueError):
            index_secid("abc")


class TestIndexBoards:
    def test_已知的三个指数都在(self):
        assert INDEX_BOARDS["1.000300"][1] == "沪深300"
        assert INDEX_BOARDS["1.000016"][1] == "上证50"
        assert INDEX_BOARDS["1.000905"][1] == "中证500"

    def test_板块代码格式正确(self):
        assert all(board.startswith("BK") for board, _ in INDEX_BOARDS.values())


class TestUnknownIndex:
    def test_不认识的指数给出可用列表(self):
        with pytest.raises(DataError, match="还不认识指数"):
            fetch_index_constituents("1.999999")

    def test_报错信息里列出支持的指数(self):
        with pytest.raises(DataError, match="沪深300"):
            fetch_index_constituents("1.999999")


@pytest.mark.network
class TestFetchConstituentsLive:
    """联网测试，默认跳过（pytest -m network 才跑）。"""

    def test_沪深300正好300只(self):
        members = fetch_index_constituents("000300")
        assert len(members) == 300

    def test_上证50正好50只(self):
        assert len(fetch_index_constituents("000016")) == 50

    def test_limit能截断(self):
        assert len(fetch_index_constituents("000300", limit=7)) == 7

    def test_返回代码是6位数字(self):
        members = fetch_index_constituents("000300", limit=20)
        assert all(code.isdigit() and len(code) == 6 for code, _ in members)

    def test_没有重复(self):
        members = fetch_index_constituents("000300")
        codes = [c for c, _ in members]
        assert len(codes) == len(set(codes))

    def test_页大小不超过接口上限(self):
        assert CONSTITUENT_PAGE_SIZE <= 100
