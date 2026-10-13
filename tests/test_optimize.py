"""参数寻优测试。

重点不是"能不能找到最优参数"，而是：
- 非法参数组合要被跳过而不是让整个寻优崩掉
- 样本内/样本外要真的分开（切分写错了整个功能就是摆设）
"""

from __future__ import annotations

import pytest

from ashare_bt.engine import EngineConfig
from ashare_bt.optimize import (
    MAX_COMBINATIONS,
    SweepRow,
    expand_grid,
    format_sweep,
    overfit_report,
    sweep,
)
from conftest import build_series


class TestExpandGrid:
    def test_展开成笛卡尔积(self):
        grid = expand_grid({"fast": [3, 5], "slow": [20, 60]})
        assert len(grid) == 4
        assert {"fast": 3, "slow": 20} in grid
        assert {"fast": 5, "slow": 60} in grid

    def test_空网格要报错(self):
        with pytest.raises(ValueError, match="不能为空"):
            expand_grid({})

    def test_某个参数没给取值要报错(self):
        with pytest.raises(ValueError, match="没有给任何取值"):
            expand_grid({"fast": []})

    def test_组合数超上限要报错(self):
        with pytest.raises(ValueError, match="超过上限"):
            expand_grid({"a": list(range(50)), "b": list(range(50))})

    def test_上限是合理的(self):
        assert MAX_COMBINATIONS >= 100


def _wavy(n: int = 300, code: str = "600519"):
    """造一段有涨有跌的行情，否则任何参数都赚或都亏，测不出差异。"""
    closes = []
    price = 10.0
    for i in range(n):
        price *= 1 + (0.006 if (i // 20) % 2 == 0 else -0.005)
        closes.append(round(price, 2))
    return build_series(closes, code=code)


class TestSweep:
    def test_返回所有合法组合(self):
        rows = sweep(
            _wavy(),
            "ma_cross",
            {"fast": [3, 5], "slow": [20, 60]},
            config=EngineConfig(initial_cash=100_000.0),
        )
        assert len(rows) == 4

    def test_非法组合被跳过而不是崩掉(self):
        """fast=20 配 slow=20 是非法的，网格里出现这种组合是常态。"""
        rows = sweep(
            _wavy(),
            "ma_cross",
            {"fast": [3, 20], "slow": [20]},
            config=EngineConfig(initial_cash=100_000.0),
        )
        assert len(rows) == 1  # 只有 fast=3 合法
        assert rows[0].params == {"fast": 3, "slow": 20}

    def test_全部非法时报错(self):
        with pytest.raises(ValueError, match="都不合法"):
            sweep(
                _wavy(),
                "ma_cross",
                {"fast": [20], "slow": [10]},
                config=EngineConfig(initial_cash=100_000.0),
            )

    def test_按样本内收益降序(self):
        rows = sweep(
            _wavy(),
            "ma_cross",
            {"fast": [3, 5, 10], "slow": [20, 60]},
            config=EngineConfig(initial_cash=100_000.0),
        )
        returns = [r.is_return for r in rows]
        assert returns == sorted(returns, reverse=True)

    def test_样本内外真的分开了(self):
        rows = sweep(
            _wavy(),
            "ma_cross",
            {"fast": [5], "slow": [20]},
            config=EngineConfig(initial_cash=100_000.0),
            split=0.6,
        )
        row = rows[0]
        assert row.out_of_sample is not None
        # 样本外区间必须晚于样本内
        assert row.out_of_sample["start"] > row.in_sample["end"]

    def test_切分比例非法要报错(self):
        with pytest.raises(ValueError, match="split"):
            sweep(
                _wavy(),
                "ma_cross",
                {"fast": [5], "slow": [20]},
                split=1.5,
            )


class TestOverfitReport:
    def _row(self, params, is_ret, oos_ret):
        return SweepRow(
            params=params,
            in_sample={"total_return": is_ret},
            out_of_sample={"total_return": oos_ret},
        )

    def test_统计各组的排名(self):
        rows = [
            self._row({"p": 1}, 0.50, -0.10),  # 样本内第一、样本外最差
            self._row({"p": 2}, 0.30, 0.05),
            self._row({"p": 3}, 0.10, 0.20),  # 样本外第一
        ]
        report = overfit_report(rows)

        assert report["best_in_sample"].params == {"p": 1}
        assert report["best_out_of_sample"].params == {"p": 3}
        assert report["oos_rank_of_best_is"] == 3
        assert report["is_rank_of_best_oos"] == 3
        assert report["mean_is_return"] == pytest.approx(0.30)

    def test_空列表要报错(self):
        with pytest.raises(ValueError):
            overfit_report([])


class TestFormat:
    def test_报告里同时出现样本内和样本外(self):
        rows = [
            SweepRow({"fast": 3, "slow": 20}, {"total_return": 0.3}, {"total_return": 0.1}),
            SweepRow({"fast": 5, "slow": 20}, {"total_return": 0.1}, {"total_return": -0.2}),
        ]
        text = format_sweep(rows, 0.7)
        assert "样本内收益" in text
        assert "样本外收益" in text
        assert "样本内冠军" in text
        assert "fast=3 slow=20" in text

    def test_过拟合时给出警告(self):
        # 样本内最好的那组，样本外恰好最差 —— 典型的过拟合形态
        rows = [
            SweepRow(
                {"p": i},
                {"total_return": 0.50 - i * 0.05},
                {"total_return": -0.50 + i * 0.05},
            )
            for i in range(10)
        ]
        text = format_sweep(rows, 0.7)
        assert "过拟合" in text
