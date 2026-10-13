"""参数寻优。

**这个模块最重要的作用不是"找到最好的参数"，而是让你亲眼看到过拟合。**

做法是把回测区间切成两段：

- **样本内**（前 70%）：拿来找哪个参数最好
- **样本外**（后 30%）：拿来检验

如果样本内最优的那组参数到了样本外表现稀烂，说明它找到的不是规律，
而是背下了答案。这是量化里最贵的一课，越早看到越好。

反过来，如果样本内好、样本外也好，那也只能说"没被证伪"，
不等于明天会继续有效。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from .engine import EngineConfig, run_backtest
from .metrics import summarize
from .models import Series
from .strategies import build

MAX_COMBINATIONS = 2000
"""参数组合数上限。防止有人写个 10 个参数各 10 个取值 = 100 亿次回测。"""

DEFAULT_SPLIT = 0.7
"""样本内占比。70% 用来挑参数，30% 用来检验。"""


@dataclass(frozen=True)
class SweepRow:
    """一组参数在样本内/样本外的表现。"""

    params: dict[str, Any]
    in_sample: dict[str, Any]
    out_of_sample: dict[str, Any] | None = None

    @property
    def is_return(self) -> float:
        return float(self.in_sample["total_return"])

    @property
    def oos_return(self) -> float | None:
        if self.out_of_sample is None:
            return None
        return float(self.out_of_sample["total_return"])

    @property
    def decay(self) -> float | None:
        """样本外相对样本内的收益衰减。负得越多说明过拟合越严重。"""
        oos = self.oos_return
        return None if oos is None else oos - self.is_return


def expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """把 {参数: [取值...]} 展开成所有组合。"""
    if not grid:
        raise ValueError("参数网格不能为空")
    for name, values in grid.items():
        if not values:
            raise ValueError(f"参数 {name} 没有给任何取值")

    keys = list(grid)
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]

    if len(combos) > MAX_COMBINATIONS:
        raise ValueError(
            f"参数组合有 {len(combos)} 组，超过上限 {MAX_COMBINATIONS}。"
            f"少放几个取值，或者分两次跑。"
        )
    return combos


def _split_days(days: Sequence[date], split: float) -> tuple[date, date]:
    """把交易日序列按比例切成两段，返回 (样本内最后一天, 样本外第一天)。"""
    if not 0 < split < 1:
        raise ValueError("split 必须落在 (0, 1) 之间")
    if len(days) < 10:
        raise ValueError(f"只有 {len(days)} 个交易日，太少了，没法切分样本内/外")

    idx = int(len(days) * split)
    idx = max(1, min(len(days) - 1, idx))
    return days[idx - 1], days[idx]


def sweep(
    series: Series,
    strategy_name: str,
    grid: dict[str, Sequence[Any]],
    config: EngineConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    split: float = DEFAULT_SPLIT,
    on_progress: Any = None,
) -> list[SweepRow]:
    """扫一遍参数网格，每组参数都做样本内 + 样本外两段回测。

    返回的列表按**样本内收益**从高到低排序 —— 也就是"如果只看历史，
    你会挑哪一组"。样本外的成绩同时列出来，用来打脸。
    """
    config = config or EngineConfig()
    combos = expand_grid(grid)

    # 先剔掉非法组合再动手。参数网格里出现非法搭配是常态
    # （比如 fast=3,5,20 配 slow=20，其中 20/20 就不合法），
    # 得跳过它继续扫，不能让它把整个寻优带崩。
    # 注意这一步必须在试跑之前做 —— 否则试跑自己就会先炸。
    valid: list[dict[str, Any]] = []
    for params in combos:
        try:
            build(strategy_name, **params)
        except ValueError:
            continue
        valid.append(params)

    skipped = len(combos) - len(valid)
    if not valid:
        raise ValueError(
            f"所有参数组合都不合法（共 {len(combos)} 组）。检查一下取值组合，"
            f"比如 ma_cross 要求 fast 必须小于 slow。"
        )

    # 先用第一组合法参数跑一次，拿到实际交易日序列，好按**真实交易日**切分
    probe = run_backtest(series, build(strategy_name, **valid[0]), config, start, end)
    if len(probe.dates) < 10:
        raise ValueError(f"回测区间只有 {len(probe.dates)} 个交易日，太少，没法做寻优")

    is_end, oos_start = _split_days(probe.dates, split)

    rows: list[SweepRow] = []
    for i, params in enumerate(valid, 1):
        if on_progress is not None:
            on_progress(i, len(valid), params)

        in_sample = summarize(
            run_backtest(series, build(strategy_name, **params), config, start, is_end),
            None,
            config.fees,
        )
        out_of_sample = summarize(
            run_backtest(series, build(strategy_name, **params), config, oos_start, end),
            None,
            config.fees,
        )
        rows.append(SweepRow(params=params, in_sample=in_sample, out_of_sample=out_of_sample))

    if skipped:
        # 挂到第一行上，让报告层能提示用户
        rows[0].in_sample["skipped_combinations"] = skipped

    rows.sort(key=lambda r: r.is_return, reverse=True)
    return rows


def overfit_report(rows: Sequence[SweepRow], top_n: int = 5) -> dict[str, Any]:
    """把"过拟合有多严重"量化出来。

    三组对比：
    - 样本内第一名，它在样本外排第几
    - 样本外第一名是谁，它在样本内排第几
    - 全部参数的平均表现（说明策略本身有没有边际）
    """
    if not rows:
        raise ValueError("没有结果")

    best_is = rows[0]
    by_oos = sorted(rows, key=lambda r: (r.oos_return is None, -(r.oos_return or 0)))
    best_oos = by_oos[0]

    oos_rank_of_best_is = next(
        (i for i, r in enumerate(by_oos, 1) if r.params == best_is.params), None
    )
    is_rank_of_best_oos = next(
        (i for i, r in enumerate(rows, 1) if r.params == best_oos.params), None
    )

    is_returns = [r.is_return for r in rows]
    oos_returns = [r.oos_return for r in rows if r.oos_return is not None]

    return {
        "count": len(rows),
        "best_in_sample": best_is,
        "best_out_of_sample": best_oos,
        "oos_rank_of_best_is": oos_rank_of_best_is,
        "is_rank_of_best_oos": is_rank_of_best_oos,
        "top": list(rows[:top_n]),
        "mean_is_return": sum(is_returns) / len(is_returns),
        "mean_oos_return": (sum(oos_returns) / len(oos_returns)) if oos_returns else None,
        "positive_oos_ratio": (
            sum(1 for r in oos_returns if r > 0) / len(oos_returns) if oos_returns else None
        ),
    }


def _fmt_params(params: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())


def format_sweep(rows: Sequence[SweepRow], split: float, top_n: int = 15) -> str:
    """渲染成终端表格。"""
    line = "─" * 72
    report = overfit_report(rows, top_n)

    skipped = rows[0].in_sample.get("skipped_combinations", 0) if rows else 0

    out = [line]
    out.append(
        f" 参数寻优   共 {report['count']} 组参数   样本内 : 样本外 = {split:.0%} : {1 - split:.0%}"
        + (f"   （跳过 {skipped} 组非法组合）" if skipped else "")
    )
    out.append(line)
    out.append(f" {'参数':<34}{'样本内收益':>12}{'样本外收益':>12}{'衰减':>12}")
    out.append(line)

    for row in report["top"]:
        decay = row.decay
        decay_text = "     n/a" if decay is None else f"{decay * 100:>+11.2f}%"
        oos_text = "     n/a" if row.oos_return is None else f"{row.oos_return * 100:>+11.2f}%"
        out.append(
            f" {_fmt_params(row.params):<34}{row.is_return * 100:>+11.2f}%{oos_text}{decay_text}"
        )

    out.append(line)
    best_is = report["best_in_sample"]
    best_oos = report["best_out_of_sample"]

    out.append(f" 样本内冠军   {_fmt_params(best_is.params)}")
    out.append(f"   样本内 {best_is.is_return * 100:+.2f}%   样本外 {best_is.oos_return * 100:+.2f}%")
    if report["oos_rank_of_best_is"] is not None:
        out.append(f"   → 它在样本外排第 {report['oos_rank_of_best_is']} / {report['count']}")
    out.append("")
    out.append(f" 样本外冠军   {_fmt_params(best_oos.params)}")
    out.append(f"   样本内 {best_oos.is_return * 100:+.2f}%   样本外 {best_oos.oos_return * 100:+.2f}%")
    if report["is_rank_of_best_oos"] is not None:
        out.append(f"   → 它在样本内只排第 {report['is_rank_of_best_oos']} / {report['count']}")
    out.append("")

    mean_is = report["mean_is_return"]
    mean_oos = report["mean_oos_return"]
    out.append(f" 全部参数平均   样本内 {mean_is * 100:+.2f}%" + (f"   样本外 {mean_oos * 100:+.2f}%" if mean_oos is not None else ""))
    if report["positive_oos_ratio"] is not None:
        out.append(f" 样本外为正的比例  {report['positive_oos_ratio']:.0%}")
    out.append(line)

    if report["oos_rank_of_best_is"] and report["oos_rank_of_best_is"] > max(3, report["count"] // 2):
        out.append(" ⚠️  样本内冠军在样本外掉到了中下游 —— 典型的过拟合。")
        out.append("    换一组参数样本内就差一点点、样本外却好得多，那种才更值得看。")
        out.append(line)

    return "\n".join(out)
