"""图表输出。

**配色**用的是经过校验的调色板：分类槽位 1/2/3（蓝 #2a78d6 / 橙 #eb6834 /
青 #1baf7a）。这三个槽位在浅色和深色背景下都通过了色盲可分辨性检验
（最差配对 CVD ΔE 9.2 / 9.4，阈值 8）—— 不是靠眼睛挑的。

**买卖点用形状区分**（▲ 买 / ▼ 卖）而不是只靠红绿 —— 红绿色盲约占总人口 8%，
只靠颜色的话他们完全读不出这张图的买卖信号。

**三张子图共用一个 x 轴，权益图只有一根 y 轴。** 双 y 轴是图表里最容易误导人的
做法（两条曲线的交叉点完全由缩放决定），所以基准和策略都换算成「起点 = 1」再叠加。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # 无窗口后端，避免在无图形环境下卡住
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .models import BacktestResult, Series, Side  # noqa: E402

# ── 调色板（来自校验过的默认实例，不要随手改）────────────────────
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_1 = "#2a78d6"  # 蓝 —— 策略
SERIES_2 = "#eb6834"  # 橙 —— 沪深300
SERIES_3 = "#1baf7a"  # 青 —— 个股买入持有

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
STATUS_WARNING = "#fab219"

FONT_STACK = ["PingFang SC", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": FONT_STACK,
            "axes.unicode_minus": False,  # 否则负号在中文环境显示成方块
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.titlesize": 10,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def _recessive_axes(ax, grid_axis: str = "y") -> None:
    """让坐标轴退到背景里去 —— 数据才是主角。"""
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


def _index_to_one(values: Sequence[float]) -> list[float]:
    base = values[0] if values and values[0] else 1.0
    return [v / base for v in values]


def plot_result(
    result: BacktestResult,
    series: Series,
    benchmarks: Mapping[str, Sequence[float]] | None = None,
    out_path: Path | str = "output/result.png",
    title: str | None = None,
) -> Path:
    """画三张图：权益对比 / 回撤 / 价格与买卖点。"""
    _setup_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dates = [date.fromisoformat(d.isoformat()) for d in result.dates]
    # 价格序列按回测日期轴对齐，停牌日沿用上一个有效价
    closes: list[float] = []
    for d in result.dates:
        idx = series.index_of(d)
        closes.append(series.bars[idx].close if idx is not None else (closes[-1] if closes else 0.0))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, height_ratios=[2.2, 1.0, 1.6]
    )
    fig.subplots_adjust(hspace=0.12, top=0.92, bottom=0.07, left=0.07, right=0.93)

    # ── 子图 1：权益曲线（起点 = 1）──────────────────────────────
    lines: list[tuple[str, list[float], str]] = [
        ("策略", _index_to_one(result.equity), SERIES_1),
    ]
    for name, curve in (benchmarks or {}).items():
        color = SERIES_2 if "300" in name or "沪" in name else SERIES_3
        lines.append((name, _index_to_one(curve), color))

    for name, curve, color in lines:
        ax1.plot(dates, curve, linewidth=2.0, color=color, label=name, solid_capstyle="round")

    # 线尾直接标名字 —— 图例要来回对颜色，直标不用。
    # 三条曲线在末尾通常已经分开，真撞上了也能从图例读出来。
    for name, curve, _color in lines:
        ax1.annotate(
            name,
            xy=(dates[-1], curve[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    ax1.axhline(1.0, color=BASELINE, linewidth=0.8, linestyle="--", zorder=0)
    ax1.set_ylabel("净值（起点 = 1）", fontsize=9)
    ax1.legend(loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    ax1.set_title(title or f"{result.code} {result.name} · {result.strategy_name}", loc="left", color=INK_PRIMARY)
    _recessive_axes(ax1)

    # ── 子图 2：回撤 ────────────────────────────────────────────
    peak = result.equity[0] if result.equity else 1.0
    drawdown: list[float] = []
    for value in result.equity:
        peak = max(peak, value)
        drawdown.append((value / peak - 1.0) * 100 if peak else 0.0)

    ax2.fill_between(dates, drawdown, 0, color=STATUS_CRITICAL, alpha=0.25, linewidth=0)
    ax2.plot(dates, drawdown, linewidth=1.4, color=STATUS_CRITICAL)
    ax2.set_ylabel("回撤 %", fontsize=9)
    # 顶端的 0 要显示成 "0%" 而不是 "-0%" —— 浮点数会给出负零
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0%" if abs(v) < 0.5 else f"{v:.0f}%"))
    _recessive_axes(ax2)

    # ── 子图 3：价格与买卖点 ────────────────────────────────────
    ax3.plot(dates, closes, linewidth=1.0, color=INK_MUTED, label="收盘价")

    fills = [f for f in result.fills if date.fromisoformat(f.date.isoformat()) in set(dates)]
    date_index = {d: i for i, d in enumerate(dates)}
    for side, marker, color, name in (
        (Side.BUY, "^", STATUS_GOOD, "买入"),
        (Side.SELL, "v", STATUS_CRITICAL, "卖出"),
    ):
        xs = [date.fromisoformat(f.date.isoformat()) for f in fills if f.side is side]
        ys = [closes[date_index[date.fromisoformat(f.date.isoformat())]] for f in fills if f.side is side]
        if xs:
            ax3.plot(
                xs,
                ys,
                linestyle="none",
                marker=marker,
                markersize=7,
                color=color,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=name,
                zorder=5,
            )

    rejected = [r for r in result.rejections if date.fromisoformat(r.date.isoformat()) in date_index]
    if rejected:
        xs = [date.fromisoformat(r.date.isoformat()) for r in rejected]
        ys = [closes[date_index[date.fromisoformat(r.date.isoformat())]] for r in rejected]
        ax3.plot(
            xs,
            ys,
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=STATUS_WARNING,
            markeredgewidth=1.6,
            label="被拒（涨跌停/T+1）",
            zorder=4,
        )

    ax3.set_ylabel("价格（前复权）", fontsize=9)
    ax3.legend(loc="upper left", fontsize=8.5, ncol=4, labelcolor=INK_SECONDARY)
    _recessive_axes(ax3)

    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax3.get_xticklabels():
        label.set_rotation(0)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
