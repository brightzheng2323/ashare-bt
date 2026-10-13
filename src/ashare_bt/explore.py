"""探索：把一只股票的各项数据摊开来看看。

不是回测，是**体检**。回答的是"这只股票现在是什么状态、我这些策略各自
想说啥" —— 在动手写新策略之前，先看看数据长什么样通常比直接调参有效。

所有指标都通过 `History` 视图计算，和引擎里跑的是同一套代码，
所以这里显示的信号和回测里的信号一定一致（不会出现"探索说买、回测没买"）。
"""

from __future__ import annotations

from datetime import date

from .chips import ChipDistribution
from .indicators import bollinger, macd, rsi, sma
from .models import Series
from .rules import BOARD_LABEL, board_of, is_st, limit_pct
from .strategies import REGISTRY, build
from .strategies.base import History

CHIP_LOOKBACK = 500
"""算筹码分布时回看的 K 线根数，和 ChipPeak 的默认值保持一致。"""


def _bar(ratio: float, width: int = 40) -> str:
    """按占比画横条。ratio 是相对峰值的比例（0 ~ 1）。"""
    if ratio <= 0:
        return ""
    return "█" * max(1, int(round(ratio * width)))


def build_chips(series: Series, index: int, lookback: int = CHIP_LOOKBACK) -> ChipDistribution:
    """把筹码分布推进到 index 这一天。"""
    chips = ChipDistribution(bins=400)
    first = max(0, index - lookback + 1)
    for j in range(first, index + 1):
        bar = series.bars[j]
        avg = (bar.high + bar.low + bar.close) / 3.0
        chips.update(bar.high, bar.low, avg, bar.turnover)
    return chips


def chip_chart(chips: ChipDistribution, close: float, rows: int = 18) -> list[str]:
    """把筹码分布画成横条图，只在现价处标一条线。

    只标现价，不标平均成本那些 —— 一条参考线足够，多了反而看不清分布本身。
    """
    prices = chips.prices()
    weights = chips.weights

    # 只展示有筹码的价格段，否则绝大部分格子是 0，图会很难看
    active = [k for k, w in enumerate(weights) if w > 1e-4]
    if not active:
        return ["  （没有筹码数据 —— 换手率可能缺失）"]

    lo_idx, hi_idx = active[0], active[-1]
    span = hi_idx - lo_idx + 1
    step = max(1, span // rows)  # 抽稀到大约 rows 行，免得终端刷屏

    picked = list(range(lo_idx, hi_idx + 1, step))
    peak = max(weights[k] for k in picked) or 1.0

    close_idx = min(range(len(prices)), key=lambda k: abs(prices[k] - close))
    out: list[str] = []
    for k in reversed(picked):
        marker = " ◀ 现价" if abs(k - close_idx) < step else ""
        out.append(f"  {prices[k]:>9.2f} │{_bar(weights[k] / peak):<40} {weights[k]:>6.2%}{marker}")
    return out


def indicator_panel(history: History) -> list[str]:
    out: list[str] = []
    close = history.close[-1]

    for window in (5, 10, 20, 60, 120):
        value = sma(history.close, window)
        if value is None:
            out.append(f"  MA{window:<4} 数据不足")
        else:
            gap = (close / value - 1) * 100
            where = "上方" if gap >= 0 else "下方"
            out.append(f"  MA{window:<4} {value:>9.2f}    现价在均线{where} {abs(gap):>5.2f}%")

    rsi_value = rsi(history.close, 14)
    out.append(f"  RSI14  {'数据不足' if rsi_value is None else f'{rsi_value:>9.2f}'}")

    macd_value = macd(history.close)
    if macd_value is None:
        out.append("  MACD   数据不足")
    else:
        dif, dea, hist = macd_value
        cross = "金叉状态（DIF 在 DEA 上方）" if dif > dea else "死叉状态（DIF 在 DEA 下方）"
        out.append(f"  MACD   DIF {dif:>8.3f}  DEA {dea:>8.3f}  柱 {hist:>8.3f}   {cross}")

    bands = bollinger(history.close, 20, 2.0)
    if bands is None:
        out.append("  布林带 数据不足")
    else:
        lower, middle, upper = bands
        pos = (close - lower) / (upper - lower) * 100 if upper > lower else 50.0
        out.append(f"  布林带 {lower:>8.2f} ~ {upper:>8.2f}   现价位于带内 {pos:>5.1f}% 处")

    return out


def strategy_signals(history: History) -> list[tuple[str, str]]:
    """问每个内置策略：按你现在看到的，你想拿多少仓位。"""
    out: list[tuple[str, str]] = []
    for name in sorted(REGISTRY):
        strategy = build(name)
        try:
            target = strategy.target(history)
        except Exception as exc:  # 策略出错不该让整个探索挂掉
            out.append((name, f"报错：{exc}"))
            continue
        out.append((name, "满仓" if target >= 1.0 else "空仓" if target <= 0.0 else f"{target:.0%}"))
    return out


def explore(series: Series, as_of: date | None = None, recent: int = 10) -> str:
    """生成探索报告。"""
    index = series.index_or_before(as_of) if as_of else len(series.bars) - 1
    if index is None:
        raise ValueError(f"{as_of} 之前没有数据")

    bar = series.bars[index]
    history = History(series, index)

    line = "─" * 72
    out: list[str] = [line]

    board = board_of(series.code)
    tags = [BOARD_LABEL[board]]
    if is_st(series.name):
        tags.append("ST")
    pct = limit_pct(series.code, series.name, bar.date)
    tags.append("无涨跌停限制" if pct is None else f"涨跌停 ±{pct:.0%}")

    out.append(f" {series.code} {series.name}   {' · '.join(tags)}")
    out.append(f" 截至 {bar.date}   收盘 {bar.close:.2f}")
    out.append(line)

    out.append(" 数据概览")
    out.append(f"   全历史 {series.first_date} → {series.last_date}   共 {len(series.bars)} 根 K 线")
    out.append(f"   区间内最高 {max(b.high for b in series.bars):.2f}   最低 {min(b.low for b in series.bars):.2f}")
    if series.dropped_bars:
        out.append(f"   ⚠️  丢弃了 {series.dropped_bars} 根非正价数据（前复权副作用）")
    out.append(line)

    out.append(f" 最近 {recent} 个交易日")
    out.append(f"   {'日期':<12}{'开盘':>9}{'最高':>9}{'最低':>9}{'收盘':>9}{'涨跌':>9}{'换手':>8}")
    for j in range(max(0, index - recent + 1), index + 1):
        b = series.bars[j]
        prev = series.bars[j - 1].close if j > 0 else b.close
        change = (b.close / prev - 1) * 100 if prev else 0.0
        out.append(
            f"   {b.date.isoformat():<12}{b.open:>9.2f}{b.high:>9.2f}{b.low:>9.2f}"
            f"{b.close:>9.2f}{change:>+8.2f}%{b.turnover:>7.2f}%"
        )
    out.append(line)

    out.append(" 技术指标")
    out.extend(indicator_panel(history))
    out.append(line)

    chips = build_chips(series, index)
    state = chips.state(bar.close)
    out.append(" 筹码分布")
    out.append(
        f"   获利盘 {state.profit_ratio:.1%}   平均成本 {state.avg_cost:.2f}   "
        f"90% 成本区间 {state.cost_low:.2f} ~ {state.cost_high:.2f}   集中度 {state.concentration:.3f}"
    )
    out.append("")
    out.extend(chip_chart(chips, bar.close))
    out.append(line)

    out.append(" 各策略此刻的信号")
    for name, signal in strategy_signals(history):
        out.append(f"   {name:<14} {signal}")
    out.append(line)
    out.append(" 提示：这些信号是用「今天收盘价」算出来的，引擎会到明天的开盘价才成交。")
    out.append(line)

    return "\n".join(out)
