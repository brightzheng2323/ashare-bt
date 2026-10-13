"""技术指标。

**所有函数都是「因果」的**：只读取传入序列的末尾若干元素，返回「截至最后一个
元素」的值，绝不使用前向或居中窗口 —— 那是未来函数，会让回测收益虚高得离谱，
而且几乎不可能从结果里看出来。

调用约定：
- 输入是「到今日为止」的序列，通常直接传 `h.close`
- 返回 float，数据还不够时返回 `None`（不要返回 0 —— 0 是合法的价格）
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def sma(values: Sequence[float], window: int, min_periods: int | None = None) -> float | None:
    """简单移动平均。返回最新 window 个数的均值。"""
    if window <= 0:
        raise ValueError("window 必须为正整数")
    need = window if min_periods is None else min_periods
    if len(values) < need:
        return None
    tail = values[-window:]
    return sum(tail) / len(tail)


def _ema_series(values: Sequence[float], window: int) -> list[float | None]:
    """指数移动平均的完整序列，长度与输入一致。

    前 window-1 个位置是 None（数据不够）。种子用前 window 个数的简单平均 ——
    这是通行做法，也让 EMA 的值可复现。
    """
    out: list[float | None] = [None] * len(values)
    if window <= 0 or len(values) < window:
        return out
    multiplier = 2.0 / (window + 1)
    prev = sum(values[:window]) / window
    out[window - 1] = prev
    for i in range(window, len(values)):
        prev = values[i] * multiplier + prev * (1 - multiplier)
        out[i] = prev
    return out


def ema(values: Sequence[float], window: int) -> float | None:
    """指数移动平均的最新值。"""
    series = _ema_series(values, window)
    return series[-1] if series else None


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    """Wilder RSI，0 ~ 100。

    用的是 Wilder 本人的平滑算法（不是简单平均），所以需要从头算 ——
    每根 K 线重算一次是 O(n)，2600 根一共约 700 万次运算，可以忽略。
    """
    if period <= 0:
        raise ValueError("period 必须为正整数")
    if len(values) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(delta if delta > 0 else 0.0)
        losses.append(-delta if delta < 0 else 0.0)

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float] | None:
    """MACD。返回 (DIF, DEA, 柱状值)。

    DIF = EMA(fast) - EMA(slow)
    DEA = DIF 的 signal 日 EMA
    柱  = 2 × (DIF - DEA)          ← A 股常见的画法，乘了 2
    """
    if not 0 < fast < slow:
        raise ValueError("必须满足 0 < fast < slow")
    if len(values) < slow + signal:
        return None

    ema_fast = _ema_series(values, fast)
    ema_slow = _ema_series(values, slow)

    # 注意这里是**跳过** None 而不是遇到 None 就 break。
    # 两条 EMA 的预热长度不同（fast 短、slow 长），开头一段 fast 有值而 slow 没有；
    # 写成 break 的话循环在第一个元素就退出了，dif_full 永远是空列表，
    # 于是 macd() 永远返回 None —— 而且调用方拿到 None 只会安静地空仓，不报错。
    dif_full = [
        f - s for f, s in zip(ema_fast, ema_slow) if f is not None and s is not None
    ]

    if len(dif_full) < signal:
        return None

    dea_series = _ema_series(dif_full, signal)
    dea = dea_series[-1]
    if dea is None:
        return None
    dif = dif_full[-1]
    return dif, dea, 2.0 * (dif - dea)


def bollinger(
    values: Sequence[float],
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[float, float, float] | None:
    """布林带。返回 (下轨, 中轨, 上轨)。

    中轨是简单均线，上下轨是 ±num_std 倍标准差。
    标准差用总体口径（除以 window 而不是 window-1）—— 这是布林带的惯例。
    """
    if window <= 0:
        raise ValueError("window 必须为正整数")
    if len(values) < window:
        return None

    tail = values[-window:]
    middle = sum(tail) / window
    variance = sum((x - middle) ** 2 for x in tail) / window
    std = math.sqrt(variance)
    return middle - num_std * std, middle, middle + num_std * std


def highest(values: Sequence[float], window: int) -> float | None:
    """最近 window 个数的最大值。"""
    if len(values) < window:
        return None
    return max(values[-window:])


def lowest(values: Sequence[float], window: int) -> float | None:
    """最近 window 个数的最小值。"""
    if len(values) < window:
        return None
    return min(values[-window:])
