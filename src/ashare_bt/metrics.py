"""绩效指标。

计算方式本身就有很多坑：年化用 365 天会低估约 40%、最大回撤算在收益率上而不是
权益上、夏普用年度数据点算。这里每个函数都只保留一种正确算法。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

from .fees import FeeConfig
from .models import BacktestResult, Fill, RejectReason, Series, Side, Trade

TRADING_DAYS_PER_YEAR = 252
"""A 股一年大约 243 个交易日，惯例用 252 折算。绝不能用 365 —— 那是自然日。"""


def daily_returns(equity: Sequence[float]) -> list[float]:
    """逐日收益率。"""
    return [
        equity[i] / equity[i - 1] - 1.0
        for i in range(1, len(equity))
        if equity[i - 1] > 0
    ]


def total_return(equity: Sequence[float]) -> float:
    """累计收益率。"""
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    return equity[-1] / equity[0] - 1.0


def annualized_return(equity: Sequence[float], periods: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """年化收益率。

    用的是**交易日**数而不是自然日。一年约 243 个交易日，惯例按 252 折算。
    如果误按 365 算：2600 根日线本是 10.3 年，会被算成 7.1 年 ——
    年数变小，年化收益就被**高估**约四成。
    """
    n = len(equity)
    if n < 2 or equity[0] <= 0 or equity[-1] <= 0:
        return None
    years = (n - 1) / periods
    if years <= 0:
        return None
    return (equity[-1] / equity[0]) ** (1.0 / years) - 1.0


def max_drawdown(equity: Sequence[float]) -> tuple[float, int, int]:
    """最大回撤。返回 (回撤幅度(负数), 峰值下标, 谷值下标)。

    必须算在**权益**上而不是收益率上。收益率算出来的回撤是错的 ——
    下跌 50% 再涨 50% 并不会回到原点，把百分比直接相加就丢掉了这个复利效应。
    """
    if not equity:
        return 0.0, 0, 0

    peak = equity[0]
    peak_idx = 0
    worst = 0.0
    worst_peak, worst_trough = 0, 0

    for i, value in enumerate(equity):
        if value > peak:
            peak = value  # 创新高就重置峰值
            peak_idx = i
        if peak > 0:
            drawdown = value / peak - 1.0
            if drawdown < worst:
                worst = drawdown
                worst_peak, worst_trough = peak_idx, i

    return worst, worst_peak, worst_trough


def sharpe(
    equity: Sequence[float],
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """夏普比率。

    用**逐日**收益率算再乘 √252。用年度权益点算会得到完全不同的数，
    而且样本太少没有统计意义。

    收益完全没有波动时（比如全程空仓）分母是 0，返回 None 而不是除零。
    """
    returns = daily_returns(equity)
    if len(returns) < 2:
        return None

    excess = [r - risk_free / periods for r in returns]
    mean = sum(excess) / len(excess)
    variance = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return mean / std * math.sqrt(periods)


def trade_stats(trades: Sequence[Trade]) -> dict[str, Any]:
    """交易层面的统计。"""
    if not trades:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_holding_days": None,
            "best": None,
            "worst": None,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = sum(t.pnl for t in losses)

    if gross_loss == 0:
        # 一次都没亏过 —— 盈亏比是无穷大，用 inf 表示，报告里显示成「∞」
        profit_factor: float | None = math.inf if gross_profit > 0 else None
    else:
        profit_factor = gross_profit / abs(gross_loss)

    return {
        "trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": len(wins) / len(trades),
        "profit_factor": profit_factor,
        "avg_win": gross_profit / len(wins) if wins else None,
        "avg_loss": gross_loss / len(losses) if losses else None,
        "avg_holding_days": sum(t.holding_days for t in trades) / len(trades),
        "best": max(t.return_pct for t in trades),
        "worst": min(t.return_pct for t in trades),
    }


def fee_breakdown(fills: Sequence[Fill], fees: FeeConfig) -> dict[str, float]:
    """按佣金 / 印花税 / 过户费拆开统计总成本。"""
    commission = stamp = transfer = 0.0
    for fill in fills:
        parts = fees.breakdown(fill.side, fill.amount)
        commission += parts.commission
        stamp += parts.stamp
        transfer += parts.transfer
    return {
        "commission": commission,
        "stamp": stamp,
        "transfer": transfer,
        "total": commission + stamp + transfer,
    }


def buy_and_hold(series: Series, dates: Sequence[date], initial_cash: float) -> list[float]:
    """按给定日期序列做买入持有，得到基准权益曲线。

    用**策略自己的日期轴**，而不是基准自己的 —— 个股停牌的日子指数照样在交易，
    两边日期不一致就没法公平对比。停牌日沿用上一个有效价格。
    """
    curve: list[float] = []
    base: float | None = None
    for d in dates:
        idx = series.index_of(d)
        if idx is None:
            curve.append(curve[-1] if curve else initial_cash)
            continue
        close = series.bars[idx].close
        if base is None:
            base = close
        curve.append(initial_cash * close / base if base else initial_cash)
    return curve


def summarize(
    result: BacktestResult,
    benchmarks: dict[str, Sequence[float]] | None = None,
    fees: FeeConfig | None = None,
    label: str = "沪深300",
) -> dict[str, Any]:
    """把所有指标汇总成一个字典。报告的文本和 --json 都从这里取。"""
    equity = result.equity
    stats: dict[str, Any] = {
        "code": result.code,
        "name": result.name,
        "strategy": result.strategy_name,
        "start": result.dates[0].isoformat() if result.dates else None,
        "end": result.dates[-1].isoformat() if result.dates else None,
        "trading_days": len(result.dates),
        "initial_cash": result.initial_cash,
        "final_equity": equity[-1] if equity else result.initial_cash,
        "total_return": total_return(equity),
        "annualized_return": annualized_return(equity),
    }

    dd, dd_peak, dd_trough = max_drawdown(equity)
    stats["max_drawdown"] = dd
    stats["max_drawdown_peak_date"] = (
        result.dates[dd_peak].isoformat() if result.dates and dd_peak < len(result.dates) else None
    )
    stats["max_drawdown_trough_date"] = (
        result.dates[dd_trough].isoformat() if result.dates and dd_trough < len(result.dates) else None
    )
    stats["sharpe"] = sharpe(equity)

    stats.update(trade_stats(result.trades))

    if fees is not None:
        stats["fees"] = fee_breakdown(result.fills, fees)

    stats["rejections"] = len(result.rejections)
    stats["rejection_detail"] = {
        reason.label: sum(1 for r in result.rejections if r.reason is reason)
        for reason in RejectReason
        if any(r.reason is reason for r in result.rejections)
    }
    stats["open_qty"] = result.open_qty
    stats["open_cost"] = result.open_cost
    stats["warnings"] = list(result.warnings)

    if benchmarks:
        stats["benchmarks"] = {
            key: {
                "total_return": total_return(curve),
                "annualized_return": annualized_return(curve),
                "max_drawdown": max_drawdown(curve)[0],
                "sharpe": sharpe(curve),
            }
            for key, curve in benchmarks.items()
        }
        stats["benchmark_label"] = label

    return stats


def _pct(value: float | None, width: int = 8) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value * 100:>+{width}.2f}%"


def _num(value: float | None, width: int = 8, digits: int = 2) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    if math.isinf(value):
        return f"{'∞':>{width}}"
    return f"{value:>{width},.{digits}f}"


def format_report(stats: dict[str, Any]) -> str:
    """把指标字典渲染成终端里对齐的文本报告。"""
    line = "─" * 64
    out: list[str] = []
    out.append(line)
    bench_label = stats.get("benchmark_label", "基准")
    out.append(f" {stats['code']} {stats['name']}   策略: {stats['strategy']}")
    out.append(f" 区间: {stats['start']} → {stats['end']}   共 {stats['trading_days']} 个交易日")
    out.append(line)

    bench = stats.get("benchmarks", {}).get(bench_label)
    out.append(f" 初始资金       {_num(stats['initial_cash'], 14)}")
    open_note = ""
    if stats.get("open_qty"):
        open_note = f"   (含未平仓 {stats['open_qty']} 股，成本 {_num(stats['open_cost'], 12).strip()})"
    out.append(f" 期末权益       {_num(stats['final_equity'], 14)}{open_note}")
    out.append(
        f" 累计收益率     {_pct(stats['total_return'], 10)}"
        + (f"      基准 {_pct(bench['total_return'], 8).strip()}" if bench else "")
    )
    out.append(
        f" 年化收益率     {_pct(stats['annualized_return'], 10)}"
        + (f"      基准 {_pct(bench['annualized_return'], 8).strip()}" if bench else "")
    )
    out.append(
        f" 最大回撤       {_pct(stats['max_drawdown'], 10)}"
        + (f"      基准 {_pct(bench['max_drawdown'], 8).strip()}" if bench else "")
    )
    out.append(
        f" 夏普比率       {_num(stats['sharpe'], 10)}"
        + (f"      基准 {_num(bench['sharpe'], 8).strip()}" if bench else "")
    )
    out.append(line)

    out.append(f" 交易次数       {stats['trade_count']:>10}      " f"胜 {stats['win_count']} / 负 {stats['loss_count']}")
    out.append(f" 胜率           {_pct(stats['win_rate'], 10)}")
    out.append(f" 盈亏比         {_num(stats['profit_factor'], 10)}")
    out.append(f" 平均持仓天数   {_num(stats['avg_holding_days'], 10, 1)}")
    out.append(f" 单笔最好/最差  {_pct(stats['best'], 10)} / {_pct(stats['worst'], 8).strip()}")

    if "fees" in stats:
        f = stats["fees"]
        out.append(
            f" 总成本         {_num(f['total'], 10)}      "
            f"佣金 {_num(f['commission'], 9).strip()} / "
            f"印花税 {_num(f['stamp'], 8).strip()} / "
            f"过户费 {_num(f['transfer'], 7).strip()}"
        )

    rej = stats.get("rejection_detail") or {}
    if rej:
        detail = " / ".join(f"{k} {v}" for k, v in rej.items())
        out.append(f" 被拒订单       {stats['rejections']:>10}      {detail}")
    else:
        out.append(f" 被拒订单       {stats['rejections']:>10}")

    out.append(line)

    for warning in stats.get("warnings", []):
        out.append(f" ⚠️  {warning}")
    if stats.get("warnings"):
        out.append(line)

    return "\n".join(out)
