"""多股票组合。

实现方式是**把本金拆成 N 个独立子账户**：每只股票各自跑一遍单股回测，
最后把每天的权益相加。

**这个做法的局限要说清楚：子账户之间不会再平衡。** A 股票赚了钱不会挪给
B 股票用，B 股票亏光了 A 也不会来救。真实的组合管理会定期调仓 ——
那是另一个层次的复杂度，这一版不做。

**但它有个明确的好处：其余的一切都是对的。** T+1、涨跌停、整手、
5 元最低佣金、涨跌停的日期规则 —— 每只股票都按真实规则单独结算，
不因为套了个"组合"的壳就放松。而且它完全复用单股引擎，
不存在两套逻辑对不上的风险。

另外注意：本金被拆成 N 份之后，每份都更小，**5 元最低佣金的影响会放大**。
这不是 bug，是真实情况 —— 小账户本来就更容易被手续费吃掉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Sequence

from .engine import EngineConfig, run_backtest
from .models import BacktestResult, Series
from .strategies import build


@dataclass
class PortfolioResult:
    """组合回测结果。"""

    codes: list[str]
    names: dict[str, str]
    weights: dict[str, float]
    dates: list[date] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    cash: list[float] = field(default_factory=list)
    position_value: list[float] = field(default_factory=list)
    per_stock: dict[str, BacktestResult] = field(default_factory=dict)
    initial_cash: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_backtest_result(self, strategy_name: str) -> BacktestResult:
        """合并成单个 BacktestResult，好复用现有的指标和报告。

        成交、交易、被拒订单直接拼在一起 —— 组合层面看这些是汇总量，
        想看某只股票的去 per_stock 里找。
        """
        combined = BacktestResult(
            code="+".join(self.codes),
            name=f"{len(self.codes)} 只股票组合",
            strategy_name=strategy_name,
            dates=list(self.dates),
            equity=list(self.equity),
            cash=list(self.cash),
            position_value=list(self.position_value),
            position_pct=[
                (pv / eq) if eq > 0 else 0.0 for pv, eq in zip(self.position_value, self.equity)
            ],
            initial_cash=self.initial_cash,
            warnings=list(self.warnings),
        )
        for result in self.per_stock.values():
            combined.fills.extend(result.fills)
            combined.rejections.extend(result.rejections)
            combined.trades.extend(result.trades)
            combined.open_qty += result.open_qty
            combined.open_cost += result.open_cost
        combined.fills.sort(key=lambda f: f.date)
        combined.trades.sort(key=lambda t: t.entry_date)
        combined.rejections.sort(key=lambda r: r.date)
        return combined


def normalize_weights(codes: Sequence[str], weights: Sequence[float] | None) -> dict[str, float]:
    if weights is None:
        equal = 1.0 / len(codes)
        return {code: equal for code in codes}
    if len(weights) != len(codes):
        raise ValueError(f"权重个数（{len(weights)}）和股票个数（{len(codes)}）对不上")
    if any(w < 0 for w in weights):
        raise ValueError("权重不能为负")
    total = sum(weights)
    if total <= 0:
        raise ValueError("权重之和必须为正")
    return {code: w / total for code, w in zip(codes, weights)}


def _align(
    dates_in: Sequence[date],
    values_in: Sequence[float],
    dates_out: Sequence[date],
    fallback: float,
) -> list[float]:
    """把一条曲线对齐到统一日期轴上。

    停牌日、未上市期间、退市之后沿用上一个有效值；
    在那只股票有数据之前，它的子账户就是一笔没动过的现金。
    """
    lookup = dict(zip(dates_in, values_in))
    aligned: list[float] = []
    last = fallback
    for d in dates_out:
        if d in lookup:
            last = lookup[d]
        aligned.append(last)
    return aligned


def run_portfolio(
    codes: Sequence[str],
    strategy_name: str,
    params: dict[str, Any] | None = None,
    config: EngineConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    weights: Sequence[float] | None = None,
    loader: Callable[[str], Series] | None = None,
) -> PortfolioResult:
    """跑一个多股票组合。

    Args:
        codes: 股票代码列表，如 ["000001", "600519"]
        strategy_name: 策略名，所有股票共用同一个策略
        params: 策略参数
        config: 引擎配置。`initial_cash` 是**组合总本金**，会按权重拆开
        weights: 各股票权重，默认等权
        loader: 取行情的函数（测试里注入假数据用）
    """
    if not codes:
        raise ValueError("至少要给一只股票")
    if len(set(codes)) != len(codes):
        raise ValueError("股票代码有重复")

    config = config or EngineConfig()
    params = params or {}
    allocation = normalize_weights(codes, weights)
    if loader is None:
        raise ValueError("必须先注入 loader（正常调用走 cache.get_backtest_series）")

    per_stock: dict[str, BacktestResult] = {}
    series_by_code: dict[str, Series] = {}
    skipped: list[tuple[str, str]] = []

    for code in codes:
        # 成分股里混进几只"这段区间还没上市"的次新股是常态（指数会纳入新上市的），
        # 单只跑不起来不该让整个组合失败 —— 跳过它，把它那份钱留作现金。
        try:
            series = loader(code)
        except Exception as exc:  # 取数失败也归到这一类
            skipped.append((code, f"取不到行情：{exc}"))
            continue

        sub_config = EngineConfig(
            initial_cash=config.initial_cash * allocation[code],
            fees=config.fees,
            slippage_bps=config.slippage_bps,
            price_eps=config.price_eps,
            apply_new_listing_rule=config.apply_new_listing_rule,
            liquidate_at_end=config.liquidate_at_end,
        )
        try:
            per_stock[code] = run_backtest(
                series, build(strategy_name, **params), sub_config, start, end
            )
        except ValueError as exc:
            skipped.append(
                (code, f"{exc}（该股数据区间 {series.first_date} → {series.last_date}）")
            )
            continue

        series_by_code[code] = series

    if not per_stock:
        detail = "; ".join(f"{c}: {why}" for c, why in skipped[:3])
        raise ValueError(f"组合里所有股票都没能跑起来。{detail}")

    # 统一日期轴 = 所有子账户日期的并集
    all_dates = sorted({d for result in per_stock.values() for d in result.dates})
    if not all_dates:
        raise ValueError("组合里的股票在回测区间内都没有数据")

    total = 0.0
    equity_curves: list[list[float]] = []
    cash_curves: list[list[float]] = []
    position_curves: list[list[float]] = []
    for code, result in per_stock.items():
        base = config.initial_cash * allocation[code]
        equity_curves.append(_align(result.dates, result.equity, all_dates, base))
        # 现金曲线在数据开始前也是 base（子账户拿着一笔还没用的现金）
        cash_curves.append(_align(result.dates, result.cash, all_dates, base))
        # 持仓市值在数据开始前是 0（还没买任何东西）
        position_curves.append(_align(result.dates, result.position_value, all_dates, 0.0))
        total += base

    # 被跳过的股票：它那份钱全程是现金，一分没动。
    # 必须算进总权益，否则组合本金就凭空少了一块，收益率会失真。
    skipped_cash = sum(config.initial_cash * allocation[code] for code, _ in skipped)
    if skipped_cash:
        equity_curves.append([skipped_cash] * len(all_dates))
        cash_curves.append([skipped_cash] * len(all_dates))
        position_curves.append([0.0] * len(all_dates))
        total += skipped_cash

    combined_equity = [sum(curve[i] for curve in equity_curves) for i in range(len(all_dates))]
    combined_cash = [sum(curve[i] for curve in cash_curves) for i in range(len(all_dates))]
    combined_position = [sum(curve[i] for curve in position_curves) for i in range(len(all_dates))]

    portfolio = PortfolioResult(
        codes=list(per_stock),
        names={code: series_by_code[code].name for code in per_stock},
        weights={code: allocation[code] for code in per_stock},
        dates=all_dates,
        equity=combined_equity,
        cash=combined_cash,
        position_value=combined_position,
        per_stock=per_stock,
        initial_cash=total,
    )

    if skipped:
        # 带上原因。全说成"没有数据"是错的 —— 取数失败和还没上市是两回事。
        shown = "；".join(f"{code}: {why}" for code, why in skipped[:3])
        more = f"；另有 {len(skipped) - 3} 只" if len(skipped) > 3 else ""
        portfolio.warnings.append(
            f"跳过 {len(skipped)} 只股票（{shown}{more}）。"
            f"它们分到的资金全程留作现金，没有参与交易。"
        )

    _add_diagnostics(portfolio, config)
    return portfolio


def _add_diagnostics(portfolio: PortfolioResult, config: EngineConfig) -> None:
    """把组合特有的坑提示出来。"""
    n = len(portfolio.codes)

    # 把子账户的警告带上来。但**不能一条一条地堆** —— 300 只股票的组合同一条
    # "买不起一手"能刷满整个屏幕。所以同类警告合并成一条。
    unaffordable: list[str] = []
    others: list[str] = []
    for code, result in portfolio.per_stock.items():
        for warning in result.warnings:
            if "资金不足以买入一手" in warning:
                unaffordable.append(portfolio.names.get(code, code))
            else:
                others.append(f"[{portfolio.names.get(code, code)}] {warning}")

    if unaffordable:
        shown = "、".join(unaffordable[:5])
        more = f" 等 {len(unaffordable)} 只" if len(unaffordable) > 5 else ""
        per_account = config.initial_cash / n
        portfolio.warnings.append(
            f"有 {len(unaffordable)} 只股票因为分到的资金买不起一手而全程没有交易"
            f"（{shown}{more}）。组合本金 {config.initial_cash:,.0f} 元均分到 {n} 只，"
            f"每只只有 {per_account:,.0f} 元。请提高 --cash，或用 --limit 减少股票数量。"
        )
    portfolio.warnings.extend(others)

    # 有效交易日数量差距很大 → 说明有些股票大部分时间根本不在场
    counts = {code: len(r.dates) for code, r in portfolio.per_stock.items()}
    if counts:
        shortest = min(counts, key=lambda c: counts[c])
        longest = max(counts, key=lambda c: counts[c])
        if counts[shortest] < counts[longest] * 0.8:
            portfolio.warnings.append(
                f"{portfolio.names[shortest]}（{shortest}）在回测区间内只有 "
                f"{counts[shortest]} 个交易日，而 {portfolio.names[longest]}（{longest}）"
                f"有 {counts[longest]} 个。差距这么大通常是上市晚、退市或长期停牌 —— "
                f"这段时间它的那部分资金一直是闲置现金，会拉低组合的资金利用率。"
            )

    # 小资金 + 均分 = 最低佣金放大
    per_account = config.initial_cash / n
    if per_account < 20_000:
        portfolio.warnings.append(
            f"组合本金被均分成 {n} 份，每份只有 {per_account:,.0f} 元。"
            f"5 元最低佣金对小账户的拖累是成倍的（一笔 10 万元的交易和一笔 1 万元的"
            f"交易，佣金都是 5 元起），所以组合跑出来的收益率天然会低于逐只单独跑。"
        )
