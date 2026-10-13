"""回测引擎。

主循环每个交易日做五件事，**顺序不能变**：

    1. 结算   T+1：昨天买的今天变成可卖
    2. 成交   执行【昨天收盘后】挂的单，按【今日开盘价】成交
    3. 盯市   用今日收盘价算权益，记进曲线
    4. 决策   策略看一眼"到今天为止"的历史，说出目标仓位
    5. 挂单   给明天开盘准备订单

关键点在于：**信号和成交永远差一天**。第 4 步产生的信号存进 `pending`，
一直等到下一轮循环的第 2 步才被执行 —— 两者是不同迭代里的不同变量，
中间只有 `pending` 这一条通道。所以没有任何代码路径能在当日收盘价成交，
未来函数不是"我们小心避免"，而是**结构上写不出来**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .fees import DEFAULT_FEES, FeeConfig
from .models import (
    BacktestResult,
    Bar,
    Fill,
    Order,
    Position,
    RejectReason,
    Rejection,
    Series,
    Side,
    Trade,
)
from .rules import (
    STAR,
    board_of,
    floor_lot,
    limit_pct,
    limit_prices,
    lot_size,
    new_listing_exempt,
)
from .strategies.base import History, Strategy


@dataclass(frozen=True)
class EngineConfig:
    initial_cash: float = 100_000.0
    """初始资金（元）。"""

    fees: FeeConfig = DEFAULT_FEES

    slippage_bps: float = 0.0
    """滑点，单位万分之一。默认 0（按开盘价精确成交）。

    真实交易里买价会略高、卖价会略低。设成 10 就是千分之一。
    """

    price_eps: float = 0.005
    """判断"是否触及涨跌停"的容差（元）。半分钱，用来吸收浮点误差。"""

    apply_new_listing_rule: bool = True
    """是否启用「新股上市初期不限价」规则。"""

    liquidate_at_end: bool = False
    """回测结束时是否强制平仓。

    默认**不**平仓：在最后一天的收盘价上卖出，等于"看着今天收盘价做决定"，
    是轻微的未来函数，而且要多付一笔手续费。所以默认保留持仓，
    在报告里单独列出浮动盈亏。
    """


@dataclass
class _Account:
    """可变的账户状态。引擎内部用，不对外暴露。"""

    cash: float
    position: Position = field(default_factory=Position)


@dataclass
class _OpenLot:
    """尚未平仓的持仓成本，用来算一笔完整交易的盈亏。"""

    qty: int = 0
    gross: float = 0.0  # 买入金额合计（不含费用）
    fees: float = 0.0  # 买入费用合计
    first_date: date | None = None

    @property
    def avg_price(self) -> float:
        return self.gross / self.qty if self.qty else 0.0

    def add(self, fill: Fill) -> None:
        if self.qty == 0:
            self.first_date = fill.date
        self.qty += fill.qty
        self.gross += fill.amount
        self.fees += fill.fee

    def clear(self) -> None:
        self.qty = 0
        self.gross = 0.0
        self.fees = 0.0
        self.first_date = None


def max_affordable_qty(cash: float, price: float, code: str, fees: FeeConfig) -> int:
    """在给定资金和价格下，最多能买多少股（已经过整手处理）。

    不能用 `cash / price` 直接算 —— 那样会把手续费漏掉，
    结果就是"买得起"但实际下单时资金不足。
    单笔最低 5 元佣金让小资金的算法更麻烦，所以这里用上界估计 + 逐步回退。
    """
    if price <= 0 or cash <= 0:
        return 0

    rate = fees.commission_rate + fees.transfer_fee_rate
    rough = int(cash / (price * (1 + rate)))
    qty = floor_lot(rough, code)
    step = 1 if board_of(code) == STAR else lot_size(code)

    while qty > 0:
        amount = qty * price
        if amount + fees.buy_fee(amount) <= cash + 1e-9:
            return qty
        qty -= step
    return 0


def build_order(
    target: float,
    account: _Account,
    bar: Bar,
    code: str,
    created: date,
    config: EngineConfig,
) -> Order | None:
    """把「目标仓位」翻译成「明天开盘要下的单」。

    买多少只能按**今天的收盘价**估算 —— 明天的开盘价还不知道，
    真正的股数要到成交时才能定下来（见 execute_order）。
    """
    if target == 0.0:
        if account.position.total > 0:
            return Order(Side.SELL, account.position.total, created)
        return None

    if target == 1.0:
        if account.position.total > 0:
            return None  # 已经满仓，不用动
        qty = max_affordable_qty(account.cash, bar.close, code, config.fees)
        if qty <= 0:
            return None
        return Order(Side.BUY, qty, created)

    raise NotImplementedError(
        f"当前版本只支持目标仓位 0.0 或 1.0，收到 {target!r}。"
        "（部分仓位需要额外的成本核算逻辑。）"
    )


def execute_order(
    order: Order,
    bar: Bar,
    prev_close: float,
    series: Series,
    account: _Account,
    config: EngineConfig,
    bar_index: int,
) -> Fill | Rejection:
    """尝试执行一张订单。返回成交或被拒的原因。

    检查顺序是有讲究的 —— 先查市场能不能成交（涨跌停），
    再查账户有没有资格（T+1），最后才调整数量（整手、资金）。
    """
    code = series.code

    # ── 1. 涨跌停 ───────────────────────────────────────────────
    # 涨停买不进、跌停卖不出 —— 这是 A 股和美股最直观的差别之一。
    exempt = False
    if config.apply_new_listing_rule:
        # bars_since_listing 不能直接用 bar_index —— 前复权负价被丢弃的那一段
        # 也是真实交易日。茅台丢弃了 3526 根，它序列里的第 0 根其实是上市后
        # 第 3526 个交易日，绝不属于"上市初期"。
        exempt = new_listing_exempt(board_of(code), series.dropped_bars + bar_index)

    if not exempt:
        pct = limit_pct(code, series.name, bar.date)
        if pct is not None:
            upper, lower = limit_prices(prev_close, pct)
            if order.side is Side.BUY and bar.open >= upper - config.price_eps:
                return Rejection(bar.date, order.side, order.qty, RejectReason.LIMIT_UP)
            if order.side is Side.SELL and bar.open <= lower + config.price_eps:
                return Rejection(bar.date, order.side, order.qty, RejectReason.LIMIT_DOWN)

    # ── 2. T+1 ─────────────────────────────────────────────────
    if order.side is Side.SELL and order.qty > account.position.sellable:
        return Rejection(bar.date, order.side, order.qty, RejectReason.T1)

    # ── 3. 成交价（开盘价 + 滑点，并夹在当日价格区间内）──────────
    price = _fill_price(bar, order.side, config.slippage_bps)

    # ── 4. 数量与资金 ───────────────────────────────────────────
    qty = order.qty
    if order.side is Side.BUY:
        # 按**实际成交价**重算一遍买得起多少，而不是沿用昨天收盘价估的那个数 ——
        # 隔夜跳空高开时，昨天算出的股数今天就买不起了。
        affordable = max_affordable_qty(account.cash, price, code, config.fees)
        qty = min(order.qty, affordable)
        if qty <= 0:
            return Rejection(bar.date, order.side, order.qty, RejectReason.CASH)

        # 不够一手的部分向下取整。买入必须是整手，
        # 所以 150 股实际只能买 100 股；不足一手则整单作废。
        qty = floor_lot(qty, code)
        if qty <= 0:
            return Rejection(bar.date, order.side, order.qty, RejectReason.LOT)
    else:
        # 卖出不做整手校验（送股产生的零股要能一次性卖掉）。
        # 数量上面已经由 T+1 检查保证不超过可卖数，这里不用再夹。
        qty = order.qty

    amount = qty * price
    fee = config.fees.fee(order.side, amount)

    # ── 5. 记账 ─────────────────────────────────────────────────
    if order.side is Side.BUY:
        account.cash -= amount + fee
        account.position.add_buy(qty)  # 只进 total，不进 sellable → 今天卖不掉
    else:
        account.cash += amount - fee
        account.position.apply_sell(qty)

    return Fill(date=bar.date, side=order.side, qty=qty, price=price, amount=amount, fee=fee)


def _fill_price(bar: Bar, side: Side, slippage_bps: float) -> float:
    """开盘价 + 滑点，再夹回当日价格区间。

    滑点方向对买方不利、对卖方也不利 —— 这才是真实情况。
    """
    if slippage_bps <= 0:
        return bar.open
    drift = bar.open * slippage_bps / 10_000.0
    price = bar.open + drift if side is Side.BUY else bar.open - drift
    return min(max(price, bar.low), bar.high)


def run_backtest(
    series: Series,
    strategy: Strategy,
    config: EngineConfig | None = None,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    """跑一次回测。

    注意：**引擎在完整历史上跑，只在记录时截取回测区间。**
    这不是浪费 —— 如果先把数据切成 [start, end] 再算指标，
    回测第一天的 MA20 就是"用不足 20 天的数据算出来的"，
    开头几十天的信号全是错的。这是回测代码里非常常见的 bug。
    """
    config = config or EngineConfig()
    bars = series.bars

    start_i = series.index_or_after(start) if start is not None else 0
    end_i = series.index_or_before(end) if end is not None else len(bars) - 1
    if start_i is None or end_i is None:
        raise ValueError("回测区间内没有交易日")

    if start_i > end_i:
        raise ValueError("回测区间内没有交易日")

    strategy.reset()

    account = _Account(cash=config.initial_cash)
    pending: Order | None = None
    open_lot = _OpenLot()

    result = BacktestResult(
        code=series.code,
        name=series.name,
        strategy_name=strategy.describe(),
        initial_cash=config.initial_cash,
    )

    unaffordable_days = 0
    one_lot_shortfall: tuple[float, int] | None = None

    for i in range(start_i, end_i + 1):
        bar = bars[i]
        # 算今日涨跌停要用昨收。第 0 根没有昨收（用不到，因为此时必然没有挂单）。
        prev_close = bars[i - 1].close if i > 0 else bar.open

        # ── 1. 结算：昨天买的今天可卖 ──────────────────────────
        account.position.settle()

        # ── 2. 成交：执行昨天挂的单，按今天开盘价 ──────────────
        if pending is not None:
            outcome = execute_order(pending, bar, prev_close, series, account, config, i)
            if isinstance(outcome, Fill):
                result.fills.append(outcome)
                if outcome.side is Side.BUY:
                    open_lot.add(outcome)
                else:
                    result.trades.append(_close_trade(open_lot, outcome))
                    open_lot.clear()
            else:
                result.rejections.append(outcome)
            pending = None  # 无论成交还是被拒，订单只挂一天

        # ── 3. 盯市：用今天收盘价算权益 ────────────────────────
        position_value = account.position.total * bar.close
        equity = account.cash + position_value
        result.dates.append(bar.date)
        result.equity.append(equity)
        result.cash.append(account.cash)
        result.position_value.append(position_value)
        result.position_pct.append(position_value / equity if equity > 0 else 0.0)

        # ── 4. 决策：只能看到"到今天为止" ──────────────────────
        if i == end_i:
            # 最后一天产生的信号要到下一天才可能成交，而那天已经出了区间。
            # 直接跳过 —— 否则要么撞上"未来函数"，要么下标越界。
            break
        target = strategy.target(History(series, i))

        # ── 5. 挂单：给明天开盘 ────────────────────────────────
        pending = build_order(target, account, bar, series.code, bar.date, config)

        # 想满仓却一手都买不起 —— 记下来。否则用户只会看到"收益 0%"，
        # 完全不知道是资金门槛的问题（茅台一手就要十几万）。
        if pending is None and target == 1.0 and account.position.total == 0:
            lot = lot_size(series.code)
            unaffordable_days += 1
            if one_lot_shortfall is None:
                one_lot_shortfall = (lot * bar.close, lot)

    if unaffordable_days and one_lot_shortfall is not None:
        need, lot = one_lot_shortfall
        result.warnings.append(
            f"资金不足以买入一手：{series.name} 价格约 {need / lot:,.2f} 元，"
            f"一手（{lot} 股）需要 {need:,.0f} 元，超过账户资金 "
            f"{config.initial_cash:,.0f} 元。共有 {unaffordable_days} 个交易日出现这种情况。"
            f"请用 --cash 提高本金，或换一只价格较低的股票。"
        )

    if config.liquidate_at_end and account.position.total > 0:
        _liquidate(series, bars[end_i], account, open_lot, result, config)
    else:
        result.open_qty = account.position.total
        result.open_cost = open_lot.gross + open_lot.fees

    return result


def _close_trade(open_lot: _OpenLot, fill: Fill) -> Trade:
    """把一次卖出和它对应的持仓配对，算出一笔完整交易的盈亏。"""
    entry_value = open_lot.gross + open_lot.fees
    exit_value = fill.amount - fill.fee
    pnl = exit_value - entry_value
    entry_date = open_lot.first_date or fill.date
    return Trade(
        entry_date=entry_date,
        entry_price=open_lot.avg_price,
        exit_date=fill.date,
        exit_price=fill.price,
        qty=fill.qty,
        fee=open_lot.fees + fill.fee,
        pnl=pnl,
        return_pct=pnl / entry_value if entry_value else 0.0,
        holding_days=(fill.date - entry_date).days,
    )


def _liquidate(
    series: Series,
    bar: Bar,
    account: _Account,
    open_lot: _OpenLot,
    result: BacktestResult,
    config: EngineConfig,
) -> None:
    """回测结束时按收盘价强制平仓。

    `--liquidate-at-end` 时才会走到这里。在收盘价卖了又按同一个收盘价算权益，
    等于"看着今天的收盘价做决定"，是个轻微的假设，所以默认关闭。
    """
    qty = account.position.total
    if qty <= 0:
        return
    amount = qty * bar.close
    fee = config.fees.sell_fee(amount)
    account.cash += amount - fee
    account.position.apply_sell(qty)

    fill = Fill(date=bar.date, side=Side.SELL, qty=qty, price=bar.close, amount=amount, fee=fee)
    result.fills.append(fill)
    result.trades.append(_close_trade(open_lot, fill))
    open_lot.clear()

    # 修正最后一根权益（原来按持仓市值记的，现在变成现金了）
    if result.equity:
        result.equity[-1] = account.cash
        result.cash[-1] = account.cash
        result.position_value[-1] = 0.0
        result.position_pct[-1] = 0.0
