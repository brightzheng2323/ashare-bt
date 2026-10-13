"""命令行入口。

    backtest --code 600519 --start 2020-01-01 --end 2025-01-01 --strategy ma_cross
    backtest fetch --code 600519
    backtest inspect --code 600519
    backtest list-strategies

策略自己的参数直接写在后面就行（--fast 10 --slow 30），
不用改这个文件 —— 参数名会按策略的 __init__ 签名做校验，拼错了会明确报错。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .cache import get_backtest_series, get_series
from .engine import EngineConfig, run_backtest
from .errors import DataError
from .fetch import DEFAULT_FQT, FQT_BACKWARD, FQT_LABEL
from .fees import DEFAULT_FEES, FeeConfig
from .models import Series
from .metrics import buy_and_hold, format_report, summarize
from .plot import plot_result
from .strategies import build, describe_all

# 沪深300。必须写 "1." 前缀 —— 指数代码和个股代码会撞车，不带前缀会被当成深市个股
BENCHMARK_CODE = "1.000300"
BENCHMARK_NAME = "沪深300"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USER_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"fetch", "inspect", "list-strategies"}:
        return _SUBCOMMANDS[args[0]](args[1:])
    return cmd_backtest(args)


# ── 工具 ────────────────────────────────────────────────────────────


def _parse_date(text: str | None, label: str) -> date | None:
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise SystemExit(f"❌ {label} 格式不对: {text!r}，应该是 2020-01-01 这样") from None


def _coerce(value: str) -> Any:
    """命令行参数都是字符串，这里猜一下类型。"""
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _extra_params(extra: Sequence[str]) -> dict[str, Any]:
    """把剩下的 --fast 10 --slow 30 收成策略参数字典。"""
    params: dict[str, Any] = {}
    i = 0
    while i < len(extra):
        token = extra[i]
        if not token.startswith("--"):
            raise SystemExit(f"❌ 看不懂的参数: {token!r}")
        key = token[2:].replace("-", "_")
        if i + 1 >= len(extra):
            raise SystemExit(f"❌ {token} 后面要跟一个值，比如 {token} 10")
        params[key] = _coerce(extra[i + 1])
        i += 2
    return params


def _fee_config(args: argparse.Namespace) -> FeeConfig:
    return FeeConfig(
        commission_rate=args.commission,
        commission_min=args.commission_min,
        stamp_duty_rate=args.stamp_duty,
        transfer_fee_rate=args.transfer_fee,
    )


def _load_series(code: str, args: argparse.Namespace, start: date | None) -> Series:
    """按命令行参数取行情。

    默认走后复权 + 整体缩放（见 cache.get_backtest_series），
    这样报告里的价格接近真实成交价，"能买几手"也算得对。
    用户明确指定 --fqt 1/0 或 --no-rescale 时就不动它。
    """
    cache_dir = Path(args.cache_dir)
    if args.fqt == FQT_BACKWARD and not args.no_rescale:
        return get_backtest_series(code, start=start, cache_dir=cache_dir, refresh=args.refresh)
    return get_series(code, fqt=args.fqt, cache_dir=cache_dir, refresh=args.refresh)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# ── 子命令 ──────────────────────────────────────────────────────────


def cmd_list_strategies(argv: Sequence[str]) -> int:  # noqa: ARG001
    print("可用策略：\n")
    print(describe_all())
    print("\n策略参数直接跟在命令后面，例如：")
    print("  backtest --code 600519 --strategy ma_cross --fast 10 --slow 30")
    return EXIT_OK


def _code_parser(prog: str, desc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"backtest {prog}", description=desc)
    parser.add_argument("--code", "-c", required=True, help="股票代码，如 600519 / sh600519 / 1.600519")
    parser.add_argument("--fqt", type=int, default=DEFAULT_FQT, choices=[0, 1, 2], help="复权方式")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存，强制重新下载")
    parser.add_argument("--cache-dir", default="data/cache", help="缓存目录")
    return parser


def cmd_fetch(argv: Sequence[str]) -> int:
    args = _code_parser("fetch", "只下载数据并缓存，不跑回测").parse_args(list(argv))
    try:
        series = get_series(
            args.code, fqt=args.fqt, refresh=args.refresh, cache_dir=Path(args.cache_dir)
        )
    except DataError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return EXIT_USER_ERROR

    print(f"✅ {series.code} {series.name}  {FQT_LABEL[series.fqt]}")
    print(f"   共 {len(series.bars)} 根 K 线，{series.first_date} → {series.last_date}")
    if series.dropped_bars:
        print(f"   ⚠️  丢弃了 {series.dropped_bars} 根负价前导数据（前复权的副作用）")
    print("\n   最近 5 根：")
    for bar in series.bars[-5:]:
        print(
            f"     {bar.date}  开{bar.open:>9.2f} 高{bar.high:>9.2f} "
            f"低{bar.low:>9.2f} 收{bar.close:>9.2f}  "
            f"{bar.volume:>10,.0f}手  换手{bar.turnover:.2f}%"
        )
    return EXIT_OK


def cmd_inspect(argv: Sequence[str]) -> int:
    args = _code_parser("inspect", "数据体检：看看这份行情靠不靠谱").parse_args(list(argv))
    try:
        series = get_series(
            args.code, fqt=args.fqt, refresh=args.refresh, cache_dir=Path(args.cache_dir)
        )
    except DataError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return EXIT_USER_ERROR

    bars = series.bars
    prices = [b.close for b in bars]
    gaps = [
        (bars[i - 1].date, bars[i].date, (bars[i].date - bars[i - 1].date).days)
        for i in range(1, len(bars))
        if (bars[i].date - bars[i - 1].date).days > 10
    ]

    print(f"📋 {series.code} {series.name}   {FQT_LABEL[series.fqt]}")
    print(f"   条数        {len(bars)}")
    print(f"   区间        {series.first_date} → {series.last_date}")
    print(f"   价格区间    {min(prices):,.2f} ~ {max(prices):,.2f}")
    print(f"   数据新鲜度  {series.fetched_at:%Y-%m-%d %H:%M}")
    if series.dropped_bars:
        print(f"   ⚠️  丢弃      {series.dropped_bars} 根负价前导数据")
    if gaps:
        print(f"   长停牌       {len(gaps)} 处超过 10 个自然日：")
        for prev, cur, days in gaps[:5]:
            print(f"                {prev} → {cur}（{days} 天）")
    else:
        print("   长停牌       无")
    return EXIT_OK


# ── 回测 ────────────────────────────────────────────────────────────


def _backtest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="A 股策略回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  backtest --code 600519 --start 2020-01-01 --end 2025-01-01 --strategy ma_cross\n"
            "  backtest --code 300750 --strategy rsi --rsi-period 14 --cash 200000\n"
            "  backtest list-strategies\n"
        ),
    )
    parser.add_argument("--code", "-c", required=True, help="股票代码，如 600519")
    parser.add_argument("--start", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--strategy", "-s", default="ma_cross", help="策略名（见 list-strategies）")
    parser.add_argument("--cash", type=float, default=100_000.0, help="初始资金，默认 10 万")
    parser.add_argument("--fqt", type=int, default=DEFAULT_FQT, choices=[0, 1, 2], help="复权方式")

    parser.add_argument("--commission", type=float, default=DEFAULT_FEES.commission_rate)
    parser.add_argument("--commission-min", type=float, default=DEFAULT_FEES.commission_min)
    parser.add_argument("--stamp-duty", type=float, default=DEFAULT_FEES.stamp_duty_rate)
    parser.add_argument("--transfer-fee", type=float, default=DEFAULT_FEES.transfer_fee_rate)
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="滑点，万分之一为单位")

    parser.add_argument("--no-benchmark", action="store_true", help="不拉沪深300做对比")
    parser.add_argument("--liquidate-at-end", action="store_true", help="结束时按收盘价强制平仓")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重新下载")
    parser.add_argument(
        "--no-rescale",
        action="store_true",
        help="不做价格缩放（默认会缩放到回测起点的真实价位，让手数和费用算得对）",
    )
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--no-plot", action="store_true", help="不画图")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是格式化报告")
    parser.add_argument("--quiet", action="store_true")
    return parser


def cmd_backtest(argv: Sequence[str]) -> int:
    parser = _backtest_parser()
    args, extra = parser.parse_known_args(list(argv))

    try:
        strategy = build(args.strategy, **_extra_params(extra))
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return EXIT_USER_ERROR

    start = _parse_date(args.start, "--start")
    end = _parse_date(args.end, "--end")
    cache_dir = Path(args.cache_dir)

    try:
        series = _load_series(args.code, args, start)
        benchmarks: dict[str, list[float]] = {}

        config = EngineConfig(
            initial_cash=args.cash,
            fees=_fee_config(args),
            slippage_bps=args.slippage_bps,
            liquidate_at_end=args.liquidate_at_end,
        )
        result = run_backtest(series, strategy, config=config, start=start, end=end)

        if not args.no_benchmark:
            index_series = _load_series(BENCHMARK_CODE, args, start)
            bench = buy_and_hold(index_series, result.dates, args.cash)
            if any(v != args.cash for v in bench):
                benchmarks[BENCHMARK_NAME] = bench

        benchmarks[f"{series.name} 买入持有"] = buy_and_hold(series, result.dates, args.cash)
    except DataError as exc:
        print(f"❌ 取数据失败：{exc}", file=sys.stderr)
        return EXIT_USER_ERROR
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return EXIT_USER_ERROR

    # 数据没能覆盖用户要的区间时才提醒。
    # 门槛设在 10 天：元旦/春节这种"起始日不是交易日"是正常的，不该报警。
    if start is not None and result.dates:
        gap = (result.dates[0] - start).days
        if gap > 10 or start < series.first_date:
            reason = ""
            if series.dropped_bars:
                reason = f"（前复权算出负价，丢弃了最早的 {series.dropped_bars} 根）"
            result.warnings.append(
                f"你指定从 {start} 开始，但实际从 {result.dates[0]} 开始。"
                f"该股票可用数据始于 {series.first_date}{reason}。"
            )

    stats = summarize(result, benchmarks, fees=config.fees, label=BENCHMARK_NAME)

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    elif not args.quiet:
        print(format_report(stats))

    stem = f"{series.code}_{args.strategy}_{(end or result.dates[-1]).strftime('%Y%m%d')}"
    outdir = Path(args.outdir)

    if not args.json:
        _write_csv(
            outdir / f"{stem}_equity.csv",
            ["date", "equity", "cash", "position_value", "position_pct", "drawdown"],
            [
                [
                    result.dates[i].isoformat(),
                    f"{result.equity[i]:.2f}",
                    f"{result.cash[i]:.2f}",
                    f"{result.position_value[i]:.2f}",
                    f"{result.position_pct[i]:.4f}",
                    f"{(result.equity[i] / max(result.equity[: i + 1]) - 1):.6f}",
                ]
                for i in range(len(result.dates))
            ],
        )
        _write_csv(
            outdir / f"{stem}_trades.csv",
            ["entry_date", "entry_price", "exit_date", "exit_price", "qty", "fee", "pnl", "return_pct", "holding_days"],
            [
                [
                    t.entry_date.isoformat(),
                    f"{t.entry_price:.4f}",
                    t.exit_date.isoformat(),
                    f"{t.exit_price:.4f}",
                    t.qty,
                    f"{t.fee:.2f}",
                    f"{t.pnl:.2f}",
                    f"{t.return_pct:.6f}",
                    t.holding_days,
                ]
                for t in result.trades
            ],
        )
        if result.rejections:
            _write_csv(
                outdir / f"{stem}_rejections.csv",
                ["date", "side", "qty", "reason"],
                [
                    [r.date.isoformat(), r.side.label, r.qty, r.reason.label]
                    for r in result.rejections
                ],
            )

        if not args.no_plot:
            try:
                path = plot_result(
                    result, series, benchmarks, outdir / f"{stem}.png", title=None
                )
                if not args.quiet:
                    print(f"\n 图表: {path}")
                    print(f" 明细: {outdir}/{stem}_{{equity,trades,rejections}}.csv")
            except Exception as exc:  # 画图失败不该让整个回测白跑
                print(f"⚠️  画图失败（回测结果仍然有效）：{exc}", file=sys.stderr)

    return EXIT_OK


_SUBCOMMANDS = {
    "fetch": cmd_fetch,
    "inspect": cmd_inspect,
    "list-strategies": cmd_list_strategies,
}


if __name__ == "__main__":
    raise SystemExit(main())
