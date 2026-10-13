"""本地 CSV 缓存。

**为什么是 CSV 不是 parquet：** 2600 行 × 8 列大约 300 KB。CSV 用 Excel 或
Numbers 直接双击就能打开，数据对不上时一眼能看出问题；parquet 要多装一个
40 MB 的 pyarrow，这个体量下换不来任何东西。

**策略是整文件覆盖，绝不追加。**
- 全量历史本来就是一次 HTTP 请求（约 1 秒），省这一次没意义；
- 追加对后复权（fqt=2）是安全的，对前复权（fqt=1）是错的（分红后历史价会被重写）。
  既然只对部分模式安全，那就都不用 —— 简单且永远正确。
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from .errors import DataError
from .fetch import DEFAULT_FQT, FQT_BACKWARD, FQT_NONE, fetch_series
from .models import Bar, Series
from .rules import to_secid

DEFAULT_CACHE_DIR = Path("data/cache")

CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]

# 缓存多久算过期。1 天：今天下过就不再下，明天自动更新。
DEFAULT_TTL_DAYS = 1


def _paths(cache_dir: Path, secid: str, fqt: int) -> tuple[Path, Path]:
    stem = f"{secid.replace('.', '_')}_fqt{fqt}"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.meta.json"


def save(series: Series, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _paths(cache_dir, series.secid, series.fqt)

    # 先写临时文件再改名 —— 万一写到一半被打断，也不会留下半个 CSV
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for bar in series.bars:
            writer.writerow(
                [
                    bar.date.isoformat(),
                    f"{bar.open:.4f}",
                    f"{bar.high:.4f}",
                    f"{bar.low:.4f}",
                    f"{bar.close:.4f}",
                    f"{bar.volume:.0f}",
                    f"{bar.amount:.2f}",
                    f"{bar.turnover:.4f}",
                ]
            )
    tmp_path.replace(csv_path)

    meta = {
        "code": series.code,
        "secid": series.secid,
        "name": series.name,
        "fqt": series.fqt,
        "fetched_at": series.fetched_at.isoformat(),
        "rows": len(series.bars),
        "first_date": series.first_date.isoformat(),
        "last_date": series.last_date.isoformat(),
        "dropped_bars": series.dropped_bars,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load(secid: str, fqt: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> Series | None:
    """读缓存。文件不存在或损坏都返回 None（当作没缓存，重新下载即可）。"""
    csv_path, meta_path = _paths(cache_dir, secid, fqt)
    if not csv_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, ValueError, KeyError):
        return None

    if not rows:
        return None

    try:
        bars = tuple(
            Bar(
                date=date.fromisoformat(r["date"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
                amount=float(r["amount"]),
                turnover=float(r.get("turnover") or 0.0),
            )
            for r in rows
        )
        return Series(
            code=meta["code"],
            secid=meta["secid"],
            name=meta["name"],
            bars=bars,
            fqt=int(meta["fqt"]),
            fetched_at=datetime.fromisoformat(meta["fetched_at"]),
            dropped_bars=int(meta.get("dropped_bars", 0)),
        )
    except (ValueError, KeyError, TypeError):
        return None


def is_fresh(meta_path: Path, ttl_days: int) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
    except (OSError, ValueError, KeyError):
        return False
    return (datetime.now() - fetched_at) <= timedelta(days=ttl_days)


def get_series(
    code: str,
    fqt: int = DEFAULT_FQT,
    refresh: bool = False,
    ttl_days: int = DEFAULT_TTL_DAYS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Series:
    """拿一只股票的行情：优先读缓存，缓存过期或不存在就去下载。

    这是引擎和 CLI 唯一该调用的入口，不要直接用 fetch_series ——
    那样每次回测都要重新联网下载。
    """
    secid = to_secid(code)
    _, meta_path = _paths(cache_dir, secid, fqt)

    if not refresh and is_fresh(meta_path, ttl_days):
        cached = load(secid, fqt, cache_dir)
        if cached is not None:
            return cached

    series = fetch_series(code, fqt=fqt)
    save(series, cache_dir)
    return series


def rescale(series: Series, factor: float) -> Series:
    """把整条价格序列除以一个常数。

    **收益率完全不变** —— 所有价格同比例缩放，涨跌幅、盈亏比、回测结果一个都不动。
    变的只有绝对价位，也就是"这些钱能买几手"和报告里打印的数字。
    """
    if factor <= 0:
        raise ValueError("缩放系数必须为正")

    bars = tuple(
        Bar(
            date=b.date,
            open=b.open / factor,
            high=b.high / factor,
            low=b.low / factor,
            close=b.close / factor,
            volume=b.volume,
            amount=b.amount / factor,  # 成交额同比例缩放，保持「均价 = 额/量」自洽
            turnover=b.turnover,  # 换手率是无量纲的，不动
        )
        for b in series.bars
    )
    return series._replace(bars=bars)


def get_backtest_series(
    code: str,
    start: date | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> Series:
    """拿一份可以直接用于回测的行情。**回测入口应该用这个，而不是 get_series。**

    做两件事：

    1. **用后复权**（fqt=2）。前复权对高分红股票会把价格算成 0 甚至负数，
       实测工商银行 2014 年还有 "价格 0.04 元" —— 那种数据不是失真，是没有意义。

    2. **整体缩放**，让回测起点的价格等于那天的**真实成交价**。
       后复权的代价是绝对价位被放大（平安银行现在显示 1345 元，真实约 11 元），
       而价位决定了"10 万本金能买几手"。整手取整和 5 元最低佣金都按成交金额
       起作用，价位错了这两件事就全错。

    缩放需要真实价格，所以要额外拉一次不复权数据（会走缓存，很便宜）。
    拉不到就退回未缩放的序列 —— 收益率仍然正确，只是手数会失真。
    """
    adjusted = get_series(code, fqt=FQT_BACKWARD, cache_dir=cache_dir, refresh=refresh)

    probe = start or adjusted.first_date
    idx = adjusted.index_or_after(probe)
    if idx is None:
        return adjusted

    try:
        raw = get_series(code, fqt=FQT_NONE, cache_dir=cache_dir, refresh=refresh)
    except DataError:
        return adjusted

    raw_idx = raw.index_of(adjusted.bars[idx].date)
    if raw_idx is None:
        return adjusted

    real_price = raw.bars[raw_idx].close
    if real_price <= 0:
        return adjusted

    factor = adjusted.bars[idx].close / real_price
    if factor <= 0 or abs(factor - 1.0) < 1e-9:
        return adjusted
    return rescale(adjusted, factor)
