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
from .fetch import DEFAULT_FQT, fetch_series
from .models import Bar, Series
from .rules import normalize_code, to_secid

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
