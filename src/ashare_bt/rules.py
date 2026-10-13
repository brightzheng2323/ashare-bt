"""A 股交易规则：涨跌停、整手、板块识别、价格取整。

这些规则**不是一成不变的** —— 历史上改过好几次。回测跨过变更日时必须用对，
否则算出来的收益率是错的。所以涨跌停用的是一张「生效日期表」，不是单一常数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

# ── 板块 ────────────────────────────────────────────────────────────
SSE_MAIN = "SSE_MAIN"  # 上交所主板
SZSE_MAIN = "SZSE_MAIN"  # 深交所主板
STAR = "STAR"  # 科创板
CHINEXT = "CHINEXT"  # 创业板
BSE = "BSE"  # 北交所
OTHER = "OTHER"  # 认不出来

BOARD_LABEL = {
    SSE_MAIN: "沪市主板",
    SZSE_MAIN: "深市主板",
    STAR: "科创板",
    CHINEXT: "创业板",
    BSE: "北交所",
    OTHER: "未知板块",
}


def normalize_code(code: str) -> str:
    """把各种写法统一成 6 位纯代码。

    接受 "600519"、"sh600519"、"SH600519"、"1.600519"。
    """
    text = str(code).strip().upper()
    if "." in text:  # "1.600519"
        text = text.split(".", 1)[1]
    text = re.sub(r"^(SH|SZ|BJ)", "", text)  # "sh600519"
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError(f"认不出来的股票代码: {code!r}（应该像 600519 或 sh600519）")
    return text


def to_secid(code: str) -> str:
    """转成东方财富要的 secid：1.=沪市，0.=深市。

    **指数代码和股票代码会撞车**，所以指数必须显式写市场前缀：
        000001 → 0.000001 就是平安银行（深市个股）
        000001 → 1.000001 才是上证指数（沪市指数）
    本项目要用的沪深300是 1.000300，不显式指定会被当成深市个股，取不到数据。
    """
    text = str(code).strip().upper()
    if "." in text:
        market, bare = text.split(".", 1)
        if market in ("0", "1") and re.fullmatch(r"\d{6}", bare):
            return f"{market}.{bare}"  # 显式指定了市场，照办

    bare = normalize_code(text)
    if bare.startswith(("6", "9")):  # 6xxxxx 沪市主板/科创板；9 是 B 股
        return f"1.{bare}"
    if bare.startswith(("0", "2", "3")):  # 深市
        return f"0.{bare}"
    # 北交所 4xxxxx / 8xxxxx 在东财走 0. 前缀
    return f"0.{bare}"


def board_of(code: str) -> str:
    bare = normalize_code(code)
    if bare.startswith("688"):
        return STAR  # 688xxx 科创板
    if bare.startswith(("300", "301")):
        return CHINEXT
    if bare.startswith("60"):
        return SSE_MAIN
    if bare.startswith(("000", "001", "002", "003")):
        return SZSE_MAIN
    if bare.startswith(("4", "8")):
        return BSE  # 830xxx / 43xxxx / 87xxxx 等
    return OTHER


def is_st(name: str) -> bool:
    """从股票名称判断是不是 ST 股。

    ST / *ST / S*ST 都含 "ST"。"退" 是退市整理期。
    光凭代码分不出来，所以 Series 里必须存 name。
    """
    return "ST" in name.upper() or "退" in name


# ── 涨跌停 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _LimitRule:
    board: str
    is_st: bool
    effective_from: date
    pct: float


# 查表规则：同一 (板块, 是否ST) 有多条时取 effective_from 最晚且已生效的那条
_LIMIT_RULES: tuple[_LimitRule, ...] = (
    # 沪深主板：普通股一直 ±10%；ST 股 2026-07-06 起也从 ±5% 改为 ±10%
    _LimitRule(SSE_MAIN, False, date(1996, 12, 16), 0.10),
    _LimitRule(SSE_MAIN, True, date(1996, 12, 16), 0.05),
    _LimitRule(SSE_MAIN, True, date(2026, 7, 6), 0.10),
    _LimitRule(SZSE_MAIN, False, date(1996, 12, 16), 0.10),
    _LimitRule(SZSE_MAIN, True, date(1996, 12, 16), 0.05),
    _LimitRule(SZSE_MAIN, True, date(2026, 7, 6), 0.10),
    # 创业板：2020-08-24 注册制改革，从 ±10% 改为 ±20%（ST 同为 20%）
    _LimitRule(CHINEXT, False, date(2009, 10, 30), 0.10),
    _LimitRule(CHINEXT, True, date(2009, 10, 30), 0.05),
    _LimitRule(CHINEXT, False, date(2020, 8, 24), 0.20),
    _LimitRule(CHINEXT, True, date(2020, 8, 24), 0.20),
    # 科创板：2019-07-22 开市即 ±20%
    _LimitRule(STAR, False, date(2019, 7, 22), 0.20),
    _LimitRule(STAR, True, date(2019, 7, 22), 0.20),
    # 北交所：±30%
    _LimitRule(BSE, False, date(2021, 11, 15), 0.30),
    _LimitRule(BSE, True, date(2021, 11, 15), 0.30),
)


def limit_pct(code: str, name: str, on: date) -> float | None:
    """返回某天该股票的涨跌幅限制（0.10 表示 ±10%）。

    返回 None 表示没有涨跌幅限制 —— 认不出的板块，或者上市初期。
    """
    board = board_of(code)
    st = is_st(name)
    best: _LimitRule | None = None
    for rule in _LIMIT_RULES:
        if rule.board != board or rule.is_st != st:
            continue
        if rule.effective_from <= on and (best is None or rule.effective_from > best.effective_from):
            best = rule
    return best.pct if best is not None else None


def new_listing_exempt(board: str, bars_since_listing: int) -> bool:
    """上市初期不设涨跌幅限制。

    沪深主板 / 创业板 / 科创板：前 5 个交易日
    北交所：只有上市首日
    """
    if board == BSE:
        return bars_since_listing == 0
    return bars_since_listing < 5


def round_half_up(x: float, digits: int = 2) -> float:
    """四舍五入 —— 交易所的算法。

    Python 内置的 round() 是「银行家舍入」：round(2.675, 2) 得到 2.67，
    而交易所会算成 2.68。差这一分钱会直接改变涨跌停价，进而改变成交与否。

    先把 float 转成字符串再进 Decimal，避免二进制浮点误差。
    """
    quant = Decimal(1).scaleb(-digits)
    return float(Decimal(str(x)).quantize(quant, rounding=ROUND_HALF_UP))


def limit_prices(prev_close: float, pct: float) -> tuple[float, float]:
    """由昨收价算出 (涨停价, 跌停价)，四舍五入到分。"""
    return (
        round_half_up(prev_close * (1 + pct)),
        round_half_up(prev_close * (1 - pct)),
    )


# ── 整手 ────────────────────────────────────────────────────────────


def lot_size(code: str) -> int:
    """买入的最小单位（股）。

    沪深主板、创业板：100 股，且必须是 100 的整数倍
    科创板：200 股起，之后 1 股递增
    """
    return 200 if board_of(code) == STAR else 100


def floor_lot(qty: int, code: str) -> int:
    """把想买的股数向下取整到合法数量。买不了就返回 0。"""
    lot = lot_size(code)
    if qty < lot:
        return 0
    if board_of(code) == STAR:
        return qty  # 科创板 200 股起，之后 1 股递增，不用再取整
    return (qty // lot) * lot
