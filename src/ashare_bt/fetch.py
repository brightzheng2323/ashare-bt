"""东方财富行情接口。

非官方接口，但公开、免费、不需要 token，实测稳定。接口长这样：

    https://push2his.eastmoney.com/api/qt/stock/kline/get
      ?secid=1.600519              # 1.=沪市 0.=深市
      &fields1=f1,f2,f3,f4,f5,f6
      &fields2=f51,...,f61
      &klt=101                     # 101 = 日线
      &fqt=2                       # 0 不复权 / 1 前复权 / 2 后复权
      &beg=20200101&end=20251231

一次请求就能拿到全部历史（实测 2015 年至今 2600+ 根 K 线，不需要翻页）。
"""

from __future__ import annotations

import time
from datetime import date, datetime

import requests

from .errors import DataError
from .models import Bar, Series
from .rules import normalize_code, to_secid

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

# 字段顺序。注意 **第 2、3 个是「开、收」而不是「开、高」** ——
# 东财这里返回的是 O,C,H,L 顺序，不是通常的 OHLC。
# 按 OHLC 去读，会把收盘价当成最高价：程序照跑不误，结果全错。
FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
#        日期   开    收    高    低   成交量 成交额 振幅  涨跌幅 涨跌额 换手率

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

FQT_NONE = 0
FQT_FORWARD = 1  # 前复权
FQT_BACKWARD = 2  # 后复权 —— 回测默认用这个，见下

FQT_LABEL = {0: "不复权", 1: "前复权", 2: "后复权"}

DEFAULT_FQT = FQT_BACKWARD


def fetch_series(
    code: str,
    fqt: int = DEFAULT_FQT,
    beg: date | None = None,
    end: date | None = None,
    timeout: float = 20.0,
    retries: int = 3,
) -> Series:
    """拉取一只股票的完整日线历史。

    默认拉**全部**历史而不只是回测区间，原因有两个：
    1. 一次请求就够了，成本一样；
    2. 指标需要预热 —— 回测从 2020 年开始，MA20 在 2020 年第一天就该是热的，
       不能因为数据从这个区间开始就缺前 20 天。

    **为什么默认后复权（fqt=2）：**

    东财的前复权（fqt=1）对现金分红用的是**减法**：
        前复权价_t = 当时真实价 − 之后累计每股分红
    对分红历史长的股票，这个减法会把早年价格算成 0 甚至负数。实测：

        平安银行  1909 根 K 线非正价，一直到 2009 年
        工商银行   420 根非正价，2014 年还有"价格 0.04 元"
        贵州茅台     6 根非正价，一直到 2016 年

    0.04 元的股价不是"失真"，是**没有意义** —— 用它算出来的收益率、
    涨跌停、持仓手数全是垃圾。所以前复权在这里直接出局。

    后复权用的是**乘法**，永远为正、全历史有效、收益率正确。唯一的代价是
    绝对价位被放大（平安银行后复权价 1345 元，真实约 11 元），
    而价位决定了"10 万本金能买几手"。这个代价由 cache.get_backtest_series
    解决：把整条序列除以一个常数，让**回测起点**的价格等于当天真实成交价。
    常数缩放不改变任何收益率，只是把价位搬回真实量级。
    """
    bare = normalize_code(code)
    secid = to_secid(code)
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": FIELDS2,
        "klt": "101",
        "fqt": str(fqt),
        "beg": (beg or date(1990, 1, 1)).strftime("%Y%m%d"),
        "end": (end or date(2050, 1, 1)).strftime("%Y%m%d"),
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(KLINE_URL, params=params, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return parse_klines(resp.json(), secid, fqt)
        except (requests.RequestException, ValueError) as exc:
            # 网络类问题值得重试；数据类问题（下面的 DataError）不重试，直接抛
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * attempt)

    raise DataError(
        f"请求东方财富接口失败，重试 {retries} 次都没成功: {last_error}\n"
        f"参数: {params}\n"
        f"如果是网络问题，检查一下能否访问 push2his.eastmoney.com"
    )


CONSTITUENTS_URL = "https://push2.eastmoney.com/api/qt/clist/get"

CONSTITUENT_PAGE_SIZE = 100
"""这个接口单页最多给 100 条，要 300 只就得分 3 页拉。"""

INDEX_BOARDS: dict[str, tuple[str, str]] = {
    # index secid -> (东财板块代码, 名字)
    "1.000300": ("BK0500", "沪深300"),
    "1.000016": ("BK0611", "上证50"),
    "1.000905": ("BK0701", "中证500"),
}
"""指数代码 → 板块代码的映射。东财不认指数 secid，只认板块代码，
实测 BK0500 返回恰好 300 只、BK0611 恰好 50 只、BK0701 恰好 500 只。"""


def index_secid(code: str) -> str:
    """把指数代码补全成 secid。

    **指数和个股的编码规则不一样**，不能直接用 to_secid：
        000300 → 个股规则会算成 0.000300（深市个股），实际是 1.000300（沪市指数）
        000001 → 0.000001 是平安银行，1.000001 才是上证指数
    指数的规则是：000xxx / 88xxxx / 93xxxx 在沪市，399xxx 在深市。
    """
    text = str(code).strip().upper()
    if "." in text:
        return text
    bare = normalize_code(text)
    if bare.startswith("399"):
        return f"0.{bare}"
    return f"1.{bare}"


def fetch_index_constituents(
    index: str = "1.000300",
    limit: int | None = None,
    timeout: float = 20.0,
    retries: int = 3,
) -> list[tuple[str, str]]:
    """拉取指数成分股，返回 [(代码, 名称), ...]。

    ⚠️ **拿到的是「今天」的成分股名单。** 用它回测历史会有**前视偏差**：
    今天还留在沪深300里的股票，本身就是"过去这些年表现没掉队"的一批，
    等于提前知道了答案。真实的成分股名单是每半年调整一次的，
    要做严谨的历史回测需要逐个调仓日的名单，这个接口给不了。
    """
    key = index_secid(index)
    if key not in INDEX_BOARDS:
        known = "、".join(f"{v[1]}({k.split('.')[-1]})" for k, v in INDEX_BOARDS.items())
        raise DataError(f"还不认识指数 {index!r}。目前支持：{known}")

    board, _name = INDEX_BOARDS[key]
    api = "https://push2.eastmoney.com/api/qt/clist/get"

    members: list[tuple[str, str]] = []
    page = 1
    while True:
        params = {
            "pn": page,
            "pz": CONSTITUENT_PAGE_SIZE,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f3",
            "fs": f"b:{board}",
            "fields": "f12,f14",
        }
        payload = None
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(
                    CONSTITUENTS_URL, params=params, headers=HEADERS, timeout=timeout
                )
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * attempt)

        if payload is None:
            raise DataError(f"拉取成分股失败（第 {page} 页）: {last_error}")

        data = payload.get("data")
        if not data or not data.get("diff"):
            break

        batch = [(row["f12"], row["f14"]) for row in data["diff"]]
        members.extend(batch)

        total = data.get("total", 0)
        if len(members) >= total or not batch:
            break
        page += 1

    if not members:
        raise DataError(f"指数 {index} 没有拉到任何成分股")

    # 去重但保序
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for code, name in members:
        if code not in seen:
            seen.add(code)
            unique.append((code, name))

    return unique[:limit] if limit else unique


def parse_klines(payload: dict, secid: str, fqt: int) -> Series:
    """把接口返回的 JSON 解析成 Series。"""
    data = payload.get("data")
    if not data:
        raise DataError(
            f"代码 {secid} 没有数据 —— 可能代码写错了，或者这只股票已退市 / 从未上市"
        )

    raw_klines = data.get("klines") or []
    if not raw_klines:
        raise DataError(f"代码 {secid}（{data.get('name', '?')}）在指定区间内没有 K 线数据")

    bars: list[Bar] = []
    for row in raw_klines:
        parts = row.split(",")
        if len(parts) < 7:
            raise DataError(f"K 线字段少于 7 个，接口可能改版了: {row!r}")
        bars.append(
            Bar(
                date=date.fromisoformat(parts[0]),
                open=float(parts[1]),
                close=float(parts[2]),  # ← 注意：第 3 个是收盘价
                high=float(parts[3]),
                low=float(parts[4]),
                volume=float(parts[5]),
                amount=float(parts[6]),
                # f61 换手率。老接口偶尔会少给这个字段，缺了就填 0，
                # 影响只有筹码分布，不影响价格相关的所有计算。
                turnover=float(parts[10]) if len(parts) > 10 else 0.0,
            )
        )

    bars, dropped = _clean(bars, secid)

    return Series(
        code=normalize_code(data.get("code") or secid.split(".")[-1]),
        secid=secid,
        name=data.get("name") or "",
        bars=tuple(bars),
        fqt=fqt,
        fetched_at=datetime.now(),
        dropped_bars=dropped,
    )


def _clean(bars: list[Bar], secid: str) -> tuple[list[Bar], int]:
    """排序 + 去重 + 自洽检查 + 砍掉负价前导段。

    返回 (清洗后的 bar 列表, 被砍掉的条数)。
    """
    bars.sort(key=lambda b: b.date)

    seen: set[date] = set()
    deduped: list[Bar] = []
    for bar in bars:
        if bar.date in seen:
            continue
        seen.add(bar.date)
        deduped.append(bar)

    # 非正价格一律丢弃。正常情况下后复权（默认）永远为正，这里是给
    # --fqt 1（前复权）兜底的 —— 那种模式下负价会散布在很多年里，
    # 实测工商银行 2014 年还有 "价格 0.04 元"。丢掉比留着误导强。
    usable = [b for b in deduped if b.low > 0]
    dropped = len(deduped) - len(usable)

    if not usable:
        raise DataError(
            f"{secid} 整段历史的价格都非正，无法用于回测。用 --fqt 2（后复权）试试。"
        )

    for bar in usable:
        if not (bar.low <= bar.open <= bar.high) or not (bar.low <= bar.close <= bar.high):
            raise DataError(
                f"{secid} 在 {bar.date} 的数据不自洽："
                f"开{bar.open} 高{bar.high} 低{bar.low} 收{bar.close}"
            )

    return usable, dropped
