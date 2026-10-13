"""筹码分布（成本分布）。

A 股特有的一套分析方法：估算「所有持股人的成本都分布在哪些价位」，
从而看出套牢盘在哪、主力成本区在哪。它不是交易所公布的数据，
而是**用「成交量 × 三角形分布 + 换手衰减」推算出来的**。

算法（通行做法，不是本项目发明的）：

1. **铺**：每天把成交量按三角形铺在当天 [最低价, 最高价] 区间上，峰在平均价。
   为什么是三角形：一天的成交价通常集中在均价附近，两头少 —— 三角形是个
   粗糙但够用的近似，所有人都这么算。

2. **衰**：旧筹码按当天的换手率等比衰减。换手 5% 意味着有 5% 的流通股换了
   主人，那些旧成本就被新成本「替换」掉了。

3. **读**：得到分布后可以算
   - **获利盘比例** —— 成本低于现价的筹码占比，也就是"现在有多少人是赚的"
   - **平均成本** —— 全部筹码的加权平均价
   - **90% 成本区间** —— 去掉两头各 5% 后剩下的价格范围
   - **集中度** —— 区间越窄说明筹码越集中（= 筹码越"干净"）

**已知的近似**：平均价用的是 (最高+最低+收盘)/3，而不是真实的成交均价 VWAP。
原因是 VWAP 要用 成交额/成交量 算，那是**不复权**口径，而价格序列是复权的，
两者混用会算出一个根本不在当日价格区间里的值。所以退而求其次用典型价格。

**这个模块最容易错的地方是格子（价格网格）的管理** —— 价格区间会随行情扩张，
扩张时旧筹码要重采样到新网格上。所以它单独成模块、单独测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class ChipState(NamedTuple):
    """某个时点的筹码分布快照。"""

    profit_ratio: float
    """获利盘比例，0 ~ 1。成本低于现价的筹码占比。"""

    avg_cost: float
    """全部筹码的加权平均成本。"""

    cost_low: float
    """90% 成本区间下沿。"""

    cost_high: float
    """90% 成本区间上沿。"""

    concentration: float
    """集中度 = (上沿 - 下沿) / (上沿 + 下沿)。越低越集中。"""

    @property
    def cost_band(self) -> float:
        """90% 成本区间的宽度占中值的比例。"""
        middle = (self.cost_high + self.cost_low) / 2
        return (self.cost_high - self.cost_low) / middle if middle else 0.0


class ChipDistribution:
    """筹码分布，按天增量更新。

    增量而非每天重算，是因为回测会逐日调用。全量重算在 2500 天 × 200 个格子上
    是 O(n²)，慢到没法用；增量是 O(每天覆盖的格子数)。
    """

    __slots__ = ("bins", "decay", "lo", "hi", "weights", "data_lo", "data_hi")

    def __init__(self, bins: int = 400, decay: float = 1.0) -> None:
        if bins < 10:
            raise ValueError("bins 至少为 10，否则分布太粗糙")
        if not 0 < decay <= 1.0:
            raise ValueError("decay 取 (0, 1]，1.0 表示不额外放大换手率")
        self.bins = bins
        self.decay = decay
        self.lo: float | None = None
        self.hi: float | None = None
        self.weights: list[float] = []
        # 行情真实到达过的最低/最高价。网格由它加固定比例的留白算出，
        # 留白**不能**在已经留白的区间上再累加（那样会指数级膨胀）。
        self.data_lo: float | None = None
        self.data_hi: float | None = None

    # ── 网格管理 ────────────────────────────────────────────────

    def _grid_step(self) -> float:
        assert self.lo is not None and self.hi is not None
        return (self.hi - self.lo) / self.bins

    def _resample(self, new_lo: float, new_hi: float) -> None:
        """把现有筹码搬到新的价格网格上。

        用最近邻：每个旧格子的筹码按它的中心价落到对应的新格子里。
        比线性插值糙一点，但 O(bins) 且不会因为反复插值把分布抹平。
        """
        n = self.bins
        old_lo, old_hi, old_w = self.lo, self.hi, self.weights

        if old_lo is None or old_hi is None or not old_w:
            self.lo, self.hi = new_lo, new_hi
            self.weights = [0.0] * n
            return

        new_step = (new_hi - new_lo) / n
        old_step = (old_hi - old_lo) / n
        moved = [0.0] * n
        for k in range(n):
            if old_w[k] <= 0:
                continue
            center = old_lo + (k + 0.5) * old_step
            idx = int((center - new_lo) / new_step)
            moved[min(max(idx, 0), n - 1)] += old_w[k]

        total = sum(moved)
        self.weights = [w / total for w in moved] if total > 0 else [0.0] * n
        self.lo, self.hi = new_lo, new_hi

    def _ensure_range(self, low: float, high: float) -> None:
        """保证价格网格覆盖到今天。

        留白按**行情真实区间**算，每次重算而不是在旧网格上叠加 ——
        在已留白的区间上继续加留白会指数级膨胀（实测跑 6000 天后网格会
        扩到负数）。
        """
        self.data_lo = low if self.data_lo is None else min(self.data_lo, low)
        self.data_hi = high if self.data_hi is None else max(self.data_hi, high)

        span = self.data_hi - self.data_lo
        if span <= 0:
            span = max(abs(self.data_hi) * 0.01, 1e-6)  # 一字板：给个最小宽度
        pad = span * 0.02

        new_lo = self.data_lo - pad
        new_hi = self.data_hi + pad

        if self.lo is None or self.hi is None or new_lo < self.lo or new_hi > self.hi:
            self._resample(new_lo, new_hi)

    # ── 更新 ────────────────────────────────────────────────────

    def update(self, high: float, low: float, avg: float, turnover_pct: float) -> None:
        """推进一个交易日。

        Args:
            high: 当日最高价
            low: 当日最低价
            avg: 当日平均价（用典型价格 (高+低+收)/3）
            turnover_pct: 当日换手率，百分数（接口给的原始单位）
        """
        if not (low <= avg <= high):
            raise ValueError(f"平均价 {avg} 不在当日区间 [{low}, {high}] 内")

        # 换手率可能超过 1（次新股上市初期很常见），夹到 [0, 1]
        turnover = min(max(turnover_pct / 100.0 * self.decay, 0.0), 1.0)

        if turnover <= 0:
            # 停牌、一字板等：没有换手，筹码原封不动。
            # 注意这里必须**在扩张价格网格之前**返回 —— 网格一扩张就会触发
            # 重采样，筹码的价位分布会被挪动。零换手应该是彻底无影响。
            return

        self._ensure_range(low, high)

        # 老筹码按换手率衰减 —— 这部分人被"替换"出局了
        factor = 1.0 - turnover
        self.weights = [w * factor for w in self.weights]

        # 新筹码按三角形铺进今天的价格区间
        self._add_triangle(high, low, avg, turnover)

        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def _add_triangle(self, high: float, low: float, avg: float, weight: float) -> None:
        step = self._grid_step()
        assert self.lo is not None
        n = self.bins

        k0 = min(max(int((low - self.lo) / step), 0), n - 1)
        k1 = min(max(int((high - self.lo) / step), 0), n - 1)
        if k1 < k0:
            k0 = k1

        raw: list[float] = []
        for k in range(k0, k1 + 1):
            price = self.lo + (k + 0.5) * step
            if price <= avg:
                raw.append((price - low) / (avg - low) if avg > low else 1.0)
            else:
                raw.append((high - price) / (high - avg) if high > avg else 1.0)

        raw = [max(w, 0.0) for w in raw]
        total = sum(raw)

        if total <= 0:
            # 退化成一字板：全部堆在平均价那个格子里
            k = min(max(int((avg - self.lo) / step), 0), n - 1)
            self.weights[k] += weight
            return

        for k, w in zip(range(k0, k1 + 1), raw):
            self.weights[k] += weight * w / total

    # ── 读取 ────────────────────────────────────────────────────

    @property
    def total_weight(self) -> float:
        """筹码总量。没喂过数据（或换手率一直是 0）时是 0。

        调用方要拿它判断分布是否有效 —— 空的分布算出来获利盘是 0%，
        会被误读成"极度低估"从而一路满仓。
        """
        return sum(self.weights)

    def prices(self) -> list[float]:
        step = self._grid_step()
        assert self.lo is not None
        return [self.lo + (k + 0.5) * step for k in range(self.bins)]

    def state(self, close: float) -> ChipState:
        """按当前收盘价读出各项指标。"""
        if not self.weights or sum(self.weights) <= 0:
            return ChipState(0.0, close, close, close, 0.0)

        prices = self.prices()
        weights = self.weights

        profit = sum(w for p, w in zip(prices, weights) if p <= close)
        avg_cost = sum(p * w for p, w in zip(prices, weights))

        low_95, high_95 = self._quantile_range(prices, weights, 0.05, 0.95)
        middle = (high_95 + low_95) / 2
        concentration = (high_95 - low_95) / middle if middle else 0.0

        return ChipState(
            profit_ratio=profit,
            avg_cost=avg_cost,
            cost_low=low_95,
            cost_high=high_95,
            concentration=concentration,
        )

    def _quantile_range(
        self, prices: list[float], weights: list[float], q_lo: float, q_hi: float
    ) -> tuple[float, float]:
        """找出累积权重落在 [q_lo, q_hi] 区间的价格范围。

        用来做 90% 成本区间：从低到高累加筹码，跨过 5% 的那一刻和 95% 的那一刻。
        """
        total = sum(weights)
        if total <= 0:
            return prices[0], prices[-1]

        target_lo = total * q_lo
        target_hi = total * q_hi
        cumulative = 0.0
        lo = prices[0]
        hi = prices[-1]
        found_lo = False

        for price, weight in zip(prices, weights):
            cumulative += weight
            if not found_lo and cumulative >= target_lo:
                lo = price
                found_lo = True
            if cumulative >= target_hi:
                hi = price
                break

        return lo, hi
