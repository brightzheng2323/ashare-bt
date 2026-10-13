# ashare-bt · A 股日线策略回测框架

一个用来看懂「一个策略在 A 股历史上到底赚不赚钱」的小型回测框架。

**为什么不用现成的库？** 因为 A 股有一堆特殊规则 —— T+1、涨跌停、整手交易、5 元最低佣金，
网上的回测示例代码大多不处理，跑出来的收益曲线是假的。这个项目把这些规则做进了引擎本身。

## 安装

需要 [uv](https://docs.astral.sh/uv/)（这台机器上已经有了）：

```bash
uv sync
```

## 跑一个回测

```bash
uv run backtest --code 600519 --start 2020-01-01 --end 2025-01-01 --strategy ma_cross
```

## 常用命令

```bash
uv run backtest list-strategies                 # 看有哪些策略
uv run backtest fetch --code 600519             # 只下载数据并缓存
uv run backtest inspect --code 600519           # 数据体检
uv run pytest                                   # 跑测试（不联网）
```

## 加一个自己的策略

在 `src/ashare_bt/strategies/` 下新建文件：

```python
from ashare_bt.indicators import sma
from ashare_bt.strategies.base import History, Strategy


class MyStrategy(Strategy):
    name = "my_strategy"

    def target(self, h: History) -> float:
        """返回目标仓位：1.0 = 满仓，0.0 = 空仓。

        h 是一个「只能看到今天为止」的视图，
        读明天的数据会直接报错 —— 这是刻意的。
        """
        ma = sma(h.close, 10)
        if ma is None:
            return 0.0
        return 1.0 if h.close[-1] > ma else 0.0
```

然后把它加进 `strategies/__init__.py` 的 `REGISTRY` 就能用 `--strategy my_strategy` 跑。

## 这根框架保证什么

| 保证 | 怎么做到的 |
|---|---|
| 不用当日收盘价成交（未来函数） | T 日收盘出信号，T+1 日开盘成交；信号和成交是主循环里不同迭代的变量 |
| 策略看不到未来数据 | 策略拿到的是 `History` 视图，越界读取直接抛 `LookAheadError` |
| T+1 | 当日买入的股票 `sellable = 0`，次日开盘前才结算为可卖 |
| 涨跌停买不进/卖不出 | 开盘价触及涨跌停价则拒绝，并记录拒绝原因 |
| 佣金最低 5 元 | 按笔计算，小资金回测的关键成本 |
| 指标预热不缺失 | 引擎跑全历史，只在记录时截取回测窗口 |

## 目录

```
src/ashare_bt/
  models.py       数据类：Bar / Series / Position / Order / Fill / Trade
  fetch.py        东方财富接口
  cache.py        本地 CSV 缓存
  rules.py        A 股交易规则（涨跌停、整手、板块）
  fees.py         佣金 / 印花税 / 过户费
  indicators.py   SMA / RSI / MACD / 布林带
  chips.py        筹码分布
  engine.py       回测主循环
  metrics.py      绩效指标
  plot.py         图表
  cli.py          命令行入口
```

## 数据来源

东方财富公开接口，无需 token。默认使用**后复权**（fqt=2）价格 ——
前复权在每次分红送转时会重写全部历史价格，导致同一个命令下个月跑出不同结果。

数据缓存到 `data/cache/`，默认一天内不重复下载。
