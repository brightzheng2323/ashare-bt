# ashare-bt · A 股日线策略回测框架

一个用来看懂「一个策略在 A 股历史上到底赚不赚钱」的小型回测框架。

**为什么不用现成的库？** 因为 A 股有一堆特殊规则 —— T+1、涨跌停、整手交易、5 元最低佣金，网上的回测示例代码大多不处理，跑出来的收益曲线是假的。这个项目把这些规则做进了引擎本身。

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
uv run backtest list-strategies                 # 看有哪些策略和参数
uv run backtest --code 000001 --strategy rsi --rsi-period 14 --cash 200000
uv run backtest fetch --code 600519             # 只下载数据并缓存
uv run backtest inspect --code 600519           # 数据体检
uv run backtest --code 600519 --json            # 输出 JSON，方便脚本处理
uv run pytest                                   # 跑测试（不联网）
uv run pytest -m network                        # 联网测试（默认跳过）
```

## 内置策略

| 名字 | 思路 | 参数 |
|---|---|---|
| `ma_cross` | 双均线金叉买、死叉卖 | `--fast` `--slow` |
| `macd` | DIF 上穿 DEA 买，下穿卖 | `--fast` `--slow` `--signal` |
| `bollinger` | 跌破下轨买、突破上轨卖 | `--window` `--num-std` |
| `rsi` | RSI 超卖买、超买卖 | `--period` `--buy-below` `--sell-above` |
| `chip_peak` | 获利盘稀少时买、获利盘丰厚时卖 | `--buy-below` `--sell-above` `--max-concentration` `--lookback` |

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
| 科创板 200 股起、可 1 股递增 | 按板块分别处理整手规则 |
| 涨跌停按**日期**生效 | 创业板 2020-08-24 才改成 20%，主板 ST 2026-07-06 才改成 10% |

## 关于复权：为什么用后复权 + 整体缩放

这是个踩过坑才定下来的设计。

**前复权不能用。** 东方财富的前复权对现金分红用的是减法（`前复权价 = 当时真实价 − 之后累计每股分红`），对分红历史长的股票会把早年价格算成 0 甚至负数：

| 股票 | 非正价 K 线 | 最后一根 |
|---|---|---|
| 平安银行 | 1909 根 | 2009-01-19 |
| 工商银行 | 420 根 | 2014-05-28（之后"价格" 0.04 元） |
| 中国石化 | 95 根 | 2006-11-14 |

0.04 元的股价不是"失真"，是**没有意义**。

**后复权是乘法，永远为正、全历史有效、收益率正确。** 但它把绝对价位放大了（平安银行后复权价 1345 元，真实约 11 元），而价位决定了「10 万本金能买几手」—— 整手取整和 5 元最低佣金都按成交金额起作用，价位错了这两件事就全错。

**所以：** 用后复权，再整体除以一个常数，让**回测起点**的价格等于那天的真实成交价。常数缩放不改变任何收益率，只把价位搬回真实量级。缩放需要真实价格，所以会额外拉一次不复权数据（走缓存，很便宜）。

已知的取舍：只有回测起点附近的价格是真实的，更早的历史价格会被同比例压得偏小（回测窗口内的价格是对的，不影响结果）。用 `--no-rescale` 可以关掉缩放。

## 关于筹码分布

`chips.py` 用「三角形分布 + 换手衰减」推算持仓成本结构，A 股特有的分析方法。它**不是**交易所公布的数据，是推算出来的。

- 每天把成交量按三角形铺在当天 [最低价, 最高价] 上，峰在平均价
- 旧筹码按当天换手率等比衰减
- 读出：获利盘比例、平均成本、90% 成本区间、集中度

两个已知近似：

1. 平均价用 `(最高+最低+收盘)/3` 而不是真实 VWAP。VWAP 要拿 `成交额/成交量` 算，那是**不复权**口径，和复权后的价格序列混用会算出一个根本不在当日区间里的值。
2. 只回看最近 `lookback` 根（默认 500 根）。不设上限的话分布要覆盖几十年的价格区间，网格被拉宽、近期筹码反而分不清；而且按 0.5% 日换手，500 天前的筹码只剩 8%，本来就没必要算。

## 目录

```
src/ashare_bt/
  models.py       数据类：Bar / Series / Position / Order / Fill / Trade
  fetch.py        东方财富接口
  cache.py        本地 CSV 缓存 + 复权缩放
  rules.py        A 股交易规则（涨跌停、整手、板块）
  fees.py         佣金 / 印花税 / 过户费
  indicators.py   SMA / RSI / MACD / 布林带
  chips.py        筹码分布
  engine.py       回测主循环
  metrics.py      绩效指标
  plot.py         图表
  cli.py          命令行入口
  strategies/     内置策略
```

## 数据来源

东方财富公开接口，无需 token，一次请求取全部历史。数据缓存到 `data/cache/`，默认一天内不重复下载。

## 已知不做（这一版）

部分仓位（策略只能全仓/空仓）、多股组合、日内数据、ETF/可转债、融资融券、参数寻优。

## 免责

这是个**教学和分析工具**。回测跑出来的数字不构成任何投资建议 —— 而且历史回测表现对未来收益几乎没有预测力。
