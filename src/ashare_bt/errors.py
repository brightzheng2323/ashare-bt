"""异常类型。

单独一个模块，避免 fetch / cache / cli 之间互相 import 造成循环依赖。
"""

from __future__ import annotations


class DataError(RuntimeError):
    """行情数据拿不到或者格式不对。

    这是「用户能看懂并自己解决」的错误：代码写错了、股票退市了、
    网络不通 —— 所以 CLI 会把它打成友好提示，而不是抛一堆堆栈。
    """
