"""
计时工具
========
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable


@contextmanager
def timer(name: str = "Operation"):
    """
    计时上下文管理器

    Examples
    --------
    >>> with timer("Model training"):
    ...     model.fit(X, y)
    [Model training] took 2.34s
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{name}] took {elapsed:.2f}s")


class Timer:
    """
    计时器类，支持 with 上下文、分段计时。

    Examples
    --------
    >>> with Timer() as t:
    ...     # step 1
    ...     t.lap("Step 1")
    ...     # step 2
    ...     t.lap("Step 2")
    >>> print(t.total)
    """

    def __init__(self):
        self._start_time = None
        self._lap_times = []
        self._total_time = None

    def __enter__(self):
        """支持 with 上下文"""
        self.start()
        return self

    def __exit__(self, *args):
        """退出时自动停止"""
        self.stop()

    def start(self):
        """开始计时"""
        self._start_time = time.perf_counter()
        self._lap_times = []
        self._total_time = None
        return self

    def lap(self, name: str = None) -> float:
        """
        记录分段时间

        Parameters
        ----------
        name : str, optional
            分段名称

        Returns
        -------
        float
            从上一分段开始的时间
        """
        if self._start_time is None:
            raise RuntimeError("Timer not started")

        current = time.perf_counter()

        if self._lap_times:
            elapsed = current - self._lap_times[-1][1]
        else:
            elapsed = current - self._start_time

        lap_name = name or f"Lap {len(self._lap_times) + 1}"
        self._lap_times.append((lap_name, current, elapsed))

        print(f"[{lap_name}] {elapsed:.2f}s")

        return elapsed

    def stop(self) -> float:
        """
        停止计时

        Returns
        -------
        float
            总时间
        """
        if self._start_time is None:
            raise RuntimeError("Timer not started")

        self._total_time = time.perf_counter() - self._start_time
        print(f"[Total] {self._total_time:.2f}s")

        return self._total_time

    @property
    def total(self) -> float:
        """总时间"""
        return self._total_time or 0

    @property
    def elapsed(self) -> float:
        """获取当前已过时间"""
        if self._start_time is None:
            return 0
        return time.perf_counter() - self._start_time

    def summary(self) -> dict:
        """
        获取计时摘要

        Returns
        -------
        dict
            计时信息
        """
        return {
            "total": self._total_time,
            "laps": [
                {"name": name, "time": elapsed} for name, _, elapsed in self._lap_times
            ],
        }


def timed(func: Callable) -> Callable:
    """
    函数计时装饰器

    Examples
    --------
    >>> @timed
    ... def my_function():
    ...     time.sleep(1)
    >>> my_function()
    [my_function] took 1.00s
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] took {elapsed:.2f}s")
        return result

    return wrapper
