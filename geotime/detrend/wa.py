"""
加权移动平均去趋势模块
"""

import numpy as np
import pandas as pd


class WeightedAverage:
    """
    加权移动平均计算器

    用于对时间序列数据进行加权移动平均平滑处理

    Example:
        >>> weights = np.array([0.4, 0.15, 0.15, 0.15, 0.15])
        >>> wa = WeightedAverage(weights)
        >>> trend = wa.rolling(data, window=5, min_periods=1, center=True)
    """

    def __init__(self, weights: np.ndarray):
        """
        Args:
            weights: 权重数组，长度应等于窗口大小
        """
        self.weights = np.array(weights).flatten()

    def rolling(
        self,
        data: pd.Series,
        window: int = 5,
        min_periods: int = 1,
        center: bool = True,
    ) -> pd.Series:
        """
        计算加权移动平均

        Args:
            data: 输入数据序列
            window: 窗口大小
            min_periods: 最小有效数据点数
            center: 是否居中窗口

        Returns:
            平滑后的数据序列
        """
        if len(self.weights) < window:
            raise ValueError(f"权重长度({len(self.weights)})应不小于窗口大小({window})")
        
        return data.rolling(
            window=window,
            min_periods=min_periods,
            center=center,
        ).apply(lambda x: np.dot(x, self.weights[:len(x)]))
