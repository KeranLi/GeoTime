"""
线性去趋势和标准化模块
"""

import numpy as np


def detrend_and_normalize(
    data: np.ndarray,
    time: np.ndarray,
    time_offset: float = 0.0,
) -> dict:
    """
    去除线性趋势并进行标准化

    使用一阶多项式拟合去除趋势，然后计算统计量并归一化

    Args:
        data: 输入数据
        time: 时间/位置数组
        time_offset: 时间偏移量

    Returns:
        包含以下键的字典:
        - detrended: 去趋势数据
        - std: 标准差
        - variance: 方差
        - normalized: 归一化数据
    """
    data = np.asarray(data)
    time = np.asarray(time)
    
    # 一阶多项式拟合 (线性趋势)
    coeffs = np.polyfit(time - time_offset, data, 1)
    trend = np.polyval(coeffs, time - time_offset)
    
    # 去趋势
    detrended = data - trend
    detrended = np.nan_to_num(detrended, nan=0.0)
    
    # 统计量
    std = detrended.std()
    variance = std ** 2
    
    # 归一化
    normalized = detrended / std if std > 0 else detrended
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    return {
        "detrended": detrended,
        "std": std,
        "variance": variance,
        "normalized": normalized,
    }
