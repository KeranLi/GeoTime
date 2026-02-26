"""
平滑处理模块

提供LOWESS、移动平均等平滑方法
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def lowess_smooth(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.1,
    it: int = 3,
    delta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LOWESS (Locally Weighted Scatterplot Smoothing)

    Args:
        x: 自变量数组
        y: 因变量数组
        frac: 平滑参数，使用数据的比例 (0-1)，越大越平滑
        it: 迭代次数
        delta: 加速计算的阈值

    Returns:
        (smoothed_x, smoothed_y) 元组
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("需要安装statsmodels: pip install statsmodels")
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 删除NaN值
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) == 0:
        raise ValueError("输入数据全为NaN")
    
    # 执行LOWESS
    result = sm_lowess(y, x, frac=frac, it=it, delta=delta, return_sorted=True)
    
    return result[:, 0], result[:, 1]


def moving_average(
    data: np.ndarray,
    window: int = 5,
    mode: str = 'same',
) -> np.ndarray:
    """
    简单移动平均

    Args:
        data: 输入数据
        window: 窗口大小
        mode: 边界处理模式 ('valid', 'same', 'full')

    Returns:
        平滑后的数据
    """
    data = np.asarray(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode=mode)


def exponential_smooth(
    data: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    指数平滑

    Args:
        data: 输入数据
        alpha: 平滑因子 (0-1)，越大越响应快速变化

    Returns:
        平滑后的数据
    """
    data = np.asarray(data)
    result = np.zeros_like(data)
    result[0] = data[0]
    
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result


def gaussian_smooth(
    data: np.ndarray,
    sigma: float = 2.0,
    mode: str = 'nearest',
) -> np.ndarray:
    """
    高斯平滑

    Args:
        data: 输入数据
        sigma: 高斯核标准差
        mode: 边界处理模式

    Returns:
        平滑后的数据
    """
    from scipy.ndimage import gaussian_filter1d
    
    data = np.asarray(data)
    return gaussian_filter1d(data, sigma=sigma, mode=mode)


def savgol_smooth(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Savitzky-Golay滤波

    Args:
        data: 输入数据
        window_length: 窗口长度（必须是奇数）
        polyorder: 多项式阶数

    Returns:
        平滑后的数据
    """
    from scipy.signal import savgol_filter
    
    data = np.asarray(data)
    
    # 确保窗口长度是奇数
    if window_length % 2 == 0:
        window_length += 1
    
    return savgol_filter(data, window_length, polyorder)


def interpolate_to_uniform(
    x: np.ndarray,
    y: np.ndarray,
    dx: Optional[float] = None,
    n_points: Optional[int] = None,
    kind: str = 'linear',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将非均匀采样数据插值为均匀采样

    Args:
        x: 原始x坐标
        y: 原始y值
        dx: 均匀采样间隔（与n_points二选一）
        n_points: 均匀采样点数（与dx二选一）
        kind: 插值方法 ('linear', 'cubic', 'quadratic', ...)

    Returns:
        (uniform_x, interpolated_y) 元组
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 删除NaN值
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    # 排序
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]
    
    # 创建插值函数
    f = interp1d(x, y, kind=kind, bounds_error=False, fill_value='extrapolate')
    
    # 生成均匀x坐标
    if dx is not None:
        x_uniform = np.arange(x.min(), x.max() + dx, dx)
    elif n_points is not None:
        x_uniform = np.linspace(x.min(), x.max(), n_points)
    else:
        raise ValueError("必须指定dx或n_points")
    
    y_uniform = f(x_uniform)
    
    return x_uniform, y_uniform


def resample_series(
    df: pd.DataFrame,
    age_column: str,
    value_column: str,
    new_ages: np.ndarray,
    method: str = 'linear',
) -> pd.DataFrame:
    """
    重采样时间序列到新的年龄点

    Args:
        df: 输入DataFrame
        age_column: 年龄列名
        value_column: 值列名
        new_ages: 新的年龄数组
        method: 插值方法

    Returns:
        重采样后的DataFrame
    """
    x = df[age_column].values
    y = df[value_column].values
    
    # 排序
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]
    
    # 插值
    f = interp1d(x, y, kind=method, bounds_error=False, fill_value=0)
    new_values = f(new_ages)
    
    return pd.DataFrame({
        age_column: new_ages,
        value_column: new_values,
    })


class SmoothingPipeline:
    """
    平滑处理管道

    支持链式调用多种平滑方法
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: 自变量
            y: 因变量
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.history = [('original', self.x.copy(), self.y.copy())]

    def lowess(self, frac: float = 0.1, **kwargs) -> 'SmoothingPipeline':
        """应用LOWESS平滑"""
        x_smooth, y_smooth = lowess_smooth(self.x, self.y, frac=frac, **kwargs)
        self.x, self.y = x_smooth, y_smooth
        self.history.append(('lowess', self.x.copy(), self.y.copy()))
        return self

    def moving_average(self, window: int = 5) -> 'SmoothingPipeline':
        """应用移动平均"""
        self.y = moving_average(self.y, window=window, mode='same')
        self.history.append(('ma', self.x.copy(), self.y.copy()))
        return self

    def gaussian(self, sigma: float = 2.0) -> 'SmoothingPipeline':
        """应用高斯平滑"""
        self.y = gaussian_smooth(self.y, sigma=sigma)
        self.history.append(('gaussian', self.x.copy(), self.y.copy()))
        return self

    def interpolate(self, dx: float = 1.0, kind: str = 'linear') -> 'SmoothingPipeline':
        """插值为均匀采样"""
        self.x, self.y = interpolate_to_uniform(self.x, self.y, dx=dx, kind=kind)
        self.history.append(('uniform', self.x.copy(), self.y.copy()))
        return self

    def get_result(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取最终结果"""
        return self.x, self.y

    def plot_history(self, figsize: Tuple[int, int] = (12, 8)):
        """
        绘制处理历史
        
        注意: 实际绘图逻辑在 geotime.plotting
        """
        from geotime.plotting import plot_smoothing_history
        return plot_smoothing_history(self.history, figsize)
