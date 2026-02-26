"""
离散频率分布的连续化重建

将离散的年龄-频率数据重建为连续信号，用于后续频谱分析
"""

from enum import Enum
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


class InterpolationMethod(Enum):
    """插值方法枚举"""
    LINEAR = "linear"
    CUBIC = "cubic"
    AKIMA = "akima"
    PCHIP = "pchip"
    GAUSSIAN = "gaussian"


@dataclass
class ReconstructionResult:
    """信号重建结果"""
    age_continuous: np.ndarray
    frequency_continuous: np.ndarray
    interpolation_method: str
    smoothing_sigma: Optional[float] = None
    
    def plot(self, ax=None, show_original=None):
        """
        绘制重建结果
        
        注意: 实际绘图逻辑在 geotime.plotting.metaspectral_plots.plot_reconstruction
        """
        from geotime.plotting import plot_reconstruction
        
        original_age, original_freq = None, None
        if show_original is not None:
            original_age, original_freq = show_original
        
        return plot_reconstruction(
            self.age_continuous,
            self.frequency_continuous,
            self.interpolation_method,
            original_age=original_age,
            original_freq=original_freq,
            ax=ax,
        )


def reconstruct_continuous_signal(
    age: np.ndarray,
    frequency: np.ndarray,
    method: InterpolationMethod = InterpolationMethod.CUBIC,
    n_points: Optional[int] = None,
    dx: Optional[float] = None,
    smooth_sigma: Optional[float] = None,
    boundary_extension: float = 0.1,
) -> ReconstructionResult:
    """
    将离散的年龄-频率分布重建为连续信号

    Args:
        age: 离散年龄数组
        frequency: 离散频率数组
        method: 插值方法
        n_points: 连续信号点数 (与dx二选一)
        dx: 采样间隔 (与n_points二选一)
        smooth_sigma: 高斯平滑参数 (可选)
        boundary_extension: 边界扩展比例 (防止边界效应)

    Returns:
        ReconstructionResult对象
    """
    age = np.asarray(age)
    frequency = np.asarray(frequency)
    
    # 排序
    sort_idx = np.argsort(age)
    age, frequency = age[sort_idx], frequency[sort_idx]
    
    # 确定连续信号的分辨率
    if dx is not None:
        age_min, age_max = age.min(), age.max()
        age_continuous = np.arange(age_min, age_max + dx, dx)
    elif n_points is not None:
        age_continuous = np.linspace(age.min(), age.max(), n_points)
    else:
        # 默认使用原始数据密度的5倍
        n_points = len(age) * 5
        age_continuous = np.linspace(age.min(), age.max(), n_points)
    
    # 选择插值方法
    if method == InterpolationMethod.LINEAR:
        f = interpolate.interp1d(age, frequency, kind='linear', 
                                  bounds_error=False, fill_value=0)
        freq_continuous = f(age_continuous)
        
    elif method == InterpolationMethod.CUBIC:
        # 使用样条插值
        cs = interpolate.CubicSpline(age, frequency)
        freq_continuous = cs(age_continuous)
        
    elif method == InterpolationMethod.AKIMA:
        f = interpolate.Akima1DInterpolator(age, frequency)
        freq_continuous = f(age_continuous)
        
    elif method == InterpolationMethod.PCHIP:
        f = interpolate.PchipInterpolator(age, frequency)
        freq_continuous = f(age_continuous)
        
    elif method == InterpolationMethod.GAUSSIAN:
        # 基于核密度估计的重建
        from scipy.stats import gaussian_kde
        
        # 使用频率作为权重进行核密度估计
        weights = frequency / frequency.sum()
        kde = gaussian_kde(age, weights=weights, bw_method='scott')
        freq_continuous = kde(age_continuous) * frequency.sum()
        
    else:
        raise ValueError(f"未知插值方法: {method}")
    
    # 确保非负
    freq_continuous = np.maximum(freq_continuous, 0)
    
    # 可选的高斯平滑
    if smooth_sigma is not None and smooth_sigma > 0:
        freq_continuous = gaussian_filter1d(freq_continuous, sigma=smooth_sigma)
    
    return ReconstructionResult(
        age_continuous=age_continuous,
        frequency_continuous=freq_continuous,
        interpolation_method=method.value,
        smoothing_sigma=smooth_sigma,
    )


def adaptive_reconstruction(
    age: np.ndarray,
    frequency: np.ndarray,
    target_resolution: float = 0.5,
    preserve_peaks: bool = True,
    peak_prominence: float = 0.1,
) -> ReconstructionResult:
    """
    自适应信号重建

    根据数据特征自动选择最佳重建参数

    Args:
        age: 离散年龄数组
        frequency: 离散频率数组
        target_resolution: 目标分辨率 (Ma)
        preserve_peaks: 是否保留峰值特征
        peak_prominence: 峰值显著性阈值

    Returns:
        ReconstructionResult对象
    """
    from scipy.signal import find_peaks
    
    age = np.asarray(age)
    frequency = np.asarray(frequency)
    
    # 检测峰值
    if preserve_peaks:
        peaks, properties = find_peaks(frequency, prominence=peak_prominence*frequency.max())
        
        # 在峰值附近使用更高分辨率
        age_dense = [age]
        for peak_idx in peaks:
            peak_age = age[peak_idx]
            # 峰值周围添加密集采样点
            local_range = target_resolution * 5
            local_points = np.arange(
                max(age.min(), peak_age - local_range),
                min(age.max(), peak_age + local_range),
                target_resolution / 5
            )
            age_dense.append(local_points)
        
        # 合并并排序
        age_dense = np.unique(np.concatenate(age_dense))
        
        # 插值到新网格
        f = interpolate.interp1d(age, frequency, kind='cubic', 
                                  bounds_error=False, fill_value=0)
        freq_continuous = f(age_dense)
        age_continuous = age_dense
    else:
        # 均匀采样
        n_points = int((age.max() - age.min()) / target_resolution)
        result = reconstruct_continuous_signal(
            age, frequency, 
            n_points=n_points,
            method=InterpolationMethod.CUBIC
        )
        age_continuous = result.age_continuous
        freq_continuous = result.frequency_continuous
    
    return ReconstructionResult(
        age_continuous=age_continuous,
        frequency_continuous=freq_continuous,
        interpolation_method='adaptive',
        smoothing_sigma=None,
    )


def validate_reconstruction(
    original_age: np.ndarray,
    original_freq: np.ndarray,
    reconstructed: ReconstructionResult,
) -> dict:
    """
    验证重建质量

    Args:
        original_age: 原始年龄
        original_freq: 原始频率
        reconstructed: 重建结果

    Returns:
        质量指标字典
    """
    # 在原始年龄点采样重建信号
    f = interpolate.interp1d(
        reconstructed.age_continuous,
        reconstructed.frequency_continuous,
        kind='linear',
        bounds_error=False,
        fill_value=0
    )
    reconstructed_at_original = f(original_age)
    
    # 计算误差指标
    mse = np.mean((original_freq - reconstructed_at_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original_freq - reconstructed_at_original))
    
    # 相关性
    correlation = np.corrcoef(original_freq, reconstructed_at_original)[0, 1]
    
    # 能量守恒
    original_energy = np.sum(original_freq ** 2)
    reconstructed_energy = np.sum(reconstructed.frequency_continuous ** 2)
    energy_ratio = reconstructed_energy / original_energy if original_energy > 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'original_energy': original_energy,
        'reconstructed_energy': reconstructed_energy,
        'energy_ratio': energy_ratio,
    }
