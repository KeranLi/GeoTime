"""
时频分析模块

提供STFT、Lomb-Scargle等时频分析方法

注意: 绘图功能已移至 geotime.plotting.timeseries_plots
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from scipy import signal

try:
    from astropy.timeseries import LombScargle as AstroLombScargle
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


@dataclass
class SpectrogramResult:
    """频谱图结果容器"""
    frequencies: np.ndarray
    times: np.ndarray
    power: np.ndarray
    
    def plot(self, ax=None, **kwargs):
        """
        绘制频谱图
        
        注意: 实际绘图逻辑在 geotime.plotting.timeseries_plots.plot_spectrogram
        """
        from geotime.plotting import plot_spectrogram
        return plot_spectrogram(
            self.frequencies, self.times, self.power,
            ax=ax, **kwargs
        )


@dataclass
class LombScargleResult:
    """Lomb-Scargle结果容器"""
    frequencies: np.ndarray
    power: np.ndarray
    false_alarm_probability: Optional[np.ndarray] = None
    
    def get_peaks(
        self,
        n_peaks: int = 3,
        min_frequency: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        获取主要峰值

        Args:
            n_peaks: 峰值数量
            min_frequency: 最小频率阈值

        Returns:
            {'frequencies': ..., 'periods': ..., 'powers': ...}
        """
        freqs = self.frequencies.copy()
        power = self.power.copy()
        
        # 应用频率阈值
        if min_frequency is not None:
            mask = freqs >= min_frequency
            freqs = freqs[mask]
            power = power[mask]
        
        # 找峰值
        peak_indices = signal.find_peaks(power)[0]
        peak_powers = power[peak_indices]
        
        # 排序
        top_indices = np.argsort(peak_powers)[-n_peaks:][::-1]
        top_peak_indices = peak_indices[top_indices]
        
        peak_freqs = freqs[top_peak_indices]
        peak_periods = 1 / peak_freqs
        
        return {
            'frequencies': peak_freqs,
            'periods': peak_periods,
            'powers': power[top_peak_indices],
        }
    
    def plot(self, ax=None, show_peaks: int = 3, **kwargs):
        """
        绘制Lomb-Scargle周期图
        
        注意: 实际绘图逻辑在 geotime.plotting.timeseries_plots.plot_lomb_scargle
        """
        from geotime.plotting import plot_lomb_scargle
        
        peaks = self.get_peaks(n_peaks=show_peaks) if show_peaks > 0 else None
        
        return plot_lomb_scargle(
            self.frequencies, self.power,
            show_peaks=show_peaks,
            peak_freqs=peaks['frequencies'] if peaks else None,
            peak_periods=peaks['periods'] if peaks else None,
            peak_powers=peaks['powers'] if peaks else None,
            ax=ax, **kwargs
        )


def compute_stft(
    signal_data: np.ndarray,
    sampling_rate: float = 1.0,
    window: str = 'hann',
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: bool = False,
    boundary: Optional[str] = None,
    padded: bool = True,
    axis: int = -1,
) -> SpectrogramResult:
    """
    计算短时傅里叶变换 (STFT) 频谱图

    Args:
        signal_data: 输入信号
        sampling_rate: 采样率
        window: 窗函数类型
        nperseg: 每段长度
        noverlap: 重叠点数 (默认 nperseg // 4)
        nfft: FFT长度 (默认 nperseg)
        detrend: 是否去趋势
        boundary: 边界处理
        padded: 是否填充
        axis: 计算轴

    Returns:
        SpectrogramResult对象
    """
    if noverlap is None:
        noverlap = nperseg // 4
    
    if nfft is None:
        nfft = nperseg
    
    frequencies, times, Zxx = signal.stft(
        signal_data,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        boundary=boundary,
        padded=padded,
        axis=axis,
    )
    
    # 计算功率谱
    power = np.abs(Zxx) ** 2
    
    return SpectrogramResult(
        frequencies=frequencies,
        times=times,
        power=power,
    )


def compute_lombscargle(
    time: np.ndarray,
    signal_data: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
    frequency_resolution: float = 0.001,
    normalization: str = 'standard',
    center_data: bool = True,
    fit_mean: bool = True,
    method: str = 'auto',
) -> LombScargleResult:
    """
    计算Lomb-Scargle周期图

    Args:
        time: 时间数组
        signal_data: 信号值
        frequencies: 指定频率数组 (优先)
        min_frequency: 最小频率
        max_frequency: 最大频率
        frequency_resolution: 频率分辨率
        normalization: 归一化方法
        center_data: 是否中心化数据
        fit_mean: 是否拟合均值
        method: 计算方法

    Returns:
        LombScargleResult对象
    """
    time = np.asarray(time)
    signal_data = np.asarray(signal_data)
    
    # 删除NaN值
    mask = ~(np.isnan(time) | np.isnan(signal_data))
    time, signal_data = time[mask], signal_data[mask]
    
    # 生成频率数组
    if frequencies is None:
        if min_frequency is None:
            min_frequency = 1 / (time.max() - time.min())
        if max_frequency is None:
            max_frequency = 0.5 / np.median(np.diff(time))  # Nyquist
        
        frequencies = np.arange(
            min_frequency,
            max_frequency,
            frequency_resolution
        )
    
    if ASTROPY_AVAILABLE:
        # 使用astropy的实现
        ls = AstroLombScargle(time, signal_data)
        power = ls.power(frequencies, normalization=normalization)
        
        # 计算误报概率
        try:
            fap = ls.false_alarm_probability(power.max())
        except:
            fap = None
    else:
        # 使用scipy的实现
        power = signal.lombscargle(time, signal_data, frequencies)
        fap = None
    
    return LombScargleResult(
        frequencies=frequencies,
        power=power,
        false_alarm_probability=fap,
    )


def extract_time_range(
    time: np.ndarray,
    signal: np.ndarray,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取时间范围内的信号

    Args:
        time: 时间数组
        signal: 信号数组
        start_time: 开始时间
        end_time: 结束时间

    Returns:
        (time_range, signal_range)
    """
    mask = np.ones(len(time), dtype=bool)
    
    if start_time is not None:
        mask &= (time >= start_time)
    if end_time is not None:
        mask &= (time <= end_time)
    
    return time[mask], signal[mask]


def compute_psd(
    signal_data: np.ndarray,
    sampling_rate: float = 1.0,
    nperseg: Optional[int] = None,
    method: str = 'welch',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算功率谱密度

    Args:
        signal_data: 输入信号
        sampling_rate: 采样率
        nperseg: 分段长度
        method: 方法 ('welch', 'periodogram')

    Returns:
        (frequencies, psd)
    """
    if nperseg is None:
        nperseg = min(256, len(signal_data) // 4)
    
    if method == 'welch':
        frequencies, psd = signal.welch(
            signal_data,
            fs=sampling_rate,
            nperseg=nperseg,
        )
    elif method == 'periodogram':
        frequencies, psd = signal.periodogram(
            signal_data,
            fs=sampling_rate,
        )
    else:
        raise ValueError(f"未知方法: {method}")
    
    return frequencies, psd


class TimeFrequencyAnalyzer:
    """
    时频分析器

    整合多种时频分析方法
    """

    def __init__(self, time: np.ndarray, signal: np.ndarray):
        """
        Args:
            time: 时间数组
            signal: 信号数组
        """
        self.time = np.asarray(time)
        self.signal = np.asarray(signal)
        
        # 计算采样率
        self.dt = np.median(np.diff(self.time))
        self.sampling_rate = 1.0 / self.dt if self.dt > 0 else 1.0

    def stft(
        self,
        nperseg: int = 256,
        window: str = 'hann',
        **kwargs,
    ) -> SpectrogramResult:
        """计算STFT"""
        return compute_stft(
            self.signal,
            sampling_rate=self.sampling_rate,
            nperseg=nperseg,
            window=window,
            **kwargs,
        )

    def lomb_scargle(
        self,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        n_frequencies: int = 1000,
        **kwargs,
    ) -> LombScargleResult:
        """计算Lomb-Scargle周期图"""
        min_freq = 1/max_period if max_period else None
        max_freq = 1/min_period if min_period else None
        
        frequencies = np.linspace(min_freq or 0.001, max_freq or 0.5, n_frequencies)
        
        return compute_lombscargle(
            self.time,
            self.signal,
            frequencies=frequencies,
            **kwargs,
        )

    def welch_psd(self, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算Welch PSD"""
        return compute_psd(self.signal, self.sampling_rate, nperseg, method='welch')

    def plot_summary(self, figsize: Tuple[int, int] = (14, 10)):
        """
        绘制分析摘要图
        
        注意: 实际绘图逻辑在 geotime.plotting.timeseries_plots.plot_time_frequency_summary
        """
        from geotime.plotting import plot_time_frequency_summary
        
        # STFT
        stft_result = self.stft()
        
        # Lomb-Scargle
        ls_result = self.lomb_scargle()
        
        return plot_time_frequency_summary(
            self.time, self.signal,
            stft_result.frequencies, stft_result.times, stft_result.power,
            ls_result.frequencies, ls_result.power,
            figsize=figsize,
        )
