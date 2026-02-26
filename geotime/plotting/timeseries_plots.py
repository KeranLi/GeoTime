"""
时频分析绘图模块

从 timeseries.py 和 metaspectral 中提取的绘图函数
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(
    frequencies: np.ndarray,
    times: np.ndarray,
    power: np.ndarray,
    title: str = "Spectrogram (STFT)",
    xlabel: str = "Age (Ma)",
    ylabel: str = "Frequency (1/Ma)",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    绘制频谱图

    Args:
        frequencies: 频率数组
        times: 时间数组
        power: 功率谱矩阵
        title: 图表标题
        xlabel, ylabel: 轴标签
        cmap: 颜色映射
        ax: 可选的Axes对象
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 转换为dB
    power_db = 10 * np.log10(power + 1e-10)

    im = ax.pcolormesh(
        times, frequencies, power_db,
        shading='gouraud', cmap=cmap
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label='Power (dB)')

    return fig


def plot_lomb_scargle(
    frequencies: np.ndarray,
    power: np.ndarray,
    title: str = "Lomb-Scargle Periodogram",
    xlabel: str = "Frequency (1/Ma)",
    ylabel: str = "Power",
    show_peaks: int = 3,
    peak_freqs: Optional[np.ndarray] = None,
    peak_periods: Optional[np.ndarray] = None,
    peak_powers: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    绘制Lomb-Scargle周期图

    Args:
        frequencies: 频率数组
        power: 功率数组
        show_peaks: 标注的峰值数量
        peak_freqs, peak_periods, peak_powers: 峰值信息
        ax: 可选的Axes对象
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(frequencies, power, linewidth=1.5)

    # 标注峰值
    if show_peaks > 0 and peak_freqs is not None:
        for freq, period, pwr in zip(peak_freqs, peak_periods, peak_powers):
            ax.annotate(
                f'{period:.1f} Ma',
                xy=(freq, pwr),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, frequencies.max())
    ax.grid(True, alpha=0.3)

    return fig


def plot_time_frequency_summary(
    time: np.ndarray,
    signal: np.ndarray,
    stft_freqs: np.ndarray,
    stft_times: np.ndarray,
    stft_power: np.ndarray,
    ls_freqs: np.ndarray,
    ls_power: np.ndarray,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    绘制时频分析摘要图 (3个子图)

    Args:
        time, signal: 原始信号
        stft_freqs, stft_times, stft_power: STFT结果
        ls_freqs, ls_power: Lomb-Scargle结果
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # 原始信号
    axes[0].plot(time, signal)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal')
    axes[0].grid(True, alpha=0.3)

    # STFT
    power_db = 10 * np.log10(stft_power + 1e-10)
    im = axes[1].pcolormesh(
        stft_times, stft_freqs, power_db,
        shading='gouraud', cmap='viridis'
    )
    axes[1].set_ylabel('Frequency (1/Ma)')
    axes[1].set_title('Spectrogram (STFT)')
    plt.colorbar(im, ax=axes[1])

    # Lomb-Scargle
    axes[2].plot(ls_freqs, ls_power)
    axes[2].set_xlabel('Frequency (1/Ma)')
    axes[2].set_ylabel('Power')
    axes[2].set_title('Lomb-Scargle Periodogram')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
