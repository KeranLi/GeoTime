"""
功率谱密度 (PSD) 图绘制模块
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_psd_top3(
    freq: np.ndarray,
    psd: np.ndarray,
    title: str = "Power Spectrum",
    top3_freqs: Optional[np.ndarray] = None,
    top3_periods: Optional[np.ndarray] = None,
    xlabel: str = "Frequency",
    ylabel: str = "Amplitude Spectra",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    绘制功率谱并标注前3个峰值

    Args:
        freq: 频率数组
        psd: 功率谱密度数组
        title: 图表标题
        top3_freqs: 前3个峰值频率
        top3_periods: 前3个峰值周期
        xlabel: X轴标签
        ylabel: Y轴标签
        ax: 可选的Axes对象

    Returns:
        Axes对象
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # 双对数坐标
    ax.loglog(freq, psd)
    
    # 标注前3峰值
    if top3_freqs is not None and top3_periods is not None:
        for i, (f, period) in enumerate(zip(top3_freqs, top3_periods)):
            # 找到对应的PSD值
            idx = np.argmin(np.abs(freq - f))
            psd_val = psd[idx]
            ax.annotate(
                f"{f:.2f} ({period:.1f} Ma)",
                xy=(f, psd_val),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
            )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which='both')
    
    return ax
