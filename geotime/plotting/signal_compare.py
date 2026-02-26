"""
信号对比图绘制模块

用于对比原始信号和去趋势后的信号
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


def plot_comparison(
    x: np.ndarray,
    original: np.ndarray,
    detrended: np.ndarray,
    title: str = "Signal Comparison",
    xlabel: str = "Age (Ma)",
    ylabel: str = "Value",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    original_label: str = "Original",
    detrended_label: str = "Detrended",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    绘制原始信号与去趋势信号的对比图

    Args:
        x: X轴数据 (如年龄)
        original: 原始信号
        detrended: 去趋势后的信号
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        xlim: X轴范围
        ylim: Y轴范围
        original_label: 原始信号图例标签
        detrended_label: 去趋势信号图例标签
        ax: 可选的Axes对象

    Returns:
        Axes对象
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(x, original, label=original_label, alpha=0.7)
    ax.plot(x, detrended, label=detrended_label, alpha=0.7)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax
