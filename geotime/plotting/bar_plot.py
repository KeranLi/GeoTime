"""
柱状图/折线图绘制模块
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


def plot_line(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Data",
    xlabel: str = "Age (Ma)",
    ylabel: str = "Value",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    绘制折线图

    Args:
        x: X轴数据
        y: Y轴数据
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        xlim: X轴范围
        ylim: Y轴范围
        ax: 可选的Axes对象
        **kwargs: 传递给plt.plot的其他参数

    Returns:
        Axes对象
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(x, y, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    return ax


def plot_bar(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Data",
    xlabel: str = "Age (Ma)",
    ylabel: str = "Frequency",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    绘制柱状图

    Args:
        x: X轴数据
        y: Y轴数据
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        xlim: X轴范围
        ylim: Y轴范围
        ax: 可选的Axes对象
        **kwargs: 传递给plt.bar的其他参数

    Returns:
        Axes对象
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.bar(x, y, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    return ax
