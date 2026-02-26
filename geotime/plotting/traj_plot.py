"""
轨迹矩阵可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_trajectory_matrix(
    matrix: np.ndarray,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    cmap: str = 'coolwarm',
    title: str = "Trajectory Matrix",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    可视化轨迹矩阵

    Args:
        matrix: 轨迹矩阵
        max_rows: 最大显示行数
        max_cols: 最大显示列数
        cmap: 颜色映射
        title: 图表标题
        ax: 可选的Axes对象

    Returns:
        Axes对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 切片
    if max_rows:
        matrix = matrix[:max_rows]
    if max_cols:
        matrix = matrix[:, :max_cols]
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Trajectory Index")
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Value')
    
    return ax
