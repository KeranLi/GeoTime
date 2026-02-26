"""
小波分析结果可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def wavelet_analysis_plot(
    age: np.ndarray,
    data: np.ndarray,
    time: np.ndarray,
    period: np.ndarray,
    power: np.ndarray,
    global_ws: np.ndarray,
    global_signif: np.ndarray,
    scaleavg_signif: np.ndarray,
    scale_avg: np.ndarray,
    sig95: np.ndarray,
    coi: np.ndarray,
    ylabel_data: str = "Value",
    xlim: Optional[tuple] = None,
    figsize: tuple = (15, 12),
) -> plt.Figure:
    """
    绘制完整的小波分析结果图

    包含四个子图：
    (a) 原始数据
    (b) 小波功率谱
    (c) 全局小波谱
    (d) 尺度平均时间序列

    Args:
        age: 年龄数组
        data: 原始数据
        time: 时间数组
        period: 周期数组
        power: 小波功率
        global_ws: 全局小波谱
        global_signif: 全局显著性水平
        scaleavg_signif: 尺度平均显著性
        scale_avg: 尺度平均
        sig95: 95%显著性水平
        coi: 影响锥 (Cone of Influence)
        ylabel_data: 数据Y轴标签
        xlim: X轴范围
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    if xlim is None:
        xlim = (0, 32)
    
    fig = plt.figure(figsize=figsize)
    
    # (a) 原始数据
    ax1 = fig.add_axes([0.1, 0.75, 0.65, 0.2])
    ax1.plot(age, data)
    ax1.set_xlabel('Age (Ma)')
    ax1.set_ylabel(ylabel_data)
    ax1.set_title('(a) Original Data')
    ax1.set_xlim(xlim)
    
    # (b) 小波功率谱
    ax2 = fig.add_axes([0.1, 0.37, 0.65, 0.28])
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    
    Yticks = 2 ** np.arange(
        int(np.log2(period.min())),
        int(np.log2(period.max())) + 1
    )
    
    cf = ax2.contourf(
        time, np.log2(period), np.log2(power),
        np.log2(levels), extend='both', cmap=plt.cm.seismic
    )
    ax2.contour(time, np.log2(period), sig95, levels=[-99, 1], colors='k')
    ax2.plot(time, np.log2(coi), 'k--')
    ax2.set_xlabel('Age (Ma)')
    ax2.set_ylabel('Period (Ma)')
    ax2.set_title('(b) Wavelet Power Spectrum')
    ax2.set_ylim(np.log2(period.max()), np.log2(period.min()))
    
    # 设置Y轴刻度
    ax2.set_yticks(np.log2(Yticks))
    ax2.set_yticklabels([f'{y:.1f}' for y in Yticks])
    
    # (c) 全局小波谱
    ax3 = fig.add_axes([0.77, 0.37, 0.2, 0.28])
    ax3.plot(global_ws, np.log2(period))
    ax3.plot(global_signif, np.log2(period), 'k--')
    ax3.set_xlabel('Power')
    ax3.set_title('(c) Global Wavelet Spectrum')
    ax3.set_ylim(np.log2(period.max()), np.log2(period.min()))
    ax3.set_yticklabels([])
    ax3.set_xlim(0, 1.25 * global_ws.max())
    
    # (d) 尺度平均时间序列
    ax4 = fig.add_axes([0.1, 0.07, 0.65, 0.2])
    ax4.plot(xlim, [scaleavg_signif, scaleavg_signif], 'k--', label='Significance')
    ax4.plot(time, scale_avg, label='Scale Average')
    ax4.set_xlabel('Age (Ma)')
    ax4.set_ylabel('Avg Variance')
    ax4.set_title('(d) Scale-average Time Series')
    ax4.set_xlim(xlim)
    ax4.legend()
    
    plt.colorbar(cf, ax=[ax2, ax3], location='right', label='Power')
    
    return fig
