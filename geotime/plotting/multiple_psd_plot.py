"""
多视角PSD分析图绘制模块
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional


def plot_psd_decomposition(
    age: np.ndarray,
    data: np.ndarray,
    sampling_freq: float = 1.0,
    title: str = "PSD Analysis",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    绘制多视角PSD分解图

    展示不同参数设置下的功率谱密度分析结果

    Args:
        age: 年龄数组
        data: 数据数组
        sampling_freq: 采样频率
        title: 总标题
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # 原始数据
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(age, data, label='Detrended Data', color='steelblue')
    ax1.set_xlabel('Age (Ma)')
    ax1.set_ylabel('Normalized Signal')
    ax1.set_title('Original Sequence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 零填充对比
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.psd(data, NFFT=len(data), pad_to=len(data), Fs=sampling_freq, 
            label='No padding', alpha=0.7)
    ax2.psd(data, NFFT=len(data), pad_to=len(data)*2, Fs=sampling_freq,
            label='2x padding', alpha=0.7)
    ax2.psd(data, NFFT=len(data), pad_to=len(data)*4, Fs=sampling_freq,
            label='4x padding', alpha=0.7)
    ax2.set_title('Zero Padding Effect')
    ax2.legend()
    
    # 分块大小对比
    ax3 = fig.add_subplot(gs[1, 1])
    nfft_vals = [len(data)//4, len(data)//2, len(data)]
    for nfft in nfft_vals:
        ax3.psd(data, NFFT=nfft, Fs=sampling_freq, 
                label=f'NFFT={nfft}', alpha=0.7)
    ax3.set_title('Block Size Effect')
    ax3.legend()
    
    # 重叠对比
    ax4 = fig.add_subplot(gs[1, 2])
    noverlap_vals = [0, len(data)//4, len(data)//2]
    for noverlap in noverlap_vals:
        ax4.psd(data, NFFT=len(data)//2, noverlap=noverlap, Fs=sampling_freq,
                label=f'Overlap={noverlap}', alpha=0.7)
    ax4.set_title('Overlap Effect')
    ax4.legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    return fig
