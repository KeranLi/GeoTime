"""
元频谱分析绘图模块

从 metaspectral 中提取的绘图函数
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_reconstruction(
    age_continuous: np.ndarray,
    frequency_continuous: np.ndarray,
    interpolation_method: str,
    original_age: Optional[np.ndarray] = None,
    original_freq: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    绘制信号重建结果

    Args:
        age_continuous: 连续年龄数组
        frequency_continuous: 连续频率数组
        interpolation_method: 插值方法名称
        original_age, original_freq: 原始离散数据
        ax: 可选的Axes对象
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 绘制重建信号
    ax.plot(age_continuous, frequency_continuous,
            'b-', linewidth=2, label='Reconstructed')

    # 绘制原始数据
    if original_age is not None and original_freq is not None:
        ax.bar(original_age, original_freq, alpha=0.3, color='gray',
               width=np.diff(original_age).mean() * 0.8 if len(original_age) > 1 else 1,
               label='Original')

    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Continuous Signal ({interpolation_method})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_decomposition_components(
    age: np.ndarray,
    components: List,
    n_components: int = 5,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    绘制分解成分

    Args:
        age: 年龄数组
        components: 成分列表 (包含 signal, period, index 属性)
        n_components: 绘制的成分数量
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    n_plot = min(n_components, len(components))
    fig, axes = plt.subplots(n_plot, 1, figsize=figsize, sharex=True)

    if n_plot == 1:
        axes = [axes]

    for i, (ax, comp) in enumerate(zip(axes, components[:n_plot])):
        ax.plot(age, comp.signal, linewidth=1.5)
        period_str = f"{comp.period:.1f}Ma" if hasattr(
            comp, 'period') and comp.period else "N/A"
        ax.set_ylabel(f'RC{comp.index}\n({period_str})')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Age (Ma)')
    fig.suptitle('Decomposition Components', fontsize=12)

    return fig


def plot_contribution_pie(
    contribution_df: pd.DataFrame,
    n_components: int = 6,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    绘制贡献率饼图

    Args:
        contribution_df: 贡献率DataFrame (需包含 contribution_rate 列)
        n_components: 显示的top成分数
        ax: 可选的Axes对象
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    top_components = contribution_df.head(n_components)
    others_contribution = 1 - top_components['contribution_rate'].sum()

    sizes = list(top_components['contribution_rate']) + [others_contribution]
    labels = []
    for _, row in top_components.iterrows():
        period_str = f"{row['period_Ma']:.1f}Ma" if 'period_Ma' in row else "N/A"
        labels.append(f"RC{int(row['component'])}\n({period_str})")
    labels.append('Others')

    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           textprops={'fontsize': 9})
    ax.set_title('Component Contributions')

    return fig


def plot_range_contributions(
    range_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    绘制信号范围贡献率条形图

    Args:
        range_df: 范围贡献率DataFrame
        ax: 可选的Axes对象
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 过滤掉0贡献的
    df = range_df[range_df['total_contribution_pct'] > 0].copy()

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['total_contribution_pct'])

    # 使用描述作为标签
    labels = [row['description'][:30] if 'description' in row else row.get('signal_range', 'Unknown')
              for _, row in df.iterrows()]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Contribution (%)')
    ax.set_title('Contribution by Geological Signal Range')
    ax.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['total_contribution_pct'] + 0.5, i,
                f"{row['total_contribution_pct']:.1f}%",
                va='center', fontsize=8)

    return fig


def plot_extracted_signal(
    age: np.ndarray,
    signal: np.ndarray,
    signal_name: str,
    contribution: float,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    绘制提取的特定信号

    Args:
        age: 年龄数组
        signal: 信号数组
        signal_name: 信号名称
        contribution: 贡献率
        ax: 可选的Axes对象
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(age, signal, linewidth=2, color='steelblue')
    ax.fill_between(age, signal, alpha=0.3)
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{signal_name.replace("_", " ").title()} '
                 f'(Contribution: {contribution:.1%})')
    ax.grid(True, alpha=0.3)

    return fig


def plot_metaspectral_full_analysis(
    period_name: str,
    original_ages: np.ndarray,
    original_frequencies: np.ndarray,
    reconstruction_age: np.ndarray,
    reconstruction_freq: np.ndarray,
    reconstruction_method: str,
    decomposition_components: List,
    contribution_df: pd.DataFrame,
    range_contribution_df: pd.DataFrame,
    dominant_periods: pd.DataFrame,
    extracted_signals: dict,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    绘制完整的元频谱分析图 (7个子图)

    Args:
        period_name: 地质时代名称
        original_ages, original_frequencies: 原始数据
        reconstruction_age, reconstruction_freq: 重建信号
        reconstruction_method: 重建方法
        decomposition_components: 分解成分列表
        contribution_df: 成分贡献率
        range_contribution_df: 范围贡献率
        dominant_periods: 主导周期
        extracted_signals: 提取的信号字典 {name: (age, signal, contribution)}
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 原始频率分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(original_ages, original_frequencies,
            alpha=0.5, color='gray', width=np.diff(original_ages).mean()*0.8 if len(original_ages) > 1 else 1)
    ax1.set_title('Original Frequency Distribution')
    ax1.set_xlabel('Age (Ma)')
    ax1.set_ylabel('Frequency')

    # 2. 重建信号
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(reconstruction_age, reconstruction_freq, 'b-', linewidth=2)
    ax2.set_title(f'Reconstructed Signal ({reconstruction_method})')
    ax2.set_xlabel('Age (Ma)')
    ax2.set_ylabel('Frequency')

    # 3. 分解成分 (前5个)
    ax3 = fig.add_subplot(gs[0, 2])
    for comp in decomposition_components[:5]:
        period_str = f"{comp.period:.1f}" if hasattr(
            comp, 'period') and comp.period else "N/A"
        ax3.plot(reconstruction_age, comp.signal,
                label=f'RC{comp.index} ({period_str}Ma)', alpha=0.7, linewidth=1.5)
    ax3.set_title('Decomposition Components')
    ax3.set_xlabel('Age (Ma)')
    ax3.legend(fontsize=8)

    # 4. 贡献率饼图
    ax4 = fig.add_subplot(gs[1, 0])
    plot_contribution_pie(contribution_df, n_components=6, ax=ax4)

    # 5. 信号范围贡献率
    ax5 = fig.add_subplot(gs[1, 1:])
    plot_range_contributions(range_contribution_df, ax=ax5)

    # 6. 主导周期
    ax6 = fig.add_subplot(gs[2, 0])
    n_plot = min(5, len(dominant_periods))
    ax6.bar(range(n_plot), dominant_periods['contribution_pct'].head(n_plot))
    ax6.set_xticks(range(n_plot))
    ax6.set_xticklabels([f"{p:.1f}Ma" for p in dominant_periods['period_Ma'].head(n_plot)],
                       rotation=45, fontsize=9)
    ax6.set_ylabel('Contribution (%)')
    ax6.set_title('Top 5 Dominant Periods')

    # 7. 提取的信号
    ax7 = fig.add_subplot(gs[2, 1:])
    for name, (age, sig, contrib) in extracted_signals.items():
        ax7.plot(age, sig, label=f"{name.replace('_', ' ')} ({contrib:.1%})",
                linewidth=2)
    ax7.set_xlabel('Age (Ma)')
    ax7.set_ylabel('Amplitude')
    ax7.set_title('Extracted Geological Signals')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    plt.suptitle(f'Meta-Spectrum Analysis: {period_name}',
                 fontsize=14, fontweight='bold')

    return fig


def plot_multiple_periods_comparison(
    results: dict,
    signal_name: str = 'wilson_cycle',
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    绘制多个地质时代的信号对比

    Args:
        results: {period_name: (age, signal, contribution)} 字典
        signal_name: 信号名称
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    n_periods = len(results)
    fig, axes = plt.subplots(1, n_periods, figsize=figsize, sharey=True)

    if n_periods == 1:
        axes = [axes]

    for ax, (period_name, (age, sig, contrib)) in zip(axes, results.items()):
        ax.plot(age, sig, linewidth=2)
        ax.fill_between(age, sig, alpha=0.3)
        ax.set_title(f"{period_name}\n({contrib:.1%})")
        ax.set_xlabel('Age (Ma)')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Amplitude')
    fig.suptitle(f'{signal_name.replace("_", " ").title()} Signal Comparison',
                 fontsize=12)

    return fig


def plot_smoothing_history(
    history: list,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    绘制平滑处理历史

    Args:
        history: [(name, x, y), ...] 历史记录列表
        figsize: 图表尺寸

    Returns:
        Figure对象
    """
    n_steps = len(history)
    fig, axes = plt.subplots(n_steps, 1, figsize=figsize, sharex=True)

    if n_steps == 1:
        axes = [axes]

    for ax, (name, x, y) in zip(axes, history):
        ax.plot(x, y, linewidth=1.5)
        ax.set_title(name)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Age (Ma)')
    plt.tight_layout()

    return fig
