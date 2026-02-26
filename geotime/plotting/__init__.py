"""
绘图工具模块
"""

from geotime.plotting.bar_plot import plot_line, plot_bar
from geotime.plotting.signal_compare import plot_comparison
from geotime.plotting.psd_plot import plot_psd_top3
from geotime.plotting.traj_plot import plot_trajectory_matrix
from geotime.plotting.wavelet_plot import wavelet_analysis_plot
from geotime.plotting.multiple_psd_plot import plot_psd_decomposition

# 时频分析绘图
from geotime.plotting.timeseries_plots import (
    plot_spectrogram,
    plot_lomb_scargle,
    plot_time_frequency_summary,
)

# 元频谱分析绘图
from geotime.plotting.metaspectral_plots import (
    plot_reconstruction,
    plot_decomposition_components,
    plot_contribution_pie,
    plot_range_contributions,
    plot_extracted_signal,
    plot_metaspectral_full_analysis,
    plot_multiple_periods_comparison,
    plot_smoothing_history,
)

__all__ = [
    # 基础绘图
    "plot_line",
    "plot_bar",
    "plot_comparison",
    "plot_psd_top3",
    "plot_trajectory_matrix",
    "wavelet_analysis_plot",
    "plot_psd_decomposition",
    # 时频分析绘图
    "plot_spectrogram",
    "plot_lomb_scargle",
    "plot_time_frequency_summary",
    # 元频谱分析绘图
    "plot_reconstruction",
    "plot_decomposition_components",
    "plot_contribution_pie",
    "plot_range_contributions",
    "plot_extracted_signal",
    "plot_metaspectral_full_analysis",
    "plot_multiple_periods_comparison",
    "plot_smoothing_history",
]
