"""
GeoTime - 地质时间序列分析工具包

用于处理和分析地质年代数据的时间序列分析工具，包括：
- 地质时代处理
- 碎屑锆石数据分析
- SSA (奇异谱分析)
- 小波分析
- 频谱分析 (MTM, Lomb-Scargle, STFT)
- 滤波处理 (高斯滤波、Butterworth滤波)
"""

__version__ = "1.0.0"
__author__ = "Keran Li"

# 核心模块
from geotime.detrend import WeightedAverage, SSA, Filter
from geotime.dataloader import load_data, load_frequency_data

# 新增模块
from geotime.geological import (
    GEOLOGICAL_PERIODS,
    GEOLOGICAL_PERIODS_BOUNDARIES,
    assign_geological_period,
    get_period_ages,
)
from geotime.pipeline import DetritalAnalysisPipeline, quick_analyze
from geotime.analysis import GeoTimeSeriesAnalysis, run_standard_analysis

# 元频谱分析 (新增)
from geotime.metaspectral import (
    DetritalSpectrumAnalyzer,
    SpectrumAnalysisResult,
    reconstruct_continuous_signal,
    InterpolationMethod,
    SpectralDecomposer,
    DecompositionMethod,
    calculate_contribution_rates,
    extract_specific_signals,
    SIGNAL_RANGES,
)
from geotime.metaspectral.analyzer import quick_spectrum_analysis

__all__ = [
    # 核心
    "WeightedAverage",
    "SSA", 
    "Filter",
    "load_data",
    "load_frequency_data",
    # 地质时代
    "GEOLOGICAL_PERIODS",
    "GEOLOGICAL_PERIODS_BOUNDARIES",
    "assign_geological_period",
    "get_period_ages",
    # 管道
    "DetritalAnalysisPipeline",
    "quick_analyze",
    # 通用分析
    "GeoTimeSeriesAnalysis",
    "run_standard_analysis",
    # 元频谱分析 (核心新增)
    "DetritalSpectrumAnalyzer",
    "SpectrumAnalysisResult",
    "quick_spectrum_analysis",
    "reconstruct_continuous_signal",
    "InterpolationMethod",
    "SpectralDecomposer",
    "DecompositionMethod",
    "calculate_contribution_rates",
    "extract_specific_signals",
    "SIGNAL_RANGES",
]
