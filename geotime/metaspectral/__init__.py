"""
元频谱分析模块 (Meta-Spectral Analysis)

对碎屑锆石年龄频率分布进行频谱分解，计算各周期成分的贡献率。

主要功能:
- 离散频率分布的连续化重建
- 频谱分解 (SSA, EMD, 小波)
- 各周期成分贡献率计算
- 特定时间信号提取
"""

from geotime.metaspectral.reconstruction import (
    reconstruct_continuous_signal,
    InterpolationMethod,
)
from geotime.metaspectral.decomposition import (
    SpectralDecomposer,
    DecompositionMethod,
)
from geotime.metaspectral.contribution import (
    calculate_contribution_rates,
    extract_specific_signals,
    SignalFrequencyRange,
    SIGNAL_RANGES,
)
from geotime.metaspectral.analyzer import (
    DetritalSpectrumAnalyzer,
    SpectrumAnalysisResult,
)

__all__ = [
    # 信号重建
    "reconstruct_continuous_signal",
    "InterpolationMethod",
    # 频谱分解
    "SpectralDecomposer",
    "DecompositionMethod",
    # 贡献率计算
    "calculate_contribution_rates",
    "extract_specific_signals",
    "SignalFrequencyRange",
    # 主分析器
    "DetritalSpectrumAnalyzer",
    "SpectrumAnalysisResult",
]
