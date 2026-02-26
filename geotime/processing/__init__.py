"""
数据处理工具模块
"""

from geotime.processing.interpolate import interpolate_data
from geotime.processing.wavelet import WaveletAnalysis

__all__ = [
    "interpolate_data",
    "WaveletAnalysis",
]
