"""
地质时代处理模块

提供地质时代定义、映射、划分等功能
"""

from geotime.geological.periods import (
    GEOLOGICAL_PERIODS,
    GEOLOGICAL_PERIODS_BOUNDARIES,
    assign_geological_period,
    filter_by_period,
    get_period_ages,
)

__all__ = [
    "GEOLOGICAL_PERIODS",
    "GEOLOGICAL_PERIODS_BOUNDARIES", 
    "assign_geological_period",
    "filter_by_period",
    "get_period_ages",
]
