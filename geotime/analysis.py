"""
地质时间序列分析工作流

整合数据加载、去趋势、频谱分析等步骤的统一接口
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geotime.dataloader import load_data, load_frequency_data
from geotime.detrend import WeightedAverage, SSA, Filter
from geotime.plotting import plot_comparison, plot_psd_top3, plot_trajectory_matrix


logger = logging.getLogger(__name__)


class GeoTimeSeriesAnalysis:
    """
    地质时间序列分析器

    整合完整的分析流程：数据加载 → 预处理 → SSA分解 → 频谱分析
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典，包含分析参数
        """
        self.config = config or {}
        self.data = None
        self.age = None
        self.values = None
        self.ssa_result = None
        self.spectrum_result = None
        
    def load_data(
        self,
        file_path: Union[str, Path],
        age_column: str = "Age(Ma)",
        value_column: Optional[str] = None,
        time_filter: Optional[float] = None,
    ) -> "GeoTimeSeriesAnalysis":
        """
        加载数据

        Args:
            file_path: 数据文件路径
            age_column: 年龄列名
            value_column: 数值列名 (频率分析时为None)
            time_filter: 时间过滤值

        Returns:
            self (链式调用)
        """
        file_path = Path(file_path)
        
        if value_column is None:
            # 频率统计模式
            self.age, self.values = load_frequency_data(file_path, age_column)
            logger.info(f"加载频率数据: {len(self.age)} 个年龄点")
        else:
            # 常规数据模式
            df = load_data(file_path, age_column, time_filter, [age_column, value_column])
            self.age = df[age_column].values
            self.values = df[value_column].values
            logger.info(f"加载数据: {len(self.age)} 个样本")
        
        return self

    def apply_weighted_average(
        self,
        weights: Optional[np.ndarray] = None,
        window: int = 5,
    ) -> "GeoTimeSeriesAnalysis":
        """
        应用加权移动平均去趋势

        Args:
            weights: 权重数组 (默认 [0.4, 0.15, 0.15, 0.15, 0.15])
            window: 窗口大小

        Returns:
            self
        """
        if weights is None:
            weights = np.array([0.4, 0.15, 0.15, 0.15, 0.15])
        
        wa = WeightedAverage(weights)
        series = pd.Series(self.values)
        self.values = wa.rolling(series, window=window).values
        
        logger.info(f"应用加权移动平均 (window={window})")
        return self

    def apply_ssa(
        self,
        window_size: int = 200,
        n_components: int = 4,
    ) -> "GeoTimeSeriesAnalysis":
        """
        应用SSA分解和重构

        Args:
            window_size: SSA窗口大小
            n_components: 用于重构的分量数

        Returns:
            self
        """
        # 中心化
        centered = self.values - np.mean(self.values)
        
        # SSA
        ssa = SSA(centered, window_size)
        trajectory = ssa.build_trajectory_matrix()
        components = ssa.decompose(trajectory)
        reconstructed = ssa.reconstruct(components, n_components)
        
        self.values = reconstructed
        self.ssa_result = {
            'components': components,
            'trajectory': trajectory,
        }
        
        logger.info(f"应用SSA (K={window_size}, n_components={n_components})")
        return self

    def analyze_spectrum(
        self,
        method: str = "mtm",
        nperseg: int = 128,
        find_peaks: bool = True,
        n_peaks: int = 3,
    ) -> Dict[str, Any]:
        """
        频谱分析

        Args:
            method: 分析方法 ('mtm', 'lomb_scargle')
            nperseg: MTM分段长度
            find_peaks: 是否查找峰值
            n_peaks: 查找的峰值数量

        Returns:
            分析结果字典
        """
        analyzer = Filter(self.values)
        
        # 去趋势
        detrended = analyzer.detrend()
        
        if method == "mtm":
            freqs, psd = analyzer.mtm_power_spectrum(nperseg)
        elif method == "lomb_scargle":
            freqs = np.linspace(0.01, 5, len(detrended))
            psd = analyzer.lomb_scargle(freqs)
        else:
            raise ValueError(f"未知方法: {method}")
        
        result = {
            'method': method,
            'frequencies': freqs,
            'power': psd,
            'detrended': detrended,
        }
        
        # 查找峰值
        if find_peaks:
            peak_indices, peak_heights = analyzer.find_peaks(psd, n_peaks)
            peak_freqs = freqs[peak_indices]
            peak_periods = 1 / peak_freqs
            
            result['peak_indices'] = peak_indices
            result['peak_frequencies'] = peak_freqs
            result['peak_periods'] = peak_periods
            
            logger.info(f"检测到周期: {peak_periods} Ma")
        
        self.spectrum_result = result
        return result

    def plot_comparison(
        self,
        original: Optional[np.ndarray] = None,
        title: str = "Signal Comparison",
    ) -> plt.Figure:
        """
        绘制处理前后对比图

        Args:
            original: 原始数据 (未设置则使用加载时的数据)
            title: 图表标题

        Returns:
            Figure对象
        """
        if original is None:
            # 重新加载原始数据进行对比
            original = self.values  # 简化处理
        
        from geotime.plotting import plot_comparison as pc
        fig, ax = plt.subplots(figsize=(12, 5))
        pc(self.age, original, self.values, title=title, ax=ax)
        return fig

    def plot_spectrum(self, title: str = "Power Spectrum") -> plt.Figure:
        """
        绘制频谱图

        Args:
            title: 图表标题

        Returns:
            Figure对象
        """
        if self.spectrum_result is None:
            raise RuntimeError("请先调用 analyze_spectrum()")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_psd_top3(
            self.spectrum_result['frequencies'],
            self.spectrum_result['power'],
            title=title,
            top3_freqs=self.spectrum_result.get('peak_frequencies'),
            top3_periods=self.spectrum_result.get('peak_periods'),
            ax=ax,
        )
        return fig

    def save_results(self, output_dir: Union[str, Path]):
        """
        保存分析结果

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存处理后的数据
        if self.age is not None and self.values is not None:
            pd.DataFrame({
                'Age': self.age,
                'Value': self.values,
            }).to_csv(output_dir / 'processed_data.csv', index=False)
        
        # 保存频谱结果
        if self.spectrum_result:
            pd.DataFrame({
                'Frequency': self.spectrum_result['frequencies'],
                'Power': self.spectrum_result['power'],
            }).to_csv(output_dir / 'spectrum.csv', index=False)
            
            if 'peak_periods' in self.spectrum_result:
                pd.DataFrame({
                    'Peak_Frequency': self.spectrum_result['peak_frequencies'],
                    'Peak_Period_Ma': self.spectrum_result['peak_periods'],
                }).to_csv(output_dir / 'peaks.csv', index=False)
        
        logger.info(f"结果已保存到: {output_dir}")


def run_standard_analysis(
    data_file: Union[str, Path],
    output_dir: Union[str, Path],
    age_column: str = "Age(Ma)",
    value_column: Optional[str] = None,
    ssa_window: int = 200,
    ssa_components: int = 4,
) -> GeoTimeSeriesAnalysis:
    """
    运行标准分析流程

    Args:
        data_file: 数据文件
        output_dir: 输出目录
        age_column: 年龄列名
        value_column: 数值列名
        ssa_window: SSA窗口
        ssa_components: SSA分量数

    Returns:
        分析器实例
    """
    analysis = GeoTimeSeriesAnalysis()
    
    # 加载数据
    analysis.load_data(data_file, age_column, value_column)
    
    # 如果没有指定value_column，已经是频率数据，跳过WA
    if value_column:
        analysis.apply_weighted_average()
    
    # SSA
    analysis.apply_ssa(ssa_window, ssa_components)
    
    # 频谱分析
    analysis.analyze_spectrum(method="mtm")
    
    # 保存结果
    analysis.save_results(output_dir)
    
    return analysis
