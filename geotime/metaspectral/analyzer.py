"""
碎屑锆石频谱分析主模块

整合信号重建、频谱分解和贡献率计算的完整流程
"""

from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from geotime.grouping import calculate_age_frequencies
from geotime.metaspectral.reconstruction import (
    reconstruct_continuous_signal,
    adaptive_reconstruction,
    ReconstructionResult,
    InterpolationMethod,
)
from geotime.metaspectral.decomposition import (
    SpectralDecomposer,
    DecompositionResult,
    DecompositionMethod,
)
from geotime.metaspectral.contribution import (
    calculate_contribution_rates,
    calculate_range_contributions,
    extract_specific_signals,
    find_dominant_periods,
    SIGNAL_RANGES,
)


logger = logging.getLogger(__name__)


@dataclass
class SpectrumAnalysisResult:
    """频谱分析完整结果"""
    period_name: str
    original_ages: np.ndarray = field(repr=False)
    original_frequencies: np.ndarray = field(repr=False)
    reconstruction: ReconstructionResult = field(repr=False)
    decomposition: DecompositionResult = field(repr=False)
    contribution_df: pd.DataFrame = field(repr=False)
    range_contribution_df: pd.DataFrame = field(repr=False)
    
    @property
    def total_energy(self) -> float:
        """总能量"""
        return self.decomposition.total_energy
    
    @property
    def dominant_periods(self) -> pd.DataFrame:
        """主导周期"""
        return find_dominant_periods(self.decomposition)
    
    def get_signal(self, range_name: str):
        """获取特定范围信号"""
        return extract_specific_signals(self.decomposition, range_name)
    
    def plot_full_analysis(self, figsize: Tuple[int, int] = (16, 12)):
        """
        绘制完整分析图
        
        注意: 实际绘图逻辑在 geotime.plotting.metaspectral_plots
        """
        from geotime.plotting import plot_metaspectral_full_analysis
        
        # 准备提取的信号
        extracted_signals = {}
        for name in ['supercontinent', 'wilson_cycle', 'plate_tectonic', 'orogenic']:
            try:
                sig_result = self.get_signal(name)
                extracted_signals[name] = (
                    self.reconstruction.age_continuous,
                    sig_result.combined_signal,
                    sig_result.total_contribution,
                )
            except:
                pass
        
        return plot_metaspectral_full_analysis(
            period_name=self.period_name,
            original_ages=self.original_ages,
            original_frequencies=self.original_frequencies,
            reconstruction_age=self.reconstruction.age_continuous,
            reconstruction_freq=self.reconstruction.frequency_continuous,
            reconstruction_method=self.reconstruction.interpolation_method,
            decomposition_components=self.decomposition.components,
            contribution_df=self.contribution_df,
            range_contribution_df=self.range_contribution_df,
            dominant_periods=self.dominant_periods,
            extracted_signals=extracted_signals,
            figsize=figsize,
        )


class DetritalSpectrumAnalyzer:
    """
    碎屑锆石频谱分析器

    整合完整的分析流程：
    离散频率 → 连续信号 → 频谱分解 → 贡献率计算
    """

    def __init__(
        self,
        reconstruction_method: InterpolationMethod = InterpolationMethod.CUBIC,
        decomposition_method: DecompositionMethod = DecompositionMethod.SSA,
    ):
        """
        Args:
            reconstruction_method: 信号重建方法
            decomposition_method: 频谱分解方法
        """
        self.reconstruction_method = reconstruction_method
        self.decomposition_method = decomposition_method
        self.results: Dict[str, SpectrumAnalysisResult] = {}

    def analyze_period(
        self,
        ages: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        period_name: str = "Unknown",
        bin_size: float = 1.0,
        reconstruction_params: Optional[Dict] = None,
        decomposition_params: Optional[Dict] = None,
    ) -> SpectrumAnalysisResult:
        """
        分析单个地质时代

        Args:
            ages: 年龄数组 (原始颗粒年龄或离散年龄点)
            frequencies: 频率数组 (如果ages已经是离散的年龄-频率对)
            period_name: 地质时代名称
            bin_size: 频率计算分箱大小
            reconstruction_params: 重建参数字典
            decomposition_params: 分解参数字典

        Returns:
            SpectrumAnalysisResult
        """
        logger.info(f"分析 {period_name}...")
        
        # 步骤1: 如果输入是原始年龄，计算频率分布
        if frequencies is None:
            freq_df = calculate_age_frequencies(ages, bin_size=bin_size)
            original_ages = freq_df['age'].values
            original_frequencies = freq_df['frequency'].values
        else:
            original_ages = np.asarray(ages)
            original_frequencies = np.asarray(frequencies)
        
        logger.info(f"  原始数据: {len(original_ages)} 个点")
        
        # 步骤2: 信号重建
        recon_params = reconstruction_params or {}
        reconstruction = reconstruct_continuous_signal(
            original_ages,
            original_frequencies,
            method=self.reconstruction_method,
            **recon_params,
        )
        
        logger.info(f"  重建信号: {len(reconstruction.age_continuous)} 个点, "
                   f"方法: {reconstruction.interpolation_method}")
        
        # 步骤3: 频谱分解
        decomp_params = decomposition_params or {}
        
        # 自适应窗口大小 (SSA需要)
        if self.decomposition_method == DecompositionMethod.SSA and 'window_size' not in decomp_params:
            signal_length = len(reconstruction.age_continuous)
            decomp_params['window_size'] = min(50, max(10, signal_length // 4))
        
        decomposer = SpectralDecomposer(
            reconstruction.age_continuous,
            reconstruction.frequency_continuous,
        )
        decomposition = decomposer.decompose(
            self.decomposition_method,
            **decomp_params,
        )
        
        logger.info(f"  分解完成: {len(decomposition.components)} 个成分")
        
        # 步骤4: 贡献率计算
        contribution_df = calculate_contribution_rates(decomposition)
        range_contribution_df = calculate_range_contributions(decomposition)
        
        # 打印主要贡献
        top3 = contribution_df.head(3)
        logger.info(f"  主要周期贡献:")
        for _, row in top3.iterrows():
            logger.info(f"    RC{int(row['component'])}: "
                       f"{row['period_Ma']:.1f}Ma ({row['contribution_pct']:.1f}%)")
        
        # 创建结果对象
        result = SpectrumAnalysisResult(
            period_name=period_name,
            original_ages=original_ages,
            original_frequencies=original_frequencies,
            reconstruction=reconstruction,
            decomposition=decomposition,
            contribution_df=contribution_df,
            range_contribution_df=range_contribution_df,
        )
        
        self.results[period_name] = result
        return result

    def analyze_from_dataframe(
        self,
        df: pd.DataFrame,
        age_column: str,
        period_column: str,
        period_name: str,
        **kwargs,
    ) -> SpectrumAnalysisResult:
        """
        从DataFrame分析

        Args:
            df: 输入DataFrame
            age_column: 年龄列名
            period_column: 地质时代列名 (用于过滤)
            period_name: 目标地质时代
            **kwargs: 其他参数

        Returns:
            SpectrumAnalysisResult
        """
        # 过滤特定地质时代
        period_data = df[df[period_column] == period_name][age_column].dropna()
        
        return self.analyze_period(
            ages=period_data.values,
            period_name=period_name,
            **kwargs,
        )

    def compare_periods(
        self,
        period_names: Optional[List[str]] = None,
        signal_range: str = 'wilson_cycle',
    ) -> pd.DataFrame:
        """
        比较多个地质时代的特定信号贡献

        Args:
            period_names: 地质时代列表 (None则比较所有)
            signal_range: 要比较的信号范围

        Returns:
            比较DataFrame
        """
        if period_names is None:
            period_names = list(self.results.keys())
        
        comparison = []
        
        for period_name in period_names:
            if period_name not in self.results:
                continue
            
            result = self.results[period_name]
            
            # 获取特定信号
            try:
                signal_result = result.get_signal(signal_range)
                signal_contribution = signal_result.total_contribution
            except:
                signal_contribution = 0
            
            comparison.append({
                'period': period_name,
                'n_grains': len(result.original_ages),
                'total_energy': result.total_energy,
                'dominant_period_Ma': result.dominant_periods.iloc[0]['period_Ma'] if len(result.dominant_periods) > 0 else np.nan,
                f'{signal_range}_contribution': signal_contribution,
            })
        
        return pd.DataFrame(comparison)

    def plot_period_comparison(
        self,
        period_names: Optional[List[str]] = None,
        signal_range: str = 'wilson_cycle',
        figsize: Tuple[int, int] = (14, 6),
    ):
        """
        绘制多个地质时代的信号对比
        
        注意: 实际绘图逻辑在 geotime.plotting.metaspectral_plots
        """
        from geotime.plotting import plot_multiple_periods_comparison
        
        if period_names is None:
            period_names = list(self.results.keys())
        
        # 收集信号数据
        results_dict = {}
        for period_name in period_names:
            if period_name not in self.results:
                continue
            
            result = self.results[period_name]
            try:
                signal_result = result.get_signal(signal_range)
                results_dict[period_name] = (
                    result.reconstruction.age_continuous,
                    signal_result.combined_signal,
                    signal_result.total_contribution,
                )
            except:
                pass
        
        return plot_multiple_periods_comparison(
            results_dict, signal_range, figsize
        )

    def export_results(
        self,
        output_dir: Union[str, Path],
        save_plots: bool = True,
    ):
        """
        导出分析结果

        Args:
            output_dir: 输出目录
            save_plots: 是否保存图片
        """
        import matplotlib.pyplot as plt
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for period_name, result in self.results.items():
            # 保存频率数据
            result.contribution_df.to_csv(
                output_dir / f'{period_name}_contributions.csv',
                index=False,
            )
            result.range_contribution_df.to_csv(
                output_dir / f'{period_name}_range_contributions.csv',
                index=False,
            )
            
            # 保存图片
            if save_plots:
                fig = result.plot_full_analysis()
                fig.savefig(output_dir / f'{period_name}_analysis.png', 
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        logger.info(f"结果已保存到: {output_dir}")


# 便捷函数
def quick_spectrum_analysis(
    ages: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    period_name: str = "Sample",
    return_full: bool = False,
) -> Union[pd.DataFrame, SpectrumAnalysisResult]:
    """
    快速频谱分析

    Args:
        ages: 年龄数组
        frequencies: 频率数组 (可选)
        period_name: 样品名称
        return_full: 是否返回完整结果

    Returns:
        贡献率DataFrame或完整结果
    """
    analyzer = DetritalSpectrumAnalyzer()
    result = analyzer.analyze_period(ages, frequencies, period_name)
    
    if return_full:
        return result
    else:
        return result.range_contribution_df
