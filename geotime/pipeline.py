"""
完整分析流程管道

整合地质时代处理、分组、平滑、时频分析等步骤
"""

from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geotime.geological import (
    get_period_ages,
    GEOLOGICAL_PERIODS_BOUNDARIES,
)
from geotime.grouping import calculate_age_frequencies
from geotime.smoothing import SmoothingPipeline
from geotime.timeseries import TimeFrequencyAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class PeriodAnalysisResult:
    """单个地质时代的分析结果"""
    period: str
    n_samples: int
    n_grains: int
    ages: np.ndarray = field(repr=False)
    frequencies: pd.DataFrame = field(repr=False)
    smoothed: Optional[pd.DataFrame] = None
    stft_result: Optional[Any] = None
    lomb_scargle_result: Optional[Any] = None
    
    def plot_frequency(self, ax=None, show_smoothed: bool = True):
        """绘制频率分布图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        ax.bar(self.frequencies['age'], self.frequencies['frequency'], 
               alpha=0.5, label='Raw')
        
        if show_smoothed and self.smoothed is not None:
            ax.plot(self.smoothed['age'], self.smoothed['frequency'], 
                   'r-', linewidth=2, label='Smoothed')
        
        ax.set_xlabel('Age (Ma)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.period} (n={self.n_grains})')
        ax.legend()
        
        return fig


class DetritalAnalysisPipeline:
    """
    碎屑锆石分析完整流程

    使用示例:
        >>> pipeline = DetritalAnalysisPipeline()
        >>> results = pipeline.run(
        ...     data_file='./data/Qiagtang_large.xlsx',
        ...     periods=['Upper_Triassic', 'Middle_Triassic', 'Lower_Triassic'],
        ...     smooth_frac=0.1,
        ...     stft_nperseg=256,
        ... )
    """

    def __init__(
        self,
        sample_id_column: str = 'Published_Sample_ID',
        age_column: str = 'Best Age',
        deposit_age_column: str = 'Est. Depos. Age (Ma)',
    ):
        """
        Args:
            sample_id_column: 样品ID列名
            age_column: 年龄列名
            deposit_age_column: 沉积年龄列名
        """
        self.sample_id_column = sample_id_column
        self.age_column = age_column
        self.deposit_age_column = deposit_age_column
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.period_ages: Dict[str, List[float]] = {}
        self.results: Dict[str, PeriodAnalysisResult] = {}

    def load_data(self, file_path: Union[str, Path]) -> 'DetritalAnalysisPipeline':
        """加载数据文件"""
        file_path = Path(file_path)
        logger.info(f"加载数据: {file_path}")
        
        self.raw_data = pd.read_excel(file_path)
        logger.info(f"  加载了 {len(self.raw_data)} 行数据")
        
        return self

    def extract_period_ages(
        self,
        periods: Optional[List[str]] = None,
    ) -> 'DetritalAnalysisPipeline':
        """
        按地质时代提取年龄数据

        Args:
            periods: 要分析的地质时代列表 (None则分析所有)
        """
        if self.raw_data is None:
            raise ValueError("请先调用load_data()")
        
        logger.info("提取地质时代年龄数据...")
        
        period_dict = get_period_ages(
            self.raw_data,
            self.sample_id_column,
            self.age_column,
            self.deposit_age_column,
        )
        
        # 筛选指定的地质时代
        if periods:
            self.period_ages = {
                k: v for k, v in period_dict.items()
                if k in periods
            }
        else:
            self.period_ages = period_dict
        
        logger.info(f"  提取了 {len(self.period_ages)} 个地质时代")
        for period, samples in self.period_ages.items():
            n_grains = sum(len(ages) for ages in samples.values())
            logger.info(f"    {period}: {len(samples)} 样品, {n_grains} 颗粒")
        
        return self

    def process_period(
        self,
        period: str,
        bin_size: float = 1.0,
        smooth_frac: float = 0.1,
        stft_params: Optional[Dict] = None,
        lomb_scargle_params: Optional[Dict] = None,
    ) -> PeriodAnalysisResult:
        """
        处理单个地质时代

        Args:
            period: 地质时代名称
            bin_size: 频率计算分箱大小 (Ma)
            smooth_frac: LOWESS平滑参数
            stft_params: STFT参数字典
            lomb_scargle_params: Lomb-Scargle参数字典

        Returns:
            PeriodAnalysisResult
        """
        if period not in self.period_ages:
            raise ValueError(f"地质时代 {period} 不存在")
        
        samples = self.period_ages[period]
        
        # 聚合所有年龄
        all_ages = []
        for sample_id, ages in samples.items():
            all_ages.extend(ages)
        all_ages = np.array(all_ages)
        
        n_samples = len(samples)
        n_grains = len(all_ages)
        
        logger.info(f"处理 {period}: {n_samples} 样品, {n_grains} 颗粒")
        
        # 计算频率分布
        freq_df = calculate_age_frequencies(all_ages, bin_size=bin_size)
        
        # 平滑处理
        if smooth_frac > 0:
            pipeline = SmoothingPipeline(freq_df['age'], freq_df['frequency'])
            pipeline.lowess(frac=smooth_frac)
            smoothed_age, smoothed_freq = pipeline.get_result()
            smoothed_df = pd.DataFrame({
                'age': smoothed_age,
                'frequency': smoothed_freq,
            })
        else:
            smoothed_df = None
        
        # 使用平滑后的数据进行时频分析
        analysis_data = smoothed_df if smoothed_df is not None else freq_df
        
        # STFT分析
        stft_result = None
        if stft_params is not False:
            params = stft_params or {'nperseg': min(256, len(analysis_data) // 4)}
            try:
                analyzer = TimeFrequencyAnalyzer(
                    analysis_data['age'].values,
                    analysis_data['frequency'].values,
                )
                stft_result = analyzer.stft(**params)
                logger.info(f"  STFT完成")
            except Exception as e:
                logger.warning(f"  STFT失败: {e}")
        
        # Lomb-Scargle分析
        lomb_result = None
        if lomb_scargle_params is not False:
            params = lomb_scargle_params or {}
            try:
                if stft_result is None:
                    analyzer = TimeFrequencyAnalyzer(
                        analysis_data['age'].values,
                        analysis_data['frequency'].values,
                    )
                lomb_result = analyzer.lomb_scargle(**params)
                peaks = lomb_result.get_peaks(n_peaks=3)
                logger.info(f"  Lomb-Scargle完成, 主要周期: {peaks['periods']}")
            except Exception as e:
                logger.warning(f"  Lomb-Scargle失败: {e}")
        
        result = PeriodAnalysisResult(
            period=period,
            n_samples=n_samples,
            n_grains=n_grains,
            ages=all_ages,
            frequencies=freq_df,
            smoothed=smoothed_df,
            stft_result=stft_result,
            lomb_scargle_result=lomb_result,
        )
        
        self.results[period] = result
        return result

    def run(
        self,
        data_file: Union[str, Path],
        periods: Optional[List[str]] = None,
        bin_size: float = 1.0,
        smooth_frac: float = 0.1,
        stft_params: Optional[Dict] = None,
        lomb_scargle_params: Optional[Dict] = None,
    ) -> Dict[str, PeriodAnalysisResult]:
        """
        运行完整分析流程

        Args:
            data_file: 数据文件路径
            periods: 要分析的地质时代列表
            bin_size: 频率计算分箱大小
            smooth_frac: LOWESS平滑参数
            stft_params: STFT参数字典 (False则跳过)
            lomb_scargle_params: Lomb-Scargle参数字典 (False则跳过)

        Returns:
            {地质时代: PeriodAnalysisResult}
        """
        self.load_data(data_file)
        self.extract_period_ages(periods)
        
        for period in self.period_ages.keys():
            self.process_period(
                period,
                bin_size=bin_size,
                smooth_frac=smooth_frac,
                stft_params=stft_params,
                lomb_scargle_params=lomb_scargle_params,
            )
        
        logger.info(f"分析完成，共 {len(self.results)} 个地质时代")
        return self.results

    def plot_period_comparison(
        self,
        periods: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """
        绘制多个地质时代的对比图

        Args:
            periods: 要绘制的地质时代 (None则绘制所有)
            figsize: 图表尺寸

        Returns:
            Figure对象
        """
        if periods is None:
            periods = list(self.results.keys())
        
        n_periods = len(periods)
        n_cols = 3
        n_rows = (n_periods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_periods > 1 else [axes]
        
        for i, period in enumerate(periods):
            if period in self.results:
                self.results[period].plot_frequency(ax=axes[i])
        
        # 隐藏多余的子图
        for i in range(n_periods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig

    def plot_stft_grid(
        self,
        periods: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """
        绘制STFT频谱图网格
        """
        if periods is None:
            periods = [
                p for p, r in self.results.items()
                if r.stft_result is not None
            ]
        
        n_periods = len(periods)
        n_cols = 3
        n_rows = (n_periods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_periods > 1 else [axes]
        
        for i, period in enumerate(periods):
            if period in self.results:
                result = self.results[period]
                if result.stft_result:
                    ax = axes[i]
                    power_db = 10 * np.log10(result.stft_result.power + 1e-10)
                    im = ax.pcolormesh(
                        result.stft_result.times,
                        result.stft_result.frequencies,
                        power_db,
                        shading='gouraud',
                        cmap='viridis',
                    )
                    ax.set_title(period)
                    ax.set_xlabel('Age (Ma)')
                    ax.set_ylabel('Frequency (1/Ma)')
                    plt.colorbar(im, ax=ax)
        
        for i in range(n_periods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig

    def save_results(
        self,
        output_dir: Union[str, Path],
        save_csv: bool = True,
        save_plots: bool = True,
    ):
        """
        保存分析结果

        Args:
            output_dir: 输出目录
            save_csv: 是否保存CSV
            save_plots: 是否保存图片
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存结果到: {output_dir}")
        
        for period, result in self.results.items():
            # 保存频率数据
            if save_csv:
                result.frequencies.to_csv(
                    output_dir / f'{period}_frequency.csv',
                    index=False,
                )
                
                if result.smoothed is not None:
                    result.smoothed.to_csv(
                        output_dir / f'{period}_smoothed.csv',
                        index=False,
                    )
                
                # 保存Lomb-Scargle结果
                if result.lomb_scargle_result:
                    ls_df = pd.DataFrame({
                        'frequency': result.lomb_scargle_result.frequencies,
                        'power': result.lomb_scargle_result.power,
                    })
                    ls_df.to_csv(
                        output_dir / f'{period}_lomb_scargle.csv',
                        index=False,
                    )
                    
                    # 保存峰值
                    peaks = result.lomb_scargle_result.get_peaks(n_peaks=5)
                    peaks_df = pd.DataFrame(peaks)
                    peaks_df.to_csv(
                        output_dir / f'{period}_peaks.csv',
                        index=False,
                    )
        
        # 保存图片
        if save_plots and self.results:
            fig = self.plot_period_comparison()
            fig.savefig(output_dir / 'period_comparison.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            fig = self.plot_stft_grid()
            fig.savefig(output_dir / 'stft_grid.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        logger.info("保存完成")


def quick_analyze(
    data_file: Union[str, Path],
    output_dir: Union[str, Path],
    periods: Optional[List[str]] = None,
    **kwargs,
) -> DetritalAnalysisPipeline:
    """
    快速分析函数

    Args:
        data_file: 数据文件
        output_dir: 输出目录
        periods: 地质时代列表
        **kwargs: 传递给pipeline的参数

    Returns:
        DetritalAnalysisPipeline实例
    """
    pipeline = DetritalAnalysisPipeline()
    pipeline.run(data_file, periods=periods, **kwargs)
    pipeline.save_results(output_dir)
    return pipeline
