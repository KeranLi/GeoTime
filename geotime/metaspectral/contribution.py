"""
贡献率计算模块

计算各周期成分对总信号的贡献率，支持特定时间信号的提取
"""

from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


@dataclass
class SignalFrequencyRange:
    """信号频率范围定义"""
    name: str
    min_period: float
    max_period: float
    description: str = ""
    
    def __contains__(self, period: float) -> bool:
        """检查周期是否在该范围内"""
        return self.min_period <= period < self.max_period


# 预定义的信号频率范围 (基于常见地质周期)
SIGNAL_RANGES = {
    'long_term_trend': SignalFrequencyRange(
        name='Long-term Trend',
        min_period=500,
        max_period=float('inf'),
        description='超长期趋势 (>500 Ma)',
    ),
    'supercontinent': SignalFrequencyRange(
        name='Supercontinent Cycle',
        min_period=300,
        max_period=500,
        description='超大陆周期 (300-500 Ma)',
    ),
    'wilson_cycle': SignalFrequencyRange(
        name='Wilson Cycle',
        min_period=200,
        max_period=300,
        description='Wilson旋回 (200-300 Ma)',
    ),
    'plate_tectonic': SignalFrequencyRange(
        name='Plate Tectonic',
        min_period=100,
        max_period=200,
        description='板块构造周期 (100-200 Ma)',
    ),
    'orogenic': SignalFrequencyRange(
        name='Orogenic Cycle',
        min_period=50,
        max_period=100,
        description='造山周期 (50-100 Ma)',
    ),
    'sedimentary': SignalFrequencyRange(
        name='Sedimentary Cycle',
        min_period=20,
        max_period=50,
        description='沉积周期 (20-50 Ma)',
    ),
    'milankovitch_long': SignalFrequencyRange(
        name='Milankovitch (Long)',
        min_period=10,
        max_period=20,
        description='米兰科维奇长周期 (10-20 Ma)',
    ),
    'milankovitch_short': SignalFrequencyRange(
        name='Milankovitch (Short)',
        min_period=1,
        max_period=10,
        description='米兰科维奇短周期 (1-10 Ma)',
    ),
    'high_frequency': SignalFrequencyRange(
        name='High Frequency',
        min_period=0,
        max_period=1,
        description='高频噪声 (<1 Ma)',
    ),
}


@dataclass
class ContributionResult:
    """贡献率计算结果"""
    component_index: int
    period: float
    energy: float
    contribution_rate: float
    signal_range: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'component': self.component_index,
            'period_Ma': self.period,
            'energy': self.energy,
            'contribution_rate': self.contribution_rate,
            'contribution_pct': self.contribution_rate * 100,
            'signal_range': self.signal_range,
        }


@dataclass
class SignalExtractionResult:
    """特定信号提取结果"""
    signal_name: str
    period_range: SignalFrequencyRange
    combined_signal: np.ndarray
    components: List[int]
    total_contribution: float
    dominant_period: float
    
    def plot(self, age: np.ndarray, ax=None):
        """
        绘制提取的信号
        
        注意: 实际绘图逻辑在 geotime.plotting.metaspectral_plots.plot_extracted_signal
        """
        from geotime.plotting import plot_extracted_signal
        
        return plot_extracted_signal(
            age,
            self.combined_signal,
            self.signal_name,
            self.total_contribution,
            ax=ax,
        )


def calculate_contribution_rates(
    decomposition_result,
    signal_ranges: Optional[Dict[str, SignalFrequencyRange]] = None,
) -> pd.DataFrame:
    """
    计算各成分的贡献率

    Args:
        decomposition_result: 分解结果
        signal_ranges: 信号范围定义

    Returns:
        贡献率DataFrame
    """
    if signal_ranges is None:
        signal_ranges = SIGNAL_RANGES
    
    contributions = []
    
    for comp in decomposition_result.components:
        # 确定属于哪个信号范围
        signal_range_name = None
        if comp.period is not None:
            for name, range_def in signal_ranges.items():
                if comp.period in range_def:
                    signal_range_name = name
                    break
        
        contributions.append(ContributionResult(
            component_index=comp.index,
            period=comp.period if comp.period else 0,
            energy=comp.energy,
            contribution_rate=comp.contribution,
            signal_range=signal_range_name,
        ))
    
    # 转换为DataFrame
    df = pd.DataFrame([c.to_dict() for c in contributions])
    
    # 按贡献率排序
    df = df.sort_values('contribution_rate', ascending=False)
    
    return df


def calculate_range_contributions(
    decomposition_result,
    signal_ranges: Optional[Dict[str, SignalFrequencyRange]] = None,
) -> pd.DataFrame:
    """
    计算各信号范围的总贡献率

    Args:
        decomposition_result: 分解结果
        signal_ranges: 信号范围定义

    Returns:
        范围贡献率DataFrame
    """
    if signal_ranges is None:
        signal_ranges = SIGNAL_RANGES
    
    # 计算各成分的贡献
    comp_df = calculate_contribution_rates(decomposition_result, signal_ranges)
    
    # 按范围分组
    range_stats = []
    
    for range_name in signal_ranges.keys():
        range_comps = comp_df[comp_df['signal_range'] == range_name]
        
        if len(range_comps) > 0:
            total_contribution = range_comps['contribution_rate'].sum()
            total_energy = range_comps['energy'].sum()
            dominant_period = range_comps.loc[range_comps['contribution_rate'].idxmax(), 'period_Ma']
            n_components = len(range_comps)
        else:
            total_contribution = 0
            total_energy = 0
            dominant_period = np.nan
            n_components = 0
        
        range_stats.append({
            'signal_range': range_name,
            'description': signal_ranges[range_name].description,
            'period_min_Ma': signal_ranges[range_name].min_period,
            'period_max_Ma': signal_ranges[range_name].max_period,
            'total_contribution': total_contribution,
            'total_contribution_pct': total_contribution * 100,
            'total_energy': total_energy,
            'dominant_period_Ma': dominant_period,
            'n_components': n_components,
        })
    
    return pd.DataFrame(range_stats)


def extract_specific_signals(
    decomposition_result,
    target_ranges: Union[str, List[str], SignalFrequencyRange, List[SignalFrequencyRange]],
    signal_ranges: Optional[Dict[str, SignalFrequencyRange]] = None,
) -> Union[SignalExtractionResult, List[SignalExtractionResult]]:
    """
    提取特定周期范围的信号

    Args:
        decomposition_result: 分解结果
        target_ranges: 目标范围名称或定义
        signal_ranges: 信号范围定义字典

    Returns:
        SignalExtractionResult或列表
    """
    if signal_ranges is None:
        signal_ranges = SIGNAL_RANGES
    
    # 处理单个或列表输入
    single_input = False
    if isinstance(target_ranges, (str, SignalFrequencyRange)):
        target_ranges = [target_ranges]
        single_input = True
    
    results = []
    
    for target in target_ranges:
        # 解析目标范围
        if isinstance(target, str):
            if target not in signal_ranges:
                raise ValueError(f"未知信号范围: {target}")
            period_range = signal_ranges[target]
            range_name = target
        else:
            period_range = target
            range_name = target.name
        
        # 找到属于该范围的成分
        range_components = [
            comp for comp in decomposition_result.components
            if comp.period is not None and comp.period in period_range
        ]
        
        if not range_components:
            # 没有成分匹配，返回零信号
            combined_signal = np.zeros_like(decomposition_result.original_signal)
            total_contribution = 0
            dominant_period = 0
        else:
            # 合成信号
            combined_signal = sum(comp.signal for comp in range_components)
            total_contribution = sum(comp.contribution for comp in range_components)
            
            # 主导周期 (贡献最大的)
            dominant_comp = max(range_components, key=lambda c: c.contribution)
            dominant_period = dominant_comp.period
        
        result = SignalExtractionResult(
            signal_name=range_name,
            period_range=period_range,
            combined_signal=combined_signal,
            components=[comp.index for comp in range_components],
            total_contribution=total_contribution,
            dominant_period=dominant_period,
        )
        
        results.append(result)
    
    return results[0] if single_input else results


def calculate_cumulative_contribution(
    decomposition_result,
    period_threshold: float,
) -> Dict[str, float]:
    """
    计算特定周期阈值以下的累积贡献率

    Args:
        decomposition_result: 分解结果
        period_threshold: 周期阈值 (Ma)

    Returns:
        {'below_threshold': x, 'above_threshold': y}
    """
    below_energy = 0
    above_energy = 0
    
    for comp in decomposition_result.components:
        if comp.period is not None and comp.period < period_threshold:
            below_energy += comp.energy
        else:
            above_energy += comp.energy
    
    total = below_energy + above_energy
    
    return {
        'below_threshold': below_energy / total if total > 0 else 0,
        'above_threshold': above_energy / total if total > 0 else 0,
        'threshold_Ma': period_threshold,
    }


def find_dominant_periods(
    decomposition_result,
    n_periods: int = 5,
    min_contribution: float = 0.01,
) -> pd.DataFrame:
    """
    找出主导周期

    Args:
        decomposition_result: 分解结果
        n_periods: 返回的周期数量
        min_contribution: 最小贡献率阈值

    Returns:
        DataFrame with columns: period_Ma, contribution, component
    """
    # 过滤和排序
    valid_components = [
        comp for comp in decomposition_result.components
        if comp.period is not None and comp.contribution >= min_contribution
    ]
    
    sorted_components = sorted(
        valid_components,
        key=lambda c: c.contribution,
        reverse=True
    )[:n_periods]
    
    return pd.DataFrame([
        {
            'rank': i + 1,
            'component': comp.index,
            'period_Ma': comp.period,
            'contribution': comp.contribution,
            'contribution_pct': comp.contribution * 100,
        }
        for i, comp in enumerate(sorted_components)
    ])


def compare_contributions(
    results: Dict[str, 'DecompositionResult'],
    signal_ranges: Optional[Dict[str, SignalFrequencyRange]] = None,
) -> pd.DataFrame:
    """
    比较多个分解结果的贡献率

    Args:
        results: {名称: 分解结果}字典
        signal_ranges: 信号范围定义

    Returns:
        比较DataFrame
    """
    if signal_ranges is None:
        signal_ranges = SIGNAL_RANGES
    
    all_stats = []
    
    for name, result in results.items():
        range_df = calculate_range_contributions(result, signal_ranges)
        range_df['sample'] = name
        all_stats.append(range_df)
    
    combined = pd.concat(all_stats, ignore_index=True)
    
    # 透视表
    pivot = combined.pivot(
        index='signal_range',
        columns='sample',
        values='total_contribution_pct'
    )
    
    return pivot


def export_contribution_report(
    decomposition_result,
    output_file: str,
    signal_ranges: Optional[Dict[str, SignalFrequencyRange]] = None,
):
    """
    导出贡献率报告

    Args:
        decomposition_result: 分解结果
        output_file: 输出文件路径 (.xlsx或.csv)
        signal_ranges: 信号范围定义
    """
    # 生成多个表格
    component_df = calculate_contribution_rates(decomposition_result, signal_ranges)
    range_df = calculate_range_contributions(decomposition_result, signal_ranges)
    dominant_df = find_dominant_periods(decomposition_result)
    
    # 保存
    if output_file.endswith('.xlsx'):
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            component_df.to_excel(writer, sheet_name='Components', index=False)
            range_df.to_excel(writer, sheet_name='Signal_Ranges', index=False)
            dominant_df.to_excel(writer, sheet_name='Dominant_Periods', index=False)
    else:
        # CSV格式只保存主表
        range_df.to_csv(output_file, index=False)
    
    print(f"报告已保存: {output_file}")
