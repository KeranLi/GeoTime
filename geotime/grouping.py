"""
数据分组和聚合工具

用于处理碎屑锆石等按样品分组的数据
"""

from typing import Dict, List, Optional, Callable, Union, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np


def group_by_sample(
    df: pd.DataFrame,
    sample_id_column: str = 'Published_Sample_ID',
    agg_dict: Optional[Dict[str, Union[str, List[str], Callable]]] = None,
) -> pd.DataFrame:
    """
    按样品ID分组聚合

    Args:
        df: 输入DataFrame
        sample_id_column: 样品ID列名
        agg_dict: 聚合字典，如 {'Best Age': list, 'Est. Depos. Age (Ma)': 'first'}

    Returns:
        分组后的DataFrame
    """
    if agg_dict is None:
        # 默认聚合策略
        agg_dict = {
            'Best Age': list,
            'Est. Depos. Age (Ma)': 'first',
            'Latitude': 'first',
            'Longitude': 'first',
            'Region': 'first',
        }
    
    # 只保留存在的列
    available_cols = [col for col in agg_dict.keys() if col in df.columns]
    agg_dict = {col: agg_dict[col] for col in available_cols}
    
    return df.groupby(sample_id_column).agg(agg_dict).reset_index()


def extract_ages_by_sample(
    df: pd.DataFrame,
    sample_id_column: str = 'Published_Sample_ID',
    age_column: str = 'Best Age',
    deposit_age_column: str = 'Est. Depos. Age (Ma)',
) -> Dict[str, Dict[str, any]]:
    """
    提取每个样品的年龄列表和沉积年龄

    Args:
        df: 输入DataFrame
        sample_id_column: 样品ID列名
        age_column: 年龄列名
        deposit_age_column: 沉积年龄列名

    Returns:
        {样品ID: {'ages': [...], 'deposit_age': ...}}
    """
    result = {}
    
    for sample_id, group in df.groupby(sample_id_column):
        # 转换年龄为数值
        ages = pd.to_numeric(group[age_column], errors='coerce').dropna().tolist()
        
        # 获取沉积年龄（取第一个有效值）
        deposit_ages = pd.to_numeric(group[deposit_age_column], errors='coerce').dropna()
        deposit_age = deposit_ages.iloc[0] if len(deposit_ages) > 0 else None
        
        if ages and deposit_age is not None:
            result[sample_id] = {
                'ages': ages,
                'deposit_age': deposit_age,
                'n_grains': len(ages),
            }
    
    return result


def aggregate_ages_to_series(
    ages_dict: Dict[str, Dict],
    min_grains: int = 1,
) -> pd.Series:
    """
    将多个样品的年龄聚合成一个Series

    Args:
        ages_dict: extract_ages_by_sample的输出
        min_grains: 最小颗粒数阈值

    Returns:
        包含所有年龄的pandas Series
    """
    all_ages = []
    
    for sample_id, data in ages_dict.items():
        if data['n_grains'] >= min_grains:
            all_ages.extend(data['ages'])
    
    return pd.Series(all_ages)


def calculate_age_frequencies(
    ages: Union[List[float], pd.Series, np.ndarray],
    bins: Optional[np.ndarray] = None,
    bin_size: float = 1.0,
    min_age: Optional[float] = None,
    max_age: Optional[float] = None,
) -> pd.DataFrame:
    """
    计算年龄频率分布

    Args:
        ages: 年龄数组
        bins: 自定义分箱边界
        bin_size: 分箱大小 (默认1Ma)
        min_age: 最小年龄
        max_age: 最大年龄

    Returns:
        DataFrame with 'age' and 'frequency' columns
    """
    ages = pd.to_numeric(pd.Series(ages).dropna())
    
    if min_age is None:
        min_age = ages.min()
    if max_age is None:
        max_age = ages.max()
    
    if bins is None:
        bins = np.arange(min_age, max_age + bin_size, bin_size)
    
    # 计算直方图
    counts, edges = np.histogram(ages, bins=bins)
    
    # 使用bin中心作为年龄值
    ages_center = (edges[:-1] + edges[1:]) / 2
    
    return pd.DataFrame({
        'age': ages_center,
        'frequency': counts,
    })


def merge_period_ages(
    period_ages: Dict[str, List[float]],
    periods_to_merge: List[str],
    new_period_name: str,
) -> Dict[str, List[float]]:
    """
    合并多个地质时代的年龄数据

    Args:
        period_ages: {地质时代: [年龄列表]}
        periods_to_merge: 要合并的地质时代列表
        new_period_name: 新地质时代名称

    Returns:
        更新后的字典
    """
    merged_ages = []
    
    for period in periods_to_merge:
        if period in period_ages:
            merged_ages.extend(period_ages[period])
    
    result = period_ages.copy()
    
    # 删除已合并的时代
    for period in periods_to_merge:
        result.pop(period, None)
    
    # 添加新合并的时代
    if merged_ages:
        result[new_period_name] = merged_ages
    
    return result


def split_by_deposit_age(
    df: pd.DataFrame,
    age_bins: List[Tuple[float, float]],
    deposit_age_column: str = 'Est. Depos. Age (Ma)',
    bin_labels: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    按沉积年龄范围分割数据

    Args:
        df: 输入DataFrame
        age_bins: 年龄范围列表 [(min1, max1), (min2, max2), ...]
        deposit_age_column: 沉积年龄列名
        bin_labels: 每个范围的标签

    Returns:
        {标签: DataFrame}
    """
    if bin_labels is None:
        bin_labels = [f"bin_{i}" for i in range(len(age_bins))]
    
    if len(bin_labels) != len(age_bins):
        raise ValueError("bin_labels和age_bins长度必须相同")
    
    result = {}
    
    for label, (min_age, max_age) in zip(bin_labels, age_bins):
        mask = (df[deposit_age_column] >= min_age) & (df[deposit_age_column] < max_age)
        result[label] = df[mask].copy()
    
    return result


def summarize_by_period(
    period_ages: Dict[str, Dict[str, List[float]]],
) -> pd.DataFrame:
    """
    生成地质时代的统计摘要

    Args:
        period_ages: get_period_ages的输出

    Returns:
        统计摘要DataFrame
    """
    summaries = []
    
    for period, samples in period_ages.items():
        all_ages = []
        for sample_id, ages in samples.items():
            all_ages.extend(ages)
        
        if all_ages:
            summaries.append({
                'period': period,
                'n_samples': len(samples),
                'n_grains': len(all_ages),
                'mean_age': np.mean(all_ages),
                'median_age': np.median(all_ages),
                'std_age': np.std(all_ages),
                'min_age': np.min(all_ages),
                'max_age': np.max(all_ages),
            })
    
    return pd.DataFrame(summaries)
