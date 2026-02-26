"""
地质时代定义和工具

基于国际地层委员会(ICS)的地质年代划分
"""

from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np

# 标准地质时代边界 (Ma) - ICS 2023/2024
GEOLOGICAL_PERIODS_BOUNDARIES: Dict[str, Tuple[float, float]] = {
    # 新生代 (Cenozoic)
    'Quaternary': (0.0, 2.58),
    'Neogene': (2.58, 23.03),
    'Paleogene': (23.03, 66.0),
    
    # 中生代 (Mesozoic)
    'Upper_Cretaceous': (66.0, 100.5),
    'Lower_Cretaceous': (100.5, 145.0),
    'Upper_Jurassic': (145.0, 163.5),
    'Middle_Jurassic': (163.5, 174.1),
    'Lower_Jurassic': (174.1, 201.3),
    'Upper_Triassic': (201.3, 237.0),
    'Middle_Triassic': (237.0, 247.2),
    'Lower_Triassic': (247.2, 251.9),
    
    # 古生代 (Paleozoic)
    'Lopingian': (251.9, 259.1),
    'Guadalupian': (259.1, 265.1),
    'Cisuralian': (265.1, 298.9),
    'Pennsylvanian': (298.9, 323.2),
    'Mississippian': (323.2, 358.9),
    'Late_Devonian': (358.9, 382.7),
    'Middle_Devonian': (382.7, 393.3),
    'Early_Devonian': (393.3, 419.2),
    'Pridoli': (419.2, 423.0),
    'Ludlow': (423.0, 427.4),
    'Wenlock': (427.4, 433.4),
    'Llandovery': (433.4, 443.8),
    'Late_Ordovician': (443.8, 458.4),
    'Middle_Ordovician': (458.4, 470.0),
    'Early_Ordovician': (470.0, 485.4),
    'Furongian': (485.4, 497.0),
    'Miaolingian': (497.0, 509.0),
    'Series_2': (509.0, 521.0),
    'Terreneuvian': (521.0, 541.0),
}

# 中文映射 (可选)
GEOLOGICAL_PERIODS_CN: Dict[str, str] = {
    'Quaternary': '第四纪',
    'Neogene': '新近纪',
    'Paleogene': '古近纪',
    'Upper_Cretaceous': '晚白垩世',
    'Lower_Cretaceous': '早白垩世',
    'Upper_Jurassic': '晚侏罗世',
    'Middle_Jurassic': '中侏罗世',
    'Lower_Jurassic': '早侏罗世',
    'Upper_Triassic': '晚三叠世',
    'Middle_Triassic': '中三叠世',
    'Lower_Triassic': '早三叠世',
}

# 便捷的分组
GEOLOGICAL_PERIODS: Dict[str, List[str]] = {
    'Cenozoic': ['Quaternary', 'Neogene', 'Paleogene'],
    'Mesozoic': [
        'Upper_Cretaceous', 'Lower_Cretaceous',
        'Upper_Jurassic', 'Middle_Jurassic', 'Lower_Jurassic',
        'Upper_Triassic', 'Middle_Triassic', 'Lower_Triassic'
    ],
    'Triassic': ['Upper_Triassic', 'Middle_Triassic', 'Lower_Triassic'],
    'Jurassic': ['Upper_Jurassic', 'Middle_Jurassic', 'Lower_Jurassic'],
    'Cretaceous': ['Upper_Cretaceous', 'Lower_Cretaceous'],
}


def assign_geological_period(
    age: float,
    period_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[str]:
    """
    根据年龄确定地质时代

    Args:
        age: 年龄值 (Ma)
        period_boundaries: 时代边界字典 (默认使用标准边界)

    Returns:
        地质时代名称，如果不匹配则返回None
    """
    if period_boundaries is None:
        period_boundaries = GEOLOGICAL_PERIODS_BOUNDARIES
    
    if pd.isna(age):
        return None
    
    for period_name, (lower, upper) in period_boundaries.items():
        if lower <= age < upper:
            return period_name
    
    return None


def filter_by_period(
    df: pd.DataFrame,
    age_column: str,
    period: str,
    period_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    按地质时代过滤数据

    Args:
        df: 输入DataFrame
        age_column: 年龄列名
        period: 地质时代名称
        period_boundaries: 时代边界字典

    Returns:
        过滤后的DataFrame
    """
    if period_boundaries is None:
        period_boundaries = GEOLOGICAL_PERIODS_BOUNDARIES
    
    if period not in period_boundaries:
        raise ValueError(f"未知地质时代: {period}。可用: {list(period_boundaries.keys())}")
    
    lower, upper = period_boundaries[period]
    mask = (df[age_column] >= lower) & (df[age_column] < upper)
    
    return df[mask].copy()


def get_period_ages(
    df: pd.DataFrame,
    sample_id_column: str = 'Published_Sample_ID',
    age_column: str = 'Best Age',
    deposit_age_column: str = 'Est. Depos. Age (Ma)',
    period_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    按地质时代组织年龄数据

    Args:
        df: 输入DataFrame
        sample_id_column: 样品ID列名
        age_column: 年龄列名
        deposit_age_column: 沉积年龄列名
        period_boundaries: 时代边界字典

    Returns:
        嵌套字典: {地质时代: {样品ID: [年龄列表]}}
    """
    if period_boundaries is None:
        period_boundaries = GEOLOGICAL_PERIODS_BOUNDARIES
    
    result: Dict[str, Dict[str, List[float]]] = {
        period: {} for period in period_boundaries.keys()
    }
    
    # 按样品ID分组
    grouped = df.groupby(sample_id_column)
    
    for sample_id, group in grouped:
        # 获取沉积年龄
        deposit_ages = pd.to_numeric(group[deposit_age_column], errors='coerce').dropna()
        if len(deposit_ages) == 0:
            continue
        
        deposit_age = deposit_ages.iloc[0]
        period = assign_geological_period(deposit_age, period_boundaries)
        
        if period is None:
            continue
        
        # 获取该样品的年龄列表
        ages = pd.to_numeric(group[age_column], errors='coerce').dropna().tolist()
        
        if ages:
            result[period][sample_id] = ages
    
    # 移除空的地质时代
    result = {k: v for k, v in result.items() if v}
    
    return result


def add_geological_period_column(
    df: pd.DataFrame,
    age_column: str,
    output_column: str = 'Geological_Period',
    period_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    添加地质时代列

    Args:
        df: 输入DataFrame
        age_column: 年龄列名
        output_column: 输出列名
        period_boundaries: 时代边界字典

    Returns:
        添加了地质时代列的DataFrame
    """
    df = df.copy()
    df[output_column] = df[age_column].apply(
        lambda x: assign_geological_period(x, period_boundaries)
    )
    return df


def get_period_boundary(
    period: str,
    period_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    获取地质时代边界

    Args:
        period: 地质时代名称
        period_boundaries: 时代边界字典

    Returns:
        (下限, 上限) 元组
    """
    if period_boundaries is None:
        period_boundaries = GEOLOGICAL_PERIODS_BOUNDARIES
    
    if period not in period_boundaries:
        raise ValueError(f"未知地质时代: {period}")
    
    return period_boundaries[period]


def list_periods(
    era: Optional[str] = None,
    period_boundaries: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[str]:
    """
    列出可用的地质时代

    Args:
        era: 代名称 (如 'Mesozoic', 'Cenozoic')
        period_boundaries: 时代边界字典

    Returns:
        地质时代列表
    """
    if period_boundaries is None:
        period_boundaries = GEOLOGICAL_PERIODS_BOUNDARIES
    
    if era is None:
        return list(period_boundaries.keys())
    
    if era in GEOLOGICAL_PERIODS:
        return GEOLOGICAL_PERIODS[era]
    
    raise ValueError(f"未知代: {era}。可用: {list(GEOLOGICAL_PERIODS.keys())}")
