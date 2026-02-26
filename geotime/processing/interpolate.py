"""
数据插值处理模块
"""

import pandas as pd
import numpy as np
from typing import Union


def interpolate_data(
    df: pd.DataFrame,
    age_column: str = "Age",
    value_column: str = "Thickness",
    interval: float = 1.0,
) -> pd.DataFrame:
    """
    对数据进行线性插值，使年龄间隔均匀

    Args:
        df: 包含年龄和数值列的DataFrame
        age_column: 年龄列名
        value_column: 数值列名
        interval: 插值间隔 (默认1百万年)

    Returns:
        插值后的DataFrame
    """
    # 确保数据类型正确
    df = df.copy()
    df[age_column] = pd.to_numeric(df[age_column], errors='coerce')
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
    
    # 删除NaN值并排序
    df = df.dropna(subset=[age_column, value_column])
    df = df.sort_values(age_column)
    
    if len(df) < 2:
        raise ValueError("至少需要2个有效数据点进行插值")
    
    # 使用pandas的插值功能
    min_age = df[age_column].min()
    max_age = df[age_column].max()
    
    # 创建均匀间隔的年龄点
    new_ages = np.arange(min_age, max_age + interval, interval)
    
    # 使用numpy的interp进行线性插值
    interpolated_values = np.interp(
        new_ages,
        df[age_column].values,
        df[value_column].values,
    )
    
    return pd.DataFrame({
        age_column: new_ages,
        value_column: interpolated_values,
    })
