"""
数据加载模块

用于加载和预处理地质数据文件 (Excel格式)
"""

import argparse
from pathlib import Path
from typing import Optional, Union, List

import pandas as pd


def load_data(
    file_path: Union[str, Path],
    age_column: str = "Age(Ma)",
    time_filter: Optional[float] = None,
    select_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载Excel数据文件

    Args:
        file_path: 数据文件路径
        age_column: 年龄列的名称
        time_filter: 最大年龄过滤值 (可选)
        select_columns: 要保留的列名列表 (可选)

    Returns:
        处理后的DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 列名不存在或数据格式错误
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    df = pd.read_excel(file_path)
    
    # 年龄过滤
    if time_filter is not None:
        if age_column not in df.columns:
            raise ValueError(f"年龄列 '{age_column}' 不存在，可用列: {list(df.columns)}")
        df = df[df[age_column] < time_filter].copy()
    
    # 列选择
    if select_columns:
        missing_cols = [col for col in select_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"列不存在: {missing_cols}")
        df = df[select_columns].copy()
    
    return df


def load_frequency_data(
    file_path: Union[str, Path],
    age_column: str = "Published 206Pb/238U age (Ma)",
) -> tuple[pd.Series, pd.Series]:
    """
    加载并计算年龄频率统计

    Args:
        file_path: 数据文件路径
        age_column: 年龄列名称

    Returns:
        (年龄序列, 频率序列)
    """
    df = pd.read_excel(file_path)
    
    ages = df[age_column].dropna()
    freq = ages.value_counts().sort_index()
    
    return freq.index, freq.values


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="地质数据加载工具")
    parser.add_argument("-F", "--input", dest="input_file", required=True,
                        help="输入Excel文件路径")
    parser.add_argument("-C", "--col", dest="select_columns", nargs="+",
                        help="要选择的列名")
    parser.add_argument("-A", "--age", dest="time_filter", type=float,
                        help="最大年龄过滤值")
    parser.add_argument("--age-col", default="Age(Ma)",
                        help="年龄列名称 (默认: Age(Ma))")
    
    args = parser.parse_args()
    
    df = load_data(
        args.input_file,
        age_column=args.age_col,
        time_filter=args.time_filter,
        select_columns=args.select_columns,
    )
    
    print(f"成功加载数据: {len(df)} 行")
    print(df.head())


if __name__ == "__main__":
    main()
