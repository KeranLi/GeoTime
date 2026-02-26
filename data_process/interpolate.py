import pandas as pd
import numpy as np

def interpolate_data(df, interval=1):
    """
    对给定的DataFrame进行线性插值，使Age列的时间间隔均匀。

    参数:
    df (pd.DataFrame): 包含'Age'和'Thickness'列的DataFrame。
    interval (float): 年龄间隔，数值1代表1百万年。

    返回:
    pd.DataFrame: 插值后的DataFrame。
    """
    # 确保'Age'列为数值类型
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # 对'Age'进行升序排序
    df = df.sort_values('Age')
    
    # 获取最小和最大年龄
    min_age = df['Age'].min()
    max_age = df['Age'].max()
    
    # 创建一个均匀间隔的年龄数组
    new_ages = np.arange(min_age, max_age + interval, interval)
    
    # 初始化插值后的DataFrame
    # 使用列表收集插值后的行
    interpolated_data = []
    
    # 上一个厚度值
    prev_thickness = df.loc[df['Age'] == min_age, 'Thickness'].values[0]
    
    # 对每个年龄点进行插值
    for age in new_ages:
        if age in df['Age'].values:
            # 如果当前年龄在原始数据中，直接使用对应的厚度值
            thickness = df.loc[df['Age'] == age, 'Thickness'].values[0]
        else:
            # 如果当前年龄不在原始数据中，使用线性插值确定厚度
            mask_below = df['Age'] < age
            mask_equal = df['Age'] == age
            closest_above = df.loc[mask_below, 'Age'].max()
            closest_below = df.loc[~mask_below & ~mask_equal, 'Age'].min()
            
            thickness = (age - closest_below) * (df.loc[df['Age'] == closest_above, 'Thickness'].values[0] - prev_thickness) / (closest_above - closest_below) + prev_thickness
        
        # 将插值结果添加到列表中
        interpolated_data.append({'Age': age, 'Thickness': thickness})
        
        # 更新prev_thickness为最新的厚度值
        if age in df['Age'].values:
            prev_thickness = thickness
    
    # 将收集的插值数据转换为DataFrame
    interpolated_df = pd.DataFrame(interpolated_data)
    
    return interpolated_df