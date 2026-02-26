import numpy as np

def detrend_and_normalize(dat, t, t0):
    """
    去除数据的趋势并进行标准化处理。

    参数:
    dat (array-like): 需要去趋势化的数据点。
    t (array-like): 对应于每个数据点的时间或位置。
    t0 (scalar): 用作基线的时间或位置偏移量。

    返回:
    tuple: 包含去趋势化数据、标准差、方差和归一化数据的元组。
    """
    # 计算一阶多项式拟合
    p = np.polyfit(t - t0, dat, 1)
    
    # 去趋势化数据
    dat_notrend = dat - np.polyval(p, t - t0)
    dat_notrend = np.nan_to_num(dat_notrend, nan=0.0)

    # 计算标准差
    std = dat_notrend.std()
    
    # 计算方差
    var = std ** 2
    
    # 归一化数据集
    dat_norm = dat_notrend / std
    dat_norm = np.nan_to_num(dat_norm, nan=0.0)
    
    
    return dat_notrend, std, var, dat_norm