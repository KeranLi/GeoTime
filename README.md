# GeoTime - 地质时间序列分析工具包

用于处理和分析地质年代数据的时间序列分析工具。

## 功能特性

- **数据加载**: 支持Excel格式地质数据
- **地质时代处理**: 自动映射和分组
- **碎屑锆石分析**: 完整的分析流程 (原time_group.ipynb)
- **元频谱分析** ⭐ **NEW**: 对年龄频率分布进行频谱分解，计算各周期成分贡献率
- **去趋势分析**: 加权移动平均、SSA (奇异谱分析)
- **频谱分析**: 2π-MTM、Lomb-Scargle周期图、STFT
- **滤波处理**: 高斯滤波、Butterworth低通滤波、LOWESS平滑
- **小波分析**: 连续小波变换 (CWT)
- **可视化**: 多种专业图表绘制

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 碎屑锆石分析 (原time_group.ipynb)

```python
from geotime import quick_analyze

# 一行代码完成原notebook的全部功能
pipeline = quick_analyze(
    data_file="./data/Qiagtang_large.xlsx",
    output_dir="./results/detrital_analysis",
    periods=['Lower_Triassic', 'Middle_Triassic', 'Upper_Triassic',
             'Lower_Jurassic', 'Middle_Jurassic', 'Upper_Jurassic',
             'Lower_Cretaceous', 'Upper_Cretaceous'],
    bin_size=1.0,
    smooth_frac=0.1,
)

# 查看结果
for period, result in pipeline.results.items():
    print(f"{period}: {result.n_grains} grains")
    if result.lomb_scargle_result:
        peaks = result.lomb_scargle_result.get_peaks(n_peaks=3)
        print(f"  周期: {peaks['periods']}")
```

### 2. 时间序列分析 (原CCD.py / global_detrital.py)

```python
from geotime.analysis import run_standard_analysis

# 运行完整分析流程
analysis = run_standard_analysis(
    data_file="./data/equator_pacific.xlsx",
    output_dir="./results/my_analysis",
    age_column="Age(Ma)",
    value_column="CAR(mg/cm2/kyr)",
)

# 查看检测到的周期
print(analysis.spectrum_result['peak_periods'])

# 绘制频谱图
fig = analysis.plot_spectrum()
fig.savefig("./results/spectrum.png")
```

### 3. 分步分析

```python
from geotime.analysis import GeoTimeSeriesAnalysis

# 创建分析器
analysis = GeoTimeSeriesAnalysis()

# 链式调用
(analysis
    .load_data("./data/equator_pacific.xlsx", 
               age_column="Age(Ma)", 
               value_column="CAR(mg/cm2/kyr)")
    .apply_weighted_average()
    .apply_ssa(window_size=200, n_components=4))

# 频谱分析
result = analysis.analyze_spectrum(method="mtm")
print(f"主要周期: {result['peak_periods']} Ma")
```

### 4. 元频谱分析 (Meta-Spectral Analysis) ⭐

对碎屑锆石年龄频率分布进行频谱分解，计算各周期成分的贡献率：

```python
from geotime import DetritalSpectrumAnalyzer, SIGNAL_RANGES

# 创建分析器
analyzer = DetritalSpectrumAnalyzer()

# 分析 (自动完成: 离散→连续→分解→贡献率计算)
result = analyzer.analyze_period(
    ages=detrital_ages,  # 碎屑锆石年龄数组
    period_name="Lower_Triassic",
    bin_size=2.0,  # Ma
)

# 查看各周期成分贡献率
print(result.contribution_df)
# → DataFrame: component | period_Ma | contribution_pct

# 查看各地质信号范围贡献率
print(result.range_contribution_df)
# → supercontinent: 15.2%, wilson_cycle: 23.5%, orogenic: 8.3%, ...

# 提取特定地质信号
supercontinent_signal = result.get_signal('supercontinent')
print(f"超大陆周期贡献: {supercontinent_signal.total_contribution:.1%}")

# 绘制完整分析图
result.plot_full_analysis()
```

### 5. 独立模块使用

```python
import numpy as np
import pandas as pd
from geotime.detrend import SSA, Filter
from geotime.geological import assign_geological_period
from geotime.timeseries import TimeFrequencyAnalyzer

# 地质时代判断
period = assign_geological_period(250)  # -> 'Lower_Triassic'

# SSA分析
ssa = SSA(values, window_size=100)
trajectory = ssa.build_trajectory_matrix()
components = ssa.decompose(trajectory)
reconstructed = ssa.reconstruct(components, n_components=4)

# 时频分析
analyzer = TimeFrequencyAnalyzer(age, frequency)
stft_result = analyzer.stft(nperseg=256)
lomb_result = analyzer.lomb_scargle()
peaks = lomb_result.get_peaks(n_peaks=3)
print(f"周期: {peaks['periods']} Ma")
```

## 项目结构

```
geotime/
├── __init__.py
├── dataloader.py          # 数据加载
├── analysis.py            # 通用分析工作流
├── pipeline.py            # 碎屑锆石分析管道
├── grouping.py            # 数据分组聚合
├── smoothing.py           # 平滑处理
├── timeseries.py          # 时频分析
├── geological/            # 地质时代处理
│   ├── __init__.py
│   └── periods.py
├── metaspectral/          # 元频谱分析 ⭐ NEW
│   ├── __init__.py
│   ├── reconstruction.py  # 离散→连续信号重建
│   ├── decomposition.py   # 频谱分解 (SSA/EMD/FFT)
│   ├── contribution.py    # 贡献率计算
│   └── analyzer.py        # 主分析器
├── detrend/               # 去趋势模块
│   ├── wa.py             # 加权平均
│   ├── ssa.py            # 奇异谱分析
│   ├── spectrum.py       # 频谱分析
│   └── linear.py         # 线性去趋势
├── plotting/              # 绘图模块
│   ├── bar_plot.py
│   ├── signal_compare.py
│   ├── psd_plot.py
│   ├── traj_plot.py
│   ├── wavelet_plot.py
│   └── multiple_psd_plot.py
└── processing/            # 数据处理
    ├── interpolate.py
    └── wavelet.py

examples/
├── basic_analysis.py          # 基础分析示例
├── detrital_analysis.py       # 碎屑数据示例
├── time_group_analysis.py     # time_group.ipynb重构
├── time_group_simple.py       # 最简使用示例
└── metaspectral_analysis.py   # 元频谱分析示例 ⭐ NEW
```

## 旧代码迁移

原代码仍保留在根目录，重构后的代码在 `geotime/` 目录下：

| 旧文件 | 新位置 |
|--------|--------|
| `dataloader.py` | `geotime/dataloader.py` |
| `detrend_utils/*.py` | `geotime/detrend/*.py` |
| `plot_utils/*.py` | `geotime/plotting/*.py` |
| `CCD.py` | `geotime/analysis.py` |
| `time_group.ipynb` | `geotime/pipeline.py` + `examples/time_group_analysis.py` |

### time_group.ipynb 迁移对照

原notebook中重复52次的代码块，现在只需：

```python
# 原代码 (复制粘贴52次，每次改地质时代名称)
# Cell for Middle_Triassic
freq_df = calculate_frequency(...)
smooth_age, smooth_freq = smooth_LOWESS(freq_df)
stft_result = compute_stft(smooth_freq)
lomb_result = lomb_scargle(smooth_age, smooth_freq)
plot_everything(...)

# 新代码 (一次搞定)
from geotime import quick_analyze
quick_analyze(
    data_file="./data/Qiagtang_large.xlsx",
    output_dir="./results",
    periods=['Lower_Triassic', 'Middle_Triassic', ...],  # 一次指定所有时代
)
```

## 主要改进

1. **消除重复代码**: 将time_group.ipynb中复制粘贴52次的代码抽象为可复用函数
2. **统一地质时代管理**: 内置ICS标准地质时代边界定义
3. **管道化设计**: `DetritalAnalysisPipeline`整合完整分析流程
4. **一键分析**: `quick_analyze()`函数一行代码完成全部分析
5. **灵活配置**: 支持自定义参数和分步执行
6. **自动保存**: 自动保存CSV数据和可视化图片

### 元频谱分析 (Meta-Spectral Analysis) 核心概念

**问题**: 传统的碎屑锆石分析只给出年龄分布直方图，难以量化各地质周期（如超大陆周期、Wilson旋回）的贡献。

**解决方案**:

```
离散年龄数据 → 频率分布 → 连续信号重建 → 频谱分解 → 贡献率计算
     ↑                                                        ↓
  原始数据                                           各周期成分能量占比
```

**输出示例**:

| 信号范围 | 周期范围 (Ma) | 贡献率 |
|---------|--------------|--------|
| 超大陆周期 | 300-500 | 15.2% |
| Wilson旋回 | 200-300 | 23.5% |
| 造山周期 | 50-100 | 8.3% |
| ... | ... | ... |

**应用场景**:
- 量化不同构造周期对沉积盆地的贡献
- 对比不同地质时代/区域的周期特征
- 提取特定的地质信号进行进一步分析

## 作者

Keran Li (@Keranli98)
