# 迁移指南

本文档帮助你将旧代码迁移到新的 `geotime` 包。

## 快速对照表

### time_group.ipynb → geotime.pipeline

**原notebook代码 (复制粘贴52次):**
```python
# 每个地质时代都要重复这些代码
freq_df = calculate_age_frequencies(ages, bin_size=1)
smooth_age, smooth_freq = smooth_LOWESS(freq_df['age'], freq_df['frequency'])

# STFT
f, t, Zxx = signal.stft(smooth_freq, fs=1.0, nperseg=256)

# Lomb-Scargle
frequency = np.linspace(0.001, 0.5, 1000)
power = LombScargle(smooth_age, smooth_freq).power(frequency)

# 绘图
plt.figure(figsize=(15, 10))
...
plt.savefig(f'{period}.png')
```

**新代码 (只需一次):**
```python
from geotime import quick_analyze

pipeline = quick_analyze(
    data_file="./data/Qiagtang_large.xlsx",
    output_dir="./results",
    periods=['Lower_Triassic', 'Middle_Triassic', 'Upper_Triassic',
             'Lower_Jurassic', 'Middle_Jurassic', 'Upper_Jurassic',
             'Lower_Cretaceous', 'Upper_Cretaceous'],
    bin_size=1.0,
    smooth_frac=0.1,
)
```

### 数据加载

**旧代码:**
```python
from dataloader import DataLoader

df = DataLoader('./data/file.xlsx', time_filter=100, select_col='Value')
```

**新代码:**
```python
from geotime.dataloader import load_data

df = load_data('./data/file.xlsx', time_filter=100, select_columns=['Age(Ma)', 'Value'])
```

### SSA分析

**旧代码:**
```python
from detrend_utils.ssa import SSA

ssa = SSA(series, N, K)
X = ssa.trajectory_matrix(series, K)
X = X.reshape(X.shape[0], -1)
components = ssa.decompose(X)
reconstructed = ssa.reconstruct(np.array(components), 4)
```

**新代码:**
```python
from geotime.detrend import SSA

ssa = SSA(series, window_size=K)
trajectory = ssa.build_trajectory_matrix()
components = ssa.decompose(trajectory)
reconstructed = ssa.reconstruct(components, n_components=4)
```

### 频谱分析

**旧代码:**
```python
from detrend_utils.spectrum import Filter

filter = Filter(series)
freq, psd = filter.Mtm(nperseg=128)
peaks = filter.find_peaks(psd)
```

**新代码:**
```python
from geotime.detrend import Filter

analyzer = Filter(series)
freq, psd = analyzer.mtm_power_spectrum(nperseg=128)
peak_indices, peak_heights = analyzer.find_peaks(psd, n_peaks=3)
```

### 绘图

**旧代码:**
```python
from plot_utils.signal_compare import plot_line as line

line(age, original, detrended, title='Comparison', xlimit=None, ylimit=None)
```

**新代码:**
```python
from geotime.plotting import plot_comparison

fig, ax = plt.subplots()
plot_comparison(age, original, detrended, title='Comparison', ax=ax)
```

### 地质时代处理 (新增)

```python
from geotime.geological import assign_geological_period, get_period_ages

# 判断年龄所属地质时代
period = assign_geological_period(250)  # -> 'Lower_Triassic'

# 从数据中提取各时代年龄
period_dict = get_period_ages(df, 'Sample_ID', 'Best_Age', 'Est_Depos_Age')
# -> {'Lower_Triassic': {'Sample001': [249.5, 251.2, ...], ...}, ...}
```

### 时频分析 (新增)

```python
from geotime.timeseries import TimeFrequencyAnalyzer

analyzer = TimeFrequencyAnalyzer(age, frequency)

# STFT
stft_result = analyzer.stft(nperseg=256)
stft_result.plot()

# Lomb-Scargle
lomb_result = analyzer.lomb_scargle()
peaks = lomb_result.get_peaks(n_peaks=3)
print(peaks['periods'])  # 主要周期

# 绘制摘要图
analyzer.plot_summary()
```

### 平滑处理 (新增)

```python
from geotime.smoothing import SmoothingPipeline

# 链式调用
pipeline = (SmoothingPipeline(x, y)
    .lowess(frac=0.1)
    .gaussian(sigma=2.0)
    .interpolate(dx=1.0))

x_smooth, y_smooth = pipeline.get_result()

# 或单独调用
from geotime.smoothing import lowess_smooth
x_smooth, y_smooth = lowess_smooth(x, y, frac=0.1)
```

## 新功能：分析工作流

### 碎屑锆石分析 (DetritalAnalysisPipeline)

```python
from geotime import DetritalAnalysisPipeline

analysis = (DetritalAnalysisPipeline()
    .load_data('./data/Qiagtang_large.xlsx')
    .extract_period_ages(periods=['Lower_Triassic', 'Middle_Triassic'])
    .process_period('Lower_Triassic', smooth_frac=0.1)
    .process_period('Middle_Triassic', smooth_frac=0.1))

# 获取结果
result = analysis.results['Lower_Triassic']
print(f"颗粒数: {result.n_grains}")
print(f"主要周期: {result.lomb_scargle_result.get_peaks()['periods']}")

# 绘图
analysis.plot_period_comparison()
analysis.plot_stft_grid()

# 保存
analysis.save_results('./results')
```

### 通用时间序列分析 (GeoTimeSeriesAnalysis)

```python
from geotime.analysis import GeoTimeSeriesAnalysis

analysis = (GeoTimeSeriesAnalysis()
    .load_data("./data/file.xlsx", age_column="Age(Ma)", value_column="CAR")
    .apply_weighted_average()
    .apply_ssa(window_size=200, n_components=4))

result = analysis.analyze_spectrum(method="mtm")
```

## 向后兼容

旧的代码文件仍然保留在原位置，你可以逐步迁移：

- `dataloader.py` → `geotime/dataloader.py`
- `detrend_utils/*.py` → `geotime/detrend/*.py`
- `plot_utils/*.py` → `geotime/plotting/*.py`
- `time_group.ipynb` → `geotime/pipeline.py`

建议按照以下顺序迁移：
1. 新分析使用 `geotime` 包
2. 逐步替换旧脚本
3. 最终删除旧文件

## 主要变化

1. **函数命名**: 更清晰的英文命名（如 `Mtm` → `mtm_power_spectrum`）
2. **参数命名**: 更明确的参数名（如 `n_components` 替代数字）
3. **返回值**: 更一致的返回类型（元组或字典）
4. **错误处理**: 添加了适当的异常检查
5. **文档**: 所有公共函数都有文档字符串
6. **类型注解**: 添加了Python类型提示
