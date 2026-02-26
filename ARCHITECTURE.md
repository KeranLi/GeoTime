# GeoTime 架构说明

## 模块分层

```
┌─────────────────────────────────────────────────────────────────┐
│                         应用层 (Applications)                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ examples/       │  │ notebooks/      │  │ scripts/        │  │
│  │ 示例脚本         │  │ Jupyter notebooks│  │ 批量处理脚本     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       工作流层 (Workflows)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ pipeline.py     │  │ analysis.py     │  │ metaspectral/   │  │
│  │ 碎屑锆石分析管道 │  │ 通用时间序列分析 │  │ 元频谱分析      │  │
│  │ (STFT/Lomb)     │  │ (SSA/Filter)    │  │ (信号重建+SSA   │  │
│  │                 │  │                 │  │  +贡献率计算)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       处理层 (Processing)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ grouping.py     │  │ smoothing.py    │  │ geological/     │  │
│  │ 数据分组聚合     │  │ 平滑处理(LOWESS)│  │ 地质时代处理    │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  │
│  │ timeseries.py   │  │ processing/     │  │ dataloader.py   │  │
│  │ 时频分析(STFT   │  │ 数据预处理      │  │ 数据加载        │  │
│  │ /Lomb-Scargle)  │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        算法层 (Algorithms)                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ detrend/                                                    │ │
│  │  ├── wa.py          加权移动平均                             │ │
│  │  ├── ssa.py         奇异谱分析 (核心算法)                     │ │
│  │  ├── spectrum.py    频谱分析 (Filter类)                      │ │
│  │  └── linear.py      线性去趋势                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        可视化层 (Visualization)                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ plotting/                                                   │ │
│  │  ├── bar_plot.py          柱状图/折线图                      │ │
│  │  ├── signal_compare.py    信号对比图                         │ │
│  │  ├── psd_plot.py          功率谱图                          │ │
│  │  ├── traj_plot.py         轨迹矩阵图                        │ │
│  │  ├── wavelet_plot.py      小波分析图                        │ │
│  │  ├── timeseries_plots.py  时频分析图 ← NEW                  │ │
│  │  └── metaspectral_plots.py 元频谱分析图 ← NEW               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 模块依赖关系

### 核心依赖链
```
plotting (底层，无依赖)
    ↑
detrend (基础算法，依赖: numpy/scipy)
    ↑
processing, grouping, smoothing, timeseries (数据处理)
    ↑
geological (地质专用，依赖: grouping)
    ↑
pipeline, metaspectral (工作流，依赖: 以上所有)
    ↑
examples, notebooks (应用层)
```

### 各模块职责

| 模块 | 职责 | 主要类/函数 |
|------|------|------------|
| `detrend/wa.py` | 加权移动平均 | `WeightedAverage` |
| `detrend/ssa.py` | 奇异谱分析核心 | `SSA` |
| `detrend/spectrum.py` | 基础频谱分析 | `Filter` |
| `processing/interpolate.py` | 数据插值 | `interpolate_data` |
| `grouping.py` | 数据分组聚合 | `calculate_age_frequencies` |
| `smoothing.py` | 平滑处理 | `lowess_smooth`, `SmoothingPipeline` |
| `timeseries.py` | 时频分析 | `TimeFrequencyAnalyzer`, `compute_stft` |
| `geological/periods.py` | 地质时代 | `assign_geological_period` |
| `pipeline.py` | 碎屑锆石分析 | `DetritalAnalysisPipeline` |
| `analysis.py` | 通用时间序列 | `GeoTimeSeriesAnalysis` |
| `metaspectral/` | 元频谱分析 | `DetritalSpectrumAnalyzer` |
| `plotting/` | 可视化 | 各种 `plot_*` 函数 |

## 关键设计决策

### 1. 绘图分离
- **之前**: 各模块类自带 `plot()` 方法（分散在各处）
- **之后**: 所有绘图逻辑集中在 `plotting/` 目录
- **好处**: 统一风格、便于维护、可复用

### 2. 工作流分层
```
元频谱分析 (metaspectral/) - 你的核心需求
    └─ 离散 → 连续 → SSA分解 → 贡献率计算

碎屑锆石分析 (pipeline.py) - 传统时频分析
    └─ 分组 → 平滑 → STFT/Lomb-Scargle

通用分析 (analysis.py) - 单时间序列
    └─ 加载 → SSA → 频谱分析
```

### 3. 避免重复
- `timeseries.py` 和 `detrend/spectrum.py` 有互补而非重复
  - `detrend/spectrum.py`: 基础Filter类（静态方法）
  - `timeseries.py`: 高级时频分析（整合STFT+Lomb+PSD）

- `pipeline.py` 和 `metaspectral/analyzer.py` 是不同层次
  - `pipeline.py`: 时频分析（关注时间-频率分布）
  - `metaspectral/analyzer.py`: 元频谱分析（关注周期成分贡献率）

## 使用场景选择

| 场景 | 推荐模块 | 原因 |
|------|---------|------|
| 快速分析单个样品 | `quick_spectrum_analysis()` | 一行代码 |
| 批量处理多时代 | `DetritalSpectrumAnalyzer` | 批量+对比 |
| 传统STFT分析 | `DetritalAnalysisPipeline` | 成熟流程 |
| 研究周期贡献率 | `metaspectral/` | 你的核心需求 |
| 自定义算法 | `detrend/ssa.py` | 底层接口 |

## 扩展指南

### 添加新的分解方法
1. 在 `metaspectral/decomposition.py` 添加新方法
2. 更新 `DecompositionMethod` 枚举
3. 在 `DecompositionMethod` 添加 case

### 添加新的绘图类型
1. 在 `plotting/` 添加新模块
2. 更新 `plotting/__init__.py`
3. 在对应结果类添加 `plot()` 方法调用新绘图函数

### 添加新的地质信号范围
1. 在 `metaspectral/contribution.py` 的 `SIGNAL_RANGES` 添加
2. 自动生效，无需修改其他代码
