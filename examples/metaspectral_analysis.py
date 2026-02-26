"""
元频谱分析示例

展示如何对碎屑锆石年龄频率分布进行频谱分解和贡献率计算

核心流程:
1. 离散频率分布 → 连续信号重建
2. 频谱分解 (SSA/EMD/FFT) → 各周期成分
3. 贡献率计算 → 各周期占总信号的能量比例
4. 特定信号提取 → 如超大陆周期、Wilson旋回等
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geotime import (
    DetritalSpectrumAnalyzer,
    quick_spectrum_analysis,
    InterpolationMethod,
    DecompositionMethod,
    SIGNAL_RANGES,
)
from geotime.pipeline import DetritalAnalysisPipeline


def example_basic_analysis():
    """基础分析示例"""
    print("=" * 60)
    print("示例1: 基础频谱分析")
    print("=" * 60)
    
    # 创建分析器
    analyzer = DetritalSpectrumAnalyzer(
        reconstruction_method=InterpolationMethod.CUBIC,
        decomposition_method=DecompositionMethod.SSA,
    )
    
    # 模拟一些年龄数据 (实际使用时从文件读取)
    np.random.seed(42)
    # 生成包含多个周期成分的模拟数据
    ages = np.random.normal(loc=250, scale=50, size=1000)  # 中心250Ma
    ages = np.concatenate([
        ages,
        np.random.normal(loc=400, scale=30, size=500),  # 400Ma峰值
        np.random.normal(loc=100, scale=20, size=300),  # 100Ma峰值
    ])
    ages = ages[(ages > 0) & (ages < 600)]  # 过滤有效范围
    
    # 分析
    result = analyzer.analyze_period(
        ages=ages,
        period_name="Example_Sample",
        bin_size=5.0,  # 5 Ma 分箱
        reconstruction_params={
            'n_points': 500,  # 连续信号500个点
            'smooth_sigma': 1.0,  # 轻微平滑
        },
        decomposition_params={
            'window_size': 50,
            'n_components': 10,
        },
    )
    
    # 查看贡献率
    print("\n各成分贡献率:")
    print(result.contribution_df.head(10).to_string(index=False))
    
    print("\n各地质信号范围贡献率:")
    print(result.range_contribution_df[['signal_range', 'description', 'total_contribution_pct']].to_string(index=False))
    
    # 提取特定信号
    print("\n提取超大陆周期信号 (300-500 Ma):")
    supercontinent_signal = result.get_signal('supercontinent')
    print(f"  贡献率: {supercontinent_signal.total_contribution:.2%}")
    print(f"  主导周期: {supercontinent_signal.dominant_period:.1f} Ma")
    
    # 绘制完整分析图
    fig = result.plot_full_analysis()
    plt.savefig('./results/metaspectral_example1.png', dpi=150, bbox_inches='tight')
    print("\n分析图已保存: ./results/metaspectral_example1.png")
    plt.close()


def example_real_data():
    """真实数据分析示例"""
    print("\n" + "=" * 60)
    print("示例2: 真实碎屑锆石数据分析")
    print("=" * 60)
    
    data_file = Path("./data/Qiagtang_large.xlsx")
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        print("使用模拟数据代替...")
        
        # 使用模拟数据
        np.random.seed(42)
        ages_dict = {
            'Lower_Triassic': np.concatenate([
                np.random.normal(250, 30, 200),
                np.random.normal(400, 40, 150),
                np.random.normal(180, 25, 100),
            ]),
            'Middle_Triassic': np.concatenate([
                np.random.normal(240, 35, 180),
                np.random.normal(420, 35, 120),
                np.random.normal(160, 20, 80),
            ]),
        }
    else:
        # 使用真实数据
        print("加载真实数据...")
        pipeline = DetritalAnalysisPipeline()
        pipeline.load_data(data_file)
        pipeline.extract_period_ages(periods=['Lower_Triassic', 'Middle_Triassic'])
        
        ages_dict = {}
        for period, samples in pipeline.period_ages.items():
            all_ages = []
            for sample_ages in samples.values():
                all_ages.extend(sample_ages)
            ages_dict[period] = np.array(all_ages)
    
    # 创建分析器
    analyzer = DetritalSpectrumAnalyzer(
        reconstruction_method=InterpolationMethod.CUBIC,
        decomposition_method=DecompositionMethod.SSA,
    )
    
    # 分析每个地质时代
    for period_name, ages in ages_dict.items():
        if len(ages) < 50:
            continue
        
        print(f"\n分析 {period_name}...")
        result = analyzer.analyze_period(
            ages=ages,
            period_name=period_name,
            bin_size=2.0,
            reconstruction_params={'smooth_sigma': 0.5},
            decomposition_params={'window_size': 40, 'n_components': 8},
        )
        
        # 打印关键结果
        print(f"  主导周期: {result.dominant_periods.iloc[0]['period_Ma']:.1f} Ma "
              f"({result.dominant_periods.iloc[0]['contribution_pct']:.1f}%)")
        
        # 计算特定范围的贡献
        for range_name in ['wilson_cycle', 'plate_tectonic', 'orogenic']:
            signal = result.get_signal(range_name)
            print(f"  {range_name}: {signal.total_contribution:.1%}")
    
    # 比较不同时代的Wilson旋回贡献
    print("\n" + "-" * 40)
    print("Wilson旋回贡献率对比:")
    comparison = analyzer.compare_periods(
        period_names=list(ages_dict.keys()),
        signal_range='wilson_cycle',
    )
    print(comparison.to_string(index=False))
    
    # 绘制对比图
    fig = analyzer.plot_period_comparison(
        period_names=list(ages_dict.keys()),
        signal_range='wilson_cycle',
    )
    plt.savefig('./results/metaspectral_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存: ./results/metaspectral_comparison.png")
    plt.close()


def example_advanced_decomposition():
    """高级分解方法对比"""
    print("\n" + "=" * 60)
    print("示例3: 不同分解方法对比")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    ages = np.concatenate([
        np.random.normal(200, 40, 500),
        np.random.normal(350, 30, 300),
        np.random.normal(100, 20, 200),
    ])
    ages = ages[(ages > 0) & (ages < 500)]
    
    # 计算频率分布
    from geotime.grouping import calculate_age_frequencies
    freq_df = calculate_age_frequencies(ages, bin_size=5.0)
    
    methods = [
        (DecompositionMethod.SSA, {'window_size': 30, 'n_components': 8}),
        (DecompositionMethod.FFT_BANDS, {}),
    ]
    
    results = {}
    
    for method, params in methods:
        print(f"\n使用 {method.value} 分解...")
        
        analyzer = DetritalSpectrumAnalyzer(
            reconstruction_method=InterpolationMethod.CUBIC,
            decomposition_method=method,
        )
        
        result = analyzer.analyze_period(
            ages=freq_df['age'].values,
            frequencies=freq_df['frequency'].values,
            period_name=method.value,
            decomposition_params=params,
        )
        
        results[method.value] = result
        
        # 打印前3个主导周期
        print("  主导周期:")
        for _, row in result.dominant_periods.head(3).iterrows():
            print(f"    {row['period_Ma']:.1f} Ma ({row['contribution_pct']:.1f}%)")
    
    # 绘制对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (method_name, result) in enumerate(results.items()):
        # 原始vs重建
        ax = axes[idx, 0]
        ax.bar(result.original_ages, result.original_frequencies, 
               alpha=0.3, color='gray', label='Original')
        ax.plot(result.reconstruction.age_continuous, 
               result.reconstruction.frequency_continuous, 
               'b-', label='Reconstructed')
        ax.set_title(f'{method_name} - Signal')
        ax.legend()
        
        # 成分
        ax = axes[idx, 1]
        for comp in result.decomposition.components[:4]:
            ax.plot(result.reconstruction.age_continuous, comp.signal, 
                   label=f'RC{comp.index} ({comp.period:.1f}Ma)', alpha=0.7)
        ax.set_title(f'{method_name} - Components')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('./results/metaspectral_methods_comparison.png', dpi=150)
    print("\n方法对比图已保存")
    plt.close()


def example_signal_extraction():
    """特定信号提取示例"""
    print("\n" + "=" * 60)
    print("示例4: 特定地质信号提取")
    print("=" * 60)
    
    # 生成包含多个地质周期信号的模拟数据
    np.random.seed(42)
    
    age_grid = np.linspace(0, 500, 1000)
    
    # 合成信号: 超大陆周期(400Ma) + Wilson旋回(250Ma) + 造山周期(80Ma)
    signal = (
        5 * np.sin(2 * np.pi * age_grid / 400) +  # 超大陆周期
        3 * np.sin(2 * np.pi * age_grid / 250) +  # Wilson旋回
        2 * np.sin(2 * np.pi * age_grid / 80) +   # 造山周期
        np.random.normal(0, 0.5, len(age_grid))   # 噪声
    )
    signal = np.maximum(signal, 0)  # 确保非负
    
    # 创建分析器
    analyzer = DetritalSpectrumAnalyzer()
    
    result = analyzer.analyze_period(
        ages=age_grid,
        frequencies=signal,
        period_name="Synthetic",
        reconstruction_params={'n_points': 500},
        decomposition_params={'window_size': 100, 'n_components': 10},
    )
    
    # 提取各种地质信号
    target_signals = [
        'supercontinent',
        'wilson_cycle', 
        'orogenic',
    ]
    
    fig, axes = plt.subplots(len(target_signals) + 1, 1, 
                            figsize=(12, 10), sharex=True)
    
    # 原始信号
    axes[0].plot(age_grid, signal, 'k-', linewidth=2, label='Original Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 各提取的信号
    for i, signal_name in enumerate(target_signals, 1):
        try:
            extracted = result.get_signal(signal_name)
            axes[i].plot(age_grid, extracted.combined_signal, 
                        linewidth=2, label=f'{signal_name}')
            axes[i].fill_between(age_grid, extracted.combined_signal, 
                                alpha=0.3)
            axes[i].set_ylabel('Amplitude')
            axes[i].set_title(f'{signal_name.replace("_", " ").title()} '
                            f'(Contribution: {extracted.total_contribution:.1%})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Not found: {e}', 
                        ha='center', va='center')
            axes[i].set_title(signal_name)
    
    axes[-1].set_xlabel('Age (Ma)')
    plt.tight_layout()
    plt.savefig('./results/metaspectral_signal_extraction.png', dpi=150)
    print("\n信号提取图已保存")
    plt.close()


def example_quick_analysis():
    """快速分析函数示例"""
    print("\n" + "=" * 60)
    print("示例5: 快速分析函数")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    ages = np.random.normal(300, 100, 1000)
    ages = ages[(ages > 0) & (ages < 600)]
    
    # 快速分析 - 只返回贡献率表格
    contribution_df = quick_spectrum_analysis(
        ages=ages,
        period_name="Quick_Test",
        return_full=False,
    )
    
    print("\n快速分析结果 (各信号范围贡献率):")
    print(contribution_df[['signal_range', 'period_min_Ma', 'period_max_Ma', 
                          'total_contribution_pct']].to_string(index=False))
    
    # 快速分析 - 返回完整结果
    full_result = quick_spectrum_analysis(
        ages=ages,
        period_name="Quick_Test_Full",
        return_full=True,
    )
    
    print("\n完整结果 - 主导周期:")
    print(full_result.dominant_periods.head(5).to_string(index=False))


def main():
    """运行所有示例"""
    import os
    os.makedirs('./results', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("元频谱分析工具演示")
    print("=" * 60)
    print("\n这个示例展示如何:")
    print("1. 将离散的年龄频率分布重建为连续信号")
    print("2. 使用SSA/EMD等方法进行频谱分解")
    print("3. 计算各周期成分的贡献率")
    print("4. 提取特定的地质信号 (超大陆周期、Wilson旋回等)")
    
    try:
        example_basic_analysis()
    except Exception as e:
        print(f"示例1失败: {e}")
    
    try:
        example_real_data()
    except Exception as e:
        print(f"示例2失败: {e}")
    
    try:
        example_advanced_decomposition()
    except Exception as e:
        print(f"示例3失败: {e}")
    
    try:
        example_signal_extraction()
    except Exception as e:
        print(f"示例4失败: {e}")
    
    try:
        example_quick_analysis()
    except Exception as e:
        print(f"示例5失败: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
