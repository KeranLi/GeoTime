"""
time_group.ipynb 的重构版本

使用新的 DetritalAnalysisPipeline 替代原有的重复代码
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geotime import DetritalAnalysisPipeline, quick_analyze
from geotime.geological import GEOLOGICAL_PERIODS


def main():
    # 文件路径
    data_file = Path("./data/Qiagtang_large.xlsx")
    output_dir = Path("./results/timegroup_analysis")
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        print("请确保在正确的目录下运行此脚本")
        return
    
    print("=" * 60)
    print("碎屑锆石地质时代频率分析")
    print("=" * 60)
    
    # 定义要分析的地质时代
    # 可选：三叠纪、侏罗纪、白垩纪的各个世
    periods_to_analyze = [
        # 三叠纪
        'Lower_Triassic',
        'Middle_Triassic', 
        'Upper_Triassic',
        # 侏罗纪
        'Lower_Jurassic',
        'Middle_Jurassic',
        'Upper_Jurassic',
        # 白垩纪
        'Lower_Cretaceous',
        'Upper_Cretaceous',
    ]
    
    print(f"\n要分析的地质时代: {len(periods_to_analyze)} 个")
    for p in periods_to_analyze:
        print(f"  - {p}")
    
    # 方法1: 使用完整Pipeline (推荐)
    print("\n" + "-" * 60)
    print("[方法1] 使用 DetritalAnalysisPipeline")
    print("-" * 60)
    
    pipeline = DetritalAnalysisPipeline()
    
    results = pipeline.run(
        data_file=data_file,
        periods=periods_to_analyze,
        bin_size=1.0,              # 1 Ma分箱
        smooth_frac=0.1,           # LOWESS平滑参数
        stft_params={              # STFT参数
            'nperseg': 256,
            'window': 'hann',
        },
        lomb_scargle_params={      # Lomb-Scargle参数
            'n_frequencies': 1000,
        },
    )
    
    # 打印结果摘要
    print("\n分析结果摘要:")
    for period, result in results.items():
        print(f"\n{period}:")
        print(f"  样品数: {result.n_samples}")
        print(f"  颗粒数: {result.n_grains}")
        
        if result.lomb_scargle_result:
            peaks = result.lomb_scargle_result.get_peaks(n_peaks=3)
            print(f"  主要周期:")
            for i, (period_val, freq) in enumerate(zip(peaks['periods'], peaks['frequencies']), 1):
                print(f"    {i}. {period_val:.1f} Ma ({freq:.4f} 1/Ma)")
    
    # 保存结果
    print(f"\n保存结果到: {output_dir}")
    pipeline.save_results(
        output_dir=output_dir,
        save_csv=True,
        save_plots=True,
    )
    
    # 方法2: 使用快速分析函数
    print("\n" + "-" * 60)
    print("[方法2] 使用 quick_analyze 函数")
    print("-" * 60)
    
    # 只分析侏罗纪作为示例
    jurassic_periods = GEOLOGICAL_PERIODS['Jurassic']
    
    pipeline2 = quick_analyze(
        data_file=data_file,
        output_dir=output_dir / "jurassic_only",
        periods=jurassic_periods,
        bin_size=1.0,
        smooth_frac=0.1,
    )
    
    print("\n侏罗纪分析完成!")
    
    # 方法3: 分步执行 (更灵活的控制)
    print("\n" + "-" * 60)
    print("[方法3] 分步执行")
    print("-" * 60)
    
    pipeline3 = DetritalAnalysisPipeline()
    
    # 步骤1: 加载数据
    pipeline3.load_data(data_file)
    
    # 步骤2: 提取所有地质时代的年龄
    pipeline3.extract_period_ages()
    
    # 步骤3: 只处理感兴趣的特定时代
    target_period = 'Middle_Triassic'
    if target_period in pipeline3.period_ages:
        result = pipeline3.process_period(
            target_period,
            bin_size=1.0,
            smooth_frac=0.05,  # 使用不同的平滑参数
        )
        
        print(f"\n{target_period} 处理完成:")
        print(f"  颗粒数: {result.n_grains}")
        
        # 自定义绘图
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 频率分布
        result.plot_frequency(ax=axes[0, 0])
        
        # STFT
        if result.stft_result:
            power_db = 10 * np.log10(result.stft_result.power + 1e-10)
            im = axes[0, 1].pcolormesh(
                result.stft_result.times,
                result.stft_result.frequencies,
                power_db,
                shading='gouraud',
                cmap='viridis',
            )
            axes[0, 1].set_title('STFT Spectrogram')
            axes[0, 1].set_xlabel('Age (Ma)')
            axes[0, 1].set_ylabel('Frequency (1/Ma)')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Lomb-Scargle
        if result.lomb_scargle_result:
            result.lomb_scargle_result.plot(ax=axes[1, 0], show_peaks=3)
        
        # 年龄直方图
        axes[1, 1].hist(result.ages, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Age (Ma)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Age Distribution (n={result.n_grains})')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{target_period}_detailed.png', dpi=150)
        print(f"  详细图表已保存")
    
    print("\n" + "=" * 60)
    print("所有分析完成!")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # 添加必要的import
    import matplotlib.pyplot as plt
    
    main()
