"""
基础分析示例

展示如何使用新的GeoTime API进行地质时间序列分析
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from geotime.analysis import GeoTimeSeriesAnalysis


def main():
    # 文件路径
    data_file = Path("./data/equator_pacific.xlsx")
    output_dir = Path("./results/example_analysis")
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        print("请确保在正确的目录下运行此脚本")
        return
    
    # 创建分析器
    print("=" * 50)
    print("GeoTime 地质时间序列分析")
    print("=" * 50)
    
    analysis = GeoTimeSeriesAnalysis()
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    analysis.load_data(
        data_file,
        age_column="Age(Ma)",
        value_column="CAR(mg/cm2/kyr)",
    )
    print(f"  加载了 {len(analysis.age)} 个数据点")
    
    # 2. 加权移动平均去趋势
    print("\n[2/4] 加权移动平均去趋势...")
    analysis.apply_weighted_average(window=5)
    print("  完成")
    
    # 3. SSA分解
    print("\n[3/4] SSA分解...")
    analysis.apply_ssa(window_size=200, n_components=4)
    print("  完成")
    
    # 4. 频谱分析
    print("\n[4/4] MTM频谱分析...")
    result = analysis.analyze_spectrum(
        method="mtm",
        nperseg=128,
        find_peaks=True,
        n_peaks=3,
    )
    
    print(f"\n  检测到主要周期:")
    for i, period in enumerate(result['peak_periods'], 1):
        freq = result['peak_frequencies'][i-1]
        print(f"    {i}. {period:.2f} Ma (频率: {freq:.4f})")
    
    # 保存结果
    print(f"\n保存结果到: {output_dir}")
    analysis.save_results(output_dir)
    
    # 绘制图表
    print("\n生成图表...")
    fig1 = analysis.plot_spectrum(title="MTM Power Spectrum Analysis")
    fig1.savefig(output_dir / "spectrum.png", dpi=150, bbox_inches="tight")
    print(f"  已保存: {output_dir / 'spectrum.png'}")
    
    print("\n" + "=" * 50)
    print("分析完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
