"""
碎屑数据频率分析示例

展示如何分析碎屑锆石年龄频率数据
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geotime.analysis import GeoTimeSeriesAnalysis
import matplotlib.pyplot as plt


def main():
    data_file = Path("./data/Puetz_1200K.xlsx")
    output_dir = Path("./results/detrital_analysis")
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return
    
    print("=" * 50)
    print("碎屑锆石年龄频率分析")
    print("=" * 50)
    
    analysis = GeoTimeSeriesAnalysis()
    
    # 加载频率数据 (value_column=None 表示频率统计模式)
    print("\n[1/3] 加载频率数据...")
    analysis.load_data(
        data_file,
        age_column="Published 206Pb/238U age (Ma)",
        value_column=None,  # 频率统计模式
    )
    print(f"  加载了 {len(analysis.age)} 个年龄点")
    
    # SSA分解 (跳过WA，因为已经是频率数据)
    print("\n[2/3] SSA分解...")
    analysis.apply_ssa(window_size=200, n_components=4)
    print("  完成")
    
    # 频谱分析
    print("\n[3/3] 频谱分析...")
    result = analysis.analyze_spectrum(method="mtm")
    
    print(f"\n  主要周期:")
    for i, period in enumerate(result['peak_periods'], 1):
        print(f"    {i}. {period:.2f} Ma")
    
    # 保存和绘图
    analysis.save_results(output_dir)
    
    fig = analysis.plot_spectrum(title="Detrital Zircon Age Frequency Analysis")
    fig.savefig(output_dir / "detrital_spectrum.png", dpi=150)
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
