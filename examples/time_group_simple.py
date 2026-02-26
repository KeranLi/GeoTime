"""
time_group.ipynb 的最简重构版本

只需几行代码即可完成原notebook的全部功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geotime import quick_analyze


def main():
    """最简单的使用方式"""
    
    # 一行代码完成所有分析！
    pipeline = quick_analyze(
        data_file="./data/Qiagtang_large.xlsx",
        output_dir="./results/simple_analysis",
        periods=[  # 原notebook分析的所有地质时代
            'Lower_Triassic',
            'Middle_Triassic',
            'Upper_Triassic',
            'Lower_Jurassic',
            'Middle_Jurassic',
            'Upper_Jurassic',
            'Lower_Cretaceous',
            'Upper_Cretaceous',
        ],
        bin_size=1.0,        # 1 Ma分箱
        smooth_frac=0.1,     # LOWESS平滑
    )
    
    print("分析完成！")
    print(f"结果保存在: ./results/simple_analysis")
    
    # 查看结果
    for period, result in pipeline.results.items():
        print(f"\n{period}:")
        print(f"  样品: {result.n_samples}, 颗粒: {result.n_grains}")
        
        if result.lomb_scargle_result:
            peaks = result.lomb_scargle_result.get_peaks(n_peaks=3)
            print(f"  周期: {peaks['periods']}")


if __name__ == "__main__":
    main()
