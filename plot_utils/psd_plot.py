import matplotlib.pyplot as plt
import numpy as np

def plot_psd_top3(freq, psd, title='Default', top3_freqs=None, top3_ages=None):

    # 绘制振幅谱,双对数坐标
    plt.loglog(freq, psd)  

    # 获取top3 psd值对应的索引
    top3_psd_idx = np.argsort(psd)[-3:]

    # 标注top 3峰值
    for i in range(3):
        psd_val = psd[top3_psd_idx[i]]
        plt.annotate(f"{top3_freqs[i]:.2f} ({top3_ages[i]:.1f} Ma)",
                    xy=(top3_freqs[i], psd_val))

    plt.xlabel('Frequency')
    plt.ylabel('Amplitude spectra')
    plt.title(title)
    plt.show()