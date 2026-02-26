"""
频谱分析模块

包含多种频谱分析方法：
- 高斯滤波
- 2π-MTM (多锥形法)
- Lomb-Scargle周期图
- Butterworth低通滤波
- 连续小波变换
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter


class SpectrumAnalyzer:
    """
    频谱分析器

    提供多种频谱分析和滤波方法
    """

    def __init__(self, series: np.ndarray):
        """
        Args:
            series: 输入时间序列
        """
        self.series = np.asarray(series).flatten()

    def gaussian_filter(self, sigma: float) -> np.ndarray:
        """
        高斯滤波

        Args:
            sigma: 高斯核标准差

        Returns:
            滤波后的序列
        """
        return gaussian_filter1d(self.series, sigma)

    def detrend(self) -> np.ndarray:
        """
        去除线性趋势

        Returns:
            去趋势后的序列
        """
        return signal.detrend(self.series)

    def mtm_power_spectrum(self, nperseg: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """
        2π-MTM (多锥形法) 功率谱估计

        Args:
            nperseg: 每段数据长度

        Returns:
            (频率数组, 功率谱密度数组)
        """
        freqs, psd = signal.welch(
            self.series,
            nperseg=nperseg,
            scaling='spectrum',
        )
        return freqs, psd

    def lomb_scargle(self, freqs: np.ndarray) -> np.ndarray:
        """
        Lomb-Scargle周期图

        适用于非均匀采样数据

        Args:
            freqs: 频率数组

        Returns:
            功率数组
        """
        time = np.arange(len(self.series))
        return signal.lombscargle(time, self.series, freqs)

    def find_peaks(self, data: np.ndarray, n_peaks: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        查找功率谱峰值

        Args:
            data: 功率谱数据
            n_peaks: 返回前N个峰值

        Returns:
            (峰值索引数组, 峰值周期数组)
        """
        peaks, _ = signal.find_peaks(data)
        
        # 按峰值高度排序
        peak_heights = data[peaks]
        top_indices = np.argsort(peak_heights)[::-1][:n_peaks]
        
        return peaks[top_indices], peak_heights[top_indices]

    def butterworth_lowpass(
        self,
        cutoff: float,
        fs: float,
        order: int = 5,
    ) -> np.ndarray:
        """
        Butterworth低通滤波

        Args:
            cutoff: 截止频率
            fs: 采样频率
            order: 滤波器阶数

        Returns:
            滤波后的序列
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, self.series)

    def continuous_wavelet_transform(
        self,
        widths: np.ndarray,
        wavelet: callable = None,
    ) -> np.ndarray:
        """
        连续小波变换 (CWT)

        Args:
            widths: 小波尺度数组
            wavelet: 小波函数 (默认 morlet2)

        Returns:
            小波系数矩阵
        """
        if wavelet is None:
            wavelet = signal.morlet2
        return signal.cwt(self.series, wavelet, widths)


# 保持向后兼容的别名
Filter = SpectrumAnalyzer
