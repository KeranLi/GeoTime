"""
频谱分解模块

对连续化的频率分布信号进行频谱分解，提取各周期成分
"""

from enum import Enum
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
from scipy import signal

try:
    import PyEMD
    EMD_AVAILABLE = True
except ImportError:
    EMD_AVAILABLE = False


class DecompositionMethod(Enum):
    """频谱分解方法"""
    SSA = "ssa"              # 奇异谱分析
    EMD = "emd"              # 经验模态分解
    WAVELET = "wavelet"      # 小波分解
    FFT_BANDS = "fft_bands"  # FFT频带分解
    FILTER_BANK = "filter_bank"  # 滤波器组


@dataclass
class DecompositionComponent:
    """分解成分"""
    index: int                      # 成分编号
    signal: np.ndarray              # 时域信号
    period: Optional[float] = None  # 主导周期
    energy: Optional[float] = None  # 能量
    contribution: Optional[float] = None  # 贡献率
    
    def __repr__(self):
        return (f"Component({self.index}, "
                f"period={self.period:.2f}Ma, "
                f"contribution={self.contribution:.2%})")


@dataclass
class DecompositionResult:
    """频谱分解结果"""
    method: str
    components: List[DecompositionComponent]
    original_signal: np.ndarray
    reconstructed_signal: np.ndarray
    residual: np.ndarray
    total_energy: float
    
    def get_by_period_range(
        self,
        min_period: float,
        max_period: float,
    ) -> List[DecompositionComponent]:
        """获取特定周期范围内的成分"""
        return [
            c for c in self.components
            if c.period is not None and min_period <= c.period < max_period
        ]
    
    def get_trend_component(self) -> DecompositionComponent:
        """获取趋势成分 (最大周期)"""
        max_period_comp = max(
            self.components,
            key=lambda c: c.period if c.period else 0
        )
        return max_period_comp
    
    def get_high_frequency_components(
        self,
        period_threshold: float = 10.0,
    ) -> List[DecompositionComponent]:
        """获取高频成分 (周期小于阈值)"""
        return [c for c in self.components 
                if c.period is not None and c.period < period_threshold]


class SpectralDecomposer:
    """
    频谱分解器

    对连续信号进行多尺度分解
    """

    def __init__(self, age: np.ndarray, signal: np.ndarray):
        """
        Args:
            age: 年龄数组 (Ma)
            signal: 信号数组 (频率)
        """
        self.age = np.asarray(age)
        self.signal = np.asarray(signal)
        self.dt = np.median(np.diff(age))  # 采样间隔
        self.fs = 1.0 / self.dt if self.dt > 0 else 1.0  # 采样频率

    def decompose_ssa(
        self,
        window_size: int = 100,
        n_components: Optional[int] = None,
        grouping: Optional[Dict[str, List[int]]] = None,
    ) -> DecompositionResult:
        """
        奇异谱分析 (SSA) 分解

        Args:
            window_size: 嵌入维度
            n_components: 保留成分数 (None则自动确定)
            grouping: 成分分组 {'trend': [0,1], 'periodic': [2,3,4]}

        Returns:
            DecompositionResult
        """
        from geotime.detrend import SSA
        
        # 中心化
        signal_centered = self.signal - np.mean(self.signal)
        
        # SSA分解
        ssa = SSA(signal_centered, window_size=window_size)
        trajectory = ssa.build_trajectory_matrix()
        components = ssa.decompose(trajectory)
        
        # 确定成分数
        if n_components is None:
            # 基于特征值阈值自动确定
            n_components = min(len(components), 20)
        
        # 重构各成分
        decomp_components = []
        total_energy = np.sum(self.signal ** 2)
        
        for i in range(min(n_components, len(components))):
            comp_signal = ssa.reconstruct([components[i]], n_components=1)
            
            # 计算周期 (通过自相关)
            period = self._estimate_period(comp_signal)
            
            # 计算能量和贡献率
            energy = np.sum(comp_signal ** 2)
            contribution = energy / total_energy if total_energy > 0 else 0
            
            decomp_components.append(DecompositionComponent(
                index=i,
                signal=comp_signal,
                period=period,
                energy=energy,
                contribution=contribution,
            ))
        
        # 重构信号
        reconstructed = sum(c.signal for c in decomp_components)
        reconstructed += np.mean(self.signal)  # 加回均值
        residual = self.signal - reconstructed
        
        return DecompositionResult(
            method="SSA",
            components=decomp_components,
            original_signal=self.signal,
            reconstructed_signal=reconstructed,
            residual=residual,
            total_energy=total_energy,
        )

    def decompose_emd(
        self,
        max_imf: int = 10,
    ) -> DecompositionResult:
        """
        经验模态分解 (EMD)

        Args:
            max_imf: 最大IMF数量

        Returns:
            DecompositionResult
        """
        if not EMD_AVAILABLE:
            raise ImportError("需要安装PyEMD: pip install EMD-signal")
        
        from PyEMD import EMD
        
        emd = EMD()
        imfs = emd(self.signal, max_imf=max_imf)
        
        decomp_components = []
        total_energy = np.sum(self.signal ** 2)
        
        for i, imf in enumerate(imfs):
            # 计算周期
            period = self._estimate_period(imf)
            
            # 计算能量
            energy = np.sum(imf ** 2)
            contribution = energy / total_energy if total_energy > 0 else 0
            
            decomp_components.append(DecompositionComponent(
                index=i,
                signal=imf,
                period=period,
                energy=energy,
                contribution=contribution,
            ))
        
        # EMD的residual是最后的趋势项
        reconstructed = sum(c.signal for c in decomp_components[:-1])
        residual = decomp_components[-1].signal if len(decomp_components) > 1 else np.zeros_like(self.signal)
        
        return DecompositionResult(
            method="EMD",
            components=decomp_components,
            original_signal=self.signal,
            reconstructed_signal=reconstructed,
            residual=residual,
            total_energy=total_energy,
        )

    def decompose_fft_bands(
        self,
        period_bands: Optional[List[Tuple[float, float]]] = None,
    ) -> DecompositionResult:
        """
        基于FFT的频带分解

        Args:
            period_bands: 周期频带列表 [(min_period, max_period), ...]

        Returns:
            DecompositionResult
        """
        if period_bands is None:
            # 默认频带: 长周期 >100Ma, 中周期 50-100Ma, 短周期 10-50Ma, 高频 <10Ma
            period_bands = [
                (100, np.inf),
                (50, 100),
                (20, 50),
                (10, 20),
                (0, 10),
            ]
        
        # FFT
        fft_result = np.fft.rfft(self.signal)
        freqs = np.fft.rfftfreq(len(self.signal), d=self.dt)
        
        decomp_components = []
        total_energy = np.sum(np.abs(fft_result) ** 2)
        
        for i, (min_period, max_period) in enumerate(period_bands):
            # 周期转频率 (注意: 周期越大频率越小)
            max_freq = 1 / min_period if min_period > 0 else np.inf
            min_freq = 1 / max_period if max_period < np.inf else 0
            
            # 频带滤波
            band_fft = np.zeros_like(fft_result)
            mask = (freqs >= min_freq) & (freqs < max_freq)
            band_fft[mask] = fft_result[mask]
            
            # 反变换
            band_signal = np.fft.irfft(band_fft, n=len(self.signal))
            
            # 主导周期 (频带中心)
            center_period = (min_period + max_period) / 2 if max_period < np.inf else min_period * 2
            
            # 能量
            energy = np.sum(np.abs(band_fft) ** 2)
            contribution = energy / total_energy if total_energy > 0 else 0
            
            decomp_components.append(DecompositionComponent(
                index=i,
                signal=band_signal,
                period=center_period,
                energy=energy,
                contribution=contribution,
            ))
        
        reconstructed = sum(c.signal for c in decomp_components)
        residual = self.signal - reconstructed
        
        return DecompositionResult(
            method="FFT_BANDS",
            components=decomp_components,
            original_signal=self.signal,
            reconstructed_signal=reconstructed,
            residual=residual,
            total_energy=total_energy,
        )

    def decompose_wavelet(
        self,
        wavelet: str = 'morl',
        scales: Optional[np.ndarray] = None,
        n_components: int = 5,
    ) -> DecompositionResult:
        """
        小波分解

        Args:
            wavelet: 小波类型
            scales: 尺度数组
            n_components: 成分数

        Returns:
            DecompositionResult
        """
        import pywt
        
        if scales is None:
            # 自动确定尺度范围
            scales = np.arange(1, min(128, len(self.signal)//4))
        
        # 连续小波变换
        coef, freqs = pywt.cwt(self.signal, scales, wavelet, 
                               sampling_period=self.dt)
        
        # 按能量排序选择主要成分
        energies = np.sum(np.abs(coef) ** 2, axis=1)
        top_indices = np.argsort(energies)[-n_components:][::-1]
        
        decomp_components = []
        total_energy = np.sum(energies)
        
        for i, idx in enumerate(top_indices):
            # 重构该尺度成分 (简化处理)
            scale_signal = np.real(coef[idx])
            
            # 周期
            period = 1 / freqs[idx] if freqs[idx] > 0 else np.inf
            
            # 能量
            energy = energies[idx]
            contribution = energy / total_energy if total_energy > 0 else 0
            
            decomp_components.append(DecompositionComponent(
                index=i,
                signal=scale_signal,
                period=period,
                energy=energy,
                contribution=contribution,
            ))
        
        reconstructed = sum(c.signal for c in decomp_components)
        residual = self.signal - reconstructed
        
        return DecompositionResult(
            method="WAVELET",
            components=decomp_components,
            original_signal=self.signal,
            reconstructed_signal=reconstructed,
            residual=residual,
            total_energy=total_energy,
        )

    def _estimate_period(self, signal_data: np.ndarray) -> Optional[float]:
        """估计信号的主导周期"""
        # 使用自相关
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 找第一个峰值
        peaks = signal.find_peaks(autocorr, distance=5)[0]
        
        if len(peaks) > 0:
            # 第一个峰值的位置就是周期
            period_samples = peaks[0]
            period = period_samples * self.dt
            return period
        
        return None

    def decompose(
        self,
        method: DecompositionMethod = DecompositionMethod.SSA,
        **kwargs,
    ) -> DecompositionResult:
        """
        通用分解接口

        Args:
            method: 分解方法
            **kwargs: 方法特定参数

        Returns:
            DecompositionResult
        """
        if method == DecompositionMethod.SSA:
            return self.decompose_ssa(**kwargs)
        elif method == DecompositionMethod.EMD:
            return self.decompose_emd(**kwargs)
        elif method == DecompositionMethod.FFT_BANDS:
            return self.decompose_fft_bands(**kwargs)
        elif method == DecompositionMethod.WAVELET:
            return self.decompose_wavelet(**kwargs)
        else:
            raise ValueError(f"未知分解方法: {method}")
