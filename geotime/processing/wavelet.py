"""
小波分析处理模块

实现Torrence和Compo (1998) 的小波分析方法
"""

import numpy as np
from numpy.fft import ifft
from scipy.special import gamma
from scipy.stats.distributions import chi2
from typing import Tuple


class WaveletAnalysis:
    """
    小波分析器

    实现Morlet、Paul和DOG (Mexican Hat)小波
    """

    def __init__(self):
        pass

    def _wave_bases(
        self,
        mother: str,
        k: np.ndarray,
        scale: float,
        param: float,
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        计算小波基函数

        Args:
            mother: 小波类型 ('MORLET', 'PAUL', 'DOG')
            k: 波数数组
            scale: 小波尺度
            param: 小波参数

        Returns:
            (daughter, fourier_factor, coi, dofmin)
        """
        n = len(k)
        k = np.asarray(k)
        
        # 创建正频率掩码
        k_pos = (k > 0).astype(float)
        
        if mother == 'MORLET':
            k0 = param
            expnt = -0.5 * (scale * k - k0) ** 2 * k_pos
            norm = np.sqrt(scale * k[1]) * (np.pi ** -0.25) * np.sqrt(n)
            daughter = norm * np.exp(expnt) * k_pos
            fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
            coi = fourier_factor / np.sqrt(2)
            dofmin = 2
            
        elif mother == 'PAUL':
            m = int(param)
            expnt = -scale * k * k_pos
            norm = np.sqrt(scale * k[1]) * (2 ** m / np.sqrt(m * np.prod(np.arange(2, 2 * m)))) * np.sqrt(n)
            daughter = norm * ((scale * k) ** m) * np.exp(expnt) * k_pos
            fourier_factor = 4 * np.pi / (2 * m + 1)
            coi = fourier_factor * np.sqrt(2)
            dofmin = 2
            
        elif mother == 'DOG':
            m = int(param)
            expnt = -0.5 * (scale * k) ** 2
            norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
            daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
            fourier_factor = 2 * np.pi * np.sqrt(2 / (2 * m + 1))
            coi = fourier_factor / np.sqrt(2)
            dofmin = 1
            
        else:
            raise ValueError(f"未知小波类型: {mother}")
        
        return daughter, fourier_factor, coi, dofmin

    def wavelet_transform(
        self,
        data: np.ndarray,
        dt: float,
        pad: bool = True,
        dj: float = 0.25,
        s0: float = None,
        j1: int = None,
        mother: str = 'MORLET',
        param: float = 6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        执行连续小波变换

        Args:
            data: 输入数据
            dt: 采样间隔
            pad: 是否填充到2的幂次
            dj: 尺度间隔
            s0: 最小尺度
            j1: 尺度数量
            mother: 小波类型
            param: 小波参数

        Returns:
            (wave, period, scale, coi)
        """
        n1 = len(data)
        x = data - data.mean()
        
        # 填充
        if pad:
            base2 = int(np.log2(n1) + 0.4999) + 1
            x = np.hstack([x, np.zeros(2 ** base2 - n1)])
        n = len(x)
        
        # 波数数组
        k = np.arange(1, n // 2 + 1) * (2 * np.pi / (n * dt))
        k = np.hstack([0, k, -k[np.arange((n - 1) // 2 - 1, -1, -1)]])
        
        # FFT
        f = np.fft.fft(x)
        
        # 默认参数
        if s0 is None:
            s0 = 2 * dt
        if j1 is None:
            j1 = int(np.log2(n1 * dt / s0) / dj)
        
        # 尺度数组
        scale = s0 * 2 ** (np.arange(j1 + 1) * dj)
        
        # 小波变换
        wave = np.zeros((j1 + 1, n), dtype=complex)
        for i in range(j1 + 1):
            daughter, fourier_factor, coi, dofmin = self._wave_bases(
                mother, k, scale[i], param
            )
            wave[i, :] = ifft(f * daughter)
        
        period = fourier_factor * scale
        
        # COI (影响锥)
        coi = coi * dt * np.hstack([
            1e-5,
            np.arange(1, (n1 + 1) // 2),
            np.arange(n1 // 2 - 1, 0, -1),
            1e-5,
        ])
        
        return wave[:, :n1], period, scale, coi

    def significance(
        self,
        data: np.ndarray,
        dt: float,
        scale: np.ndarray,
        sigtest: int = 0,
        lag1: float = 0.0,
        siglvl: float = 0.95,
        dof: float = -1,
        mother: str = 'MORLET',
        param: float = 6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算小波显著性水平

        Args:
            data: 输入数据
            dt: 采样间隔
            scale: 尺度数组
            sigtest: 显著性检验类型
            lag1: 滞后1自相关
            siglvl: 显著性水平
            dof: 自由度
            mother: 小波类型
            param: 小波参数

        Returns:
            (signif, fft_theor)
        """
        variance = np.var(data)
        n = len(data)
        
        # 小波参数
        if mother == 'MORLET':
            k0 = param
            fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
            empir = np.array([2., -1, -1, -1])
            if k0 == 6:
                empir[1:] = [0.776, 2.32, 0.60]
        elif mother == 'PAUL':
            m = param
            fourier_factor = 4 * np.pi / (2 * m + 1)
            empir = np.array([2., -1, -1, -1])
            if m == 4:
                empir[1:] = [1.132, 1.17, 1.5]
        elif mother == 'DOG':
            m = param
            fourier_factor = 2 * np.pi * np.sqrt(2 / (2 * m + 1))
            empir = np.array([1, -1, -1, -1])
            if m == 2:
                empir[1:] = [3.541, 1.43, 1.4]
            elif m == 6:
                empir[1:] = [1.966, 1.37, 0.97]
        
        period = scale * fourier_factor
        dofmin = empir[0]
        cdelta = empir[1]
        gamma_fac = empir[2]
        dj0 = empir[3]
        
        freq = dt / period
        fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1 ** 2)
        fft_theor = variance * fft_theor
        
        if dof == -1:
            dof = dofmin
        
        if sigtest == 0:
            dof = dofmin
            chisquare = chi2.ppf(siglvl, dof) / dof
            signif = fft_theor * chisquare
        else:
            signif = fft_theor  # 简化处理
        
        return signif, fft_theor
