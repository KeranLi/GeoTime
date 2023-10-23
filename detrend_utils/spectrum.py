from scipy import signal
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter

class Filter:

    def __init__(self, series):
        self.series = series

    # 高斯滤波
    def gaussian(self, sigma):
        return gaussian_filter1d(self.series, sigma)  

    # 脱趋
    def detrend(self):
        return signal.detrend(self.series)
    
    # 2pi-MTM
    def Mtm(self, nperseg):
        return signal.welch(self.series,
                             nperseg=nperseg,
                             scaling='spectrum',
    )

    # 周期图
    def lombscargle(self, freqs):
        time = np.arange(len(self.series))
        power = signal.lombscargle(time, self.series, freqs)
        return power

    # 求峰值
    def find_peaks(self, data):
        peaks, _ = signal.find_peaks(data)
        return peaks
    
    # Butterworth滤波
    def butter_lowpass(self, cutoff, fs, order):
        b, a = butter(order, cutoff, fs=fs, btype='low')
        return lfilter(b, a, self.series)
    
    # 小波变换
    def cwt(self, widths):
        t, dt = np.linspace(0, 1, len(self.series), retstep=True) 
        fs = 1/dt
        cwtm = signal.cwt(self.series, signal.morlet2, widths)
        return cwtm