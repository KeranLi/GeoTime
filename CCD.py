"""
@Athour: Keran Li@Keranli98@outlook.com

@Logs@20230922: the matlpotlib.pyplot.bar has some errors in the inner functional
frameworks baceuse the (x,y)(waiting to be plotted) would be replaces as 'None'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import DataLoader
from plot_utils.bar_plot import plot_bar as bar
from detrend_utils.wa import WeightedAverage
from plot_utils.signal_compare import plot_line as line
from detrend_utils.ssa import SSA
from plot_utils.traj_plot import plot_trajectory_matrix as traj
from detrend_utils.spectrum import Filter
from plot_utils.psd_plot import plot_psd_top3 as psd

# from ./data import file
CAR = DataLoader('./data/equator_pacific.xlsx',
                time_filter=None,
                select_col='CAR(mg/cm2/kyr)'
                )

Age = DataLoader('./data/equator_pacific.xlsx',
                 time_filter=None)

# check the shapes of CAR and Age by comparing their colunn lenth
# If their lenth are same, print
if len(Age) == len(CAR):
    print('----------The Age shape is: {}'.format(Age.shape), end='----------\n')
    print('----------The CAR shape is: {}'.format(CAR.shape), end='----------\n')
    print('----------Congratulations! Same shapes cause a successful loading! The length of Age and CAR are same-----------')
else:
    print('----------The Age shape is: {}'.format(Age.shape), end='----------\n')
    print('----------The CAR shape is: {}'.format(CAR.shape), end='----------\n')
    print('----------Keep trying! Different shapes cause a failed loading! The length of Age and CAR are not same----------')

# plot the bar histgram of the Age and CAR by bar_plot.py
bar(
    np.array(Age),
    np.array(CAR),
    title='The global statistics of CAR vs. Age',
    auto_axis=False,
    xlimit=[32,0],
    ylimit=[0,5000]
)

# The first step of wa detrend is to set your weights
# here, I use the 40%wa detrend and the window size is 5
weights = np.array([[0.4], [0.15], [0.15], [0.15], [0.15]])
wa = WeightedAverage(weights)
car_40_per_wa_trend = wa.rolling(CAR, window=5, min_periods=1, center=True)

# plot the original and 40%wa detrended signals
line(
    np.array(Age),
    np.array(CAR),
    np.array(car_40_per_wa_trend),
    title='The camparasion of global original and 40%wa detrended lines',
    xlimit=None,
    ylimit=None
)

# SSA analysis
series = np.array(car_40_per_wa_trend) #Tranlating into ndarray

# Centralization
series = series - np.mean(series)

# ----------set hyper-parameters of the trajectory matrix start----------
N = len(series)
K = 200
ssa = SSA(series, N, K)
# ----------set hyper-parameters of the trajectory matrix over----------

# I don't have such a good a idea so that nowadays I just use a copy of X to visulization 
x_copy = np.copy(series)
x_copy = ssa.trajectory_matrix(x_copy, K)
# visulize the trajectory matrix
traj(
    x_copy,
    max_rows = 500,
    max_cols = 5000
)

# calculate the trajectory matrix
X = ssa.trajectory_matrix(series, K)
"""
@Note: I'm not sure why in jupyter would decrease one dimension
"""
X = X.reshape(X.shape[0], -1)
# decomposition and detrending by reconstructing
components = ssa.decompose(X)
reconstructed = ssa.reconstruct(np.array(components),4)

# plot the original and SSA_RC1-RC4 detrended signals
line(
    np.array(Age),
    np.array(CAR),
    np.array(reconstructed),
    title='The camparasion of global original and SSA_RC1-4 detrended lines',
    xlimit=None,
    ylimit=None
)

# --------Gaissian filter start---------
# set the hyper-parameter sigma
sigma = 0.1
# input reconstructed
filter = Filter(reconstructed)
gaussian_filter = filter.gaussian(sigma)
# --------Gaissian filter over---------

# plot the original and Gaussion detrended signals
line(
    np.array(Age),
    np.array(CAR),
    np.array(reconstructed),
    title='The camparasion of global original and Gaussion detrended lines',
    xlimit=None,
    ylimit=None
)

# Many spectrum detrended methods rely on the detrend
filter = Filter(reconstructed)
detrended = filter.detrend()

# --------2pi-MTM filter start---------
# set the hyper-parameter nperseg
nperseg = 128
# input detrended
MTM_freq, MTM_psd = filter.Mtm(nperseg)
# find peaks
MTM_peaks = filter.find_peaks(MTM_psd)
# index the frequencies of the most highest 3 peaks
MTM_top3_peak_idx =  MTM_peaks[:3]
# Top3 frequcies
MTM_top3_freqs = MTM_freq[MTM_top3_peak_idx]
# Period calculation
MTM_top3_ages = 1/MTM_top3_freqs
print('---------Congratulations!The 2Pi-MTM analysis is over!The periods are: {} Ma'.format(MTM_top3_ages), end='----------\n')
# --------2pi-MTM filter over---------

# plot the 2Pi-MTM signals
psd(
    np.array(MTM_freq),
    np.array(MTM_psd),
    title='The global 2Pi-MTM analysis',
    top3_freqs=MTM_top3_freqs,
    top3_ages=MTM_top3_ages
)

# --------Lomb-Scargle filter start---------
# set the hyper-parameter freqs
LS_freqs = np.linspace(0.01, 5, len(detrended))
# input detrended
LS_power = filter.lombscargle(LS_freqs)
# find peaks
LS_peaks = filter.find_peaks(LS_power)
# index the frequencies of the most highest 3 peaks
LS_top3_peak_idx =  LS_peaks[:3]
# Top3 frequcies
LS_top3_freqs = LS_freqs[LS_top3_peak_idx]
# Period calculation
LS_top3_ages = 1/LS_top3_freqs
print('----------Congratulations!The Lomb-Scargle analysis is over!The periods are: {} Ma'.format(LS_top3_ages), end='----------\n')
# --------Lomb-Scargle filter over---------

#----------plot the Lomb-Scargle signals----------
fig, axs = plt.subplots(2, 1, sharex=False)

axs[0].plot(np.arange(len(detrended)), detrended)
axs[0].invert_xaxis()
axs[1].plot(LS_freqs, LS_power)
axs[1].invert_xaxis()

plt.show()
#----------plot the Lomb-Scargle signals----------

# --------Low Pass Filter filter start---------
# set the hyper-parameter freqs
order = 20
# sampling frequency
fs = 30
# low pass cutoff frequenct
cutoff = 10
# input detrended
LPF = filter.butter_lowpass(cutoff, fs, order)
# --------Low Pass Filter filter over---------

#----------plot the Low Pass Filter----------
fig, axs = plt.subplots(2, 1, sharex=False)

axs[0].plot(np.arange(len(detrended)), detrended)
axs[0].invert_xaxis()
axs[1].plot(np.arange(len(detrended)), LPF)
axs[1].invert_xaxis()

plt.show()
#----------plot the Low Pass Filter----------

# --------wavelet filter start---------
# set the hyper-parameter freqs
w = 1.
freq = np.linspace(1, fs/2, 20)
width = w*fs / (16*freq*np.pi)
# input detrended
wavelet = filter.cwt(width)

plt.pcolormesh(np.arange(len(detrended)), freq, np.abs(wavelet), cmap='viridis', shading='gouraud')
plt.show()
# --------wavelet filter over---------