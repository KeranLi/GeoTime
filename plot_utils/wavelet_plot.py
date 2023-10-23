import numpy as np
import matplotlib.pyplot as plt

#原始数据序列
xlim = np.array([0,32])
fig = plt.figure(figsize=(15,15))
f_ax1 = fig.add_axes([0.1,0.75,0.65,0.2])
f_ax1.plot(age,STD_Y_reconstructed) 
f_ax1.set_xlabel('Age (Ma)')
f_ax1.set_ylabel('Signal')
f_ax1.set_title('(a) SSA_RC1_Detrend CAR in the Equator Pacific (32Ma)')

#小波
f_ax2 = fig.add_axes([0.1, 0.37 ,0.65 ,0.28])
levels = [0.0625,0.125,0.25,0.5,1,2,4,8,16]
Yticks = np.arange(2**np.trunc(np.log2(np.min(period))),2**np.trunc(np.log2(np.max(period)))+1,1)
f_ax2.contourf(time,np.log2(period),np.log2(power),np.log2(levels),extend='both')
f_ax2.set_xlabel('Age (Ma)')
f_ax2.set_ylabel('Period (Ma)')
f_ax2.set_title('(b) CAR in the Equator Pacific (32Ma) SST Wavelet Power Spectrum')  
f_ax2.set_ylim(np.log2(np.max(period)),np.log2(np.min(period)))
f_ax2.contour(time,np.log2(period),sig95,levels=[-99,1],colors ='k')
f_ax2.plot(time,np.log2(coi),'k')
f_ax2.set_yticks(np.log2(Yticks)[[0,1,2,3,4,9,14,19,29]])
f_ax2.set_yticklabels(Yticks[[0,1,2,3,4,9,14,19,29]])

#全局功率谱  
f_ax3 = fig.add_axes([0.77, 0.37, 0.2, 0.28])
f_ax3.plot(global_ws,np.log2(period))
f_ax3.plot(global_signif,np.log2(period),'--')
f_ax3.set_xlabel('Power')
f_ax3.set_title('(c) Global Wavelet Spectrum')  
f_ax3.set_ylim(np.log2([np.max(period),np.min(period)]))
f_ax3.set_yticks(np.log2(Yticks))
f_ax3.set_yticklabels('')
f_ax3.set_xlim(0,1.25*np.max(global_ws))

#局部功率谱
f_ax4 = fig.add_axes([0.1,0.07 ,0.65, 0.2]) 
f_ax4.plot(xlim,scaleavg_signif+np.array([0,0]),'--')
f_ax4.plot(time,scale_avg)
f_ax4.set_xlabel('Age (Ma)')
f_ax4.set_ylabel('Avg variance')
f_ax4.set_title('(d) Scale-average Time Series')

plt.show()