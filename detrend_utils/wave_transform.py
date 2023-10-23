import numpy as np
from numpy.fft import ifft
from scipy.special import gamma
from scipy.stats.distributions import chi2

"""
Parameters:
n is the lenth of sequence
dt is the temporal resolution
time is the sequence
pad = 0 do not padding/pad = 1 do padding
dj is usually defined as half of the dt
s0 is twice of the dt
j1 ususlly denoted as log2(dt)/dj
lag is the threshhold of the confidence level
mother is the type of wave. Here, we can choose "MORLET", "PAUL" and "DOG"
"""

class WaveletAnalyzer:
    def __init__(self, Y, time, pad, dj, s0, j1, lag1, mother, param):
            self.Y = Y
            self.time = time
            self.dt = time[1] - time[0]  # Calculate dt from the time array
            self.pad = pad
            self.dj = dj
            self.s0 = s0
            self.J1 = j1
            self.lag1 = lag1
            self.mother = mother
            self.param = param

    @Method
    def wave_bases(mother,k,scale,param):
        n = k.shape[0]
        k_tmp = k.copy()
        k_tmp[k_tmp>0]=1
        k_tmp[k_tmp<=0]=0
        if (mother =='MORLET'):
            k0 = param
            expnt = (-1)*(scale*k - k0)**2/2*k_tmp
            norm = np.sqrt(scale*k[1])*(np.pi**(-0.25))*np.sqrt(n)
            daughter = norm*np.exp(expnt)
            daughter = daughter*k_tmp     
            fourier_factor = (4*np.pi)/(k0 + np.sqrt(2 + k0**2))
            coi = fourier_factor/np.sqrt(2)
            dofmin = 2
        elif (mother =='PAUL'):
            m = param
            expnt = (-1)*(scale*k)*k_tmp
            norm = np.sqrt(scale*k[1])*(2**m/np.sqrt(m*np.prod(np.arange(2,2*m))))*np.sqrt(n)
            daughter = norm*((scale*k)**m)*np.exp(expnt)
            daughter = daughter*k_tmp  
            fourier_factor = 4*np.pi/(2*m+1)
            coi = fourier_factor*np.sqrt(2)
            dofmin = 2
        elif (mother =='DOG'):
            m = param
            expnt = (-1)*(scale*k)**2/2.0
            norm = np.sqrt(scale*k[1]/gamma(m+0.5))*np.sqrt(n)
            daughter = (-1)*norm*((1j)**m)*((scale*k)**m)*np.exp(expnt)
            fourier_factor = 2*np.pi*np.sqrt(2/(2*m+1))
            coi = fourier_factor/np.sqrt(2)
            dofmin = 1
        return daughter,fourier_factor,coi,dofmin

    def wavelet(Y,dt,pad,dj,s0,J1,mother,param):
        n1 = Y.shape[0]
        x = (Y - Y.mean())
        if (pad == 1):
            base2 = np.trunc(np.log(n1)/np.log(2) + 0.4999).astype('int32')
            x = np.hstack((x,np.zeros((2**(base2+1)-n1))))
        n = x.shape[0]
        k = np.arange(1,n//2+1,1)
        k = k*((2*np.pi)/(n*dt))
        k = np.hstack((0,k,((-1)*k[np.arange((n-1)//2-1,-1,-1)])))
        f = np.fft.fft(x)
        scale = s0*2**(np.arange(0,J1+1)*dj)
        period = scale
        wave = np.zeros((np.array(J1+1).astype('int32'),n))
        wave = wave + (1j)*wave
        for i in range(1,np.array(J1+2).astype('int32')):
            daughter,fourier_factor,coi,dofmin = wave_bases(mother,k,scale[i-1],param)
            wave[i-1,:] = ifft(f*daughter)
        period = fourier_factor*scale
        coi = coi*dt*np.hstack((1e-5,np.arange(1,(n1+1)//2,1),np.arange(1,n1//2,1)[::-1],1e-5))
        wave = wave[:,:n1]
        return wave,period,scale,coi

    def wave_signif(Y,dt,scale1,sigtest,lag1,siglvl,dof,mother,param):
        n1 = np.array([Y]).shape
        J1 = scale1.shape[0] - 1
        scale[:J1+1] = scale1
        s0 = np.min(scale)
        dj = np.log(scale[1]/scale[0])/np.log(2)
        variance = Y
        if (mother =='MORLET'):
            k0 = param
            fourier_factor = (4*np.pi)/(k0 + np.sqrt(2 + k0**2))
            empir = np.array([2.,-1,-1,-1])
            if (k0==6):
                empir[1:]= np.array([0.776,2.32,0.60])
        elif (mother =='PAUL'):
            m = param
            fourier_factor = 4*np.pi/(2*m+1)
            empir = np.array([2.,-1,-1,-1])
            if (m==4):
                empir[1:]= np.array([1.132,1.17,1.5])
        elif (mother =='DOG'):
            m = param
            fourier_factor = 2*np.pi*np.sqrt(2/(2*m+1))
            empir = np.array([1,-1,-1,-1])
            if (m==2):
                empir[1:]= np.array([3.541,1.43,1.4])
            elif (m==6):
                empir[1:]= np.array([1.966,1.37,0.97])
        period = scale*fourier_factor
        dofmin = empir[0]
        Cdelta = empir[1]
        gamma_fac = empir[2]
        dj0 = empir[3]
        freq = dt / period
        fft_theor = (1-lag1**2) / (1-2*lag1*np.cos(freq*2*np.pi)+lag1**2)
        fft_theor = variance*fft_theor
        signif = fft_theor
        if (np.array(dof).sum() == -1):
            dof = dofmin
        if (sigtest == 0):
            dof = dofmin
            chisquare = chi2.ppf(siglvl,dof)/dof
            signif = fft_theor*chisquare
        elif (sigtest == 1):
            if (dof.shape[0] == 1):
                dof=np.zeros((J1+1))+dof
            dof[dof<1]=1
            dof = dofmin*np.sqrt(1 + (dof*dt/gamma_fac/ scale)**2 )
            dof[dof<dofmin]=dofmin
            for i in np.arange(1,J1+2):
                chisquare = chi2.ppf(siglvl,dof[i-1])/dof[i-1]
                signif[i-1] = fft_theor[i-1]*chisquare
        elif (sigtest == 2):
            s1 = dof[0]
            s2 = dof[1]
            avg = np.array(np.where((scale >= s1)&(scale < s2))).reshape(-1)
            navg = avg.shape[0]
            Savg = 1/np.sum(1/ scale[avg],axis=(0))
            Smid = np.exp((np.log(s1)+np.log(s2))/2.)
            dof = (dofmin*navg*Savg/Smid)*np.sqrt(1 + (navg*dj/dj0)**2)
            fft_theor = Savg*np.sum(fft_theor[avg]/ scale[avg])
            chisquare = chi2.ppf(siglvl,dof)/dof
            signif = (dj*dt/Cdelta/Savg)*fft_theor*chisquare    
        return signif,fft_theor
    
    def calculate_wavelet(Y_reconstructed, dt, pad, dj, s0, j1, mother):
        variance = Y_reconstructed.std() * Y_reconstructed.std()
        STD_Y_reconstructed = ((Y_reconstructed - Y_reconstructed.mean()) / Y_reconstructed.std())

        wave, period, scale, coi = WaveletAnalyzer.wavelet(STD_Y_reconstructed, dt, pad, dj, s0, j1, mother, 6)
        power = (np.abs(wave)) ** 2

        signif, fft_theor = WaveletAnalyzer.wave_signif(1, dt, scale, 0, lag1, 0.95, 0, mother, 6)
        sig95 = (signif.T).reshape((-1, 1)) * (np.zeros((n)) + 1).reshape((1, -1))
        sig95 = power / sig95

        global_ws = variance * (np.sum(power.T, axis=0) / n)
        dof = n - scale
        global_signif, _ = WaveletAnalyzer.wave_signif(variance, dt, scale, 1, lag1, 0.95, dof, mother, 6)
        avg = np.where((scale >= 2) & (scale < 8))
        Cdelta = 0.776
        scale_avg = (scale.T).reshape((-1, 1)) * (np.zeros((n)) + 1).reshape((1, -1))
        scale_avg = power / scale_avg
        scale_avg = variance * dj * dt / Cdelta * np.sum(scale_avg[avg, :], axis=(0, 1))
        scaleavg_signif, _ = WaveletAnalyzer.wave_signif(variance, dt, scale, 2, lag1, 0.95, np.array([2, 7.9]), mother, 6)
        
        return {
            'variance': variance,
            'STD_Y_reconstructed': STD_Y_reconstructed,
            'wave': wave,
            'period': period,
            'scale': scale,
            'coi': coi,
            'power': power,
            'sig95': sig95,
            'global_ws': global_ws,
            'dof': dof,
            'global_signif': global_signif,
            'avg': avg,
            'Cdelta': Cdelta,
            'scale_avg': scale_avg,
            'scaleavg_signif': scaleavg_signif
        }