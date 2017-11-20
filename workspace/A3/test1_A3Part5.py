import sys
sys.path.append('../../software/models/')
from dftModel import dftAnal, dftSynth
from scipy.signal import get_window
from A3Part5 import zpFFTsizeExpt
import numpy as np
import matplotlib.pyplot as plt

f = 110.0 # input sinusoid frequency [Hz]
fs = 1000.0 # sampling frequency [Hz]
T = 1/fs # sampling interval [sec]
    
M = 512 # number of samples of input signal
nv = np.arange(M) # sample indices
x = np.cos(2*np.pi*f*nv*T) # input signal

# Case-1: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 256
# Case-2: Input signal x (512 samples), window w2 (512 samples), and FFT size of 512
# Case-3: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 512 (Implicitly does a 
#         zero-padding of xseg by 256 samples)

mX1_80, mX2_80, mX3_80 = zpFFTsizeExpt(x, fs)

# M = len(x)/2
xseg = x[:M/2]
w1 = get_window('hamming',M/2)
w2 = get_window('hamming',M)
x3 = np.zeros(M)
x3[:256] = xseg*w1
x4 = np.zeros(2*M)
x4[:256] = xseg*w1

N4 = 1024 # FFT size
mX4, pX4 = dftAnal(xseg, w1, N4)
mX4_160 = mX4[:160]
'''
f, axarr = plt.subplots(2)
axarr[0].plot(xseg*w1, label='xseg*w1 (256)')
axarr[0].plot(x*w2, label='x*w2 (512)')
axarr[0].plot(x3, label='xseg*w1 (512 zero-padded)')
axarr[0].plot(x4, label='xseg*w1 (1024 zero-padded)')
axarr[0].set_xlabel('n')
axarr[0].set_ylabel('Input signal')
axarr[0].legend()
axarr[1].plot(np.arange(80)*fs/256, mX1_80, label='xseg (256), w1 (256), FFT (256)')
axarr[1].plot(np.arange(80)*fs/512, mX2_80, label='x (512), w2 (512), FFT (512)')
axarr[1].plot(np.arange(80)*fs/512, mX3_80, label='xseg (256), w1 (256), FFT_zp (512)')
axarr[1].plot(np.arange(200)*fs/1024, mX4_200, label='xseg (256), w1 (256), FFT_zp (1024)')
axarr[1].legend()
axarr[1].set_ylabel('Magnitude [dB]')
axarr[1].set_xlabel('Frequency [Hz]')
'''
plt.plot(np.arange(40)*fs/256, mX1_80[:40], label='xseg (256), w1 (256), FFT (256)')
plt.plot(np.arange(80)*fs/512, mX2_80, label='x (512), w2 (512), FFT (512)')
plt.plot(np.arange(80)*fs/512, mX3_80, label='xseg (256), w1 (256), FFT_zp (512)')
plt.plot(np.arange(160)*fs/1024, mX4_160, label='xseg (256), w1 (256), FFT_zp (1024)')
plt.legend(loc='best')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')

plt.show(block=False)

# Results
# Case 1 has the lowest spectral/frequency resolution (accuracy) with 256 samples.
# Case 2 has the narrowest/sharpest peak and double the frequency resolution of Case 1 so is the most accurate.
# Case 3 has the same frequency resolution/accuracy as Case 1 because it has the same amount of information. It    basically interpolates the FFT from Case 1 to make it smoother and makes the amplitude estimates more accurate for any resolvable signal components. Hence, you can see the peak around 110 Hz is smoother compared to Case 1.

