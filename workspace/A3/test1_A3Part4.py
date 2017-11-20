import sys
sys.path.append('../../software/models/')
from dftModel import dftAnal, dftSynth
from scipy.signal import get_window
from A3Part4 import suppressFreqDFTmodel
import numpy as np
import matplotlib.pyplot as plt

# 40 Hz - 250 samples per cycle
# 100 Hz - 100 samples per cycle
# 200 Hz - 50 samples per cycle
# 1000 Hz = 10 samples per cycle

fs = 10000.0 # sampling rate
M = 511 # number of samples in input signal
fs = 10000.0 # sampling rate
T = 1/fs
nv = np.arange(M)

# Test case 1: For an input signal with 40 Hz, 100 Hz, 200 Hz, 1000 Hz components, yfilt will only contain
# 100 Hz, 200 Hz and 1000 Hz components. 

x = np.cos(2*np.pi*40.0*nv*T) + np.cos(2*np.pi*100.0*nv*T) + np.cos(2*np.pi*200.0*nv*T) + np.cos(2*np.pi*1000.0*nv*T)
w = get_window('hamming', M)
outputScaleFactor = sum(w)
N = 1024

mX, pX = dftAnal(x, w, N)
y = dftSynth(mX, pX, w.size)*outputScaleFactor

kX = np.arange(N/2 + 1) # frequency index up to N/2 + 1
fv = kX*fs/N # frequency spectrum
fmax = 70 # filter out frequencies below 70 Hz

fmaxIdx = np.argmax(np.ceil(fv) > 70) # find first index where frequency > 70
mXfilt = mX.copy()
mXfilt[:fmaxIdx+1] = -120 # set all magnitudes <= 70 Hz to -120 dB
# add 1 to fmaxIdx to capture one frequency above 70 Hz

yfilt = dftSynth(mXfilt, pX, w.size)*outputScaleFactor

#y, yfilt = suppressFreqDFTmodel(x, fs, N)
#mXfilt, pXfilt = dftAnal(yfilt, w, N)

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(3, sharex=True)
axarr[0].set_title('Input signal')
axarr[0].plot(x,label='input')
axarr[0].plot(y,label='unfiltered output (windowed)')
axarr[0].plot(yfilt, label='filtered output (windowed)')
axarr[0].set_ylabel('y')
axarr[0].legend()
axarr[1].plot(mX,label='unfiltered output')
axarr[1].plot(mXfilt, label='filtered output')
axarr[1].legend()
axarr[1].set_ylabel('Magnitude (dB)')
axarr[2].plot(pX)
axarr[2].set_ylabel('Phase (rad)')

plt.show(block = False)
