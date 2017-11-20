import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import math
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

# Test case 1
#M = 100
#window = 'blackmanharris'

# Test case 2
#M = 120
#window = 'boxcar'

# Test case 3
M = 256
window = 'hamming'

w = get_window(window, M)         # get the window 

N = 8*M
w_zeropad = np.zeros(N)
w_zeropad[:M] = w
X = fft(w_zeropad) # FFT of zero-padded window

absX = abs(X) # magnitude of DFT
absX[absX<eps] = eps # add small number to magnitude so you can take the log
mX = 20*np.log10(absX) # magnitude of DFT in dB
mX = fftshift(mX) # center around zero frequency

# Collect magnitudes of main lobe into numpy array

mX_mainLobe = np.array([])

# Start with the positive frequencies
maxIdx = np.argmax(mX)
mX_mainLobe = np.array(mX[maxIdx-1:maxIdx+2])

upperIdx = maxIdx + 2
while mX[upperIdx] < mX[upperIdx - 1]:
    mX_mainLobe = np.append(mX_mainLobe, mX[upperIdx])
    upperIdx +=1
    
# Now add negative frequencies
lowerIdx = maxIdx - 2
while mX[lowerIdx] < mX[lowerIdx +1]:
    mX_mainLobe = np.append(mX[lowerIdx],mX_mainLobe)
    lowerIdx -=1

f, axarr = plt.subplots(3)
axarr[0].plot(np.arange(-M/2, M/2), w)
axarr[0].set_title('Blackman-Harris, M = 100, N = 8*M')
axarr[0].set_ylabel('w[n]')
axarr[0].set_xlabel('n')
axarr[1].plot(np.arange(-N/2, N/2), mX)
axarr[1].set_ylabel('mX (dB)')
axarr[1].set_xlabel('k')
axarr[2].plot(np.arange(mX_mainLobe.size)-mX_mainLobe.size/2, mX_mainLobe)
axarr[2].set_ylabel('mX (dB)')
axarr[2].set_xlabel('k')
plt.show(block = False)


