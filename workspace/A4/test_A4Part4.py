import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

'''
Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.
'''

# computeEngEnv(inputFile, window, M, N, H)

inputFile = '/home/victor/sms-tools/sounds/piano.wav'
window = 'blackman'

M = 513
N = 2048
H = 128

(fs, x) = UF.wavread(inputFile)
w = get_window(window, M)

xmX, xpX = stft.stftAnal(x, w, N, H)
# xmX.shape = (1325, 513), where x.size/H = 1325 is the number of frames
# N-M = 511, so each frame is padded with 511 zeros
# the zero padded fftbuffer should have the first 257 samples of x, then 511 zeros, followed by the remaining 256 samples of x

# Find bins corresponding to 0 < f < freqMid and freqMid < f < freqMax
K = xmX.shape[0] # number of frames
freqMid = 3000.0
freqMax = 10000.0
endBinLowFreq = freqMid/(fs/float(N))

if endBinLowFreq == np.floor(endBinLowFreq):
    startBinHighFreq = endBinLowFreq + 1
    endBinLowFreq -=1
else:
    endBinLowFreq = np.floor(endBinLowFreq)
    startBinHighFreq = endBinLowFreq + 1
    
endBinHighFreq = freqMax/(fs/float(N))

if endBinHighFreq == np.floor(endBinHighFreq):
    endBinHighFreq -=1
else:
    endBinHighFreq = np.floor(endBinHighFreq)

endBinLowFreq = int(endBinLowFreq)    
startBinHighFreq = int(startBinHighFreq)
endBinHighFreq = int(endBinHighFreq)

# For each frame, calculate the energy in each frequency range (energy envelope) in dB

engEnv = np.zeros([K,2])

for ii in np.arange(K):
    # xmX is in units of dB, so we have to convert it to gain before calculating the energy
    engEnv[ii,0] = 10*np.log10(sum((10**(xmX[ii,1:endBinLowFreq+1]/20))**2))
    engEnv[ii,1] = 10*np.log10(sum((10**(xmX[ii,startBinHighFreq:endBinHighFreq+1]/20))**2))

ODF = np.zeros(engEnv.shape)

for ii in np.arange(K-1):
    # Calculate low frequency range ODF
    ODF_low = engEnv[ii+1,0] - engEnv[ii,0]
    if ODF_low > 0:
        ODF[ii,0] = ODF_low
    else:
        ODF[ii,0] = 0
        
    # Calculate high frequency range ODF
    ODF_high = engEnv[ii+1,1] - engEnv[ii,1]
    if ODF_high > 0:
        ODF[ii,1] = ODF_high
    else:
        ODF[ii,1] = 0

# Test cases are in loadTestCases.py
# In the ipython shell type: >> import loadTestCases
# >> testcase = loadTestCases.load(1) for test case 1
# There are 4 test cases. Test case 1 in the assignment is equivalent to loadTestCases.load(3)
# The output testcase is a dictionary type. To access the numpy array, type >> testcase['output']

plt.figure(1, figsize=(9.5, 6))

plt.subplot(211)
# numFrames = int(mX[:,0].size)
frmTime = H*np.arange(K)/float(fs)                             
binFreq = np.arange(N/2+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(xmX))
# plt.title('mX (piano.wav), M=1001, N=1024, H=256')
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.autoscale(tight=True)

plt.subplot(212)
# numFrames = int(pX[:,0].size)
# frmTime = H*np.arange(numFrames)/float(fs)                             
# binFreq = np.arange(N/2+1)*float(fs)/N                         
# plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(pX),axis=0))
plt.plot(frmTime, ODF[:,0], label='ODF low')
plt.plot(frmTime, ODF[:,1], label='ODF high')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude (dB)')
plt.autoscale(tight=True)

plt.tight_layout()
plt.show(block=False)

