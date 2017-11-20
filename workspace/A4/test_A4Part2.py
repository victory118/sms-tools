import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF
eps = np.finfo(float).eps

'''
Test case 1: If you run your code using piano.wav file with 'blackman' window, M = 513, N = 2048 and 
H = 128, the output SNR values should be around: (67.57748352378475, 304.68394693221649).

inputFile = '/home/victor/sms-tools/sounds/piano.wav'
window = 'blackman'

M = 513
N = 2048
H = 128

Test case 2: If you run your code using sax-phrase-short.wav file with 'hamming' window, M = 512, 
N = 1024 and H = 64, the output SNR values should be around: (89.510506656299285, 306.18696700251388).

inputFile = '/home/victor/sms-tools/sounds/sax-phrase-short.wav'
window = 'hamming'

M = 512
N = 1024
H = 64

Test case 3: If you run your code using rain.wav file with 'hann' window, M = 1024, N = 2048 and 
H = 128, the output SNR values should be around: (74.631476225366825, 304.26918192997738).

inputFile = '/home/victor/sms-tools/sounds/rain.wav'
window = 'hann'

M = 1024
N = 2048
H = 128
'''

# computeSNR(inputFile, window, M, N, H)

inputFile = '/home/victor/sms-tools/sounds/rain.wav'
window = 'hann'

M = 1024
N = 2048
H = 128

(fs, x) = UF.wavread(inputFile)
w = get_window(window, M)

y = stft.stft(x, w, N, H) # output sound

'''
With the input signal and the obtained output, compute two different SNR values for the following cases:

1) SNR1: Over the entire length of the input and the output signals.
2) SNR2: For the segment of the signals left after discarding M samples from both the start and the 
end, where M is the analysis window length. Note that this computation is done after STFT analysis 
and synthesis.
'''

# SNR1
Esig1 = sum(abs(x)**2)
Enoise1 = sum(abs(x-y)**2)
SNR1 = 10*np.log10(Esig1/Enoise1)

# SNR2
Esig2 = sum(abs(x[M:-M])**2)
Enoise2 = sum(abs(x[M:-M] - y[M:-M])**2)
SNR2 = 10*np.log10(Esig2/Enoise2)

f, axarr = plt.subplots(3,sharex=True)
axarr[0].plot(x)
axarr[0].set_ylabel('x[n]')
axarr[0].set_xlabel('n')
axarr[1].plot(y)
axarr[1].set_ylabel('y[n]')
axarr[2].plot(x-y)
axarr[1].set_ylabel('x[n] - y[n]')
axarr[2].set_xlabel('n')
plt.show(block = False)
