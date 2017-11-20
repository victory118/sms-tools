import numpy as np
from scipy.signal import get_window
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import dftModel as DFT
import utilFunctions as UF
import matplotlib.pyplot as plt

"""
Test case 1: If you run your code with inputFile = '../../sounds/sine-490.wav', f = 490.0 Hz, the optimal
values are M = 1101, N = 2048, fEst = 489.963 and the freqency estimation error is 0.037.

Test case 2: If you run your code with inputFile = '../../sounds/sine-1000.wav', f = 1000.0 Hz, the optimal
values are M = 1101, N = 2048, fEst = 1000.02 and the freqency estimation error is 0.02.

Test case 3: If you run your code with inputFile = '../../sounds/sine-200.wav', f = 200.0 Hz, the optimal
values are M = 1201, N = 2048, fEst = 200.038 and the freqency estimation error is 0.038.
"""

inputFile = '../../sounds/sine-490.wav'
f = 490.0
inputFile = '../../sounds/sine-1000.wav'
f = 1000.0
inputFile = '../../sounds/sine-200.wav'
f = 200.0
window = 'blackman'
t = -40 # magnitude threshold in dB for peak picking

(fs, x) = UF.wavread(inputFile)

centerIdx = int(0.5*fs) # get the index of the sample at 0.5 sec.

k = 0
fEst = 0.0

while abs(fEst-f) >= 0.05:
    k = k + 1
    
    # Increase window size(M)
    M = 100*k + 1
    
    # Take FFT size(N) to be smallest power of 2 larger than M
    N = int(2**math.ceil(math.log(M,2)))
    
    w = get_window(window, M)
    
    mX, pX = DFT.dftAnal(x[centerIdx-k*50:centerIdx+k*50+1], w, N)

    # Since the DFT size N is a power of 2 and hence even, there are N/2 + 1 independent frequency components
    ploc = UF.peakDetection(mX, t)

    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
    
    fEst = iploc[0]*fs/N
    
    






