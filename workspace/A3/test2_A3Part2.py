from A3Part2 import optimalZeropad
import numpy as np
import matplotlib.pyplot as plt

M = 210
fs = 10000.0
f = 250.0
T = 1/fs
nv = np.arange(M)

x = np.cos(2*np.pi*f*nv*T)

mX = optimalZeropad(x, fs, f)

plt.plot(mX)
plt.show()
