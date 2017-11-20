from A3Part1 import minimizeEnergySpreadDFT
import numpy as np
import matplotlib.pyplot as plt

N = 1000
fs = 48000.0
f1 = 300.0
f2 = 800.0
T = 1.0/fs
nv = np.arange(N)

x = np.cos(2*np.pi*f1*nv*T) + np.cos(2*np.pi*f2*nv*T)

mX = minimizeEnergySpreadDFT(x, fs, f1, f2)

plt.plot(mX)
plt.show()
