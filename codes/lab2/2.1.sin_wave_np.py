import numpy as np
import matplotlib.pyplot as plt
fs = 1000
t = np.linspace(0, 1, fs)
signal = np.sin(2*np.pi*5*t)
plt.plot(t, signal)
plt.title("5 Hz sine signal")
plt.show()

