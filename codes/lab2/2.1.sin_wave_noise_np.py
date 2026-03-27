import numpy as np
import matplotlib.pyplot as plt
fs = 1000
t = np.linspace(0, 1, fs)
signal = np.sin(2*np.pi*5*t)
noise = 0.2 * np.random.randn(len(signal))
noisy_signal = signal + noise
plt.plot(t, noisy_signal)
plt.title("5 Hz sine signal")
plt.show()
