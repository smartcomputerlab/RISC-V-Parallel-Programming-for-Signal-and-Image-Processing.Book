import numpy as np
import matplotlib.pyplot as plt
# -----------------------------
# 1. Sampling parameters
# -----------------------------
fs = 1000               # Sampling frequency in Hz
T = 1.0                 # Duration in seconds
N = int(fs * T)         # Number of samples
t = np.linspace(0, T, N, endpoint=False)
# -----------------------------
# 2. Generate signal
# -----------------------------
freq = 5                # 5 Hz sine wave
signal = np.sin(2 * np.pi * freq * t) + 0.1*np.random.randn(N)
# -----------------------------
# 3. Integrate signal
# -----------------------------
dt = 1/fs
integral_signal = np.cumsum(signal) * dt
# -----------------------------
# 4. Combined plot
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(t, signal, label="Original Signal")
plt.plot(t, integral_signal, color='orange', label="Integrated Signal")
plt.title("Original Signal and Its Integral")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

