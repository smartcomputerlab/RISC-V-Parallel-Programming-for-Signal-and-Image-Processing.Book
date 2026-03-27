import numpy as np
import matplotlib.pyplot as plt
# -----------------------------
# 1. Sampling parameters
# -----------------------------
fs = 1000                 # Sampling frequency (Hz)
T = 1.0                   # Signal duration (seconds)
N = int(fs * T)           # Number of samples
t = np.linspace(0, T, N, endpoint=False)
# -----------------------------
# 2. Generate signal
# -----------------------------
f1 = 50                   # 50 Hz component
f2 = 120                  # 120 Hz component
signal = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
# -----------------------------
# 3. Compute FFT
# -----------------------------
fft_values = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, 1/fs)
magnitude = np.abs(fft_values)
# -----------------------------
# 4. Plot time-domain signal
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(t[:200], signal[:200])
plt.title("Input Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
# -----------------------------
# 5. Plot frequency spectrum
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(freqs[:N//2], magnitude[:N//2])
plt.title("FFT Spectrum (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

