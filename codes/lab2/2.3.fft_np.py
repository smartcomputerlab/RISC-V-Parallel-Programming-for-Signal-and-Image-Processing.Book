#fft_np.py
import numpy as np
import matplotlib.pyplot as plt
import time
import math
# ---------------------------
# Sampling parameters
# ---------------------------
fs = 1000          # Sampling frequency (Hz)
T = 1.0            # Signal duration (seconds)
N = int(fs * T)    # Number of samples
t = np.linspace(0, T, N, endpoint=False)
# ---------------------------
# Generate signal
# ---------------------------
f1 = 50    # 50 Hz
f2 = 120   # 120 Hz
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
# ---------------------------
# Warm-up FFT
# ---------------------------
np.fft.fft(signal[:1024])
# ---------------------------
# Measure FFT execution time
# ---------------------------
start = time.perf_counter()
fft_values = np.fft.fft(signal)
end = time.perf_counter()
execution_time = end - start
print(f"FFT size: {N}")
print(f"Execution time: {execution_time:.6f} seconds")
# ---------------------------
# Estimate FFT FLOPs
# ---------------------------
flops_est = 5 * N * math.log2(N)        # Approximate operations
gflops = flops_est / execution_time / 1e9
print(f"Estimated performance: {gflops:.2f} GFLOPS")
# ---------------------------
# Frequency axis and magnitude
# ---------------------------
freqs = np.fft.fftfreq(N, 1/fs)
magnitude = np.abs(fft_values)
# ---------------------------
# Plot time-domain signal
# ---------------------------
plt.figure(figsize=(10,4))
plt.plot(t[:200], signal[:200])   # plot first 200 samples for clarity
plt.title("Input Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
# ---------------------------
# Plot frequency spectrum
# ---------------------------
plt.figure(figsize=(10,4))
plt.plot(freqs[:N//2], magnitude[:N//2])
plt.title("Frequency Spectrum (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

