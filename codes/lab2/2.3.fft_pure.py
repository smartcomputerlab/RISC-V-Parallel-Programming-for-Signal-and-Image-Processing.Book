# fft_pure.py
import matplotlib.pyplot as plt
import time
import math
import cmath
# ---------------------------
# Sampling parameters
# ---------------------------
fs = 1000          # Sampling frequency (Hz)
T = 1.0            # Signal duration (seconds)
N = int(fs * T)    # Number of samples
# ---------------------------
# Create time axis (replace np.linspace)
# ---------------------------
t = []
for n in range(N):
    t.append(n / fs)
# ---------------------------
# Generate signal (replace NumPy vector operations)
# ---------------------------
f1 = 50
f2 = 120
signal = []
for ti in t:
    value = math.sin(2 * math.pi * f1 * ti) + 0.5 * math.sin(2 * math.pi * f2 * ti)
    signal.append(value)
# ---------------------------
# Manual DFT (replace np.fft.fft)
# ---------------------------
def dft(x):
    N = len(x)
    X = []
    for k in range(N):
        s = 0j
        for n in range(N):
            angle = -2j * math.pi * k * n / N
            s += x[n] * cmath.exp(angle)
        X.append(s)
    return X

# ---------------------------
# Warm-up
# ---------------------------
dft(signal[:64])   # smaller size so it doesn't take too long
# ---------------------------
# Measure execution time
# ---------------------------
start = time.perf_counter()
fft_values = dft(signal)
end = time.perf_counter()
execution_time = end - start
print(f"DFT size: {N}")
print(f"Execution time: {execution_time:.6f} seconds")
# ---------------------------
# Estimate FLOPs (same formula used)
# ---------------------------
flops_est = 5 * N * math.log2(N)
gflops = flops_est / execution_time / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPS")
# ---------------------------
# Frequency axis (replace np.fft.fftfreq)
# ---------------------------
freqs = []
for k in range(N):
    freqs.append(k * fs / N)

# ---------------------------
# Magnitude (replace np.abs)
# ---------------------------
magnitude = []
for v in fft_values:
    magnitude.append(abs(v))

# ---------------------------
# Plot time-domain signal
# ---------------------------
plt.figure(figsize=(10,4))
plt.plot(t[:200], signal[:200])
plt.title("Input Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
# ---------------------------
# Plot frequency spectrum
# ---------------------------
plt.figure(figsize=(10,4))
plt.plot(freqs[:N//2], magnitude[:N//2])
plt.title("Frequency Spectrum (DFT - Pure Python)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

