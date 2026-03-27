import numpy as np
import matplotlib.pyplot as plt
import time
# ---------------------------
# 1. Generate large input signal
# ---------------------------
fs = 1000               # Sampling frequency (Hz)
T = 10_000              # Duration in seconds
N = int(fs * T)         # Number of samples (~10M)
t = np.linspace(0, T, N, endpoint=False, dtype=np.float32)
# Signal: 5 Hz sine wave + high-frequency noise
freq = 5
signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
signal += 0.05 * np.random.randn(N).astype(np.float32)  # noise
# ---------------------------
# 2. Design FIR low-pass filter
# ---------------------------
num_taps = 101            # Filter length
cutoff = 10               # Cutoff frequency in Hz
fc = cutoff / (fs / 2)    # Normalized cutoff
h = np.sinc(2 * fc * (np.arange(num_taps) - (num_taps - 1) / 2)).astype(np.float32)
h *= np.hamming(num_taps).astype(np.float32)
h /= np.sum(h)
# ---------------------------
# 3. Warm-up convolution
# ---------------------------
np.convolve(signal[:1024], h, mode='valid')
# ---------------------------
# 4. FIR filtering with timing
# ---------------------------
start = time.perf_counter()
filtered_signal = np.convolve(signal, h, mode='same')
end = time.perf_counter()
elapsed_time = end - start
print(f"FIR filter size: {num_taps} taps")
print(f"Input signal size: {N}")
print(f"Execution time: {elapsed_time:.6f} seconds")
# ---------------------------
# 5. Estimate FLOPs and GFLOPS
# ---------------------------
# FLOPs per output sample: num_taps mult + num_taps add ≈ 2*num_taps
flops = 2 * N * num_taps
gflops = flops / elapsed_time / 1e9
print(f"Estimated performance: {gflops:.2f} GFLOPS")
# ---------------------------
# 6. Plot a small portion of the signal
# ---------------------------
plt.figure(figsize=(12,4))
plt.plot(t[:2000], signal[:2000], label='Original Signal')
plt.plot(t[:2000], filtered_signal[:2000], label='Filtered Signal', linewidth=2)
plt.title("FIR Filtering Example (10M+ Samples)")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude")
plt.legend(); plt.grid(True); plt.show()

