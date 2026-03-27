import numpy as np
import matplotlib.pyplot as plt
import time
import math
# -----------------------------
# 1. Generate synthetic audio
# -----------------------------
fs = 16000
duration = 2.0
N = int(fs * duration)
t = np.linspace(0, duration, N, endpoint=False, dtype=np.float32)
audio = (0.6*np.sin(2*np.pi*440*t) +
         0.3*np.sin(2*np.pi*1000*t) +
         0.05*np.random.randn(N)).astype(np.float32)
# -----------------------------
# 2. Frame parameters
# -----------------------------
frame_size =1024
hop_size = 512
num_frames = (N - frame_size) // hop_size
window = np.hanning(frame_size).astype(np.float32)
freqs = np.fft.rfftfreq(frame_size, 1/fs)
print("Extracting spectral centroid...")
print("Frames:", num_frames)
# -----------------------------
# 3. Processing with timing
# -----------------------------
start = time.perf_counter()
centroids = []
for i in range(num_frames):
    start_idx = i * hop_size
    frame = audio[start_idx:start_idx+frame_size]
    windowed = frame * window
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    centroids.append(centroid)
end = time.perf_counter()
elapsed = end - start
centroids = np.array(centroids)
print(f"Execution time: {elapsed:.6f} seconds")
# -----------------------------
# 4. FLOP estimation
# -----------------------------
Nf = frame_size
flops_per_frame = (
    Nf +                                  # window multiply
    5 * Nf * math.log2(Nf) +              # FFT approx
    3 * (Nf/2) +                          # magnitude
    2 * (Nf/2) +                          # centroid mult + add
    1                                      # final division
)
total_flops = flops_per_frame * num_frames
gflops = total_flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.2f} GFLOPS")
# -----------------------------
# 5. Plot result
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(centroids)
plt.title("Spectral Centroid Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.show()

