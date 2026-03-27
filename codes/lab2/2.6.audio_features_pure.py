# audio_feature_gflops_pure_python.py
import matplotlib.pyplot as plt
import time
import math
import random
# ---------------------------
# 1. Generate test audio signal
# ---------------------------
fs = 16000
duration = 2.0
N = int(fs * duration)
# Time axis
t = [n / fs for n in range(N)]
# Example signal: 440 Hz sine wave + noise
freq = 440
audio = []
for ti in t:
    value = 0.5 * math.sin(2 * math.pi * freq * ti) + 0.05 * random.gauss(0,1)
    audio.append(value)
# ---------------------------
# 2. Frame the signal
# ---------------------------
frame_size = 1024
hop_size = 512
num_frames = 1 + (len(audio) - frame_size) // hop_size
frames = []
for i in range(num_frames):
    start = i * hop_size
    frame = audio[start:start+frame_size]
    frames.append(frame)
# ---------------------------
# 3. Feature extraction with timing
# ---------------------------
start = time.perf_counter()
energy = []
zcr = []
mean_frame = []
std_frame = []
for frame in frames:
    # Short-Time Energy
    e = 0.0
    for x in frame:
        e += x * x
    energy.append(e)
    # Zero-Crossing Rate
    crossings = 0
    for i in range(1, len(frame)):
        if frame[i-1] * frame[i] < 0:
            crossings += 1
    zcr.append(crossings / frame_size)
    # Mean
    m = sum(frame) / frame_size
    mean_frame.append(m)
    # Standard deviation
    s = 0.0
    for x in frame:
        s += (x - m) ** 2
    std_frame.append(math.sqrt(s / frame_size))
end = time.perf_counter()

elapsed_time = end - start
print(f"Feature extraction execution time: {elapsed_time:.6f} seconds")
# ---------------------------
# 4. Estimate FLOPs and GFLOPS
# ---------------------------
flops_ste = num_frames * 2 * frame_size
flops_zcr = num_frames * 3 * frame_size
flops_mean = num_frames * 2 * frame_size
flops_std = num_frames * 4 * frame_size
total_flops = flops_ste + flops_zcr + flops_mean + flops_std
gflops = total_flops / elapsed_time / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPS")
# ---------------------------
# 5. Plot results (first 200 frames)
# ---------------------------
time_frames = [(i * hop_size) / fs for i in range(num_frames)]
plt.figure(figsize=(12,8))
plt.subplot(4,1,1)
plt.plot(t[:2000], audio[:2000])
plt.title("Audio Signal (first 2000 samples)")
plt.grid(True)
plt.subplot(4,1,2)
plt.plot(time_frames[:200], energy[:200])
plt.title("Short-Time Energy")
plt.grid(True)
plt.subplot(4,1,3)
plt.plot(time_frames[:200], zcr[:200])
plt.title("Zero-Crossing Rate")
plt.grid(True)
plt.subplot(4,1,4)
plt.plot(time_frames[:200], std_frame[:200])
plt.title("Frame Standard Deviation")
plt.grid(True)
plt.tight_layout()
plt.show()

