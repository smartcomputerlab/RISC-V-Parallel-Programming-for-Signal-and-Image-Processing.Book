# audio_features_np.py
import numpy as np
import matplotlib.pyplot as plt
import time
# ---------------------------
# 1. Generate test audio signal (float32)
# ---------------------------
fs = 16000             # Sampling frequency (Hz)
duration = 2.0         # seconds
t = np.linspace(0, duration, int(fs*duration), endpoint=False, dtype=np.float32)
# Example signal: 440 Hz sine wave + noise
freq = 440
audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32) + \
        0.05 * np.random.randn(len(t)).astype(np.float32)
# ---------------------------
# 2. Frame the signal into a matrix
# ---------------------------
frame_size = 1024
hop_size = 512
num_frames = 1 + (len(audio) - frame_size) // hop_size
frames = np.zeros((num_frames, frame_size), dtype=np.float32)
for i in range(num_frames):
    start = i * hop_size
    frames[i] = audio[start:start+frame_size]
# ---------------------------
# 3. Feature extraction with timing
# ---------------------------
start = time.perf_counter()
# Short-Time Energy
energy = np.sum(frames**2, axis=1)
# Zero-Crossing Rate
zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / (2*frame_size)
# Mean and Std per frame
mean_frame = np.mean(frames, axis=1)
std_frame = np.std(frames, axis=1)
end = time.perf_counter()
elapsed_time = end - start
print(f"Feature extraction execution time: {elapsed_time:.6f} seconds")
# ---------------------------
# 4. Estimate FLOPs and GFLOPS
# ---------------------------
flops_ste = num_frames * 2 * frame_size           # sum of squares
flops_zcr = num_frames * 3 * frame_size           # diff, sign, abs
flops_mean = num_frames * 2 * frame_size          # sum + divide
flops_std = num_frames * 4 * frame_size           # (x-mean)^2 + sum + div + sqrt
total_flops = flops_ste + flops_zcr + flops_mean + flops_std
gflops = total_flops / elapsed_time / 1e9
print(f"Estimated performance: {gflops:.2f} GFLOPS")
# ---------------------------
# 5. Plot results (first 200 frames for clarity)
# ---------------------------
time_frames = np.arange(num_frames) * hop_size / fs
plt.figure(figsize=(12,8))
plt.subplot(4,1,1)
plt.plot(t[:2000], audio[:2000])
plt.title("Audio Signal (first 2000 samples)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplot(4,1,2)
plt.plot(time_frames[:200], energy[:200])
plt.title("Short-Time Energy")
plt.xlabel("Time [s]")
plt.ylabel("Energy")
plt.grid(True)
plt.subplot(4,1,3)
plt.plot(time_frames[:200], zcr[:200])
plt.title("Zero-Crossing Rate")
plt.xlabel("Time [s]")
plt.ylabel("ZCR")
plt.grid(True)
plt.subplot(4,1,4)
plt.plot(time_frames[:200], std_frame[:200])
plt.title("Frame Standard Deviation")
plt.xlabel("Time [s]")
plt.ylabel("Std")
plt.grid(True)
plt.tight_layout()
plt.show()

