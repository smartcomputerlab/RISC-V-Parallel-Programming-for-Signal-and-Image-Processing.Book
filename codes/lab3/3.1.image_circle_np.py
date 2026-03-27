# image_circle_performance.py
import numpy as np
import cv2
import time
# -------------------------------------------------
# 1. Image parameters
# -------------------------------------------------
H, W = 512, 512
radius = 64
center = (W // 2, H // 2)
num_pixels = H * W
# -------------------------------------------------
# 2. Vectorized benchmark (NumPy)
# -------------------------------------------------
start_vec = time.perf_counter()
# Create red image
image_vec = np.zeros((H, W, 3), dtype=np.uint8)
image_vec[:, :, 2] = 255
# Draw white circle using mask
Y, X = np.ogrid[:H, :W]
mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2
image_vec[mask] = 255
end_vec = time.perf_counter()
time_vec = end_vec - start_vec
print(f"[Vectorized] Execution time: {time_vec:.6f} s")
# -------------------------------------------------
# 3. Scalar benchmark (Python loops)
# -------------------------------------------------
start_s = time.perf_counter()
image_s = np.zeros((H, W, 3), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        image_s[i, j, 2] = 255
        if (i - center[1])**2 + (j - center[0])**2 <= radius**2:
            image_s[i, j, :] = 255

end_s = time.perf_counter()
time_s = end_s - start_s
print(f"[Scalar] Execution time: {time_s:.6f} s")
# -------------------------------------------------
# 4. Speedup
# -------------------------------------------------
speedup = time_s / time_vec
print(f"Estimated speedup (Vectorized vs Scalar): {speedup:.2f}x")
# -------------------------------------------------
# 5. FLOP estimation
# -------------------------------------------------
# Circle equation operations per pixel:
# (x-cx)^2 + (y-cy)^2 <= r^2
# Operations:
# subtraction (2)
# square (2)
# addition (1)
# comparison (1)
flops_per_pixel = 6
total_flops = num_pixels * flops_per_pixel
gflops_vec = total_flops / time_vec / 1e9
gflops_scalar = total_flops / time_s / 1e9
print("\n--- Performance estimation ---")
print(f"Total FLOPs: {total_flops:.2e}")
print(f"Vectorized performance: {gflops_vec:.3f} GFLOPs")
print(f"Scalar performance: {gflops_scalar:.6f} GFLOPs")
# -------------------------------------------------
# 6. Memory bandwidth estimation
# -------------------------------------------------
# Memory accesses per pixel
# read X,Y + write image
bytes_per_pixel = 3   # RGB output
total_bytes = num_pixels * bytes_per_pixel
bandwidth_vec = total_bytes / time_vec / 1e9
bandwidth_scalar = total_bytes / time_s / 1e9
print("\n--- Memory bandwidth estimation ---")
print(f"Total memory moved: {total_bytes/1e6:.2f} MB")
print(f"Vectorized bandwidth: {bandwidth_vec:.3f} GB/s")
print(f"Scalar bandwidth: {bandwidth_scalar:.6f} GB/s")
# -------------------------------------------------
# 7. Save vectorized image
# -------------------------------------------------
cv2.imwrite("red_circle_vectorized.png", image_vec)

