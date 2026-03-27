import cv2
import numpy as np
import sys
import time
import os
# -----------------------------
# 1. Check arguments
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python resize_gray.py input_image")
    sys.exit(1)

input_filename = sys.argv[1]
name, ext = os.path.splitext(input_filename)
output_filename = name + "_resized_gray" + ext
# -----------------------------
# 2. Read RGB image
# -----------------------------
image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
if image is None:
    print("Error: Could not read image:", input_filename)
    sys.exit(1)
H, W, C = image.shape
print(f"Original image size: {H} x {W} x {C}")
# -----------------------------
# 3. Resize image
# -----------------------------
# Reduce size by factor (e.g., 2)
scale_factor = 0.5
new_H = int(H * scale_factor)
new_W = int(W * scale_factor)
start = time.perf_counter()
resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
resize_time = time.perf_counter() - start
print(f"Resized image size: {new_H} x {new_W} x {C}")
print(f"Resize execution time: {resize_time:.6f} seconds")
# -----------------------------
# 4. Convert to grayscale (NumPy vectorized)
# -----------------------------
start = time.perf_counter()
resized_f = resized.astype(np.float32)
B = resized_f[:, :, 0]
G = resized_f[:, :, 1]
R = resized_f[:, :, 2]
gray = 0.114 * B + 0.587 * G + 0.299 * R
gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
grayscale_time = time.perf_counter() - start
print(f"Grayscale conversion execution time: {grayscale_time:.6f} seconds")
# -----------------------------
# 5. GFLOPS estimation
# -----------------------------
# 3 multiplications + 2 additions per pixel
flops = new_H * new_W * 5
gflops = flops / grayscale_time / 1e9
print(f"Estimated GFLOPS: {gflops:.6f}")
# -----------------------------
# 6. Memory bandwidth estimation
# -----------------------------
# 3 channels read + 1 channel write, float32 = 4 bytes
bytes_moved = new_H * new_W * 4 * 4
bandwidth = bytes_moved / grayscale_time / 1e9
print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
# -----------------------------
# 7. Save grayscale image
# -----------------------------
cv2.imwrite(output_filename, gray_uint8)
print("Resized grayscale image saved as:", output_filename)

