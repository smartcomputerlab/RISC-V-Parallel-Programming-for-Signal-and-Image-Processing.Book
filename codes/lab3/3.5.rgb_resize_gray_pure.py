# resize_gray_pure_python.py
import cv2
import sys
import time
import os
# -----------------------------
# 1. Check arguments
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python resize_gray_pure_python.py input_image")
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
# 3. Resize image (pure Python)
# -----------------------------
scale_factor = 0.5
new_H = int(H * scale_factor)
new_W = int(W * scale_factor)
start = time.perf_counter()
# Create resized image
resized = [[[0,0,0] for _ in range(new_W)] for _ in range(new_H)]
for y in range(new_H):
    for x in range(new_W):
        # Corresponding pixel in original image
        src_y = int(y / scale_factor)
        src_x = int(x / scale_factor)
        resized[y][x] = image[src_y][src_x]

resize_time = time.perf_counter() - start
print(f"Resized image size: {new_H} x {new_W} x {C}")
print(f"Resize execution time: {resize_time:.6f} seconds")
# -----------------------------
# 4. Convert to grayscale (pure Python)
# -----------------------------
start = time.perf_counter()
gray = [[0 for _ in range(new_W)] for _ in range(new_H)]
for y in range(new_H):
    for x in range(new_W):
        # OpenCV loads BGR
        B = float(resized[y][x][0])
        G = float(resized[y][x][1])
        R = float(resized[y][x][2])
        gray_value = 0.114 * B + 0.587 * G + 0.299 * R
        # Clamp to 0..255
        if gray_value < 0:
            gray_value = 0
        elif gray_value > 255:
            gray_value = 255
        gray[y][x] = int(gray_value)

grayscale_time = time.perf_counter() - start
print(f"Grayscale conversion execution time: {grayscale_time:.6f} seconds")
# -----------------------------
# 5. GFLOPS estimation
# -----------------------------
flops = new_H * new_W * 5
gflops = flops / grayscale_time / 1e9
print(f"Estimated GFLOPS: {gflops:.6f}")
# -----------------------------
# 6. Memory bandwidth estimation
# -----------------------------
bytes_moved = new_H * new_W * 4 * 4
bandwidth = bytes_moved / grayscale_time / 1e9
print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
# -----------------------------
# 7. Save grayscale image
# -----------------------------
# Convert Python list to OpenCV image
import numpy as np
gray_image = np.array(gray, dtype=np.uint8)
cv2.imwrite(output_filename, gray_image)
print("Grayscale image saved as:", output_filename)

