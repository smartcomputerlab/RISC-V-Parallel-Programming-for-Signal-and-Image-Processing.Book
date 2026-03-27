import cv2
import numpy as np
import sys
import time
import os
# -----------------------------
# 1. Check arguments
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python rotate_image_perf.py input_image")
    sys.exit(1)
input_filename = sys.argv[1]
name, ext = os.path.splitext(input_filename)
output_filename = name + "_rotated" + ext
# -----------------------------
# 2. Read RGB image
# -----------------------------
image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
if image is None:
    print("Error: Could not read image:", input_filename)
    sys.exit(1)
H, W, C = image.shape
print(f"Image size: {H} x {W} x {C}")
# Convert to float32 to simulate floating-point operations
image_f = image.astype(np.float32)
# -----------------------------
# 3. Rotate image (vectorized)
# -----------------------------
start = time.perf_counter()
# Rotate 90° clockwise: transpose + reverse rows
rotated = np.transpose(image_f, (1, 0, 2))[:, ::-1, :]
end = time.perf_counter()
elapsed = end - start
print(f"Rotation execution time: {elapsed:.6f} seconds")
# -----------------------------
# 4. GFLOPS estimation
# -----------------------------
# Count each pixel move as 1 operation
num_elements = H * W * C
# Transpose + row flip ≈ 2 operations per element
flops = num_elements * 2
gflops = flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPS")
# -----------------------------
# 5. Memory bandwidth estimation
# -----------------------------
# Read original image + write rotated image, float32 = 4 bytes per channel
bytes_moved = num_elements * 2 * 4  # read + write
bandwidth = bytes_moved / elapsed / 1e9
print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
# -----------------------------
# 6. Save rotated image
# -----------------------------
# Convert back to uint8 for saving
cv2.imwrite(output_filename, np.clip(rotated, 0, 255).astype(np.uint8))
print("Rotated image saved as:", output_filename)

