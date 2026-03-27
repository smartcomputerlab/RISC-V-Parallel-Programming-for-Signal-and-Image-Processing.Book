import cv2
import numpy as np
import sys, time, os
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
# Convert to float32
image_f = image.astype(np.float32)
# -----------------------------
# 3. Allocate rotated image
# -----------------------------
rotated = np.zeros((W, H, C), dtype=np.float32)
# -----------------------------
# 4. Rotate image manually
# -----------------------------
start = time.perf_counter()
for y in range(H):
    for x in range(W):
        for c in range(C):
            rotated[x, H - 1 - y, c] = image_f[y, x, c]
end = time.perf_counter()
elapsed = end - start
print(f"Rotation execution time: {elapsed:.6f} seconds")
# -----------------------------
# 5. GFLOPS estimation
# -----------------------------
num_elements = H * W * C
# Approximate operations per element
flops = num_elements * 2
gflops = flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPS")
# -----------------------------
# 6. Memory bandwidth estimation
# -----------------------------
bytes_moved = num_elements * 2 * 4
bandwidth = bytes_moved / elapsed / 1e9
print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
# -----------------------------
# 7. Save rotated image
# -----------------------------
cv2.imwrite(output_filename, np.clip(rotated, 0, 255).astype(np.uint8))
print("Rotated image saved as:", output_filename)

