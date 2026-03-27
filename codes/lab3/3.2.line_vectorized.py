import numpy as np
import cv2
import time
# -----------------------------
# 1. Image parameters
# -----------------------------
H, W = 512, 512
line_thickness = 2
num_pixels = H * W
# -----------------------------
# 2. Create red image
# -----------------------------
start_time = time.perf_counter()
# Initialize red image (BGR format)
image = np.zeros((H, W, 3), dtype=np.uint8)
image[:, :, 2] = 255   # Red channel
# -----------------------------
# 3. Generate random line coordinates
# -----------------------------
y0, x0 = np.random.randint(0, H), np.random.randint(0, W)
y1, x1 = np.random.randint(0, H), np.random.randint(0, W)
# Draw black line
cv2.line(image, (x0, y0), (x1, y1), color=(0, 0, 0), thickness=line_thickness)
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Red image with random black line created in {elapsed:.6f} seconds")
print(f"Line coordinates: ({x0}, {y0}) → ({x1}, {y1})")
# -----------------------------
# 4. Save image
# -----------------------------
output_filename = "red_image_black_line.png"
cv2.imwrite(output_filename, image)
print(f"Saved image as {output_filename}")
# -----------------------------
# 5. Performance estimation
# -----------------------------
# Estimate operations
# Image initialization: writing 3 values per pixel
flops_init = num_pixels * 3
# Random coordinate generation (approx)
flops_rand = 20
# Bresenham-like line algorithm approx operations
line_length = int(np.sqrt((x1-x0)**2 + (y1-y0)**2))
flops_line = line_length * 10
total_flops = flops_init + flops_rand + flops_line
# GFLOPs
gflops = total_flops / elapsed / 1e9
print("\n--- Performance estimation ---")
print(f"Estimated FLOPs: {total_flops:.2e}")
print(f"Estimated performance: {gflops:.6f} GFLOPs")
# -----------------------------
# 6. Memory bandwidth estimation
# -----------------------------
# Memory written: entire image
bytes_image = H * W * 3
# Approx memory touched by line drawing
bytes_line = line_length * 3
total_bytes = bytes_image + bytes_line
bandwidth = total_bytes / elapsed / 1e9
print("\n--- Memory bandwidth estimation ---")
print(f"Total data moved: {total_bytes/1e6:.3f} MB")
print(f"Estimated memory bandwidth: {bandwidth:.3f} GB/s")

