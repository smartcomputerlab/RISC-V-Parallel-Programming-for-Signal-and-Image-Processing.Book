import numpy as np
import time
# -----------------------------
# 1. Image parameters
# -----------------------------
H, W = 512, 512
line_thickness = 2
num_pixels = H * W
# -----------------------------
# 2. Start performance timer
# -----------------------------
start_time = time.perf_counter()
# -----------------------------
# 3. Create red image
# -----------------------------
image = np.zeros((H, W, 3), dtype=np.uint8)
image[:, :, 2] = 255
# -----------------------------
# 4. Generate random line coordinates
# -----------------------------
y0, x0 = np.random.randint(0, H), np.random.randint(0, W)
y1, x1 = np.random.randint(0, H), np.random.randint(0, W)
# -----------------------------
# 5. Draw line (Bresenham-like)
# -----------------------------
dx = abs(x1 - x0)
dy = abs(y1 - y0)
sx = 1 if x0 < x1 else -1
sy = 1 if y0 < y1 else -1
err = dx - dy
x, y = x0, y0
line_pixels = 0
while True:
    y_min = max(0, y - line_thickness // 2)
    y_max = min(H, y + line_thickness // 2 + 1)
    x_min = max(0, x - line_thickness // 2)
    x_max = min(W, x + line_thickness // 2 + 1)
    image[y_min:y_max, x_min:x_max, :] = 0
    line_pixels += 1
    if x == x1 and y == y1:
        break
    e2 = 2 * err
    if e2 > -dy:
        err -= dy
        x += sx
    if e2 < dx:
        err += dx
        y += sy
# -----------------------------
# 6. End performance timer
# -----------------------------
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Red image with random black line created in {elapsed:.6f} seconds")
print(f"Line coordinates: ({x0}, {y0}) → ({x1}, {y1})")
# -----------------------------
# 7. Save image
# -----------------------------
import cv2
output_filename = "red_image_black_line_numpy.png"
cv2.imwrite(output_filename, image)
print(f"Saved image as {output_filename}")
# -------------------------------------------------
# 8. Performance estimation
# -------------------------------------------------
# Image initialization operations
flops_init = num_pixels * 3
# Bresenham operations per pixel (approx)
# comparison, addition, subtraction, multiplication
flops_per_pixel = 12
flops_line = line_pixels * flops_per_pixel
total_flops = flops_init + flops_line
gflops = total_flops / elapsed / 1e9
print("\n--- Performance estimation ---")
print(f"Line length (pixels): {line_pixels}")
print(f"Estimated FLOPs: {total_flops:.2e}")
print(f"Estimated performance: {gflops:.6f} GFLOPs")
# -------------------------------------------------
# 9. Memory bandwidth estimation
# -------------------------------------------------
# Image memory size
bytes_image = H * W * 3
# Memory accessed by line drawing
bytes_line = line_pixels * line_thickness * line_thickness * 3
total_bytes = bytes_image + bytes_line
bandwidth = total_bytes / elapsed / 1e9
print("\n--- Memory bandwidth estimation ---")
print(f"Total data moved: {total_bytes/1e6:.3f} MB")
print(f"Estimated memory bandwidth: {bandwidth:.3f} GB/s")

