# red_image_black_line_pure_python.py
import time
import random
import cv2
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
# image[y][x] = [B, G, R]
image = []
for y in range(H):
    row = []
    for x in range(W):
        row.append([0, 0, 255])   # red pixel
    image.append(row)
# -----------------------------
# 4. Generate random line coordinates
# -----------------------------
y0, x0 = random.randint(0, H-1), random.randint(0, W-1)
y1, x1 = random.randint(0, H-1), random.randint(0, W-1)
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
    for yy in range(y_min, y_max):
        for xx in range(x_min, x_max):
            image[yy][xx] = [0, 0, 0]   # black pixel

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
# Convert Python list → OpenCV image
import numpy as np
image_np = np.array(image, dtype=np.uint8)
output_filename = "red_image_black_line_pure_python.png"
cv2.imwrite(output_filename, image_np)
print(f"Saved image as {output_filename}")
# -------------------------------------------------
# 8. Performance estimation
# -------------------------------------------------
flops_init = num_pixels * 3
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
bytes_image = H * W * 3
bytes_line = line_pixels * line_thickness * line_thickness * 3
total_bytes = bytes_image + bytes_line
bandwidth = total_bytes / elapsed / 1e9
print("\n--- Memory bandwidth estimation ---")
print(f"Total data moved: {total_bytes/1e6:.3f} MB")
print(f"Estimated memory bandwidth: {bandwidth:.3f} GB/s")

