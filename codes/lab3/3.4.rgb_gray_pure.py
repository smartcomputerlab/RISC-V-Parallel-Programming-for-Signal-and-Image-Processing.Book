# rgb_to_gray_pure_python.py
import cv2
import sys
import time
import os
# -----------------------------
# 1. Check command-line argument
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python rgb_to_gray_pure_python.py input_image")
    sys.exit(1)
input_filename = sys.argv[1]
name, ext = os.path.splitext(input_filename)
output_filename = name + "_gray" + ext
# -----------------------------
# 2. Read RGB image
# -----------------------------
image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
if image is None:
    print("Error: Could not read image:", input_filename)
    sys.exit(1)

H, W, C = image.shape
print("Image size:", H, "x", W, "x", C)
# -----------------------------
# 3. RGB → Gray conversion (pure Python)
# -----------------------------
start = time.perf_counter()
# Create output grayscale image
gray = image.copy()
for y in range(H):
    for x in range(W):
        # OpenCV loads BGR
        B = float(image[y][x][0])
        G = float(image[y][x][1])
        R = float(image[y][x][2])
        gray_value = 0.114 * B + 0.587 * G + 0.299 * R
        # Clamp to 0–255
        if gray_value < 0:
            gray_value = 0
        elif gray_value > 255:
            gray_value = 255
        gray[y][x] = int(gray_value)

end = time.perf_counter()
elapsed = end - start
print(f"Execution time: {elapsed:.6f} seconds")
# -----------------------------
# 4. GFLOPS estimation
# -----------------------------
flops = H * W * 5   # 3 multiplications + 2 additions per pixel
gflops = flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPS")
# -----------------------------
# 5. Memory bandwidth estimation
# -----------------------------
bytes_moved = H * W * (3 + 1) * 4   # same estimate as original program
bandwidth = bytes_moved / elapsed / 1e9
print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
# -----------------------------
# 6. Save grayscale image
# -----------------------------
cv2.imwrite(output_filename, gray)
print("Grayscale image saved as:", output_filename)

