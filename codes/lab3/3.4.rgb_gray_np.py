import numpy as np
import cv2, sys, time, os
if len(sys.argv) != 2:
    print("Usage: python rgb_to_gray.py input_image")
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
# Convert to float32 for proper weighted computation
image_f = image.astype(np.float32)
# -----------------------------
# 3. RGB → Gray conversion
# -----------------------------
start = time.perf_counter()
# OpenCV loads BGR format
B = image_f[:, :, 0]
G = image_f[:, :, 1]
R = image_f[:, :, 2]
gray = 0.114 * B + 0.587 * G + 0.299 * R
end = time.perf_counter()
elapsed = end - start
print(f"Execution time: {elapsed:.6f} seconds")
# Convert back to uint8
gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
# -----------------------------
# 4. GFLOPS estimation
# -----------------------------
flops = H * W * 5   # 3 multiplications + 2 additions per pixel
gflops = flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPS")
# -----------------------------
# 5. Memory bandwidth estimation
# -----------------------------
# Read: 3 channels float32 (4 bytes each) = 12 bytes per pixel
# Write: 1 channel float32 = 4 bytes per pixel
# Total bytes moved
bytes_moved = H * W * (3 + 1) * 4  # float32 = 4 bytes
bandwidth = bytes_moved / elapsed / 1e9  # GB/s
print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
# -----------------------------
# 6. Save grayscale image
# -----------------------------
cv2.imwrite(output_filename, gray_uint8)
print("Grayscale image saved as:", output_filename)

