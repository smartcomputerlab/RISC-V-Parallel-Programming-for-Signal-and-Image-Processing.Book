import cv2
import numpy as np
import sys
import time
import os

if len(sys.argv) != 2:
    print("Usage: python negate_image.py input_image")
    print("Example: python negate_image.py lena.png")
    sys.exit(1)
input_filename = sys.argv[1]
# Generate output filename
name, ext = os.path.splitext(input_filename)
output_filename = name + "_negated" + ext
# -----------------------------
# 2. Read image
# -----------------------------
image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
if image is None:
    print("Error: Could not read image:", input_filename)
    sys.exit(1)
H, W, C = image.shape
print("Image size:", H, "x", W, "x", C)
# -----------------------------
# 3. Negate image with timing
# -----------------------------
start = time.perf_counter()
negated = 255 - image   # Vectorized NumPy operation
end = time.perf_counter()
elapsed = end - start
print(f"Execution time: {elapsed:.6f} seconds")
# -----------------------------
# 4. GFLOPS estimation
# -----------------------------
flops = H * W * C     # 1 subtraction per channel
gflops = flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPS")
# -----------------------------
# 5. Save output image
# -----------------------------
cv2.imwrite(output_filename, negated)
print("Negated image saved as:", output_filename)

