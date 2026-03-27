import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time
import sys
# -----------------------------
# Check input argument
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python3 gaussian_blur_image.py IMAGE_FILE")
    sys.exit(1)

image_file = sys.argv[1]
# -----------------------------
# Load image
# -----------------------------
img = plt.imread(image_file)
# Convert to float32 for processing
img = img.astype(np.float32)
# Resize to 512x512 if necessary
H, W = 512, 512
img = img[:H, :W, :3]
print("Gaussian Blur using SciPy")
print(f"Image size: {img.shape}")

# Output image
blur = np.zeros_like(img)
sigma = 2
# -----------------------------
# Warm-up
# -----------------------------
for c in range(3):
    gaussian_filter(img[:,:,c], sigma=sigma)
# -----------------------------
# Benchmark
# -----------------------------
start = time.perf_counter()
for c in range(3):
    blur[:,:,c] = gaussian_filter(img[:,:,c], sigma=sigma)
end = time.perf_counter()
elapsed = end - start
print(f"Execution time: {elapsed:.6f} seconds")
# Prevent optimization removal
print("Checksum:", np.sum(blur))
# -----------------------------
# FLOP estimation
# -----------------------------
kernel_size = int(6*sigma + 1)
flops_per_pixel = 2 * kernel_size * kernel_size
H, W, C = img.shape
total_flops = H * W * C * flops_per_pixel
gflops = total_flops / elapsed / 1e9
print(f"Estimated FLOPs: {total_flops:.3e}")
print(f"Estimated performance: {gflops:.3f} GFLOPS")
# -----------------------------
# Memory bandwidth estimation
# -----------------------------
bytes_moved = H * W * C * 4 * 2
bandwidth = bytes_moved / elapsed / 1e9
print(f"Estimated memory bandwidth: {bandwidth:.3f} GB/s")
# -----------------------------
# Display images
# -----------------------------
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.title("Gaussian Blur")
plt.imshow(blur)
plt.axis("off")
plt.show()

