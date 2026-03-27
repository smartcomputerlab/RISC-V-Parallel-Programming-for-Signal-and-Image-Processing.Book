# gaussian_blur_scipy_mp.py

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import sys
import cv2

# -----------------------------
# Worker function (SciPy Gaussian blur on RGB block)
# -----------------------------
def blur_worker(chunk):
    # Apply Gaussian blur to all channels at once
    return gaussian_filter(chunk, sigma=(1.5, 1.5, 0), mode='reflect')

# -----------------------------
# Main program
# -----------------------------
def main():

    if len(sys.argv) < 3:
        print("Usage: python3 gaussian_blur_scipy_mp.py IMAGE_FILE NUM_CPUS")
        sys.exit(1)

    image_file = sys.argv[1]
    nproc = int(sys.argv[2])

    print("Multiprocessing Gaussian Blur Benchmark (SciPy version)")
    print(f"Input image: {image_file}")
    print(f"Processes: {nproc}")

    # -----------------------------
    # Load image as RGB float32
    # -----------------------------
    img_cv = cv2.imread(image_file, cv2.IMREAD_COLOR)

    if img_cv is None:
        print("Error loading image")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W, C = img_rgb.shape

    print(f"Image size: {H} x {W} RGB")

    # -----------------------------
    # Split image into chunks (rows)
    # -----------------------------
    chunks = np.array_split(img_rgb, nproc, axis=0)

    # -----------------------------
    # Warm-up
    # -----------------------------
    blur_worker(chunks[0])

    # -----------------------------
    # Benchmark
    # -----------------------------
    start = time.perf_counter()

    with mp.Pool(nproc) as pool:
        results = pool.map(blur_worker, chunks)

    end = time.perf_counter()
    elapsed = end - start

    # -----------------------------
    # Combine results
    # -----------------------------
    blurred = np.vstack(results)

    print(f"Execution time: {elapsed:.6f} seconds")
    print("Checksum:", np.sum(blurred))

    # -----------------------------
    # FLOP estimation
    # -----------------------------
    # Gaussian kernel approx 7x7 (sigma=1.5)
    kernel_size = 7

    flops_per_pixel = 2 * kernel_size * kernel_size
    total_flops = H * W * C * flops_per_pixel
    gflops = total_flops / elapsed / 1e9

    print(f"Estimated FLOPs: {total_flops:.3e}")
    print(f"Estimated Performance: {gflops:.3f} GFLOPS")

    # -----------------------------
    # Memory bandwidth estimation
    # -----------------------------
    bytes_moved = H * W * C * 4 * 2   # read + write (float32)
    bandwidth = bytes_moved / elapsed / 1e9

    print(f"Estimated memory bandwidth: {bandwidth:.3f} GB/s")

    # -----------------------------
    # Display result
    # -----------------------------
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Gaussian Blur (SciPy)")
    plt.imshow(blurred)
    plt.axis("off")

    plt.show()

    # -----------------------------
    # Save output image
    # -----------------------------
    blur_out = (blurred * 255).astype(np.uint8)
    blur_out_bgr = cv2.cvtColor(blur_out, cv2.COLOR_RGB2BGR)

    cv2.imwrite("gaussian_blur_output_scipy.png", blur_out_bgr)

    print("Output saved: gaussian_blur_output_scipy.png")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()

