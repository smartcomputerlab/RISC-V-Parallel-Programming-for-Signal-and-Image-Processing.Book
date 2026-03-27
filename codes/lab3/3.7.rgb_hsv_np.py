import numpy as np
import cv2
import time
import sys

def rgb_to_hsv_numpy(image):
    # Normalize RGB values
    img = image.astype(np.float32) / 255.0
    r = img[:,:,2]
    g = img[:,:,1]
    b = img[:,:,0]
    cmax = np.maximum(np.maximum(r,g),b)
    cmin = np.minimum(np.minimum(r,g),b)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax
    mask = delta != 0
    mask_r = (cmax == r) & mask
    mask_g = (cmax == g) & mask
    mask_b = (cmax == b) & mask
    h[mask_r] = (60 * ((g[mask_r]-b[mask_r]) / delta[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g]-r[mask_g]) / delta[mask_g]) + 120)
    h[mask_b] = (60 * ((r[mask_b]-g[mask_b]) / delta[mask_b]) + 240)
    s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    hsv = np.stack((h,s,v),axis=2)
    return hsv

def main():

    if len(sys.argv) != 2:
        print("Usage: python rgb_to_hsv_numpy_perf.py image.png")
        return
    input_name = sys.argv[1]
    img = cv2.imread(input_name)
    if img is None:
        print("Error loading image")
        return
    height, width, ch = img.shape
    pixels = height * width
    print("Image size:", width, "x", height)
    # Warm-up (important for fair measurement)
    _ = rgb_to_hsv_numpy(img)
    t0 = time.time()
    hsv = rgb_to_hsv_numpy(img)
    t1 = time.time()
    exec_time = t1 - t0
    print("Execution time:", exec_time, "seconds")
    # Save result for visualization
    hsv_vis = hsv.copy()
    hsv_vis[:,:,0] = hsv_vis[:,:,0] / 360.0
    hsv_vis = (hsv_vis * 255).astype(np.uint8)
    cv2.imwrite("output_hsv.png", hsv_vis)
    # ------------------------------------------------
    # Performance estimation
    # ------------------------------------------------
    # Approximate number of floating-point operations per pixel
    flops_per_pixel = 40
    total_flops = pixels * flops_per_pixel
    gflops = total_flops / exec_time / 1e9
    print("Estimated FLOPs       :", "{:.3e}".format(total_flops))
    print("Estimated Performance :", "{:.3f}".format(gflops), "GFLOPS")
    # ------------------------------------------------
    # Memory bandwidth estimation
    # ------------------------------------------------
    bytes_read  = pixels * 3      # RGB
    bytes_write = pixels * 3      # HSV
    total_bytes = bytes_read + bytes_write
    bandwidth = total_bytes / exec_time / 1e9
    print("Estimated Memory Bandwidth:", "{:.3f}".format(bandwidth), "GB/s")

if __name__ == "__main__":
    main()

