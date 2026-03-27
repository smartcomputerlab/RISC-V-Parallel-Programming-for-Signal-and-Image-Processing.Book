import numpy as np
import cv2
import time
import sys

def hsv_to_rgb_numpy(hsv):

    # Normalize values
    h = hsv[:,:,0].astype(np.float32) * 360.0 / 255.0
    s = hsv[:,:,1].astype(np.float32) / 255.0
    v = hsv[:,:,2].astype(np.float32) / 255.0
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    cond0 = (h >= 0) & (h < 60)
    cond1 = (h >= 60) & (h < 120)
    cond2 = (h >= 120) & (h < 180)
    cond3 = (h >= 180) & (h < 240)
    cond4 = (h >= 240) & (h < 300)
    cond5 = (h >= 300) & (h < 360)
    r[cond0], g[cond0], b[cond0] = c[cond0], x[cond0], 0
    r[cond1], g[cond1], b[cond1] = x[cond1], c[cond1], 0
    r[cond2], g[cond2], b[cond2] = 0, c[cond2], x[cond2]
    r[cond3], g[cond3], b[cond3] = 0, x[cond3], c[cond3]
    r[cond4], g[cond4], b[cond4] = x[cond4], 0, c[cond4]
    r[cond5], g[cond5], b[cond5] = c[cond5], 0, x[cond5]
    r = (r + m)
    g = (g + m)
    b = (b + m)
    rgb = np.stack((b,g,r),axis=2)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

def main():

    if len(sys.argv) != 2:
        print("Usage: python hsv_to_rgb_numpy.py hsv_image.png")
        return
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Error loading image")
        return
    height, width, _ = img.shape
    pixels = height * width
    print("Image size:", width, "x", height)
    # Warm-up
    _ = hsv_to_rgb_numpy(img)
    t0 = time.time()
    rgb = hsv_to_rgb_numpy(img)
    t1 = time.time()
    exec_time = t1 - t0
    print("Execution time:", exec_time, "seconds")
    cv2.imwrite("output_rgb.png", rgb)
    # -------------------------------
    # Performance estimation
    # -------------------------------
    flops_per_pixel = 35
    total_flops = pixels * flops_per_pixel
    gflops = total_flops / exec_time / 1e9
    print("Estimated FLOPs       :", "{:.3e}".format(total_flops))
    print("Estimated Performance :", "{:.3f}".format(gflops), "GFLOPS")
    # -------------------------------
    # Memory bandwidth estimation
    # -------------------------------
    bytes_read = pixels * 3
    bytes_write = pixels * 3
    total_bytes = bytes_read + bytes_write
    bandwidth = total_bytes / exec_time / 1e9
    print("Estimated Memory Bandwidth:", "{:.3f}".format(bandwidth), "GB/s")

if __name__ == "__main__":
    main()

