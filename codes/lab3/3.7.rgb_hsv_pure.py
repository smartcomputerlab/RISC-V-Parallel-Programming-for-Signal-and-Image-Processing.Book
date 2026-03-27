import cv2
import numpy as np
import time
import sys

def rgb_to_hsv_serial(image):
    height, width, _ = image.shape
    hsv = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            b = image[i, j, 0] / 255.0
            g = image[i, j, 1] / 255.0
            r = image[i, j, 2] / 255.0
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            h = 0.0
            if delta != 0:
                if cmax == r:
                    h = (60 * ((g - b) / delta) + 360) % 360
                elif cmax == g:
                    h = (60 * ((b - r) / delta) + 120)
                else:
                    h = (60 * ((r - g) / delta) + 240)
            if cmax != 0:
                s = delta / cmax
            else:
                s = 0
            v = cmax
            hsv[i, j, 0] = h
            hsv[i, j, 1] = s
            hsv[i, j, 2] = v
    return hsv

def main():
    if len(sys.argv) != 2:
        print("Usage: python rgb_to_hsv_serial.py image.png")
        return

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Error loading image")
        return
    height, width, _ = img.shape
    pixels = height * width
    print("Image size:", width, "x", height)
    # Warm-up
    _ = rgb_to_hsv_serial(img)
    t0 = time.time()
    hsv = rgb_to_hsv_serial(img)
    t1 = time.time()
    exec_time = t1 - t0
    print("Execution time:", exec_time, "seconds")
    # Save result
    hsv_vis = hsv.copy()
    hsv_vis[:,:,0] = hsv_vis[:,:,0] / 360.0
    hsv_vis = (hsv_vis * 255).astype(np.uint8)
    cv2.imwrite("output_hsv_serial.png", hsv_vis)
    # ------------------------------------------------
    # Performance estimation
    # ------------------------------------------------
    flops_per_pixel = 40
    total_flops = pixels * flops_per_pixel
    gflops = total_flops / exec_time / 1e9
    print("Estimated FLOPs       :", "{:.3e}".format(total_flops))
    print("Estimated Performance :", "{:.6f}".format(gflops), "GFLOPS")
    # ------------------------------------------------
    # Memory bandwidth estimation
    # ------------------------------------------------
    bytes_read = pixels * 3
    bytes_write = pixels * 3
    total_bytes = bytes_read + bytes_write
    bandwidth = total_bytes / exec_time / 1e9
    print("Estimated Memory Bandwidth:", "{:.6f}".format(bandwidth), "GB/s")

if __name__ == "__main__":
    main()

