# gaussian_blur_rgb_pure_python_mp.py
import multiprocessing as mp
import cv2
import sys
import time
import numpy as np
# -------------------------------------------------
# 1. Read arguments
# -------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python gaussian_blur_rgb_pure_python_mp.py image_file [num_cpu]")
    sys.exit()

image_file = sys.argv[1]
NUM_CPU = int(sys.argv[2]) if len(sys.argv) > 2 else mp.cpu_count()
print("\nGaussian Blur (Pure Python) RGB with multiprocessing")
print("Input image:", image_file)
print("CPU cores:", NUM_CPU)
# -------------------------------------------------
# 2. Load RGB image
# -------------------------------------------------
img_cv = cv2.imread(image_file, cv2.IMREAD_COLOR)
if img_cv is None:
    print("Error loading image")
    sys.exit()
HEIGHT, WIDTH, CHANNELS = img_cv.shape
print("Image size:", WIDTH, "x", HEIGHT, "Channels:", CHANNELS)
# Convert image to Python list
image = img_cv.tolist()
# -------------------------------------------------
# 3. Gaussian kernel (5x5)
# -------------------------------------------------
kernel = [
    [1, 4, 6, 4, 1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1, 4, 6, 4, 1]
]
norm = 256.0
# -------------------------------------------------
# 4. Worker function for one channel
# -------------------------------------------------
def gaussian_block_channel(args):
    start_row, block = args
    h = len(block)
    w = len(block[0])
    result = [[ [0]*CHANNELS for _ in range(w) ] for _ in range(h)]  # RGB
    for y in range(2, h-2):
        for x in range(2, w-2):
            acc = [0.0]*CHANNELS
            for ky in range(-2,3):
                for kx in range(-2,3):
                    pixel = block[y+ky][x+kx]  # list of R,G,B
                    weight = kernel[ky+2][kx+2]
                    for c in range(CHANNELS):
                        acc[c] += pixel[c] * weight
            for c in range(CHANNELS):
                result[y][x][c] = int(acc[c] / norm)
    return (start_row, result)

# -------------------------------------------------
# 5. Split image rows
# -------------------------------------------------
chunk = HEIGHT // NUM_CPU
tasks = []
for i in range(NUM_CPU):
    start = i * chunk
    end = (i+1)*chunk if i < NUM_CPU-1 else HEIGHT
    block = image[start:end]
    tasks.append((start, block))
# -------------------------------------------------
# 6. Parallel processing
# -------------------------------------------------
t0 = time.perf_counter()
with mp.Pool(NUM_CPU) as pool:
    results = pool.map(gaussian_block_channel, tasks)
t1 = time.perf_counter()
elapsed = t1 - t0
print("Processing time:", elapsed, "seconds")
# -------------------------------------------------
# 7. Assemble final image
# -------------------------------------------------
blur = [[[0]*CHANNELS for _ in range(WIDTH)] for _ in range(HEIGHT)]
for start, block in results:
    for i,row in enumerate(block):
        blur[start+i] = row
# -------------------------------------------------
# 8. Performance estimation
# -------------------------------------------------
pixels = WIDTH * HEIGHT * CHANNELS
flops_per_pixel = 50  # approximate for 5x5 convolution per channel
total_flops = pixels * flops_per_pixel
gflops = total_flops / elapsed / 1e9
print("Estimated performance:", round(gflops,3), "GFLOPS")
# -------------------------------------------------
# 9. Save output image
# -------------------------------------------------
blur_cv = np.array(blur, dtype='uint8')
cv2.imwrite("gaussian_blur_rgb_pure_python.png", blur_cv)
print("Output saved: gaussian_blur_rgb_pure_python.png")

