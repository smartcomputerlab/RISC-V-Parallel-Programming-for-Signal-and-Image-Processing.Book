# sobel_pure_python_mp.py
import multiprocessing as mp
import time
import sys
import cv2
import numpy as np
# -------------------------------------------------
# 1. Arguments
# -------------------------------------------------
if len(sys.argv) > 1:
    image_file = sys.argv[1]
else:
    print("Usage: python sobel_pure_python_mp.py image_file num_cpu")
    sys.exit()
if len(sys.argv) > 2:
    NUM_CPU = int(sys.argv[2])
else:
    NUM_CPU = mp.cpu_count()
print("\nSobel Filter (Pure Python) using multiprocessing")
print("Input image:", image_file)
print("CPU cores:", NUM_CPU)
# -------------------------------------------------
# 2. Load image and convert to grayscale 2D list
# -------------------------------------------------
img_cv = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
if img_cv is None:
    print("Error: cannot load image")
    sys.exit()
HEIGHT, WIDTH = img_cv.shape
print("Image size:", WIDTH, "x", HEIGHT)
# Convert to nested list
image = img_cv.tolist()
# -------------------------------------------------
# 3. Sobel kernels
# -------------------------------------------------
Kx = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

Ky = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]
# -------------------------------------------------
# 4. Worker function
# -------------------------------------------------
def sobel_block(args):
    start_row, end_row, block = args
    block_height = len(block)
    block_width = len(block[0])
    # Initialize output block
    result = [[0]*block_width for _ in range(block_height)]
    for y in range(1, block_height-1):
        for x in range(1, block_width-1):
            gx = 0
            gy = 0
            for ky in range(-1,2):
                for kx in range(-1,2):
                    val = block[y+ky][x+kx]
                    gx += val * Kx[ky+1][kx+1]
                    gy += val * Ky[ky+1][kx+1]
            mag = (gx*gx + gy*gy)**0.5
            result[y][x] = int(min(255, mag))
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
    tasks.append((start, end, block))
# -------------------------------------------------
# 6. Parallel processing
# -------------------------------------------------
t0 = time.perf_counter()
with mp.Pool(NUM_CPU) as pool:
    results = pool.map(sobel_block, tasks)
# -------------------------------------------------
# 7. Assemble final image
# -------------------------------------------------
sobel_img = [[0]*WIDTH for _ in range(HEIGHT)]
for start, block in results:
    for i, row in enumerate(block):
        sobel_img[start+i] = row
t1 = time.perf_counter()
elapsed = t1 - t0
print("Processing time:", elapsed, "seconds")
# -------------------------------------------------
# 8. Estimate performance in GFLOPs
# -------------------------------------------------
# Per pixel (inside the valid region):
# 9 multiplications for gx
# 9 multiplications for gy
# 18 additions (accumulations)
# gx*gx + gy*gy -> 2 multiplications + 1 addition
# sqrt -> 1 operation (approx)
# -----------------------------------------------
flops_per_pixel = 9 + 9 + 18 + 2 + 1 + 1   # ≈ 40 FLOPs per pixel
valid_pixels = (HEIGHT - 2) * (WIDTH - 2)
total_flops = valid_pixels * flops_per_pixel
gflops = total_flops / elapsed / 1e9
print("Estimated performance:", gflops, "GFLOPs")
# -------------------------------------------------
# 9. Save output using OpenCV
# -------------------------------------------------
sobel_cv = np.array(sobel_img, dtype=np.uint8)
cv2.imwrite("sobel_pure_python.png", sobel_cv)
print("Output saved: sobel_pure_python.png")

