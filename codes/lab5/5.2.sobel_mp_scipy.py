# sobel_mp_scipy_rgb.py
import numpy as np
import multiprocessing as mp
import cv2
import time
import sys
from scipy.ndimage import sobel

# -------------------------------------------------
# 1. Arguments
# -------------------------------------------------
if len(sys.argv) > 1:
    image_file = sys.argv[1]
else:
    print("Usage: python sobel_mp_scipy_rgb.py image_file num_cpu")
    sys.exit()

if len(sys.argv) > 2:
    NUM_CPU = int(sys.argv[2])
else:
    NUM_CPU = mp.cpu_count()

print("\nSobel Filter RGB (SciPy + multiprocessing)")
print("Input image:", image_file)
print("CPU cores:", NUM_CPU)

# -------------------------------------------------
# 2. Load RGB image
# -------------------------------------------------
image = cv2.imread(image_file, cv2.IMREAD_COLOR)

if image is None:
    print("Error: cannot load image")
    sys.exit()

image = image.astype(np.float32)
HEIGHT, WIDTH, CHANNELS = image.shape
print("Image size:", WIDTH, "x", HEIGHT, " - Channels:", CHANNELS)

# -------------------------------------------------
# 3. Worker function using SciPy Sobel (RGB)
# -------------------------------------------------
def sobel_block(args):
    start_row, block = args

    # Output block
    result = np.zeros_like(block)

    # Apply Sobel per channel
    for c in range(3):
        gx = sobel(block[:, :, c], axis=1, mode='nearest')
        gy = sobel(block[:, :, c], axis=0, mode='nearest')

        magnitude = np.sqrt(gx**2 + gy**2)
        result[:, :, c] = magnitude

    return (start_row, result)

# -------------------------------------------------
# 4. Split image into row blocks for multiprocessing
# -------------------------------------------------
chunk = HEIGHT // NUM_CPU
tasks = []

for i in range(NUM_CPU):
    start = i * chunk
    end = (i + 1) * chunk if i < NUM_CPU - 1 else HEIGHT
    block = image[start:end, :, :]
    tasks.append((start, block))

# -------------------------------------------------
# 5. Parallel computation
# -------------------------------------------------
t0 = time.perf_counter()

with mp.Pool(NUM_CPU) as pool:
    results = pool.map(sobel_block, tasks)

t1 = time.perf_counter()
elapsed = t1 - t0

print("Processing time:", elapsed, "seconds")

# -------------------------------------------------
# 6. Assemble final RGB image
# -------------------------------------------------
sobel_img = np.zeros_like(image)

for start, block in results:
    sobel_img[start:start + block.shape[0], :, :] = block

# -------------------------------------------------
# 7. Performance estimation in GFLOPs
# -------------------------------------------------
# Same Sobel complexity per pixel = 38 FLOPs
# Now applied to 3 channels

flops_per_pixel = 38 * 3
valid_pixels = (HEIGHT - 2) * (WIDTH - 2)

total_flops = valid_pixels * flops_per_pixel
gflops = total_flops / elapsed / 1e9

print("Estimated performance:", round(gflops, 3), "GFLOPs")

# -------------------------------------------------
# 8. Save output image
# -------------------------------------------------
sobel_img = np.clip(sobel_img, 0, 255).astype(np.uint8)
cv2.imwrite("sobel_output_rgb_scipy.png", sobel_img)

print("Output saved: sobel_output_rgb_scipy.png")

