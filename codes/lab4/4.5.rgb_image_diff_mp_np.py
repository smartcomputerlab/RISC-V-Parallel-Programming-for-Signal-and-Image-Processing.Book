# rgb_image_diff_mp.py
import numpy as np
import multiprocessing as mp
import cv2
import sys
import time
# -------------------------------------------------
# 1. Read arguments
# -------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: python rgb_image_diff_mp.py image1 image2 num_cpu")
    sys.exit()

img1_file = sys.argv[1]
img2_file = sys.argv[2]

if len(sys.argv) > 3:
    NUM_CPU = int(sys.argv[3])
else:
    NUM_CPU = mp.cpu_count()

print("\nRGB Image Difference using NumPy + multiprocessing")
print("Image1:", img1_file)
print("Image2:", img2_file)
print("CPU cores:", NUM_CPU)
# -------------------------------------------------
# 2. Load images in color
# -------------------------------------------------
img1 = cv2.imread(img1_file, cv2.IMREAD_COLOR)
img2 = cv2.imread(img2_file, cv2.IMREAD_COLOR)
if img1 is None or img2 is None:
    print("Error: cannot load images")
    sys.exit()
if img1.shape != img2.shape:
    print("Error: Images must have the same size")
    sys.exit()
HEIGHT, WIDTH, CHANNELS = img1.shape
print("Image size:", WIDTH, "x", HEIGHT, "Channels:", CHANNELS)
# Convert to int16 to avoid overflow
img1 = img1.astype(np.int16)
img2 = img2.astype(np.int16)
# -------------------------------------------------
# 3. Worker function
# -------------------------------------------------
def diff_block(args):
    start, block1, block2 = args
    diff = np.abs(block1 - block2)
    return (start, diff)
# -------------------------------------------------
# 4. Split rows for multiprocessing
# -------------------------------------------------
chunk = HEIGHT // NUM_CPU
tasks = []
for i in range(NUM_CPU):
    start = i * chunk
    end = (i + 1) * chunk if i < NUM_CPU - 1 else HEIGHT
    block1 = img1[start:end, :, :]
    block2 = img2[start:end, :, :]
    tasks.append((start, block1, block2))
# -------------------------------------------------
# 5. Parallel computation
# -------------------------------------------------
t0 = time.perf_counter()
with mp.Pool(NUM_CPU) as pool:
    results = pool.map(diff_block, tasks)
# -------------------------------------------------
# 6. Assemble result image
# -------------------------------------------------
diff_img = np.zeros_like(img1)
for start, block in results:
    diff_img[start:start + block.shape[0], :, :] = block
t1 = time.perf_counter()
elapsed = t1 - t0
print("Processing time:", elapsed, "seconds")
# -------------------------------------------------
# 7. Estimate performance in GFLOPs
# -------------------------------------------------
# For each pixel and channel:
# block1 - block2  -> 1 subtraction
# abs(...)         -> 1 operation
# --------------------------------
flops_per_element = 2
total_elements = HEIGHT * WIDTH * CHANNELS
total_flops = total_elements * flops_per_element
gflops = total_flops / elapsed / 1e9
print("Estimated performance:", gflops, "GFLOPs")
# -------------------------------------------------
# 8. Save output image
# -------------------------------------------------
diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)
output_file = "rgb_image_difference.png"
cv2.imwrite(output_file, diff_img)
print("Output saved:", output_file)

