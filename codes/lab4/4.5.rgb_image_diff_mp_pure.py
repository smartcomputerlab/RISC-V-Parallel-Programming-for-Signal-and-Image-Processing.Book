# rgb_image_diff_mp_no_numpy.py
import cv2
import sys
import multiprocessing as mp
import time
if len(sys.argv) < 3:
    print("Usage: python rgb_image_diff_mp_no_numpy.py image1 image2 num_cpu")
    sys.exit()
img1_file = sys.argv[1]
img2_file = sys.argv[2]
if len(sys.argv) > 3:
    NUM_CPU = int(sys.argv[3])
else:
    NUM_CPU = mp.cpu_count()
print("\nRGB Image Difference (Pure Python + multiprocessing)")
print("Image1:", img1_file)
print("Image2:", img2_file)
print("CPU cores:", NUM_CPU)
# -------------------------------------------------
# 2. Load images
# -------------------------------------------------
img1_cv = cv2.imread(img1_file, cv2.IMREAD_COLOR)
img2_cv = cv2.imread(img2_file, cv2.IMREAD_COLOR)
if img1_cv is None or img2_cv is None:
    print("Error: cannot load images"); sys.exit()
if img1_cv.shape != img2_cv.shape:
    print("Error: Images must have the same size");  sys.exit()
HEIGHT, WIDTH, CHANNELS = img1_cv.shape
print("Image size:", WIDTH, "x", HEIGHT, "Channels:", CHANNELS)
# Convert images to nested Python lists
img1 = img1_cv.tolist()
img2 = img2_cv.tolist()
# -------------------------------------------------
# 3. Worker function
# -------------------------------------------------
def diff_block(args):
    start_row, block1, block2 = args
    diff_block = []
    for y in range(len(block1)):
        row = []
        for x in range(len(block1[0])):
            pixel1 = block1[y][x]
            pixel2 = block2[y][x]
            # Absolute difference per channel
            diff_pixel = [abs(int(pixel1[c]) - int(pixel2[c])) for c in range(3)]
            row.append(diff_pixel)
        diff_block.append(row)
    return (start_row, diff_block)
# -------------------------------------------------
# 4. Split rows for multiprocessing
# -------------------------------------------------
chunk = HEIGHT // NUM_CPU
tasks = []
for i in range(NUM_CPU):
    start = i * chunk
    end = (i + 1) * chunk if i < NUM_CPU - 1 else HEIGHT
    block1 = img1[start:end]
    block2 = img2[start:end]
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
diff_img_cv = img1_cv.copy()
diff_img_cv[:] = 0
for start, block in results:
    for y, row in enumerate(block):
        for x, pixel in enumerate(row):
            diff_img_cv[start + y, x] = pixel
t1 = time.perf_counter()
elapsed = t1 - t0
print("Processing time:", elapsed, "seconds")
# -------------------------------------------------
# 7. Estimate performance in GFLOPs
# -------------------------------------------------
# Per pixel and per channel:
# subtraction -> 1 operation
# abs()       -> 1 operation
# ---------------------------------
flops_per_channel = 2
flops_per_pixel = flops_per_channel * CHANNELS
total_pixels = HEIGHT * WIDTH
total_flops = total_pixels * flops_per_pixel
gflops = total_flops / elapsed / 1e9
print("Estimated performance:", gflops, "GFLOPs")
output_file = "rgb_image_difference_no_numpy.png"
cv2.imwrite(output_file, diff_img_cv)
print("Output saved:", output_file)

