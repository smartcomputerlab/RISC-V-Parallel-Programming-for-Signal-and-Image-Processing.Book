# sobel_mp_numpy_file.py
import numpy as np
import multiprocessing as mp
import cv2
import time
import sys
# -------------------------------------------------
# 1. Arguments
# -------------------------------------------------
if len(sys.argv) > 1:
    image_file = sys.argv[1]
else:
    print("Usage: python sobel_mp_numpy_file_v2.py image_file num_cpu")
    sys.exit()
if len(sys.argv) > 2:
    NUM_CPU = int(sys.argv[2])
else:
    NUM_CPU = mp.cpu_count()
print("\nSobel Filter (NumPy + multiprocessing)")
print("Input image:", image_file)
print("CPU cores:", NUM_CPU)
# -------------------------------------------------
# 2. Load grayscale image
# -------------------------------------------------
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: cannot load image")
    sys.exit()
image = image.astype(np.float32)
HEIGHT, WIDTH = image.shape
print("Image size:", WIDTH, "x", HEIGHT)
# -------------------------------------------------
# 3. Sobel kernels as NumPy arrays
# -------------------------------------------------
Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
# -------------------------------------------------
# 4. Worker function
# -------------------------------------------------
def sobel_block(args):
    start_row, end_row, block = args
    H_block, W_block = block.shape
    gx = np.zeros_like(block)
    gy = np.zeros_like(block)
    # Convolution using NumPy slicing
    gx[1:-1,1:-1] = (
        Kx[0,0]*block[:-2,:-2] + Kx[0,1]*block[:-2,1:-1] + Kx[0,2]*block[:-2,2:] +
        Kx[1,0]*block[1:-1,:-2] + Kx[1,1]*block[1:-1,1:-1] + Kx[1,2]*block[1:-1,2:] +
        Kx[2,0]*block[2:,:-2] + Kx[2,1]*block[2:,1:-1] + Kx[2,2]*block[2:,2:]
    )
    gy[1:-1,1:-1] = (
        Ky[0,0]*block[:-2,:-2] + Ky[0,1]*block[:-2,1:-1] + Ky[0,2]*block[:-2,2:] +
        Ky[1,0]*block[1:-1,:-2] + Ky[1,1]*block[1:-1,1:-1] + Ky[1,2]*block[1:-1,2:] +
        Ky[2,0]*block[2:,:-2] + Ky[2,1]*block[2:,1:-1] + Ky[2,2]*block[2:,2:]
    )
    magnitude = np.sqrt(gx**2 + gy**2)
    return (start_row, magnitude)
# -------------------------------------------------
# 5. Split image into row blocks for multiprocessing
# -------------------------------------------------
chunk = HEIGHT // NUM_CPU
tasks = []
for i in range(NUM_CPU):
    start = i*chunk
    end = (i+1)*chunk if i < NUM_CPU-1 else HEIGHT
    block = image[start:end,:]
    tasks.append((start,end,block))
# -------------------------------------------------
# 6. Parallel computation
# -------------------------------------------------
t0 = time.perf_counter()
with mp.Pool(NUM_CPU) as pool:
    results = pool.map(sobel_block, tasks)
t1 = time.perf_counter()
elapsed = t1 - t0
print("Processing time:", elapsed, "seconds")
# -------------------------------------------------
# 7. Assemble final image
# -------------------------------------------------
sobel_img = np.zeros_like(image)
for start, block in results:
    sobel_img[start:start+block.shape[0],:] = block
# -------------------------------------------------
# 8. Performance estimation in GFLOPs
# -------------------------------------------------
# NumPy convolution estimate:
# Each pixel (excluding borders):
# gx: 9 multiplications + 8 additions
# gy: 9 multiplications + 8 additions
# gx^2 + gy^2: 2 multiplications + 1 addition
# sqrt: 1 operation
# -----------------------------------------------
flops_per_pixel = 9+8 + 9+8 + 2+1 + 1   # = 38 FLOPs per pixel
valid_pixels = (HEIGHT-2)*(WIDTH-2)
total_flops = valid_pixels * flops_per_pixel
gflops = total_flops / elapsed / 1e9
print("Estimated performance:", round(gflops,3), "GFLOPs")
# -------------------------------------------------
# 9. Save output image
# -------------------------------------------------
sobel_img = np.clip(sobel_img,0,255).astype(np.uint8)
cv2.imwrite("sobel_output_numpy.png", sobel_img)
print("Output saved: sobel_output_numpy.png")

