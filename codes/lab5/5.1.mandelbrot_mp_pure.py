# mandelbrot_mp_scalar_args.py
import multiprocessing as mp
import time
import sys
# -------------------------------------------------
# 1. Read command-line arguments
# -------------------------------------------------
if len(sys.argv) > 1:
    SIZE = int(sys.argv[1])
else:
    SIZE = 800   # default image size
if len(sys.argv) > 2:
    num_workers = int(sys.argv[2])
else:
    num_workers = mp.cpu_count()  # default: all cores
print(f"\nMandelbrot image size: {SIZE} x {SIZE}")
print(f"Using {num_workers} CPU cores")
# -------------------------------------------------
# 2. Mandelbrot parameters
# -------------------------------------------------
MAX_ITER = 200
RE_START = -2.0
RE_END   = 1.0
IM_START = -1.5
IM_END   = 1.5
# -------------------------------------------------
# 3. Worker function
# -------------------------------------------------
def compute_rows(row_range):
    start_row, end_row = row_range
    block = []
    for y in range(start_row, end_row):
        row = []
        imag = IM_START + (y / SIZE) * (IM_END - IM_START)
        for x in range(SIZE):
            real = RE_START + (x / SIZE) * (RE_END - RE_START)
            c = complex(real, imag)
            z = 0
            iteration = 0
            while abs(z) <= 2 and iteration < MAX_ITER:
                z = z*z + c
                iteration += 1
            row.append(iteration)
        block.append(row)
    return (start_row, block)
# -------------------------------------------------
# 4. Split rows among CPUs
# -------------------------------------------------
chunk = SIZE // num_workers
row_blocks = []
for i in range(num_workers):
    start = i * chunk
    end = (i + 1) * chunk if i < num_workers - 1 else SIZE
    row_blocks.append((start, end))
# -------------------------------------------------
# 5. Parallel computation
# -------------------------------------------------
t0 = time.perf_counter()
with mp.Pool(num_workers) as pool:
    results = pool.map(compute_rows, row_blocks)
# -------------------------------------------------
# 6. Assemble final image
# -------------------------------------------------
image = [[0]*SIZE for _ in range(SIZE)]
for start_row, block in results:
    for i, row in enumerate(block):
        image[start_row + i] = row
t1 = time.perf_counter()
elapsed = t1 - t0
print(f"Mandelbrot computation time: {elapsed:.4f} seconds")
# -------------------------------------------------
# 7. Performance estimation
# -------------------------------------------------
# Approximate operations per iteration ~10 FLOPs
total_flops = SIZE * SIZE * MAX_ITER * 10
gflops = total_flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.2f} GFLOPS")
# -------------------------------------------------
# 8. Optional image saving (requires Pillow)
# -------------------------------------------------
try:
    from PIL import Image
    img = Image.new("L", (SIZE, SIZE))
    pixels = []
    for row in image:
        for val in row:
            pixels.append(int(255 * val / MAX_ITER))
    img.putdata(pixels)
    img.save("mandelbrot_scalar.png")
    print("Image saved as mandelbrot_scalar.png")
except ImportError:
    print("Pillow not installed, skipping image saving.")

