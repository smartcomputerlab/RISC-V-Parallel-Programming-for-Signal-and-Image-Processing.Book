# mandelbrot_mp_numpy.py
import numpy as np
import multiprocessing as mp
import time
import sys
# -------------------------------------------------
# 1. Read image size from command line
# -------------------------------------------------
if len(sys.argv) > 1:
    SIZE = int(sys.argv[1])
else:
    SIZE = 1000   # default
print(f"\nMandelbrot image size: {SIZE} x {SIZE}")
# -------------------------------------------------
# 2. Mandelbrot parameters
# -------------------------------------------------
MAX_ITER = 200
RE_START, RE_END = -2.0, 1.0
IM_START, IM_END = -1.5, 1.5
# -------------------------------------------------
# 3. Worker function
# -------------------------------------------------
def mandelbrot_rows(row_range):
    start_row, end_row = row_range
    rows = end_row - start_row
    # Create coordinate grid for these rows
    y = np.linspace(IM_START, IM_END, SIZE)[start_row:end_row]
    x = np.linspace(RE_START, RE_END, SIZE)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    output = np.zeros(C.shape, dtype=np.int32)
    for i in range(MAX_ITER):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        output[mask] = i
    return (start_row, output)
# -------------------------------------------------
# 4. Split rows among CPUs
# -------------------------------------------------
num_workers = mp.cpu_count()
chunk = SIZE // num_workers
row_blocks = []
for i in range(num_workers):
    start = i * chunk
    end = (i + 1) * chunk if i < num_workers - 1 else SIZE
    row_blocks.append((start, end))
print(f"Using {num_workers} CPU cores")
# -------------------------------------------------
# 5. Parallel computation
# -------------------------------------------------
t0 = time.perf_counter()
with mp.Pool(num_workers) as pool:
    results = pool.map(mandelbrot_rows, row_blocks)
# -------------------------------------------------
# 6. Assemble final image
# -------------------------------------------------
image = np.zeros((SIZE, SIZE), dtype=np.int32)
for start_row, block in results:
    image[start_row:start_row + block.shape[0], :] = block
t1 = time.perf_counter()
elapsed = t1 - t0
print(f"Mandelbrot computation time: {elapsed:.4f} seconds")
# -------------------------------------------------
# 7. Performance estimation
# -------------------------------------------------
# Approximate operations per iteration: ~10 FLOPs
total_flops = SIZE * SIZE * MAX_ITER * 10
gflops = total_flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.2f} GFLOPS")
# -------------------------------------------------
# 8. Save image
# -------------------------------------------------
try:
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap="hot", extent=[RE_START, RE_END, IM_START, IM_END])
    plt.title("Mandelbrot Set")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.colorbar()
    plt.savefig("mandelbrot.png", dpi=300)
    print("Image saved as mandelbrot.png")

except ImportError:
    print("Matplotlib not installed. Skipping image saving.")

