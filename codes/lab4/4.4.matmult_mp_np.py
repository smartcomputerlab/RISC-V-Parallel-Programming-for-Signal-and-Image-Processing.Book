import numpy as np
import multiprocessing as mp
import time
import sys
# -------------------------------------------------
# 1. Read matrix size from command line
# -------------------------------------------------
if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 2000   # default size
print(f"\nMatrix size: {N} x {N}")
# -------------------------------------------------
# 2. Generate matrices
# -------------------------------------------------
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
num_workers = mp.cpu_count()
print(f"CPU cores used: {num_workers}")
# -------------------------------------------------
# 3. Worker function
# -------------------------------------------------
def compute_block(rows):
    start, end = rows
    # Vectorized matrix multiplication
    return np.dot(A[start:end], B)
# -------------------------------------------------
# 4. Divide matrix rows into blocks
# -------------------------------------------------
chunk = N // num_workers
blocks = []
for i in range(num_workers):
    start = i * chunk
    end = (i + 1) * chunk if i < num_workers - 1 else N
    blocks.append((start, end))
# -------------------------------------------------
# 5. Parallel execution
# -------------------------------------------------
start_time = time.perf_counter()
with mp.Pool(num_workers) as pool:
    results = pool.map(compute_block, blocks)
# Combine results
C = np.vstack(results)
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\nExecution time: {elapsed:.4f} seconds")
# -------------------------------------------------
# 6. Performance estimation
# -------------------------------------------------
# FLOPs for matrix multiplication
# 2 * N^3
flops = 2 * (N**3)
gflops = flops / elapsed / 1e9
print("\n--- Performance estimation ---")
print(f"Estimated FLOPs: {flops:.2e}")
print(f"Performance: {gflops:.2f} GFLOPs")
# -------------------------------------------------
# 7. Memory bandwidth estimation
# -------------------------------------------------
bytes_A = N * N * 8
bytes_B = N * N * 8
bytes_C = N * N * 8
total_bytes = bytes_A + bytes_B + bytes_C
bandwidth = total_bytes / elapsed / 1e9
print("\n--- Memory estimation ---")
print(f"Memory moved: {total_bytes/1e9:.2f} GB")
print(f"Estimated bandwidth: {bandwidth:.2f} GB/s")

