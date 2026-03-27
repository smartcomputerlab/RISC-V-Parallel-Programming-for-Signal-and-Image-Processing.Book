# matrix_mul_float32_mp_arg2.py
import time
import sys
import random
import struct
import multiprocessing as mp
# -------------------------------------------------
# 1. Read command-line arguments
# -------------------------------------------------
if len(sys.argv) > 1:
    N = int(sys.argv[1])  # matrix size
else:
    N = 200  # default size

if len(sys.argv) > 2:
    num_workers = int(sys.argv[2])  # number of CPU cores
else:
    num_workers = mp.cpu_count()  # default: all available cores
print(f"\nMatrix size: {N} x {N}")
print(f"Using {num_workers} CPU cores")
# -------------------------------------------------
# 2. Generate matrices (float32) without NumPy
# -------------------------------------------------
def random_float32():
    """Return a random float32 number."""
    return struct.unpack('f', struct.pack('f', random.random()))[0]
A = [[random_float32() for _ in range(N)] for _ in range(N)]
B = [[random_float32() for _ in range(N)] for _ in range(N)]
# Shared memory for result matrix C
manager = mp.Manager()
C = manager.list([[0.0 for _ in range(N)] for _ in range(N)])
print("Matrices A and B generated (float32)")
# -------------------------------------------------
# 3. Worker function to compute a block of rows
# -------------------------------------------------
def compute_rows(row_range):
    start, end = row_range
    for i in range(start, end):
        for j in range(N):
            sum_ = 0.0
            for k in range(N):
                sum_ += A[i][k] * B[k][j]
            C[i][j] = sum_
# -------------------------------------------------
# 4. Split rows among specified CPU cores
# -------------------------------------------------
chunk = N // num_workers
blocks = []
for i in range(num_workers):
    start = i * chunk
    end = (i + 1) * chunk if i < num_workers - 1 else N
    blocks.append((start, end))
# -------------------------------------------------
# 5. Run parallel multiplication
# -------------------------------------------------
start_time = time.perf_counter()
processes = []
for block in blocks:
    p = mp.Process(target=compute_rows, args=(block,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\nMatrix multiplication completed in {elapsed:.6f} seconds")
# -------------------------------------------------
# 6. Performance estimation
# -------------------------------------------------
flops = 2 * N**3
gflops = flops / elapsed / 1e9
print("\n--- Performance estimation ---")
print(f"Estimated FLOPs: {flops:.2e}")
print(f"Estimated performance: {gflops:.6f} GFLOPs")
# -------------------------------------------------
# 7. Memory estimation
# -------------------------------------------------
bytes_per_element = 4  # float32
total_bytes = 3 * N * N * bytes_per_element  # A, B, C
bandwidth = total_bytes / elapsed / 1e9
print("\n--- Memory estimation ---")
print(f"Total memory moved: {total_bytes/1e6:.2f} MB")
print(f"Estimated bandwidth: {bandwidth:.2f} GB/s")

