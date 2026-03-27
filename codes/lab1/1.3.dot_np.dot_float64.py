# dot_product_numpy_bw.py
import numpy as np
import time
# Vector size
N = 10_000_000   # 10 million elements
# Create double precision vectors
a = np.random.rand(N).astype(np.float64)
b = np.random.rand(N).astype(np.float64)
# Warm-up (important for fair timing)
np.dot(a[:1000], b[:1000])
# -----------------------------
# Measure execution time
# -----------------------------
start = time.perf_counter()
result = np.dot(a, b)       # NumPy BLAS / vectorized kernel (RVV if supported)
end = time.perf_counter()
elapsed = end - start
# -----------------------------
# Performance metrics
# -----------------------------
print(f"Dot product result: {result}")
print(f"Vector size: {N}")
print(f"Execution time: {elapsed:.6f} seconds")
# FLOPS estimation
flops = 2 * N                 # 1 multiply + 1 add per element
gflops = flops / elapsed / 1e9
print(f"Estimated FLOPs: {flops:.3e}")
print(f"Performance: {gflops:.2f} GFLOPS")
# -----------------------------
# Memory bandwidth estimation
# -----------------------------
bytes_read = 2 * N * 8        # read a[i] and b[i] (float64)
bandwidth = bytes_read / elapsed / 1e9
print(f"Data moved: {bytes_read/1e6:.2f} MB")
print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")

