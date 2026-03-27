#addser_float64.py
import numpy as np
import time
N = 10_000_000
a = np.random.rand(N)
b = np.random.rand(N)
t0 = time.time()
# Serial addition using Python loop
c = [a[i] + b[i] for i in range(N)]
t1 = time.time()
elapsed = t1 - t0
print("Time:", elapsed)
# -------------------------------------------------
# Performance estimation in GFLOPs
# -------------------------------------------------
# Each iteration performs 1 floating-point addition
flops_per_element = 1
total_flops = N * flops_per_element
gflops = total_flops / elapsed / 1e9
print("Estimated performance:", gflops, "GFLOPs")

