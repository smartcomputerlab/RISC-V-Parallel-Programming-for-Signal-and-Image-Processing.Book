# addvect_float64.py
import numpy as np
import time
size = 10_000_000
a = np.random.rand(size).astype(np.float64)
b = np.random.rand(size).astype(np.float64)
s_time = time.time()
# Vectorized addition
c = a + b
e_time = time.time()
elapsed = e_time - s_time
print(f"Time of {size} : {elapsed:.4f} sec")
# -------------------------------------------------
# Performance estimation in GFLOPs
# -------------------------------------------------
# Each element requires 1 floating-point addition
flops_per_element = 1
total_flops = size * flops_per_element
gflops = total_flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.4f} GFLOPs")

