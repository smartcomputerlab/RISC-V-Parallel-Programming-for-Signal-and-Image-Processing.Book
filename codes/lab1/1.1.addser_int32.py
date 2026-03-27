# addser_int32.py
import numpy as np
import time
N = 10_000_000
# Create two int32 arrays
a = np.random.randint(0, 1000, size=N, dtype=np.int32)
b = np.random.randint(0, 1000, size=N, dtype=np.int32)
t0 = time.time()
# Serial element-by-element addition (no vectorization)
c = [a[i] + b[i] for i in range(N)]
t1 = time.time()
elapsed = t1 - t0
print("Time:", elapsed)
# -------------------------------------------------
# Performance estimation (integer operations)
# -------------------------------------------------
# Each iteration performs 1 integer addition
ops_per_element = 1
total_ops = N * ops_per_element
gops = total_ops / elapsed / 1e9
print("Estimated performance:", gops, "GOPS")

