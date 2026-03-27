# addser_int8.py
import numpy as np
import time
N = 10_000_000
# Create two int8 arrays
a = np.random.randint(-128, 127, size=N, dtype=np.int8)
b = np.random.randint(-128, 127, size=N, dtype=np.int8)
t0 = time.time()
# Serial element-by-element addition
c = [int(a[i]) + int(b[i]) for i in range(N)]
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

