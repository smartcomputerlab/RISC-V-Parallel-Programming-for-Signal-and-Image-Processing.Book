# addvect_int32.py
import numpy as np
import time
size = 10_000_000
# Create two int32 arrays
a = np.random.randint(0, 1000, size=size, dtype=np.int32)
b = np.random.randint(0, 1000, size=size, dtype=np.int32)
s_time = time.time()
# Vectorized integer addition
c = a + b
e_time = time.time()
elapsed = e_time - s_time
print(f"Time of {size} : {elapsed:.4f} sec")
# -------------------------------------------------
# Performance estimation (integer operations)
# -------------------------------------------------
# Each element performs 1 integer addition
ops_per_element = 1
total_ops = size * ops_per_element
gops = total_ops / elapsed / 1e9

print(f"Estimated performance: {gops:.4f} GOPS")

