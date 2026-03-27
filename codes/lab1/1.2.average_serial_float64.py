# average_serial_float64.py
import numpy as np
import time
size = 1_000_000
# Create vector (float64)
a = np.random.rand(size)
start = time.time()
# Serial computation of sum
total = 0.0
for i in range(size):
    total += a[i]

average = total / size
end = time.time()
elapsed = end - start
print(f"Serial average: {average}")
print(f"Time: {elapsed:.6f} sec")
# -------------------------------------------------
# Performance estimation in GFLOPs
# -------------------------------------------------
# Each iteration performs 1 addition for sum
# Division for average counts as 1 FLOP
flops = size + 1
gflops = flops / elapsed / 1e9

print(f"Estimated performance: {gflops:.6f} GFLOPs")

