# average_vector_float64.py
import numpy as np
import time
size = 1_000_000
# Create vector (float64)
a = np.random.rand(size)
start = time.time()
# Vectorized computation of average
average = np.mean(a)
end = time.time()
elapsed = end - start
print(f"Vectorized average: {average}")
print(f"Time: {elapsed:.6f} sec")
# -------------------------------------------------
# Performance estimation in GFLOPs
# -------------------------------------------------
# np.mean performs sum of 'size' elements (1 addition per element)
# and 1 division at the end
flops = size + 1
gflops = flops / elapsed / 1e9

print(f"Estimated performance: {gflops:.6f} GFLOPs")
import numpy as np
import time
size = 1_000_000
# Create vector (float64)
a = np.random.rand(size)
start = time.time()
# Vectorized computation of average
average = np.mean(a)
end = time.time()
elapsed = end - start
print(f"Vectorized average: {average}")
print(f"Time: {elapsed:.6f} sec")
# -------------------------------------------------
# Performance estimation in GFLOPs
# -------------------------------------------------
# np.mean performs sum of 'size' elements (1 addition per element)
# and 1 division at the end
flops = size + 1
gflops = flops / elapsed / 1e9
print(f"Estimated performance: {gflops:.6f} GFLOPs")

