import numpy as np
import time
# Number of samples
N = 50_000_000
print("Computing Pi using NumPy vector operations...")
print("Samples:", N)
t0 = time.perf_counter()
# Generate vector of x values
x = np.linspace(0.0, 1.0, N, dtype=np.float64)
# Vectorized computation
y = 4.0 / (1.0 + x*x)
# Integration (mean value method)
pi_est = np.mean(y)
t1 = time.perf_counter()
elapsed_time = t1 - t0
print("Estimated Pi =", pi_est)
print("Error =", abs(np.pi - pi_est))
print("Time =", elapsed_time, "seconds")
# ---------------------------
# Estimate FLOPs and GFLOPS
# ---------------------------
# FLOPs: x*x (1 mul) + 1 + (1 div) + mean (N adds + 1 div)
flops = 4 * N
gflops = flops / elapsed_time / 1e9
print(f"Estimated performance: {gflops:.2f} GFLOPS")

