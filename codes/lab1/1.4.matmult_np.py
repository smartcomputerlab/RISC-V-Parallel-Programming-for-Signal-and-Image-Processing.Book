# mmult_perf.py
import numpy as np
import time
import sys

def matrix_multiply(size):
    print("Matrix multiplication using NumPy")
    print("Matrix size:", size, "x", size)
    # Generate matrices (float64)
    A = np.random.rand(size, size).astype(np.float64)
    B = np.random.rand(size, size).astype(np.float64)
    # Warm-up (important for fair timing)
    C = A @ B
    # ---------------------------
    # Measure execution time
    # ---------------------------
    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    print(f"Execution time: {elapsed_time:.6f} seconds")
    # ---------------------------
    # Prevent optimization removal
    # ---------------------------
    print("Checksum:", np.sum(C))
    # ---------------------------
    # FLOP estimation
    # ---------------------------
    flops = 2 * size**3
    gflops = flops / elapsed_time / 1e9
    print(f"Total FLOPs: {flops:.3e}")
    print(f"Estimated performance: {gflops:.2f} GFLOPS")
    # ---------------------------
    # Memory bandwidth estimation
    # ---------------------------
    bytes_moved = 3 * size**2 * 8   # A read + B read + C write
    bandwidth = bytes_moved / elapsed_time / 1e9
    print(f"Data moved: {bytes_moved/1e6:.2f} MB")
    print(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python matrix_numpy.py SIZE")
        print("Example: python matrix_numpy.py 256")
        sys.exit(1)

    size = int(sys.argv[1])
    matrix_multiply(size)

