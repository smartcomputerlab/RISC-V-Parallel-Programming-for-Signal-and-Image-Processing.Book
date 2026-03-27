import numpy as np
import time
import multiprocessing as mp
import sys

# -----------------------------
# Worker function
# -----------------------------
def worker_add(args):
    a_chunk, b_chunk = args
    return (a_chunk + b_chunk) & 0xFF  # 8-bit wrap-around: using RVV instructions !

def checksum(buf):
    return int(np.sum(buf))

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 add_mp.py VECTOR_SIZE(KB) NUM_PROCESSES")
        return

    N = int(sys.argv[1]) * 1024  # total bytes
    nproc = int(sys.argv[2])     # number of processes

    print(f"Vector size: {N} bytes, Processes: {nproc}")

    # -----------------------------
    # Create vectors
    # -----------------------------
    A = np.array([i & 0xFF for i in range(N)], dtype=np.uint8)
    B = np.array([(3*i + 7) & 0xFF for i in range(N)], dtype=np.uint8)

    # Split vectors into chunks
    chunk_size = N // nproc
    chunks = []

    for i in range(nproc):
        start = i * chunk_size
        end = N if i == nproc - 1 else (i + 1) * chunk_size
        chunks.append((A[start:end], B[start:end]))

    # -----------------------------
    # Multiprocessing pool
    # -----------------------------
    t0 = time.time()

    with mp.Pool(processes=nproc) as pool:
        results = pool.map(worker_add, chunks)

    t1 = time.time()

    # Combine results
    C = np.concatenate(results)

    # -----------------------------
    # Performance
    # -----------------------------
    elapsed = t1 - t0

    print(f"Multiprocessing time: {elapsed:.6f} s")
    print(f"Checksum: {checksum(C)}")

    # -----------------------------
    # Memory bandwidth estimation
    # -----------------------------
    # 3 vectors moved: read A + read B + write C
    bytes_moved = 3 * N
    bw = bytes_moved / elapsed / 1e9

    print(f"Estimated memory bandwidth: {bw:.3f} GB/s")

    # -----------------------------
    # GFLOPs estimation
    # -----------------------------
    # One arithmetic operation per element (A[i] + B[i])
    operations = N

    gflops = operations / elapsed / 1e9

    print(f"Estimated performance: {gflops:.6f} GFLOPs")


if __name__ == "__main__":
    main()


