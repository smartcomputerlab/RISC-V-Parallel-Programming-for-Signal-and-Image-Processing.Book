import time
import multiprocessing as mp
import sys

# -----------------------------
# Worker function (pure Python)
# -----------------------------
def worker_add(args):
    a_chunk, b_chunk = args
    n = len(a_chunk)
    c_chunk = [0] * n
    for i in range(n):
        c_chunk[i] = (a_chunk[i] + b_chunk[i]) & 0xFF  # 8-bit wrap-around
    return c_chunk

def checksum(buf):
    return sum(buf)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 add_mp_py.py VECTOR_SIZE(KB) NUM_PROCESSES")
        return

    N = int(sys.argv[1]) * 1024  # total bytes
    nproc = int(sys.argv[2])     # number of processes

    print(f"Vector size: {N} bytes, Processes: {nproc}")

    # -----------------------------
    # Create vectors
    # -----------------------------
    A = [i & 0xFF for i in range(N)]
    B = [(3*i + 7) & 0xFF for i in range(N)]

    # -----------------------------
    # Split vectors into chunks
    # -----------------------------
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
    C = []
    for chunk in results:
        C.extend(chunk)

    # -----------------------------
    # Performance
    # -----------------------------
    elapsed = t1 - t0

    print(f"Multiprocessing time: {elapsed:.6f} s")
    print(f"Checksum: {checksum(C)}")

    # -----------------------------
    # Memory bandwidth estimation
    # -----------------------------
    # read A + read B + write C
    bytes_moved = 3 * N
    bw = bytes_moved / elapsed / 1e9

    print(f"Estimated memory bandwidth: {bw:.3f} GB/s")

    # -----------------------------
    # Performance estimation
    # -----------------------------
    # One addition per element
    operations = N

    gops = operations / elapsed / 1e9   # integer operations per second
    gflops = operations / elapsed / 1e9 # same formula, shown for comparison

    print(f"Estimated performance: {gops:.6f} GOPS")
    print(f"(Equivalent GFLOPs estimate): {gflops:.6f} GFLOPs")

if __name__ == "__main__":
    main()

