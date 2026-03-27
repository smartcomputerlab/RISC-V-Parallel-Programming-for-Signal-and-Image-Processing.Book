import multiprocessing as mp
import time
import sys
# -----------------------------
# Worker function (pure Python)
# -----------------------------
def worker_pi(args):
    start_idx, end_idx, dx = args
    total = 0.0
    for i in range(start_idx, end_idx):
        x = i * dx
        total += 4.0 / (1.0 + x*x)
    return total * dx
# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 pi_mp_py.py NUM_SAMPLES NUM_PROCESSES")
        return
    N = int(sys.argv[1])       # total number of rectangles
    nproc = int(sys.argv[2])   # number of processes
    print(f"Calculating Pi with {N} samples using {nproc} processes (pure Python inside workers)...")
    dx = 1.0 / N
    # Split indices for each process
    chunk_size = N // nproc
    args_list = []
    for i in range(nproc):
        start = i * chunk_size
        end = N if i == nproc-1 else (i+1)*chunk_size
        args_list.append((start, end, dx))
    # -----------------------------
    # Multiprocessing pool
    # -----------------------------
    t0 = time.time()
    with mp.Pool(processes=nproc) as pool:
        results = pool.map(worker_pi, args_list)
    t1 = time.time()
    # Sum results from all processes
    pi_est = sum(results)
    elapsed = t1 - t0
    print(f"Estimated Pi: {pi_est}")
    print(f"Error: {abs(3.141592653589793 - pi_est)}")
    print(f"Execution time: {elapsed:.6f} s")
    # -----------------------------
    # Estimate memory bandwidth
    # -----------------------------
    bytes_moved = nproc * chunk_size * 16  # float64 read + float64 accumulation
    bw = bytes_moved / elapsed / 1e9
    print(f"Estimated memory bandwidth: {bw:.3f} GB/s")
    # -----------------------------
    # Estimate GFLOPs
    # -----------------------------
    # Per iteration:
    # x = i * dx          -> 1 multiplication
    # x*x                 -> 1 multiplication
    # 1.0 + x*x           -> 1 addition
    # 4.0 / (...)         -> 1 division
    # total += ...        -> 1 addition
    # ------------------------------------
    flops_per_iter = 5
    total_flops = N * flops_per_iter
    gflops = total_flops / elapsed / 1e9
    print(f"Estimated performance: {gflops:.6f} GFLOPs")

# -----------------------------
if __name__ == "__main__":
    main()

