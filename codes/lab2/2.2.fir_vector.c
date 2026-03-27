#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main() {
    // ---------------------------
    // 1. Parameters
    // ---------------------------
    const int fs = 1000;          // Sampling frequency
    const double T = 10000.0;     // Duration in seconds
    const long N = (long)(fs * T); // Number of samples (~10M)
    const int num_taps = 101;     // Filter length
    const double cutoff = 10.0;   // Low-pass cutoff frequency
    printf("FIR Filtering in C (double) with %ld samples\n", N);
    printf("Filter length: %d taps\n", num_taps);
    // ---------------------------
    // 2. Allocate memory
    // ---------------------------
    double *signal = malloc(N * sizeof(double));
    double *filtered = malloc(N * sizeof(double));
    double *h = malloc(num_taps * sizeof(double));
    if(signal == NULL || filtered == NULL || h == NULL) {
        printf("Memory allocation failed!\n"); return 1;
    }
    // ---------------------------
    // 3. Generate input signal (5 Hz sine + noise)
    // ---------------------------
    srand(0);
    for(long i = 0; i < N; i++) {
        double t = (double)i / fs;
        signal[i] = 0.5 * sin(2.0 * M_PI * 5.0 * t) + 0.05 * ((double)rand() / RAND_MAX - 0.5);
    }
    // ---------------------------
    // 4. Design FIR filter (sinc + Hamming)
    // ---------------------------
    double fc = cutoff / (fs / 2.0); // normalized cutoff
    for(int n = 0; n < num_taps; n++) {
        double k = n - (num_taps - 1) / 2.0;
        h[n] = (k == 0.0) ? 2.0 * fc : sin(2.0 * M_PI * fc * k) / (M_PI * k);
        h[n] *= 0.54 - 0.46 * cos(2.0 * M_PI * n / (num_taps - 1)); // Hamming
    }
    // Normalize filter
    double sum_h = 0.0;
    for(int n = 0; n < num_taps; n++) sum_h += h[n];
    for(int n = 0; n < num_taps; n++) h[n] /= sum_h;
    // ---------------------------
    // 5. FIR filtering with timing
    // ---------------------------
    clock_t start = clock();
    for(long i = 0; i < N; i++) {
        double acc = 0.0;
        for(int j = 0; j < num_taps; j++) {
            long idx = i - j;
            if(idx >= 0)
                acc += signal[idx] * h[j];
        }
        filtered[i] = acc;
    }
    clock_t end = clock();
    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %.6f seconds\n", elapsed_sec);
    // ---------------------------
    // 6. Compute checksum to prevent optimization
    // ---------------------------
    double checksum = 0.0;
    for(long i = 0; i < N; i++) checksum += filtered[i];
    printf("Checksum: %.6f\n", checksum);
    // ---------------------------
    // 7. Estimate FLOPs and GFLOPS
    // ---------------------------
    double flops = 2.0 * N * num_taps; // 1 multiply + 1 add per tap per output
    double gflops = flops / elapsed_sec / 1e9;
    printf("Estimated performance: %.2f GFLOPS\n", gflops);
    // ---------------------------
    // 8. Free memory
    // ---------------------------
    free(signal);
    free(filtered);
    free(h);
    return 0;
}
