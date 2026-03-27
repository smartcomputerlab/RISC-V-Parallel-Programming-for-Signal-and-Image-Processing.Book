// fft_signal_benchmark.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#define FS 1000.0
#define T 1.0
int main() {
    int N = (int)(FS * T);
    double *signal;
    fftw_complex *fft_out;
    fftw_plan plan;
    signal = (double*) fftw_malloc(sizeof(double) * N);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2 + 1));
    double t;
    double f1 = 50.0;
    double f2 = 120.0;
    // ---------------------------
    // Generate signal
    // ---------------------------
    for(int i = 0; i < N; i++) {
        t = (double)i / FS;
        signal[i] = sin(2*M_PI*f1*t) + 0.5*sin(2*M_PI*f2*t);
    }
    // ---------------------------
    // Create FFT plan
    // ---------------------------
    plan = fftw_plan_dft_r2c_1d(N, signal, fft_out, FFTW_ESTIMATE);
    // ---------------------------
    // Warm-up FFT
    // ---------------------------
    fftw_execute(plan);
    // ---------------------------
    // Measure FFT execution time
    // ---------------------------
    clock_t start = clock();
    fftw_execute(plan);
    clock_t end = clock();
    double execution_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("FFT size: %d\n", N);
    printf("Execution time: %f seconds\n", execution_time);
    // ---------------------------
    // Estimate FFT FLOPs
    // ---------------------------
    double flops_est = 5.0 * N * log2((double)N);
    double gflops = flops_est / execution_time / 1e9;
    printf("Estimated performance: %.2f GFLOPS\n", gflops);
    // ---------------------------
    // Print part of spectrum
    // ---------------------------
    printf("\nFrequency Spectrum (first 10 bins)\n");
    for(int i = 0; i < 10; i++) {
        double freq = (double)i * FS / N;
        double real = fft_out[i][0];
        double imag = fft_out[i][1];
        double magnitude = sqrt(real*real + imag*imag);
        printf("%6.1f Hz  ->  %f\n", freq, magnitude);
    }
    // ---------------------------
    // Cleanup
    // ---------------------------
    fftw_destroy_plan(plan);
    fftw_free(signal);
    fftw_free(fft_out);
    return 0;
}
