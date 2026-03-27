#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage: %s NUM_SAMPLES\n", argv[0]);
        printf("Example: %s 50000000\n", argv[0]); return 1;
    }
    long N = atol(argv[1]);
    printf("Computing Pi using C (float64)\n");
    printf("Number of samples: %ld\n", N);
    double *x = malloc(N * sizeof(double));
    double *y = malloc(N * sizeof(double));
    if(x == NULL || y == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Generate x values
    for(long i = 0; i < N; i++) {
        x[i] = (double)i / (double)(N - 1);
    }
    clock_t start = clock();
    // Vectorized-like computation
    for(long i = 0; i < N; i++) {
        y[i] = 4.0 / (1.0 + x[i]*x[i]);
    }
    // Integration (mean)
    double sum = 0.0;
    for(long i = 0; i < N; i++) { sum += y[i]; }
    double pi_est = sum / (double)N;
    clock_t end = clock();
    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Estimated Pi = %.15f\n", pi_est);
    printf("Error = %.15f\n", fabs(M_PI - pi_est));
    printf("Execution time = %.6f seconds\n", elapsed_sec);
    // ---------------------------
    // Estimate FLOPs and GFLOPS
    // ---------------------------
    // x*x (1 mul), 1.0 + x*x (1 add), 4.0 / (...) (1 div) per element
    // mean: N adds + 1 div → total ~4*N FLOPs
    double flops = 4.0 * N;
    double gflops = flops / elapsed_sec / 1e9;
    printf("Estimated performance: %.2f GFLOPS\n", gflops);
    free(x);free(y);
    return 0;
}

