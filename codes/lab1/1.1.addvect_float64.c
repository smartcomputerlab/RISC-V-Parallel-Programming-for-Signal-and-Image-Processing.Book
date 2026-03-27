//  add_vect_float64_perf.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 10000000

int main() {
    // Allocate memory
    double *a = (double*) malloc(SIZE * sizeof(double));
    double *b = (double*) malloc(SIZE * sizeof(double));
    double *c = (double*) malloc(SIZE * sizeof(double));
    if (a == NULL || b == NULL || c == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    // Initialize vectors
    for (long i = 0; i < SIZE; i++) {
        a[i] = (double) rand() / RAND_MAX;
        b[i] = (double) rand() / RAND_MAX;
    }
    // Start timing (only addition loop)
    clock_t start = clock();
    for (long i = 0; i < SIZE; i++) {
        c[i] = a[i] + b[i];
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time for %d double additions: %f seconds\n", SIZE, time_spent);
    // -------------------------------------------------
    // Performance estimation in GFLOPs
    // -------------------------------------------------
    // Each iteration performs 1 floating-point addition
    double flops = (double) SIZE;
    double gflops = flops / (time_spent * 1e9);
    printf("Estimated performance: %f GFLOPs\n", gflops);
    // Simple checksum (avoid optimization removal)
    double checksum = 0.0;
    for (long i = 0; i < SIZE; i++)
        checksum += c[i];
    printf("Checksum: %f\n", checksum);
    free(a);
    free(b);
    free(c);
    return 0;
}

