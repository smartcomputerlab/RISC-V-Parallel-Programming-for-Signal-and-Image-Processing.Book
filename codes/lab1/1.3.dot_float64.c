#include <stdio.h>
#include <stdlib.h>
#include <time.h>
size_t N = 10000000;
double dot_f64(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += a[i] * b[i];  // 1 multiplication + 1 addition per iteration
    return sum;
}

int main(int argc, char **argv)
{
    clock_t start, end;
    double elapsed_sec;
    if (argc != 2) {
        printf("Usage: %s vector_size\n", argv[0]); return 1; }
    N = (size_t) atoll(argv[1]);
    // Allocate memory
    double *a = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));
    if (!a || !b) {  printf("Memory allocation failed\n"); return 1;}
    // Initialize vectors
    for (size_t i = 0; i < N; i++) {  a[i] = 1.0; b[i] = 2.0; }
    double result = 0.0;
    // Start timing
    start = clock();
    result = dot_f64(a, b, N);
    end = clock();
    elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Scalar execution time in sec: %.6f\n", elapsed_sec);
    printf("Scalar dot product = %.6f\n", result);
    // -------------------------------------------------
    // Performance estimation in GFLOPs
    // -------------------------------------------------
    // Each iteration: 1 multiplication + 1 addition = 2 FLOPs
    double flops_per_element = 2.0;
    double total_flops = N * flops_per_element;
    double gflops = total_flops / elapsed_sec / 1e9;
    printf("Estimated performance: %.6f GFLOPs\n", gflops);
    free(a); free(b);
    return 0;
}

