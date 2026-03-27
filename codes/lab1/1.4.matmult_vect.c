#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage: %s MATRIX_SIZE\n", argv[0]);
        printf("Example: %s 256\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    printf("Matrix multiplication in C\n");
    printf("Matrix size: %d x %d\n", N, N);
    // Allocate matrices
    double **A = malloc(N * sizeof(double*));
    double **B = malloc(N * sizeof(double*));
    double **C = malloc(N * sizeof(double*));
    for(int i=0; i<N; i++) {
        A[i] = malloc(N * sizeof(double));
        B[i] = malloc(N * sizeof(double));
        C[i] = malloc(N * sizeof(double));
    }
    // Initialize matrices with random values
    srand((unsigned int)time(NULL));
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
    // ---------------------------
    // Measure execution time
    // ---------------------------
    clock_t start = clock();
    // Standard matrix multiplication (C = A * B)
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            double sum = 0.0;
            for(int k=0; k<N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    clock_t end = clock();
    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %.6f seconds\n", elapsed_sec);
    // ---------------------------
    // Prevent compiler optimization
    // ---------------------------
    double checksum = 0.0;
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            checksum += C[i][j];
    printf("Checksum: %.6f\n", checksum);
    // ---------------------------
    // Estimate FLOPs and GFLOPS
    // ---------------------------
    double flops = 2.0 * N * N * N;  // 2*N^3 FLOPs
    double gflops = flops / elapsed_sec / 1e9;
    printf("Estimated performance: %.2f GFLOPS\n", gflops);
    // ---------------------------
    // Free memory
    // ---------------------------
    for(int i=0; i<N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A); free(B); free(C);
    return 0;
}

