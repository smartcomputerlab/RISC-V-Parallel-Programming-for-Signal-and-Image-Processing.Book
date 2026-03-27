// gaussian_blur_rgb_omp.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#define KERNEL_SIZE 5
#define KERNEL_RADIUS 2
#define NORM 256.0

int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 4, 6, 4, 1},
    {4,16,24,16,4},
    {6,24,36,24,6},
    {4,16,24,16,4},
    {1, 4, 6, 4, 1}
};

int main(int argc, char **argv)
{
    if(argc < 2) {
        printf("Usage: %s image_file [num_threads]\n", argv[0]);
        return 1;
    }
    int num_threads = omp_get_max_threads();
    if(argc > 2) num_threads = atoi(argv[2]);
    printf("Gaussian Blur RGB with OpenMP\n");
    printf("Threads: %d\n", num_threads);
    // Load RGB image
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(img.empty()) {
        printf("Error loading image\n");
        return 1;
    }
    int HEIGHT = img.rows;
    int WIDTH  = img.cols;
    int CHANNELS = img.channels();
    printf("Image size: %dx%d, Channels: %d\n", WIDTH, HEIGHT, CHANNELS);
    // Convert to float
    cv::Mat img_f;
    img.convertTo(img_f, CV_32F);
    // Output image
    cv::Mat blur = cv::Mat::zeros(HEIGHT, WIDTH, CV_32FC3);
    // -------------------------------
    // Benchmark
    // -------------------------------
    double t0 = omp_get_wtime();
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(int y = KERNEL_RADIUS; y < HEIGHT - KERNEL_RADIUS; y++) {
        for(int x = KERNEL_RADIUS; x < WIDTH - KERNEL_RADIUS; x++) {
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for(int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
                for(int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                    cv::Vec3f pixel = img_f.at<cv::Vec3f>(y+ky, x+kx);
                    int weight = kernel[ky+KERNEL_RADIUS][kx+KERNEL_RADIUS];
                    for(int c = 0; c < 3; c++)
                        acc[c] += pixel[c] * weight;
                }
            }
            for(int c = 0; c < 3; c++)
                blur.at<cv::Vec3f>(y,x)[c] = acc[c] / NORM;
        }
    }
    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;
    printf("Processing time: %.6f s\n", elapsed);
    // -------------------------------
    // Performance estimation
    // -------------------------------
    int pixels = HEIGHT * WIDTH * CHANNELS;
    int flops_per_pixel = KERNEL_SIZE*KERNEL_SIZE*2; // mult + add
    double total_flops = (double)pixels * flops_per_pixel;
    double gflops = total_flops / elapsed / 1e9;
    printf("Estimated GFLOPS: %.3f\n", gflops);
    // Convert back to 8-bit and save
    cv::Mat blur_u8;
    blur.convertTo(blur_u8, CV_8UC3);
    cv::imwrite("gaussian_blur_rgb_omp.png", blur_u8);
    printf("Output saved: gaussian_blur_rgb_omp.png\n");
    return 0;
}

