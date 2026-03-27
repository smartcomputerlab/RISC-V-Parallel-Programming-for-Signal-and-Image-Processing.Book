# Introduction
This book is devoted to exploring how modern RISC-V processors can provide an effective solution for parallel programming and high-performance processing using Python and optimized scientific libraries.
RISC-V represents a significant shift in the field of processor design. Unlike traditional proprietary instruction set architectures, RISC-V is open, modular, and freely available. This openness has encouraged rapid innovation in universities, research laboratories, and industry alike. Designers are free to implement, extend, and optimize the architecture according to the needs of specific applications. The clean and well-structured design of RISC-V also makes it particularly suitable for both high-performance computing platforms and modern embedded systems. As a result, it has become one of the most promising foundations for the next generation of computing technologies.

A central theme of this book is performance through parallelism. Modern RISC-V processors combine multicore architectures with powerful vector-processing capabilities. Multicore systems improve performance by executing multiple tasks simultaneously, while the RISC-V vector extension enables a single instruction to operate on multiple data elements at once. This form of data-level parallelism is especially well suited to applications in signal and image processing, where the same operations are repeatedly applied to large data sets. By combining multicore execution with vector operations, RISC-V offers a scalable and energy-efficient computing model capable of addressing demanding real-world problems.
Another important aspect of this book is the role of Python as a high-level development environment. Python has become one of the most widely used languages in scientific and engineering applications thanks to its clear syntax, readability, and extensive ecosystem of libraries.

Optimized libraries such as NumPy, SciPy, and OpenCV allow developers to implement complex algorithms quickly and with relatively little code. Although Python itself is a high-level language, many of its scientific libraries rely on highly optimized low-level implementations. This makes it possible to combine the productivity of Python with the performance of optimized numerical routines, including those that take advantage of vector instructions and other hardware-acceleration features available on modern RISC-V processors.

This book aims to bridge the gap between architectural concepts and practical algorithm development. It explains how computationally intensive operations can be implemented efficiently while remaining accessible to readers who may not have extensive experience in low-level programming. Through a series of practical examples, the book demonstrates how modern RISC-V processors and high-level programming tools can work together to create efficient, portable, and scalable solutions.
Ultimately, this book is intended for students, researchers, and engineers who are interested in high-performance computing, signal processing, image processing, and modern processor architectures.

Whether the reader is approaching RISC-V for the first time or seeking to deepen their understanding of parallel computing techniques, the goal of this work is to provide a clear and practical path toward efficient algorithm design on one of the most promising computing platforms of our time


# Table of Contents
0. Introduction	6
0.1 From SciPy/NumPy to Python, C and -RISC-V Assembly Programming Stack	7
0.2 Example Use-Cases	9
0.3 RISC-V platform for SIMD/MIMD programming	10
0.4 The NumPy package	11
0.5 Functionalities of the NumPy and SciPy libraries	12
0.6 Functionalities of the SciPy Library	15
0.7 Common filtering and image-processing functions with descriptions.	17
0.8 How SciPy uses NumPy	20

Lab 1 : SIMD - Basic vector programming and processing	21
1.1 Adding two vectors	21
1.1.1 Serial execution (interpretation): addser_float64.py (default)	21
1.1.2 Parallel (RVV) execution	24
1.1.3 Parallel execution with optimized C code	26
1.2 Average value of vector	28
1.2.1 Serial average	28
2.2.1 Vector average (NumPy mean() operator)	28
1.3 dot product	30
1.3.1 NumPy (accelerated) implementaion with dot operator	30
1.3.2 C (RVV optimized) implementation	31
1.4 Matrix multiplication	32
1.4.1 Matrix multiplication with NumPy @ operator	32
1.4.2 C (RVV optimized) implementation	33
1.5 Pi value calculations	35
1.5.1 NumPy implemetation with RVV acceleration	35
1.5.2 C (accelerated) implemetation	36

Lab 2 :  Simple Signal Processing	37
2.1 Signal processing elements	37
2.1.1 Sine-Wave generation and plotting	37
2.1.2 Add noise to a signal	37
2.1.3 Moving average filter (simple smoothing)	38
2.1.4 Compute the spectrum (FFT)	38
2.1.5 Signal integration	40
2.2 Low-pass FIR filter: function and performance with RVV	41
2.2.1 FIR implementation with NumPy (hamming(), convolve() ,..)	42
2.2.2 FIR implementation with C (optimized)	45
2.2.3 FIR implementation with SciPy signal.lfilter(h,1.0,signal_in)	46
2.3 Simple FFT processing	48
2.3.1 Optimized implementation with Numpy fft.fft(signal)	48
2.3.2 Implementation with “pure” Python (no vector instructions)	51
2.3.3 C implementation with fftw3.h	53
2.4 Generating Synthetic Audio	55
2.5 Fully Vectorized Audio Processing	58
2.6 Audio feature extraction (matrix calculation)	63
2.6.1 Vectorized version with NumPy	63
2.6.2 Pure Python version (non vectorized)	68

Lab 3 :  Simple Image Processing	70
3.1 Simple image generation (circle)	70
3.2 Simple image generation (line)	72
3.2.1 Version with openCV and RVV	72
3.2.2 Image – line generation with NumPy and RVV	73
3.2.3 Image – line generation with pure Python	75
3.3 Simple image negation	77
3.3.1 Image negation accelerated with vectorized NumPy	77
3.3.2 Image negation with pure Python (no acceleration)	78
3.4 Image conversion: RGB to grayscale	80
3.4.1 Image conversion with NumPy accleration	80
3.4.2 Image conversion with pure Python (no accleration)	81
3.5 Image resize (cv2.resize()) + grayscale np.clip() conversion	83
3.5.1 Image resize with NumPy acceleration	83
3.5.2 Image resize -with pure Python - no acceleration	84
3.6 Image Rotation	87
3.6.1 Image rotation with np.transpose()	87
3.6.2 Image rotation without np.transpose()	88
3.7 Image conversion RGB to HSV	90
3.7.1 Image conversion with NumPy acceleration	90
3.7.2 Image conversion - no acceleration	91
3.7.3 Image back-conversion HSV to RGB	93

Lab 4 :  MIMD (+SIMD) processing with Python	95
4.1 Purpose and functions of the multiprocessing library	95
            4.1.1 Process Creation	95
4.1.2 Process Pools	96
4.1.3 Pool Functions	96
4.1.4 Example: Parallel Image Processing	96
4.2 Vector addition with multiprocessing	97
4.2.1 Vector addition: multiprocessing and NumPy (vector)	97
4.2.2 Vector addition: multiprocessing and pure Python (no vector)	98
4.3 Pi calculation	100
4.3.1 Pi multiprocessing and vector (NumPy)	100
4.3.2 Pi multiprocessing and pure Python (no vector)	101
4.4 Matrix muliplication	103
4.4.1 Matrix muliplication : multiprocessing and vector (NumPy)	103
4.4.2 Matrix muliplication : multiprocessing only	106
4.5 Image comparison: multiprocessing	108
4.5.1 RGB image generation	108
4.5.2 RGB image difference with multiprocessing and NumPy	108
4.5.3 RGB image difference with multiprocessing only	110

Lab 5 :  Image processing (filtering) with multiprocessing and vector instructions	113
5.1 Mandelbrot set	113
5.1.1 Mandelbrot set: vector (NumPy) and multiprocessing	113
5.1.2 Mandelbrot set : “pure” multiprocessing	114
5.2 Sobel filter	119
5.2.1 Sobel filter : multiprocessing only	119
5.2.2 Sobel filter: vector (NumPy) + multiprocessing	121
5.2.3 Sobel filter: vector (SciPy) + multiprocessing	125
5.3 Gaussian blur	127
5.3.1 Gaussian blur: vector acceleration only with NumPy	127
5.3.2 Gaussian blur: multiprocessing only	129
5.3.3 Gaussian blur: multiprocessing + vector (NumPy)	131
5.3.4 Gaussian blur: multiprocessing + vector (SciPy: gaussian_filter())	133
5.3.5 Gaussian blur: vectorized C and openMP	135

6. Summary	138
