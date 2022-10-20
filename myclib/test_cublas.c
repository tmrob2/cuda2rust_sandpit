//#include <iostream>
#include<stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// input is a host device mem vector
void cuda_call_spaxy(const float *x, float *y, int N, float alpha ) {
    cudaError_t cudaStat;
    // create a device handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // allocate the GPU memory for x, and y
    float *d_x, *d_y;

    cudaStat = cudaMalloc ((void**)&d_x, N*sizeof(float));
    cudaStat = cudaMalloc ((void**)&d_y, N*sizeof(float));

    float *alpha_;
    alpha_ = &alpha;

    cublasSetVector(N, sizeof(x[0]), x, 1, d_x, 1);
    cublasSetVector(N, sizeof(y[0]), y, 1, d_y, 1);

    cublasSaxpy(handle, N, alpha_, d_x, 1, d_y, 1);

    cublasGetVector(N, sizeof(d_y[0]), d_y, 1, y, 1);

    cudaFree(d_x); cudaFree(d_y);

    // destroy the hanle
    cublasDestroy(handle);
}