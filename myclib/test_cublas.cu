//#include <iostream>
#include<stdio.h>
#include <math.h>
#include <stdlib.h>
//#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define W 512
#define H (512 * 1024)

// input is a host device mem vector
extern "C" {
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
}


__global__ void expand_kernel(
    const float* vector, 
    const unsigned vlen, 
    float* matrix,
    const unsigned mdim
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        float myval = vector[idx%vlen];
        while (idx < mdim*vlen){
            matrix[idx] = myval;
            idx += gridDim.x*blockDim.x;
        }
    }



extern "C" {
void call_reshape(const float *x, float *y, int w, int num_actions){

    float *d_x, *d_y;
    cudaMalloc ((void**)&d_x, w*sizeof(float));
    cudaMalloc ((void**)&d_y, w*num_actions*sizeof(float));
    cudaMemcpy(d_x, x, w * ( sizeof(float)), cudaMemcpyHostToDevice);

    expand_kernel<<<w, 256>>>(d_x, w, d_y, num_actions);

    cudaMemcpy(y, d_y, w * num_actions * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x); cudaFree(d_y);
}
}


