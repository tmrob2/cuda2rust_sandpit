#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
//#include <cuda.h>
#include <cusparse_v2.h>

// What we are interested in constructing a CSR matrix
// Assume that we already have a sparse matrix on host memory using 
// CXSparse and we just want to convert it to device memory 
extern "C" {
void csr_spmv(
    int *csr_row, 
    int *csr_col, 
    float *csr_vals, 
    float *x,
    float *y,
    int nnz, 
    int sizeof_row, 
    int m, 
    int n
    ) {
    // sizeof_row is the size of csr_row
    // sizeof_col is the size of csr_col
    // m number of rows in the matrix
    // n number of cols in the matrix
    // nnz is the size of the csr_vals
    // create a sparse handle
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t descrC = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dCsrRowPtr, *dCsrColPtr;
    float *dCsrValPtr;

    // allocate device memory to store the sparse CSR
    cudaMalloc((void **)&dCsrValPtr, sizeof(float) * nnz);
    cudaMalloc((void **)&dCsrColPtr, sizeof(int) * nnz);
    cudaMalloc((void **)&dCsrRowPtr, sizeof(int) * sizeof_row);

    // Free the device memory allocated to the coo ptrs once they
    // the conversion from coo to csr has been completed
    cudaMemcpy(dCsrValPtr, csr_vals, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrColPtr, csr_col, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrRowPtr, csr_row, sizeof(int) * sizeof_row, cudaMemcpyHostToDevice);

    // create the sparse CSR matrix in device memory
    status = cusparseCreateCsr(
        &descrC, // MATRIX DESCRIPTION
        m, // NUMBER OF ROWS
        n, // NUMBER OF COLS
        nnz, // NUMBER OF NON ZERO VALUES
        dCsrRowPtr, // ROWS OFFSETS
        dCsrColPtr, // COL INDICES
        dCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    );

    float alpha = 1.0;
    float beta = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    float *dX, *dY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cudaMalloc((void**)&dX, m * sizeof(float));
    cudaMalloc((void**)&dY, n * sizeof(float));

    // copy the vector from host memory to device memory
    cudaMemcpy(dX, x, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    cusparseCreateDnVec(&vecX, m, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n, dY, CUDA_R_32F);

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrC, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alpha, descrC, vecX, &beta, vecY, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    // Any algorithms get inserted here

    cudaMemcpy(y, dY, n *sizeof(float), cudaMemcpyDeviceToHost);

    //destroy the vector descriptors
    cusparseDestroySpMat(descrC);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    // Free the device memory
    cudaFree(dCsrColPtr);
    cudaFree(dCsrRowPtr);
    cudaFree(dCsrValPtr);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dBuffer);

}

}