#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include<cuda.h>
#include <cusparse_v2.h>

// What we are interested in constructing a CSR matrix
// Assume that we already have a sparse matrix on host memory using 
// CXSparse and we just want to convert it to device memory 
void create_sparse_from_csr(
    int *csr_row, 
    int *csr_col, 
    float *csr_vals, 
    int nnz, 
    int sizeof_row, 
    int sizeof_col,
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
    cusparseSpMatDescr_t descrC = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

    status = cusparseCreate(&handle);

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dCsrRowPtr, *dCsrColPtr;
    float *dCsrValPtr;

    // allocate device memory to store the sparse CSR
    status = cudaMalloc((void **)&dCsrValPtr, sizeof(float) * nnz);
    status = cudaMalloc((void **)&dCsrRowPtr, sizeof(int) * sizeof_row);
    status = cudaMalloc((void **)&dCsrColPtr, sizeof(int) * sizeof_col);

    // Free the device memory allocated to the coo ptrs once they
    // the conversion from coo to csr has been completed
    cudaMemcpy(dCsrRowPtr, csr_row, sizeof(int) * sizeof_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrColPtr, csr_col, sizeof(int) * sizeof_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrValPtr, csr_vals, sizeof(float) * nnz, cudaMemcpyHostToDevice);

    // create the sparse CSR matrix in device memory
    status = cusparseCreateCsr(
        &descrC, // MATRIX DESCRIPTION
        m, // NUMBER OF ROWS
        n, // NUMBER OF COLS
        nnz, // NUMBER OF NON ZERO VALUES
        dCsrColPtr, // COL INDICES
        dCsrRowPtr, // ROWS OFFSETS
        dCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    );
    // Free the device memory
    cudaFree(dCsrColPtr);
    cudaFree(dCsrRowPtr);
    cudaFree(dCsrValPtr);

    status = cusparseDestroy(handle);
    cusparseDestroySpMat(descrC);
}

