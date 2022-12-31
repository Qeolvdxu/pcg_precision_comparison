#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "my_crs_matrix.h"

#include <cuda_runtime.h>
#include <cusparse.h>

__global__ void cgkernel()
{
	
}

__host__ my_crs_2_cusparse(my_crs_matrix* A, cusparseHandle_t cusparseHandle);
{
	int nnz = A->nz;
	int n = A->n;

	// Allocate memory for the CSR matrix
	int *ptr, *indices, *data;
	cudaMalloc((void **)&ptr, (n+1) * sizeof(int));
	cudaMalloc((void **)&indices, nnz * sizeof(int));
	cudaMalloc((void **)&data, nnz * sizeof(PRECI_DT));

	// Copy data from host to device
	cudaMemcpy(ptr, A->rowptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(indices, A->col, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data, A->val, nnz * sizeof(PRECI_DT), cudaMemcpyHostToDevice);

	// Create the CSR matrix
	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	cusparseHybMat_t csrMat;
	cusparseCreateHybMat(&csrMat);
	cusparseDcsr2hyb(cusparseHandle, n, n, descr, data, ptr, indices, csrMat, 0, CUSPARSE_HYB_PARTITION_AUTO);
}

int main(void)
{
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    
    my_crs_matrix* A = my_crs_read("test_subjects/");
	
    my_crs_2_cusparse(A, cusparseHandle);
    // Run, Time and copy data from CG
    clock_t t;
    t = clock();
    cgkernel<<<1,1>>>();

    cudaDeviceSynchronize();
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("cg took %f seconds\n", time_taken);
    cudaMemcpy(h_x, d_x, sizeof(PRECI_DT)*size, cudaMemcpyDeviceToHost);

    cusparseDestroy(cusparseHandle);
    return 0;
  }
