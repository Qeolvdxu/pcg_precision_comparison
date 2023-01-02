#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "../include/my_crs_matrix.h"
#include "../include/CuCG.cuh"

__global__ void cgkernel()
{
  return;
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

