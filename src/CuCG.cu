#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cusparse.h>

//#include "../include/CuCG.cuh"

//#include "../include/my_crs_matrix.h"
#define PRECI_DT double 
#define PRECI_S "%lf "
#define PRECI_CUDA CUDA_R_64F

// compressed sparse row matrix
typedef struct {
  int n;
  int m;
  int nz;
  PRECI_DT *val;
  int *col;
  int *rowptr;
} my_crs_matrix;


__global__ int cgkernel(cusparseSpMatDescr_t A,
			cusparseSpMatDescr_t M,
			PRECI_DT *b,
			PRECI_DT *x,
			int max_iter,
			PRECI_DT tolerance)

{

  return max_iter;
}


//cusparseSpMatDescr_t device_A = my_crs_2_cusparse(host_A, device_A, cusparseHandle);

__host__ cusparseSpMatDescr_t my_crs_2_cusparse(my_crs_matrix* A)
{
  cusparseSpMatDescr_t desc;
  int nnz = A->nz;
  int n = A->n;

  // Allocate memory for the CSR matrix
  int *ptr, *indices;
  PRECI_DT* data;
  cudaMalloc((void **)&ptr, (n+1) * sizeof(int));
  cudaMalloc((void **)&indices, nnz * sizeof(int));
  cudaMalloc((void **)&data, nnz * sizeof(PRECI_DT));

  // Copy data from host to device
  cudaMemcpy(ptr, A->rowptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(indices, A->col, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(data, A->val, nnz * sizeof(PRECI_DT), cudaMemcpyHostToDevice);

  //Create HHthe CSR matrix
  cusparseCreateCsr(&desc, n, n, nnz, A->rowptr, A->col, A->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, PRECI_CUDA);
  return desc;
}

    __host__ void my_crs_free(my_crs_matrix *M) {
    free(M->val);
    free(M->col);
    free(M->rowptr);
    free(M);

    return;
}

__host__ my_crs_matrix *my_crs_read(char *name) {
  printf("test 1\n");

  my_crs_matrix *M = (my_crs_matrix*)malloc(sizeof(my_crs_matrix));
  printf("test 2\n");


  FILE *file;
  if ((file = fopen(name, "r"))) {
    int i;

    fscanf(file, "%d %d %d", &M->m, &M->n, &M->nz);
    M->val = (PRECI_DT*)malloc(sizeof(PRECI_DT) * M->nz);
    M->col = (int*)malloc(sizeof(int) * M->nz);
    M->rowptr = (int*)malloc(sizeof(int) * M->n);

    for (i = 0; i <= M->n; i++)
      fscanf(file, "%d ", &M->rowptr[i]);
    for (i = 0; i < M->nz; i++)
      fscanf(file, "%d ", &M->col[i]);
    for (i = 0; i < M->nz; i++)
      fscanf(file, PRECI_S, &M->val[i]);

    fclose(file);
  } else {
    printf("ERROR: could not open file %s\n", name);
    M->n = -1;
  }
  printf("test 3\n");


  return M;
}

extern "C" void call_CuCG(void)
{
  cusparseHandle_t cusparseHandle;
  cusparseStatus_t status = cusparseCreate(&cusparseHandle);
  if (status != CUSPARSE_STATUS_SUCCESS)
    {
      printf("Error creating cusparse Handle!"); 
    }
  else
    {
      my_crs_matrix *host_A;
      host_A = my_crs_read((char*)"../test_subjects/rcm/bcsstk10.mtx.rcm.csr");

      PRECI_DT* h_x = new PRECI_DT[host_A->n];
      PRECI_DT* h_b = new PRECI_DT[host_A->n];

      for(int i=0;i<host_A->n;i++)
	{
	  h_x[i] = 1;
	  h_b[i] = 0;
	}
  

      PRECI_DT* d_x;
      PRECI_DT* d_b;

      cudaMalloc(&d_x, host_A->n * sizeof(PRECI_DT));
      cudaMalloc(&d_b, host_A->n * sizeof(PRECI_DT));

      cudaMemcpy(d_x, h_x, host_A->n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, host_A->n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);

      cusparseSpMatDescr_t device_A = my_crs_2_cusparse(host_A);
      //cgkernel<<<1,(host_A->n*8)+host_A->nz>>>(device_A, device_A, d_b,d_x,8000,1e-7);


      my_crs_free(host_A);
    }
  cusparseDestroy(cusparseHandle);
  printf("Done!\n");
    
    
  return;
}

int main (void)
{
  call_CuCG();
  return 0;
}