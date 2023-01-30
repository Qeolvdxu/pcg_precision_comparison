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

__global__ void cgkernel(cusparseSpMatDescr_t A,
			 cusparseSpMatDescr_t M,
			PRECI_DT *b,
			PRECI_DT *x,
			int max_iter,
			PRECI_DT tolerance)
{
  printf("Hello cuda!");
  return;
}



__host__ cusparseSpMatDescr_t cusparse_crs_read(char* name)
{
  cusparseSpMatDescr_t desc;
  PRECI_DT* val;
  int* col;
  int* rowptr;

  int n = 0;
  int m = 0;
  int nz = 0;
  FILE *file;
  if ((file = fopen(name, "r"))) {
    int i;

    fscanf(file, "%d %d %d", &m, &n, &nz);

    /*PRECI_DT* val = new PRECI_DT[nz];
      int* col = new int[nz];
    int* rowptr = new int[n];*/

     val = (PRECI_DT*)malloc(sizeof(PRECI_DT)*nz);
     col = (int*)malloc(sizeof(int)*nz);
     rowptr = (int*)malloc(sizeof(int)*n+1);
    

     for (i = 0; i <= n; i++)
       fscanf(file, "%d ", &rowptr[i]);

     for (i = 0; i < nz; i++)
       fscanf(file, "%d ", &col[i]);
       for (i = 0; i < nz; i++)
    fscanf(file, PRECI_S, &val[i]);



       fclose(file);


       // Allocate memory for the CSR matrix
      int *ptr, *indices;
      PRECI_DT* data;
      cudaMalloc((void **)&ptr, (n+1) * sizeof(int));
      cudaMalloc((void **)&indices, nz * sizeof(int));
      cudaMalloc((void **)&data, nz * sizeof(PRECI_DT));

      // Copy data from host to device
      cudaMemcpy(ptr, rowptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(indices, col, nz * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(data, val, nz * sizeof(PRECI_DT), cudaMemcpyHostToDevice);

      //Create HHthe CSR matrix
      cusparseCreateCsr(&desc, n, n, nz, rowptr, col, val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, PRECI_CUDA);

  } else {
    printf("ERROR: could not open file %s\n", name);
    n = -1;
  }
  return desc;
}


extern "C" void call_CuCG(void)
{
  printf("Creating cusparse handle!\n");
  cusparseHandle_t cusparseHandle;
  cusparseStatus_t status = cusparseCreate(&cusparseHandle);
  if (status != CUSPARSE_STATUS_SUCCESS)
    {
      printf("Error creating cusparse Handle!\n"); 
    }
     else
       {
	 printf("reading matrix file...\n");
	 cusparseSpMatDescr_t device_A = cusparse_crs_read((char*)"../test_subjects/rcm/bcsstk10.mtx.rcm.csr");

	 int n=10;
	 int nz = 20;

	 printf("allocating host vectors...\n");
	 PRECI_DT* h_x = new PRECI_DT[n];
	 PRECI_DT* h_b = new PRECI_DT[n];

	 printf("setting host vector values...\n");
	 for(int i=0;i<n-1;i++)
	   {
	     h_x[i] = 1;
	     h_b[i] = 0;
	   }
	     printf("Created Host Vectors!\n");  

		    PRECI_DT* d_x;
      PRECI_DT* d_b;

      cudaMalloc(&d_x, n * sizeof(PRECI_DT));
      cudaMalloc(&d_b, n * sizeof(PRECI_DT));

      cudaMemcpy(d_x, h_x, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);



      printf("Calling cgkernel...\n");
      cgkernel<<<1,(n*8)+nz>>>(device_A, device_A, d_b,d_x,8000,1e-7);
      cusparseDestroy(cusparseHandle);
       }
  printf("Done!\n");
    
    
    return;
}

int main (void)
{
  call_CuCG();
  return 0;
}