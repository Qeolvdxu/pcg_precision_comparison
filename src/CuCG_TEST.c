#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "../include/my_crs_matrix.h"
#include "../include/CuCG.cuh"

int main(void)
{
  cusparseHandle_t cusparseHandle;
  cusparseCreate(&cusparseHandle);
    
  my_crs_matrix* A = my_crs_read("../test_subjects/");
	
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
