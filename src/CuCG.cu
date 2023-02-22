#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "../include/CuCG.h"

//#include "../include/my_crs_matrix.h"
#define PRECI_DT double 
#define PRECI_S "%lf "
#define PRECI_CUDA CUDA_R_64F

typedef struct {
  cusparseDnVecDescr_t desc;
  PRECI_DT*            val;
} my_cuda_vector;

typedef struct {
  cusparseSpMatDescr_t desc;
  int n;
  int m;
  int nz;
  PRECI_DT *val;
  int *col;
  int *rowptr;
} my_cuda_csr_matrix;

__host__ void cusparse_conjugate_gradient(my_cuda_csr_matrix *A,
					  my_cuda_csr_matrix *M,
					  my_cuda_vector *b,
					  my_cuda_vector *x,
					  int max_iter,
			 PRECI_DT tolerance,
					  cusparseHandle_t* handle,
					  cublasHandle_t* handle_blas)

{
  int n = A->n;
  // Make r vector
  my_cuda_vector *r_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  PRECI_DT* h_r = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  for(int i=0;i<n;i++) h_r[i] = 1;
  cudaMalloc((void**)&r_vec->val, n * sizeof(PRECI_DT));
  cudaMemcpy(r_vec->val, h_r, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cusparseCreateDnVec(&r_vec->desc, n, r_vec->val,PRECI_CUDA);

  // Make p vector
  my_cuda_vector *p_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  PRECI_DT* h_p = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  for(int i=0;i<n;i++) h_p[i] = 1;
  cudaMalloc((void**)&p_vec->val, n * sizeof(PRECI_DT));
  cudaMemcpy(p_vec->val, h_p, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cusparseCreateDnVec(&p_vec->desc, n, p_vec->val,PRECI_CUDA);

  // Make q vector
  my_cuda_vector *q_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  PRECI_DT* h_q = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  for(int i=0;i<n;i++) h_q[i] = 1;
  cudaMalloc((void**)&q_vec->val, n * sizeof(PRECI_DT));
  cudaMemcpy(q_vec->val, h_q, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cusparseCreateDnVec(&q_vec->desc, n, q_vec->val,PRECI_CUDA);

  // Make z vector
  my_cuda_vector *z_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  PRECI_DT* h_z = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  for(int i=0;i<n;i++) h_z[i] = 1;
  cudaMalloc((void**)&z_vec->val, n * sizeof(PRECI_DT));
  cudaMemcpy(z_vec->val, h_z, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cusparseCreateDnVec(&z_vec->desc, n, z_vec->val,PRECI_CUDA);

  PRECI_DT alpha = 0.0;
  PRECI_DT beta = 0.0;
  const double ne_one = -1.0;
  const double n_one = 1.0;
  const double one = 0.0;

  int iter = 0;
  int j = 0;

  PRECI_DT v = 0;
  PRECI_DT Rho = 0;
  PRECI_DT Rtmp = 0;

  PRECI_DT res_norm = 0;
  PRECI_DT init_norm = 0;
  PRECI_DT ratio = 0;

  
  double Tiny = 0.1e-28;

  // x is already zero
  
  size_t bufferSizeMV;
  void* buff;
  cusparseSpMV_bufferSize(*handle,CUSPARSE_OPERATION_NON_TRANSPOSE, &n_one, A->desc, b->desc, &one, x->desc, PRECI_CUDA, CUSPARSE_MV_ALG_DEFAULT, &bufferSizeMV);
  cudaMalloc(&buff, bufferSizeMV);

  //matvec(A,x,r);
  cusparseSpMV(*handle,
	       CUSPARSE_OPERATION_NON_TRANSPOSE,//operation
	       &n_one,//alpha
	       A->desc,//matrix
	       x->desc,//vector
	       &one,//beta
	       r_vec->desc,//answer
	       PRECI_CUDA,//data type
	       CUSPARSE_MV_ALG_DEFAULT,//algorithm
	       buff//buffer
	       );

  // r = b - r
  cublasDaxpy(*handle_blas, n, &ne_one, r_vec->val, 1, b->val, 1);
  cublasDcopy(*handle_blas,n,b->val, 1, r_vec->val, 1);

  // z = r
  cublasDcopy(*handle_blas,n,r_vec->val, 1, z_vec->val, 1);

  // p = z
  cublasDcopy(*handle_blas,n,z_vec->val, 1, p_vec->val, 1);

  cublasDnrm2(*handle_blas, n, r_vec->val, 1, &res_norm);
  init_norm = res_norm;
  ratio = 1.0;

  while (iter <= max_iter && ratio > tolerance)
  {
    printf("Iteration : %d",iter);
    iter++;

  // z = r
  cublasDcopy(*handle_blas,n,r_vec->val, 1, z_vec->val, 1);

  // Rho = r z dot prod
  cublasDdot(*handle_blas, n, r_vec->val, 1, z_vec->val, 1, &Rho);

  if (iter == 1)
      cublasDcopy(*handle_blas,n,z_vec->val, 1, p_vec->val, 1);
  else
  {
    beta = Rho / (v + Tiny);
    // p = z + (beta * p)
    cublasDaxpy(*handle_blas, n, &beta, p_vec->val, 1, z_vec->val, 1);
  }

  cusparseSpMV(*handle,
	       CUSPARSE_OPERATION_NON_TRANSPOSE,//operation
	       &n_one,//alpha
	       A->desc,//matrix
	       p_vec->desc,//vector
	       &one,//beta
	       q_vec->desc,//answer
	       PRECI_CUDA,//data type
	       CUSPARSE_MV_ALG_DEFAULT,//algorithm
	       buff//buffer
	       );
  
  // Rtmp = p q dot prod
  cublasDdot(*handle_blas, n, p_vec->val, 1, q_vec->val, 1, &Rtmp);

  // v = r z dot prod
  cublasDdot(*handle_blas, n, r_vec->val, 1, z_vec->val, 1, &v);

  //alpha
  alpha = Rho / (Rtmp + Tiny);
  
  // x = x + alpha * p
  cublasDaxpy(*handle_blas, n, &alpha, p_vec->val, 1, x->val, 1);

  // r = r - alpha * q
  alpha *= -1;
  cublasDaxpy(*handle_blas, n, &alpha,q_vec->val,1,r_vec->val,1);
  alpha *= -1;


  Rho = 0.0;
  cublasDnrm2(*handle_blas, n, r_vec->val, 1, &res_norm);
  ratio = res_norm/init_norm;

  if (iter > 0) {
    cusparseSpMV(*handle,
	       CUSPARSE_OPERATION_NON_TRANSPOSE,//operation
	       &n_one,//alpha
	       A->desc,//matrix
	       x->desc,//vector
	       &one,//beta
	       r_vec->desc,//answer
	       PRECI_CUDA,//data type
	       CUSPARSE_MV_ALG_DEFAULT,//algorithm
	       buff//buffer
	       );
    cublasDaxpy(*handle_blas, n, &ne_one, r_vec->val, 1, b->val, 1);
  }

    cudaDeviceSynchronize();
    int error = cudaGetLastError();
    printf("%s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));

  }

  return;
}



__host__ my_cuda_csr_matrix* cusparse_crs_read(char* name)
{
  my_cuda_csr_matrix *M = (my_cuda_csr_matrix*)malloc(sizeof(my_cuda_csr_matrix));
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

    for (i = 0; i < nz; i++) fscanf(file, "%d ", &col[i]);
    for (i = 0; i < nz; i++)
      fscanf(file, PRECI_S, &val[i]);

    printf("rowptr : ");


    for( i = 0; i < 5; i++)
      printf("%d, ",rowptr[i]);
    printf("\n");


    printf("col : ");

    for( i = 0; i < 5; i++)
      printf("%d, ",col[i]);
    printf("\n");

    printf("val : ");

    for( i = 0; i < 5; i++)
      printf("%lf, ",val[i]);
    printf("\n");




    fclose(file);


    // Allocate memory for the CSR matrix
    cudaMalloc((void**)&M->rowptr, (n+1) * sizeof(int));
    cudaMalloc((void**)&M->col, nz * sizeof(int));
    cudaMalloc((void**)&M->val, nz * sizeof(PRECI_DT));


    // Copy data from host to device
    cudaMemcpy(M->rowptr, rowptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(M->col, col, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(M->val, val, nz * sizeof(PRECI_DT), cudaMemcpyHostToDevice);


    M->n = n;
    M->m = m;
    M->nz = nz;
    cusparseCreateCsr(&M->desc, n, n, nz, M->rowptr, M->col, M->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, PRECI_CUDA);
    //Create the CSR matrix
   
  } else {
    printf("ERROR: could not open file %s\n", name);
    n = -1;
  }
  return M;
}


void call_CuCG(void)
{
  printf("Creating cusparse handle!\n");
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  cusparseHandle_t cusparseHandle;
  cusparseStatus_t status = cusparseCreate(&cusparseHandle);
  if (status != CUSPARSE_STATUS_SUCCESS)
		 {
		   printf("Error creating cusparse Handle!\n"); 
		 }
  else
    {
      printf("reading matrix file...\n");
      my_cuda_csr_matrix *A_matrix = cusparse_crs_read((char*)"../test_subjects/rcm/bcsstk10.mtx.rcm.csr");


      int64_t n=A_matrix->n;

      printf("creating %d vectors... ",A_matrix->n);

      // Make x vector
      my_cuda_vector *x_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
      PRECI_DT* h_x = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
      for(int i=0;i<n;i++) h_x[i] = 0;
      cudaMalloc((void**)&x_vec->val, n * sizeof(PRECI_DT));
      cudaMemcpy(x_vec->val, h_x, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
      cusparseCreateDnVec(&x_vec->desc, n, x_vec->val,PRECI_CUDA);

      // Make b vector
      my_cuda_vector *b_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
      PRECI_DT* h_b = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
      for(int i=0;i<n;i++) h_b[i] = 1;
      cudaMalloc((void**)&b_vec->val, n * sizeof(PRECI_DT));
      cudaMemcpy(b_vec->val, h_b, n * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
      cusparseCreateDnVec(&b_vec->desc, n, b_vec->val,PRECI_CUDA);



      printf("Created Vectors!\n");  
      

      for (int i = 0; i < 10; i++)
	printf(PRECI_S,h_x[i]);
      printf("\n");


      for (int i = 0; i < 10; i++)
	printf(PRECI_S,h_b[i]);
      printf("\n");



      printf("Calling CG func...");
      cusparse_conjugate_gradient(A_matrix, A_matrix, b_vec,x_vec,8000,1e-7, &cusparseHandle, &cublasHandle);
      printf("Done!\n");

      cudaMemcpy(h_x, x_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_b, b_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);

      for (int i = 0; i < 10; i++)
	 printf(PRECI_S,h_x[i]);
       printf("\n");



       for (int i = 0; i < 10; i++)
	 printf(PRECI_S,h_b[i]);
      printf("\n");



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
