#include <stdlib.h>
#include <omp.h>
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
                      my_cuda_vector *r_vec,
                      my_cuda_vector *p_vec,
                      my_cuda_vector *q_vec,
                      my_cuda_vector *z_vec,
					  int max_iter,
					 PRECI_DT tolerance,
                      int* iter,
                      double* elapsed,
					  cusparseHandle_t* handle,
					  cublasHandle_t* handle_blas)

{
  #ifdef ENABLE_TESTS
    printf("start cg!");
  #endif
  int n = A->n;

  #ifdef ENABLE_TESTS
  PRECI_DT* onex = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  PRECI_DT* onez = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  PRECI_DT* oner = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  PRECI_DT* oneq = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
  PRECI_DT* onep = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
	int* rowptr;
	int* col;
	PRECI_DT* val;
    val = (PRECI_DT*)malloc(sizeof(PRECI_DT)*A->nz);
    col = (int*)malloc(sizeof(int)*A->nz);
    rowptr = (int*)malloc(sizeof(int)*n+1);

   /* cudaMemcpy(rowptr, M->rowptr, (A->n+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col, M->col, A->nz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(val, M->val, A->nz * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
    printf("CUDA rowptr : ");
    for( int i = 0; i <= A->n; i++)
      printf("%d ",rowptr[i]);
    printf("\n");

    printf("CUDA col : ");
    for( int i = 0; i < A->nz; i++)
      printf("%d ",col[i]);
    printf("\n");

    printf("CUDA val : ");
    for( int i = 0; i < A->nz; i++)
      printf("%lf ",val[i]);
    printf("\n");
    */
  #endif
  size_t pitch;




  cublasStatus_t sb;
  
  PRECI_DT alpha = 1.0;
  PRECI_DT beta = 0.0;
  const double ne_one = -1.0;
  const double n_one = 1.0;
  const double one = 0.0;

  int itert = 0;

  PRECI_DT v = 0;
  PRECI_DT Rho = 0;
  PRECI_DT Rtmp = 0;

  PRECI_DT res_norm = 0;
  PRECI_DT init_norm = 0;
  PRECI_DT ratio = 0;

  
  double Tiny = 0.1e-28;
  double minus_alpha = 0.0;

  // x is already zero
  
  size_t bufferSizeMV;
  void* buff;
  cusparseSpMV_bufferSize(*handle,CUSPARSE_OPERATION_NON_TRANSPOSE, &n_one, A->desc,
                          b->desc, &one, x->desc, PRECI_CUDA, CUSPARSE_MV_ALG_DEFAULT,
                          &bufferSizeMV);
  cudaMalloc(&buff, bufferSizeMV);


  /*cudaMemcpy(onex, x->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(onep, p_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(oneq, q_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(oner, r_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(onez, z_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  printf("\INITIAL VEC CREATION\n x1 = %lf \t alpha= %lf \t beta= %lf "
	 "\n v "
	 "= %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm "
	 "ratio(%lf) > tolerance(%lf)\n\n\n",
	 iter, onex[0], alpha, beta, v, oner[0], onep[0], oneq[0], onez[0], ratio,
	 tolerance);*/


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
  //cudaDeviceSynchronize();

  // r = b - r
  cublasDaxpy(*handle_blas, n, &ne_one, r_vec->val, 1, b->val, 1);
  //cudaDeviceSynchronize();
  cublasDcopy(*handle_blas,n,b->val, 1, r_vec->val, 1);
  //cudaDeviceSynchronize();

  // z = r
  if (M)
      //z = MT\(M\r);
      M=A;
  else
      // z = r
      cublasDcopy(*handle_blas,n,r_vec->val, 1, z_vec->val, 1);
  //cudaDeviceSynchronize();

  // p = z
  cublasDcopy(*handle_blas,n,z_vec->val, 1, p_vec->val, 1);
  //cudaDeviceSynchronize();
  cublasDnrm2(*handle_blas, n, r_vec->val, 1, &res_norm);
  //cudaDeviceSynchronize();
  init_norm = res_norm;
  ratio = 1.0;

  #ifdef ENABLE_TESTS
  cudaMemcpy(onex, x->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(onep, p_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(oneq, q_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(oner, r_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);

  cudaMemcpy(onez, z_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
  /*printf("PREQUEL \n x1 = %lf \t alpha= %lf \t beta= %lf "
	 "\n v "
	 "= %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm "
	 "ratio(%lf) > tolerance(%e)\n\n\n",
	 iter, onex[0], alpha, beta, v, oner[0], onep[0], oneq[0], onez[0], ratio,
	 tolerance);*/
  #endif
/* //CUDA_EVENT_TIME
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEventRecord(start, stream);
  */

  // WALL TIME
  double start;
  double end;
  start = omp_get_wtime();

  while (itert <= max_iter && ratio > tolerance)
    {
  #ifdef ENABLE_TESTS
      printf("\nITERATION %d\n",itert);
  #endif
      itert++;

      if (M)
          //z = MT\(M\r);
          M=A;
      else
          // z = r
          cublasDcopy(*handle_blas,n,r_vec->val, 1, z_vec->val, 1);
      //cudaDeviceSynchronize();
  #ifdef ENABLE_TESTS
      cudaMemcpy(onez, z_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("z[1] = %lf\n",onez[1]);
  #endif

      // Rho = r z dot prod
      cublasDdot(*handle_blas, n, r_vec->val, 1, z_vec->val, 1, &Rho);
      //cudaDeviceSynchronize();
  #ifdef ENABLE_TESTS
      printf("Rho = %lf\n",Rho);
  #endif

      // p = z + (beta * p)
      // p = (beta * z) + p
      if (itert == 1)
	    {
	      cublasDcopy(*handle_blas,n,z_vec->val, 1, p_vec->val, 1);
//	      cudaDeviceSynchronize();
	    }
      else
	    {
	      beta = Rho / (v + Tiny);
	      cublasDscal(*handle_blas, n, &beta, p_vec->val, 1);
	      cublasDaxpy(*handle_blas, n, &n_one, z_vec->val, 1, p_vec->val, 1);
 	//      cudaDeviceSynchronize();
	    }
  #ifdef ENABLE_TESTS
	    printf("beta = %lf\n",beta);
	    cudaMemcpy(onep, p_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
	    printf("p[1] = %lf\n",onep[1]);
  #endif
	

      cusparseSpMV_bufferSize(*handle,CUSPARSE_OPERATION_NON_TRANSPOSE, &n_one, A->desc, p_vec->desc, &one, q_vec->desc, PRECI_CUDA, CUSPARSE_MV_ALG_DEFAULT, &bufferSizeMV);
      cudaMalloc(&buff, bufferSizeMV);

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
//      cudaDeviceSynchronize();

#ifdef ENABLE_TESTS
      cudaMemcpy(oneq, q_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("q[1] = %lf\n",oneq[1]);
#endif
      
      // Rtmp = p q dot prod
      cublasDdot(*handle_blas, n, p_vec->val, 1, q_vec->val, 1, &Rtmp);
//      cudaDeviceSynchronize();
#ifdef ENABLE_TESTS
      printf("Rtmp = %lf\n",Rtmp);
#endif

      // v = r z dot prod
      cublasDdot(*handle_blas, n, r_vec->val, 1, z_vec->val, 1, &v);
//      cudaDeviceSynchronize();
#ifdef ENABLE_TESTS
      cudaMemcpy(onep, p_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("v = %lf\n",v);
#endif

      //alpha
      alpha = Rho / (Rtmp + Tiny);
#ifdef ENABLE_TESTS
      printf("alpha = %lf\n",alpha);
#endif
      
      // x = x + alpha * p
      cublasDaxpy(*handle_blas, n, &alpha, p_vec->val, 1, x->val, 1);
//      cudaDeviceSynchronize();

#ifdef ENABLE_TESTS
      cudaMemcpy(onex, x->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("x[1] = %lf\n",onex[1]);
#endif

      // r = r - alpha * q
      minus_alpha = -alpha;
      cublasDaxpy(*handle_blas, n, &minus_alpha,q_vec->val,1,r_vec->val,1);
//      cudaDeviceSynchronize();
#ifdef ENABLE_TESTS
      cudaMemcpy(oner, r_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("r[1] = %lf\n", oner[1]);
#endif

      Rho = 0.0;
      cublasDnrm2(*handle_blas, n, r_vec->val, 1, &res_norm);
//      cudaDeviceSynchronize();
#ifdef ENABLE_TESTS
      printf("res_norm = %lf\n", res_norm);
#endif

      ratio = res_norm/init_norm;
#ifdef ENABLE_TESTS
      printf("ratio = %lf\n", ratio);
#endif

      if (iter > 0) {
        // A*x=r
        cusparseSpMV_bufferSize(*handle,CUSPARSE_OPERATION_NON_TRANSPOSE, &n_one, A->desc, x->desc, &one, r_vec->desc, PRECI_CUDA, CUSPARSE_MV_ALG_DEFAULT, &bufferSizeMV);
        cudaMalloc(&buff, bufferSizeMV);
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
//        cudaDeviceSynchronize();
#ifdef ENABLE_TESTS
      cudaMemcpy(oner, r_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("r[1] = %lf\n", oner[1]);
#endif
      //r = b - r
        cublasDaxpy(*handle_blas, n, &ne_one, b->val, 1, r_vec->val, 1);
  //      cudaDeviceSynchronize();
        cublasDscal(*handle_blas, n, &ne_one, r_vec->val, 1);
    //    cudaDeviceSynchronize();
#ifdef ENABLE_TESTS
      cudaMemcpy(oner, r_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("r[1] = %lf\n", oner[1]);
#endif
      }

//      cudaDeviceSynchronize();
#ifdef ENABLE_TESTS
      int error = cudaGetLastError();
      printf("%s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
#endif
    /*
      cudaMemcpy(onex, x->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      cudaMemcpy(onep, p_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      cudaMemcpy(oneq, q_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      cudaMemcpy(oner, r_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      cudaMemcpy(onez, z_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      printf("\nend of iteration %d\n x1 = %lf \t alpha= %lf \t beta= %lf \t res_norm = %lf"
            "\n v "
            "= %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm "
            "ratio(%lf) > tolerance(%lf)\n\n\n",
            iter, onex[0], alpha, beta, res_norm, v, oner[0], onep[0], oneq[0], onez[0], ratio,
            tolerance);*/

      //printf("\e[1;1H\e[2J");
      }
  cudaDeviceSynchronize();

  // WALL TIME
  end = omp_get_wtime();
/* //CUDA EVENT TIME
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
*/
#ifdef ENABLE_TESTS
    printf("TOtal of %d CuCG ITerations\n",itert);
#endif

    *elapsed = (end - start)*1000;
    *iter = itert;
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
    for (i = 0; i < nz; i++)
      fscanf(file, "%d ", &col[i]);
    for (i = 0; i < nz; i++)
      fscanf(file, PRECI_S, &val[i]);

    #ifdef ENABLE_TESTS
       printf("READ rowptr : ");
       for( i = 0; i <= n; i++)
        printf("%d ",rowptr[i]);
       printf("\n");

       printf("READ col : ");
       for( i = 0; i < nz; i++)
         printf("%d ",col[i]);
       printf("\n");

       printf("READ val : ");
       for( i = 0; i < nz; i++)
         printf("%lf ",val[i]);
       printf("\n");
    #endif

    fclose(file);
    size_t pitch;
    // Allocate memory for the CSR matrix
    //M->rowptr = cudaMalloc(sizeof(int)*(n+1));
    //M->col = cudaMalloc(sizeof(int)*(nz));
    //M->val = cudaMalloc(sizeof(int)*(PRECI_DT));
    cudaMallocPitch((void**)&M->rowptr,&pitch, (n+1) * sizeof(int),1);
    cudaMallocPitch((void**)&M->col,&pitch, nz * sizeof(int),1);
    cudaMallocPitch((void**)&M->val,&pitch, nz * sizeof(PRECI_DT),1);


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


void call_CuCG(char* name, char* m_name, PRECI_DT* h_b, PRECI_DT* h_x, int maxit,
               PRECI_DT tol, int* iter, double* elapsed, double* mem_elapsed)
{
  //printf("Creating cusparse handle!\n");
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  cusparseHandle_t cusparseHandle;
  cusparseStatus_t status = cusparseCreate(&cusparseHandle);

  int n = 0;
  int m = 0;
  int nz = 0;

  FILE *file;
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    printf("Error creating cusparse Handle!\n"); 
  }
  else if ((file = fopen(name, "r")))
  {
      size_t pitch;


      int i;
      fscanf(file, "%d %d %d", &m, &n, &nz);
      int64_t n_t=n;

  PRECI_DT* val = (PRECI_DT*)malloc(sizeof(PRECI_DT)*nz);
  int* col = (int*)malloc(sizeof(int)*nz);
  int* rowptr = (int*)malloc(sizeof(int)*n+1);
  PRECI_DT* h_rpqz = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);


    for (i = 0; i <= n; i++)
      fscanf(file, "%d ", &rowptr[i]);
    for (i = 0; i < nz; i++)
      fscanf(file, "%d ", &col[i]);
    for (i = 0; i < nz; i++)
      fscanf(file, PRECI_S, &val[i]);

    #ifdef ENABLE_TESTS
      printf("READ rowptr : ");
       for( i = 0; i <= n; i++)
        printf("%d ",rowptr[i]);
       printf("\n");

       printf("READ col : ");
       for( i = 0; i < nz; i++)
         printf("%d ",col[i]);
       printf("\n");

       printf("READ val : ");
       for( i = 0; i < nz; i++)
         printf("%lf ",val[i]);
       printf("end\n");
    #endif
    //fclose(file);

  // Copy data from host to device


  /*EVENT TIME
   cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEventRecord(start, stream);*/

  // WALL TIME
  double start;
  double end;
  start = omp_get_wtime();

  my_cuda_csr_matrix *A_matrix = (my_cuda_csr_matrix*)malloc(sizeof(my_cuda_csr_matrix));
  my_cuda_vector *b_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *x_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *r_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *p_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *q_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *z_vec = (my_cuda_vector*)malloc(sizeof(my_cuda_vector));
  my_cuda_csr_matrix *M_matrix;
  A_matrix->n = n;
  A_matrix->m = m;
  A_matrix->nz = nz;

  cudaMallocPitch((void**)&A_matrix->rowptr,&pitch, ( n +1) * sizeof(int),1);
  cudaMallocPitch((void**)&A_matrix->col,&pitch, nz * sizeof(int),1);
  cudaMallocPitch((void**)&A_matrix->val,&pitch, nz * sizeof(PRECI_DT),1);
  cudaMallocPitch((void**)&x_vec->val,&pitch, n_t * sizeof(PRECI_DT),1);
  cudaMallocPitch((void**)&b_vec->val, &pitch, n_t * sizeof(PRECI_DT),1);
  cudaMallocPitch((void**)&r_vec->val,&pitch, n_t * sizeof(PRECI_DT), 1);
  cudaMallocPitch((void**)&p_vec->val,&pitch, n_t * sizeof(PRECI_DT),1);
  cudaMallocPitch((void**)&q_vec->val,&pitch, n_t * sizeof(PRECI_DT),1);
  cudaMallocPitch((void**)&z_vec->val,&pitch, n_t * sizeof(PRECI_DT),1);

  cudaMemcpy(A_matrix->rowptr, rowptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(A_matrix->col, col, nz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(A_matrix->val, val, nz * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cudaMemcpy(x_vec->val, h_x, n_t * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cudaMemcpy(b_vec->val, h_b, n_t * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cudaMemcpy(r_vec->val, h_rpqz, n_t * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cudaMemcpy(p_vec->val, h_rpqz, n_t * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cudaMemcpy(q_vec->val, h_rpqz, n_t * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cudaMemcpy(z_vec->val, h_rpqz, n_t * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();


  cusparseCreateDnVec(&b_vec->desc, n_t, b_vec->val,PRECI_CUDA);
  cusparseCreateDnVec(&x_vec->desc, n_t, x_vec->val,PRECI_CUDA);
  cusparseCreateDnVec(&r_vec->desc, n_t, r_vec->val,PRECI_CUDA);
  cusparseCreateDnVec(&p_vec->desc, n_t, p_vec->val,PRECI_CUDA);
  cusparseCreateDnVec(&q_vec->desc, n_t, q_vec->val,PRECI_CUDA);
  cusparseCreateDnVec(&z_vec->desc, n_t, z_vec->val,PRECI_CUDA);
  cusparseCreateCsr(&A_matrix->desc, n, n, nz, A_matrix->rowptr, A_matrix->col, A_matrix->val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, PRECI_CUDA);
  cudaDeviceSynchronize();
  /* //EVENT TIME
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  *mem_elapsed = milliseconds;*/
   //WALL
  end = omp_get_wtime();
    *mem_elapsed = (end - start)*1000;




    //Create the CSR matrix


   if (m_name)
       M_matrix = cusparse_crs_read((char*)m_name);


      //printf("creating vectors... %d",A_matrix->n);
      //PRECI_DT* h_x = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);

      //PRECI_DT* h_b = (PRECI_DT*)malloc(sizeof(PRECI_DT)*n);
      //for(int i=0;i<n;i++) h_b[i] = 1;

  // vectors
/* //WALL
    gettimeofday(&end, 0);
    double seconds = end.tv_sec - begin.tv_sec;
    double microseconds = end.tv_usec - begin.tv_usec;
    *elapsed = seconds + microseconds * 1e-6 * 1000;
   */

#ifdef ENABLE_TESTS
      printf("Created Vectors!\n");
     /* for (int i = 0; i < n_t; i++)
	printf(PRECI_S,h_x[i]);
      printf("\n");

      for (int i = 0; i < n_t; i++)
	printf(PRECI_S,h_b[i]);
      printf("\n");*/
      printf("Calling CG func...\n");
#endif

      cusparse_conjugate_gradient(A_matrix, NULL, b_vec,x_vec,r_vec,p_vec,q_vec,z_vec,
                                  maxit,tol, iter, elapsed, &cusparseHandle, &cublasHandle);

#ifdef ENABLE_TESTS
      printf("Done!\n");
#endif

      cudaMemcpy(h_x, x_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_b, b_vec->val, n * sizeof(PRECI_DT), cudaMemcpyDeviceToHost);

#ifdef ENABLE_TESTS
     /* for (int i = 0; i < 10; i++)
	 printf(PRECI_S,h_x[i]);
       printf("\n");

       for (int i = 0; i < 10; i++)
	 printf(PRECI_S,h_b[i]);*/
      printf("\n");
#endif

      cusparseDestroySpMat(A_matrix->desc);
      cudaFree(A_matrix->val);
      cudaFree(A_matrix->rowptr);
      cudaFree(A_matrix->col);
      free(A_matrix);

      if (m_name)
      {
          cusparseDestroySpMat(M_matrix->desc);
          cudaFree(M_matrix->val);
          cudaFree(M_matrix->rowptr);
          cudaFree(M_matrix->col);
          free(M_matrix);
      }
      cudaFree(x_vec->val);
      cusparseDestroyDnVec(x_vec->desc);
      free(x_vec);

      free(h_rpqz);

      cudaFree(b_vec->val);
      cusparseDestroyDnVec(b_vec->desc);
      free(b_vec);

  // free everything
    cudaFree(p_vec->val);
    cusparseDestroyDnVec(p_vec->desc);
    free(p_vec);

    cudaFree(z_vec->val);
    cusparseDestroyDnVec(z_vec->desc);
    free(z_vec);

    cudaFree(q_vec->val);
    cusparseDestroyDnVec(q_vec->desc);
    free(q_vec);

    cudaFree(r_vec->val);
    cusparseDestroyDnVec(r_vec->desc);
    free(r_vec);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    }
  else {
    printf("ERROR: could not open file %s\n", name);
  }
  //printf("Done!\n");

  return;
}

/*int main (void)
  {
  call_CuCG();
  return 0;
  }*/
