#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/CUSTOMIZE.h"
#include "../include/CuCG.h"
#include "../include/trisolv.h"

typedef struct {
  cusparseDnVecDescr_t desc;
  double *val;
} my_cuda_vector;

typedef struct {
  cusparseSpMatDescr_t desc;
  cusparseMatDescr_t desctwo;
  int n;
  int m;
  int nz;
  double *val;
  int *col;
  int *rowptr;
} my_cuda_csr_matrix;

void printGPUVector(const double *gpuVector, int size) {
  // Allocate memory for a temporary host vector to store the data from the GPU
  double *hostVector = (double *)malloc(size * sizeof(double));

  // Copy data from the GPU to the host
  cudaMemcpy(hostVector, gpuVector, size * sizeof(double),
             cudaMemcpyDeviceToHost);

  // Print the vector elements
  printf("GPU Vector: ");
  for (int i = 0; i < size; i++) {
    printf("%.2f ", hostVector[i]);
  }
  printf("\n");

  // Free the host memory
  free(hostVector);
}

void cusparse_conjugate_gradient(
    my_cuda_csr_matrix *A, double *val_host_M, int *col_host_M,
    int *rowptr_host_M, double *val_host_A, int *col_host_A, int *rowptr_host_A,
    my_cuda_csr_matrix *M, my_cuda_vector *b, my_cuda_vector *x,
    my_cuda_vector *r_vec, my_cuda_vector *p_vec, my_cuda_vector *q_vec,
    my_cuda_vector *z_vec, my_cuda_vector *y_vec, int max_iter,
    double tolerance, int *iter, double *elapsed, double *mem_elapsed,
    double *fault_elapsed, cusparseHandle_t *handle,
    cublasHandle_t *handle_blas) {
#ifdef ENABLE_TESTS
  printf("start cg!\n");
  // tolerance = 1e-6;
#endif

  int fault_freq = 10;

  double faultcheck_start;
  double faultcheck_end;
  // fault_elapsed = 0.0;

  double memcheck_start;
  double memcheck_end;

  double s_abft_tol = tolerance;

  /*double *val_host_M = malloc(sizeof(double) * M->nz);
  int *col_host_M = malloc(sizeof(int) * M->nz);
  int *rowptr_host_M = malloc(sizeof(int) * (M->n + 1));
  cudaMemcpy(rowptr_host_M, M->rowptr, (M->n + 1) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(col_host_M, M->col, M->nz * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(val_host_M, M->val, M->nz * sizeof(double),
             cudaMemcpyDeviceToHost);

  double *val_host_A = malloc(sizeof(double) * A->nz);
  int *col_host_A = malloc(sizeof(int) * A->nz);
  int *rowptr_host_A = malloc(sizeof(int) * (A->n + 1));
  cudaMemcpy(rowptr_host_A, A->rowptr, (A->n + 1) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(col_host_A, A->col, A->nz * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(val_host_A, A->val, A->nz * sizeof(double),
             cudaMemcpyDeviceToHost);*/

  double *fault_vec_one = (double *)malloc(sizeof(double) * A->n);
  double *fault_vec_two = (double *)malloc(sizeof(double) * A->n);

  int error;

  cusparseStatus_t M_status;
  cusparseStatus_t MT_status;

  int n = A->n;

  double *xfive = (double *)malloc(sizeof(double) * 5);

#ifdef ENABLE_TESTS
  double *onex = (double *)malloc(sizeof(double) * n);
  double *onez = (double *)malloc(sizeof(double) * n);
  double *oner = (double *)malloc(sizeof(double) * n);
  double *oneq = (double *)malloc(sizeof(double) * n);
  double *onep = (double *)malloc(sizeof(double) * n);
#endif

  double beta = 0.0;
  double alpha = 0.0;
  const double ne_one = -1.0;
  const double n_one = 1.0;
  const double one = 0.0;

  int itert = 0;

  double v = 0;
  double Rho = 0;
  double Rtmp = 0;

  double res_norm = 0;
  double init_norm = 0;
  double ratio = 0;

  double Tiny = 1e-27;
  double minus_alpha = 0.0;

  // x is already zero

  size_t bufferSizeMV;
  void *buff;

#ifdef ENABLE_TESTS
  cudaMemcpy(onex, x->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(onep, p_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(oneq, q_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(oner, r_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(onez, z_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  printf("\nINITIAL VEC CREATION\n x1 = %lf \t alpha= %lf \t beta= %lf "
         "\n v "
         "= %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm "
         "ratio(%lf) > tolerance(%lf)\n\n\n",
         onex[0], alpha, beta, v, oner[0], onep[0], oneq[0], onez[0], ratio,
         tolerance);
#endif
  // r = A * x
  cusparseSpMV_bufferSize(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &n_one,
                          A->desc, b->desc, &one, x->desc, CUDA_PRECI_DT_DEVICE,
                          CUSPARSE_MV_ALG_DEFAULT, &bufferSizeMV);
  // printf("bufferSizeMV: %zu\n", bufferSizeMV);
  cudaMalloc(&buff, bufferSizeMV);

  cusparseSpMV(*handle,
               CUSPARSE_OPERATION_NON_TRANSPOSE, // operation
               &n_one,                           // alpha
               A->desc,                          // matrix
               x->desc,                          // vector
               &one,                             // beta
               r_vec->desc,                      // answer
               CUDA_PRECI_DT_DEVICE,             // data type
               CUSPARSE_MV_ALG_DEFAULT,          // algorithm
               buff                              // buffer
  );

  if (itert % fault_freq == 0) {
    cudaMemcpy(rowptr_host_A, A->rowptr, (A->n + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(col_host_A, A->col, A->nz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(val_host_A, A->val, A->nz * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(fault_vec_one, x->val, A->n * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(fault_vec_two, r_vec->val, A->n * sizeof(double),
               cudaMemcpyDeviceToHost);
    faultcheck_start = omp_get_wtime();
    if (1 == s_abft_spmv(val_host_A, col_host_A, rowptr_host_A, A->n,
                         fault_vec_one, fault_vec_two, s_abft_tol)) {
      /*printf("ERROR (ITERATION %d): S-ABFT DETECTED FAULT IN SPMV 1 A*x=r \n",
             itert);*/
      // exit(1);
    }
    faultcheck_end = omp_get_wtime();
    *fault_elapsed += (faultcheck_end - faultcheck_start) * 1000;
  }

  // r = b - r
  AXPY_FUN(*handle_blas, n, &ne_one, r_vec->val, 1, b->val, 1);
  COPY_FUN(*handle_blas, n, b->val, 1, r_vec->val, 1);

  /*  if (M) { // z = MT\(M\r);
      csrsv2Info_t info;
      cusparseCreateCsrsv2Info(&info);
      SPSV_FUN(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M->m, M->nz, &n_one,
               M->desctwo, M->val, M->rowptr, M->col, info, r_vec->val,
               z_vec->val, CUSPARSE_SOLVE_POLICY_NO_LEVEL, NULL);
    } else {*/
  // z = r
  // COPY_FUN(*handle_blas, n, r_vec->val, 1, z_vec->val, 1);
  //}
  //
  // p = z
  COPY_FUN(*handle_blas, n, z_vec->val, 1, p_vec->val, 1);
  // COPY_FUN(*handle_blas, n, x->val, 1, z_vec->val, 1);
  NORM_FUN(*handle_blas, n, r_vec->val, 1, &res_norm);
  init_norm = res_norm;
  ratio = 1.0;

#ifdef ENABLE_TESTS
  cudaMemcpy(onex, x->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(onep, p_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(oneq, q_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(oner, r_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaMemcpy(onez, z_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
/*printf("PREQUEL \n x1 = %lf \t alpha= %lf \t beta= %lf "
       "\n v "
       "= %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm "
       "ratio(%lf) > tolerance(%e)\n\n\n",
       iter, onex[0], alpha, beta, v, oner[0], onep[0], oneq[0], onez[0], ratio,
       tolerance);*/
#endif

  // WALL TIME
  double start;
  double end;

  csrsv2Info_t info_nontrans;
  csrsv2Info_t info_trans;
  int bufferSize_nontrans;
  void *pBuffer_nontrans;
  int bufferSize_trans;
  void *pBuffer_trans;
  // error = cudaGetLastError();
  // printf("1 %s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
  if (M != NULL) {
    // printf("PRECOND!\n");
    //  Create the cuSPARSE CSR triangular solve info structures
    cusparseCreateCsrsv2Info(&info_nontrans);

    cusparseCreateCsrsv2Info(&info_trans);

    // Get the buffer size for the non-transpose analysis phase
    cusparseDcsrsv2_bufferSize(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M->m,
                               M->nz, M->desctwo, M->val, M->rowptr, M->col,
                               info_nontrans, &bufferSize_nontrans);
    cudaMalloc(&pBuffer_nontrans, bufferSize_nontrans);
    // bufferSize_nontrans = 0;

    // Allocate the buffer for the non-transpose analysis phase

    // Perform the analysis phase for CUSPARSE_OPERATION_NON_TRANSPOSE with
    // CUSPARSE_SOLVE_POLICY_NO_LEVEL
    /*error = cudaGetLastError();
    printf("2 %s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
    error = cudaGetLastError();
    printf("2.5 %s - %s\n", cudaGetErrorName(error),
    cudaGetErrorString(error));*/

    cusparseDcsrsv2_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M->m,
                             M->nz, M->desctwo, M->val, M->rowptr, M->col,
                             info_nontrans, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                             pBuffer_nontrans);
    error = cudaGetLastError();
    // printf("3 %s - %s\n", cudaGetErrorName(error),
    // cudaGetErrorString(error));

    // Get the buffer size for the transpose analysis phase
    cusparseDcsrsv2_bufferSize(*handle, CUSPARSE_OPERATION_TRANSPOSE, M->m,
                               M->nz, M->desctwo, M->val, M->rowptr, M->col,
                               info_trans, &bufferSize_trans);
    cudaMalloc(&pBuffer_trans, bufferSize_trans);
    // bufferSize_trans = 0;

    // printf("bufferSizeLT: %d\n", bufferSize_nontrans);
    // printf("bufferSizeUT: %d\n", bufferSize_trans);
    //  Allocate the buffer for the transpose analysis phase

    // Perform the analysis phase for CUSPARSE_OPERATION_TRANSPOSE with
    // CUSPARSE_SOLVE_POLICY_USE_LEVEL
    cusparseDcsrsv2_analysis(*handle, CUSPARSE_OPERATION_TRANSPOSE, M->m, M->nz,
                             M->desctwo, M->val, M->rowptr, M->col, info_trans,
                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_trans);
    error = cudaGetLastError();
    // printf("4 %s - %s\n", cudaGetErrorName(error),
    // cudaGetErrorString(error));
  }

  start = omp_get_wtime();
  while (itert < max_iter && ratio > tolerance) {
#ifdef ENABLE_TESTS
    printf("\nITERATION %d\n", itert);
#endif
    itert++;

    // Check X value for faults every nth iteration
    /*if (itert % 5 == 0) {
      faultcheck_start = omp_get_wtime();
      faultcheck_end = omp_get_wtime();
      *fault_elapsed += (faultcheck_end - faultcheck_start) * 1000;
    }*/

    if (M != NULL) { // z = MT\(M\r);
      // printf("APPLYING PRECOND\n");
      //    M*y=r
      /*printf("Y ");
      printGPUVector(y_vec->val, M->n);
      printf("R ");
      printGPUVector(r_vec->val, M->n);
      printf(" SOLVED! ");*/
      M_status = cusparseDcsrsv2_solve(
          *handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M->m, M->nz, &n_one,
          M->desctwo, M->val, M->rowptr, M->col, info_nontrans, r_vec->val,
          y_vec->val, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_nontrans);
      /*printf("Y ");
      printGPUVector(y_vec->val, M->n);
      printf("R ");
      printGPUVector(r_vec->val, M->n);*/
      // error = cudaGetLastError();
      //  printf("fs %s - %s\n", cudaGetErrorName(error),
      // cudaGetErrorString(error));
      if (itert % fault_freq == 0) {
        memcheck_start = omp_get_wtime();
        cudaMemcpy(fault_vec_one, r_vec->val, M->n * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(fault_vec_two, y_vec->val, M->n * sizeof(double),
                   cudaMemcpyDeviceToHost);
        memcheck_end = omp_get_wtime();
        *mem_elapsed += (memcheck_end - memcheck_start) * 1000;
        // cudaDeviceSynchronize();
        faultcheck_start = omp_get_wtime();
        if (1 == s_abft_forsub(val_host_M, col_host_M, rowptr_host_M, n,
                               fault_vec_one, fault_vec_two, s_abft_tol)) {
          /*printf("ERROR GPU (ITERATION %d): S-ABFT DETECTED FAULT IN FORWARD "
                 "SUB \n",
                 itert);*/
          /*printf("Error: cusparseDcsrsv2_solve failed with status %d\n",
                 M_status);*/
          // exit(1);
        }
        faultcheck_end = omp_get_wtime();
        *fault_elapsed += (faultcheck_end - faultcheck_start) * 1000;
      }
      //  M*z=y
      MT_status = cusparseDcsrsv2_solve(
          *handle, CUSPARSE_OPERATION_TRANSPOSE, M->m, M->nz, &n_one,
          M->desctwo, M->val, M->rowptr, M->col, info_trans, y_vec->val,
          z_vec->val, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_trans);
      error = cudaGetLastError();
      // printf("bs %s - %s\n", cudaGetErrorName(error),
      // cudaGetErrorString(error));

      if (itert % fault_freq == 0) {
        memcheck_start = omp_get_wtime();
        cudaMemcpy(fault_vec_one, y_vec->val, M->n * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(fault_vec_two, z_vec->val, M->n * sizeof(double),
                   cudaMemcpyDeviceToHost);
        memcheck_end = omp_get_wtime();
        *mem_elapsed += (memcheck_end - memcheck_start) * 1000;
        // cudaDeviceSynchronize();

        faultcheck_start = omp_get_wtime();
        if (1 == s_abft_backsub(val_host_M, col_host_M, rowptr_host_M, n,
                                fault_vec_one, fault_vec_two, s_abft_tol)) {
          /*printf("ERROR GPU (ITERATION %d): S-ABFT DETECTED FAULT IN BACKWARD
             " "SUB \n", itert);*/
          /*printf("Error: cusparseDcsrsv2_solve failed with status %d\n",
                 MT_status);*/
          // exit(1);
        }
        faultcheck_end = omp_get_wtime();
        *fault_elapsed += (faultcheck_end - faultcheck_start) * 1000;
      }
    } else // z = r
      COPY_FUN(*handle_blas, n, r_vec->val, 1, z_vec->val, 1);

#ifdef ENABLE_TESTS
    cudaMemcpy(onez, z_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(oner, r_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("z[%d] = %lf & r[%d] = %lf\n", 1, onez[1], 1, oner[1]);
#endif

    // Rho = r z dot prod
    DOT_FUN(*handle_blas, n, r_vec->val, 1, z_vec->val, 1, &Rho);

#ifdef ENABLE_TESTS
    printf("Rho = %lf\n", Rho);
#endif

    // p = z + (beta * p)
    if (itert == 1) {
      COPY_FUN(*handle_blas, n, z_vec->val, 1, p_vec->val, 1);
    } else {
      beta = Rho / (v + Tiny);
      SCAL_FUN(*handle_blas, n, &beta, p_vec->val, 1);
      AXPY_FUN(*handle_blas, n, &n_one, z_vec->val, 1, p_vec->val, 1);
    }

#ifdef ENABLE_TESTS
    cudaMemcpy(onep, p_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("p[1] = %lf\n", onep[1]);
#endif

    cusparseSpMV(*handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, // operation
                 &n_one,                           // alpha
                 A->desc,                          // matrix
                 p_vec->desc,                      // vector
                 &one,                             // beta
                 q_vec->desc,                      // answer
                 CUDA_PRECI_DT_DEVICE,             // data type
                 CUSPARSE_MV_ALG_DEFAULT,          // algorithm
                 buff                              // buffer
    );
    if (itert % fault_freq == 0) {
      memcheck_start = omp_get_wtime();
      cudaMemcpy(fault_vec_one, p_vec->val, A->n * sizeof(double),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(fault_vec_two, q_vec->val, A->n * sizeof(double),
                 cudaMemcpyDeviceToHost);
      memcheck_end = omp_get_wtime();
      *mem_elapsed += (memcheck_end - memcheck_start) * 1000;
      faultcheck_start = omp_get_wtime();
      if (1 == s_abft_spmv(val_host_A, col_host_A, rowptr_host_A, A->n,
                           fault_vec_one, fault_vec_two, s_abft_tol)) {
        /*printf(
            "ERROR GPU (ITERATION %d): S-ABFT DETECTED FAULT IN SPMV 2 A*p=q\n",
            itert);*/
        // exit(1);
      }
      faultcheck_end = omp_get_wtime();
      *fault_elapsed += (faultcheck_end - faultcheck_start) * 1000;
    }
#ifdef ENABLE_TESTS
    cudaMemcpy(oneq, q_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("q[1] = %lf\n", oneq[1]);
#endif

    // Rtmp = p q dot prod
    DOT_FUN(*handle_blas, n, p_vec->val, 1, q_vec->val, 1, &Rtmp);

#ifdef ENABLE_TESTS
    printf("Rtmp = %lf\n", Rtmp);
#endif

    // v = r z dot prod
    DOT_FUN(*handle_blas, n, r_vec->val, 1, z_vec->val, 1, &v);

#ifdef ENABLE_TESTS
    cudaMemcpy(onep, p_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("v = %lf\n", v);
#endif

    // alpha
    alpha = Rho / (Rtmp + Tiny);

#ifdef ENABLE_TESTS
    printf("alpha = %lf\n", alpha);
#endif

    // x = x + alpha * p
    AXPY_FUN(*handle_blas, n, &alpha, p_vec->val, 1, x->val, 1);

#ifdef ENABLE_TESTS
    cudaMemcpy(onex, x->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("x[1] = %lf\n", onex[1]);
#endif

    // r = r - alpha * q
    minus_alpha = -alpha;
    AXPY_FUN(*handle_blas, n, &minus_alpha, q_vec->val, 1, r_vec->val, 1);

#ifdef ENABLE_TESTS
    cudaMemcpy(oner, r_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("r[1] = %lf\n", oner[1]);
#endif

    Rho = 0.0;
    NORM_FUN(*handle_blas, n, r_vec->val, 1, &res_norm);

#ifdef ENABLE_TESTS
    printf("res_norm = %lf\n", res_norm);
#endif

    ratio = res_norm / init_norm;

#ifdef ENABLE_TESTS
    printf("ratio = %lf\n", ratio);
#endif

    cusparseSpMV(*handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, // operation
                 &n_one,                           // alpha
                 A->desc,                          // matrix
                 x->desc,                          // vector
                 &one,                             // beta
                 r_vec->desc,                      // answer
                 CUDA_PRECI_DT_DEVICE,             // data type
                 CUSPARSE_MV_ALG_DEFAULT,          // algorithm
                 buff                              // buffer
    );
    if (itert % fault_freq == 0) {
      memcheck_start = omp_get_wtime();
      cudaMemcpy(fault_vec_one, x->val, A->n * sizeof(double),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(fault_vec_two, r_vec->val, A->n * sizeof(double),
                 cudaMemcpyDeviceToHost);
      memcheck_end = omp_get_wtime();
      *mem_elapsed += (memcheck_end - memcheck_start) * 1000;
      faultcheck_start = omp_get_wtime();
      if (1 == s_abft_spmv(val_host_A, col_host_A, rowptr_host_A, A->n,
                           fault_vec_one, fault_vec_two, s_abft_tol)) {
        /*printf(
            "ERROR GPU (ITERATION %d): S-ABFT DETECTED FAULT IN SPMV 3 A*x=r\n",
            itert);*/
        // exit(1);
      }
      faultcheck_end = omp_get_wtime();
      *fault_elapsed += (faultcheck_end - faultcheck_start) * 1000;
    }

#ifdef ENABLE_TESTS
    cudaMemcpy(oner, r_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("r[1] = %lf\n", oner[1]);
#endif

    // r = b - r
    AXPY_FUN(*handle_blas, n, &ne_one, b->val, 1, r_vec->val, 1);
    SCAL_FUN(*handle_blas, n, &ne_one, r_vec->val, 1);

#ifdef ENABLE_TESTS
    cudaMemcpy(oner, r_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("r[1] = %lf\n", oner[1]);
    error = cudaGetLastError();
    printf("%s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
    // printf("\e[1;1H\e[2J");
#endif
  }
  cudaDeviceSynchronize();

  // WALL TIME
  end = omp_get_wtime();
  *elapsed = (end - start) * 1000;

  cudaFree(buff);
  cudaFree(pBuffer_trans);
  cudaFree(pBuffer_nontrans);

#ifdef ENABLE_TESTS
  printf("TOtal of %d CuCG ITerations\n", itert);
#endif

  *iter = itert;
  return;
}

my_cuda_csr_matrix *cusparse_crs_read(char *name) {
  my_cuda_csr_matrix *M =
      (my_cuda_csr_matrix *)malloc(sizeof(my_cuda_csr_matrix));
  double *val;
  int *col;
  int *rowptr;

  int n = 0;
  int m = 0;
  int nz = 0;

  FILE *file;
  if ((file = fopen(name, "r"))) {
    int i;

    if (fscanf(file, "%d %d %d", &m, &n, &nz) < 0) {
      printf("error scanning head file %s\n", name);
    }

    val = (double *)malloc(sizeof(double) * nz);

    col = (int *)malloc(sizeof(int) * nz);
    rowptr = (int *)malloc(sizeof(int) * n + 1);

    for (i = 0; i <= n; i++) {
      if (fscanf(file, "%d ", &rowptr[i]) < 0) {
        printf("error scanning rowptr file %s\n", name);
        break;
      }
    }
    for (i = 0; i < nz; i++) {
      if (fscanf(file, "%d ", &col[i]) < 0) {
        printf("error scanning col file %s\n", name);
        break;
      }
    }
    for (i = 0; i < nz; i++) {
      if (fscanf(file, CUDA_PRECI_S, &val[i]) < 0) {
        printf("error scanning val file %s\n", name);
        break;
      }
    }

#ifdef ENABLE_TESTS
    printf("READ rowptr : ");
    for (i = 0; i <= n; i++)
      printf("%d ", rowptr[i]);
    printf("\n");

    printf("READ col : ");
    for (i = 0; i < nz; i++)
      printf("%d ", col[i]);
    printf("\n");

    printf("READ val : ");
    for (i = 0; i < nz; i++)
      printf(CUDA_PRECI_S, val[i]);
    printf("\n");
#endif

    fclose(file);

    cudaMalloc((void **)&M->rowptr, (n + 1) * sizeof(int));
    cudaMalloc((void **)&M->col, nz * sizeof(int));
    cudaMalloc((void **)&M->val, nz * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(M->rowptr, rowptr, (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(M->col, col, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(M->val, val, nz * sizeof(double), cudaMemcpyHostToDevice);

    M->n = n;
    M->m = m;
    M->nz = nz;
    cusparseCreateCsr(&M->desc, n, n, nz, M->rowptr, M->col, M->val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_PRECI_DT_DEVICE);
    // Create the CSR matrix

  } else {
    printf("ERROR: could not open file %s\n", name);
    n = -1;
  }
  return M;
}

void call_CuCG(char *name, char *m_name, double *h_b, double *h_x, int maxit,
               double tol, int *iter, double *elapsed, double *mem_elapsed,
               double *fault_elapsed) {

  int error;
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  cusparseHandle_t cusparseHandle;
  cusparseCreate(&cusparseHandle);

  int n = 0;
  int m = 0;
  int nz = 0;

  int m_n = 0;
  int m_m = 0;
  int m_nz = 0;

  FILE *file;
  FILE *m_file;

  if (cusparseHandle == NULL || cublasHandle == NULL) {
    printf("Error creating cuSparse or cuBLAS handle!\n");
    return;
  }

  if ((file = fopen(name, "r")) == NULL) {
    printf("Error opening file %s\n", name);
    return;
  }

  if (fscanf(file, "%d %d %d", &m, &n, &nz) < 0) {
    printf("Error scanning head file %s\n", name);
    fclose(file);
    return;
  }
  int n_t = n;

  double *val = (double *)malloc(sizeof(double) * nz);
  int *col = (int *)malloc(sizeof(int) * nz);
  int *rowptr = (int *)malloc(sizeof(int) * (n + 1));
  double *h_rpqz = (double *)malloc(sizeof(double) * n);

  for (int i = 0; i <= n; i++) {
    if (fscanf(file, "%d ", &rowptr[i]) < 0) {
      printf("Error scanning rowptr file %s\n", name);
      fclose(file);
      free(val);
      free(col);
      free(rowptr);
      free(h_rpqz);
      return;
    }
  }
  for (int i = 0; i < nz; i++) {
    if (fscanf(file, "%d ", &col[i]) < 0) {
      printf("Error scanning col file %s\n", name);
      fclose(file);
      free(val);
      free(col);
      free(rowptr);
      free(h_rpqz);
      return;
    }
  }
  for (int i = 0; i < nz; i++) {
    if (fscanf(file, CUDA_PRECI_S, &val[i]) < 0) {
      printf("Error scanning val file %s\n", name);
      fclose(file);
      free(val);
      free(col);
      free(rowptr);
      free(h_rpqz);
      return;
    }
  }
  fclose(file);

  if (m_name && (m_file = fopen(m_name, "r")) == NULL) {
    printf("Error opening preconditioner file %s\n", m_name);
    free(val);
    free(col);
    free(rowptr);
    free(h_rpqz);
    return;
  }

  double *m_val;
  int *m_col;
  int *m_rowptr;
  if (m_name && fscanf(m_file, "%d %d %d", &m_m, &m_n, &m_nz) > 0) {
    m_val = (double *)malloc(sizeof(double) * m_nz);
    m_col = (int *)malloc(sizeof(int) * m_nz);
    m_rowptr = (int *)malloc(sizeof(int) * (m_n + 1));

    for (int i = 0; i <= m_n; i++) {
      if (fscanf(m_file, "%d ", &m_rowptr[i]) < 0) {
        printf("Error scanning m rowptr file %s\n", m_name);
        fclose(m_file);
        free(val);
        free(col);
        free(rowptr);
        free(h_rpqz);
        free(m_val);
        free(m_col);
        free(m_rowptr);
        return;
      }
    }
    for (int i = 0; i < m_nz; i++) {
      if (fscanf(m_file, "%d ", &m_col[i]) < 0) {
        printf("Error scanning m col file %s\n", m_name);
        fclose(m_file);
        free(val);
        free(col);
        free(rowptr);
        free(h_rpqz);
        free(m_val);
        free(m_col);
        free(m_rowptr);
        return;
      }
    }
    for (int i = 0; i < m_nz; i++) {
      if (fscanf(m_file, CUDA_PRECI_S, &m_val[i]) < 0) {
        printf("Error scanning m val file %s\n", m_name);
        fclose(m_file);
        free(val);
        free(col);
        free(rowptr);
        free(h_rpqz);
        free(m_val);
        free(m_col);
        free(m_rowptr);
        return;
      }
    }

    // printVector("M val: ", m_val, m_nz);
    fclose(m_file);
  }

  double start;
  double end;
  start = omp_get_wtime();

  my_cuda_csr_matrix *A_matrix =
      (my_cuda_csr_matrix *)malloc(sizeof(my_cuda_csr_matrix));
  my_cuda_csr_matrix *M_matrix = NULL;
  my_cuda_vector *b_vec = (my_cuda_vector *)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *x_vec = (my_cuda_vector *)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *r_vec = (my_cuda_vector *)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *p_vec = (my_cuda_vector *)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *q_vec = (my_cuda_vector *)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *z_vec = (my_cuda_vector *)malloc(sizeof(my_cuda_vector));
  my_cuda_vector *y_vec;

  A_matrix->n = n;
  A_matrix->m = m;
  A_matrix->nz = nz;

  if (m_name) {
    // printf("PRECOND READING!\n");
    M_matrix = (my_cuda_csr_matrix *)malloc(sizeof(my_cuda_csr_matrix));
    M_matrix->n = m_n;
    M_matrix->m = m_m;
    M_matrix->nz = m_nz;

    cudaMalloc((void **)&M_matrix->rowptr, (m_n + 1) * sizeof(int));
    cudaMalloc((void **)&M_matrix->col, m_nz * sizeof(int));
    cudaMalloc((void **)&M_matrix->val, m_nz * sizeof(double));

    cudaMemcpy(M_matrix->rowptr, m_rowptr, (m_n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(M_matrix->col, m_col, m_nz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(M_matrix->val, m_val, m_nz * sizeof(double),
               cudaMemcpyHostToDevice);

    cusparseCreateCsr(&M_matrix->desc, M_matrix->n, M_matrix->n, M_matrix->nz,
                      M_matrix->rowptr, M_matrix->col, M_matrix->val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_PRECI_DT_DEVICE);

    cusparseCreateMatDescr(&M_matrix->desctwo);
    cusparseSetMatType(M_matrix->desctwo, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(M_matrix->desctwo, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(M_matrix->desctwo, CUSPARSE_DIAG_TYPE_NON_UNIT);
    y_vec = (my_cuda_vector *)malloc(sizeof(my_cuda_vector));
    cudaMalloc((void **)&y_vec->val, m_n * sizeof(double));
    cudaMemcpy(y_vec->val, h_rpqz, m_n * sizeof(double),
               cudaMemcpyHostToDevice);
    cusparseCreateDnVec(&y_vec->desc, m_n, y_vec->val, CUDA_PRECI_DT_DEVICE);
  }

  cudaMalloc((void **)&A_matrix->rowptr, (n + 1) * sizeof(int));
  cudaMalloc((void **)&A_matrix->col, nz * sizeof(int));
  cudaMalloc((void **)&A_matrix->val, nz * sizeof(double));
  cudaMalloc((void **)&x_vec->val, n_t * sizeof(double));
  cudaMalloc((void **)&b_vec->val, n_t * sizeof(double));
  cudaMalloc((void **)&r_vec->val, n_t * sizeof(double));
  cudaMalloc((void **)&p_vec->val, n_t * sizeof(double));
  cudaMalloc((void **)&q_vec->val, n_t * sizeof(double));
  cudaMalloc((void **)&z_vec->val, n_t * sizeof(double));

  cudaMemcpy(A_matrix->rowptr, rowptr, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(A_matrix->col, col, nz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(A_matrix->val, val, nz * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_vec->val, h_x, n_t * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_vec->val, h_b, n_t * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(r_vec->val, h_rpqz, n_t * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(p_vec->val, h_rpqz, n_t * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(q_vec->val, h_rpqz, n_t * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(z_vec->val, h_rpqz, n_t * sizeof(double), cudaMemcpyHostToDevice);

  cusparseCreateDnVec(&b_vec->desc, n_t, b_vec->val, CUDA_PRECI_DT_DEVICE);
  cusparseCreateDnVec(&x_vec->desc, n_t, x_vec->val, CUDA_PRECI_DT_DEVICE);
  cusparseCreateDnVec(&r_vec->desc, n_t, r_vec->val, CUDA_PRECI_DT_DEVICE);
  cusparseCreateDnVec(&p_vec->desc, n_t, p_vec->val, CUDA_PRECI_DT_DEVICE);
  cusparseCreateDnVec(&q_vec->desc, n_t, q_vec->val, CUDA_PRECI_DT_DEVICE);
  cusparseCreateDnVec(&z_vec->desc, n_t, z_vec->val, CUDA_PRECI_DT_DEVICE);
  cusparseCreateCsr(&A_matrix->desc, n, n, nz, A_matrix->rowptr, A_matrix->col,
                    A_matrix->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_PRECI_DT_DEVICE);

  cudaDeviceSynchronize();

  end = omp_get_wtime();
  *mem_elapsed = (end - start) * 1000;

  // error = cudaGetLastError();
  // printf("CallCuCG: %s - %s\n", cudaGetErrorName(error),
  //       cudaGetErrorString(error));

  if (m_name && M_matrix) {
    // printf("PASSING PRECOND\n");
    cusparse_conjugate_gradient(
        A_matrix, m_val, m_col, m_rowptr, val, col, rowptr, M_matrix, b_vec,
        x_vec, r_vec, p_vec, q_vec, z_vec, y_vec, maxit, tol, iter, elapsed,
        mem_elapsed, fault_elapsed, &cusparseHandle, &cublasHandle);
  } else {
    printf("NOT PASSING PRECOND\n");
    cusparse_conjugate_gradient(A_matrix, NULL, NULL, NULL, val, col, rowptr,
                                NULL, b_vec, x_vec, r_vec, p_vec, q_vec, z_vec,
                                NULL, maxit, tol, iter, elapsed, mem_elapsed,
                                fault_elapsed, &cusparseHandle, &cublasHandle);
  }

  cudaMemcpy(h_x, x_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, b_vec->val, n * sizeof(double), cudaMemcpyDeviceToHost);

  free(val);
  free(col);
  free(rowptr);
  free(h_rpqz);

  cudaFree(A_matrix->val);
  cudaFree(A_matrix->rowptr);
  cudaFree(A_matrix->col);
  cusparseDestroySpMat(A_matrix->desc);
  free(A_matrix);

  if (m_name) {
    free(m_val);
    free(m_col);
    free(m_rowptr);
    cudaFree(M_matrix->val);
    cudaFree(M_matrix->col);
    cudaFree(M_matrix->rowptr);
    cusparseDestroyMatDescr(M_matrix->desctwo);
    cusparseDestroySpMat(M_matrix->desc);
    free(M_matrix);

    cudaFree(y_vec->val);
    cusparseDestroyDnVec(y_vec->desc);
    free(y_vec);
  }

  cudaFree(x_vec->val);
  cusparseDestroyDnVec(x_vec->desc);
  free(x_vec);

  cudaFree(b_vec->val);
  cusparseDestroyDnVec(b_vec->desc);
  free(b_vec);

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
