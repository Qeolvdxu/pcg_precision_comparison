#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "CCG.h"
#include "my_crs_matrix.h"
#include "trisolv.h"

void CCG(my_crs_matrix *A, my_crs_matrix *M, double *b, double *x, int max_iter,
         double tolerance, int *iter, double *elapsed, double *fault_elapsed,
         int k, int *crit_index) {
  int fault_freq = 1;
  int n = A->n;
  double s_abft_tol = tolerance;
  double *r = (double *)malloc(n * sizeof(double));
  double *p = (double *)malloc(n * sizeof(double));
  double *q = (double *)malloc(n * sizeof(double));
  double *z = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));
  double *fault_buff = (double *)malloc(n * sizeof(double));
  // double *temp = (double *)malloc(n * sizeof(double));

  double *acChecksum = (double *)malloc(n * sizeof(double));
  calculate_checksum(A->val, A->col, A->rowptr, A->n, acChecksum);


  //#ifdef STORE_PATH
  double **path = NULL;
  //#endif

  double alpha = 0.0;
  double beta = 0.0;

  my_crs_matrix *MT;
  if (M) {
    MT = (my_crs_matrix *)malloc(sizeof(my_crs_matrix));
    MT = sparse_transpose(M);
    if (!isLowerTriangular(M)) {
      printf("Error: The precondtioner is NOT lower triangular.\n");
      // exit(1);
    }
  }
  int j = 0;

  double v = 0.0;
  double Rho = 0.0;
  double Rtmp = 0.0;

  double res_norm = 0.0;
  double init_norm = 0.0;
  double ratio = 0.0;

  double Tiny = 1e-27;

  // x = zeros
  for (int i = 0; i < n; i++)
    x[i] = 0;

  // r = b - A*x
  matvec(A, x, r);
  for (int i = 0; i < n; i++)
    r[i] = b[i] - r[i];

  for (j = 0; j < n; j++)
    z[j] = 0.0;

  for (int i = 0; i < n; i++)
    p[i] = z[i];
  for (int i = 0; i < n; i++)
    q[i] = 1;

  // x = zeros
  for (int i = 0; i < n; i++)
    x[i] = 0;

  res_norm = init_norm = norm(n, r);
  if (init_norm == 0.0)
    init_norm = 1.0;
  ratio = 1.0;

  // WALL TIME
  double start;
  double end;
  double fault_start;
  double fault_end;
  start = omp_get_wtime();

  // main CG loop
  int itert = 0;
  while (itert < max_iter && ratio > tolerance) {
    itert++;

#ifdef ENABLE_TESTS
    printf("\nITERATION %d\n", itert);
#endif

    // Precondition
    // z = MT\(M\r);
    if (M && M->val) {
      forwardSubstitutionCSR(M, r, y);
      // matvec(M, y, temp);
      //  printVector("M*y=r\n r ", r, n);
      //  printVector("ans ", temp, n);

#ifdef PRECOND_FAULT_CHECK
      /*fault_start = omp_get_wtime();
      if (itert % fault_freq == 0) {
        if (1 ==
            s_abft_forsub(M->val, M->col, M->rowptr, M->n, r, y, s_abft_tol)) {
              printf(" ERROR CPU (ITERATION %d): S-ABFT DETECTED FAULT IN FORWARD"
              "SUB \n",
              itert);
            // exit(1);
        }
        fault_end = omp_get_wtime();
        *fault_elapsed += (fault_end - fault_start) * 1000;
      }*/
#endif
      backwardSubstitutionCSR(MT, y, z);
      // matvec(MT, z, temp);
      //  printVector("MT*z=y\n y ", y, n);
      //  printVector("ans ", temp, n);
#ifdef PRECOND_FAULT_CHECK
      /*if (itert % fault_freq == 0) {
        fault_start = omp_get_wtime();
        if (1 ==
          s_abft_backsub(M->val, M->col, M->rowptr, M->n, y, z, s_abft_tol)) {
            /*printf("ERROR CPU (ITERATION %d): S-ABFT DETECTED FAULT IN BACKWARD
            " "SUB \n", itert);
          // exit(1);
        }
        fault_end = omp_get_wtime();
        *fault_elapsed += (fault_end - fault_start) * 1000;
      }*/
#endif
    } else
      for (j = 0; j < n; j++)
        z[j] = r[j];

#ifdef ENABLE_TESTS
    printf("z[1] = %lf\n", z[1]);
#endif

    // Rho = r * z
    for (j = 0, Rho = 0.0; j < n; j++)
      Rho += r[j] * z[j];
#ifdef ENABLE_TESTS
    printf("Rho = %lf\n", Rho);
#endif
    // p = z + beta * p
    if (itert == 1) {
      for (j = 0; j < n; j++)
        p[j] = z[j];
    } else {
      // beta = dot(r,z) / v
      //
      // beta = Rho / (v + Tiny);
      beta = Rho / (v + Tiny);
      for (j = 0; j < n; j++)
        p[j] = z[j] + (beta * p[j]);
    }
#ifdef ENABLE_TESTS
    printf("beta = %.11lf\n%.11lf / (%.11lf + %.11lf)\n", beta, Rho, v, Tiny);
    printf("p[1] = %lf\n", p[1]);
#endif

    // q = A*p
// inject the error
#ifdef INJECT_ERROR
    if (itert == 5 && k != -1) 
      vecErrorInj(p, n, k);
#endif

    matvec(A, p, q);

#ifdef FAULT_CHECK
  printf("CHECKINNN FAULTS!!\n");
    if (itert % fault_freq == 0) {
      fault_start = omp_get_wtime();
      if (0 != abft_spmv_selective(A->val, A->col, A->rowptr, n, p, q, fault_buff, s_abft_tol, n/4, crit_index))
      {
        printf("CPU FAULT DETECTED!\n");
      }

      //s_abft_spmv(acChecksum, A->n, p, q, s_abft_tol);
      fault_end = omp_get_wtime();
      *fault_elapsed += (fault_end - fault_start) * 1000;
    }
#endif

#ifdef ENABLE_TESTS
    printf("S-ABFT : 0\n");
    printf("q[1] = %lf\n", q[1]);
#endif

    for (j = 0, Rtmp = 0.0; j < n; j++)
      Rtmp += p[j] * q[j];
#ifdef ENABLE_TESTS
    printf("Rtmp = %lf\n", Rtmp);
#endif

    // v = early dot(r,z)
    v = dot(r, z, n);
#ifdef ENABLE_TESTS
    printf("v = %lf\n", v);
#endif

    // alpha = v / dot(p,q)
    alpha = Rho / (Rtmp + Tiny);
#ifdef ENABLE_TESTS
    printf("alpha = %lf\n", alpha);
#endif

    // x = x + alpha * p
    for (j = 0; j < n; j++)
      x[j] = x[j] + (alpha * p[j]);
#ifdef ENABLE_TESTS
    printf("x[1] = %.11lf\n", x[1]);
#endif

    // r = r - alpha * q
    for (j = 0; j < n; j++)
      r[j] -= alpha * q[j];
#ifdef ENABLE_TESTS
    printf("r[1] = %lf\n", r[1]);
#endif

    Rho = 0.0;
    res_norm = norm(n, r);
#ifdef ENABLE_TESTS
    printf("res norm = %lf\n", res_norm);
#endif

    ratio = res_norm / init_norm;
#ifdef ENABLE_TESTS
    printf("ratio = %0.11lf\n", ratio);
#endif
    if (itert > 1) {
      // printf("RVECONE = %.50lf\n", r[1]);
      matvec(A, x, r);
      // printf("RVECONE = %.50lf\n", r[1]);

      /*if (itert % fault_freq == 0) {
        fault_start = omp_get_wtime();
        if (1 ==
            s_abft_spmv(A->val, A->col, A->rowptr, A->n, x, r, s_abft_tol)) {
          printf("ERROR CPU (ITERATION %d): S-ABFT DETECTED FAULT IN SPMV 3 "
          "A*x=r \n",
          itert);
        }
        fault_end = omp_get_wtime();
        *fault_elapsed += (fault_end - fault_start) * 1000;
      }*/
#ifdef ENABLE_TESTS
      printf("S-ABFT : 0\n");
      printf("r[1] = %lf\n", r[1]);
#endif
      for (j = 0; j < n; j++)
        r[j] = b[j] - r[j];
    }
#ifdef ENABLE_TESTS
    printf("r[1] = %lf\n", r[1]);
#endif
    /*printf("\nend of iteration %d\n x1 = %lf \t alpha= %lf \t beta= %lf \t"
      "res_norm ="
      "%lf"
      "\n v "
      "= %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm "
      "ratio(%lf) > tolerance(%lf)\n\n\n",
      iter, x[0], alpha, beta, res_norm, v, r[0], p[0], q[0], z[0],
      ratio, tolerance);*/
    //#ifdef STORE_PATH
    path = (double **)realloc(path, itert * sizeof(double *));
    path[itert - 1] = x;
    //#endif

#ifdef ENABLE_TESTS
    fflush(stdout);
    // printf("\e[1;1H\e[2J");
#endif
  }
  *iter = itert;

#ifdef STORE_PATH
  FILE *file = fopen("path.csv", "w");
  for (int i = 0; i < itert; i++) {
    for (int j = 0; j < n; j++) {
      fprintf(file, "%lf", path[i][j]);
      if (j < n - 1) {
        fprintf(file, ",");
      }
    }
    fprintf(file, "\n");
  }
  fclose(file);
#endif
  //  WALL
  end = omp_get_wtime();
  *elapsed = (end - start) * 1000;

  free(r);
  free(p);
  free(q);
  free(z);
  // exit(1);
  return;
}

// find z = M^(-1)r
/*void precondition(my_crs_matrix *M, my_crs_matrix *L, double *r,
  double *z)
{
  int n = M->n;
  int i, j;

  double *y = (double *)malloc(sizeof * n);
  printf("test 1\n");

  for (i = 0; i < n; i++) {
    y[i] = r[i];
    for (j = M->rowptr[i]; j < M->rowptr[i + 1]; j++) {
      if (M->col[j] < i)
        continue;
      y[i] -= L[j] * y[M->col[j]];
    }
    y[i] /= L[M->rowptr[i]];
  }
  printf("test 2\n");

  for (int i = n - 1; i >= 0; i--) {
    z[i] = y[i];
    for (int j = M->rowptr[i]; j < M->rowptr[i + 1]; j++) {
      if (M->col[j] <= i)
        continue;
      z[i] -= L[j] * z[M->col[j]];
    }
    z[i] /= L[M->rowptr[i]];
  }
  free(L);
  free(y);
}
*/
void forwardsub(my_crs_matrix *A, double *b, double *x) {
  int n = A->n;
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    int row_start = A->rowptr[i];
    int row_end = A->rowptr[i + 1];
    for (int j = row_start; j < row_end; j++) {
      int col = A->col[j];
      sum += A->val[j] * x[col];
    }
    x[i] = (b[i] - sum) / A->val[row_end - 1];
  }
}

double matvec_dot(my_crs_matrix *A, double *x, double *y, int n) {

  double result = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      result = result + x[i] * A->val[j] * y[A->col[j]];
      // printf("result += %lf * %lf * %lf\n", x[i] , A->val[j] ,
      // y[A->col[j]]);
    }
    /*     if (result != result && i % 20 == 0)
    printf("NaN moment :(\n");*/
  }
  return result;
}

// find the dot product of two vectors

double dot(double *v, double *u, int n) {

  double x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;
}

void faulty_matvec(my_crs_matrix *A, double *x, double *y, int k) {
  int n = A->n;
  vecErrorInj(x, n, k);
  for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      // printf("%d ? %d\n", A->col[j], n);
      y[i] += A->val[j] * x[A->col[j]];
    }
  }
}

void matvec(my_crs_matrix *A, double *x, double *y) {
  int n = A->n;
  for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      // printf("%d ? %d\n", A->col[j], n);
      y[i] += A->val[j] * x[A->col[j]];
    }
  }
}

// find the norm of a vector
double norm(int n, double *v) {
  double ssq, scale, absvi;
  int i;

  if (n == 1)
    return fabs(v[0]);

  scale = 0.0;
  ssq = 1.0;

  for (i = 0; i < n; i++) {
    if (v[i] != 0) {
      absvi = fabs(v[i]);
      if (scale < absvi) {
        ssq = 1.0 + ssq * (scale / absvi) * (scale / absvi);
        scale = absvi;
      } else
        ssq = ssq + (absvi / scale) * (absvi / scale);
    }
  }
  return scale * sqrt(ssq);
}
