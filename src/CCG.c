#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "../include/CCG.h"
#include "../include/CUSTOMIZE.h"
#include "../include/my_crs_matrix.h"

void CCG(my_crs_matrix *A, my_crs_matrix *M, double *b, double *x, int max_iter,
         double tolerance, int *iter, double *elapsed) {
  int n = A->n;
  double *r = (double *)malloc(n * sizeof(double));

  double *p = (double *)malloc(n * sizeof(double));
  double *q = (double *)malloc(n * sizeof(double));
  double *z = (double *)malloc(n * sizeof(double));

  double alpha = 0.0;
  double beta = 0.0;

  int j = 0;

  double v = 0.0;
  double Rho = 0.0;
  double Rtmp = 0.0;

  double res_norm = 0.0;
  double init_norm = 0.0;
  double ratio = 0.0;

  double Tiny = 0.1e-27;

  // x = zeros
  for (int i = 0; i < n; i++)
    x[i] = 0;

  // r = b - A*x
  matvec(A, x, r);
  for (int i = 0; i < n; i++)
    r[i] = (C_PRECI_DT)b[i] - (C_PRECI_DT)r[i];

  // z = MT\(M\r);
  if (M)
    forwardsub(M, (C_PRECI_DT *)r, z);
  else
    for (j = 0; j < n; j++)
      z[j] = r[j];

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
  start = omp_get_wtime();

  // main CG loop
  int itert = 0;
  while (itert < max_iter && ratio > tolerance) {

#ifdef ENABLE_TESTS
    printf("\nITERATION %d\n", itert);
#endif

    itert++;

    // Precondition
    // z = MT\(M\r);
    if (M)
      forwardsub(M, (C_PRECI_DT *)r, z);
    else
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
      // beta = (C_PRECI_DT)Rho / ((C_PRECI_DT)v + (C_PRECI_DT)Tiny);
      beta = Rho / (v + Tiny);
      for (j = 0; j < n; j++)
        p[j] = (C_PRECI_DT)z[j] + ((C_PRECI_DT)beta * (C_PRECI_DT)p[j]);
    }
#ifdef ENABLE_TESTS
    printf("beta = %.11lf\n%.11lf / (%.11lf + %.11lf)\n", beta, Rho, v, Tiny);
    printf("p[1] = %lf\n", p[1]);
#endif

    // q = A*p
    matvec(A, p, q);
#ifdef ENABLE_TESTS
    printf("q[1] = %lf\n", q[1]);
#endif

    for (j = 0, Rtmp = 0.0; j < n; j++)
      Rtmp += (C_PRECI_DT)p[j] * (C_PRECI_DT)q[j];
#ifdef ENABLE_TESTS
    printf("Rtmp = %lf\n", Rtmp);
#endif

    // v = early dot(r,z)
    v = dot(r, z, n);
#ifdef ENABLE_TESTS
    printf("v = %lf\n", v);
#endif

    // alpha = v / dot(p,q)
    alpha = (C_PRECI_DT)Rho / ((C_PRECI_DT)Rtmp + (C_PRECI_DT)Tiny);
#ifdef ENABLE_TESTS
    printf("alpha = %lf\n", alpha);
#endif

    // x = x + alpha * p
    for (j = 0; j < n; j++)
      x[j] = (C_PRECI_DT)x[j] + ((C_PRECI_DT)alpha * (C_PRECI_DT)p[j]);
#ifdef ENABLE_TESTS
    printf("x[1] = %.11lf\n", x[1]);
#endif

    // r = r - alpha * q
    for (j = 0; j < n; j++)
      r[j] -= (C_PRECI_DT)alpha * (C_PRECI_DT)q[j];
#ifdef ENABLE_TESTS
    printf("r[1] = %lf\n", r[1]);
#endif

    Rho = 0.0;
    res_norm = norm(n, r);
#ifdef ENABLE_TESTS
    printf("res norm = %lf\n", res_norm);
#endif

    ratio = (C_PRECI_DT)res_norm / (C_PRECI_DT)init_norm;
#ifdef ENABLE_TESTS
    printf("ratio = %0.11lf\n", ratio);
#endif
    if (itert > 1) {
      matvec(A, x, r);
#ifdef ENABLE_TESTS
      printf("r[1] = %lf\n", r[1]);
#endif
      for (j = 0; j < n; j++)
        r[j] = (C_PRECI_DT)b[j] - (C_PRECI_DT)r[j];
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
           iter, x[0], alpha, beta, res_norm, v, r[0], p[0], q[0], z[0], ratio,
           tolerance);*/

#ifdef ENABLE_TESTS
    fflush(stdout);
    // printf("\e[1;1H\e[2J");
#endif
  }
  *iter = itert;
  // printf("\n %d TOTAL ITERATIONS \n", itert);

  // CPU TIME
  // end = clock();
  //*elapsed = 1000 * (((double)(end - start)) / CLOCKS_PER_SEC);

  // WALL
  end = omp_get_wtime();
  *elapsed = (end - start) * 1000;

  free(r);
  free(p);
  free(q);
  free(z);
  return;
}

// find z = M^(-1)r
/*void precondition(my_crs_matrix *M, my_crs_matrix *L, double *r,
  C_PRECI_DT *z)
{
  int n = M->n;
  int i, j;

  C_PRECI_DT *y = (C_PRECI_DT *)malloc(sizeof(C_PRECI_DT) * n);
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
void forwardsub(my_crs_matrix *A, C_PRECI_DT *b, double *x) {
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

double matvec_dot(my_crs_matrix *A, C_PRECI_DT *x, C_PRECI_DT *y, int n) {

  double result = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      result = (C_PRECI_DT)result + (C_PRECI_DT)x[i] * (C_PRECI_DT)A->val[j] *
                                        (C_PRECI_DT)y[A->col[j]];
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
    x += (C_PRECI_DT)v[i] * (C_PRECI_DT)u[i];

  return x;
}

void matvec(my_crs_matrix *A, double *x, double *y) {
  int n = A->n;
  for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      // printf("%d ? %d\n", A->col[j], n);
      y[i] += (C_PRECI_DT)A->val[j] * (C_PRECI_DT)x[A->col[j]];
    }
  }
}

// find the norm of a vector
double norm(int n, double *v) {
  double ssq, scale, absvi;
  int i;

  if (n == 1)
    return fabs((C_PRECI_DT)v[0]);

  scale = 0.0;
  ssq = 1.0;

  for (i = 0; i < n; i++) {
    if ((C_PRECI_DT)v[i] != 0) {
      absvi = fabs((C_PRECI_DT)v[i]);
      if ((C_PRECI_DT)scale < (C_PRECI_DT)absvi) {
        ssq = 1.0 + (C_PRECI_DT)ssq * ((C_PRECI_DT)scale / (C_PRECI_DT)absvi) *
                        ((C_PRECI_DT)scale / (C_PRECI_DT)absvi);
        scale = absvi;
      } else
        ssq = (C_PRECI_DT)ssq + ((C_PRECI_DT)absvi / (C_PRECI_DT)scale) *
                                    ((C_PRECI_DT)absvi / (C_PRECI_DT)scale);
    }
  }
  return (C_PRECI_DT)scale * (C_PRECI_DT)sqrt((C_PRECI_DT)ssq);
}
