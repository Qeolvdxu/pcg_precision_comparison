
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

void CCG(my_crs_matrix *A, my_crs_matrix *M, C_PRECI_DT *b, C_PRECI_DT *x,
         int max_iter, C_PRECI_DT tolerance, int *iter, C_PRECI_DT *elapsed) {

  int n = A->n;
  C_PRECI_DT *r = (C_PRECI_DT *)malloc(n * sizeof(C_PRECI_DT));

  C_PRECI_DT *p = (C_PRECI_DT *)malloc(n * sizeof(C_PRECI_DT));
  C_PRECI_DT *q = (C_PRECI_DT *)malloc(n * sizeof(C_PRECI_DT));
  C_PRECI_DT *z = (C_PRECI_DT *)malloc(n * sizeof(C_PRECI_DT));

  C_PRECI_DT alpha = 0.0;
  C_PRECI_DT beta = 0.0;

  int j = 0;

  C_PRECI_DT v = 0;
  C_PRECI_DT Rho = 0;
  C_PRECI_DT Rtmp = 0;

  C_PRECI_DT res_norm = 0;
  C_PRECI_DT init_norm = 0;
  C_PRECI_DT ratio = 0;

  double Tiny = 0.1e-28;

  // x = zeros
  for (int i = 0; i < n; i++)
    x[i] = 0;

  // r = b - A*x
  matvec(A, x, r);
  for (int i = 0; i < n; i++)
    r[i] = b[i] - r[i];

  // z = MT\(M\r);
  if (M)
    precondition(M, r, z);
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

  /*printf("** %lf | %d | %d ** \n", A->val[1], A->col[1], A->rowptr[1]);
  printf("iteration PREQUEL\n x0 = %lf \t alpha= %lf \t beta= %lf \n r0 = %lf "
         "\n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm ratio(%lf) > "
         "tolerance(%lf)\n\n\n",
         x[0], alpha, beta, r[0], p[0], q[0], z[0], ratio, tolerance);*/

  // WALL TIME
  double start;
  double end;
  start = omp_get_wtime();

  // CPU TIME
  // clock_t start, end;
  // double cpu_time_used;
  // start = clock();

  // main CG loop
  int itert = 0;
  //  printf("%d \n", *iter);
  while (itert < max_iter && ratio > tolerance) {
    // printf("%d < %d && %f > %f\n", itert, max_iter, ratio, tolerance);
// next iteration
#ifdef ENABLE_TESTS
    printf("\nITERATION %d\n", itert);
#endif
    itert++;

    // Precondition
    // z = MT\(M\r);
    if (M)
      precondition(M, r, z);
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
      beta = Rho / (v + Tiny);
      for (j = 0; j < n; j++)
        p[j] = z[j] + (beta * p[j]);
    }
#ifdef ENABLE_TESTS
    printf("beta = %lf\n", beta);
    printf("p[1] = %lf\n", p[1]);
#endif

    // q = A*p
    matvec(A, p, q);
#ifdef ENABLE_TESTS
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
    printf("x[1] = %lf\n", x[1]);
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
    printf("ratio = %lf\n", ratio);
#endif
    if (itert > 1) {
      matvec(A, x, r);
#ifdef ENABLE_TESTS
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
/*void precondition(my_crs_matrix *M, my_crs_matrix *L, C_PRECI_DT *r,
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
void precondition(my_crs_matrix *M, C_PRECI_DT *r, C_PRECI_DT *z) {
  int n = M->n;

  for (int i = 0; i < n; i++) {
    z[i] = r[i] / M->val[M->rowptr[i]];
    for (int j = M->rowptr[i] + 1; j < M->rowptr[i + 1]; j++)
      r[M->col[j]] -= M->val[j] * z[i];
  }

  /*for (int i = 0; i < n; i++) {
    int start = M->rowptr[i];
    int end = M->rowptr[i + 1];
    double *row_vals = &M->val[start];
    int *row_indices = &M->col[start];

    z[i] = r[i];
    for (int j = start; j < end; j++) {
      int col_index = row_indices[j];
      double val = row_vals[j];
      if (col_index < i && col_index > 0) {
        printf("%d\n", col_index);
        z[i] -= val * z[col_index];
      }
    }
    z[i] /= M->val[start];
  }*/
  // forward subsitution
  /*z[0] = r[0] / M->val[0];
  C_PRECI_DT comp = 0.0;

  for (int i = 1; i < M->n; i++) {
    comp = comp * 0;
    for (int j = M->rowptr[i]; j < M->rowptr[i + 1]; j++) {
      int colIndex = M->col[j];
      // printf("colindex = %d\n", colIndex);
      comp += M->val[j] * z[colIndex];
      // printf("comp += %lf * %lf\n = %lf\n", M->val[j], z[colIndex], comp);
    }
    z[i] = (r[i] - comp) / M->val[M->rowptr[i]];
    printf("z[%d] += (%lf - %lf) / [%d] %lf = %lf\n\n", i, r[i], comp,
           M->rowptr[i], M->val[M->rowptr[i]], z[i]);
  }*/
}

C_PRECI_DT matvec_dot(my_crs_matrix *A, C_PRECI_DT *x, C_PRECI_DT *y, int n) {

  C_PRECI_DT result = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      result += x[i] * A->val[j] * y[A->col[j]];
      // printf("result += %lf * %lf * %lf\n", x[i] , A->val[j] ,
      // y[A->col[j]]);
    }
    /*     if (result != result && i % 20 == 0)
           printf("NaN moment :(\n");*/
  }
  return result;
}

// find the dot product of two vectors

C_PRECI_DT dot(C_PRECI_DT *v, C_PRECI_DT *u, int n) {

  C_PRECI_DT x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;
}

void matvec(my_crs_matrix *A, C_PRECI_DT *x, C_PRECI_DT *y) {
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
C_PRECI_DT norm(int n, C_PRECI_DT *v) {
  C_PRECI_DT ssq, scale, absvi;
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
