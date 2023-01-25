#include "../include/my_crs_matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../include/CCG.h"

int conjugant_gradient(my_crs_matrix *A, my_crs_matrix *M, PRECI_DT *b,
                       PRECI_DT *x, int max_iter, PRECI_DT tolerance)

{
  int n = A->n;
  PRECI_DT *r = (PRECI_DT *)malloc(n * sizeof(PRECI_DT));

  PRECI_DT *p = (PRECI_DT *)malloc(n * sizeof(PRECI_DT));
  PRECI_DT *q = (PRECI_DT *)malloc(n * sizeof(PRECI_DT));
  PRECI_DT *z = (PRECI_DT *)malloc(n * sizeof(PRECI_DT));

  PRECI_DT alpha = 0.0;
  PRECI_DT beta = 0.0;

  int iter = 0;
  int j = 0;

  PRECI_DT v = 0;
  PRECI_DT Rho = 0;
  PRECI_DT Rtmp = 0;

  PRECI_DT res_norm = 0;
  PRECI_DT init_norm = 0;
  PRECI_DT ratio = 0;

  double Tiny = 0.1e-28;

  // x = zeros
  for (int i = 0; i < n; i++)
    x[i] = 0;

  // r = b - A*x
  matvec(A, x, r);
  for (int i = 0; i < n; i++)
    r[i] = b[i] - r[i];

  // z = MT\(M\r)
  // precondition(M, r, z);
  for (int i = 0; i < n; i++)
    z[i] = r[i];

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

  printf("** %lf | %d | %d ** \n", A->val[1], A->col[1], A->rowptr[1]);
  printf("iteration PREQUEL\n x0 = %lf \t alpha= %lf \t beta= %lf \n r0 = %lf "
         "\n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm ratio(%lf) > "
         "tolerance(%lf)\n\n\n",
         x[0], alpha, beta, r[0], p[0], q[0], z[0], ratio, tolerance);

  // main CG loop
  while (iter <= max_iter && ratio > tolerance) {
    // next iteration
    iter++;

    // Precondition
    // precondition(M, r, z);
    for (j = 0; j < n; j++)
      z[j] = r[j];

    for (j = 0, Rho = 0.0; j < n; j++)
      Rho += r[j] * z[j];

    // beta = dot(r,z) / v

    // p = z + beta * p
    if (iter == 1) {
      for (j = 0; j < n; j++)
        p[j] = z[j];
    } else {
      beta = Rho / (v + Tiny);
      for (j = 0; j < n; j++)
        p[j] = z[j] + (beta * p[j]);
    }

    // q = A*p
    matvec(A, p, q);

    for (j = 0, Rtmp = 0.0; j < n; j++)
      Rtmp += p[j] * q[j];

    // v = early dot(r,z)
    v = dot(r, z, n);

    printf("p*q[1]=%lf\n", dot(p, q, n));
    // alpha = v / dot(p,q)
    alpha = Rho / (Rtmp + Tiny);
    // x = x + alpha * p
    printf("x[0] = %lf + %lf * %lf = ", x[0], alpha, p[0]);
    for (j = 0; j < n; j++)
      x[j] = x[j] + (alpha * p[j]);
    printf("%lf\n", x[0]);

    // r = r - alpha * q
    for (j = 0; j < n; j++)
      r[j] -= alpha * q[j];

    Rho = 0.0;
    res_norm = norm(n, r);
    ratio = res_norm / init_norm;

    if (iter > 0) {
      matvec(A, x, r);
      for (j = 0; j < n; j++)
        r[j] = b[j] - r[j];
    }
    printf("\nend of iteration %d\n x1 = %lf \t alpha= %lf \t beta= %lf "
           "\n v "
           "= %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm "
           "ratio(%lf) > tolerance(%lf)\n\n\n",
           iter, x[0], alpha, beta, v, r[0], p[0], q[0], z[0], ratio,
           tolerance);
    printf("\e[1;1H\e[2J");
    
  }
  free(r);
  free(p);
  free(q);
  free(z);
  return iter;
}

// incomplete Choleskys
void ichol(my_crs_matrix *M, double *L)

{
  int n = M->n;
  int i, j, k;
  double s;
  for (i = 0; i < n; i++) {
    for (j = M->rowptr[i]; j < M->rowptr[i + 1]; j++) {
      if (M->col[j] < i)
        continue;
      s = 0;
      for (k = M->rowptr[i]; k < j; k++) {
        if (M->col[k] < i)
          continue;
        s += L[k] * L[M->rowptr[M->col[k]]] + i - M->col[k];
      }
      L[j] = (i == M->col[j]) ? sqrt(M->val[j] - s)
                              : (1.0 / L[M->rowptr[M->col[j]] + i - M->col[j]] *
                                 (M->val[j] - s));
    }
  }
}

void precondition(my_crs_matrix *M, PRECI_DT *r, PRECI_DT *z)

// find z = M^(-1)r
{
  int n = M->n;
  int i, j;

  PRECI_DT *L = (PRECI_DT *)malloc(sizeof(PRECI_DT) * n);

  ichol(M, L);

  PRECI_DT *y = (PRECI_DT *)malloc(n * sizeof(PRECI_DT));
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

PRECI_DT matvec_dot(my_crs_matrix *A, PRECI_DT *x, PRECI_DT *y, int n)
{

  PRECI_DT result = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      result += x[i] * A->val[j] * y[A->col[j]];
      // printf("result += %lf * %lf * %lf\n", x[i] , A->val[j] , y[A->col[j]]);
    }
    /*     if (result != result && i % 20 == 0)
           printf("NaN moment :(\n");*/
  }
  return result;
}

// find the dot product of two vectors

PRECI_DT dot(PRECI_DT *v, PRECI_DT *u, int n) {

  PRECI_DT x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;
}

void matvec(my_crs_matrix *A, PRECI_DT *x, PRECI_DT *y)
{
  PRECI_DT test;
  int n = A->n;
  for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      // printf("%d ? %d\n", A->col[j], n);
      test = x[A->col[j]];
      y[i] += A->val[j] * x[A->col[j]];
    }
  }
}

// find the norm of a vector
PRECI_DT norm(int n, PRECI_DT *v) {
  PRECI_DT ssq, scale, absvi;
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
