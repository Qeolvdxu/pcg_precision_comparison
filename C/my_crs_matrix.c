#include "my_crs_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// find the dot product of two vectors
static double dotprod(int n, double *v, double *u) {
  double x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;
}

// find the norm of a vector
static double norm(int n, double *v) {

  double norm;
  int i;

  for (i = 0, norm = 0.0; i < n; i++)
    norm += v[i] * v[i];

  norm = sqrt(norm);
  return norm;
}

// multiply a my_crs_matrix with a vector
double *my_crs_times_vec(my_crs_matrix *M, double *v) {
  int i;
  double *ans = malloc(sizeof(double) * M->n);
  for (i = 0; i < M->n; i++)
    ans[i] = 0;

  int row = 0;
  for (i = 0; i < M->nz; i++) {

    if (i == M->rowptr[row])
      row++;
    ans[row - 1] += M->val[i] * v[M->col[i]];
    printf("ans %d += %lf * %lf]\n", row - 1, M->val[i], v[M->col[i]]);
  }
  return ans;
}

double *my_crs_cg(my_crs_matrix *M, PRECI_DT *b, double tol, int maxit) {
  int i, j, v;
  // allocate vectors
  double *x = malloc(sizeof(double) * M->n);
  for (i = 0; i < M->n; i++)
    x[i] = 0;
  double alpha, beta;
  double *p = malloc(sizeof(double) * M->n);
  double *r; // = malloc(sizeof(double) * M->n);
  double *q; // = malloc(sizeof(double) * M->n);
  double *z = malloc(sizeof(double) * M->n);

  // Set up to iterate
  r = my_crs_times_vec(M, x);
  for (i = 0; i < M->n; i++) {
    printf("%lf, ", r[i]);
    r[i] = b[i] - r[i];
  }
  for (i = 0; i < M->n; i++)
    z[i] = r[i];
  for (i = 0; i < M->n; i++)
    p[i] = z[i];

  // Start iteration
  i = 0;
  while (i <= maxit && norm(M->n, r) / norm(M->n, b) > tol) {
    printf("\n\ni:%d\nnorm_r: %lf norm_b: %lf\n", i, norm(M->n, r),
           norm(M->n, b));
    q = my_crs_times_vec(M, p);
    v = dotprod(M->n, r, z);

    // alpha =  / dot(p,q)
    alpha = v / dotprod(M->n, p, q);

    // x = x + alpha*p
    for (j = 0; j < M->n; j++) {
      x[j] += alpha * p[j];
    }

    // r = r - alpha*q
    for (j = 0; j < M->n; j++)
      r[j] -= alpha * q[j];
    beta = dotprod(M->n, r, z) / v;

    // p = z + beta * p;
    for (j = 0; j < M->n; j++) {
      p[j] = z[j] + beta * p[j];
      printf("%lf %lf %lf\n", p[j], z[j], beta * p[j]);
    }
    free(q);
    i++;
  }

  free(p);
  // free(q);
  free(z);
  free(r);

  return x;
}
    // read matrix file into a my_csr_matrix variable
my_crs_matrix *my_crs_read(char *name) {
  my_crs_matrix *M = malloc(sizeof(my_crs_matrix));
  FILE *file = fopen(name, "r");
  int i;

  fscanf(file, "%d %d %d", &M->m, &M->n, &M->nz);
  M->val = malloc(sizeof(PRECI_DT) * M->nz);

  M->col = malloc(sizeof(int) * M->nz);
  M->rowptr = malloc(sizeof(int) * M->n);

  for (i = 0; i < M->n; i++)
    fscanf(file, "%d ", &M->rowptr[i]);
  for (i = 0; i < M->nz; i++)
    fscanf(file, "%d ", &M->col[i]);
  for (i = 0; i < M->nz; i++)
    fscanf(file, PRECI_S, &M->val[i]);

  fclose(file);
  return M;
}

    // Free my_csr_matrix variable
void my_crs_free(my_crs_matrix *M) {
  free(M->val);
  free(M->col);
  free(M->rowptr);
  free(M);

  return;
}

    // Print my_csr_matrix Matrix
void my_crs_print(my_crs_matrix *M) {
  int i = 0;
  printf("rowptr,");
  for (i = 0; i < M->n; i++)
    printf("%d, ", M->rowptr[i]);
  printf("\n\n");

  printf("index,");
  for (i = 0; i < M->nz; i++)
    printf("%d, ", M->col[i]);
  printf("\n\n");

  printf("values,");
  for (i = 0; i < M->nz; i++) {
    printf(PRECI_S, M->val[i]);
    printf(", ");
  }
  printf("\n\n");
}