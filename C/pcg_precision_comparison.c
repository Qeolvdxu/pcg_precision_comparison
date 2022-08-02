#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define PRECI_DT double
#define PRECI_S "%lf "


typedef struct my_csr_matrix {
  int n;
  int m;
  int nz;
  PRECI_DT *val;
  int *col;
  int *rowptr;
} my_csr_matrix;

static double dotprod(int n, double *v, double *u) {
  double x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;
}

static double norm(int n, double *v) {

  double norm;
  int i;

  for (i = 0, norm = 0.0; i < n; i++)
    norm += v[i] * v[i];

  norm = sqrt(norm);
  return norm;
}

double *crsprod(my_csr_matrix *M, double *v) {
  int i, j, k;
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

int crs_cg(my_csr_matrix *M, PRECI_DT *b, double tol, int maxit) {
  int i = 0;

  double *x = malloc(sizeof(double) * M->n);
  for (i = 0; i < M->n; i++)
    x[i] = 0;
  double alpha, beta;
  double *p = malloc(sizeof(double) * M->n);
  double *r = malloc(sizeof(double) * M->n);
  double *q = malloc(sizeof(double) * M->n);
  double *z = malloc(sizeof(double) * M->n);
  int v;
  // r = b - M * x;
  r = crsprod(M, x);
  for (i = 0; i < M->n; i++) {
    printf("%lf, ", r[i]);
    r[i] = b[i] - r[i];
  }
  z = r;
  p = z;
  i = 0;
  while (i <= maxit && norm(M->n, r) / norm(M->n, b) > tol) {
    printf("i:%d\nnorm_r: %lf norm_b: %lf\n\n", i, norm(M->n, r),
           norm(M->n, b));
    q = crsprod(M, p);
    v = dotprod(M->n, r, z);

    // alpha =  / dot(p,q)
    alpha = v / dotprod(M->n, p, q);

    // x = x + alpha*p

    // r = r - alpha*q

    beta = dotprod(M->n, r, z) / v;

    // p = z + beta * p;

    i++;
  }
     return 0;
}

my_csr_matrix *csrread(char *name) {
  struct my_csr_matrix *M = malloc(sizeof(struct my_csr_matrix));
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

void free_matrix(my_csr_matrix * M) {
  free(M->val);
  free(M->col);
  free(M->rowptr);
  free(M);

  return;
}

void print_csr(my_csr_matrix * M) {
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

int main(void) {
  int i;

  struct my_csr_matrix *test = csrread("./test_subjects/bob.mtx.crs");
  print_csr(test);
  double *v = malloc(sizeof(double) * test->n);
  double *a = malloc(sizeof(double) * test->n);
  double *b = malloc(sizeof(double) * test->n);
  for (i = 0; i < test->n; i++)

    v[0] = 10;

  v[1] = 11;

  v[2] = 12;

  for (i = 0; i < test->n; i++)
    b[i] = 1;

  crs_cg(test, b, 1e-6, 8000);
  free_matrix(test);
  return 0;
}
