#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct my_csr_matrix {
  int n;
  int m;
  int nz;
  long double *val;
  int* col;
  int* rowptr;
} my_csr_matrix;

static double norm(int n, double*v)
{
  double norm;
  int i;

  for (i=0, norm=0.0; i<n; i++) norm += v[i]*v[i];

  norm = sqrt(norm);
  return norm;
}

/* my_csr_matrix csr_transpose(const my_csr_matrix input) {
   my_csr_matrix res;
  return res;
  }*/

int my_pcg(my_csr_matrix *A,
           //	   int* b,
           int maxiter, double tol)
// my_csr_matrix* M)
{
  // Set up
  int n = A->n;
  double *R = malloc(1);
  double inorm = norm(n, R);
  // Iterate
  int Iter = 0;
  int i = 0;
  while (i <= maxiter && norm(n, R) / inorm > tol) {
    Iter++;
  }

  return 0;
}

my_csr_matrix *mm2csr(char *name) {

  struct my_csr_matrix *M = malloc(sizeof(struct my_csr_matrix));
  FILE *file = fopen(name, "r");
  int row;

  fscanf(file, "%d %d %d", &M->m, &M->n, &M->nz);
    M->val = malloc(sizeof(long double) * M->nz);
    M->col = malloc(sizeof(int) * M->nz);
    M->rowptr = malloc(sizeof(int) * M->n);

    int *rowctr = malloc(sizeof(int) * M->n);

    int i = 0;
    for (i = 0; i < M->n; i++)
      rowctr[i] = 0;

    i = 0;
    for (i = 0; i < M->nz; i++) {
      fscanf(file, "%d %d %Lf", &row, &M->col[i], &M->val[i]);
      rowctr[row]++;
    }

    M->rowptr[0] = 0;
    for (i = 1; i < M->n; i++)
      M->rowptr[i] = rowctr[i] + M->rowptr[i - 1];

    fclose(file);
    return M;
    }

        void
        free_matrix(my_csr_matrix * M) {
      free(M->val);
      free(M->col);
      // free(M->rowptr);
      free(M);
      return;
    }

int main(void) {

  struct my_csr_matrix *bob = mm2csr("./test_subjects/bob.mtx");
  int i = 0, j = 0;

  for (i = 0; i < bob->nz; i++)
    printf("%Lf, ", bob->val[i]);
  printf("\n");

  for (i = 0; i < bob->nz; i++)
    printf("%d, ", bob->col[i]);
  printf("\n");

  for (i = 0; i < bob->n; i++)
    printf("%d, ", bob->rowptr[i]);
  printf("\n");

  i += j * 2;
  free_matrix(bob);
  return 0;
}
