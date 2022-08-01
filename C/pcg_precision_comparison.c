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

/* static double norm(int n, double *v) {
   double norm;
  int i;

  for (i=0, norm=0.0; i<n; i++) norm += v[i]*v[i];

  norm = sqrt(norm);
  return norm;
  }*/

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
  printf("%d, ", rowctr[0]);

  for (i = 1; i < M->n; i++) {
    M->rowptr[i] = rowctr[i - 1] + M->rowptr[i - 1];
    // printf("%d, ", rowctr[i]);
  }
  printf("\n");
  fclose(file);
  free(rowctr);
  return M;
  }

  void free_matrix(my_csr_matrix * M) {
    free(M->val);
    free(M->col);
    free(M->rowptr);
    free(M);

    return;
  }

  void print_csr(my_csr_matrix *M) {
    int i = 0;

    printf("values,");
    for (i = 0; i < M->nz; i++)
      printf("%Lf, ", M->val[i]);
    printf("\n\n");

    printf("index,");
    for (i = 0; i < M->nz; i++)
      printf("%d, ", M->col[i]);
    printf("\n\n");

    printf("rowptr,");
    for (i = 0; i < M->n; i++)
      printf("%d, ", M->rowptr[i]);
    printf("\n\n");
  }
  int main(void) {

    struct my_csr_matrix *bob = mm2csr("./test_subjects/bob.mtx");
    print_csr(bob);

    struct my_csr_matrix *rob = mm2csr("./test_subjects/rob.mtx");
    print_csr(rob);

    free_matrix(bob);
    free_matrix(rob);

    return 0;
  }
